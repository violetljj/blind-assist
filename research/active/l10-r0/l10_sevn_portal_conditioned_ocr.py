#!/usr/bin/env python3
"""Compare frozen SEVN V2 with a portal-conditioned OCR successor.

The successor keeps the V2 renderer, recognizer weights, exact-token rule,
portal model, mask topology, and truth scorer.  It adds one deficit-triggered
observation branch: only when V2 returns UNKNOWN_TARGET_TEXT, surviving portal
masks define canonical upper-portal credential crops for another pass through
the same recognizer.  A prior V2 proposal is never overridden.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from rapidocr import RapidOCR
from ultralytics import YOLO

from l10_panolab import require, sha256_file, utc_now
import l10_sevn_fresh_pan_panel as panel
import l10_sevn_pixel_replay as v1
import l10_sevn_pixel_topology_replay as v2


SCHEMA = "blindassist-l10-sevn-portal-conditioned-ocr-protocol-v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def verify_sha256(spec: dict[str, Any]) -> Path:
    path = resolve(spec["path"])
    require(path.is_file(), f"missing frozen input: {path}")
    require(sha256_file(path) == spec["sha256"], f"SHA-256 mismatch: {path}")
    return path


def cohort_identity(public: dict[str, Any]) -> tuple[set[tuple[str, str]], set[int]]:
    addresses = {
        (str(episode["mission"]["street_name"]), str(episode["mission"]["house_number"]))
        for episode in public["episodes"]
    }
    frames = {int(row["frame_id"]) for row in public["observations"].values()}
    return addresses, frames


def cross_episode_frame_reuse(public: dict[str, Any]) -> dict[str, list[str]]:
    owners: dict[int, list[str]] = {}
    for episode in public["episodes"]:
        episode_id = episode["episode_id"]
        start = episode["start_observation_id"]
        observation_ids = {start}
        observation_ids.update(edge["to_observation_id"] for edge in episode["transitions"][start].values())
        for frame in {int(public["observations"][item]["frame_id"]) for item in observation_ids}:
            owners.setdefault(frame, []).append(episode_id)
    return {str(frame): rows for frame, rows in owners.items() if len(rows) > 1}


def portal_conditioned_specs(
    portals: list[dict[str, Any]],
    image: np.ndarray,
    contract: dict[str, Any],
    association_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    ranked = sorted(
        portals,
        key=lambda row: (-row["summary"]["confidence"], row["summary"]["candidate_id"]),
    )[: int(contract["maximum_portals"])]
    specs = []
    for portal in ranked:
        summary = portal["summary"]
        topology = v2.portal_topology(portal, association_contract)
        x1, y1, x2, _ = summary["mask_box_xyxy"]
        crop_x1 = max(0, math.floor(x1 - topology["horizontal_radius"]))
        crop_x2 = min(width, math.ceil(x2 + topology["horizontal_radius"]))
        crop_y1 = max(0, math.floor(y1 - topology["vertical_radius"]))
        crop_y2 = min(height, math.ceil(topology["upper_cutoff_y"] + topology["vertical_radius"]))
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        if crop_width < int(contract["minimum_crop_side_pixels"]) or crop_height < int(
            contract["minimum_crop_side_pixels"]
        ):
            continue
        scale = min(
            float(contract["maximum_magnification"]),
            float(contract["canonical_long_side_pixels"]) / max(crop_width, crop_height),
        )
        scale = max(1.0, scale)
        specs.append(
            {
                "pass_id": f"PORTAL_CREDENTIAL_{summary['candidate_id']}",
                "source_portal_id": summary["candidate_id"],
                "x": crop_x1,
                "y": crop_y1,
                "width": crop_width,
                "height": crop_height,
                "output_width": max(1, round(crop_width * scale)),
                "output_height": max(1, round(crop_height * scale)),
                "magnification": round(scale, 6),
                "credential_region": "V2_ADAPTIVE_UPPER_MASK_NEIGHBORHOOD",
            }
        )
    return specs


def portal_conditioned_ocr(
    engine: RapidOCR,
    image: np.ndarray,
    portals: list[dict[str, Any]],
    contract: dict[str, Any],
    v2_contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    receipts = []
    specs = portal_conditioned_specs(
        portals,
        image,
        contract,
        v2_contract["association"],
    )
    for spec in specs:
        current, seconds = v2.ocr_words_for_pass(engine, image, spec)
        for row in current:
            row["source_portal_id"] = spec["source_portal_id"]
        rows.extend(current)
        receipts.append({**spec, "raw_word_count": len(current), "seconds": round(seconds, 6)})
    height, width = image.shape[:2]
    deduplicated = v2.deduplicate_ocr(
        rows,
        width,
        height,
        v2_contract["ocr"]["deduplication"],
    )
    return deduplicated, {
        "branch": "PORTAL_CONDITIONED_CREDENTIAL_OBSERVATION",
        "passes": receipts,
        "raw_word_count": len(rows),
        "deduplicated_word_count": len(deduplicated),
        "seconds": round(sum(row["seconds"] for row in receipts), 6),
    }


def bind_from_rows(
    ocr_rows: list[dict[str, Any]],
    portals: list[dict[str, Any]],
    target: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    exact = [
        row
        for row in ocr_rows
        if row["canonical"] == target and row["score"] >= float(contract["ocr"]["minimum_score"])
    ]
    topology_by_id = {
        row["summary"]["candidate_id"]: v2.portal_topology(row, contract["association"])
        for row in portals
    }
    pairs = []
    pair_audit = []
    for text in exact:
        for portal in portals:
            candidate_id = portal["summary"]["candidate_id"]
            topology = v2.pair_topology(text, portal, topology_by_id[candidate_id])
            pair_audit.append(
                {
                    "text": text,
                    "portal_id": candidate_id,
                    "portal_confidence": portal["summary"]["confidence"],
                    "topology": topology,
                }
            )
            if topology["admissible"]:
                key = (
                    topology["upper_mask_distance_ratio"],
                    topology["lateral_offset_portal_widths"],
                    topology["vertical_offset_portal_heights"],
                    -portal["summary"]["confidence"],
                    -text["score"],
                    candidate_id,
                )
                pairs.append((key, text, portal, topology))
    pairs.sort(key=lambda row: row[0])
    selected = None
    if not exact:
        state = "UNKNOWN_TARGET_TEXT"
    elif not portals:
        state = "UNKNOWN_PORTAL_MASK"
    elif not pairs:
        state = "UNKNOWN_NO_ADMISSIBLE_MASK_TOPOLOGY"
    else:
        _, text, portal, topology = pairs[0]
        state = "PIXEL_BOUND_MASK_PORTAL_PROPOSAL"
        selected = {
            "text": text,
            "portal": portal["summary"],
            "topology": topology,
        }
    return {
        "state": state,
        "target_house_number_canonical": target,
        "exact_target_text_candidates": exact,
        "exact_target_text_full_pass": any("FULL" in row["observed_passes"] for row in exact),
        "exact_target_text_tile_pass": any(
            any(source.startswith("TILE_") for source in row["observed_passes"])
            for row in exact
        ),
        "exact_target_text_portal_pass": any(
            any(source.startswith("PORTAL_CREDENTIAL_") for source in row["observed_passes"])
            for row in exact
        ),
        "deduplicated_ocr_words": ocr_rows,
        "portal_mask_candidates": [row["summary"] for row in portals],
        "topology_pair_audit": pair_audit,
        "selected_binding": selected,
    }


def infer_successor(
    image: np.ndarray,
    mission: dict[str, Any],
    ocr_engine: RapidOCR,
    portal_model: YOLO,
    postprocessor: Any,
    v2_contract: dict[str, Any],
    successor_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    baseline, masks = v2.infer_binding(
        image,
        mission,
        ocr_engine,
        portal_model,
        postprocessor,
        v2_contract,
    )
    successor = copy.deepcopy(baseline)
    successor["exact_target_text_portal_pass"] = False
    successor["observation_branch"] = "V2_RESULT_RETAINED"
    successor["portal_conditioned_receipt"] = {
        "executed": False,
        "reason": "V2_STATE_NOT_UNKNOWN_TARGET_TEXT",
        "passes": [],
        "raw_word_count": 0,
        "deduplicated_word_count": 0,
        "seconds": 0.0,
    }
    if baseline["state"] != "UNKNOWN_TARGET_TEXT":
        return baseline, successor, masks

    portals = [
        {"summary": summary, "mask": masks[summary["candidate_id"]]}
        for summary in baseline["portal_mask_candidates"]
    ]
    if not portals:
        successor["portal_conditioned_receipt"]["reason"] = "NO_SURVIVING_PORTAL_MASK"
        return baseline, successor, masks

    portal_rows, receipt = portal_conditioned_ocr(
        ocr_engine,
        image,
        portals,
        successor_contract,
        v2_contract,
    )
    height, width = image.shape[:2]
    combined = v2.deduplicate_ocr(
        [*baseline["deduplicated_ocr_words"], *portal_rows],
        width,
        height,
        v2_contract["ocr"]["deduplication"],
    )
    successor = bind_from_rows(
        combined,
        portals,
        v1.canonical_token(mission["house_number"]),
        v2_contract,
    )
    successor["observation_branch"] = "PORTAL_CONDITIONED_AFTER_UNKNOWN_TARGET_TEXT"
    successor["runtime_receipt"] = copy.deepcopy(baseline["runtime_receipt"])
    successor["portal_conditioned_receipt"] = {"executed": True, **receipt}
    return baseline, successor, masks


def score_output(
    output: dict[str, Any],
    masks: dict[str, np.ndarray],
    target_box: list[float],
    truth_visible: bool,
    evaluation_contract: dict[str, Any],
) -> dict[str, Any]:
    selected = output["selected_binding"]
    if selected is None:
        mask_metrics = None
        outcome = "UNKNOWN"
    else:
        selected_mask = masks[selected["portal"]["candidate_id"]]
        mask_metrics = v2.truth_mask_metrics(selected_mask, target_box, evaluation_contract)
        outcome = "CORRECT_TARGET_DOOR" if mask_metrics["correct"] else "WRONG_DOOR"
    return {
        "target_house_number_visible": bool(truth_visible),
        "exact_target_text": bool(output["exact_target_text_candidates"]),
        "selected_mask_truth_metrics": mask_metrics,
        "outcome": outcome,
    }


def summarize(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    truth_visible = sum(row["truth"]["target_house_number_visible"] for row in rows)
    evaluations = [row["evaluation"][arm] for row in rows]
    exact_visible = sum(
        evaluation["exact_target_text"] and row["truth"]["target_house_number_visible"]
        for row, evaluation in zip(rows, evaluations)
    )
    correct = sum(row["outcome"] == "CORRECT_TARGET_DOOR" for row in evaluations)
    wrong = sum(row["outcome"] == "WRONG_DOOR" for row in evaluations)
    return {
        "episode_count": len(rows),
        "truth_visible_house_number_opportunities": truth_visible,
        "exact_target_text_when_truth_visible": exact_visible,
        "exact_target_text_when_truth_visible_rate": round(exact_visible / truth_visible, 6)
        if truth_visible
        else 0.0,
        "correct_target_door_bindings": correct,
        "correct_target_door_binding_rate": round(correct / len(rows), 6),
        "wrong_door_bindings": wrong,
        "unknown": len(rows) - correct - wrong,
    }


def scenario_metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    output = {}
    for scenario in panel.PAN_SCENARIOS:
        current = [row for row in rows if row["scenario_class"] == scenario]
        output[scenario] = summarize(current, arm)
    return output


def evaluate_gates(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    successor: dict[str, Any],
    disjointness: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    exact_gain = (
        successor["exact_target_text_when_truth_visible"]
        - baseline["exact_target_text_when_truth_visible"]
    )
    exact_rate_gain = (
        successor["exact_target_text_when_truth_visible_rate"]
        - baseline["exact_target_text_when_truth_visible_rate"]
    )
    correct_gain = (
        successor["correct_target_door_bindings"] - baseline["correct_target_door_bindings"]
    )
    baseline_correct_ids = {
        row["episode_id"]
        for row in rows
        if row["evaluation"]["baseline"]["outcome"] == "CORRECT_TARGET_DOOR"
    }
    retained = sum(
        row["episode_id"] in baseline_correct_ids
        and row["evaluation"]["successor"]["outcome"] == "CORRECT_TARGET_DOOR"
        for row in rows
    )
    delta = {
        "exact_target_text_gain": exact_gain,
        "exact_target_text_rate_gain": round(exact_rate_gain, 6),
        "correct_target_door_binding_gain": correct_gain,
        "wrong_door_binding_delta": (
            successor["wrong_door_bindings"] - baseline["wrong_door_bindings"]
        ),
        "baseline_correct_bindings_retained": retained,
        "baseline_correct_bindings_total": len(baseline_correct_ids),
        "portal_conditioned_branch_executed": sum(
            row["runtime_output"]["successor"]["portal_conditioned_receipt"]["executed"]
            for row in rows
        ),
        "portal_conditioned_exact_token_recoveries": sum(
            not row["evaluation"]["baseline"]["exact_target_text"]
            and row["evaluation"]["successor"]["exact_target_text"]
            for row in rows
        ),
    }
    gates = {
        "all_episodes_rendered": len(rows) == int(gate["episode_count"]),
        "minimum_truth_visible_house_number_opportunities": (
            successor["truth_visible_house_number_opportunities"]
            >= int(gate["minimum_truth_visible_house_number_opportunities"])
        ),
        "minimum_successor_visible_text_exact_ocr_rate": (
            successor["exact_target_text_when_truth_visible_rate"]
            >= float(gate["minimum_successor_visible_text_exact_ocr_rate"])
        ),
        "minimum_exact_ocr_rate_gain": exact_rate_gain >= float(gate["minimum_exact_ocr_rate_gain"]),
        "minimum_correct_binding_gain": correct_gain >= int(gate["minimum_correct_binding_gain"]),
        "maximum_successor_wrong_bindings": (
            successor["wrong_door_bindings"] <= int(gate["maximum_successor_wrong_bindings"])
        ),
        "maximum_wrong_binding_delta": (
            delta["wrong_door_binding_delta"] <= int(gate["maximum_wrong_binding_delta"])
        ),
        "retain_all_baseline_correct_bindings": retained == len(baseline_correct_ids),
        "dual_reference_address_disjoint": disjointness["reference_address_overlap_count"] == 0,
        "dual_reference_panorama_frame_disjoint": disjointness["reference_frame_overlap_count"] == 0,
        "within_panel_cross_episode_frame_unique": (
            disjointness["within_panel_cross_episode_reused_frames"] == {}
        ),
    }
    return delta, gates


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output.resolve()
    require(not output_path.exists(), f"refusing to overwrite result: {output_path}")
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    require(protocol.get("schema") == SCHEMA, "successor protocol schema mismatch")
    require(
        sha256_file(Path(__file__).resolve()) == protocol["evaluator"]["sha256"],
        "evaluator hash mismatch",
    )

    frozen = protocol["frozen_inputs"]
    source_path = verify_sha256(frozen["public_source"])
    truth_path = verify_sha256(frozen["evaluator_truth"])
    source_protocol_path = verify_sha256(frozen["source_protocol"])
    source_materializer_path = verify_sha256(frozen["source_materializer"])
    selection_receipt_path = verify_sha256(frozen["selection_receipt"])
    reference_paths = [verify_sha256(spec) for spec in frozen["reference_public_sources"]]
    v2_protocol_path = verify_sha256(frozen["v2_protocol"])
    v2_evaluator_path = verify_sha256(frozen["v2_evaluator"])
    require(Path(v2.__file__).resolve() == v2_evaluator_path.resolve(), "loaded V2 evaluator mismatch")

    source = read_json(source_path)
    truth = read_json(truth_path)
    references = [read_json(path) for path in reference_paths]
    selection_receipt = read_json(selection_receipt_path)
    v2_protocol = read_json(v2_protocol_path)
    per_scenario = int(protocol["cohort"]["episodes_per_scenario"])
    panel.validate_pan_public_truth(source, truth, per_scenario)
    require(
        selection_receipt["outputs"]["public_source"]["sha256"] == sha256_file(source_path),
        "selection receipt/public source mismatch",
    )
    require(
        selection_receipt["outputs"]["evaluator_truth"]["sha256"] == sha256_file(truth_path),
        "selection receipt/truth mismatch",
    )

    addresses, frames = cohort_identity(source)
    reference_addresses: set[tuple[str, str]] = set()
    reference_frames: set[int] = set()
    for reference in references:
        current_addresses, current_frames = cohort_identity(reference)
        reference_addresses.update(current_addresses)
        reference_frames.update(current_frames)
    address_overlap = sorted(addresses.intersection(reference_addresses))
    frame_overlap = sorted(frames.intersection(reference_frames))
    reused_frames = cross_episode_frame_reuse(source)
    disjointness = {
        "panel_distinct_address_count": len(addresses),
        "panel_distinct_panorama_frame_count": len(frames),
        "reference_address_count": len(reference_addresses),
        "reference_panorama_frame_count": len(reference_frames),
        "reference_address_overlap_count": len(address_overlap),
        "reference_address_overlap": address_overlap,
        "reference_frame_overlap_count": len(frame_overlap),
        "reference_frame_overlap": frame_overlap,
        "within_panel_cross_episode_reused_frames": reused_frames,
    }
    require(not address_overlap, f"reference address overlap: {address_overlap}")
    require(not frame_overlap, f"reference frame overlap: {frame_overlap}")
    require(not reused_frames, f"within-panel frame reuse: {reused_frames}")

    archive_path = args.highres_archive.resolve()
    archive_contract = v2_protocol["high_resolution_archive"]
    require(archive_path.is_file(), f"missing high-resolution archive: {archive_path}")
    require(archive_path.stat().st_size == int(archive_contract["bytes"]), "archive byte mismatch")
    archive_md5 = v1.md5_file(archive_path)
    require(archive_md5 == archive_contract["md5"], "archive MD5 mismatch")
    print(json.dumps({"archive_md5_verified": archive_md5}), flush=True)

    versions = {
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "rapidocr": importlib.metadata.version("rapidocr"),
        "onnxruntime-gpu": importlib.metadata.version("onnxruntime-gpu"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "numpy": importlib.metadata.version("numpy"),
    }
    require(versions == protocol["runtime"]["versions"], f"runtime version mismatch: {versions}")
    require(torch.cuda.is_available(), "CUDA unavailable")

    model_path = verify_sha256(v2_protocol["models"]["portal_model"]["weights"])
    postprocessor_path = verify_sha256(v2_protocol["models"]["portal_model"]["postprocessor"])
    verify_sha256(v2_protocol["models"]["portal_model"]["model_card"])
    ocr_model_root = resolve(v2_protocol["models"]["ocr"]["model_root"])
    for filename, expected in v2_protocol["models"]["ocr"]["sha256"].items():
        require(sha256_file(ocr_model_root / filename) == expected, f"OCR model mismatch: {filename}")

    postprocessor = v1.load_module(postprocessor_path)
    portal_model = YOLO(str(model_path))
    require(
        portal_model.task == "segment"
        and portal_model.names == {0: "doorway", 1: "door", 2: "people", 3: "window", 4: "mirror"},
        "portal model ontology mismatch",
    )
    ocr_engine = RapidOCR(
        params={
            "Global.model_root_dir": str(ocr_model_root),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": int(
                v2_protocol["pixel_contract"]["ocr"]["intra_op_threads"]
            ),
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )

    rows = []
    started = time.perf_counter()
    action_map = v2_protocol["action_control"]["scenario_to_action"]
    archive_member = archive_contract["member_pattern"]
    with zipfile.ZipFile(archive_path, "r") as archive_zip:
        for sequence, episode in enumerate(source["episodes"], start=1):
            episode_id = episode["episode_id"]
            episode_truth = truth["episodes"][episode_id]
            scenario = episode_truth["scenario_class"]
            action = action_map[scenario]
            start_id = episode["start_observation_id"]
            transition = episode["transitions"][start_id][action]
            require(transition["action_executed"], f"{episode_id}: frozen action unavailable")
            observation_id = transition["to_observation_id"]
            observation = source["observations"][observation_id]
            headings = observation["viewport_headings_degrees"]
            require(len(headings) == 1, f"{episode_id}: expected one viewport")
            frame_id = int(observation["frame_id"])
            member_name = archive_member.format(frame_id=frame_id)
            info = archive_zip.getinfo(member_name)
            encoded = archive_zip.read(info)
            panorama = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            require(panorama is not None, f"{episode_id}: panorama decode failed")
            require(
                list(panorama.shape) == v2_protocol["render_contract"]["panorama_shape_hwc"],
                "panorama shape mismatch",
            )
            viewport, render_receipt = v1.render_viewport(
                panorama,
                float(observation["camera_pose"]["panorama_angle_degrees"]),
                float(headings[0]),
                float(observation["horizontal_fov_degrees"]),
            )
            render_receipt.update(
                {
                    "archive_member": member_name,
                    "archive_member_crc32": f"{info.CRC:08x}",
                    "archive_member_bytes": info.file_size,
                    "archive_member_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )

            baseline_output, successor_output, masks = infer_successor(
                viewport,
                episode["mission"],
                ocr_engine,
                portal_model,
                postprocessor,
                v2_protocol["pixel_contract"],
                protocol["successor_observation"],
            )
            observation_truth = episode_truth["observations"][observation_id]
            target_box = v1.annotation_to_viewport(
                episode_truth["target_door_annotation"],
                render_receipt,
                float(v2_protocol["render_contract"]["annotation_canvas_width"]),
                float(v2_protocol["render_contract"]["annotation_canvas_height"]),
            )
            baseline_evaluation = score_output(
                baseline_output,
                masks,
                target_box,
                bool(observation_truth["target_house_number_visible"]),
                v2_protocol["pixel_contract"]["evaluation"],
            )
            successor_evaluation = score_output(
                successor_output,
                masks,
                target_box,
                bool(observation_truth["target_house_number_visible"]),
                v2_protocol["pixel_contract"]["evaluation"],
            )
            rows.append(
                {
                    "episode_id": episode_id,
                    "scenario_class": scenario,
                    "frozen_action": action,
                    "observation_id": observation_id,
                    "mission": episode["mission"],
                    "render_receipt": render_receipt,
                    "runtime_output": {
                        "baseline": baseline_output,
                        "successor": successor_output,
                    },
                    "truth": {
                        "binding_state": observation_truth["binding_state"],
                        "target_house_number_visible": bool(
                            observation_truth["target_house_number_visible"]
                        ),
                        "target_door_box_xyxy": [round(value, 6) for value in target_box],
                    },
                    "evaluation": {
                        "baseline": baseline_evaluation,
                        "successor": successor_evaluation,
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "episode": episode_id,
                        "progress": f"{sequence}/{len(source['episodes'])}",
                        "baseline": baseline_evaluation["outcome"],
                        "successor": successor_evaluation["outcome"],
                        "portal_branch": successor_output["portal_conditioned_receipt"]["executed"],
                        "portal_exact": successor_output["exact_target_text_portal_pass"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    actual_device = str(next(portal_model.model.parameters()).device)
    require(actual_device.startswith("cuda"), f"silent model fallback: {actual_device}")
    baseline_metrics = summarize(rows, "baseline")
    successor_metrics = summarize(rows, "successor")
    delta, gates = evaluate_gates(
        rows,
        baseline_metrics,
        successor_metrics,
        disjointness,
        protocol["gate"],
    )
    gate_met = all(gates.values())
    decision = (
        "L10_SEVN_PORTAL_CONDITIONED_OCR_FRESH_PAN_DEVELOPMENT_GATE_MET"
        if gate_met
        else "L10_SEVN_PORTAL_CONDITIONED_OCR_FRESH_PAN_DEVELOPMENT_GATE_NOT_MET"
    )
    result = {
        "schema": "blindassist-l10-sevn-portal-conditioned-ocr-result-v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "claim_scope": "DUAL_REFERENCE_DISJOINT_SAME_SOURCE_SEVN_FRESH_PAN_DEVELOPMENT",
        "question": protocol["question"],
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "public_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "evaluator_truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "reference_public_sources": [
                {"path": str(path), "sha256": sha256_file(path)} for path in reference_paths
            ],
            "source_protocol": {
                "path": str(source_protocol_path),
                "sha256": sha256_file(source_protocol_path),
            },
            "source_materializer": {
                "path": str(source_materializer_path),
                "sha256": sha256_file(source_materializer_path),
            },
            "selection_receipt": {
                "path": str(selection_receipt_path),
                "sha256": sha256_file(selection_receipt_path),
            },
            "v2_protocol": {"path": str(v2_protocol_path), "sha256": sha256_file(v2_protocol_path)},
            "v2_evaluator": {"path": str(v2_evaluator_path), "sha256": sha256_file(v2_evaluator_path)},
            "high_resolution_archive": {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "md5": archive_md5,
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "runtime": {
            "versions": versions,
            "torch_cuda": torch.version.cuda,
            "device": actual_device,
            "device_name": torch.cuda.get_device_name(0),
        },
        "representation_boundary": protocol["representation_boundary"],
        "disjointness": disjointness,
        "baseline_metrics": baseline_metrics,
        "successor_metrics": successor_metrics,
        "delta": delta,
        "scenario_metrics": {
            "baseline": scenario_metrics(rows, "baseline"),
            "successor": scenario_metrics(rows, "successor"),
        },
        "gates": gates,
        "gate_met": gate_met,
        "wall_seconds": round(time.perf_counter() - started, 4),
        "episode_results": rows,
        "retention": {
            "panorama_crop_tile_mask_or_overlay_files_written": 0,
            "reconstructible_intermediate_images_retained": 0,
            "durable_result": str(output_path),
            "source_archive_retained": str(archive_path),
        },
        "non_claims": protocol["non_claims"],
    }
    output_path.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--highres-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "baseline_metrics": result["baseline_metrics"],
                "successor_metrics": result["successor_metrics"],
                "delta": result["delta"],
                "gates": result["gates"],
                "wall_seconds": result["wall_seconds"],
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
