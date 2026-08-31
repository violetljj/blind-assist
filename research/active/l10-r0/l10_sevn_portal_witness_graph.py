#!/usr/bin/env python3
"""Evaluate portal-specific exact-token witness edges on fresh SEVN PAN views.

Every surviving portal instance requests its own upper-portal OCR observation.
If exactly one portal observation emits the requested exact token, that typed
portal -> observation -> token edge authorizes the portal proposal.  Multiple
portal witnesses remain set-valued and UNKNOWN.  Existing V2 proposals are
never replaced or revoked.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
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
import l10_sevn_portal_conditioned_ocr as predecessor


SCHEMA = "blindassist-l10-sevn-portal-witness-graph-protocol-v1"


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


def portal_witness_rows(
    engine: RapidOCR,
    image: np.ndarray,
    portals: list[dict[str, Any]],
    observation_contract: dict[str, Any],
    v2_contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    receipts = []
    specs = predecessor.portal_conditioned_specs(
        portals,
        image,
        observation_contract,
        v2_contract["association"],
    )
    for spec in specs:
        current, seconds = v2.ocr_words_for_pass(engine, image, spec)
        for row in current:
            row["source_portal_id"] = spec["source_portal_id"]
        rows.extend(current)
        receipts.append({**spec, "raw_word_count": len(current), "seconds": round(seconds, 6)})
    return rows, {
        "executed": True,
        "branch": "PORTAL_SPECIFIC_EXACT_TOKEN_WITNESS_GRAPH",
        "passes": receipts,
        "raw_word_count": len(rows),
        "seconds": round(sum(row["seconds"] for row in receipts), 6),
    }


def infer_successor(
    image: np.ndarray,
    mission: dict[str, Any],
    ocr_engine: RapidOCR,
    portal_model: YOLO,
    postprocessor: Any,
    v2_contract: dict[str, Any],
    observation_contract: dict[str, Any],
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
    successor["portal_witness_set"] = []
    successor["portal_witness_receipt"] = {
        "executed": False,
        "reason": "V2_ALREADY_EMITTED_A_PROPOSAL",
        "passes": [],
        "raw_word_count": 0,
        "seconds": 0.0,
    }
    if baseline["selected_binding"] is not None:
        return baseline, successor, masks
    if baseline["state"] not in {
        "UNKNOWN_TARGET_TEXT",
        "UNKNOWN_NO_ADMISSIBLE_MASK_TOPOLOGY",
    }:
        successor["portal_witness_receipt"]["reason"] = "BASELINE_DEFICIT_NOT_ELIGIBLE"
        return baseline, successor, masks

    portals = [
        {"summary": summary, "mask": masks[summary["candidate_id"]]}
        for summary in baseline["portal_mask_candidates"]
    ]
    if not portals:
        successor["portal_witness_receipt"]["reason"] = "NO_SURVIVING_PORTAL_MASK"
        return baseline, successor, masks

    raw_rows, receipt = portal_witness_rows(
        ocr_engine,
        image,
        portals,
        observation_contract,
        v2_contract,
    )
    target = v1.canonical_token(mission["house_number"])
    exact_rows = [
        row
        for row in raw_rows
        if row["canonical"] == target and row["score"] >= float(v2_contract["ocr"]["minimum_score"])
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in exact_rows:
        grouped.setdefault(row["source_portal_id"], []).append(row)
    witness_set = []
    for portal_id in sorted(grouped):
        best = sorted(
            grouped[portal_id],
            key=lambda row: (-row["score"], row["source_pass"], tuple(row["box_xyxy"])),
        )[0]
        witness_set.append(
            {
                "portal_id": portal_id,
                "text": best,
                "supporting_exact_rows": len(grouped[portal_id]),
            }
        )

    successor["portal_witness_receipt"] = receipt
    successor["portal_witness_set"] = witness_set
    successor["exact_target_text_portal_pass"] = bool(witness_set)
    if witness_set:
        successor["exact_target_text_candidates"] = [row["text"] for row in witness_set]
    if len(witness_set) == 1:
        witness = witness_set[0]
        portal = next(row for row in portals if row["summary"]["candidate_id"] == witness["portal_id"])
        successor["state"] = "PIXEL_BOUND_PORTAL_SPECIFIC_EXACT_TOKEN_WITNESS"
        successor["selected_binding"] = {
            "text": witness["text"],
            "portal": portal["summary"],
            "authority_edge": {
                "type": "PORTAL_TO_LOCAL_OBSERVATION_TO_EXACT_TOKEN",
                "source_portal_id": witness["portal_id"],
                "unique_portal_witness_count": 1,
            },
        }
        successor["observation_branch"] = "UNIQUE_PORTAL_SPECIFIC_EXACT_TOKEN"
    elif len(witness_set) > 1:
        successor["state"] = "UNKNOWN_AMBIGUOUS_PORTAL_SPECIFIC_EXACT_TOKEN_SET"
        successor["selected_binding"] = None
        successor["observation_branch"] = "SET_VALUED_PORTAL_SPECIFIC_EXACT_TOKEN"
    else:
        successor["observation_branch"] = "NO_PORTAL_SPECIFIC_EXACT_TOKEN"
    return baseline, successor, masks


def summarize(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    metrics = predecessor.summarize(rows, arm)
    emitted = metrics["correct_target_door_bindings"] + metrics["wrong_door_bindings"]
    metrics["binding_precision"] = round(metrics["correct_target_door_bindings"] / emitted, 6) if emitted else 0.0
    return metrics


def evaluate_gates(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    successor: dict[str, Any],
    disjointness: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    correct_gain = successor["correct_target_door_bindings"] - baseline["correct_target_door_bindings"]
    wrong_delta = successor["wrong_door_bindings"] - baseline["wrong_door_bindings"]
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
    unique_witness_rows = [
        row
        for row in rows
        if row["runtime_output"]["successor"]["observation_branch"]
        == "UNIQUE_PORTAL_SPECIFIC_EXACT_TOKEN"
    ]
    unique_correct = sum(
        row["evaluation"]["successor"]["outcome"] == "CORRECT_TARGET_DOOR"
        for row in unique_witness_rows
    )
    delta = {
        "correct_target_door_binding_gain": correct_gain,
        "correct_target_door_binding_rate_gain": round(
            successor["correct_target_door_binding_rate"] - baseline["correct_target_door_binding_rate"],
            6,
        ),
        "wrong_door_binding_delta": wrong_delta,
        "baseline_correct_bindings_retained": retained,
        "baseline_correct_bindings_total": len(baseline_correct_ids),
        "portal_witness_branch_executed": sum(
            row["runtime_output"]["successor"]["portal_witness_receipt"]["executed"]
            for row in rows
        ),
        "unique_portal_witness_proposals": len(unique_witness_rows),
        "unique_portal_witness_correct": unique_correct,
        "unique_portal_witness_wrong": len(unique_witness_rows) - unique_correct,
        "ambiguous_portal_witness_sets": sum(
            row["runtime_output"]["successor"]["observation_branch"]
            == "SET_VALUED_PORTAL_SPECIFIC_EXACT_TOKEN"
            for row in rows
        ),
    }
    gates = {
        "all_episodes_rendered": len(rows) == int(gate["episode_count"]),
        "minimum_truth_visible_house_number_opportunities": (
            successor["truth_visible_house_number_opportunities"]
            >= int(gate["minimum_truth_visible_house_number_opportunities"])
        ),
        "minimum_successor_correct_binding_rate": (
            successor["correct_target_door_binding_rate"]
            >= float(gate["minimum_successor_correct_binding_rate"])
        ),
        "minimum_correct_binding_gain": correct_gain >= int(gate["minimum_correct_binding_gain"]),
        "minimum_successor_binding_precision": (
            successor["binding_precision"] >= float(gate["minimum_successor_binding_precision"])
        ),
        "maximum_successor_wrong_bindings": (
            successor["wrong_door_bindings"] <= int(gate["maximum_successor_wrong_bindings"])
        ),
        "maximum_wrong_binding_delta": wrong_delta <= int(gate["maximum_wrong_binding_delta"]),
        "retain_all_baseline_correct_bindings": retained == len(baseline_correct_ids),
        "cumulative_reference_address_disjoint": disjointness["reference_address_overlap_count"] == 0,
        "cumulative_reference_panorama_frame_disjoint": disjointness["reference_frame_overlap_count"] == 0,
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
    require(protocol.get("schema") == SCHEMA, "protocol schema mismatch")
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
    archive_receipt_path = verify_sha256(frozen["archive_integrity_receipt"])
    require(Path(v2.__file__).resolve() == v2_evaluator_path.resolve(), "loaded V2 evaluator mismatch")

    source = read_json(source_path)
    truth = read_json(truth_path)
    references = [read_json(path) for path in reference_paths]
    selection_receipt = read_json(selection_receipt_path)
    v2_protocol = read_json(v2_protocol_path)
    archive_receipt = read_json(archive_receipt_path)
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

    addresses, frames = predecessor.cohort_identity(source)
    reference_addresses: set[tuple[str, str]] = set()
    reference_frames: set[int] = set()
    for reference in references:
        current_addresses, current_frames = predecessor.cohort_identity(reference)
        reference_addresses.update(current_addresses)
        reference_frames.update(current_frames)
    address_overlap = sorted(addresses.intersection(reference_addresses))
    frame_overlap = sorted(frames.intersection(reference_frames))
    reused_frames = predecessor.cross_episode_frame_reuse(source)
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
    prior_archive = archive_receipt["inputs"]["high_resolution_archive"]
    require(prior_archive["bytes"] == archive_path.stat().st_size, "archive receipt byte mismatch")
    require(prior_archive["md5"] == archive_contract["md5"], "archive receipt MD5 mismatch")
    archive_md5 = prior_archive["md5"]

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
                protocol["portal_witness_observation"],
            )
            observation_truth = episode_truth["observations"][observation_id]
            target_box = v1.annotation_to_viewport(
                episode_truth["target_door_annotation"],
                render_receipt,
                float(v2_protocol["render_contract"]["annotation_canvas_width"]),
                float(v2_protocol["render_contract"]["annotation_canvas_height"]),
            )
            baseline_evaluation = predecessor.score_output(
                baseline_output,
                masks,
                target_box,
                bool(observation_truth["target_house_number_visible"]),
                v2_protocol["pixel_contract"]["evaluation"],
            )
            successor_evaluation = predecessor.score_output(
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
                    "runtime_output": {"baseline": baseline_output, "successor": successor_output},
                    "truth": {
                        "binding_state": observation_truth["binding_state"],
                        "target_house_number_visible": bool(
                            observation_truth["target_house_number_visible"]
                        ),
                        "target_door_box_xyxy": [round(value, 6) for value in target_box],
                    },
                    "evaluation": {"baseline": baseline_evaluation, "successor": successor_evaluation},
                }
            )
            print(
                json.dumps(
                    {
                        "episode": episode_id,
                        "progress": f"{sequence}/{len(source['episodes'])}",
                        "baseline": baseline_evaluation["outcome"],
                        "successor": successor_evaluation["outcome"],
                        "branch": successor_output["observation_branch"],
                        "witnesses": len(successor_output["portal_witness_set"]),
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
        "L10_SEVN_PORTAL_SPECIFIC_TEXT_WITNESS_GRAPH_FRESH_PAN_DEVELOPMENT_GATE_MET"
        if gate_met
        else "L10_SEVN_PORTAL_SPECIFIC_TEXT_WITNESS_GRAPH_FRESH_PAN_DEVELOPMENT_GATE_NOT_MET"
    )
    result = {
        "schema": "blindassist-l10-sevn-portal-witness-graph-result-v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "claim_scope": "CUMULATIVE_REFERENCE_DISJOINT_SAME_SOURCE_SEVN_FRESH_PAN_DEVELOPMENT",
        "question": protocol["question"],
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "public_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "evaluator_truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "reference_public_sources": [
                {"path": str(path), "sha256": sha256_file(path)} for path in reference_paths
            ],
            "source_protocol": {"path": str(source_protocol_path), "sha256": sha256_file(source_protocol_path)},
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
            "archive_integrity_receipt": {
                "path": str(archive_receipt_path),
                "sha256": sha256_file(archive_receipt_path),
            },
            "high_resolution_archive": {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "md5": archive_md5,
                "verification": "REUSED_HASH_BOUND_PREDECESSOR_RECEIPT_PLUS_PER_MEMBER_CRC_AND_SHA256",
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
            "baseline": predecessor.scenario_metrics(rows, "baseline"),
            "successor": predecessor.scenario_metrics(rows, "successor"),
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
