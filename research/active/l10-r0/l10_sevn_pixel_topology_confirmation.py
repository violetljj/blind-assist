#!/usr/bin/env python3
"""Run frozen SEVN V2 on an address- and panorama-disjoint same-source panel.

The evaluator imports the frozen V2 inference and truth-scoring functions by
hash. New SEVN panoramas are streamed from the official ZIP and every crop,
OCR tile, raster mask, and overlap calculation remains in memory. Only the
detailed confirmation result is written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from rapidocr import RapidOCR
from ultralytics import YOLO

from l10_panolab import require, sha256_file, utc_now
from l10_sevn_panolab import validate_public, validate_truth
import l10_sevn_pixel_replay as v1
import l10_sevn_pixel_topology_replay as v2


SCHEMA = "blindassist-l10-sevn-v2-address-panorama-disjoint-confirmation-protocol-v1"


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
        observation_ids = {episode["start_observation_id"]}
        observation_ids.update(
            edge["to_observation_id"]
            for edge in episode["transitions"][episode["start_observation_id"]].values()
        )
        frames = {int(public["observations"][item]["frame_id"]) for item in observation_ids}
        for frame_id in frames:
            owners.setdefault(frame_id, []).append(episode_id)
    return {str(frame): episode_ids for frame, episode_ids in owners.items() if len(episode_ids) > 1}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    truth_visible = sum(row["evaluation"]["target_house_number_visible"] for row in rows)
    exact = sum(row["evaluation"]["exact_target_text"] for row in rows)
    exact_visible = sum(
        row["evaluation"]["exact_target_text"]
        and row["evaluation"]["target_house_number_visible"]
        for row in rows
    )
    correct = sum(row["evaluation"]["outcome"] == "CORRECT_TARGET_DOOR" for row in rows)
    wrong = sum(row["evaluation"]["outcome"] == "WRONG_DOOR" for row in rows)
    return {
        "episode_count": total,
        "truth_visible_house_number_opportunities": truth_visible,
        "exact_target_text": exact,
        "exact_target_text_when_truth_visible": exact_visible,
        "exact_target_text_when_truth_visible_rate": round(exact_visible / truth_visible, 6)
        if truth_visible
        else 0.0,
        "exact_target_text_full_pass": sum(
            row["runtime_output"]["exact_target_text_full_pass"] for row in rows
        ),
        "exact_target_text_tile_pass": sum(
            row["runtime_output"]["exact_target_text_tile_pass"] for row in rows
        ),
        "exact_target_text_tile_only_gain": sum(
            row["runtime_output"]["exact_target_text_tile_pass"]
            and not row["runtime_output"]["exact_target_text_full_pass"]
            for row in rows
        ),
        "joined_mask_topology_proposal": sum(
            row["runtime_output"]["selected_binding"] is not None for row in rows
        ),
        "correct_target_door_bindings": correct,
        "correct_target_door_binding_rate": round(correct / total, 6),
        "wrong_door_bindings": wrong,
        "unknown": total - correct - wrong,
    }


def evaluate_gates(
    metrics: dict[str, Any],
    disjointness: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, bool]:
    return {
        "all_episodes_rendered": metrics["episode_count"] == int(gate["episode_count"]),
        "minimum_truth_visible_house_number_opportunities": (
            metrics["truth_visible_house_number_opportunities"]
            >= int(gate["minimum_truth_visible_house_number_opportunities"])
        ),
        "minimum_visible_text_exact_ocr_rate": (
            metrics["exact_target_text_when_truth_visible_rate"]
            >= float(gate["minimum_visible_text_exact_ocr_rate"])
        ),
        "minimum_correct_target_door_bindings": (
            metrics["correct_target_door_bindings"]
            >= int(gate["minimum_correct_target_door_bindings"])
        ),
        "maximum_wrong_door_bindings": (
            metrics["wrong_door_bindings"] <= int(gate["maximum_wrong_door_bindings"])
        ),
        "reference_address_disjoint": disjointness["reference_address_overlap_count"] == 0,
        "reference_panorama_frame_disjoint": disjointness["reference_frame_overlap_count"] == 0,
        "within_panel_cross_episode_frame_unique": (
            disjointness["within_panel_cross_episode_reused_frames"] == {}
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output.resolve()
    require(not output_path.exists(), f"refusing to overwrite confirmation result: {output_path}")
    protocol_path = args.protocol.resolve()
    protocol = v1.read_json(protocol_path)
    require(protocol.get("schema") == SCHEMA, "confirmation protocol schema mismatch")
    require(
        sha256_file(Path(__file__).resolve()) == protocol["evaluator"]["sha256"],
        "confirmation evaluator hash mismatch",
    )

    source_path = v1.verify_sha256(protocol["frozen_inputs"]["public_source"])
    truth_path = v1.verify_sha256(protocol["frozen_inputs"]["evaluator_truth"])
    reference_path = v1.verify_sha256(protocol["frozen_inputs"]["reference_public_source"])
    source_protocol_path = v1.verify_sha256(protocol["frozen_inputs"]["source_protocol"])
    source_materializer_path = v1.verify_sha256(protocol["frozen_inputs"]["source_materializer"])
    selection_receipt_path = v1.verify_sha256(protocol["frozen_inputs"]["selection_receipt"])
    v2_protocol_path = v1.verify_sha256(protocol["frozen_v2"]["protocol"])
    v2_evaluator_path = v1.verify_sha256(protocol["frozen_v2"]["evaluator"])
    require(Path(v2.__file__).resolve() == v2_evaluator_path.resolve(), "loaded V2 evaluator path mismatch")
    v2_protocol = v1.read_json(v2_protocol_path)
    require(
        v2_protocol.get("schema") == "blindassist-l10-sevn-multiscale-mask-topology-protocol-v2",
        "frozen V2 protocol schema mismatch",
    )
    require(
        v2_protocol["evaluator"]["sha256"] == protocol["frozen_v2"]["evaluator"]["sha256"],
        "V2 protocol/evaluator hash disagreement",
    )

    source = v1.read_json(source_path)
    truth = v1.read_json(truth_path)
    reference = v1.read_json(reference_path)
    selection_receipt = v1.read_json(selection_receipt_path)
    per_scenario = int(protocol["cohort"]["episodes_per_scenario"])
    validate_public(source, per_scenario)
    validate_truth(truth, source, per_scenario)
    validate_public(reference, int(protocol["reference_panel"]["episodes_per_scenario"]))
    require(
        selection_receipt["outputs"]["public_source"]["sha256"] == sha256_file(source_path),
        "selection receipt/public source mismatch",
    )
    require(
        selection_receipt["outputs"]["evaluator_truth"]["sha256"] == sha256_file(truth_path),
        "selection receipt/truth mismatch",
    )
    require(
        len(source["episodes"]) == int(protocol["gate"]["episode_count"]),
        "confirmation episode count mismatch",
    )

    addresses, frames = cohort_identity(source)
    reference_addresses, reference_frames = cohort_identity(reference)
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
    require(archive_path.stat().st_size == int(archive_contract["bytes"]), "archive byte size mismatch")
    archive_md5 = v1.md5_file(archive_path)
    require(archive_md5 == archive_contract["md5"], "archive Zenodo MD5 mismatch")
    print(json.dumps({"archive_md5_verified": archive_md5}), flush=True)

    versions = {
        "torch": importlib.metadata.version("torch"),
        "ultralytics": ultralytics.__version__,
        "rapidocr": importlib.metadata.version("rapidocr"),
        "onnxruntime": importlib.metadata.version("onnxruntime"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "numpy": importlib.metadata.version("numpy"),
    }
    require(versions == v2_protocol["runtime"]["versions"], f"runtime version mismatch: {versions}")
    require(torch.cuda.is_available(), "CUDA unavailable")

    model_path = v1.verify_sha256(v2_protocol["models"]["portal_model"]["weights"])
    postprocessor_path = v1.verify_sha256(v2_protocol["models"]["portal_model"]["postprocessor"])
    v1.verify_sha256(v2_protocol["models"]["portal_model"]["model_card"])
    ocr_model_root = v1.resolve(v2_protocol["models"]["ocr"]["model_root"])
    for filename, expected in v2_protocol["models"]["ocr"]["sha256"].items():
        require(sha256_file(ocr_model_root / filename) == expected, f"OCR model hash mismatch: {filename}")

    postprocessor = v1.load_module(postprocessor_path)
    portal_model = YOLO(str(model_path))
    expected_names = {0: "doorway", 1: "door", 2: "people", 3: "window", 4: "mirror"}
    require(
        portal_model.task == "segment" and portal_model.names == expected_names,
        "portal model ontology mismatch",
    )
    ocr_engine = RapidOCR(params={
        "Global.model_root_dir": str(ocr_model_root),
        "Global.log_level": "error",
        "EngineConfig.onnxruntime.intra_op_num_threads": int(
            v2_protocol["pixel_contract"]["ocr"]["intra_op_threads"]
        ),
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })

    rows = []
    started = time.perf_counter()
    action_map = v2_protocol["action_control"]["scenario_to_action"]
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
            member_name = archive_contract["member_pattern"].format(frame_id=frame_id)
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
            render_receipt.update({
                "archive_member": member_name,
                "archive_member_crc32": f"{info.CRC:08x}",
                "archive_member_bytes": info.file_size,
                "archive_member_sha256": hashlib.sha256(encoded).hexdigest(),
            })

            runtime_output, candidate_masks = v2.infer_binding(
                viewport,
                episode["mission"],
                ocr_engine,
                portal_model,
                postprocessor,
                v2_protocol["pixel_contract"],
            )
            observation_truth = episode_truth["observations"][observation_id]
            target_box = v1.annotation_to_viewport(
                episode_truth["target_door_annotation"],
                render_receipt,
                float(v2_protocol["render_contract"]["annotation_canvas_width"]),
                float(v2_protocol["render_contract"]["annotation_canvas_height"]),
            )
            selected = runtime_output["selected_binding"]
            if selected is None:
                mask_metrics = None
                outcome = "UNKNOWN"
            else:
                selected_mask = candidate_masks[selected["portal"]["candidate_id"]]
                mask_metrics = v2.truth_mask_metrics(
                    selected_mask,
                    target_box,
                    v2_protocol["pixel_contract"]["evaluation"],
                )
                outcome = "CORRECT_TARGET_DOOR" if mask_metrics["correct"] else "WRONG_DOOR"
            evaluation = {
                "truth_binding_state": observation_truth["binding_state"],
                "target_house_number_visible": bool(observation_truth["target_house_number_visible"]),
                "exact_target_text": bool(runtime_output["exact_target_text_candidates"]),
                "target_door_box_xyxy": [round(value, 6) for value in target_box],
                "selected_mask_truth_metrics": mask_metrics,
                "outcome": outcome,
            }
            rows.append({
                "episode_id": episode_id,
                "scenario_class": scenario,
                "frozen_action": action,
                "observation_id": observation_id,
                "mission": episode["mission"],
                "render_receipt": render_receipt,
                "runtime_output": runtime_output,
                "evaluation": evaluation,
            })
            print(json.dumps({
                "episode": episode_id,
                "progress": f"{sequence}/{len(source['episodes'])}",
                "action": action,
                "state": runtime_output["state"],
                "outcome": outcome,
                "tile_only_ocr": (
                    runtime_output["exact_target_text_tile_pass"]
                    and not runtime_output["exact_target_text_full_pass"]
                ),
            }, ensure_ascii=False), flush=True)

    actual_device = str(next(portal_model.model.parameters()).device)
    require(actual_device.startswith("cuda"), f"silent model fallback: {actual_device}")
    metrics = summarize(rows)
    gates = evaluate_gates(metrics, disjointness, protocol["gate"])
    decision = (
        "L10_SEVN_V2_ADDRESS_PANORAMA_DISJOINT_SAME_SOURCE_CONFIRMATION_GATE_MET"
        if all(gates.values())
        else "L10_SEVN_V2_ADDRESS_PANORAMA_DISJOINT_SAME_SOURCE_CONFIRMATION_GATE_NOT_MET"
    )
    result = {
        "schema": "blindassist-l10-sevn-v2-address-panorama-disjoint-confirmation-result-v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "claim_scope": "ADDRESS_AND_PANORAMA_DISJOINT_SAME_SOURCE_SEVN_DEVELOPMENT_CONFIRMATION",
        "question": protocol["question"],
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "public_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "evaluator_truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "reference_public_source": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
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
            "frozen_v2_protocol": {
                "path": str(v2_protocol_path),
                "sha256": sha256_file(v2_protocol_path),
            },
            "frozen_v2_evaluator": {
                "path": str(v2_evaluator_path),
                "sha256": sha256_file(v2_evaluator_path),
            },
            "high_resolution_archive": {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "md5": archive_md5,
            },
            "confirmation_evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "runtime": {
            "versions": versions,
            "torch_cuda": torch.version.cuda,
            "device": actual_device,
            "device_name": torch.cuda.get_device_name(
                int(v2_protocol["pixel_contract"]["portal_model"]["cuda_device"])
            ),
        },
        "frozen_algorithm_boundary": protocol["frozen_v2"]["boundary"],
        "disjointness": disjointness,
        "metrics": metrics,
        "scenario_metrics": v2.scenario_metrics(rows),
        "gates": gates,
        "gate_met": all(gates.values()),
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
    print(json.dumps({
        "decision": result["decision"],
        "disjointness": result["disjointness"],
        "metrics": result["metrics"],
        "gates": result["gates"],
        "wall_seconds": result["wall_seconds"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
