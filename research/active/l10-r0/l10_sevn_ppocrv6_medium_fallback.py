#!/usr/bin/env python3
"""Evaluate a PP-OCRv6 medium OCR expert behind the frozen SEVN V2 baseline."""

from __future__ import annotations

import argparse
import copy
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
from paddleocr import PaddleOCR
from rapidocr import RapidOCR
from ultralytics import YOLO

from l10_panolab import require, sha256_file, utc_now
import l10_sevn_fresh_pan_panel as panel
import l10_sevn_pixel_replay as v1
import l10_sevn_pixel_topology_replay as v2
import l10_sevn_portal_conditioned_ocr as predecessor
import l10_sevn_portal_witness_graph as witness


SCHEMA = "blindassist-l10-sevn-ppocrv6-medium-fallback-protocol-v1"


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


def verify_model_directory(spec: dict[str, Any]) -> Path:
    root = resolve(spec["path"])
    require(root.is_dir(), f"missing model directory: {root}")
    for filename, expected in spec["sha256"].items():
        path = root / filename
        require(path.is_file(), f"missing model file: {path}")
        require(sha256_file(path) == expected, f"model SHA-256 mismatch: {path}")
    return root


def medium_full_view(
    pipeline: PaddleOCR,
    image: np.ndarray,
    v2_contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    outputs = list(pipeline.predict(image))
    require(len(outputs) == 1, f"unexpected medium OCR result count: {len(outputs)}")
    payload = outputs[0].json["res"]
    texts = payload["rec_texts"]
    scores = payload["rec_scores"]
    polygons = payload["rec_polys"]
    require(len(texts) == len(scores) == len(polygons), "medium OCR row count mismatch")
    rows = [
        {
            "text": str(text),
            "canonical": v1.canonical_token(text),
            "score": round(float(score), 8),
            "box_xyxy": [round(value, 6) for value in v1.polygon_box(polygon)],
            "source_pass": "PPOCRV6_MEDIUM_FULL",
            "observed_passes": ["PPOCRV6_MEDIUM_FULL"],
        }
        for text, score, polygon in zip(texts, scores, polygons)
    ]
    height, width = image.shape[:2]
    deduplicated = v2.deduplicate_ocr(
        rows,
        width,
        height,
        v2_contract["ocr"]["deduplication"],
    )
    return deduplicated, {
        "executed": True,
        "branch": "PPOCRV6_MEDIUM_FULL_VIEW",
        "raw_word_count": len(rows),
        "deduplicated_word_count": len(deduplicated),
        "text_detection_parameters": payload["text_det_params"],
        "seconds": round(time.perf_counter() - started, 6),
    }


def infer_successor(
    image: np.ndarray,
    mission: dict[str, Any],
    primary_ocr: RapidOCR,
    medium_ocr: PaddleOCR,
    portal_model: YOLO,
    postprocessor: Any,
    v2_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    baseline, masks = v2.infer_binding(
        image,
        mission,
        primary_ocr,
        portal_model,
        postprocessor,
        v2_contract,
    )
    successor = copy.deepcopy(baseline)
    successor["observation_branch"] = "V2_RESULT_RETAINED"
    successor["medium_exact_target_candidates"] = []
    successor["medium_receipt"] = {
        "executed": False,
        "reason": "V2_ALREADY_EMITTED_A_PROPOSAL",
        "raw_word_count": 0,
        "deduplicated_word_count": 0,
        "seconds": 0.0,
    }
    if baseline["selected_binding"] is not None:
        return baseline, successor, masks
    if baseline["state"] not in {
        "UNKNOWN_TARGET_TEXT",
        "UNKNOWN_NO_ADMISSIBLE_MASK_TOPOLOGY",
    }:
        successor["medium_receipt"]["reason"] = "BASELINE_DEFICIT_NOT_ELIGIBLE"
        return baseline, successor, masks

    rows, receipt = medium_full_view(medium_ocr, image, v2_contract)
    target = v1.canonical_token(mission["house_number"])
    exact = [
        row
        for row in rows
        if row["canonical"] == target
        and row["score"] >= float(v2_contract["ocr"]["minimum_score"])
    ]
    successor["medium_receipt"] = receipt
    successor["medium_ocr_words"] = rows
    successor["medium_exact_target_candidates"] = exact
    if not exact:
        successor["observation_branch"] = "PPOCRV6_MEDIUM_NO_EXACT_TOKEN"
        return baseline, successor, masks

    portals = [
        {"summary": summary, "mask": masks[summary["candidate_id"]]}
        for summary in baseline["portal_mask_candidates"]
    ]
    candidate = predecessor.bind_from_rows(exact, portals, target, v2_contract)
    candidate["runtime_receipt"] = copy.deepcopy(baseline["runtime_receipt"])
    candidate["medium_receipt"] = receipt
    candidate["medium_ocr_words"] = rows
    candidate["medium_exact_target_candidates"] = exact
    if candidate["selected_binding"] is None:
        candidate["observation_branch"] = "PPOCRV6_MEDIUM_EXACT_NO_ADMISSIBLE_TOPOLOGY"
    else:
        candidate["state"] = "PIXEL_BOUND_PPOCRV6_MEDIUM_MASK_PORTAL_PROPOSAL"
        candidate["selected_binding"]["authority_edge"] = {
            "type": "PPOCRV6_MEDIUM_EXACT_TOKEN_TO_UNCHANGED_MASK_TOPOLOGY",
            "lexical_source": "PPOCRV6_MEDIUM_DET_REC_ONNX",
            "exact_candidate_count": len(exact),
        }
        candidate["observation_branch"] = "PPOCRV6_MEDIUM_EXACT_TO_MASK_TOPOLOGY"
    return baseline, candidate, masks


def evaluate_gates(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    successor: dict[str, Any],
    disjointness: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    exact_gain = successor["exact_target_text_when_truth_visible"] - baseline["exact_target_text_when_truth_visible"]
    exact_rate_gain = successor["exact_target_text_when_truth_visible_rate"] - baseline["exact_target_text_when_truth_visible_rate"]
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
    emitted = successor["correct_target_door_bindings"] + successor["wrong_door_bindings"]
    precision = round(successor["correct_target_door_bindings"] / emitted, 6) if emitted else 0.0
    delta = {
        "exact_target_text_gain": exact_gain,
        "exact_target_text_rate_gain": round(exact_rate_gain, 6),
        "correct_target_door_binding_gain": correct_gain,
        "correct_target_door_binding_rate_gain": round(successor["correct_target_door_binding_rate"] - baseline["correct_target_door_binding_rate"], 6),
        "wrong_door_binding_delta": wrong_delta,
        "successor_binding_precision": precision,
        "baseline_correct_bindings_retained": retained,
        "baseline_correct_bindings_total": len(baseline_correct_ids),
        "medium_branch_executed": sum(row["runtime_output"]["successor"]["medium_receipt"]["executed"] for row in rows),
        "medium_exact_recoveries": sum(
            not row["evaluation"]["baseline"]["exact_target_text"]
            and row["evaluation"]["successor"]["exact_target_text"]
            for row in rows
        ),
        "medium_proposals": sum(
            row["runtime_output"]["successor"]["observation_branch"]
            == "PPOCRV6_MEDIUM_EXACT_TO_MASK_TOPOLOGY"
            for row in rows
        ),
    }
    gates = {
        "all_episodes_rendered": len(rows) == int(gate["episode_count"]),
        "minimum_truth_visible_house_number_opportunities": successor["truth_visible_house_number_opportunities"] >= int(gate["minimum_truth_visible_house_number_opportunities"]),
        "minimum_successor_visible_text_exact_ocr_rate": successor["exact_target_text_when_truth_visible_rate"] >= float(gate["minimum_successor_visible_text_exact_ocr_rate"]),
        "minimum_exact_ocr_rate_gain": exact_rate_gain >= float(gate["minimum_exact_ocr_rate_gain"]),
        "minimum_successor_correct_binding_rate": successor["correct_target_door_binding_rate"] >= float(gate["minimum_successor_correct_binding_rate"]),
        "minimum_correct_binding_gain": correct_gain >= int(gate["minimum_correct_binding_gain"]),
        "minimum_successor_binding_precision": precision >= float(gate["minimum_successor_binding_precision"]),
        "maximum_successor_wrong_bindings": successor["wrong_door_bindings"] <= int(gate["maximum_successor_wrong_bindings"]),
        "maximum_wrong_binding_delta": wrong_delta <= int(gate["maximum_wrong_binding_delta"]),
        "retain_all_baseline_correct_bindings": retained == len(baseline_correct_ids),
        "cumulative_reference_address_disjoint": disjointness["reference_address_overlap_count"] == 0,
        "cumulative_reference_panorama_frame_disjoint": disjointness["reference_frame_overlap_count"] == 0,
        "within_panel_cross_episode_frame_unique": disjointness["within_panel_cross_episode_reused_frames"] == {},
    }
    return delta, gates


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output.resolve()
    require(not output_path.exists(), f"refusing to overwrite result: {output_path}")
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    require(protocol.get("schema") == SCHEMA, "protocol schema mismatch")
    require(sha256_file(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "evaluator hash mismatch")

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
    panel.validate_pan_public_truth(source, truth, int(protocol["cohort"]["episodes_per_scenario"]))
    require(selection_receipt["outputs"]["public_source"]["sha256"] == sha256_file(source_path), "selection receipt/public source mismatch")
    require(selection_receipt["outputs"]["evaluator_truth"]["sha256"] == sha256_file(truth_path), "selection receipt/truth mismatch")

    addresses, frames = predecessor.cohort_identity(source)
    reference_addresses: set[tuple[str, str]] = set()
    reference_frames: set[int] = set()
    for reference in references:
        current_addresses, current_frames = predecessor.cohort_identity(reference)
        reference_addresses.update(current_addresses)
        reference_frames.update(current_frames)
    reused_frames = predecessor.cross_episode_frame_reuse(source)
    disjointness = {
        "panel_distinct_address_count": len(addresses),
        "panel_distinct_panorama_frame_count": len(frames),
        "reference_address_count": len(reference_addresses),
        "reference_panorama_frame_count": len(reference_frames),
        "reference_address_overlap_count": len(addresses.intersection(reference_addresses)),
        "reference_frame_overlap_count": len(frames.intersection(reference_frames)),
        "within_panel_cross_episode_reused_frames": reused_frames,
    }
    require(disjointness["reference_address_overlap_count"] == 0, "reference address overlap")
    require(disjointness["reference_frame_overlap_count"] == 0, "reference frame overlap")
    require(not reused_frames, f"within-panel frame reuse: {reused_frames}")

    archive_path = args.highres_archive.resolve()
    archive_contract = v2_protocol["high_resolution_archive"]
    require(archive_path.is_file(), f"missing high-resolution archive: {archive_path}")
    require(archive_path.stat().st_size == int(archive_contract["bytes"]), "archive byte mismatch")
    prior_archive = archive_receipt["inputs"]["high_resolution_archive"]
    require(prior_archive["bytes"] == archive_path.stat().st_size, "archive receipt byte mismatch")
    require(prior_archive["md5"] == archive_contract["md5"], "archive receipt MD5 mismatch")

    versions = {
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "rapidocr": importlib.metadata.version("rapidocr"),
        "onnxruntime-gpu": importlib.metadata.version("onnxruntime-gpu"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "numpy": importlib.metadata.version("numpy"),
        "paddleocr": importlib.metadata.version("paddleocr"),
        "paddlex": importlib.metadata.version("paddlex"),
    }
    require(versions == protocol["runtime"]["versions"], f"runtime version mismatch: {versions}")
    require(torch.cuda.is_available(), "CUDA unavailable for portal model")

    model_path = verify_sha256(v2_protocol["models"]["portal_model"]["weights"])
    postprocessor_path = verify_sha256(v2_protocol["models"]["portal_model"]["postprocessor"])
    verify_sha256(v2_protocol["models"]["portal_model"]["model_card"])
    primary_root = resolve(v2_protocol["models"]["ocr"]["model_root"])
    for filename, expected in v2_protocol["models"]["ocr"]["sha256"].items():
        require(sha256_file(primary_root / filename) == expected, f"primary OCR model mismatch: {filename}")
    medium_det_root = verify_model_directory(protocol["models"]["medium_detection"])
    medium_rec_root = verify_model_directory(protocol["models"]["medium_recognition"])

    postprocessor = v1.load_module(postprocessor_path)
    portal_model = YOLO(str(model_path))
    require(portal_model.task == "segment", "portal model task mismatch")
    primary_ocr = RapidOCR(params={
        "Global.model_root_dir": str(primary_root),
        "Global.log_level": "error",
        "EngineConfig.onnxruntime.intra_op_num_threads": int(v2_protocol["pixel_contract"]["ocr"]["intra_op_threads"]),
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })
    medium_ocr = PaddleOCR(
        text_detection_model_dir=str(medium_det_root),
        text_recognition_model_dir=str(medium_rec_root),
        engine="onnxruntime",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    rows = []
    started = time.perf_counter()
    action_map = v2_protocol["action_control"]["scenario_to_action"]
    member_pattern = archive_contract["member_pattern"]
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
            frame_id = int(observation["frame_id"])
            member_name = member_pattern.format(frame_id=frame_id)
            info = archive_zip.getinfo(member_name)
            encoded = archive_zip.read(info)
            panorama = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            require(panorama is not None, f"{episode_id}: panorama decode failed")
            require(list(panorama.shape) == v2_protocol["render_contract"]["panorama_shape_hwc"], "panorama shape mismatch")
            viewport, render_receipt = v1.render_viewport(
                panorama,
                float(observation["camera_pose"]["panorama_angle_degrees"]),
                float(observation["viewport_headings_degrees"][0]),
                float(observation["horizontal_fov_degrees"]),
            )
            render_receipt.update({
                "archive_member": member_name,
                "archive_member_crc32": f"{info.CRC:08x}",
                "archive_member_bytes": info.file_size,
                "archive_member_sha256": hashlib.sha256(encoded).hexdigest(),
            })
            baseline_output, successor_output, masks = infer_successor(
                viewport,
                episode["mission"],
                primary_ocr,
                medium_ocr,
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
            visible = bool(observation_truth["target_house_number_visible"])
            baseline_evaluation = predecessor.score_output(baseline_output, masks, target_box, visible, v2_protocol["pixel_contract"]["evaluation"])
            successor_evaluation = predecessor.score_output(successor_output, masks, target_box, visible, v2_protocol["pixel_contract"]["evaluation"])
            rows.append({
                "episode_id": episode_id,
                "scenario_class": scenario,
                "frozen_action": action,
                "observation_id": observation_id,
                "mission": episode["mission"],
                "render_receipt": render_receipt,
                "runtime_output": {"baseline": baseline_output, "successor": successor_output},
                "truth": {
                    "binding_state": observation_truth["binding_state"],
                    "target_house_number_visible": visible,
                    "target_door_box_xyxy": [round(value, 6) for value in target_box],
                },
                "evaluation": {"baseline": baseline_evaluation, "successor": successor_evaluation},
            })
            print(json.dumps({
                "episode": episode_id,
                "progress": f"{sequence}/{len(source['episodes'])}",
                "baseline": baseline_evaluation["outcome"],
                "successor": successor_evaluation["outcome"],
                "branch": successor_output["observation_branch"],
                "medium_exact": len(successor_output["medium_exact_target_candidates"]),
            }), flush=True)

    actual_device = str(next(portal_model.model.parameters()).device)
    require(actual_device.startswith("cuda"), f"silent portal fallback: {actual_device}")
    baseline_metrics = witness.summarize(rows, "baseline")
    successor_metrics = witness.summarize(rows, "successor")
    delta, gates = evaluate_gates(rows, baseline_metrics, successor_metrics, disjointness, protocol["gate"])
    gate_met = all(gates.values())
    decision = "L10_SEVN_PPOCRV6_MEDIUM_EXPERT_FRESH_PAN_DEVELOPMENT_GATE_MET" if gate_met else "L10_SEVN_PPOCRV6_MEDIUM_EXPERT_FRESH_PAN_DEVELOPMENT_GATE_NOT_MET"
    result = {
        "schema": "blindassist-l10-sevn-ppocrv6-medium-fallback-result-v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "claim_scope": "CUMULATIVE_REFERENCE_DISJOINT_SAME_SOURCE_SEVN_FRESH_PAN_DEVELOPMENT",
        "question": protocol["question"],
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "public_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "evaluator_truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "reference_public_sources": [{"path": str(path), "sha256": sha256_file(path)} for path in reference_paths],
            "source_protocol": {"path": str(source_protocol_path), "sha256": sha256_file(source_protocol_path)},
            "source_materializer": {"path": str(source_materializer_path), "sha256": sha256_file(source_materializer_path)},
            "selection_receipt": {"path": str(selection_receipt_path), "sha256": sha256_file(selection_receipt_path)},
            "v2_protocol": {"path": str(v2_protocol_path), "sha256": sha256_file(v2_protocol_path)},
            "v2_evaluator": {"path": str(v2_evaluator_path), "sha256": sha256_file(v2_evaluator_path)},
            "archive_integrity_receipt": {"path": str(archive_receipt_path), "sha256": sha256_file(archive_receipt_path)},
            "high_resolution_archive": {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "md5": prior_archive["md5"],
                "verification": "REUSED_HASH_BOUND_PREDECESSOR_RECEIPT_PLUS_PER_MEMBER_CRC_AND_SHA256",
            },
            "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
        "runtime": {
            "versions": versions,
            "torch_cuda": torch.version.cuda,
            "portal_device": actual_device,
            "device_name": torch.cuda.get_device_name(0),
            "medium_ocr_device": "cpu",
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
            "panorama_viewport_mask_or_overlay_files_written": 0,
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
        "baseline_metrics": result["baseline_metrics"],
        "successor_metrics": result["successor_metrics"],
        "delta": result["delta"],
        "gates": result["gates"],
        "wall_seconds": result["wall_seconds"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
