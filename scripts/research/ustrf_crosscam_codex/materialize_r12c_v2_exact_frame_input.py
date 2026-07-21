#!/usr/bin/env python3
"""Materialize the Bangkok-replaced R1.2c v2 exact-frame replay input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image

from contract import load_json, sha256_file, write_json
from diagnostic_contract import require


INPUT_SCHEMA = "blindassist_ustrf_crosscam_exact_frame_android_input_v1"
REMOTE_ROOT = "ustrf-crosscam-r12c"
EVENT_ID = "bangkok_tactile_cone_intrusion"


def raw_rgb_sha256(path: Path) -> str:
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--parent-input", type=Path, required=True)
    parser.add_argument("--continuous-v2", type=Path, required=True)
    parser.add_argument("--replacement-contract", type=Path, required=True)
    parser.add_argument("--oracle-result", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    parent = load_json(args.parent_input)
    continuous = load_json(args.continuous_v2)
    replacement = load_json(args.replacement_contract)
    oracle = load_json(args.oracle_result)
    canary = load_json(args.canary)
    require(parent.get("schema") == INPUT_SCHEMA, "parent exact-frame input schema mismatch")
    require(parent.get("dataset_role") == "seen_diagnostic_not_held_out", "parent role drifted")
    require(oracle.get("all_positive_truth_geometry_consistent") is True, "R1.2c v2 oracle did not pass")
    require(oracle["authorization"]["london_768_candidate_execution_authorized"] is True,
            "London 768 candidate is not authorized")
    require(canary["gate"]["canary_passed"] is True, "London 768 mechanical canary did not pass")
    require(canary["candidate_id"] == "r12c_c1_sameweights_fp16_768_gpu_london_only",
            "unexpected canary candidate")
    require(canary["model_sha256"] == sha256_file(args.model), "canary/model SHA mismatch")

    old_sources = {row["event_id"]: row for row in parent["sources"]}
    require(len(old_sources) == 12 and "japan_path_intrusion" in old_sources, "parent inventory drifted")
    old_sources.pop("japan_path_intrusion")
    event = replacement["event"]
    geometry = replacement["geometry_contract"]
    video = (repo / replacement["source"]["local_video_path"]).resolve()
    require(video.is_file() and sha256_file(video) == replacement["source"]["video_sha256"],
            "Bangkok source video drifted")

    output_dir = args.output_dir.resolve()
    capture = cv2.VideoCapture(str(video))
    require(capture.isOpened(), "OpenCV cannot open Bangkok video")
    frame_rows: list[dict[str, object]] = []
    staged_files: list[dict[str, str]] = []
    exact_by_timestamp: dict[int, Path] = {}
    try:
        for timestamp_ms in range(event["window_ms"][0], event["window_ms"][1] + 1, 500):
            path = output_dir / "frames" / EVENT_ID / f"f{timestamp_ms:09d}.png"
            require(not path.exists(), f"refusing to overwrite exact frame: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, image = capture.read()
            require(ok and image is not None, f"Bangkok/{timestamp_ms}: OpenCV decode failed")
            require(cv2.imwrite(str(path), image), f"Bangkok/{timestamp_ms}: PNG write failed")
            height, width = image.shape[:2]
            remote = f"{REMOTE_ROOT}/frames/{EVENT_ID}/{path.name}"
            frame_rows.append({"timestamp_ms": timestamp_ms, "image_path": remote,
                               "image_sha256": sha256_file(path), "width": width, "height": height})
            staged_files.append({"host_path": str(path), "device_relative_path": remote,
                                 "sha256": sha256_file(path)})
            exact_by_timestamp[timestamp_ms] = path
    finally:
        capture.release()

    anchor_rows = []
    for anchor in geometry["anchors"]:
        timestamp = int(anchor["timestamp_ms"])
        exact = exact_by_timestamp[timestamp]
        review = (repo / anchor["image_path"]).resolve()
        with Image.open(exact) as exact_image, Image.open(review) as review_image:
            require(review_image.size == (exact_image.width * 2, exact_image.height * 2),
                    f"Bangkok/{timestamp}: review/exact dimensions do not preserve the frozen 2x basis")
        bbox = anchor.get("target_bbox_xyxy_px")
        bbox_norm = None if bbox is None else [
            round(bbox[0] / geometry["frame_width"], 9),
            round(bbox[1] / geometry["frame_height"], 9),
            round(bbox[2] / geometry["frame_width"], 9),
            round(bbox[3] / geometry["frame_height"], 9),
        ]
        anchor_rows.append({
            "frame_id": anchor["frame_id"],
            "timestamp_ms": timestamp,
            "frame_sha256": sha256_file(exact),
            "frame_width": geometry["frame_width"],
            "frame_height": geometry["frame_height"],
            "visibility": "visible" if anchor.get("contact_xy_px") is not None else "occluded",
            "bbox_xyxy_norm": bbox_norm,
            "contact_xy_norm": None if anchor.get("contact_xy_px") is None else [
                round(anchor["contact_xy_px"][0] / geometry["frame_width"], 9),
                round(anchor["contact_xy_px"][1] / geometry["frame_height"], 9),
            ],
            "oracle_role": anchor["role"],
            "reference_review_image_sha256": anchor["image_sha256"],
            "reference_review_is_exact_transport_pixels": raw_rgb_sha256(exact) == raw_rgb_sha256(review),
            "coordinate_scaling_rule": geometry["runtime_scaling_rule"],
        })

    route_anchor = next(row for row in geometry["anchors"] if row["timestamp_ms"] == 336000)
    bangkok = {
        "event_id": EVENT_ID,
        "source_id": replacement["source"]["source_id"],
        "parent_round": "r12c_seen_replacement",
        "dataset_role": "seen_diagnostic_not_held_out",
        "expected_class": "positive",
        "clip_window_ms": event["window_ms"],
        "alertable_start_ms": event["alertable_start_ms"],
        "known_not_visible_from_ms": event["route_clear_from_ms"],
        "known_not_visible_until_ms": None,
        "gate_eligible": True,
        "diagnostic_role": "gate_diagnostic",
        "target_instance_id": event["target_instance_id"],
        "detector_label_allowlist": ["traffic cone"],
        "expected_route_relation": "inside",
        "target_anchors": anchor_rows,
        "primary_anchor_timestamp_ms": 336000,
        "route_polygon_xy_norm": route_anchor["route_polygon_xy_norm"],
        "route_polygon_anchor_timestamp_ms": 336000,
        "route_proxy_is_geometry_truth": False,
        "source_video_sha256": sha256_file(video),
        "frame_transport": "hash_bound_exact_timestamp_png_v1",
        "frames": frame_rows,
    }
    sources = [bangkok if row["event_id"] == "japan_path_intrusion" else row for row in parent["sources"]]
    require(len(sources) == 12 and len({row["event_id"] for row in sources}) == 12, "v2 source inventory invalid")
    expected_ids = {row["event_id"] for row in continuous["events"]}
    require({row["event_id"] for row in sources} == expected_ids, "exact-frame inventory differs from continuous v2")

    android_input = {
        "schema": INPUT_SCHEMA,
        "contract_id": continuous["contract_id"],
        "transport_contract_sha256": sha256_file(args.continuous_v2),
        "dataset_role": "seen_diagnostic_not_held_out",
        "candidate": {
            "candidate_id": canary["candidate_id"],
            "selection_rule": "single_preregistered_candidate_after_r12c_v2_oracle",
            "canary_output_sha256": sha256_file(args.canary),
            "model_sha256": sha256_file(args.model),
            "input_size": 768,
            "execution_backend": "gpu_delegate",
            "confidence_threshold": 0.05,
            "target_anchor_iou_threshold": 0.30,
            "nms_iou_threshold": 0.45,
        },
        "replay_contract": parent["replay_contract"],
        "device_gate": parent["device_gate"],
        "event_gate": {**parent["event_gate"], "positive_event_recall_at_least": 1.0},
        "sources": sources,
        "authority": {"new_held_out_read": False, "r13_sources_read": False, "benchmark_only": True,
                      "training_authorized": False, "production_model_replacement_authorized": False},
    }
    input_path = output_dir / "android-input" / "android_r12c_v2_exact_frame_input.json"
    write_json(input_path, android_input)
    input_remote = f"{REMOTE_ROOT}/input.json"
    staged_files.append({"host_path": str(input_path), "device_relative_path": input_remote,
                         "sha256": sha256_file(input_path)})
    receipt_path = output_dir / "android-input" / "host_staging_receipt.json"
    write_json(receipt_path, {
        "schema": "blindassist_ustrf_crosscam_exact_frame_staging_v2",
        "input_sha256": sha256_file(input_path),
        "parent_input_sha256": sha256_file(args.parent_input),
        "continuous_v2_sha256": sha256_file(args.continuous_v2),
        "oracle_result_sha256": sha256_file(args.oracle_result),
        "model_sha256": sha256_file(args.model),
        "canary_sha256": sha256_file(args.canary),
        "new_held_out_read": False,
        "r13_slot_consumed": False,
        "source_count": len(sources),
        "new_frame_count": len(frame_rows),
        "reused_parent_source_count": 11,
        "files": staged_files,
    })
    print(json.dumps({"ok": True, "input": str(input_path), "input_sha256": sha256_file(input_path),
                      "source_count": len(sources), "new_frame_count": len(frame_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
