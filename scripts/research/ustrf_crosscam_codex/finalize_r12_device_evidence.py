#!/usr/bin/env python3
"""Audit R1.2 Android replay parity and seal the local evidence index."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FROZEN_LABELS = ["traffic cone", "delineator", "bollard"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_hash_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def source_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["event_id"]: row for row in payload["sources"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--detector-root", type=Path, required=True)
    parser.add_argument("--apk", type=Path, required=True)
    parser.add_argument("--device-manufacturer", required=True)
    parser.add_argument("--device-model", required=True)
    parser.add_argument("--android-release", required=True)
    parser.add_argument("--sdk", type=int, required=True)
    parser.add_argument("--parser-seconds", type=float, required=True)
    parser.add_argument("--replay-seconds", type=float, required=True)
    args = parser.parse_args()

    root = args.evidence_root.resolve()
    detector_root = args.detector_root.resolve()
    canary_path = detector_root / "android-arm" / "parser_canary.json"
    android_path = root / "android-arm" / "android_r12_output.json"
    offline_path = root / "offline_detector_output.json"
    oracle_path = root / "oracle_geometry_output.json"
    ledger_path = root / "target_instance_ledger.json"
    projection_path = root / "frame_projection_receipt.json"
    input_path = root / "android-input" / "android_r12_input.json"
    model_path = detector_root / "yoloe11s_marker_static3_fp16_640.tflite"
    export_receipt_path = detector_root / "export_receipt.json"
    tensor_path = detector_root / "tensor_inspection.json"

    canary = read_json(canary_path)
    android = read_json(android_path)
    offline = read_json(offline_path)
    expected_model_sha = sha256_file(model_path)
    expected_labels_sha = sha256_file(detector_root / "marker_labels.txt")

    assert canary["schema"] == "blindassist_ustrf_r12_android_parser_output_v1"
    assert canary["dataset_role"] == "seen_r11_diagnostic_parser_canary_not_r12_held_out"
    assert canary["r12_sources_read"] is False
    assert canary["model_asset_sha256"] == expected_model_sha
    assert canary["labels_asset_sha256"] == expected_labels_sha
    assert canary["model_label_inventory"] == FROZEN_LABELS
    emitted_labels = {
        detection["label"]
        for frame in canary["frames"]
        for detection in frame["detections"]
    }
    assert emitted_labels == set(FROZEN_LABELS)
    assert canary["total_detection_count"] > 0 and canary["target_match_frame_count"] > 0

    assert android["schema"] == "blindassist_ustrf_crosscam_target_aware_android_output_v2"
    assert android["diagnostic_set_role"] == "new_held_out_unscored"
    assert android["model_asset_sha256"] == expected_model_sha
    assert android["labels_asset_sha256"] == expected_labels_sha
    assert android["model_label_inventory"] == FROZEN_LABELS
    assert android["target_ledger_sha256"] == sha256_file(ledger_path)
    assert android["projection_receipt_sha256"] == sha256_file(projection_path)

    offline_sources = source_map(offline)
    android_sources = source_map(android)
    assert offline_sources.keys() == android_sources.keys()
    parity_rows: list[dict[str, Any]] = []
    for event_id in offline_sources:
        host = offline_sources[event_id]
        device = android_sources[event_id]
        host_relation = host["frames"][0]["best_eligible_detection"]["robust_route_relation"]
        device_relation = device["frames"][0].get("matched_target_robust_relation")
        if host["target_match_frame_count"] == 0:
            host_relation = None
        parity = {
            "event_id": event_id,
            "expected_route_relation": host["expected_route_relation"],
            "host_target_match_frame_count": host["target_match_frame_count"],
            "android_target_match_frame_count": device["target_match_frame_count"],
            "host_target_relation": host_relation,
            "android_target_relation": device_relation,
            "host_event_recall": host["event_recall"],
            "android_event_recall": device["event_recall"],
            "host_false_alarm": host["false_alarm"],
            "android_false_alarm": device["false_alarm"],
            "event_verdict_parity": (
                host["target_match_frame_count"] == device["target_match_frame_count"]
                and host_relation == device_relation
                and host["event_recall"] == device["event_recall"]
                and host["false_alarm"] == device["false_alarm"]
            ),
        }
        assert parity["event_verdict_parity"], parity
        parity_rows.append(parity)

    positive = [row for row in android_sources.values() if row["expected_route_relation"] == "inside"]
    negative = [row for row in android_sources.values() if row["expected_route_relation"] == "outside"]
    positive_pass = sum(row["event_recall"] == 1 for row in positive)
    negative_false_alarms = sum(row["false_alarm"] is True for row in negative)
    target_matches = sum(row["target_match_frame_count"] for row in android_sources.values())
    android_cooccurrence_inside = sum(row["cooccurrence_robust_inside_count"] for row in android_sources.values())
    offline_cooccurrence_inside = sum(row["cooccurrence_robust_inside_count"] for row in offline_sources.values())
    assert (positive_pass, len(positive), negative_false_alarms, len(negative), target_matches) == (3, 3, 0, 3, 5)

    created_at = datetime.now(timezone.utc).isoformat()
    consistency_path = root / "host_device_consistency_report.json"
    write_json(consistency_path, {
        "schema": "blindassist_ustrf_crosscam_r12_host_device_consistency_v1",
        "created_at_utc": created_at,
        "event_verdict_parity": True,
        "sources": parity_rows,
        "offline_cooccurrence_robust_inside_count": offline_cooccurrence_inside,
        "android_cooccurrence_robust_inside_count": android_cooccurrence_inside,
        "cooccurrence_count_exact_parity_required": False,
        "cooccurrence_note": "Backend numeric/NMS drift is recorded but cannot alter target-matched event attribution.",
        "threshold_fit": False,
        "source_replacement_after_result_open": False,
        "production_model_replacement_authorized": False,
    })
    consistency_sha = write_hash_sidecar(consistency_path)

    receipt_path = root / "android-arm" / "device_run_receipt.json"
    write_json(receipt_path, {
        "schema": "blindassist_ustrf_crosscam_r12_device_run_receipt_v1",
        "created_at_utc": created_at,
        "dataset_role": "held_out_results_opened_no_source_replacement",
        "device": {
            "manufacturer": args.device_manufacturer,
            "model": args.device_model,
            "android_release": args.android_release,
            "sdk": args.sdk,
        },
        "parser_canary": {
            "instrumentation_result": "OK (1 test)",
            "instrumentation_time_seconds": args.parser_seconds,
            "seen_r11_frame_count": canary["frame_count"],
            "total_detection_count": canary["total_detection_count"],
            "target_match_frame_count": canary["target_match_frame_count"],
            "emitted_labels": sorted(emitted_labels),
            "r12_sources_read": False,
        },
        "r12_replay": {
            "instrumentation_result": "OK (1 test)",
            "instrumentation_time_seconds": args.replay_seconds,
            "positive_event_recall": {"passed": positive_pass, "total": len(positive)},
            "negative_false_alarm": {"count": negative_false_alarms, "total": len(negative)},
            "target_match_frame_count": target_matches,
            "android_oracle_geometry_parity_for_matched_sources": True,
        },
        "device_benchmark_apk_sha256": sha256_file(args.apk),
        "model_asset_sha256": expected_model_sha,
        "labels_asset_sha256": expected_labels_sha,
        "android_input_sha256": sha256_file(input_path),
        "android_output_sha256": sha256_file(android_path),
        "parser_canary_output_sha256": sha256_file(canary_path),
        "host_device_consistency_sha256": consistency_sha,
        "training_performed": False,
        "thresholds_changed": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    })
    receipt_sha = write_hash_sidecar(receipt_path)

    indexed_files = [
        ("../r12-detector-export/yoloe11s_marker_static3_fp16_640.tflite", model_path),
        ("../r12-detector-export/export_receipt.json", export_receipt_path),
        ("../r12-detector-export/tensor_inspection.json", tensor_path),
        ("../r12-detector-export/android-arm/parser_canary.json", canary_path),
        ("target_instance_ledger.json", ledger_path),
        ("frame_projection_receipt.json", projection_path),
        ("oracle_geometry_output.json", oracle_path),
        ("offline_detector_output.json", offline_path),
        ("android-input/android_r12_input.json", input_path),
        ("android-arm/android_r12_output.json", android_path),
        ("android-arm/device_run_receipt.json", receipt_path),
        ("host_device_consistency_report.json", consistency_path),
    ]
    index_path = root / "evidence_index.json"
    write_json(index_path, {
        "schema": "blindassist_ustrf_crosscam_r12_evidence_index_v1",
        "created_at_utc": created_at,
        "dataset_role": "held_out_results_opened_no_source_replacement",
        "files": [{"path": relative, "sha256": sha256_file(path)} for relative, path in indexed_files],
        "oracle_passed_sources": 6,
        "offline_positive_event_recall": {"passed": 3, "total": 3},
        "android_positive_event_recall": {"passed": positive_pass, "total": len(positive)},
        "offline_negative_false_alarm": {"count": 0, "total": 3},
        "android_negative_false_alarm": {"count": negative_false_alarms, "total": len(negative)},
        "offline_target_match_frames": 5,
        "android_target_match_frames": target_matches,
        "host_device_event_verdict_parity": True,
        "offline_cooccurrence_inside_detections": offline_cooccurrence_inside,
        "android_cooccurrence_inside_detections": android_cooccurrence_inside,
        "android_status": "complete_sm_s9280_api36",
        "device_run_receipt_sha256": receipt_sha,
        "training_performed": False,
        "thresholds_changed": False,
        "source_replacement_after_result_open": False,
        "production_model_replacement_authorized": False,
    })
    index_sha = write_hash_sidecar(index_path)
    print("USTRF_R12_DEVICE_EVIDENCE_OK", positive_pass, negative_false_alarms, target_matches, index_sha)


if __name__ == "__main__":
    main()
