from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from run_host_canonical_coverage import RAW_BYTES_PER_FRAME, read_exact
from run_host_coverage import decode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device-receipt", type=Path, required=True)
    parser.add_argument("--canonical-raw", type=Path, required=True)
    parser.add_argument("--host-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    detector = config["detector"]
    labels = Path(detector["labels_path"]).read_text(encoding="utf-8").splitlines()
    device = json.loads(args.device_receipt.read_text(encoding="utf-8"))
    host = json.loads(args.host_ledger.read_text(encoding="utf-8"))
    if len(device["frames"]) != len(host["frames"]):
        raise ValueError("device/host frame count mismatch")
    identity_exact = threshold_state_equal = 0
    detection_count_equal = pre_nms_count_equal = 0
    max_confidence_delta = max_box_delta = 0.0
    with gzip.open(args.canonical_raw, "rb") as stream:
        for device_row, host_row in zip(device["frames"], host["frames"], strict=True):
            identity = (device_row["source_name"], device_row["frame_id"])
            if identity != (host_row["source_name"], host_row["frame_id"]):
                raise ValueError(f"frame identity mismatch: {identity}")
            raw = np.frombuffer(read_exact(stream, RAW_BYTES_PER_FRAME, str(identity)), dtype="<f4").reshape(1, 84, 2100)
            letterbox = host_row["letterbox"]
            android_detections, android_diagnostics = decode(
                raw,
                tuple(host_row["source_size"]),
                (letterbox["scale"], letterbox["dx"], letterbox["dy"]),
                labels,
                float(detector["confidence_threshold"]),
                float(detector["nms_iou_threshold"]),
                320,
            )
            host_detections = host_row["post_nms_detections_canonical_320"]
            pre_nms_count_equal += int(android_diagnostics["pre_nms_candidate_count"] == host_row["pre_nms_candidate_count"])
            detection_count_equal += int(len(android_detections) == len(host_detections))
            same_identity = len(android_detections) == len(host_detections)
            if same_identity:
                for android_detection, host_detection in zip(android_detections, host_detections, strict=True):
                    if (
                        android_detection["prediction_index"] != host_detection["prediction_index"]
                        or android_detection["class_id"] != host_detection["class_id"]
                        or android_detection["label"] != host_detection["label"]
                    ):
                        same_identity = False
                        break
                    max_confidence_delta = max(
                        max_confidence_delta,
                        abs(float(android_detection["confidence"]) - float(host_detection["confidence"])),
                    )
                    max_box_delta = max(
                        max_box_delta,
                        max(abs(float(a) - float(b)) for a, b in zip(android_detection["box"], host_detection["box"], strict=True)),
                    )
            identity_exact += int(same_identity)
            threshold_state_equal += int(
                (float(device_row["android_raw_person_max_confidence"]) >= float(detector["confidence_threshold"]))
                == (float(host_row["raw_person_max_confidence"]) >= float(detector["confidence_threshold"]))
            )
        if stream.read(1):
            raise ValueError("canonical raw stream has trailing records")
    frame_count = len(host["frames"])
    result = {
        "schema": "blindassist_ustrf_detector_canonical_semantic_parity_v1",
        "authority": "parity_summary_only_no_target_truth_access",
        "frame_count": frame_count,
        "input_tensor_exact_match_count": host["input_tensor_exact_match_count"],
        "raw_output_within_frozen_tolerance_count": host["raw_output_within_frozen_tolerance_count"],
        "raw_gate_status": host["G1_android_host_parity"],
        "person_threshold_state_equal_count": threshold_state_equal,
        "pre_nms_candidate_count_equal_count": pre_nms_count_equal,
        "post_nms_detection_count_equal_count": detection_count_equal,
        "post_nms_detection_identity_exact_count": identity_exact,
        "matched_detection_max_confidence_delta": max_confidence_delta,
        "matched_detection_max_box_coordinate_delta": max_box_delta,
        "G1b_canonical_semantic_parity": "pass" if all(
            value == frame_count for value in (
                host["input_tensor_exact_match_count"], threshold_state_equal,
                pre_nms_count_equal, detection_count_equal, identity_exact,
            )
        ) else "fail",
        "does_not_override_raw_gate": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
