#!/usr/bin/env python3
"""Evaluate the frozen YOLOE candidate on oracle-cleared R1.2 target frames."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from contract import sha256_file
from diagnostic_contract import load_projection, load_target_ledger
from projected_corridor_geometry import classify_contact_point, robust_relation


FROZEN_CLASSES = ["traffic cone", "delineator", "bollard"]
UNCERTAINTY_RATIOS = [0.01, 0.02, 0.03]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def extract_frame(video_path: Path, timestamp_ms: int) -> Any:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"cannot decode {video_path} at {timestamp_ms}ms")
    return frame


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    ledger_path = args.target_ledger.resolve()
    projection_path = args.projection_receipt.resolve()
    oracle_path = args.oracle_output.resolve()
    prereg_path = args.source_preregistration.resolve()
    weights_path = args.weights.resolve()
    output = args.output.resolve()
    cache_dir = args.cache_dir.resolve()
    embedding_cache_dir = args.embedding_cache_dir.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    if args.candidate_class != FROZEN_CLASSES:
        raise ValueError("candidate class order drifted")
    if sha256_file(weights_path) != args.expected_weights_sha256.lower():
        raise ValueError("weights SHA-256 mismatch")

    ledger = load_target_ledger(ledger_path)
    if ledger["diagnostic_set_role"] != "new_held_out_unscored":
        raise ValueError("R1.2 detector evaluation requires frozen held-out ledger")
    projection = load_projection(projection_path, ledger_path, ledger)
    oracle = load(oracle_path)
    if oracle.get("target_ledger_sha256") != sha256_file(ledger_path):
        raise ValueError("oracle/ledger hash mismatch")
    if oracle.get("projection_receipt_sha256") != sha256_file(projection_path):
        raise ValueError("oracle/projection hash mismatch")
    if not all(item.get("oracle_geometry_passed") is True for item in oracle["sources"]):
        raise ValueError("all six oracle sources must pass before detector evaluation")
    prereg = load(prereg_path)
    if prereg.get("dataset_role") != "new_held_out_unscored":
        raise ValueError("source preregistration role mismatch")
    if ledger.get("source_preregistration_sha256") != sha256_file(prereg_path):
        raise ValueError("ledger/source preregistration hash mismatch")

    source_by_event = {item["event_id"]: item for item in prereg["held_out_events"]}
    projection_by_event = {item["event_id"]: item for item in projection["events"]}
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache_dir)
    old_cwd = Path.cwd()
    os.chdir(embedding_cache_dir)
    try:
        import torch
        import ultralytics
        from ultralytics import YOLOE

        model = YOLOE(str(weights_path))
        model.set_classes(args.candidate_class)
        if list(model.names.values()) != args.candidate_class:
            raise ValueError("model class inventory drifted")
        source_results = []
        for event in ledger["events"]:
            event_id = event["event_id"]
            source = source_by_event[event_id]
            video_path = (repo_root / source["local_video_path"]).resolve()
            if sha256_file(video_path) != source["video_sha256"]:
                raise ValueError(f"source hash mismatch: {event_id}")
            projection_frames = {
                item["frame_id"]: item for item in projection_by_event[event_id]["frames"]
            }
            target = event["target_instance"]
            allowlist = set(target["detector_label_allowlist"])
            frame_rows = []
            matched_count = 0
            matched_inside_count = 0
            cooccurrence_inside_count = 0
            for frozen in target["frames"]:
                if frozen["visibility"] != "visible":
                    continue
                frame = extract_frame(video_path, frozen["timestamp_ms"])
                height, width = frame.shape[:2]
                prediction = model.predict(
                    frame,
                    imgsz=args.image_size,
                    conf=args.confidence,
                    max_det=args.maximum_detections,
                    verbose=False,
                )[0]
                target_norm = frozen["bbox_xyxy_norm"]
                target_box = [
                    target_norm[0] * width,
                    target_norm[1] * height,
                    target_norm[2] * width,
                    target_norm[3] * height,
                ]
                polygon = projection_frames[frozen["frame_id"]]["route_polygon_xy_norm"]
                detections = []
                for score, class_id, box in zip(
                    prediction.boxes.conf.cpu().numpy(),
                    prediction.boxes.cls.cpu().numpy(),
                    prediction.boxes.xyxy.cpu().numpy(),
                ):
                    label = str(model.names[int(class_id)])
                    xyxy = [float(value) for value in box.tolist()]
                    contact = [(xyxy[0] + xyxy[2]) / 2.0, xyxy[3]]
                    profiles = [
                        classify_contact_point(
                            contact,
                            frame_width=width,
                            frame_height=height,
                            polygon_xy_norm=polygon,
                            uncertainty_frame_ratio=ratio,
                        )
                        for ratio in UNCERTAINTY_RATIOS
                    ]
                    robust = robust_relation([item.relation for item in profiles])
                    detections.append(
                        {
                            "label": label,
                            "confidence": float(score),
                            "bbox_xyxy_px": xyxy,
                            "target_iou": iou(target_box, xyxy),
                            "eligible_label": label in allowlist,
                            "robust_route_relation": robust,
                        }
                    )
                eligible = [item for item in detections if item["eligible_label"]]
                best = max(eligible, key=lambda item: item["target_iou"], default=None)
                matched = bool(best and best["target_iou"] >= args.target_match_iou)
                if matched:
                    matched_count += 1
                    if best["robust_route_relation"] == "inside":
                        matched_inside_count += 1
                cooccurrence_inside_count += sum(
                    item["robust_route_relation"] == "inside" and item is not best
                    for item in detections
                )
                frame_rows.append(
                    {
                        "frame_id": frozen["frame_id"],
                        "timestamp_ms": frozen["timestamp_ms"],
                        "decoded_size": [width, height],
                        "target_match": matched,
                        "best_eligible_detection": best,
                        "detections": detections,
                    }
                )
            expected = target["expected_route_relation"]
            source_results.append(
                {
                    "event_id": event_id,
                    "source_id": event["source_id"],
                    "target_instance_id": target["target_instance_id"],
                    "expected_route_relation": expected,
                    "visible_target_frame_count": len(frame_rows),
                    "target_match_frame_count": matched_count,
                    "matched_target_inside_frame_count": matched_inside_count,
                    "cooccurrence_robust_inside_count": cooccurrence_inside_count,
                    "event_recall": int(matched_inside_count > 0) if expected == "inside" else None,
                    "false_alarm": bool(matched_inside_count > 0) if expected == "outside" else None,
                    "frames": frame_rows,
                }
            )
    finally:
        os.chdir(old_cwd)

    report = {
        "schema": "blindassist_ustrf_crosscam_r12_offline_detector_output_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_role": "held_out_results_opened_no_source_replacement",
        "target_ledger_sha256": sha256_file(ledger_path),
        "projection_receipt_sha256": sha256_file(projection_path),
        "oracle_output_sha256": sha256_file(oracle_path),
        "source_preregistration_sha256": sha256_file(prereg_path),
        "weights_sha256": sha256_file(weights_path),
        "frozen_classes": args.candidate_class,
        "image_size": args.image_size,
        "confidence": args.confidence,
        "target_match_iou": args.target_match_iou,
        "maximum_detections": args.maximum_detections,
        "runtime": {"ultralytics": ultralytics.__version__, "torch": torch.__version__},
        "sources": source_results,
        "summary": {
            "positive_event_recall": sum(item["event_recall"] or 0 for item in source_results),
            "positive_source_count": sum(item["event_recall"] is not None for item in source_results),
            "negative_false_alarm_count": sum(item["false_alarm"] is True for item in source_results),
            "negative_source_count": sum(item["false_alarm"] is not None for item in source_results),
            "target_match_frame_count": sum(item["target_match_frame_count"] for item in source_results),
        },
        "authority": {
            "threshold_fit": False,
            "source_replacement_authorized": False,
            "training_performed": False,
            "android_runtime_change_authorized": False,
            "production_model_replacement_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(output) + ".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    print(
        "USTRF_R12_OFFLINE_DETECTOR_OK",
        report["summary"]["positive_event_recall"],
        report["summary"]["positive_source_count"],
        report["summary"]["negative_false_alarm_count"],
        report["summary"]["negative_source_count"],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--target-ledger", type=Path, required=True)
    parser.add_argument("--projection-receipt", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--source-preregistration", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-class", action="append", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--target-match-iou", type=float, default=0.30)
    parser.add_argument("--maximum-detections", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
