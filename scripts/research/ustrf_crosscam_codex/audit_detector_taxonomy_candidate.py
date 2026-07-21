#!/usr/bin/env python3
"""Audit a frozen prompted detector on the seen R1.1 target ledger.

This is a taxonomy/readiness audit only. It must run on the seen diagnostic set
before any replacement held-out source is preregistered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


SCHEMA = "blindassist_ustrf_detector_taxonomy_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def iou_xyxy(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def validate_inputs(
    ledger: dict[str, Any], manifest: dict[str, Any], classes: list[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if ledger.get("diagnostic_set_role") != "seen_diagnostic_not_held_out":
        raise ValueError("taxonomy audit requires the seen R1.1 diagnostic ledger")
    if manifest.get("diagnostic_set_role") != "seen_diagnostic_not_held_out":
        raise ValueError("taxonomy audit requires the seen R1.1 source manifest")
    if len(classes) != len(set(classes)) or not classes:
        raise ValueError("candidate classes must be non-empty and unique")
    required = {"traffic cone", "delineator", "bollard"}
    missing = sorted(required - set(classes))
    if missing:
        raise ValueError(f"candidate class contract is incomplete: {missing}")
    events = ledger.get("events")
    sources = manifest.get("sources")
    if not isinstance(events, list) or not isinstance(sources, list):
        raise ValueError("ledger events and manifest sources must be arrays")
    source_by_event = {str(item["event_id"]): item for item in sources}
    if set(source_by_event) != {str(item["event_id"]) for item in events}:
        raise ValueError("ledger/manifest event sets differ")
    return events, source_by_event


def extract_frame(video_path: Path, timestamp_ms: int) -> Any:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot decode {video_path} at {timestamp_ms}ms")
    return frame


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    ledger_path = args.ledger.resolve()
    source_manifest_path = args.source_manifest.resolve()
    weights_path = args.weights.resolve()
    embedding_cache_dir = args.embedding_cache_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    ledger = load_json(ledger_path)
    manifest = load_json(source_manifest_path)
    events, source_by_event = validate_inputs(ledger, manifest, args.candidate_class)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if args.expected_weights_sha256:
        actual = sha256_file(weights_path)
        if actual != args.expected_weights_sha256.lower():
            raise ValueError(f"weights SHA-256 mismatch: {actual}")

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
            raise ValueError("model class inventory differs from frozen candidate classes")

        results: list[dict[str, Any]] = []
        for event in events:
            event_id = str(event["event_id"])
            source = source_by_event[event_id]
            video_path = (repo_root / str(source["video_path"])).resolve()
            if sha256_file(video_path) != str(source["video_sha256"]):
                raise ValueError(f"source video SHA-256 mismatch: {event_id}")
            target = event["target_instance"]
            allowlist = set(map(str, target["detector_label_allowlist"]))
            frame_results: list[dict[str, Any]] = []
            for frozen in target["frames"]:
                if frozen.get("visibility") != "visible":
                    continue
                frame = extract_frame(video_path, int(frozen["timestamp_ms"]))
                prediction = model.predict(
                    frame,
                    imgsz=args.image_size,
                    conf=args.confidence,
                    max_det=args.max_detections,
                    verbose=False,
                )[0]
                height, width = frame.shape[:2]
                target_norm = list(map(float, frozen["bbox_xyxy_norm"]))
                target_px = [
                    target_norm[0] * width,
                    target_norm[1] * height,
                    target_norm[2] * width,
                    target_norm[3] * height,
                ]
                detections: list[dict[str, Any]] = []
                if prediction.boxes is not None:
                    for score, class_id, box in zip(
                        prediction.boxes.conf.cpu().numpy(),
                        prediction.boxes.cls.cpu().numpy(),
                        prediction.boxes.xyxy.cpu().numpy(),
                    ):
                        label = str(model.names[int(class_id)])
                        xyxy = [float(value) for value in box.tolist()]
                        detections.append(
                            {
                                "label": label,
                                "confidence": float(score),
                                "xyxy_px": xyxy,
                                "target_iou": iou_xyxy(target_px, xyxy),
                                "eligible_label": label in allowlist,
                            }
                        )
                eligible = [item for item in detections if item["eligible_label"]]
                best = max(eligible, key=lambda item: item["target_iou"], default=None)
                frame_results.append(
                    {
                        "frame_id": frozen["frame_id"],
                        "timestamp_ms": frozen["timestamp_ms"],
                        "decoded_size": [width, height],
                        "detection_count": len(detections),
                        "detections": detections,
                        "target_match": bool(best and best["target_iou"] >= args.match_iou),
                        "best_eligible_detection": best,
                    }
                )
            results.append(
                {
                    "event_id": event_id,
                    "semantic_type": target["semantic_type"],
                    "detector_label_allowlist": sorted(allowlist),
                    "frames": frame_results,
                    "matched_frames": sum(item["target_match"] for item in frame_results),
                    "visible_frames": len(frame_results),
                }
            )
    finally:
        os.chdir(old_cwd)

    observed_labels = sorted(
        {
            detection["label"]
            for event in results
            for frame in event["frames"]
            for detection in frame["detections"]
        }
    )
    matched_labels = sorted(
        {
            str(frame["best_eligible_detection"]["label"])
            for event in results
            for frame in event["frames"]
            if frame["target_match"]
        }
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_role": "pre_preregistration_taxonomy_readiness_on_seen_diagnostic_only",
        "candidate": {
            "kind": "YOLOE prompted segmentation detector with frozen static classes",
            "weights_sha256": sha256_file(weights_path),
            "classes": args.candidate_class,
            "class_inventory_exact": True,
            "image_size": args.image_size,
            "confidence": args.confidence,
            "match_iou": args.match_iou,
            "runtime": {
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "opencv": cv2.__version__,
            },
        },
        "source_ledger_sha256": sha256_file(ledger_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "events": results,
        "summary": {
            "candidate_classes": args.candidate_class,
            "observed_labels": observed_labels,
            "matched_labels": matched_labels,
            "visible_frames": sum(item["visible_frames"] for item in results),
            "matched_frames": sum(item["matched_frames"] for item in results),
            "taxonomy_contract_passed": set(matched_labels)
            >= {"traffic cone", "delineator", "bollard"},
        },
        "authority": {
            "new_held_out_sources_read": False,
            "threshold_fit": False,
            "training_performed": False,
            "android_runtime_authorized": False,
            "production_model_replacement_authorized": False,
        },
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256")
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-class", action="append", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--max-detections", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(
        "USTRF_TAXONOMY_AUDIT_OK",
        report["summary"]["visible_frames"],
        report["summary"]["matched_frames"],
        ",".join(report["summary"]["matched_labels"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
