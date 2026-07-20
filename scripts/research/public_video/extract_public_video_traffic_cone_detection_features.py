#!/usr/bin/env python3
"""Extract interpretable visual features for prompt-free traffic-cone detections."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import scan_public_video_prompt_free_exit_candidates as discovery
import public_video_chromatic_marker_policy as chromatic
import public_video_tristate_contract as prospective


SCHEMA = "blindassist_public_video_traffic_cone_detection_features_v1"
TARGET_CLASS = "traffic cone"


def box_visual_features(
    image_bgr: np.ndarray,
    xyxy: Sequence[float],
) -> dict[str, float | bool]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or len(xyxy) != 4:
        raise ValueError("image or detection box shape is invalid")
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)
    left = max(0, min(width, int(np.floor(x1))))
    top = max(0, min(height, int(np.floor(y1))))
    right = max(0, min(width, int(np.ceil(x2))))
    bottom = max(0, min(height, int(np.ceil(y2))))
    if right <= left or bottom <= top:
        raise ValueError("traffic-cone detection box is empty after clipping")
    crop = image_bgr[top:bottom, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    warm = (((hue <= 25) | (hue >= 170)) & (saturation >= 80) & (value >= 60))
    high_saturation = (saturation >= 100) & (value >= 50)
    bright_neutral = (saturation <= 55) & (value >= 150)
    dark = value <= 60
    geometry = discovery.nearfield_corridor_geometry(
        [left, top, right, bottom], [height, width]
    )
    width_norm = (right - left) / width
    height_norm = (bottom - top) / height
    return {
        **geometry,
        "width_norm": float(width_norm),
        "height_norm": float(height_norm),
        "aspect_width_over_height": float(width_norm / max(height_norm, 1e-12)),
        "mean_saturation": float(saturation.mean() / 255.0),
        "mean_value": float(value.mean() / 255.0),
        "warm_color_fraction": float(warm.mean()),
        "high_saturation_fraction": float(high_saturation.mean()),
        "bright_neutral_fraction": float(bright_neutral.mean()),
        "dark_fraction": float(dark.mean()),
    }


def scan_source(
    model: Any,
    source: dict[str, Any],
    *,
    sample_interval_ms: int,
    image_size: int,
    confidence: float,
    target_classes: set[str],
) -> dict[str, Any]:
    video_path = Path(source["local_video_path"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"video timing metadata is invalid: {video_path}")
    duration_ms = int(round(frame_count / fps * 1000.0))
    start_ms, end_ms = discovery.source_scan_range(source, duration_ms)
    samples: list[dict[str, Any]] = []
    try:
        for timestamp_ms in range(start_ms, end_ms, sample_interval_ms):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            predictions = model.predict(
                frame,
                imgsz=image_size,
                conf=confidence,
                max_det=100,
                verbose=False,
            )
            if len(predictions) != 1:
                raise RuntimeError("prompt-free detector must return one result per frame")
            result = predictions[0]
            detections: list[dict[str, Any]] = []
            if result.boxes is not None:
                for score, class_id, box in zip(
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.xyxy.cpu().numpy(),
                ):
                    class_name = str(model.names[int(class_id)])
                    if class_name not in target_classes:
                        continue
                    detections.append({
                        "class_name": class_name,
                        "confidence": float(score),
                        "xyxy": [float(value) for value in box.tolist()],
                        "features": box_visual_features(frame, box.tolist()),
                    })
            samples.append({
                "timestamp_ms": timestamp_ms,
                "target_detection_count": len(detections),
                "traffic_cone_detection_count": sum(
                    detection["class_name"] == TARGET_CLASS for detection in detections
                ),
                "detections": detections,
            })
    finally:
        capture.release()
    return {
        **source,
        "video_sha256": common.sha256_file(video_path),
        "video_duration_ms": duration_ms,
        "scan_start_ms": start_ms,
        "scan_end_ms": end_ms,
        "sample_count": len(samples),
        "traffic_cone_detection_count": sum(
            sample["traffic_cone_detection_count"] for sample in samples
        ),
        "target_detection_count": sum(
            sample["target_detection_count"] for sample in samples
        ),
        "samples": samples,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.source_registry, args.weights, args.cache_dir, args.output]
    if args.contract is not None:
        paths.append(args.contract)
    for path in paths:
        mil.reject_independent_direction(path)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    registry = common.load_json(args.source_registry)
    sources = discovery.validate_registry(registry, args.source_registry.resolve())
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(args.cache_dir.resolve())
    import torch
    import ultralytics
    from ultralytics import YOLOE

    model = YOLOE(str(args.weights))
    target_classes = set(args.target_class or [TARGET_CLASS])
    missing = sorted(target_classes - set(model.names.values()))
    if missing:
        raise ValueError(f"target classes are absent from the frozen vocabulary: {missing}")
    contract_attestation: dict[str, str] | None = None
    if args.contract is not None:
        contract, contract_attestation = prospective.load_contract(args.contract)
        chromatic.validate_extractor_binding(
            contract,
            weights_sha256=common.sha256_file(args.weights),
            sample_interval_ms=args.sample_interval_ms,
            image_size=args.image_size,
            confidence=args.confidence,
            target_classes=target_classes,
        )
    results = [
        scan_source(
            model,
            source,
            sample_interval_ms=args.sample_interval_ms,
            image_size=args.image_size,
            confidence=args.confidence,
            target_classes=target_classes,
        )
        for source in sources
    ]
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(args.source_registry.resolve()),
        "source_registry_sha256": common.sha256_file(args.source_registry),
        "prospective_contract": contract_attestation,
        "model": {
            "kind": "YOLOE prompt-free segmentation model with fixed built-in vocabulary",
            "weights": str(args.weights.resolve()),
            "weights_sha256": common.sha256_file(args.weights),
            "target_classes": sorted(target_classes),
            "image_size": args.image_size,
            "confidence": args.confidence,
            "text_prompt_used": False,
            "runtime": {
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "opencv": cv2.__version__,
            },
        },
        "sampling": {"sample_interval_ms": args.sample_interval_ms},
        "sources": results,
        "summary": {
            "source_count": len(results),
            "sample_count": sum(result["sample_count"] for result in results),
            "traffic_cone_detection_count": sum(
                result["traffic_cone_detection_count"] for result in results
            ),
            "target_detection_count": sum(
                result["target_detection_count"] for result in results
            ),
        },
        "evidence_limit": "Post-r7.10 feature diagnosis only; no extracted statistic is training truth, calibration, blind evidence, Android runtime authorization, or production evidence.",
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(
        common.sha256_file(args.output) + "\n", encoding="ascii"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--sample-interval-ms", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument(
        "--target-class",
        action="append",
        help="Frozen vocabulary class to inspect; repeat for multiple classes.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
