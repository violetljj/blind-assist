#!/usr/bin/env python3
"""Produce causal target observations from RGB only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1]) + max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.0


def choose_target(candidates: list[dict[str, Any]], previous: list[float] | None):
    if not candidates:
        return None, 0.0
    if previous is None:
        selected = max(candidates, key=lambda row: row["confidence"])
        return selected, 0.0
    selected = max(candidates, key=lambda row: 0.7 * iou(row["bbox_xyxy"], previous) + 0.3 * row["confidence"])
    return selected, iou(selected["bbox_xyxy"], previous)


def main() -> int:
    parser = argparse.ArgumentParser(description="RGB-only ADT target observation adapter")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.10)
    args = parser.parse_args()

    from ultralytics import YOLO
    import ultralytics

    model = YOLO(str(args.model))
    results = model.predict(source=str(args.video), stream=True, verbose=False, device="cpu", imgsz=args.imgsz, conf=args.confidence)
    frames = []
    events = []
    previous_bbox = None
    previous_nearness = None
    approach_ema = 0.0
    previous_visible = False
    ever_visible = False
    fps = None
    frame_size = None
    for frame_index, result in enumerate(results):
        fps = float(result.speed.get("fps", 0.0)) or fps
        height, width = result.orig_shape
        frame_size = {"width": int(width), "height": int(height)}
        candidates = []
        for box, confidence, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist(), strict=True):
            if str(result.names[int(class_id)]).casefold() == args.target_class.casefold():
                candidates.append({"bbox_xyxy": [float(value) for value in box], "confidence": float(confidence)})
        selected, association_iou = choose_target(candidates, previous_bbox)
        if selected is None:
            visible = False
            bbox = None
            confidence = 0.0
            bearing = None
            nearness = None
            approach_rate = None
            tracking_quality = 0.0
            observation_quality = 0.0
        else:
            visible = True
            bbox = selected["bbox_xyxy"]
            confidence = selected["confidence"]
            center_x = (bbox[0] + bbox[2]) / 2.0
            bearing = (center_x - width / 2.0) / (width / 2.0)
            nearness = math.sqrt(max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height)))
            raw_rate = 0.0 if previous_nearness is None else (nearness - previous_nearness) * 30.0
            approach_ema = 0.8 * approach_ema + 0.2 * raw_rate
            approach_rate = approach_ema
            tracking_quality = confidence * (0.5 + 0.5 * association_iou) if previous_bbox is not None else confidence * 0.5
            observation_quality = min(confidence, tracking_quality + 0.15)
            previous_bbox = bbox
            previous_nearness = nearness

        if visible and not previous_visible:
            events.append({"frame_index": frame_index, "event": "REACQUIRED" if ever_visible else "ACQUIRED"})
        elif not visible and previous_visible:
            events.append({"frame_index": frame_index, "event": "LOST"})
        ever_visible = ever_visible or visible
        previous_visible = visible
        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_s": frame_index / 30.0,
                "target_visible": visible,
                "target_class": args.target_class,
                "bbox_xyxy": bbox,
                "target_confidence": confidence,
                "target_bearing_normalized": bearing,
                "bearing_unit": "normalized_image_x",
                "relative_nearness": nearness,
                "approach_rate_per_s": approach_rate,
                "tracking_quality": tracking_quality,
                "observation_quality": observation_quality,
            }
        )

    output = {
        "schema_version": "ba_adt_rgb_observation_v1",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT-1-CANARY",
        "input": {"video": args.video.name, "sha256": sha256(args.video), "role": "RGB_SYSTEM_INPUT"},
        "groundtruth_argument_supported": False,
        "model": {"path_name": args.model.name, "sha256": sha256(args.model), "runtime": f"ultralytics-{ultralytics.__version__}", "target_class": args.target_class, "imgsz": args.imgsz, "confidence_floor": args.confidence},
        "frame_count": len(frames),
        "frame_size": frame_size,
        "visible_frame_count": sum(row["target_visible"] for row in frames),
        "events": events,
        "frames": frames,
        "limitations": ["preview_mp4_frame_order_time_proxy", "normalized_image_bearing_not_calibrated_degrees", "bbox_scale_nearness_not_metric_distance", "single_class_greedy_iou_association"],
        "claim_ceiling": "rgb_only_observation_mechanics_no_accuracy_or_navigation_claim",
        "terminal": "ADT1_RGB_OBSERVATIONS_PRODUCED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": output["terminal"], "frames": len(frames), "visible_frames": output["visible_frame_count"], "events": len(events)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
