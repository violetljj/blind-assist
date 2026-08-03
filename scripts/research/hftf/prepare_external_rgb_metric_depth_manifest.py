#!/usr/bin/env python3
"""Extract controlled RGB frames and shared person torso ROIs from one video."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TORSO_FRACTIONS = (0.25, 0.18, 0.75, 0.65)


def sample_source_indices(
    source_fps: float, target_fps: float, frame_count: int
) -> list[int]:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS values must be positive")
    effective_target = min(source_fps, target_fps)
    selected = []
    next_sample_s = 0.0
    for source_index in range(frame_count):
        timestamp_s = source_index / source_fps
        if timestamp_s + 1e-12 < next_sample_s:
            continue
        selected.append(source_index)
        next_sample_s += 1.0 / effective_target
    return selected


def select_largest_person(
    boxes_xyxy: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, float] | None:
    if not len(boxes_xyxy):
        return None
    widths = np.maximum(0.0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
    heights = np.maximum(0.0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    areas = widths * heights
    index = max(
        range(len(boxes_xyxy)),
        key=lambda value: (float(areas[value]), float(scores[value])),
    )
    return boxes_xyxy[index], float(scores[index])


def torso_roi_from_person_box(
    box_xyxy: np.ndarray | list[float],
    image_shape: tuple[int, ...],
) -> list[int]:
    height, width = image_shape[:2]
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    box_width = x1 - x0
    box_height = y1 - y0
    if box_width <= 0 or box_height <= 0:
        raise ValueError("person box must have positive area")
    left, top, right, bottom = TORSO_FRACTIONS
    roi = [
        max(0, min(width - 1, int(round(x0 + left * box_width)))),
        max(0, min(height - 1, int(round(y0 + top * box_height)))),
        max(1, min(width, int(round(x0 + right * box_width)))),
        max(1, min(height, int(round(y0 + bottom * box_height)))),
    ]
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise ValueError("derived torso ROI is empty")
    return roi


def prepare(
    *,
    input_video: Path,
    output_dir: Path,
    weights: Path,
    sequence_id: str,
    scenario: str,
    camera_motion: str,
    truth_depth_m: float | None,
    truth_direction: str | None,
    intrinsics: list[float],
    target_fps: float,
    confidence: float,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {input_video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("video has no valid FPS")
    effective_target_fps = min(source_fps, target_fps)
    next_sample_s = 0.0
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights), task="detect")

    rows = []
    source_index = -1
    sampled_index = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        source_index += 1
        timestamp_s = source_index / source_fps
        if timestamp_s + 1e-12 < next_sample_s:
            continue
        next_sample_s += 1.0 / effective_target_fps
        result = model.predict(
            source=bgr,
            classes=[0],
            conf=confidence,
            iou=0.5,
            imgsz=640,
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        boxes_xyxy = (
            boxes.xyxy.detach().cpu().numpy()
            if boxes is not None and len(boxes)
            else np.empty((0, 4), dtype=np.float32)
        )
        scores = (
            boxes.conf.detach().cpu().numpy()
            if boxes is not None and len(boxes)
            else np.empty((0,), dtype=np.float32)
        )
        selected = select_largest_person(boxes_xyxy, scores)
        if selected is None:
            raise ValueError(
                f"no person detection at source frame {source_index}; "
                "repeat the controlled capture or lower --confidence"
            )
        person_box, person_score = selected
        torso_roi = torso_roi_from_person_box(person_box, bgr.shape)
        frame_path = frames_dir / f"{sampled_index:06d}.png"
        if not cv2.imwrite(str(frame_path), bgr):
            raise OSError(f"failed to write frame: {frame_path}")
        row: dict[str, Any] = {
            "sequence_id": sequence_id,
            "frame_index": sampled_index,
            "timestamp_ns": round(timestamp_s * 1_000_000_000),
            "frame_path": str(frame_path.relative_to(output_dir)),
            "scenario": scenario,
            "camera_motion": camera_motion,
            "torso_roi_xyxy_px": torso_roi,
            "person_box_xyxy_px": [float(value) for value in person_box],
            "person_confidence": person_score,
            "intrinsics_fx_fy_cx_cy": intrinsics,
        }
        if truth_depth_m is not None:
            row["truth_depth_m"] = truth_depth_m
        if truth_direction is not None:
            row["truth_direction"] = truth_direction
        rows.append(row)
        sampled_index += 1
    capture.release()
    if len(rows) < 7:
        raise ValueError(
            f"sequence has {len(rows)} sampled frames; at least seven required"
        )
    manifest = output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    elapsed_s = (
        (int(rows[-1]["timestamp_ns"]) - int(rows[0]["timestamp_ns"]))
        / 1_000_000_000.0
    )
    effective_fps = (
        (len(rows) - 1) / elapsed_s if elapsed_s > 0 else None
    )
    receipt = {
        "sequence_id": sequence_id,
        "source_video": str(input_video.resolve()),
        "source_fps": source_fps,
        "target_fps": target_fps,
        "effective_fps": effective_fps,
        "sampled_frames": len(rows),
        "detected_frames": len(rows),
        "manifest": str(manifest.resolve()),
        "torso_fractions_within_person_xyxy": list(TORSO_FRACTIONS),
        "selection": "largest COCO person per controlled single-person frame",
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument(
        "--scenario",
        choices=("static", "approach", "recede", "lateral"),
        required=True,
    )
    parser.add_argument(
        "--camera-motion", choices=("static", "walking"), required=True
    )
    truth = parser.add_mutually_exclusive_group(required=True)
    truth.add_argument("--truth-depth-m", type=float)
    truth.add_argument(
        "--truth-direction",
        choices=("approach", "recede", "stable_or_lateral"),
    )
    parser.add_argument(
        "--intrinsics-fx-fy-cx-cy",
        type=float,
        nargs=4,
        required=True,
    )
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.target_fps <= 0:
        parser.error("--target-fps must be positive")
    if args.truth_depth_m is not None and args.truth_depth_m <= 0:
        parser.error("--truth-depth-m must be positive")
    return args


def main() -> None:
    args = parse_args()
    receipt = prepare(
        input_video=args.input_video,
        output_dir=args.output_dir,
        weights=args.weights,
        sequence_id=args.sequence_id,
        scenario=args.scenario,
        camera_motion=args.camera_motion,
        truth_depth_m=args.truth_depth_m,
        truth_direction=args.truth_direction,
        intrinsics=list(args.intrinsics_fx_fy_cx_cy),
        target_fps=args.target_fps,
        confidence=args.confidence,
        device=args.device,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
