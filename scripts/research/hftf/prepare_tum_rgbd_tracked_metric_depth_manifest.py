#!/usr/bin/env python3
"""Prepare complete ByteTrack person tracks from a TUM-format RGB-D segment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from prepare_bonn_rgbd_metric_depth_manifest import (
    associate_nearest,
    normalize_depth_image,
    read_tum_index,
    sample_timestamp_pairs,
)
from prepare_external_rgb_metric_depth_manifest import torso_roi_from_person_box
from produce_external_rgb_metric_depth_observations import robust_roi_median


def complete_track_ids(
    tracks: dict[int, list[dict[str, Any]]], frame_count: int
) -> list[int]:
    expected = list(range(frame_count))
    return sorted(
        track_id
        for track_id, rows in tracks.items()
        if [int(row["frame_index"]) for row in rows] == expected
        and all(bool(row["truth_admissible"]) for row in rows)
    )


def semantic_torso_roi(
    keypoints_xy: np.ndarray,
    keypoint_confidence: np.ndarray,
    image_shape: tuple[int, ...],
    minimum_confidence: float,
) -> list[int] | None:
    torso_indices = [5, 6, 11, 12]
    points = np.asarray(keypoints_xy, dtype=np.float64)[torso_indices]
    confidence = np.asarray(keypoint_confidence, dtype=np.float64)[torso_indices]
    if (
        points.shape != (4, 2)
        or confidence.shape != (4,)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(confidence))
        or np.any(confidence < minimum_confidence)
    ):
        return None
    height, width = image_shape[:2]
    left = max(0, min(width - 1, int(round(float(np.min(points[:, 0]))))))
    top = max(0, min(height - 1, int(round(float(np.min(points[:, 1]))))))
    right = max(1, min(width, int(round(float(np.max(points[:, 0]))))))
    bottom = max(1, min(height, int(round(float(np.max(points[:, 1]))))))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def prepare(
    *,
    sequence_root: Path,
    output_dir: Path,
    weights: Path,
    sequence_id: str,
    start_s: float,
    duration_s: float,
    target_fps: float,
    max_association_delta_s: float,
    minimum_truth_valid_fraction: float,
    intrinsics: list[float],
    anchor_mode: str,
    pose_keypoint_confidence: float,
    confidence: float,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    associated = associate_nearest(
        read_tum_index(sequence_root / "rgb.txt"),
        read_tum_index(sequence_root / "depth.txt"),
        max_association_delta_s,
    )
    sampled = sample_timestamp_pairs(
        associated,
        start_s=start_s,
        duration_s=duration_s,
        target_fps=target_fps,
    )
    if len(sampled) < 7:
        raise ValueError("segment has fewer than seven paired frames")

    model = YOLO(str(weights))
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    first_timestamp = sampled[0][0]
    for frame_index, (
        rgb_timestamp,
        rgb_relative,
        depth_timestamp,
        depth_relative,
    ) in enumerate(sampled):
        rgb_path = (sequence_root / rgb_relative).resolve()
        depth_path = (sequence_root / depth_relative).resolve()
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth_raw = normalize_depth_image(
            cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED), depth_path
        )
        if bgr is None:
            raise OSError(f"failed to read RGB frame: {rgb_path}")
        if depth_raw.shape != bgr.shape[:2]:
            raise ValueError("registered RGB/depth shape mismatch")
        result = model.track(
            source=bgr,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=confidence,
            iou=0.5,
            imgsz=640,
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.detach().cpu().numpy().astype(int)
        boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        keypoints_xy = None
        keypoint_confidence = None
        if result.keypoints is not None:
            keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
            keypoint_confidence = result.keypoints.conf.detach().cpu().numpy()
        for detection_index, (track_id, person_box, person_score) in enumerate(
            zip(ids, boxes_xyxy, scores, strict=True)
        ):
            if anchor_mode == "pose_torso":
                if keypoints_xy is None or keypoint_confidence is None:
                    continue
                torso_roi = semantic_torso_roi(
                    keypoints_xy[detection_index],
                    keypoint_confidence[detection_index],
                    bgr.shape,
                    pose_keypoint_confidence,
                )
                if torso_roi is None:
                    continue
            else:
                torso_roi = torso_roi_from_person_box(person_box, bgr.shape)
            truth_depth, _, truth_valid_fraction = robust_roi_median(
                depth_raw.astype(np.float32) / 5000.0,
                tuple(torso_roi),
            )
            tracks[int(track_id)].append(
                {
                    "frame_index": frame_index,
                    "timestamp_ns": round((rgb_timestamp - first_timestamp) * 1e9),
                    "frame_path": str(rgb_path),
                    "scenario": "person_walking",
                    "camera_motion": "static",
                    "torso_roi_xyxy_px": torso_roi,
                    "person_box_xyxy_px": [float(value) for value in person_box],
                    "person_confidence": float(person_score),
                    "intrinsics_fx_fy_cx_cy": intrinsics,
                    "truth_depth_m": truth_depth,
                    "truth_depth_valid_fraction": truth_valid_fraction,
                    "truth_admissible": (
                        truth_depth is not None
                        and truth_valid_fraction >= minimum_truth_valid_fraction
                    ),
                    "rgb_depth_timestamp_delta_s": abs(
                        depth_timestamp - rgb_timestamp
                    ),
                    "track_id": int(track_id),
                    "tracking_source": "yolo11n_bytetrack",
                    "anchor_source": anchor_mode,
                    "truth_source": "registered_rgbd_sensor_depth_not_model_input",
                }
            )

    admitted = complete_track_ids(tracks, len(sampled))
    if not admitted:
        raise ValueError("segment has no complete admissible person track")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for track_id in admitted:
        track_sequence = f"{sequence_id}-track-{track_id:03d}"
        for row in tracks[track_id]:
            row = dict(row)
            row.pop("truth_admissible")
            row["sequence_id"] = track_sequence
            output_rows.append(row)
    manifest = output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    receipt = {
        "sequence_id": sequence_id,
        "manifest": str(manifest.resolve()),
        "sampled_frames": len(sampled),
        "admitted_track_ids": admitted,
        "admitted_tracks": len(admitted),
        "output_rows": len(output_rows),
        "start_s": start_s,
        "requested_duration_s": duration_s,
        "target_fps": target_fps,
        "selection": "all ByteTrack IDs present in every sampled frame",
        "minimum_truth_valid_fraction": minimum_truth_valid_fraction,
        "anchor_mode": anchor_mode,
        "pose_keypoint_confidence": pose_keypoint_confidence,
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-association-delta-s", type=float, default=0.02)
    parser.add_argument("--minimum-truth-valid-fraction", type=float, default=0.8)
    parser.add_argument(
        "--intrinsics-fx-fy-cx-cy", type=float, nargs=4, required=True
    )
    parser.add_argument(
        "--anchor-mode",
        choices=("bbox_torso", "pose_torso"),
        default="bbox_torso",
    )
    parser.add_argument("--pose-keypoint-confidence", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 < args.minimum_truth_valid_fraction <= 1:
        parser.error("--minimum-truth-valid-fraction must be in (0, 1]")
    if not 0 <= args.pose_keypoint_confidence <= 1:
        parser.error("--pose-keypoint-confidence must be in [0, 1]")
    receipt = prepare(
        sequence_root=args.sequence_root,
        output_dir=args.output_dir,
        weights=args.weights,
        sequence_id=args.sequence_id,
        start_s=args.start_s,
        duration_s=args.duration_s,
        target_fps=args.target_fps,
        max_association_delta_s=args.max_association_delta_s,
        minimum_truth_valid_fraction=args.minimum_truth_valid_fraction,
        intrinsics=list(args.intrinsics_fx_fy_cx_cy),
        anchor_mode=args.anchor_mode,
        pose_keypoint_confidence=args.pose_keypoint_confidence,
        confidence=args.confidence,
        device=args.device,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
