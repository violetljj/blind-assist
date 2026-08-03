#!/usr/bin/env python3
"""Prepare an RGB-only metric-depth manifest from a Bonn/TUM RGB-D sequence.

RGB frames are the model inputs. Registered depth frames are read only here to
derive per-frame torso metric truth for the shared target ROI.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from prepare_external_rgb_metric_depth_manifest import (
    select_largest_person,
    torso_roi_from_person_box,
)
from produce_external_rgb_metric_depth_observations import robust_roi_median


BONN_INTRINSICS = [542.822841, 542.576870, 315.593520, 237.756098]


def normalize_depth_image(depth: np.ndarray | None, path: Path) -> np.ndarray:
    if depth is None:
        raise OSError(f"failed to read depth frame: {path}")
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(
            f"depth frame must be single-channel, got shape {depth.shape}: {path}"
        )
    return depth


def read_tum_index(path: Path) -> list[tuple[float, Path]]:
    rows: list[tuple[float, Path]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: expected timestamp path")
        timestamp = float(parts[0])
        if not math.isfinite(timestamp):
            raise ValueError(f"{path}:{line_number}: non-finite timestamp")
        rows.append((timestamp, Path(parts[1])))
    if not rows:
        raise ValueError(f"index contains no rows: {path}")
    return sorted(rows)


def associate_nearest(
    rgb_rows: list[tuple[float, Path]],
    depth_rows: list[tuple[float, Path]],
    max_delta_s: float,
) -> list[tuple[float, Path, float, Path]]:
    if max_delta_s < 0:
        raise ValueError("max_delta_s must be non-negative")
    pairs: list[tuple[float, Path, float, Path]] = []
    depth_index = 0
    for rgb_timestamp, rgb_path in rgb_rows:
        while (
            depth_index + 1 < len(depth_rows)
            and abs(depth_rows[depth_index + 1][0] - rgb_timestamp)
            <= abs(depth_rows[depth_index][0] - rgb_timestamp)
        ):
            depth_index += 1
        depth_timestamp, depth_path = depth_rows[depth_index]
        if abs(depth_timestamp - rgb_timestamp) <= max_delta_s:
            pairs.append(
                (rgb_timestamp, rgb_path, depth_timestamp, depth_path)
            )
    return pairs


def sample_timestamp_pairs(
    pairs: list[tuple[float, Path, float, Path]],
    *,
    start_s: float,
    duration_s: float,
    target_fps: float,
) -> list[tuple[float, Path, float, Path]]:
    if start_s < 0 or duration_s <= 0 or target_fps <= 0:
        raise ValueError("start_s must be non-negative; duration and FPS positive")
    if not pairs:
        return []
    sequence_start = pairs[0][0]
    lower = sequence_start + start_s
    upper = lower + duration_s
    selected = []
    next_sample = lower
    for pair in pairs:
        timestamp = pair[0]
        if timestamp + 1e-12 < lower:
            continue
        if timestamp >= upper:
            break
        if timestamp + 1e-12 < next_sample:
            continue
        selected.append(pair)
        next_sample += 1.0 / target_fps
    return selected


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
    scenario: str,
    camera_motion: str,
    confidence: float,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    rgb_rows = read_tum_index(sequence_root / "rgb.txt")
    depth_rows = read_tum_index(sequence_root / "depth.txt")
    associated = associate_nearest(
        rgb_rows, depth_rows, max_association_delta_s
    )
    sampled = sample_timestamp_pairs(
        associated,
        start_s=start_s,
        duration_s=duration_s,
        target_fps=target_fps,
    )
    if len(sampled) < 7:
        raise ValueError(
            f"selected segment has {len(sampled)} paired frames; at least seven required"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights), task="detect")
    rows: list[dict[str, Any]] = []
    association_deltas = []
    truth_valid_fractions = []
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
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise OSError(f"failed to read RGB frame: {rgb_path}")
        depth_raw = normalize_depth_image(depth_raw, depth_path)
        if depth_raw.shape != bgr.shape[:2]:
            raise ValueError(
                f"registered RGB/depth shape mismatch: {bgr.shape[:2]} vs {depth_raw.shape}"
            )

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
            raise ValueError(f"no person detection at sampled frame {frame_index}")
        person_box, person_score = selected
        torso_roi = torso_roi_from_person_box(person_box, bgr.shape)
        truth_depth, _, truth_valid_fraction = robust_roi_median(
            depth_raw.astype(np.float32) / 5000.0,
            tuple(torso_roi),
        )
        if (
            truth_depth is None
            or truth_valid_fraction < minimum_truth_valid_fraction
        ):
            raise ValueError(
                f"insufficient registered depth at sampled frame {frame_index}: "
                f"valid_fraction={truth_valid_fraction:.6f}"
            )

        timestamp_ns = round((rgb_timestamp - first_timestamp) * 1e9)
        association_delta = abs(depth_timestamp - rgb_timestamp)
        association_deltas.append(association_delta)
        truth_valid_fractions.append(truth_valid_fraction)
        rows.append(
            {
                "sequence_id": sequence_id,
                "frame_index": frame_index,
                "timestamp_ns": timestamp_ns,
                "frame_path": str(rgb_path),
                "scenario": scenario,
                "camera_motion": camera_motion,
                "torso_roi_xyxy_px": torso_roi,
                "person_box_xyxy_px": [float(value) for value in person_box],
                "person_confidence": person_score,
                "intrinsics_fx_fy_cx_cy": intrinsics,
                "truth_depth_m": truth_depth,
                "truth_depth_valid_fraction": truth_valid_fraction,
                "rgb_depth_timestamp_delta_s": association_delta,
                "truth_source": "registered_rgbd_sensor_depth_not_model_input",
            }
        )

    manifest = output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    elapsed_s = (rows[-1]["timestamp_ns"] - rows[0]["timestamp_ns"]) / 1e9
    receipt = {
        "sequence_id": sequence_id,
        "sequence_root": str(sequence_root.resolve()),
        "manifest": str(manifest.resolve()),
        "sampled_frames": len(rows),
        "start_s": start_s,
        "requested_duration_s": duration_s,
        "effective_duration_s": elapsed_s,
        "target_fps": target_fps,
        "intrinsics_fx_fy_cx_cy": intrinsics,
        "scenario": scenario,
        "camera_motion": camera_motion,
        "effective_fps": (len(rows) - 1) / elapsed_s if elapsed_s > 0 else None,
        "max_rgb_depth_timestamp_delta_s": max(association_deltas),
        "minimum_truth_valid_fraction": min(truth_valid_fractions),
        "truth_firewall": "registered depth used only to create evaluation truth",
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sequence-id", default="bonn-person-tracking-canary")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-association-delta-s", type=float, default=0.02)
    parser.add_argument("--minimum-truth-valid-fraction", type=float, default=0.8)
    parser.add_argument(
        "--intrinsics-fx-fy-cx-cy",
        type=float,
        nargs=4,
        default=BONN_INTRINSICS,
    )
    parser.add_argument("--scenario", default="person_tracking")
    parser.add_argument("--camera-motion", default="tracking")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 < args.minimum_truth_valid_fraction <= 1:
        parser.error("--minimum-truth-valid-fraction must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
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
        scenario=args.scenario,
        camera_motion=args.camera_motion,
        confidence=args.confidence,
        device=args.device,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
