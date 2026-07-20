#!/usr/bin/env python3
"""GPU audit of source-native ego-compensated dynamic tracks in Virtual KITTI 2 text GT.

The output measures tracking/kinematics in VKITTI's camera convention.  VKITTI text ground truth
does not carry a frame timestamp receipt, so every time-to-collision value is reported in frame
intervals and is deliberately not authorized as physical seconds or a USTRF safety input.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter=" ", skipinitialspace=True))


def _matrix(row: dict[str, str]) -> np.ndarray:
    return np.array([
        [float(row["r1,1"]), float(row["r1,2"]), float(row["r1,3"]), float(row["t1"])],
        [float(row["r2,1"]), float(row["r2,2"]), float(row["r2,3"]), float(row["t2"])],
        [float(row["r3,1"]), float(row["r3,2"]), float(row["r3,3"]), float(row["t3"])],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _planar(camera_xyz: np.ndarray) -> np.ndarray:
    # VKITTI states camera x/y/z; use forward=z, lateral=x while retaining the source convention.
    return np.array([camera_xyz[2], camera_xyz[0]], dtype=np.float32)


def audit(sequence: Path, *, motion_threshold_m_per_frame: float = .05, horizon_frames: float = 30.0) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for VKITTI dynamic-track audit")
    pose_by_frame: dict[int, dict[int, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for row in _rows(sequence / "pose.txt"):
        pose_by_frame[int(row["frame"])][int(row["cameraID"])][row["trackID"]] = row
    extrinsics: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for row in _rows(sequence / "extrinsic.txt"):
        extrinsics[int(row["frame"])][int(row["cameraID"])] = _matrix(row)
    moving: dict[int, dict[int, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    for row in _rows(sequence / "bbox.txt"):
        moving[int(row["frame"])][int(row["cameraID"])][row["trackID"]] = row["isMoving"].lower() == "true"
    frames = sorted(set(pose_by_frame) & set(extrinsics))
    previous_current: list[np.ndarray] = []
    current_positions: list[np.ndarray] = []
    labels: list[bool] = []
    identifiers: list[dict[str, Any]] = []
    for first, second in zip(frames, frames[1:]):
        if second != first + 1 or 0 not in extrinsics[first] or 0 not in extrinsics[second]:
            continue
        first_tracks = pose_by_frame[first].get(0, {})
        second_tracks = pose_by_frame[second].get(0, {})
        transform_previous_camera_to_current = extrinsics[second][0] @ np.linalg.inv(extrinsics[first][0])
        for track_id in sorted(set(first_tracks) & set(second_tracks), key=int):
            p0 = first_tracks[track_id]
            p1 = second_tracks[track_id]
            first_camera = np.array([float(p0["camera_space_X"]), float(p0["camera_space_Y"]), float(p0["camera_space_Z"]), 1.0], dtype=np.float32)
            second_camera = np.array([float(p1["camera_space_X"]), float(p1["camera_space_Y"]), float(p1["camera_space_Z"])], dtype=np.float32)
            prior_in_current = transform_previous_camera_to_current @ first_camera
            # Keep only front-facing source observations; this is a source-data screening choice.
            if second_camera[2] <= 0.0 or prior_in_current[2] <= 0.0:
                continue
            previous_current.append(_planar(prior_in_current[:3]))
            current_positions.append(_planar(second_camera))
            labels.append(bool(moving[second].get(0, {}).get(track_id, False)))
            identifiers.append({"first_frame": first, "second_frame": second, "track_id": int(track_id)})
    if not current_positions:
        raise ValueError("no consecutive front-facing camera-0 track pairs")
    device = torch.device("cuda")
    previous = torch.tensor(np.stack(previous_current), device=device)
    current = torch.tensor(np.stack(current_positions), device=device)
    labels_t = torch.tensor(labels, dtype=torch.bool, device=device)
    delta = current - previous
    speed_per_frame = torch.linalg.vector_norm(delta, dim=1)
    predicted_moving = speed_per_frame >= motion_threshold_m_per_frame
    true_positive = int((predicted_moving & labels_t).sum().item())
    false_positive = int((predicted_moving & ~labels_t).sum().item())
    false_negative = int((~predicted_moving & labels_t).sum().item())
    true_negative = int((~predicted_moving & ~labels_t).sum().item())
    denom_precision = max(1, true_positive + false_positive)
    denom_recall = max(1, true_positive + false_negative)
    squared = (delta * delta).sum(dim=1)
    ttc_frames = (-(current * delta).sum(dim=1) / squared.clamp_min(1e-8)).clamp(0.0, horizon_frames)
    closest = current + delta * ttc_frames[:, None]
    report = {
        "format": "blindassist_vkitti2_source_native_dynamic_track_audit_v1",
        "sequence": str(sequence),
        "source_camera_id": 0,
        "source_camera_convention": "VKITTI KITTI-like camera coordinates; planar audit uses forward=z, lateral=x",
        "consecutive_track_pair_count": len(identifiers),
        "source_moving_pair_count": int(labels_t.sum().item()),
        "motion_threshold_m_per_frame": motion_threshold_m_per_frame,
        "classification": {
            "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative, "true_negative": true_negative,
            "precision": true_positive / denom_precision, "recall": true_positive / denom_recall,
        },
        "kinematics": {
            "median_ego_compensated_delta_m_per_frame": float(speed_per_frame.median().item()),
            "p95_ego_compensated_delta_m_per_frame": float(torch.quantile(speed_per_frame, .95).item()),
            "moving_pair_with_ttc_within_horizon_frames": int((labels_t & (ttc_frames < horizon_frames)).sum().item()),
            "median_ttc_frames_for_source_moving": float(ttc_frames[labels_t].median().item()) if bool(labels_t.any()) else None,
            "median_closest_distance_m_for_source_moving": float(torch.linalg.vector_norm(closest[labels_t], dim=1).median().item()) if bool(labels_t.any()) else None,
        },
        "sample_pairs": [{**identifiers[index], "source_is_moving": labels[index], "ego_compensated_delta_m_per_frame": float(speed_per_frame[index].item()), "ttc_frames": float(ttc_frames[index].item())} for index in range(min(20, len(identifiers)))],
        "compute_backend": {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)},
        "timestamp_receipt_present": False,
        "physical_ttc_seconds_admitted": False,
        "ustrf_motion_input_admitted": False,
        "reason": "source has exact frame order, pose and track geometry but no bound per-frame timestamp or verified mapping to a human body frame",
    }
    qa = sequence / "qa"; qa.mkdir(exist_ok=True)
    (qa / "vkitti_dynamic_track_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--motion-threshold-m-per-frame", type=float, default=.05)
    parser.add_argument("--horizon-frames", type=float, default=30.0)
    args = parser.parse_args()
    report = audit(args.sequence, motion_threshold_m_per_frame=args.motion_threshold_m_per_frame, horizon_frames=args.horizon_frames)
    print(json.dumps({"pairs": report["consecutive_track_pair_count"], "moving_pairs": report["source_moving_pair_count"], "recall": report["classification"]["recall"], "physical_ttc_seconds_admitted": report["physical_ttc_seconds_admitted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
