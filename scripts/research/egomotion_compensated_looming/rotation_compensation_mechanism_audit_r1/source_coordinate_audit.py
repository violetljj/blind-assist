from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.ecological_response_discovery_r0 import (
    runner as discovery,
)


PROTOCOL_ID = "RCLE_ROTATION_COMPENSATION_MECHANISM_AUDIT_R1"
PAIR_START = 343
PAIR_END_INCLUSIVE = 462
NEGATIVE_Z_BASIS = np.diag((1.0, -1.0, -1.0))


def homography(
    rotation: np.ndarray, scale: float = 0.5
) -> np.ndarray:
    full = discovery.INTRINSIC @ rotation @ np.linalg.inv(
        discovery.INTRINSIC
    )
    scaling = np.diag((scale, scale, 1.0))
    return scaling @ full @ np.linalg.inv(scaling)


def candidate_homographies(
    previous_quaternion: np.ndarray,
    current_quaternion: np.ndarray,
) -> dict[str, np.ndarray]:
    previous_wxyz = discovery.quaternion_rotation_wxyz(
        previous_quaternion
    )
    current_wxyz = discovery.quaternion_rotation_wxyz(current_quaternion)
    relative = current_wxyz.T @ previous_wxyz
    previous_legacy = discovery.quaternion_rotation_xyzw(
        previous_quaternion
    )
    current_legacy = discovery.quaternion_rotation_xyzw(
        current_quaternion
    )
    legacy = current_legacy.T @ previous_legacy
    return {
        "identity_no_rotation": np.eye(3, dtype=np.float64),
        "official_wxyz_direct": homography(relative),
        "official_wxyz_reverse": homography(relative.T),
        "legacy_xyzw_direct": homography(legacy),
        "negative_z_basis_conjugated": homography(
            NEGATIVE_Z_BASIS @ relative @ NEGATIVE_Z_BASIS.T
        ),
        "t_cam_imu_rotation_conjugated": homography(
            discovery.T_CAM_IMU_ROTATION
            @ relative
            @ discovery.T_CAM_IMU_ROTATION.T
        ),
    }


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (points.astype(np.float64), np.ones(len(points), dtype=np.float64))
    )
    mapped = (matrix @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def tracked_points(
    previous: np.ndarray, current: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=8.0,
        blockSize=7,
    )
    if points is None:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, winSize=(21, 21), maxLevel=3
    )
    if forward is None or status is None:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, forward, None, winSize=(21, 21), maxLevel=3
    )
    if backward is None or back_status is None:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )
    initial = points.reshape(-1, 2).astype(np.float64)
    observed = forward.reshape(-1, 2).astype(np.float64)
    returned = backward.reshape(-1, 2).astype(np.float64)
    valid = (
        (status.reshape(-1) > 0)
        & (back_status.reshape(-1) > 0)
        & np.isfinite(observed).all(axis=1)
        & (np.linalg.norm(returned - initial, axis=1) <= 1.5)
    )
    return initial[valid], observed[valid]


def run(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    frame_rows = discovery.load_csv(source_root / "iphone/frames.csv", 2)
    poses = discovery.load_pose_series(source_root / "ground-truth/pose.csv")
    capture = cv2.VideoCapture(
        str(source_root / "iphone/frames.mov")
    )
    if not capture.isOpened():
        raise ValueError("VIDEO_OPEN_FAILED")
    capture.set(cv2.CAP_PROP_POS_FRAMES, PAIR_START)
    maps = discovery.build_undistort_maps(720, 1280)
    ok, bgr = capture.read()
    if not ok:
        raise ValueError("VIDEO_FIRST_FRAME_DECODE_FAILED")
    previous = discovery.preprocess_frame(bgr, 0.5, maps)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260728)
    candidate_errors: dict[str, list[float]] = {}
    candidate_cosines: dict[str, list[float]] = {}
    pair_winners: Counter[str] = Counter()
    pair_rows: list[dict[str, Any]] = []
    for pair_index in range(PAIR_START, PAIR_END_INCLUSIVE + 1):
        ok, bgr = capture.read()
        if not ok:
            raise ValueError(f"VIDEO_FRAME_DECODE_FAILED:{pair_index + 1}")
        current = discovery.preprocess_frame(bgr, 0.5, maps)
        initial, observed = tracked_points(previous, current)
        previous_pose = discovery.interpolate_pose(
            poses, float(frame_rows[pair_index, 0])
        )
        current_pose = discovery.interpolate_pose(
            poses, float(frame_rows[pair_index + 1, 0])
        )
        candidates = candidate_homographies(
            previous_pose[1], current_pose[1]
        )
        medians: dict[str, float] = {}
        height, width = previous.shape
        for name, matrix in candidates.items():
            predicted = transform(initial, matrix)
            valid = (
                np.isfinite(predicted).all(axis=1)
                & (predicted[:, 0] >= 0)
                & (predicted[:, 0] < width)
                & (predicted[:, 1] >= 0)
                & (predicted[:, 1] < height)
            )
            errors = np.linalg.norm(observed[valid] - predicted[valid], axis=1)
            observed_motion = observed[valid] - initial[valid]
            predicted_motion = predicted[valid] - initial[valid]
            denominator = np.linalg.norm(
                observed_motion, axis=1
            ) * np.linalg.norm(predicted_motion, axis=1)
            directional = denominator > 0.25
            cosines = np.sum(
                observed_motion[directional] * predicted_motion[directional],
                axis=1,
            ) / denominator[directional]
            candidate_errors.setdefault(name, []).extend(errors.tolist())
            candidate_cosines.setdefault(name, []).extend(cosines.tolist())
            medians[name] = float(np.median(errors)) if errors.size else np.inf
        winner = min(medians, key=medians.get)
        pair_winners[winner] += 1
        pair_rows.append(
            {
                "pair_index": pair_index,
                "track_count": int(len(initial)),
                "median_endpoint_error_px": medians,
                "winner": winner,
            }
        )
        previous = current
    capture.release()
    summaries = {}
    for name in candidate_errors:
        errors = np.asarray(candidate_errors[name], dtype=np.float64)
        cosines = np.asarray(candidate_cosines[name], dtype=np.float64)
        summaries[name] = {
            "track_observation_count": int(errors.size),
            "median_endpoint_error_px": float(np.median(errors)),
            "p90_endpoint_error_px": float(np.quantile(errors, 0.9)),
            "median_direction_cosine": (
                float(np.median(cosines)) if cosines.size else None
            ),
            "pair_winner_count": int(pair_winners[name]),
        }
    return {
        "schema": "rcle.rotation_compensation.source_coordinate_audit.v1",
        "protocol_id": PROTOCOL_ID,
        "pair_start": PAIR_START,
        "pair_end_inclusive": PAIR_END_INCLUSIVE,
        "pair_count": len(pair_rows),
        "image_arm": "official_undistorted_half_resolution",
        "candidates": summaries,
        "best_by_global_median_endpoint_error": min(
            summaries,
            key=lambda name: summaries[name]["median_endpoint_error_px"],
        ),
        "pair_rows": pair_rows,
        "claim_ceiling": "SOURCE_COORDINATE_DEVELOPMENT_DIAGNOSTIC",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_root)
    discovery.write_json(args.output.resolve(), result)
    print(
        {
            "best": result["best_by_global_median_endpoint_error"],
            "candidates": result["candidates"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
