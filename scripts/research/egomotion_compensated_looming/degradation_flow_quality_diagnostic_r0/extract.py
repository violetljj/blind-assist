from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.ecological_response_discovery_r0 import (
    runner as r3,
)


PROTOCOL_ID = "RCLE_DEGRADATION_FLOW_QUALITY_DIAGNOSTIC_R0"
PAIR_COUNT = 601
ALLOWED_SESSIONS = {13, 14, 15, 17}
SEALED_SESSION = 16
RESIZE_SCALE = 0.5
FB_LIMIT_PX = 1.0
GATE_FB_MEDIAN_LIMIT_PX = 0.75
GATE_MIN_FEATURES = 60
GATE_MIN_FB_TRACKS = 60
GATE_MIN_FB_FRACTION = 0.5
GATE_MIN_OCCUPIED_CELLS = 5
ROLLING_MEDIAN_PAIRS = 31


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float64)
    return ranks / float(len(values) - 1)


def centered_rolling_median(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or width % 2 != 1:
        raise ValueError("ROLLING_WIDTH_MUST_BE_POSITIVE_ODD")
    radius = width // 2
    result = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        lo = max(0, index - radius)
        hi = min(len(values), index + radius + 1)
        result[index] = float(np.median(values[lo:hi]))
    return result


def valid_laplacian_variance(gray: np.ndarray, valid: np.ndarray) -> float:
    pixels = cv2.Laplacian(gray, cv2.CV_64F)[valid > 0]
    if pixels.size == 0:
        return float("nan")
    return float(np.var(pixels))


def flow_metrics(
    previous: np.ndarray,
    current: np.ndarray,
    previous_valid: np.ndarray,
) -> dict[str, Any]:
    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=8.0,
        mask=previous_valid,
        blockSize=7,
    )
    detected = 0 if points is None else int(len(points))
    valid_pixels = int(np.count_nonzero(previous_valid))
    base = {
        "detected_feature_count": detected,
        "detected_features_per_valid_megapixel": (
            detected / (valid_pixels / 1_000_000.0) if valid_pixels else None
        ),
        "forward_track_fraction": 0.0,
        "forward_backward_consistent_count": 0,
        "forward_backward_consistent_fraction": 0.0,
        "median_forward_backward_error_px": None,
        "occupied_grid_cells": 0,
    }
    if points is None or detected == 0:
        return base
    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, **lk
    )
    if forward is None or forward_status is None:
        return base
    forward_ok = forward_status.reshape(-1).astype(bool)
    base["forward_track_fraction"] = float(np.mean(forward_ok))
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, forward, None, **lk
    )
    if backward is None or backward_status is None:
        return base
    backward_ok = backward_status.reshape(-1).astype(bool)
    errors = np.linalg.norm(
        backward.reshape(-1, 2) - points.reshape(-1, 2), axis=1
    )
    jointly_tracked = forward_ok & backward_ok & np.isfinite(errors)
    consistent = jointly_tracked & (errors <= FB_LIMIT_PX)
    consistent_count = int(np.count_nonzero(consistent))
    base["forward_backward_consistent_count"] = consistent_count
    base["forward_backward_consistent_fraction"] = (
        consistent_count / detected
    )
    if np.any(jointly_tracked):
        base["median_forward_backward_error_px"] = float(
            np.median(errors[jointly_tracked])
        )
    if consistent_count:
        xy = points.reshape(-1, 2)[consistent]
        height, width = previous.shape
        columns = np.minimum((xy[:, 0] * 3 / width).astype(int), 2)
        rows = np.minimum((xy[:, 1] * 3 / height).astype(int), 2)
        base["occupied_grid_cells"] = int(
            len(set((int(y), int(x)) for x, y in zip(columns, rows)))
        )
    return base


def gate_reasons(metrics: dict[str, Any], fb_limit: float) -> list[str]:
    reasons: list[str] = []
    if metrics["detected_feature_count"] < GATE_MIN_FEATURES:
        reasons.append("FEATURES_LT_60")
    if metrics["forward_backward_consistent_count"] < GATE_MIN_FB_TRACKS:
        reasons.append("FB_TRACKS_LT_60")
    if (
        metrics["forward_backward_consistent_fraction"]
        < GATE_MIN_FB_FRACTION
    ):
        reasons.append("FB_FRACTION_LT_0P50")
    median_error = metrics["median_forward_backward_error_px"]
    if median_error is None or median_error > fb_limit:
        reasons.append(f"FB_MEDIAN_GT_{str(fb_limit).replace('.', 'P')}")
    if metrics["occupied_grid_cells"] < GATE_MIN_OCCUPIED_CELLS:
        reasons.append("OCCUPIED_CELLS_LT_5")
    return reasons


def extract(
    session: int,
    source_root: Path,
    output_dir: Path,
    contract_path: Path,
) -> dict[str, Any]:
    if session == SEALED_SESSION:
        raise PermissionError("SEALED_UNSEEN_SESSION_ACCESS_FORBIDDEN")
    if session not in ALLOWED_SESSIONS:
        raise ValueError("SESSION_NOT_IN_FROZEN_SET")
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    contract_path = contract_path.resolve()
    if output_dir.exists():
        raise FileExistsError("OUTPUT_DIRECTORY_EXISTS")
    video_path = source_root / "iphone/frames.mov"
    frame_csv_path = source_root / "iphone/frames.csv"
    pose_path = source_root / "ground-truth/pose.csv"
    for path in (video_path, frame_csv_path, pose_path, contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    frame_rows = r3.load_csv(frame_csv_path, 2)
    poses = r3.load_pose_series(pose_path)
    timestamps = frame_rows[: PAIR_COUNT + 1, 0]
    if len(timestamps) != PAIR_COUNT + 1:
        raise ValueError("FROZEN_FRAME_COUNT_UNAVAILABLE")
    capture = cv2.VideoCapture(os.fspath(video_path))
    if not capture.isOpened():
        raise ValueError("VIDEO_OPEN_FAILED")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (720, 1280):
        raise ValueError(f"VIDEO_SHAPE:{width}x{height}")
    undistort_maps = r3.build_undistort_maps(width, height)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260728)
    ok, previous_bgr = capture.read()
    if not ok:
        raise ValueError("VIDEO_FIRST_FRAME_DECODE_FAILED")
    previous, previous_valid = r3.preprocess_frame_with_mask(
        previous_bgr, RESIZE_SCALE, undistort_maps
    )
    previous_sharpness = valid_laplacian_variance(
        previous, previous_valid
    )
    rows: list[dict[str, Any]] = []
    speeds: list[float] = []
    angular_speeds: list[float] = []
    for pair_index in range(PAIR_COUNT):
        ok, current_bgr = capture.read()
        if not ok:
            raise ValueError(f"VIDEO_FRAME_DECODE_FAILED:{pair_index + 1}")
        current, current_valid = r3.preprocess_frame_with_mask(
            current_bgr, RESIZE_SCALE, undistort_maps
        )
        current_sharpness = valid_laplacian_variance(current, current_valid)
        previous_timestamp = float(timestamps[pair_index])
        current_timestamp = float(timestamps[pair_index + 1])
        dt = current_timestamp - previous_timestamp
        if not 0.0 < dt <= 0.1:
            raise ValueError(f"PAIR_DT:{pair_index}:{dt}")
        previous_pose = r3.interpolate_pose(poses, previous_timestamp)
        current_pose = r3.interpolate_pose(poses, current_timestamp)
        _, angular_speed, translation_speed = r3.pair_geometry(
            previous_pose,
            current_pose,
            dt,
            quaternion_component_order="wxyz",
            pose_to_camera_rotation=r3.T_CAM_IMU_ROTATION,
        )
        metrics = flow_metrics(previous, current, previous_valid)
        reasons = gate_reasons(metrics, GATE_FB_MEDIAN_LIMIT_PX)
        rows.append(
            {
                "session": session,
                "pair_index": pair_index,
                "previous_timestamp_s": previous_timestamp,
                "current_timestamp_s": current_timestamp,
                "dt_s": dt,
                "sharpness_laplacian_variance": float(
                    np.mean([previous_sharpness, current_sharpness])
                ),
                "translation_speed_m_per_s": translation_speed,
                "angular_speed_deg_per_s": angular_speed,
                **metrics,
                "flow_quality_gate_accept": not reasons,
                "flow_quality_gate_reasons": reasons,
                "risk_label_accessed": False,
                "response_accessed_during_extraction": False,
            }
        )
        speeds.append(translation_speed)
        angular_speeds.append(angular_speed)
        previous = current
        previous_valid = current_valid
        previous_sharpness = current_sharpness
    capture.release()

    speed_values = np.asarray(speeds, dtype=np.float64)
    angular_values = np.asarray(angular_speeds, dtype=np.float64)
    speed_residual = np.abs(
        speed_values
        - centered_rolling_median(speed_values, ROLLING_MEDIAN_PAIRS)
    )
    angular_residual = np.abs(
        angular_values
        - centered_rolling_median(angular_values, ROLLING_MEDIAN_PAIRS)
    )
    gait_score = np.maximum(
        rank_average(speed_residual), rank_average(angular_residual)
    )
    for index, row in enumerate(rows):
        row["translation_speed_highpass_abs"] = float(speed_residual[index])
        row["angular_speed_highpass_abs"] = float(angular_residual[index])
        row["gait_oscillation_proxy_score"] = float(gait_score[index])

    output_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = output_dir / "proxy_ledger.jsonl"
    ledger_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    summary = {
        "schema": "rcle.degradation_flow_quality.proxy_summary.v1",
        "protocol_id": PROTOCOL_ID,
        "session": session,
        "pair_count": len(rows),
        "contract_sha256": sha256_file(contract_path),
        "input_sha256": {
            "frames.mov": sha256_file(video_path),
            "frames.csv": sha256_file(frame_csv_path),
            "pose.csv": sha256_file(pose_path),
        },
        "proxy_ledger_sha256": sha256_file(ledger_path),
        "gate_rejected_fraction": float(
            np.mean([not row["flow_quality_gate_accept"] for row in rows])
        ),
        "risk_label_accessed": False,
        "response_accessed_during_extraction": False,
        "sealed_session_accessed": False,
    }
    summary_path = output_dir / "proxy_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = extract(
        args.session, args.source_root, args.output_dir, args.contract
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
