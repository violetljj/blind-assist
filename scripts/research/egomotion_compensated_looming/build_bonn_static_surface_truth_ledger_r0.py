#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


FX = 542.822841
FY = 542.576870
CX = 315.593520
CY = 237.756098
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
MAX_POSE_JOIN_DELTA_SECONDS = 0.040
MAP_REDUCTION_VOXEL_METERS = 0.05
ROI_U_FRACTION = (0.25, 0.75)
ROI_V_FRACTION = (0.15, 0.90)
MIN_STATIC_MAP_POINTS_IN_ROI = 100
STATIC_RANGE_DEPTH_QUANTILE = 0.05
MAX_CLOSING_DIFFERENCE_DELTA_SECONDS = 0.060
MIN_EVALUATED_FRAMES_PER_SEQUENCE = 100

T_ROS = np.asarray(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
T_M = np.asarray(
    [
        [1.0157, 0.1828, -0.2389, 0.0113],
        [0.0009, -0.8431, -0.6413, -0.0098],
        [-0.3009, 0.6147, -0.8085, 0.0111],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(data: bytes, columns: int) -> list[list[str]]:
    result = [
        line.split()
        for line in data.decode("utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if not result or any(len(row) != columns for row in result):
        raise ValueError("invalid source index")
    times = [float(row[0]) for row in result]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timestamps must be strictly increasing")
    return result


def nearest_pose(
    poses: list[list[str]], pose_times: list[float], timestamp: float
) -> tuple[list[str], float] | None:
    index = bisect.bisect_left(pose_times, timestamp)
    candidates = poses[max(0, index - 1) : min(len(poses), index + 1)]
    selected = min(candidates, key=lambda row: abs(float(row[0]) - timestamp))
    delta = abs(float(selected[0]) - timestamp)
    return (
        (selected, delta)
        if delta <= MAX_POSE_JOIN_DELTA_SECONDS
        else None
    )


def quaternion_xyzw_rotation(values: np.ndarray) -> np.ndarray:
    x, y, z, w = values / np.linalg.norm(values)
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def pose_matrix(row: list[str]) -> np.ndarray:
    values = np.asarray([float(value) for value in row[1:]], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_xyzw_rotation(values[3:7])
    matrix[:3, 3] = values[:3]
    return matrix


def official_map_from_camera(pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_ROS) @ pose @ T_ROS @ T_M


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def reduce_map(points: np.ndarray) -> np.ndarray:
    minimum = points.min(axis=0).astype(np.float64)
    cells = np.floor(
        (points.astype(np.float64) - minimum) / MAP_REDUCTION_VOXEL_METERS
    ).astype(np.int64)
    widths = cells.max(axis=0) + 1
    keys = cells[:, 0] + widths[0] * (
        cells[:, 1] + widths[1] * cells[:, 2]
    )
    _, indexes = np.unique(keys, return_index=True)
    return points[np.sort(indexes)].astype(np.float64)


def static_surface_observation(
    map_points: np.ndarray, map_from_camera: np.ndarray
) -> dict[str, Any] | None:
    camera_points = transform_points(
        map_points, np.linalg.inv(map_from_camera)
    )
    z = camera_points[:, 2]
    front = z > 0.0
    if not front.any():
        return None
    x = camera_points[:, 0]
    y = camera_points[:, 1]
    u = FX * x / np.maximum(z, 1e-12) + CX
    v = FY * y / np.maximum(z, 1e-12) + CY
    selected = (
        front
        & (u >= ROI_U_FRACTION[0] * IMAGE_WIDTH)
        & (u < ROI_U_FRACTION[1] * IMAGE_WIDTH)
        & (v >= ROI_V_FRACTION[0] * IMAGE_HEIGHT)
        & (v < ROI_V_FRACTION[1] * IMAGE_HEIGHT)
    )
    count = int(selected.sum())
    if count < MIN_STATIC_MAP_POINTS_IN_ROI:
        return None
    selected_depth = z[selected]
    return {
        "static_map_point_count_in_roi": count,
        "static_surface_depth_q05_meters": float(
            np.quantile(selected_depth, STATIC_RANGE_DEPTH_QUANTILE)
        ),
        "static_surface_depth_median_meters": float(
            np.median(selected_depth)
        ),
        "static_surface_depth_minimum_meters": float(
            selected_depth.min()
        ),
    }


def build_sequence(
    archive_path: Path,
    sequence_id: str,
    expected_sha256: str,
    window: dict[str, Any],
    map_points: np.ndarray,
) -> dict[str, Any]:
    if sha256(archive_path) != expected_sha256:
        raise ValueError("discovery archive SHA-256 mismatch")
    prefix = f"{sequence_id}/"
    with zipfile.ZipFile(archive_path) as archive:
        rgb = rows(archive.read(f"{prefix}rgb.txt"), 2)
        poses = rows(archive.read(f"{prefix}groundtruth.txt"), 8)
    pose_times = [float(row[0]) for row in poses]
    frame_rows = [
        row
        for row in rgb
        if window["start_timestamp"]
        <= float(row[0])
        < window["end_timestamp_exclusive"]
    ]
    units: list[dict[str, Any]] = []
    previous_evaluated: dict[str, Any] | None = None
    for index, rgb_row in enumerate(frame_rows):
        timestamp = float(rgb_row[0])
        base = {
            "claim_id": "C2_STATIC_SURFACE_CLOSING_RETENTION",
            "source_family": "BONN_RGBD_DYNAMIC",
            "capture_cluster_id": "BONN_SHARED_CAPTURE_VOLUME_R0",
            "session_id": sequence_id,
            "unit_id": f"{sequence_id}:rgb:{rgb_row[0]}",
            "frame_index_within_window": index,
            "rgb_timestamp": timestamp,
            "rgb_member": f"{prefix}{rgb_row[1]}",
            "evidence_grade": "A",
            "truth_provenance": (
                "LEICA_BLK360_STATIC_MAP_PLUS_OPTITRACK_POSE_"
                "OFFICIAL_TRANSFORM"
            ),
            "interpolated_fraction": 0.0,
            "transform_chain_status": (
                "BONN_OFFICIAL_TRANSFORM_CHAIN_GEOMETRY_VALIDATED"
            ),
        }
        pose_match = nearest_pose(poses, pose_times, timestamp)
        if pose_match is None:
            units.append(
                {
                    **base,
                    "eligible": False,
                    "evaluated": False,
                    "abstained": True,
                    "abstention_reason": (
                        "POSE_JOIN_EXCEEDS_FROZEN_40MS_HARD_CAP"
                    ),
                    "time_sync_status": "FAIL_POSE_JOIN",
                }
            )
            previous_evaluated = None
            continue
        pose_row, pose_delta = pose_match
        map_from_camera = official_map_from_camera(pose_matrix(pose_row))
        observation = static_surface_observation(
            map_points, map_from_camera
        )
        if observation is None:
            units.append(
                {
                    **base,
                    "eligible": False,
                    "evaluated": False,
                    "abstained": True,
                    "abstention_reason": (
                        "INSUFFICIENT_STATIC_MAP_POINTS_IN_FROZEN_ROI"
                    ),
                    "time_sync_status": "PASS",
                    "pose_join_delta_seconds": pose_delta,
                    "camera_origin_map_xyz_meters": (
                        map_from_camera[:3, 3].tolist()
                    ),
                }
            )
            previous_evaluated = None
            continue
        unit = {
            **base,
            **observation,
            "eligible": True,
            "evaluated": True,
            "abstained": False,
            "abstention_reason": None,
            "time_sync_status": "PASS",
            "pose_timestamp": float(pose_row[0]),
            "pose_join_delta_seconds": pose_delta,
            "camera_origin_map_xyz_meters": (
                map_from_camera[:3, 3].tolist()
            ),
            "static_surface_closing_rate_meters_per_second": None,
            "closing_rate_status": "ABSTAIN_NO_PREVIOUS_CONTIGUOUS_TRUTH",
        }
        if previous_evaluated is not None:
            delta = timestamp - previous_evaluated["rgb_timestamp"]
            if 0.0 < delta <= MAX_CLOSING_DIFFERENCE_DELTA_SECONDS:
                unit["static_surface_closing_rate_meters_per_second"] = (
                    previous_evaluated[
                        "static_surface_depth_q05_meters"
                    ]
                    - unit["static_surface_depth_q05_meters"]
                ) / delta
                unit["closing_rate_status"] = "EVALUATED_BACKWARD_DIFFERENCE"
        units.append(unit)
        previous_evaluated = unit
    evaluated = [item for item in units if item["evaluated"]]
    closing = [
        item
        for item in evaluated
        if item["static_surface_closing_rate_meters_per_second"] is not None
    ]
    return {
        "sequence_id": sequence_id,
        "window_start_timestamp": window["start_timestamp"],
        "window_end_timestamp_exclusive": window[
            "end_timestamp_exclusive"
        ],
        "units": units,
        "counts": {
            "frozen_rgb_frame_count": len(units),
            "evaluated_frame_count": len(evaluated),
            "abstained_frame_count": len(units) - len(evaluated),
            "closing_rate_evaluated_frame_count": len(closing),
        },
        "continuous_truth_summary": {
            "static_surface_depth_q05_minimum_meters": (
                min(
                    item["static_surface_depth_q05_meters"]
                    for item in evaluated
                )
                if evaluated
                else None
            ),
            "static_surface_depth_q05_maximum_meters": (
                max(
                    item["static_surface_depth_q05_meters"]
                    for item in evaluated
                )
                if evaluated
                else None
            ),
            "closing_rate_median_meters_per_second": (
                float(
                    np.median(
                        [
                            item[
                                "static_surface_closing_rate_meters_per_second"
                            ]
                            for item in closing
                        ]
                    )
                )
                if closing
                else None
            ),
        },
    }


def build(
    archive_audit: dict[str, Any],
    pose_ledger: dict[str, Any],
    map_receipt: dict[str, Any],
    transform_receipt: dict[str, Any],
    archive_dir: Path,
    map_points_path: Path,
) -> dict[str, Any]:
    if (
        transform_receipt["terminal"]
        != "BONN_OFFICIAL_TRANSFORM_CHAIN_GEOMETRY_VALIDATED"
    ):
        raise ValueError("Bonn transform authority not validated")
    expected_map_sha = map_receipt["deterministic_geometry_reduction"][
        "output_sha256"
    ]
    if sha256(map_points_path) != expected_map_sha:
        raise ValueError("static-map geometry SHA-256 mismatch")
    with np.load(map_points_path) as payload:
        map_points = reduce_map(payload["xyz_meters"])
    archives = {
        item["sequence_id"]: item for item in archive_audit["archives"]
    }
    windows = {
        item["sequence_id"]: item["windows"][0]
        for item in pose_ledger["sequences"]
    }
    sequences = [
        build_sequence(
            archive_dir / item["archive_filename"],
            item["sequence_id"],
            item["archive_sha256"],
            windows[item["sequence_id"]],
            map_points,
        )
        for item in archive_audit["archives"]
    ]
    enough = all(
        item["counts"]["evaluated_frame_count"]
        >= MIN_EVALUATED_FRAMES_PER_SEQUENCE
        for item in sequences
    )
    counts = {
        "sequence_count": len(sequences),
        "frozen_rgb_frame_count": sum(
            item["counts"]["frozen_rgb_frame_count"] for item in sequences
        ),
        "evaluated_frame_count": sum(
            item["counts"]["evaluated_frame_count"] for item in sequences
        ),
        "abstained_frame_count": sum(
            item["counts"]["abstained_frame_count"] for item in sequences
        ),
        "closing_rate_evaluated_frame_count": sum(
            item["counts"]["closing_rate_evaluated_frame_count"]
            for item in sequences
        ),
    }
    return {
        "schema_version": "bonn_static_surface_truth_ledger_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "claim_id": "C2_STATIC_SURFACE_CLOSING_RETENTION",
        "source_family": "BONN_RGBD_DYNAMIC",
        "frozen_input_receipts": {
            "archive_audit_sha256": None,
            "pose_ledger_sha256": None,
            "static_map_geometry_sha256": None,
            "transform_validation_sha256": None,
            "static_map_points_sha256": expected_map_sha,
        },
        "truth_contract": {
            "map_reduction_voxel_meters": MAP_REDUCTION_VOXEL_METERS,
            "reduced_static_map_point_count": len(map_points),
            "roi_u_fraction": list(ROI_U_FRACTION),
            "roi_v_fraction": list(ROI_V_FRACTION),
            "minimum_static_map_points_in_roi": (
                MIN_STATIC_MAP_POINTS_IN_ROI
            ),
            "static_range_depth_quantile": STATIC_RANGE_DEPTH_QUANTILE,
            "maximum_pose_join_delta_seconds": (
                MAX_POSE_JOIN_DELTA_SECONDS
            ),
            "maximum_closing_difference_delta_seconds": (
                MAX_CLOSING_DIFFERENCE_DELTA_SECONDS
            ),
            "minimum_evaluated_frames_per_sequence": (
                MIN_EVALUATED_FRAMES_PER_SEQUENCE
            ),
            "selection_independent_of_rgb_pixels_or_candidate_signal": True,
            "no_alarm_or_event_threshold": True,
        },
        "sequences": sequences,
        "counts": counts,
        "read_firewall": {
            "discovery_rgb_member_read_or_decode_count": 0,
            "discovery_depth_member_read_or_decode_count": 0,
            "validation_or_holdout_read_count": 0,
            "old_window_selection_tuning_acceptance_reads": 0,
            "candidate_signal_computed": False,
        },
        "claim_effect": {
            "Bonn_C1": "ABSTAIN_NO_PURE_ROTATION_DISCOVERY_WINDOW",
            "Bonn_C2_static_truth": (
                "AVAILABLE_WITH_UNIT_LEVEL_ABSTENTION"
                if enough
                else "HOLD_INSUFFICIENT_STATIC_MAP_FOV_SUPPORT"
            ),
            "algorithm_result": "NOT_RUN",
        },
        "terminal": (
            "BONN_STATIC_SURFACE_CONTINUOUS_TRUTH_AVAILABLE"
            if enough
            else "HOLD_BONN_STATIC_SURFACE_TRUTH_SUPPORT"
        ),
        "status": "VALID",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-audit", required=True, type=Path)
    parser.add_argument("--pose-ledger", required=True, type=Path)
    parser.add_argument("--map-receipt", required=True, type=Path)
    parser.add_argument("--transform-receipt", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--map-points", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "archive_audit": json.loads(
            args.archive_audit.read_text(encoding="utf-8")
        ),
        "pose_ledger": json.loads(
            args.pose_ledger.read_text(encoding="utf-8")
        ),
        "map_receipt": json.loads(
            args.map_receipt.read_text(encoding="utf-8")
        ),
        "transform_receipt": json.loads(
            args.transform_receipt.read_text(encoding="utf-8")
        ),
    }
    receipt = build(
        **inputs,
        archive_dir=args.archive_dir,
        map_points_path=args.map_points,
    )
    receipt["frozen_input_receipts"].update(
        {
            "archive_audit_sha256": sha256(args.archive_audit),
            "pose_ledger_sha256": sha256(args.pose_ledger),
            "static_map_geometry_sha256": sha256(args.map_receipt),
            "transform_validation_sha256": sha256(
                args.transform_receipt
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "terminal": receipt["terminal"],
                **receipt["counts"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["terminal"].endswith("AVAILABLE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
