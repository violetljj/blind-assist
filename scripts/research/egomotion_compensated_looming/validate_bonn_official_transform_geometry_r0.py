#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FX = 542.822841
FY = 542.576870
CX = 315.593520
CY = 237.756098
DEPTH_SCALE = 5000.0
DEPTH_PIXEL_STRIDE = 8
MAX_POSE_JOIN_DELTA_SECONDS = 0.040
MAX_DEPTH_METERS = 10.0
VOXEL_METERS = 0.05
SUPPORT_NEIGHBOR_RADIUS_VOXELS = 1
MIN_PER_SEQUENCE_MEDIAN_SUPPORT = 0.25
MIN_GLOBAL_MEDIAN_OFFICIAL_MINUS_INVERSE_SUPPORT = 0.20
MIN_EVALUATED_SAMPLES_PER_SEQUENCE = 2

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
    if delta > MAX_POSE_JOIN_DELTA_SECONDS:
        return None
    return selected, delta


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


def depth_points(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    image = np.asarray(Image.open(io.BytesIO(data)), dtype=np.uint16)
    if image.shape != (480, 640):
        raise ValueError("unexpected Bonn depth shape")
    rows_, columns = np.mgrid[
        0 : image.shape[0] : DEPTH_PIXEL_STRIDE,
        0 : image.shape[1] : DEPTH_PIXEL_STRIDE,
    ]
    z = image[::DEPTH_PIXEL_STRIDE, ::DEPTH_PIXEL_STRIDE].astype(
        np.float64
    ) / DEPTH_SCALE
    valid = (z > 0.0) & (z <= MAX_DEPTH_METERS)
    z = z[valid]
    u = columns[valid].astype(np.float64)
    v = rows_[valid].astype(np.float64)
    points = np.column_stack(
        ((u - CX) * z / FX, (v - CY) * z / FY, z)
    )
    return points, {
        "decoded_shape": list(image.shape),
        "sampled_pixel_count": int(rows_.size),
        "valid_depth_point_count": len(points),
        "valid_depth_fraction": float(len(points) / rows_.size),
    }


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def packed_voxel_keys(
    points: np.ndarray, origin: np.ndarray, widths: np.ndarray
) -> np.ndarray:
    cells = np.floor((points - origin) / VOXEL_METERS).astype(np.int64)
    return cells[:, 0] + widths[0] * (
        cells[:, 1] + widths[1] * cells[:, 2]
    )


def support_fraction(
    points: np.ndarray,
    map_keys: np.ndarray,
    origin: np.ndarray,
    widths: np.ndarray,
) -> float:
    cells = np.floor((points - origin) / VOXEL_METERS).astype(np.int64)
    supported = np.zeros(len(cells), dtype=bool)
    for dz in range(
        -SUPPORT_NEIGHBOR_RADIUS_VOXELS,
        SUPPORT_NEIGHBOR_RADIUS_VOXELS + 1,
    ):
        for dy in range(
            -SUPPORT_NEIGHBOR_RADIUS_VOXELS,
            SUPPORT_NEIGHBOR_RADIUS_VOXELS + 1,
        ):
            for dx in range(
                -SUPPORT_NEIGHBOR_RADIUS_VOXELS,
                SUPPORT_NEIGHBOR_RADIUS_VOXELS + 1,
            ):
                shifted = cells + np.asarray([dx, dy, dz], dtype=np.int64)
                keys = shifted[:, 0] + widths[0] * (
                    shifted[:, 1] + widths[1] * shifted[:, 2]
                )
                indexes = np.searchsorted(map_keys, keys)
                inside = indexes < len(map_keys)
                matched = np.zeros(len(keys), dtype=bool)
                matched[inside] = map_keys[indexes[inside]] == keys[inside]
                supported |= matched
    return float(supported.mean()) if len(supported) else 0.0


def build_voxel_index(
    map_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = map_points.min(axis=0).astype(np.float64)
    maximum = map_points.max(axis=0).astype(np.float64)
    origin = minimum - 2.0 * VOXEL_METERS
    widths = (
        np.ceil((maximum - origin) / VOXEL_METERS).astype(np.int64) + 5
    )
    keys = np.unique(packed_voxel_keys(map_points, origin, widths))
    return keys, origin, widths


def validate(
    sample_freeze: dict[str, Any],
    map_receipt: dict[str, Any],
    archive_dir: Path,
    map_points_path: Path,
) -> dict[str, Any]:
    expected_map_sha = map_receipt["deterministic_geometry_reduction"][
        "output_sha256"
    ]
    if sha256(map_points_path) != expected_map_sha:
        raise ValueError("static-map geometry SHA-256 mismatch")
    with np.load(map_points_path) as payload:
        map_points = payload["xyz_meters"].astype(np.float64)
    map_minimum = map_points.min(axis=0)
    map_maximum = map_points.max(axis=0)
    map_keys, origin, widths = build_voxel_index(map_points)

    sequence_results: list[dict[str, Any]] = []
    for sequence in sample_freeze["sequences"]:
        archive_path = archive_dir / sequence["archive_filename"]
        if sha256(archive_path) != sequence["archive_sha256"]:
            raise ValueError("discovery archive SHA-256 mismatch")
        prefix = f"{sequence['sequence_id']}/"
        with zipfile.ZipFile(archive_path) as archive:
            poses = rows(archive.read(f"{prefix}groundtruth.txt"), 8)
            pose_times = [float(row[0]) for row in poses]
            samples: list[dict[str, Any]] = []
            for frozen in sequence["samples"]:
                pose_match = nearest_pose(
                    poses, pose_times, frozen["rgb_timestamp"]
                )
                if pose_match is None:
                    nearest_delta = min(
                        abs(value - frozen["rgb_timestamp"])
                        for value in pose_times
                    )
                    samples.append(
                        {
                            **frozen,
                            "eligible": False,
                            "evaluated": False,
                            "abstained": True,
                            "abstention_reason": (
                                "POSE_JOIN_EXCEEDS_FROZEN_40MS_HARD_CAP"
                            ),
                            "nearest_pose_join_delta_seconds": nearest_delta,
                            "depth_member_read_or_decoded": False,
                        }
                    )
                    continue
                pose_row, pose_delta = pose_match
                points, depth_audit = depth_points(
                    archive.read(frozen["depth_member"])
                )
                map_from_camera = official_map_from_camera(
                    pose_matrix(pose_row)
                )
                official_points = transform_points(points, map_from_camera)
                inverse_direction_points = transform_points(
                    points, np.linalg.inv(map_from_camera)
                )
                inside_map_bbox = np.all(
                    (official_points >= map_minimum)
                    & (official_points <= map_maximum),
                    axis=1,
                )
                official_support = support_fraction(
                    official_points, map_keys, origin, widths
                )
                inverse_support = support_fraction(
                    inverse_direction_points, map_keys, origin, widths
                )
                samples.append(
                    {
                        **frozen,
                        **depth_audit,
                        "eligible": True,
                        "evaluated": True,
                        "abstained": False,
                        "abstention_reason": None,
                        "depth_member_read_or_decoded": True,
                        "pose_timestamp": float(pose_row[0]),
                        "pose_join_delta_seconds": pose_delta,
                        "official_camera_origin_map_xyz_meters": (
                            map_from_camera[:3, 3].tolist()
                        ),
                        "official_transformed_depth_minimum_xyz_meters": (
                            official_points.min(axis=0).tolist()
                        ),
                        "official_transformed_depth_maximum_xyz_meters": (
                            official_points.max(axis=0).tolist()
                        ),
                        "official_transformed_depth_inside_map_bbox_fraction": (
                            float(inside_map_bbox.mean())
                        ),
                        "official_neighbor_voxel_support_fraction": (
                            official_support
                        ),
                        "inverse_direction_negative_control_support_fraction": (
                            inverse_support
                        ),
                        "official_minus_inverse_support_fraction": (
                            official_support - inverse_support
                        ),
                    }
                )
        evaluated_samples = [
            item for item in samples if item.get("evaluated")
        ]
        official_values = [
            item["official_neighbor_voxel_support_fraction"]
            for item in evaluated_samples
        ]
        inverse_values = [
            item["inverse_direction_negative_control_support_fraction"]
            for item in evaluated_samples
        ]
        if len(evaluated_samples) < MIN_EVALUATED_SAMPLES_PER_SEQUENCE:
            raise ValueError("insufficient evaluated transform samples")
        sequence_results.append(
            {
                "sequence_id": sequence["sequence_id"],
                "samples": samples,
                "frozen_sample_count": len(samples),
                "evaluated_sample_count": len(evaluated_samples),
                "abstained_sample_count": len(samples) - len(evaluated_samples),
                "median_official_support_fraction": float(
                    np.median(official_values)
                ),
                "median_inverse_direction_support_fraction": float(
                    np.median(inverse_values)
                ),
            }
        )

    all_samples = [
        sample
        for sequence in sequence_results
        for sample in sequence["samples"]
        if sample.get("evaluated")
    ]
    official = np.asarray(
        [
            item["official_neighbor_voxel_support_fraction"]
            for item in all_samples
        ]
    )
    inverse = np.asarray(
        [
            item["inverse_direction_negative_control_support_fraction"]
            for item in all_samples
        ]
    )
    per_sequence_pass = all(
        item["median_official_support_fraction"]
        >= MIN_PER_SEQUENCE_MEDIAN_SUPPORT
        for item in sequence_results
    )
    median_advantage = float(np.median(official - inverse))
    direction_pass = (
        median_advantage
        >= MIN_GLOBAL_MEDIAN_OFFICIAL_MINUS_INVERSE_SUPPORT
    )
    passed = bool(per_sequence_pass and direction_pass)
    return {
        "schema_version": "bonn_official_transform_geometry_validation_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "source_family": "BONN_RGBD_DYNAMIC",
        "frozen_input_receipts": {
            "sample_freeze_sha256": None,
            "static_map_geometry_sha256": None,
            "static_map_points_sha256": expected_map_sha,
        },
        "transform_under_test": {
            "formula": (
                "T_map_from_camera_i = inverse(T_ROS) * T_i * "
                "T_ROS * T_m"
            ),
            "status_before_test": (
                "OFFICIAL_FIRST_POSE_FORMULA_EXTENDED_PER_FRAME_AS_"
                "EXPLICIT_INFERENCE"
            ),
            "negative_control": "INVERSE_DIRECTION_APPLIED_TO_CAMERA_POINTS",
            "T_ROS_is_self_inverse": bool(
                np.allclose(np.linalg.inv(T_ROS), T_ROS)
            ),
            "T_m_determinant": float(np.linalg.det(T_M[:3, :3])),
        },
        "frozen_validation_contract": {
            "depth_scale": DEPTH_SCALE,
            "depth_pixel_stride": DEPTH_PIXEL_STRIDE,
            "maximum_pose_join_delta_seconds": (
                MAX_POSE_JOIN_DELTA_SECONDS
            ),
            "maximum_depth_meters": MAX_DEPTH_METERS,
            "voxel_meters": VOXEL_METERS,
            "support_neighbor_radius_voxels": (
                SUPPORT_NEIGHBOR_RADIUS_VOXELS
            ),
            "minimum_per_sequence_median_support_fraction": (
                MIN_PER_SEQUENCE_MEDIAN_SUPPORT
            ),
            "minimum_evaluated_samples_per_sequence": (
                MIN_EVALUATED_SAMPLES_PER_SEQUENCE
            ),
            "minimum_global_median_official_minus_inverse_support_fraction": (
                MIN_GLOBAL_MEDIAN_OFFICIAL_MINUS_INVERSE_SUPPORT
            ),
        },
        "map_index": {
            "downsampled_point_count": len(map_points),
            "occupied_voxel_count": len(map_keys),
            "minimum_xyz_meters": map_minimum.tolist(),
            "maximum_xyz_meters": map_maximum.tolist(),
            "voxel_origin_xyz_meters": origin.tolist(),
            "voxel_widths": widths.tolist(),
        },
        "sequences": sequence_results,
        "aggregate": {
            "sample_count": len(all_samples),
            "median_official_support_fraction": float(
                np.median(official)
            ),
            "minimum_official_support_fraction": float(official.min()),
            "median_inverse_direction_support_fraction": float(
                np.median(inverse)
            ),
            "median_official_minus_inverse_support_fraction": (
                median_advantage
            ),
            "per_sequence_support_gate_passed": per_sequence_pass,
            "direction_negative_control_gate_passed": direction_pass,
        },
        "read_firewall": {
            "discovery_depth_member_read_or_decode_count": len(all_samples),
            "discovery_rgb_member_read_or_decode_count": 0,
            "validation_or_holdout_read_count": 0,
            "old_window_selection_tuning_acceptance_reads": 0,
            "candidate_signal_computed": False,
        },
        "unit_level_abstention": {
            "frozen_sample_count": sum(
                item["frozen_sample_count"] for item in sequence_results
            ),
            "evaluated_sample_count": len(all_samples),
            "abstained_sample_count": sum(
                item["abstained_sample_count"] for item in sequence_results
            ),
            "gate_weakened_after_abstention": False,
        },
        "claim_effect": {
            "C1_ROTATION_LEAKAGE_SUPPRESSION": "ABSTAIN_NO_C1_WINDOW",
            "C2_STATIC_SURFACE_CLOSING_RETENTION": (
                "TRANSFORM_INPUT_AUTHORITY_AVAILABLE"
                if passed
                else "ABSTAIN_TRANSFORM_AUTHORITY_FAILED"
            ),
            "algorithm_result": "NOT_RUN",
        },
        "terminal": (
            "BONN_OFFICIAL_TRANSFORM_CHAIN_GEOMETRY_VALIDATED"
            if passed
            else "HOLD_BONN_STATIC_MAP_TRANSFORM_AUTHORITY"
        ),
        "status": "VALID",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-freeze", required=True, type=Path)
    parser.add_argument("--map-receipt", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--map-points", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_freeze = json.loads(
        args.sample_freeze.read_text(encoding="utf-8")
    )
    map_receipt = json.loads(args.map_receipt.read_text(encoding="utf-8"))
    receipt = validate(
        sample_freeze,
        map_receipt,
        args.archive_dir,
        args.map_points,
    )
    receipt["frozen_input_receipts"]["sample_freeze_sha256"] = sha256(
        args.sample_freeze
    )
    receipt["frozen_input_receipts"]["static_map_geometry_sha256"] = sha256(
        args.map_receipt
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
                **receipt["aggregate"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["terminal"].endswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
