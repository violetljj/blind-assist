"""Bonn RGB-D metadata and frozen SATOM-R0 source mechanics."""

from __future__ import annotations

import bisect
import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np


BONN_INTRINSICS = np.asarray(
    [[542.822841, 0.0, 315.593520], [0.0, 542.576870, 237.756098], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
BONN_T_ROS = np.asarray(
    [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
BONN_T_MARKER = np.asarray(
    [
        [1.0157, 0.1828, -0.2389, 0.0113],
        [0.0009, -0.8431, -0.6413, -0.0098],
        [-0.3009, 0.6147, -0.8085, 0.0111],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
WORLD_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
DEPTH_SCALE = 5000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def rank_hash(sequence_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{sequence_id}".encode("utf-8")).hexdigest().upper()


def read_index(path: Path, columns: int) -> list[list[str]]:
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if not rows or any(len(row) != columns for row in rows):
        raise ValueError(f"invalid Bonn index: {path}")
    timestamps = [float(row[0]) for row in rows]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"non-monotonic Bonn index: {path}")
    return rows


def _normalize_quaternion(values: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not math.isfinite(norm) or norm <= 0:
        raise ValueError("invalid Bonn quaternion")
    return quaternion / norm


def _slerp(left: Sequence[float], right: Sequence[float], alpha: float) -> np.ndarray:
    q0 = _normalize_quaternion(left)
    q1 = _normalize_quaternion(right)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(q0 + alpha * (q1 - q0))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return _normalize_quaternion(
        math.sin((1.0 - alpha) * angle) / sine * q0
        + math.sin(alpha * angle) / sine * q1
    )


def _quaternion_matrix_xyzw(values: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(values)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def interpolate_camera_pose(
    poses: Sequence[Sequence[str]], timestamp_s: float, maximum_bracket_s: float
) -> tuple[np.ndarray, tuple[float, float]] | None:
    pose_times = [float(row[0]) for row in poses]
    index = bisect.bisect_left(pose_times, timestamp_s)
    if index == 0 or index == len(poses):
        return None
    left, right = poses[index - 1], poses[index]
    left_time, right_time = float(left[0]), float(right[0])
    if timestamp_s - left_time > maximum_bracket_s or right_time - timestamp_s > maximum_bracket_s:
        return None
    alpha = (timestamp_s - left_time) / (right_time - left_time)
    translation = (1.0 - alpha) * np.asarray(left[1:4], dtype=np.float64)
    translation += alpha * np.asarray(right[1:4], dtype=np.float64)
    quaternion = _slerp([float(value) for value in left[4:8]], [float(value) for value in right[4:8]], alpha)
    raw_pose = np.eye(4, dtype=np.float64)
    raw_pose[:3, :3] = _quaternion_matrix_xyzw(quaternion)
    raw_pose[:3, 3] = translation
    camera_pose = BONN_T_ROS @ raw_pose @ BONN_T_ROS @ BONN_T_MARKER
    return np.ascontiguousarray(camera_pose), (left_time, right_time)


def associate_rgb_depth(
    rgb_rows: Sequence[Sequence[str]],
    depth_rows: Sequence[Sequence[str]],
    maximum_delta_s: float,
) -> list[tuple[Sequence[str], Sequence[str]]]:
    depth_times = [float(row[0]) for row in depth_rows]
    used: set[int] = set()
    pairs = []
    for rgb in rgb_rows:
        timestamp = float(rgb[0])
        index = bisect.bisect_left(depth_times, timestamp)
        candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(depth_rows) and candidate not in used]
        if not candidates:
            continue
        selected = min(candidates, key=lambda candidate: (abs(depth_times[candidate] - timestamp), candidate))
        if abs(depth_times[selected] - timestamp) <= maximum_delta_s:
            used.add(selected)
            pairs.append((rgb, depth_rows[selected]))
    return pairs


def frozen_frame_rows(sequence_root: Path, contract: dict) -> list[dict]:
    rgb = read_index(sequence_root / "rgb.txt", 2)
    depth = read_index(sequence_root / "depth.txt", 2)
    poses = read_index(sequence_root / "groundtruth.txt", 8)
    pairs = associate_rgb_depth(rgb, depth, float(contract["maximum_rgb_depth_delta_s"]))
    eligible = []
    for rgb_row, depth_row in pairs:
        pose = interpolate_camera_pose(poses, float(rgb_row[0]), float(contract["maximum_pose_bracket_s"]))
        if pose is not None:
            eligible.append((rgb_row, depth_row, pose))
    count = int(contract["frames_per_parent"])
    cadence_s = 1.0 / float(contract["sampling_hz"])
    required_duration = cadence_s * (count - 1)
    if not eligible or float(eligible[-1][0][0]) - float(eligible[0][0][0]) < required_duration:
        raise ValueError(f"insufficient frozen cadence support: {sequence_root.name}")
    maximum_offset = float(eligible[-1][0][0]) - float(eligible[0][0][0]) - required_duration
    offset_fraction = int(rank_hash(sequence_root.name, str(contract["frame_start_seed"]))[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    start = float(eligible[0][0][0]) + maximum_offset * offset_fraction
    eligible_times = [float(row[0][0]) for row in eligible]
    selected_indexes: list[int] = []
    previous = -1
    for frame_index in range(count):
        target = start + frame_index * cadence_s
        insertion = bisect.bisect_left(eligible_times, target)
        candidates = [index for index in (insertion - 1, insertion) if previous < index < len(eligible)]
        if not candidates:
            raise ValueError(f"cadence selection exhausted: {sequence_root.name}/{frame_index}")
        selected = min(candidates, key=lambda index: (abs(eligible_times[index] - target), index))
        if abs(eligible_times[selected] - target) > float(contract["maximum_cadence_jitter_s"]):
            raise ValueError(f"cadence jitter exceeded: {sequence_root.name}/{frame_index}")
        selected_indexes.append(selected)
        previous = selected
    output = []
    for frame_index, selected in enumerate(selected_indexes):
        rgb_row, depth_row, (pose, bracket) = eligible[selected]
        gravity_down = pose[:3, :3].T @ (-WORLD_UP)
        gravity_down /= np.linalg.norm(gravity_down)
        output.append(
            {
                "frame_index": frame_index,
                "rgb_timestamp_s": float(rgb_row[0]),
                "rgb_relative_path": str(rgb_row[1]),
                "depth_timestamp_s": float(depth_row[0]),
                "depth_relative_path": str(depth_row[1]),
                "pose_bracket_s": [float(bracket[0]), float(bracket[1])],
                "world_from_camera": pose,
                "gravity_down_camera": gravity_down,
            }
        )
    return output


def estimate_camera_height_m(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    gravity_down_camera: np.ndarray,
    quantile: float,
    minimum_m: float,
    maximum_m: float,
) -> float:
    """Frozen robust floor-distance proxy, run separately on prior and truth."""
    depth = np.asarray(depth_m, dtype=np.float64)
    height, width = depth.shape
    yy, xx = np.mgrid[0:height:4, 0:width:4]
    z = depth[::4, ::4]
    valid = np.isfinite(z) & (z >= 0.08) & (z <= 4.0)
    if not np.any(valid):
        raise ValueError("no valid depth for camera-height estimation")
    x = (xx - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (yy - intrinsics[1, 2]) * z / intrinsics[1, 1]
    gravity = np.asarray(gravity_down_camera, dtype=np.float64)
    drop = gravity[0] * x + gravity[1] * y + gravity[2] * z
    candidates = drop[valid & np.isfinite(drop) & (drop > minimum_m * 0.5)]
    if candidates.size < 64:
        raise ValueError("insufficient floor-distance candidates")
    estimate = float(np.quantile(candidates, quantile))
    if not minimum_m <= estimate <= maximum_m:
        raise ValueError(f"camera-height estimate outside frozen bounds: {estimate}")
    return estimate
