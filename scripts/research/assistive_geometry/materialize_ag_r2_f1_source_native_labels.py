#!/usr/bin/env python3
"""Materialize the frozen source-only R2 F1 supervision corpus from TUM RGB-D."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from ag_st_tum_rgbd import (
    TumIndexRow,
    TumPoseRow,
    _read_member,
    _tar_member_map,
    interpolate_camera_to_world,
    pair_rgb_depth_unique,
    parse_tum_index,
    parse_tum_poses,
)
from build_ag_st_factor_labels import (
    _pairwise_point_to_plane_edges,
    _pairwise_scalar_edges,
    backproject_depth_grid,
    compute_dense_normals,
)
from validate_ag_r2_f1_supervision_contract import (
    DEFAULT_CONTRACT,
    sha256_file,
    validate_path,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = REPO_ROOT / "artifacts.local/downloads/ag-r2-f1-supervision-tum13-r0"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-source-native-labels-tum13-r0"
METRIC_PROVENANCE = 1
SUPPORT_PROVENANCE = 2
BOUNDARY_PROVENANCE = 3
UNKNOWN_PROVENANCE = 0
REQUIRED_F1_SUPERVISION_FIELDS = {
    "metric_depth_m_hw",
    "metric_depth_valid_hw",
    "support_truth_hw",
    "support_truth_valid_hw",
    "support_plane_normal_camera_xyz",
    "camera_height_m",
    "support_plane_valid",
    "obstacle_evidence_truth_hw",
    "boundary_distance_px_hw",
    "evidence_truth_valid_hw",
}
FORBIDDEN_TASK_FIELD_TOKENS = {
    "clearance",
    "occupancy",
    "risk",
    "final_state",
    "ttc",
}


@dataclass(frozen=True)
class SelectedFrame:
    parent_id: str
    role: str
    orientation: str
    rgb: TumIndexRow
    depth: TumIndexRow
    rgb_u8_hwc: np.ndarray
    depth_m_hw: np.ndarray
    depth_valid_hw: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    pose_bracketing_gap_seconds: float
    gravity_up_camera: np.ndarray | None
    accelerometer_sample_count: int
    accelerometer_norm_mps2: float | None
    source_archive_sha256: str
    rgb_member_sha256: str
    depth_member_sha256: str
    metadata_member_sha256: str

    @property
    def frame_id(self) -> str:
        return f"{self.parent_id}__rgb{self.rgb.row_index:06d}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def parse_accelerometer_optional(text: str) -> np.ndarray | None:
    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        require(len(values) == 4 and all(math.isfinite(value) for value in values), "TUM accelerometer row invalid")
        rows.append(values)
    if not rows:
        return None
    result = np.asarray(rows, dtype=np.float64)
    require(np.all(np.diff(result[:, 0]) >= 0), "TUM accelerometer timestamps nonmonotonic")
    return result


def select_three_metadata_frames(
    token: str,
    parent_id: str,
    role: str,
    rgb_rows: list[TumIndexRow],
    depth_rows: list[TumIndexRow],
    pose_rows: list[TumPoseRow],
) -> list[tuple[TumIndexRow, TumIndexRow]]:
    pairing = pair_rgb_depth_unique(rgb_rows, depth_rows)
    rgb_by_index = {row.row_index: row for row in rgb_rows}
    eligible: list[TumIndexRow] = []
    for rgb_index in sorted(pairing, key=lambda index: rgb_by_index[index].timestamp_seconds):
        rgb = rgb_by_index[rgb_index]
        try:
            interpolate_camera_to_world(pose_rows, rgb.timestamp_seconds)
        except ValueError:
            continue
        eligible.append(rgb)
    require(len(eligible) >= 3, f"insufficient pose-bound RGB-D pairs: {parent_id}")
    selected: list[tuple[TumIndexRow, TumIndexRow]] = []
    count = len(eligible)
    for tercile in range(3):
        start = tercile * count // 3
        end = (tercile + 1) * count // 3
        bucket = eligible[start:end]
        require(bool(bucket), f"empty frame tercile: {parent_id}/{tercile}")
        rgb = min(
            bucket,
            key=lambda value: digest_text(
                f"{token}:FRAME:{role}:{parent_id}:{tercile}:"
                f"{value.relative_path}:{pairing[value.row_index].relative_path}"
            ),
        )
        selected.append((rgb, pairing[rgb.row_index]))
    return selected


def load_rgb(value: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(value)) as image:
        result = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    require(result.shape == (480, 640, 3), "TUM RGB shape drift")
    return result


def load_depth(value: bytes) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(io.BytesIO(value)) as image:
        raw = np.asarray(image).copy()
    require(raw.shape == (480, 640) and raw.dtype == np.uint16, "TUM depth payload drift")
    valid = raw > 0
    return raw.astype(np.float32) / 5000.0, valid


def intrinsics_matrix(values: list[float]) -> np.ndarray:
    require(len(values) == 4, "TUM intrinsics drift")
    fx, fy, cx, cy = (float(value) for value in values)
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def gravity_at_timestamp(
    accelerometer: np.ndarray | None,
    timestamp_seconds: float,
    window_seconds: float,
    imu_to_rgb: np.ndarray,
) -> tuple[np.ndarray | None, int, float | None]:
    if accelerometer is None:
        return None, 0, None
    delta = np.abs(accelerometer[:, 0] - timestamp_seconds)
    local = accelerometer[delta <= window_seconds, 1:]
    if len(local) < 3:
        return None, len(local), None
    vector = np.median(local, axis=0)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or not 8.0 <= norm <= 12.0:
        return None, len(local), norm
    gravity = np.asarray(imu_to_rgb, dtype=np.float64) @ (vector / norm)
    gravity_norm = float(np.linalg.norm(gravity))
    if not math.isfinite(gravity_norm) or gravity_norm <= 1e-9:
        return None, len(local), norm
    return gravity / gravity_norm, len(local), norm


def load_parent_frames(
    row: dict[str, Any],
    source_dir: Path,
    source: dict[str, Any],
    token: str,
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    parent_id = str(row["parent_id"])
    archive_path = source_dir / f"{parent_id}.tgz"
    require(archive_path.is_file(), f"source archive missing: {parent_id}")
    require(archive_path.stat().st_size == int(row["content_length"]), f"source archive length drift: {parent_id}")
    archive_sha = sha256_file(archive_path)
    intrinsics = intrinsics_matrix(source["intrinsics_fx_fy_cx_cy"][str(row["family"])])
    imu_to_rgb = np.asarray(source["imu_to_rgb_optical_rotation"], dtype=np.float64)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _tar_member_map(archive, parent_id)
        for required in ("rgb.txt", "depth.txt", "groundtruth.txt", "accelerometer.txt"):
            require(required in members, f"required source member missing: {parent_id}/{required}")
        metadata_bytes = {
            name: _read_member(archive, members[name])
            for name in ("rgb.txt", "depth.txt", "groundtruth.txt", "accelerometer.txt")
        }
        published_rgb_rows = parse_tum_index(metadata_bytes["rgb.txt"].decode("utf-8"))
        published_depth_rows = parse_tum_index(metadata_bytes["depth.txt"].decode("utf-8"))
        rgb_rows = [value for value in published_rgb_rows if value.relative_path in members]
        depth_rows = [value for value in published_depth_rows if value.relative_path in members]
        require(len(rgb_rows) == len(published_rgb_rows), f"published RGB member missing: {parent_id}")
        require(len(depth_rows) == len(published_depth_rows), f"published depth member missing: {parent_id}")
        pose_rows = parse_tum_poses(metadata_bytes["groundtruth.txt"].decode("utf-8"))
        accelerometer = parse_accelerometer_optional(metadata_bytes["accelerometer.txt"].decode("utf-8"))
        selected = select_three_metadata_frames(
            token,
            parent_id,
            str(row["role"]),
            rgb_rows,
            depth_rows,
            pose_rows,
        )
        metadata_sha = sha256_json(
            {name: sha256_bytes(value) for name, value in metadata_bytes.items()}
        )
        frames: list[SelectedFrame] = []
        for rgb, depth in selected:
            rgb_bytes = _read_member(archive, members[rgb.relative_path])
            depth_bytes = _read_member(archive, members[depth.relative_path])
            camera_to_world, pose_gap = interpolate_camera_to_world(
                pose_rows,
                rgb.timestamp_seconds,
            )
            gravity, sample_count, acceleration_norm = gravity_at_timestamp(
                accelerometer,
                rgb.timestamp_seconds,
                float(source["accelerometer_window_seconds"]),
                imu_to_rgb,
            )
            depth_m, depth_valid = load_depth(depth_bytes)
            frames.append(
                SelectedFrame(
                    parent_id=parent_id,
                    role=str(row["role"]),
                    orientation=str(row["orientation"]),
                    rgb=rgb,
                    depth=depth,
                    rgb_u8_hwc=load_rgb(rgb_bytes),
                    depth_m_hw=depth_m,
                    depth_valid_hw=depth_valid,
                    intrinsics=intrinsics,
                    camera_to_world=camera_to_world,
                    pose_bracketing_gap_seconds=pose_gap,
                    gravity_up_camera=gravity,
                    accelerometer_sample_count=sample_count,
                    accelerometer_norm_mps2=acceleration_norm,
                    source_archive_sha256=archive_sha,
                    rgb_member_sha256=sha256_bytes(rgb_bytes),
                    depth_member_sha256=sha256_bytes(depth_bytes),
                    metadata_member_sha256=metadata_sha,
                )
            )
    return frames, {
        "parent_id": parent_id,
        "source_archive": str(archive_path.resolve()),
        "source_archive_bytes": archive_path.stat().st_size,
        "source_archive_sha256": archive_sha,
        "selected_rgb_row_indices": [frame.rgb.row_index for frame in frames],
        "selected_depth_row_indices": [frame.depth.row_index for frame in frames],
        "accelerometer_stream_nonempty": accelerometer is not None,
    }


def parent_world_up(frames: list[SelectedFrame]) -> tuple[np.ndarray | None, dict[str, Any]]:
    vectors: list[np.ndarray] = []
    for frame in frames:
        if frame.gravity_up_camera is None:
            continue
        value = frame.camera_to_world[:3, :3] @ frame.gravity_up_camera
        vectors.append(value / np.linalg.norm(value))
    if len(vectors) < 2:
        return None, {
            "valid_frame_count": len(vectors),
            "maximum_angle_deg": None,
            "status": "UNKNOWN_INSUFFICIENT_SOURCE_GRAVITY",
        }
    values = np.stack(vectors)
    mean = np.sum(values, axis=0)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm <= 1e-9:
        return None, {
            "valid_frame_count": len(vectors),
            "maximum_angle_deg": None,
            "status": "UNKNOWN_INCONSISTENT_SOURCE_GRAVITY",
        }
    mean /= mean_norm
    angles = np.degrees(np.arccos(np.clip(values @ mean, -1.0, 1.0)))
    maximum = float(np.max(angles))
    if maximum > 15.0:
        return None, {
            "valid_frame_count": len(vectors),
            "maximum_angle_deg": maximum,
            "status": "UNKNOWN_INCONSISTENT_SOURCE_GRAVITY",
        }
    return mean, {
        "valid_frame_count": len(vectors),
        "maximum_angle_deg": maximum,
        "status": "SOURCE_GRAVITY_VALID",
        "world_up_unit": mean.tolist(),
    }


def horizontal_world_heights(
    frame: SelectedFrame,
    world_up: np.ndarray,
) -> np.ndarray:
    normals_camera, normal_valid = compute_dense_normals(
        frame.depth_m_hw,
        frame.depth_valid_hw,
        frame.intrinsics,
    )
    points_camera = backproject_depth_grid(frame.depth_m_hw, frame.intrinsics)
    rotation = frame.camera_to_world[:3, :3]
    translation = frame.camera_to_world[:3, 3]
    normals_world = np.einsum("...j,ij->...i", normals_camera, rotation)
    points_world = np.einsum("...j,ij->...i", points_camera, rotation) + translation
    world_height = np.einsum("...i,i->...", points_world, world_up)
    horizontal = (
        frame.depth_valid_hw
        & normal_valid
        & np.isfinite(world_height)
        & (frame.depth_m_hw <= 5.0)
        & (
            np.abs(np.einsum("...i,i->...", normals_world, world_up))
            >= math.cos(math.radians(20.0))
        )
    )
    return world_height[horizontal].astype(np.float64)[::2]


def persistent_height_modes(frame_heights: list[np.ndarray]) -> list[dict[str, Any]]:
    usable = [values for values in frame_heights if len(values) > 0]
    if len(usable) < 2:
        return []
    all_values = np.concatenate(usable)
    if len(all_values) == 0 or not np.all(np.isfinite(all_values)):
        return []
    height_bin_m = 0.04
    mode_radius_bins = 2
    low, high = np.quantile(all_values, (0.002, 0.998))
    first_bin = int(math.floor(float(low) / height_bin_m)) - 1
    last_bin = int(math.ceil(float(high) / height_bin_m)) + 1
    bin_ids = np.arange(first_bin, last_bin + 1, dtype=np.int64)
    frame_counts: list[np.ndarray] = []
    for values in usable:
        ids = np.floor(values / height_bin_m).astype(np.int64)
        counts = np.zeros(len(bin_ids), dtype=np.int64)
        inside = (ids >= first_bin) & (ids <= last_bin)
        np.add.at(counts, ids[inside] - first_bin, 1)
        frame_counts.append(counts)
    stacked = np.stack(frame_counts)
    total = np.sum(stacked, axis=0)
    smooth = np.convolve(total.astype(np.float64), [0.25, 0.5, 0.25], mode="same")
    peaks = [
        index
        for index in range(1, len(bin_ids) - 1)
        if smooth[index] >= smooth[index - 1] and smooth[index] > smooth[index + 1]
    ]
    peaks.sort(key=lambda index: (-smooth[index], int(bin_ids[index])))
    selected: list[int] = []
    for index in peaks:
        if any(abs(index - prior) <= mode_radius_bins for prior in selected):
            continue
        selected.append(index)
    candidates: list[dict[str, Any]] = []
    total_minimum = max(384, int(math.ceil(0.002 * len(all_values))))
    for index in selected:
        left = max(0, index - mode_radius_bins)
        right = min(len(bin_ids), index + mode_radius_bins + 1)
        per_frame = np.sum(stacked[:, left:right], axis=1)
        frame_minima = np.asarray(
            [max(96, int(math.ceil(0.002 * len(values)))) for values in usable],
            dtype=np.int64,
        )
        persistent = per_frame >= frame_minima
        support_count = int(np.sum(per_frame))
        if int(np.sum(persistent)) < 2 or support_count < total_minimum:
            continue
        center = (float(bin_ids[index]) + 0.5) * height_bin_m
        near = np.abs(all_values - center) <= 0.10
        refined = float(np.median(all_values[near]))
        residual = float(np.median(np.abs(all_values[near] - refined)))
        candidates.append(
            {
                "world_height_m": refined,
                "median_absolute_residual_m": residual,
                "persistent_frame_count": int(np.sum(persistent)),
                "support_sample_count": support_count,
                "support_fraction": support_count / len(all_values),
            }
        )
    return sorted(candidates, key=lambda value: float(value["world_height_m"]))


def support_identity(
    frames: list[SelectedFrame],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    world_up, gravity_receipt = parent_world_up(frames)
    if world_up is None:
        return None, {"gravity": gravity_receipt, "status": "UNKNOWN_GRAVITY"}
    heights = [horizontal_world_heights(frame, world_up) for frame in frames]
    modes = persistent_height_modes(heights)
    if not modes:
        return None, {
            "gravity": gravity_receipt,
            "horizontal_sample_counts": [len(value) for value in heights],
            "status": "UNKNOWN_NO_PERSISTENT_SUPPORT_HEIGHT",
        }
    lowest = modes[0]
    identity = {
        "world_up_unit": world_up,
        "support_world_height_m": float(lowest["world_height_m"]),
        "median_absolute_residual_m": max(float(lowest["median_absolute_residual_m"]), 0.01),
    }
    return identity, {
        "gravity": gravity_receipt,
        "horizontal_sample_counts": [len(value) for value in heights],
        "persistent_modes": modes,
        "selected_lowest_mode": lowest,
        "status": "SOURCE_SEQUENCE_SUPPORT_IDENTITY_VALID",
    }


def orient_frame(
    frame: SelectedFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    rgb = frame.rgb_u8_hwc
    depth = frame.depth_m_hw
    valid = frame.depth_valid_hw
    intrinsics = frame.intrinsics.copy()
    pose = frame.camera_to_world.copy()
    gravity = None if frame.gravity_up_camera is None else frame.gravity_up_camera.copy()
    if frame.orientation == "LANDSCAPE_IDENTITY":
        return rgb, depth, valid, intrinsics, pose, gravity
    require(frame.orientation == "PORTRAIT_ROT90_CLOCKWISE", "unsupported orientation")
    old_height = depth.shape[0]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    output_intrinsics = np.asarray(
        [[fy, 0.0, old_height - 1.0 - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    new_from_old = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    output_pose = pose.copy()
    output_pose[:3, :3] = pose[:3, :3] @ new_from_old.T
    output_gravity = None if gravity is None else new_from_old @ gravity
    return (
        np.rot90(rgb, k=-1).copy(),
        np.rot90(depth, k=-1).copy(),
        np.rot90(valid, k=-1).copy(),
        output_intrinsics,
        output_pose,
        output_gravity,
    )


def verify_orientation_projection(
    source_intrinsics: np.ndarray,
    output_intrinsics: np.ndarray,
    orientation: str,
) -> bool:
    if orientation == "LANDSCAPE_IDENTITY":
        return bool(np.allclose(source_intrinsics, output_intrinsics, atol=1e-9))
    source_points = np.asarray(
        [[-0.3, -0.2, 1.0], [0.1, 0.25, 2.0], [0.7, -0.4, 3.0]],
        dtype=np.float64,
    )
    new_from_old = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    source_pixels = (source_intrinsics @ source_points.T).T
    source_pixels = source_pixels[:, :2] / source_pixels[:, 2:]
    output_points = source_points @ new_from_old.T
    output_pixels = (output_intrinsics @ output_points.T).T
    output_pixels = output_pixels[:, :2] / output_pixels[:, 2:]
    expected = np.stack((479.0 - source_pixels[:, 1], source_pixels[:, 0]), axis=1)
    return bool(np.allclose(output_pixels, expected, atol=1e-6))


def unknown_factors(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    zeros = np.zeros(shape, dtype=np.float32)
    false = np.zeros(shape, dtype=np.bool_)
    nan = np.full(shape, np.nan, dtype=np.float32)
    return {
        "dense_normal_diagnostic_camera_xyz_hwc": np.zeros((*shape, 3), dtype=np.float32),
        "normal_valid_diagnostic_hw": false.copy(),
        "support_truth_hw": zeros.copy(),
        "support_truth_valid_hw": false.copy(),
        "support_plane_normal_camera_xyz": np.zeros(3, dtype=np.float32),
        "camera_height_m": np.asarray(np.nan, dtype=np.float32),
        "support_plane_valid": np.asarray(False, dtype=np.bool_),
        "support_signed_plane_residual_m_hw": nan.copy(),
        "support_plane_fit_residual_diagnostic_m": np.asarray(np.nan, dtype=np.float32),
        "obstacle_evidence_truth_hw": zeros.copy(),
        "boundary_probability_diagnostic_hw": zeros.copy(),
        "boundary_distance_px_hw": nan.copy(),
        "boundary_seed_diagnostic_hw": false.copy(),
        "evidence_truth_valid_hw": false.copy(),
        "support_provenance_hw": np.zeros(shape, dtype=np.uint8),
        "support_plane_provenance_code": np.asarray(UNKNOWN_PROVENANCE, dtype=np.uint8),
        "evidence_provenance_hw": np.zeros(shape, dtype=np.uint8),
    }


def source_geometric_factors(
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    gravity_up_camera: np.ndarray | None,
    identity: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    shape = tuple(int(value) for value in depth_m.shape)
    if gravity_up_camera is None or identity is None:
        return unknown_factors(shape)
    up = np.asarray(gravity_up_camera, dtype=np.float64)
    up /= np.linalg.norm(up)
    world_up = np.asarray(identity["world_up_unit"], dtype=np.float64)
    expected_up = camera_to_world[:3, :3].T @ world_up
    expected_up /= np.linalg.norm(expected_up)
    if float(np.dot(up, expected_up)) < math.cos(math.radians(15.0)):
        return unknown_factors(shape)
    camera_world_height = float(np.dot(camera_to_world[:3, 3], world_up))
    camera_height = camera_world_height - float(identity["support_world_height_m"])
    if not 0.45 <= camera_height <= 2.20:
        return unknown_factors(shape)

    points = backproject_depth_grid(depth_m, intrinsics)
    normals, normal_valid = compute_dense_normals(depth_m, depth_valid, intrinsics)
    heights = (
        np.einsum("...i,i->...", points, up) + camera_height
    ).astype(np.float32)
    residual = max(float(identity["median_absolute_residual_m"]), 0.01)
    height_sigma = max(0.06, residual + 0.04)
    height_score = np.exp(-0.5 * np.square(heights / height_sigma))
    normal_alignment = np.abs(np.einsum("...i,i->...", normals, up))
    cosine_limit = math.cos(math.radians(25.0))
    normal_score = np.clip(
        (normal_alignment - cosine_limit) / (1.0 - cosine_limit),
        0.0,
        1.0,
    )
    preliminary_support_valid = depth_valid & normal_valid
    support = np.zeros(shape, dtype=np.float32)
    support[preliminary_support_valid] = (
        height_score[preliminary_support_valid] * normal_score[preliminary_support_valid]
    ).astype(np.float32)

    lower = 1.0 / (1.0 + np.exp(-(heights - 0.08) / 0.04))
    upper = 1.0 / (1.0 + np.exp((heights - 2.00) / 0.15))
    obstacle = np.zeros(shape, dtype=np.float32)
    obstacle[depth_valid] = (
        lower[depth_valid] * upper[depth_valid] * (1.0 - support[depth_valid])
    ).astype(np.float32)

    point_plane_edge, normal_edge, neighbor_count = _pairwise_point_to_plane_edges(
        points,
        normals,
        normal_valid,
        depth_valid,
    )
    support_edge, _ = _pairwise_scalar_edges(
        support,
        preliminary_support_valid,
        np.full(shape, 0.20, dtype=np.float32),
    )
    point_plane_strength = np.clip((point_plane_edge - 0.15) / 0.30, 0.0, 1.0)
    support_transition_strength = np.clip((support_edge - 0.20) / 0.40, 0.0, 1.0)
    corroboration = np.maximum(normal_edge, support_transition_strength)
    boundary_probability = np.clip(
        point_plane_strength * (1.0 + 0.25 * corroboration),
        0.0,
        1.0,
    ).astype(np.float32)
    physical_boundary_valid = depth_valid & (neighbor_count > 0)
    boundary_probability[~physical_boundary_valid] = 0.0
    support_valid = preliminary_support_valid & (boundary_probability < 0.50)
    support[~support_valid] = 0.0
    evidence_valid = physical_boundary_valid & depth_valid
    obstacle[~evidence_valid] = 0.0
    boundary_probability[~evidence_valid] = 0.0
    boundary_seed = evidence_valid & (boundary_probability >= 0.50)
    if np.any(boundary_seed):
        boundary_distance = np.minimum(distance_transform_edt(~boundary_seed), 32.0).astype(np.float32)
    else:
        boundary_distance = np.full(shape, 32.0, dtype=np.float32)
    boundary_distance[~evidence_valid] = np.nan
    signed_residual = heights.copy()
    signed_residual[~support_valid] = np.nan
    return {
        "dense_normal_diagnostic_camera_xyz_hwc": normals.astype(np.float32),
        "normal_valid_diagnostic_hw": normal_valid.astype(np.bool_),
        "support_truth_hw": support,
        "support_truth_valid_hw": support_valid.astype(np.bool_),
        "support_plane_normal_camera_xyz": up.astype(np.float32),
        "camera_height_m": np.asarray(camera_height, dtype=np.float32),
        "support_plane_valid": np.asarray(True, dtype=np.bool_),
        "support_signed_plane_residual_m_hw": signed_residual,
        "support_plane_fit_residual_diagnostic_m": np.asarray(residual, dtype=np.float32),
        "obstacle_evidence_truth_hw": obstacle,
        "boundary_probability_diagnostic_hw": boundary_probability,
        "boundary_distance_px_hw": boundary_distance,
        "boundary_seed_diagnostic_hw": boundary_seed.astype(np.bool_),
        "evidence_truth_valid_hw": evidence_valid.astype(np.bool_),
        "support_provenance_hw": np.where(support_valid, SUPPORT_PROVENANCE, UNKNOWN_PROVENANCE).astype(np.uint8),
        "support_plane_provenance_code": np.asarray(SUPPORT_PROVENANCE, dtype=np.uint8),
        "evidence_provenance_hw": np.where(evidence_valid, BOUNDARY_PROVENANCE, UNKNOWN_PROVENANCE).astype(np.uint8),
    }


def build_payload(
    frame: SelectedFrame,
    identity: dict[str, Any] | None,
    identity_receipt_sha256: str,
    contract_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rgb, depth, valid, intrinsics, pose, gravity = orient_frame(frame)
    require(verify_orientation_projection(frame.intrinsics, intrinsics, frame.orientation), "orientation projection invariant failed")
    factors = source_geometric_factors(
        depth,
        valid,
        intrinsics,
        pose,
        gravity,
        identity,
    )
    camera_receipt = {
        "sample_id": frame.frame_id,
        "orientation": frame.orientation,
        "intrinsics": intrinsics.tolist(),
        "camera_to_world": pose.tolist(),
        "rgb_timestamp": frame.rgb.timestamp_seconds,
        "depth_timestamp": frame.depth.timestamp_seconds,
        "pose_bracketing_gap_seconds": frame.pose_bracketing_gap_seconds,
    }
    camera_receipt_sha256 = sha256_json(camera_receipt)
    metric_provenance = np.where(valid, METRIC_PROVENANCE, UNKNOWN_PROVENANCE).astype(np.uint8)
    payload: dict[str, np.ndarray] = {
        "sample_id": np.asarray(frame.frame_id),
        "parent_id": np.asarray(frame.parent_id),
        "role": np.asarray(frame.role),
        "orientation": np.asarray(frame.orientation),
        "rgb_u8_hwc": rgb.astype(np.uint8),
        "metric_depth_m_hw": depth.astype(np.float32),
        "metric_depth_valid_hw": valid.astype(np.bool_),
        "metric_depth_provenance_hw": metric_provenance,
        "intrinsics_output": intrinsics.astype(np.float64),
        "camera_to_world_output": pose.astype(np.float64),
        "gravity_up_camera_xyz": (
            np.asarray(gravity, dtype=np.float32)
            if gravity is not None
            else np.full(3, np.nan, dtype=np.float32)
        ),
        "camera_geometry_receipt_sha256": np.asarray(camera_receipt_sha256),
        "support_identity_receipt_sha256": np.asarray(identity_receipt_sha256),
        "label_transform_contract_sha256": np.asarray(contract_sha256),
        **factors,
    }
    report = {
        "shape_hw": list(depth.shape),
        "metric_depth_valid_pixels": int(np.sum(valid)),
        "support_plane_valid": bool(factors["support_plane_valid"]),
        "support_valid_pixels": int(np.sum(factors["support_truth_valid_hw"])),
        "support_positive_pixels_ge_0_5": int(
            np.sum(factors["support_truth_valid_hw"] & (factors["support_truth_hw"] >= 0.5))
        ),
        "evidence_valid_pixels": int(np.sum(factors["evidence_truth_valid_hw"])),
        "boundary_seed_pixels": int(np.sum(factors["boundary_seed_diagnostic_hw"])),
        "camera_geometry_receipt_sha256": camera_receipt_sha256,
    }
    return payload, report


def arrays_equal(first: np.ndarray, second: np.ndarray) -> bool:
    if first.dtype != second.dtype or first.shape != second.shape:
        return False
    if first.dtype.kind in {"f", "c"}:
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def run(contract_path: Path, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    validation = validate_path(contract_path)
    require(validation["passed"], "supervision contract static validation failed")
    require(source_dir.is_dir(), f"source directory missing: {source_dir}")
    require(not output_dir.exists(), f"output directory exists: {output_dir}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_sha = sha256_file(contract_path)
    source = contract["source_contract"]
    cohort = contract["cohort_contract"]
    token = str(cohort["assignment_token"])

    all_frames: list[SelectedFrame] = []
    source_receipts: list[dict[str, Any]] = []
    parent_rows: dict[str, dict[str, Any]] = {}
    for row in cohort["parents"]:
        frames, receipt = load_parent_frames(row, source_dir, source, token)
        require(len(frames) == 3, f"selected frame count drift: {row['parent_id']}")
        all_frames.extend(frames)
        source_receipts.append(receipt)
        parent_rows[str(row["parent_id"])] = row
    require(len(all_frames) == 39, "total selected frame count drift")

    identities: dict[str, dict[str, Any] | None] = {}
    identity_receipts: dict[str, dict[str, Any]] = {}
    identity_shas: dict[str, str] = {}
    for parent_id in sorted(parent_rows):
        frames = [frame for frame in all_frames if frame.parent_id == parent_id]
        identity, receipt = support_identity(frames)
        identities[parent_id] = identity
        identity_receipts[parent_id] = receipt
        identity_shas[parent_id] = sha256_json(receipt)

    output_dir.mkdir(parents=True, exist_ok=False)
    frame_receipts: list[dict[str, Any]] = []
    totals_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    totals_by_parent: dict[str, Counter[str]] = defaultdict(Counter)
    provenance_exact = True
    supervision_fields_complete = True
    unknown_fail_closed = True
    task_firewall_exact = True
    for frame in all_frames:
        payload, report = build_payload(
            frame,
            identities[frame.parent_id],
            identity_shas[frame.parent_id],
            contract_sha,
        )
        provenance_exact &= set(np.unique(payload["metric_depth_provenance_hw"]).tolist()).issubset({0, METRIC_PROVENANCE})
        provenance_exact &= set(np.unique(payload["support_provenance_hw"]).tolist()).issubset({0, SUPPORT_PROVENANCE})
        provenance_exact &= set(np.unique(payload["evidence_provenance_hw"]).tolist()).issubset({0, BOUNDARY_PROVENANCE})
        supervision_fields_complete &= REQUIRED_F1_SUPERVISION_FIELDS.issubset(payload)
        task_firewall_exact &= not any(
            token in key.lower()
            for key in payload
            for token in FORBIDDEN_TASK_FIELD_TOKENS
        )
        metric_valid = payload["metric_depth_valid_hw"].astype(np.bool_)
        support_valid = payload["support_truth_valid_hw"].astype(np.bool_)
        evidence_valid = payload["evidence_truth_valid_hw"].astype(np.bool_)
        unknown_fail_closed &= bool(np.all(payload["metric_depth_m_hw"][~metric_valid] == 0.0))
        unknown_fail_closed &= bool(np.all(payload["support_truth_hw"][~support_valid] == 0.0))
        unknown_fail_closed &= bool(
            np.all(payload["obstacle_evidence_truth_hw"][~evidence_valid] == 0.0)
        )
        unknown_fail_closed &= bool(
            np.all(np.isnan(payload["boundary_distance_px_hw"][~evidence_valid]))
        )
        unknown_fail_closed &= bool(
            np.all(payload["metric_depth_provenance_hw"][~metric_valid] == UNKNOWN_PROVENANCE)
            and np.all(payload["support_provenance_hw"][~support_valid] == UNKNOWN_PROVENANCE)
            and np.all(payload["evidence_provenance_hw"][~evidence_valid] == UNKNOWN_PROVENANCE)
        )
        output_path = output_dir / f"{frame.frame_id}.npz"
        np.savez_compressed(output_path, **payload)
        with np.load(output_path, allow_pickle=False) as written:
            require(set(written.files) == set(payload), "output field set drift")
            require(
                all(arrays_equal(np.asarray(written[key]), value) for key, value in payload.items()),
                "output array roundtrip drift",
            )
        receipt = {
            "sample_id": frame.frame_id,
            "parent_id": frame.parent_id,
            "role": frame.role,
            "orientation": frame.orientation,
            "rgb_timestamp": frame.rgb.timestamp_seconds,
            "depth_timestamp": frame.depth.timestamp_seconds,
            "association_delta_seconds": abs(frame.rgb.timestamp_seconds - frame.depth.timestamp_seconds),
            "pose_bracketing_gap_seconds": frame.pose_bracketing_gap_seconds,
            "accelerometer_sample_count": frame.accelerometer_sample_count,
            "accelerometer_norm_mps2": frame.accelerometer_norm_mps2,
            "source_archive_sha256": frame.source_archive_sha256,
            "rgb_member_sha256": frame.rgb_member_sha256,
            "depth_member_sha256": frame.depth_member_sha256,
            "metadata_member_sha256": frame.metadata_member_sha256,
            "support_identity_receipt_sha256": identity_shas[frame.parent_id],
            "label_transform_contract_sha256": contract_sha,
            "output": str(output_path.resolve()),
            "output_bytes": output_path.stat().st_size,
            "output_sha256": sha256_file(output_path),
            "field_count": len(payload),
            **report,
        }
        frame_receipts.append(receipt)
        for key in (
            "metric_depth_valid_pixels",
            "support_valid_pixels",
            "support_positive_pixels_ge_0_5",
            "evidence_valid_pixels",
            "boundary_seed_pixels",
        ):
            totals_by_role[frame.role][key] += int(report[key])
            totals_by_parent[frame.parent_id][key] += int(report[key])
        totals_by_parent[frame.parent_id]["support_plane_valid_frames"] += int(report["support_plane_valid"])

    joint_parents = sorted(
        parent_id
        for parent_id, counts in totals_by_parent.items()
        if counts["metric_depth_valid_pixels"] > 0
        and counts["support_plane_valid_frames"] > 0
        and counts["support_valid_pixels"] > 0
        and counts["support_positive_pixels_ge_0_5"] > 0
        and counts["evidence_valid_pixels"] > 0
        and counts["boundary_seed_pixels"] > 0
    )
    joint_by_role = Counter(str(parent_rows[parent]["role"]) for parent in joint_parents)
    held_parent_ids = {
        role: [
            str(row["parent_id"])
            for row in cohort["parents"]
            if row["role"] == role
        ]
        for role in ("CHECKPOINT_SELECTION", "TRAIN_CANARY")
    }
    source_archives_exact = (
        len(source_receipts) == 13
        and sum(int(row["source_archive_bytes"]) for row in source_receipts)
        == int(cohort["expected_download_bytes"])
        and all(len(str(row["source_archive_sha256"])) == 64 for row in source_receipts)
    )
    role_counts = Counter(frame.role for frame in all_frames)
    gates = {
        "F1_S01_BINDINGS_EXACT": bool(validation["gates"]["bindings_exact"]),
        "F1_S02_SOURCE_ARCHIVES_EXACT": source_archives_exact,
        "F1_S03_ROLE_AND_ORIENTATION_EXACT": (
            role_counts == {"FIT": 27, "CHECKPOINT_SELECTION": 6, "TRAIN_CANARY": 6}
            and validation["gates"]["held_orientations_complete"]
        ),
        "F1_S04_FRAME_SELECTION_EXACT": len(frame_receipts) == 39,
        "F1_S05_SCHEMA_AND_PROVENANCE_COMPLETE": (
            all(row["field_count"] == frame_receipts[0]["field_count"] for row in frame_receipts)
            and all(len(row["camera_geometry_receipt_sha256"]) == 64 for row in frame_receipts)
            and provenance_exact
            and supervision_fields_complete
        ),
        "F1_S06_JOINT_PARENT_COVERAGE": (
            len(joint_parents) >= 12
            and joint_by_role["FIT"] >= 8
            and set(held_parent_ids["CHECKPOINT_SELECTION"]).issubset(joint_parents)
            and set(held_parent_ids["TRAIN_CANARY"]).issubset(joint_parents)
        ),
        "F1_S07_UNKNOWN_FAIL_CLOSED": unknown_fail_closed and all(
            row["support_plane_valid"]
            or (row["support_valid_pixels"] == 0 and row["evidence_valid_pixels"] == 0)
            for row in frame_receipts
        ),
        "F1_S08_UNCERTAINTY_RESIDUAL_ONLY": bool(
            validation["gates"]["uncertainty_residual_only"]
        ),
        "F1_S09_TASK_FIREWALL": bool(
            validation["gates"]["source_only_labels"] and task_firewall_exact
        ),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_source_native_label_frontdoor_result_v1",
        "status": (
            "F1_SUPERVISION_FRONTDOOR_SATISFIED_MODEL_AND_OPTIMIZER_STILL_REQUIRE_SEPARATE_LOCK"
            if passed
            else "F1_SUPERVISION_FRONTDOOR_UNSATISFIED_NO_OPTIMIZER"
        ),
        "contract": str(contract_path.resolve()),
        "contract_sha256": contract_sha,
        "contract_validation": validation,
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "source_parent_count": len(source_receipts),
        "frame_count": len(frame_receipts),
        "role_frame_counts": dict(role_counts),
        "joint_factor_parent_count": len(joint_parents),
        "joint_factor_parents": joint_parents,
        "joint_factor_parent_counts_by_role": dict(joint_by_role),
        "source_receipts": source_receipts,
        "support_identity_receipts": identity_receipts,
        "coverage_by_role": {role: dict(values) for role, values in totals_by_role.items()},
        "coverage_by_parent": {parent: dict(values) for parent, values in totals_by_parent.items()},
        "gates": gates,
        "passed": passed,
        "frames": frame_receipts,
        "decision": {
            "formal_source_only_labels_materialized": True,
            "teacher_pixels_admitted": False,
            "uncertainty_proxy_targets_admitted": False,
            "f1_model_or_optimizer_authorized": False,
            "next_action_if_pass": (
                "Freeze model, nonlearned baselines, loss normalization, checkpoint schedule, seeds and optimizer budget in a separate execution lock."
                if passed
                else None
            ),
        },
        "claim_boundary": "Source-native F1 label materialization and pre-optimizer supervision frontdoor only; not factor learnability, task utility, reducer evidence, deployment, product or safety evidence.",
    }
    result_path = output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.contract.resolve(), args.source_dir.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "source_parent_count",
                    "frame_count",
                    "joint_factor_parent_count",
                    "joint_factor_parent_counts_by_role",
                    "gates",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
