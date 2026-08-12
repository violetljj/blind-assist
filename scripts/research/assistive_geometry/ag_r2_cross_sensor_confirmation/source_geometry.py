"""Frozen ETH3D source-native geometry recipe for F2.

Only calibration identities may construct a session support identity.  Score
identities are then materialized against that immutable identity after the
conditioned factor completion has been sealed by the runner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from .contract import require
from .model_only import scale_intrinsics

MIN_CALIBRATION_GRAVITY_FRAMES = 8
MIN_PERSISTENT_SUPPORT_FRAMES = 8
HEIGHT_BIN_M = 0.04
MODE_RADIUS_BINS = 2
MIN_HORIZONTAL_POINTS_PER_FRAME = 96
MIN_HORIZONTAL_POINTS_TOTAL = 1536


def _validate_source_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    require(
        set(frame) == {
            "parent_id", "frame_id", "depth_m", "depth_known", "intrinsics",
            "camera_to_world", "gravity_up_camera_xyz",
        },
        "F2_SOURCE_FRAME_KEY_SET",
    )
    depth = np.asarray(frame["depth_m"])
    known = np.asarray(frame["depth_known"])
    intrinsics = np.asarray(frame["intrinsics"])
    transform = np.asarray(frame["camera_to_world"])
    gravity = np.asarray(frame["gravity_up_camera_xyz"])
    require(depth.dtype == np.dtype("float64") and depth.ndim == 2, "F2_SOURCE_DEPTH_SCHEMA")
    require(known.dtype == np.dtype("bool") and known.shape == depth.shape, "F2_SOURCE_DEPTH_KNOWN_SCHEMA")
    require(bool(np.all(np.isfinite(depth[known]))) and bool(np.all(depth[known] > 0.0)), "F2_SOURCE_DEPTH_KNOWN_INVALID")
    require(bool(np.all(np.isnan(depth[~known]))), "F2_SOURCE_DEPTH_UNKNOWN_NOT_NAN")
    require(intrinsics.dtype == np.dtype("float64") and intrinsics.shape == (3, 3), "F2_SOURCE_K_SCHEMA")
    require(transform.dtype == np.dtype("float64") and transform.shape == (4, 4), "F2_SOURCE_POSE_SCHEMA")
    require(gravity.dtype == np.dtype("float64") and gravity.shape == (3,), "F2_SOURCE_GRAVITY_SCHEMA")
    require(
        bool(np.all(np.isfinite(intrinsics))) and bool(np.all(np.isfinite(transform)))
        and bool(np.all(np.isfinite(gravity))),
        "F2_SOURCE_NONFINITE_GEOMETRY",
    )
    require(intrinsics[0, 0] > 0.0 and intrinsics[1, 1] > 0.0, "F2_SOURCE_K_INVALID")
    require(bool(np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)), "F2_SOURCE_POSE_BOTTOM_ROW")
    rotation = transform[:3, :3]
    require(
        bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5))
        and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-5,
        "F2_SOURCE_POSE_ROTATION",
    )
    norm = float(np.linalg.norm(gravity))
    require(norm > 0.0, "F2_SOURCE_GRAVITY_ZERO")
    return {
        "parent_id": str(frame["parent_id"]),
        "frame_id": str(frame["frame_id"]),
        "depth_m": depth,
        "depth_known": known,
        "intrinsics": intrinsics,
        "camera_to_world": transform,
        "gravity_up_camera_xyz": gravity / norm,
    }


def backproject(depth_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    rows, columns = np.indices(depth_m.shape, dtype=np.float64)
    x = (columns - intrinsics[0, 2]) * depth_m / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) * depth_m / intrinsics[1, 1]
    return np.stack((x, y, depth_m), axis=-1)


def dense_normals(
    depth_m: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = backproject(np.nan_to_num(depth_m, nan=0.0), intrinsics)
    horizontal = np.zeros_like(points)
    vertical = np.zeros_like(points)
    horizontal[:, 1:-1] = points[:, 2:] - points[:, :-2]
    vertical[1:-1] = points[2:] - points[:-2]
    neighborhood = np.zeros(valid.shape, dtype=bool)
    neighborhood[1:-1, 1:-1] = (
        valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
        & valid[:-2, 1:-1] & valid[2:, 1:-1]
    )
    normals = np.cross(horizontal, vertical)
    norm = np.linalg.norm(normals, axis=-1)
    normal_valid = neighborhood & np.isfinite(norm) & (norm > 1.0e-8)
    normals[normal_valid] /= norm[normal_valid, None]
    normals[~normal_valid] = 0.0
    return normals, normal_valid


def _parent_world_up(frames: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    vectors = []
    camera_vectors = []
    for raw in frames:
        frame = _validate_source_frame(raw)
        camera = frame["gravity_up_camera_xyz"]
        world = frame["camera_to_world"][:3, :3] @ camera
        vectors.append(world / np.linalg.norm(world))
        camera_vectors.append(camera)
    require(len(vectors) == 12, "F2_CALIBRATION_FRAME_COUNT")
    values = np.stack(vectors)
    mean = np.sum(values, axis=0)
    require(float(np.linalg.norm(mean)) > 1.0e-9, "F2_CALIBRATION_GRAVITY_CANCELLED")
    mean /= np.linalg.norm(mean)
    angles = np.degrees(np.arccos(np.clip(values @ mean, -1.0, 1.0)))
    require(int(np.sum(np.isfinite(angles))) >= MIN_CALIBRATION_GRAVITY_FRAMES, "F2_CALIBRATION_GRAVITY_INSUFFICIENT")
    require(float(np.max(angles)) <= 15.0, "F2_CALIBRATION_GRAVITY_INCONSISTENT")
    camera_mean = np.sum(np.stack(camera_vectors), axis=0)
    require(float(np.linalg.norm(camera_mean)) > 1.0e-9, "F2_CALIBRATION_CAMERA_GRAVITY_CANCELLED")
    camera_mean /= np.linalg.norm(camera_mean)
    return mean, {
        "world_up_unit": mean.tolist(),
        "gravity_up_camera_xyz": camera_mean.tolist(),
        "maximum_world_angle_deg": float(np.max(angles)),
        "valid_frame_count": len(vectors),
    }


def _horizontal_world_heights(frame: Mapping[str, Any], world_up: np.ndarray) -> np.ndarray:
    value = _validate_source_frame(frame)
    depth = np.nan_to_num(value["depth_m"], nan=0.0)
    normals_camera, normal_valid = dense_normals(depth, value["depth_known"], value["intrinsics"])
    points_camera = backproject(depth, value["intrinsics"])
    rotation = value["camera_to_world"][:3, :3]
    translation = value["camera_to_world"][:3, 3]
    normals_world = np.einsum("...j,ij->...i", normals_camera, rotation)
    points_world = np.einsum("...j,ij->...i", points_camera, rotation) + translation
    heights = np.einsum("...i,i->...", points_world, world_up)
    horizontal = (
        value["depth_known"] & normal_valid & np.isfinite(heights) & (depth <= 5.0)
        & (np.abs(np.einsum("...i,i->...", normals_world, world_up)) >= math.cos(math.radians(20.0)))
    )
    return heights[horizontal].astype(np.float64)[::2]


def _persistent_height_modes(frame_heights: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    require(len(frame_heights) == 12, "F2_SUPPORT_CALIBRATION_COUNT")
    require(all(value.ndim == 1 for value in frame_heights), "F2_SUPPORT_HEIGHT_RANK")
    usable = [value for value in frame_heights if value.size > 0]
    require(len(usable) >= MIN_PERSISTENT_SUPPORT_FRAMES, "F2_SUPPORT_USABLE_FRAMES")
    all_values = np.concatenate(usable)
    require(all_values.size >= MIN_HORIZONTAL_POINTS_TOTAL and bool(np.all(np.isfinite(all_values))), "F2_SUPPORT_TOTAL_POINTS")
    low, high = np.quantile(all_values, (0.002, 0.998))
    first_bin = math.floor(float(low) / HEIGHT_BIN_M) - 1
    last_bin = math.ceil(float(high) / HEIGHT_BIN_M) + 1
    bin_ids = np.arange(first_bin, last_bin + 1, dtype=np.int64)
    counts = []
    for values in usable:
        ids = np.floor(values / HEIGHT_BIN_M).astype(np.int64)
        row = np.zeros(len(bin_ids), dtype=np.int64)
        inside = (ids >= first_bin) & (ids <= last_bin)
        np.add.at(row, ids[inside] - first_bin, 1)
        counts.append(row)
    stacked = np.stack(counts)
    total = np.sum(stacked, axis=0)
    smooth = np.convolve(total.astype(np.float64), [0.25, 0.5, 0.25], mode="same")
    peaks = [index for index in range(1, len(bin_ids) - 1) if smooth[index] >= smooth[index - 1] and smooth[index] > smooth[index + 1]]
    peaks.sort(key=lambda index: (-smooth[index], int(bin_ids[index])))
    selected: list[int] = []
    for index in peaks:
        if not any(abs(index - prior) <= MODE_RADIUS_BINS for prior in selected):
            selected.append(index)
    candidates = []
    for index in selected:
        left = max(0, index - MODE_RADIUS_BINS)
        right = min(len(bin_ids), index + MODE_RADIUS_BINS + 1)
        per_frame = np.sum(stacked[:, left:right], axis=1)
        minima = np.asarray(
            [max(MIN_HORIZONTAL_POINTS_PER_FRAME, math.ceil(0.002 * len(values))) for values in usable],
            dtype=np.int64,
        )
        persistent = per_frame >= minima
        count = int(np.sum(per_frame))
        if int(np.sum(persistent)) < MIN_PERSISTENT_SUPPORT_FRAMES or count < MIN_HORIZONTAL_POINTS_TOTAL:
            continue
        center = (float(bin_ids[index]) + 0.5) * HEIGHT_BIN_M
        near = np.abs(all_values - center) <= 0.10
        refined = float(np.median(all_values[near]))
        residual = float(np.median(np.abs(all_values[near] - refined)))
        candidates.append({
            "world_height_m": refined,
            "median_absolute_residual_m": residual,
            "persistent_frame_count": int(np.sum(persistent)),
            "support_sample_count": count,
        })
    return sorted(candidates, key=lambda row: row["world_height_m"])


def derive_session_context(
    parent_id: str,
    calibration_frames: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(len(calibration_frames) == 12, "F2_SESSION_CALIBRATION_COUNT")
    validated = [_validate_source_frame(frame) for frame in calibration_frames]
    require(all(frame["parent_id"] == parent_id for frame in validated), "F2_SESSION_PARENT_MISMATCH")
    require(len({frame["frame_id"] for frame in validated}) == 12, "F2_SESSION_FRAME_DUPLICATE")
    world_up, gravity_receipt = _parent_world_up(validated)
    heights = [_horizontal_world_heights(frame, world_up) for frame in validated]
    modes = _persistent_height_modes(heights)
    require(bool(modes), "F2_SESSION_SUPPORT_MODE_MISSING")
    selected = modes[0]
    camera_heights = np.asarray(
        [float(np.dot(frame["camera_to_world"][:3, 3], world_up) - selected["world_height_m"]) for frame in validated],
        dtype=np.float64,
    )
    require(bool(np.all(np.isfinite(camera_heights))), "F2_SESSION_HEIGHT_NONFINITE")
    median = float(np.median(camera_heights))
    mad = float(np.median(np.abs(camera_heights - median)))
    identity = {
        "world_up_unit": world_up,
        "support_world_height_m": float(selected["world_height_m"]),
        "median_absolute_residual_m": max(float(selected["median_absolute_residual_m"]), 0.01),
    }
    context = {
        "parent_id": parent_id,
        "camera_height_m": median,
        "camera_height_mad_m": mad,
        "gravity_up_camera_xyz": gravity_receipt["gravity_up_camera_xyz"],
    }
    receipt = {
        "parent_id": parent_id,
        "identity": {
            "world_up_unit": world_up.tolist(),
            "support_world_height_m": identity["support_world_height_m"],
            "median_absolute_residual_m": identity["median_absolute_residual_m"],
        },
        "context": context,
        "gravity": gravity_receipt,
        "horizontal_sample_counts": [int(value.size) for value in heights],
        "persistent_modes": modes,
        "selected_lowest_mode": selected,
        "calibration_camera_heights_m": camera_heights.tolist(),
        "persistent_frame_minimum": MIN_PERSISTENT_SUPPORT_FRAMES,
        "total_point_minimum": MIN_HORIZONTAL_POINTS_TOTAL,
    }
    return context, receipt


def _pairwise_scalar_edge(value: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.zeros(value.shape, dtype=np.float64)
    count = np.zeros(value.shape, dtype=np.int16)
    for row_shift, column_shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        source_rows = slice(max(0, -row_shift), value.shape[0] - max(0, row_shift))
        source_columns = slice(max(0, -column_shift), value.shape[1] - max(0, column_shift))
        target_rows = slice(max(0, row_shift), value.shape[0] - max(0, -row_shift))
        target_columns = slice(max(0, column_shift), value.shape[1] - max(0, -column_shift))
        pair_valid = valid[source_rows, source_columns] & valid[target_rows, target_columns]
        delta = np.abs(value[source_rows, source_columns] - value[target_rows, target_columns])
        for rows, columns in ((source_rows, source_columns), (target_rows, target_columns)):
            view = maximum[rows, columns]
            np.maximum(view, np.where(pair_valid, delta, 0.0), out=view)
            count[rows, columns] += pair_valid.astype(np.int16)
    return maximum, count


def _point_plane_edges(
    points: np.ndarray,
    normals: np.ndarray,
    normal_valid: np.ndarray,
    depth_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_edge = np.zeros(depth_valid.shape, dtype=np.float64)
    normal_edge = np.zeros(depth_valid.shape, dtype=np.float64)
    count = np.zeros(depth_valid.shape, dtype=np.int16)
    for row_shift, column_shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        sr = slice(max(0, -row_shift), depth_valid.shape[0] - max(0, row_shift))
        sc = slice(max(0, -column_shift), depth_valid.shape[1] - max(0, column_shift))
        tr = slice(max(0, row_shift), depth_valid.shape[0] - max(0, -row_shift))
        tc = slice(max(0, column_shift), depth_valid.shape[1] - max(0, -column_shift))
        pair = depth_valid[sr, sc] & depth_valid[tr, tc] & normal_valid[sr, sc] & normal_valid[tr, tc]
        delta = points[tr, tc] - points[sr, sc]
        plane = np.maximum(
            np.abs(np.einsum("...i,...i->...", delta, normals[sr, sc])),
            np.abs(np.einsum("...i,...i->...", delta, normals[tr, tc])),
        )
        angle = 1.0 - np.abs(np.einsum("...i,...i->...", normals[sr, sc], normals[tr, tc]))
        for rows, columns in ((sr, sc), (tr, tc)):
            np.maximum(point_edge[rows, columns], np.where(pair, plane, 0.0), out=point_edge[rows, columns])
            np.maximum(normal_edge[rows, columns], np.where(pair, angle, 0.0), out=normal_edge[rows, columns])
            count[rows, columns] += pair.astype(np.int16)
    return point_edge, normal_edge, count


def _nearest_resize(value: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    source_height, source_width = value.shape[:2]
    output_height, output_width = output_hw
    rows = np.minimum(((np.arange(output_height) + 0.5) * source_height / output_height).astype(int), source_height - 1)
    columns = np.minimum(((np.arange(output_width) + 0.5) * source_width / output_width).astype(int), source_width - 1)
    return value[np.ix_(rows, columns)]


def materialize_score_truth(
    score_frame: Mapping[str, Any],
    identity_receipt: Mapping[str, Any],
    output_hw: tuple[int, int],
) -> dict[str, Any]:
    frame = _validate_source_frame(score_frame)
    identity = identity_receipt["identity"]
    world_up = np.asarray(identity["world_up_unit"], dtype=np.float64)
    expected_up = frame["camera_to_world"][:3, :3].T @ world_up
    expected_up /= np.linalg.norm(expected_up)
    require(
        float(np.dot(frame["gravity_up_camera_xyz"], expected_up)) >= math.cos(math.radians(15.0)),
        "F2_SCORE_GRAVITY_INCONSISTENT",
    )
    depth = _nearest_resize(frame["depth_m"], output_hw).astype(np.float64)
    valid = _nearest_resize(frame["depth_known"], output_hw).astype(bool)
    depth[~valid] = np.nan
    intrinsics = scale_intrinsics(frame["intrinsics"], frame["depth_m"].shape, output_hw)
    camera_world_height = float(np.dot(frame["camera_to_world"][:3, 3], world_up))
    camera_height = camera_world_height - float(identity["support_world_height_m"])
    if not 0.45 <= camera_height <= 2.20:
        support_known = np.zeros(output_hw, dtype=bool)
        evidence_known = np.zeros(output_hw, dtype=bool)
        support = np.full(output_hw, np.nan, dtype=np.float64)
        residual = np.full(output_hw, np.nan, dtype=np.float64)
        obstacle = np.full(output_hw, np.nan, dtype=np.float64)
        boundary = np.full(output_hw, np.nan, dtype=np.float64)
    else:
        depth_zero = np.nan_to_num(depth, nan=0.0)
        points = backproject(depth_zero, intrinsics)
        normals, normal_valid = dense_normals(depth_zero, valid, intrinsics)
        up = frame["gravity_up_camera_xyz"]
        heights = np.einsum("...i,i->...", points, up) + camera_height
        height_sigma = max(0.06, float(identity["median_absolute_residual_m"]) + 0.04)
        height_score = np.exp(-0.5 * np.square(heights / height_sigma))
        alignment = np.abs(np.einsum("...i,i->...", normals, up))
        cosine = math.cos(math.radians(25.0))
        normal_score = np.clip((alignment - cosine) / (1.0 - cosine), 0.0, 1.0)
        preliminary = valid & normal_valid
        support_value = np.zeros(output_hw, dtype=np.float64)
        support_value[preliminary] = height_score[preliminary] * normal_score[preliminary]
        lower = 1.0 / (1.0 + np.exp(-(heights - 0.08) / 0.04))
        upper = 1.0 / (1.0 + np.exp((heights - 2.00) / 0.15))
        obstacle_value = np.zeros(output_hw, dtype=np.float64)
        obstacle_value[valid] = lower[valid] * upper[valid] * (1.0 - support_value[valid])
        point_edge, normal_edge, neighbor_count = _point_plane_edges(points, normals, normal_valid, valid)
        support_edge, _ = _pairwise_scalar_edge(support_value, preliminary)
        point_strength = np.clip((point_edge - 0.15) / 0.30, 0.0, 1.0)
        support_strength = np.clip((support_edge - 0.20) / 0.40, 0.0, 1.0)
        probability = np.clip(point_strength * (1.0 + 0.25 * np.maximum(normal_edge, support_strength)), 0.0, 1.0)
        physical = valid & (neighbor_count > 0)
        support_known = preliminary & physical & (probability < 0.50)
        evidence_known = physical
        seed = evidence_known & (probability >= 0.50)
        boundary_value = np.minimum(distance_transform_edt(~seed), 32.0) if np.any(seed) else np.full(output_hw, 32.0)
        support = np.full(output_hw, np.nan, dtype=np.float64)
        support[support_known] = support_value[support_known]
        residual = np.full(output_hw, np.nan, dtype=np.float64)
        residual[support_known] = heights[support_known]
        obstacle = np.full(output_hw, np.nan, dtype=np.float64)
        obstacle[evidence_known] = obstacle_value[evidence_known]
        boundary = np.full(output_hw, np.nan, dtype=np.float64)
        boundary[evidence_known] = boundary_value[evidence_known]
    return {
        "parent_id": frame["parent_id"],
        "frame_id": frame["frame_id"],
        "fx": float(intrinsics[0, 0]),
        "fy": float(intrinsics[1, 1]),
        "truth": {
            "depth_m": depth,
            "depth_known": valid,
            "support_probability": support,
            "support_signed_residual_m": residual,
            "support_known": support_known,
            "obstacle_probability": obstacle,
            "boundary_distance_px": boundary,
            "evidence_known": evidence_known,
        },
        "camera_height_m": camera_height,
    }


def source_parent_summary(
    parent_id: str,
    eligible_pair_count: int,
    context: Mapping[str, Any],
    truths: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(len(truths) == 12, "F2_SOURCE_SUMMARY_SCORE_COUNT")
    pixel_counts = [int(np.asarray(row["truth"]["depth_known"]).size) for row in truths]
    require(all(count > 0 for count in pixel_counts), "F2_SOURCE_SUMMARY_PIXEL_COUNT")
    return {
        "parent_id": parent_id,
        "eligible_pair_count": int(eligible_pair_count),
        "calibration_count": 12,
        "score_count": 12,
        "camera_height_m": float(context["camera_height_m"]),
        "camera_height_mad_m": float(context["camera_height_mad_m"]),
        "source_depth_known_coverage": float(
            sum(int(np.sum(row["truth"]["depth_known"])) for row in truths) / sum(pixel_counts)
        ),
        "source_support_known_coverage": float(
            sum(int(np.sum(row["truth"]["support_known"])) for row in truths) / sum(pixel_counts)
        ),
        "source_boundary_known_coverage": float(
            sum(int(np.sum(row["truth"]["evidence_known"])) for row in truths) / sum(pixel_counts)
        ),
    }
