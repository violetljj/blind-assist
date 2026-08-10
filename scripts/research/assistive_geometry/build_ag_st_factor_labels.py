#!/usr/bin/env python3
"""Build selective Assistive Geometry factor pseudo-labels from AG-ST outputs.

The factory is intentionally source-first. Registered sensor depth wins wherever
it is valid; the MapAnything Teacher only fills missing source regions. Teacher
pixels are graded with model confidence, distance-to-anchor residual, and
multi-view reprojection consistency. Low-quality pixels remain UNKNOWN.

This is reversible WILD_LAB materialization. The outputs are pseudo-labels with
explicit provenance, never objective ground truth or task-state supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from arkitscenes_truth_reader import (  # noqa: E402
    WORLD_UP,
    parse_trajectory,
    unproject_depth,
)
from run_ag_st_stage0a import (  # noqa: E402
    compute_selective_metrics,
    load_factor_source_frame,
    select_train_videos,
    sha256_file,
)


DEFAULT_STAGE0A_RESULT = (
    REPO_ROOT
    / "artifacts.local"
    / "experiments"
    / "ag-st-stage0a-mapanything-apache-train16-block64-r1"
    / "result.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local"
    / "experiments"
    / "ag-st-superteacher-factor-labels-train16-r0"
)

TIER_UNKNOWN = 0
TIER_C_TEACHER = 1
TIER_B_ANCHORED = 2
TIER_A_SOURCE = 3

PROVENANCE_UNKNOWN = 0
PROVENANCE_SOURCE_NATIVE = 1
PROVENANCE_TEACHER = 2

TEACHER_B_QUALITY = 0.60
TEACHER_C_QUALITY = 0.30

FORBIDDEN_TASK_TOKENS = (
    "clearance",
    "occupancy",
    "free",
    "blocked",
    "risk_score",
    "final_state",
    "ttc",
)


@dataclass(frozen=True)
class FactorLabelPolicy:
    """Geometry-only constants for pseudo-label derivation."""

    point_stride: int = 2
    plane_height_min_m: float = 0.45
    plane_height_max_m: float = 2.20
    plane_histogram_bin_m: float = 0.04
    plane_support_tolerance_m: float = 0.08
    minimum_plane_support_points: int = 80
    minimum_plane_support_fraction: float = 0.02
    minimum_obstacle_height_m: float = 0.08
    maximum_obstacle_height_m: float = 2.00
    maximum_support_tilt_degrees: float = 25.0


def fit_gravity_support_plane(
    points: np.ndarray,
    up_camera: np.ndarray,
    policy: FactorLabelPolicy,
) -> dict[str, Any] | None:
    """Fit only the gravity-aligned support height mode; no task outcome fields."""
    values = np.asarray(points, dtype=np.float64)
    up = np.asarray(up_camera, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == 3, "support points must be Nx3")
    norm = float(np.linalg.norm(up))
    require(norm > 1e-9 and np.all(np.isfinite(up)), "gravity vector invalid")
    up /= norm
    offsets = -(values @ up)
    plausible = (
        np.isfinite(offsets)
        & (offsets >= policy.plane_height_min_m)
        & (offsets <= policy.plane_height_max_m)
    )
    if int(np.sum(plausible)) < policy.minimum_plane_support_points:
        return None
    edges = np.arange(
        policy.plane_height_min_m,
        policy.plane_height_max_m + policy.plane_histogram_bin_m * 1.001,
        policy.plane_histogram_bin_m,
    )
    counts, edges = np.histogram(offsets[plausible], bins=edges)
    if not len(counts):
        return None
    candidate_bins = np.flatnonzero(counts == int(np.max(counts)))
    mode_index = int(candidate_bins[-1])
    mode_center = float((edges[mode_index] + edges[mode_index + 1]) / 2.0)
    support = plausible & (
        np.abs(offsets - mode_center) <= policy.plane_support_tolerance_m
    )
    support_count = int(np.sum(support))
    minimum_count = max(
        policy.minimum_plane_support_points,
        int(math.ceil(policy.minimum_plane_support_fraction * len(values))),
    )
    if support_count < minimum_count:
        return None
    camera_height = float(np.median(offsets[support]))
    residual = float(np.median(np.abs(offsets[support] - camera_height)))
    return {
        "normal_camera": up,
        "camera_height_m": camera_height,
        "median_residual_m": residual,
        "support_points": support_count,
        "sampled_valid_points": int(len(values)),
        "support_fraction": float(support_count / len(values)),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def robust_confidence_quality(
    confidence: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Map a higher-is-better Teacher confidence field to [0, 1]."""
    values = np.asarray(confidence, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.bool_)
    require(values.shape == mask.shape, "confidence/valid shape mismatch")
    output = np.zeros_like(values, dtype=np.float32)
    finite = mask & np.isfinite(values)
    if not np.any(finite):
        return output
    lower, upper = np.quantile(values[finite].astype(np.float64), (0.10, 0.90))
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower + 1e-6:
        output[finite] = 0.50
        return output
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    output[finite] = 0.05 + 0.95 * normalized[finite]
    return output


def propagated_anchor_signal(
    observed_depth_m: np.ndarray,
    teacher_depth_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate observed Teacher residuals without reading hidden depth values."""
    observed = np.asarray(observed_depth_m, dtype=np.float32)
    teacher = np.asarray(teacher_depth_m, dtype=np.float32)
    require(observed.shape == teacher.shape and observed.ndim == 2, "anchor shape mismatch")
    seeds = (
        np.isfinite(observed)
        & (observed > 0)
        & np.isfinite(teacher)
        & (teacher > 0)
    )
    residual = np.full_like(observed, np.inf, dtype=np.float32)
    distance = np.full_like(observed, np.inf, dtype=np.float32)
    quality = np.zeros_like(observed, dtype=np.float32)
    if not np.any(seeds):
        return residual, distance, quality
    seed_residual = np.zeros_like(observed, dtype=np.float32)
    seed_residual[seeds] = np.abs(teacher[seeds] - observed[seeds])
    distance_values, indices = distance_transform_edt(
        ~seeds,
        return_distances=True,
        return_indices=True,
    )
    residual = seed_residual[tuple(indices)].astype(np.float32)
    distance = distance_values.astype(np.float32)
    tolerance = 0.02 + 0.03 * np.maximum(teacher, 0.0)
    quality = np.exp(-residual / np.maximum(tolerance, 1e-4))
    quality *= np.exp(-distance / 160.0)
    quality[~np.isfinite(teacher) | (teacher <= 0)] = 0.0
    return residual, distance, np.clip(quality, 0.0, 1.0).astype(np.float32)


def backproject_depth_grid(depth_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    require(depth.ndim == 2, "depth must be HxW")
    require(matrix.shape == (3, 3) and np.all(np.isfinite(matrix)), "intrinsics invalid")
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    require(fx > 0 and fy > 0, "focal lengths must be positive")
    return np.stack(
        (
            (columns - cx) * depth / fx,
            (rows - cy) * depth / fy,
            depth,
        ),
        axis=-1,
    )


def compute_dense_normals(
    depth_m: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute camera-facing dense normals from continuous central 3D differences."""
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.bool_)
    require(depth.ndim == 2 and mask.shape == depth.shape, "normal input shape mismatch")
    points = backproject_depth_grid(depth, intrinsics)
    output = np.zeros((*depth.shape, 3), dtype=np.float32)
    output_valid = np.zeros_like(mask)
    if min(depth.shape) < 3:
        return output, output_valid
    tangent_u = points[1:-1, 2:] - points[1:-1, :-2]
    tangent_v = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = np.cross(tangent_u, tangent_v)
    norms = np.linalg.norm(normals, axis=-1)
    neighbors_valid = (
        mask[1:-1, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & np.isfinite(norms)
        & (norms > 1e-9)
    )
    safe_norms = np.where(neighbors_valid, norms, 1.0)
    normals = normals / safe_norms[..., None]
    center_points = points[1:-1, 1:-1]
    camera_facing = np.einsum("...i,...i->...", normals, center_points)
    normals[camera_facing > 0] *= -1.0
    normals[~neighbors_valid] = 0.0
    output[1:-1, 1:-1] = normals.astype(np.float32)
    output_valid[1:-1, 1:-1] = neighbors_valid
    return output, output_valid


def projective_depth_residual(
    source_depth_m: np.ndarray,
    source_valid: np.ndarray,
    source_intrinsics: np.ndarray,
    source_camera_to_world: np.ndarray,
    target_depth_m: np.ndarray,
    target_valid: np.ndarray,
    target_intrinsics: np.ndarray,
    target_camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject source depth into a target view and compare z-depth."""
    source_depth = np.asarray(source_depth_m, dtype=np.float32)
    source_mask = np.asarray(source_valid, dtype=np.bool_)
    target_depth = np.asarray(target_depth_m, dtype=np.float32)
    target_mask = np.asarray(target_valid, dtype=np.bool_)
    require(source_depth.ndim == 2 and source_mask.shape == source_depth.shape, "source shape mismatch")
    require(target_depth.ndim == 2 and target_mask.shape == target_depth.shape, "target shape mismatch")
    source_pose = np.asarray(source_camera_to_world, dtype=np.float64)
    target_pose = np.asarray(target_camera_to_world, dtype=np.float64)
    require(source_pose.shape == target_pose.shape == (4, 4), "pose shape mismatch")
    residual = np.full_like(source_depth, np.nan, dtype=np.float32)
    valid_output = np.zeros_like(source_mask)
    usable = source_mask & np.isfinite(source_depth) & (source_depth > 0)
    if not np.any(usable):
        return residual, valid_output

    points_camera = backproject_depth_grid(source_depth, source_intrinsics)[usable]
    world = points_camera @ source_pose[:3, :3].T + source_pose[:3, 3]
    target_camera = (world - target_pose[:3, 3]) @ target_pose[:3, :3]
    z = target_camera[:, 2]
    matrix = np.asarray(target_intrinsics, dtype=np.float64)
    u = matrix[0, 0] * target_camera[:, 0] / np.maximum(z, 1e-9) + matrix[0, 2]
    v = matrix[1, 1] * target_camera[:, 1] / np.maximum(z, 1e-9) + matrix[1, 2]
    columns = np.rint(u).astype(np.int64)
    rows = np.rint(v).astype(np.int64)
    projected = (
        np.isfinite(z)
        & (z > 0)
        & (rows >= 0)
        & (rows < target_depth.shape[0])
        & (columns >= 0)
        & (columns < target_depth.shape[1])
    )
    flat_source_indices = np.flatnonzero(usable)
    projected_indices = np.flatnonzero(projected)
    if not len(projected_indices):
        return residual, valid_output
    sampled_rows = rows[projected]
    sampled_columns = columns[projected]
    target_usable = (
        target_mask[sampled_rows, sampled_columns]
        & np.isfinite(target_depth[sampled_rows, sampled_columns])
        & (target_depth[sampled_rows, sampled_columns] > 0)
    )
    selected_source = flat_source_indices[projected_indices[target_usable]]
    sampled_rows = sampled_rows[target_usable]
    sampled_columns = sampled_columns[target_usable]
    selected_z = z[projected][target_usable]
    flat_residual = residual.reshape(-1)
    flat_valid = valid_output.reshape(-1)
    flat_residual[selected_source] = np.abs(
        selected_z - target_depth[sampled_rows, sampled_columns]
    ).astype(np.float32)
    flat_valid[selected_source] = True
    return residual, valid_output


def aggregate_multiview_residual(
    source_index: int,
    frames: list["FrameBundle"],
) -> tuple[np.ndarray, np.ndarray]:
    source = frames[source_index]
    residuals: list[np.ndarray] = []
    for target_index, target in enumerate(frames):
        if target_index == source_index:
            continue
        residual, _ = projective_depth_residual(
            source.teacher_depth_m,
            source.teacher_valid,
            source.intrinsics,
            source.camera_to_world,
            target.teacher_depth_m,
            target.teacher_valid,
            target.intrinsics,
            target.camera_to_world,
        )
        residuals.append(residual)
    if not residuals:
        return (
            np.full_like(source.teacher_depth_m, np.nan, dtype=np.float32),
            np.zeros_like(source.teacher_valid),
        )
    stack = np.stack(residuals, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(stack, axis=0).astype(np.float32)
    valid = np.any(np.isfinite(stack), axis=0)
    median[~valid] = np.nan
    return median, valid


def teacher_quality_signals(
    teacher_depth_m: np.ndarray,
    teacher_confidence: np.ndarray,
    teacher_valid: np.ndarray,
    observed_depth_m: np.ndarray,
    multiview_residual_m: np.ndarray,
    multiview_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    depth = np.asarray(teacher_depth_m, dtype=np.float32)
    valid = np.asarray(teacher_valid, dtype=np.bool_)
    confidence_quality = robust_confidence_quality(teacher_confidence, valid)
    anchor_residual, anchor_distance, anchor_quality = propagated_anchor_signal(
        observed_depth_m,
        depth,
    )
    multiview_residual = np.asarray(multiview_residual_m, dtype=np.float32)
    multiview_mask = np.asarray(multiview_valid, dtype=np.bool_)
    tolerance = 0.03 + 0.05 * np.maximum(depth, 0.0)
    multiview_quality = np.full_like(depth, 0.55, dtype=np.float32)
    multiview_quality[multiview_mask] = np.exp(
        -multiview_residual[multiview_mask]
        / np.maximum(tolerance[multiview_mask], 1e-4)
    )
    multiview_quality = np.clip(multiview_quality, 0.0, 1.0)
    confidence_only = confidence_quality * valid
    anchor_combined = np.sqrt(confidence_quality * anchor_quality) * valid
    combined = np.cbrt(confidence_quality * anchor_quality * multiview_quality) * valid
    return {
        "confidence_quality": confidence_only.astype(np.float32),
        "anchor_quality": anchor_quality.astype(np.float32),
        "anchor_combined_quality": anchor_combined.astype(np.float32),
        "combined_quality": combined.astype(np.float32),
        "anchor_residual_m": anchor_residual.astype(np.float32),
        "anchor_distance_px": anchor_distance.astype(np.float32),
        "multiview_quality": multiview_quality.astype(np.float32),
        "multiview_residual_m": multiview_residual.astype(np.float32),
        "multiview_valid": multiview_mask,
    }


def assign_quality_tiers(
    source_valid: np.ndarray,
    sensor_confidence: np.ndarray,
    teacher_valid: np.ndarray,
    teacher_quality: np.ndarray,
    anchor_quality: np.ndarray,
    multiview_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source_valid, dtype=np.bool_)
    sensor = np.asarray(sensor_confidence)
    teacher = np.asarray(teacher_valid, dtype=np.bool_)
    quality = np.asarray(teacher_quality, dtype=np.float32)
    anchor = np.asarray(anchor_quality, dtype=np.float32)
    multiview = np.asarray(multiview_valid, dtype=np.bool_)
    require(source.shape == sensor.shape == teacher.shape == quality.shape == anchor.shape == multiview.shape, "tier shape mismatch")
    tiers = np.zeros(source.shape, dtype=np.uint8)
    provenance = np.zeros(source.shape, dtype=np.uint8)
    scores = np.zeros(source.shape, dtype=np.float32)

    source_a = source & (sensor >= 2)
    source_b = source & ~source_a
    tiers[source_a] = TIER_A_SOURCE
    tiers[source_b] = TIER_B_ANCHORED
    provenance[source] = PROVENANCE_SOURCE_NATIVE
    scores[source_a] = 0.98
    scores[source_b] = 0.90

    teacher_only = ~source & teacher
    teacher_b = (
        teacher_only
        & (quality >= TEACHER_B_QUALITY)
        & (multiview | (anchor >= 0.75))
    )
    teacher_c = teacher_only & ~teacher_b & (quality >= TEACHER_C_QUALITY)
    tiers[teacher_b] = TIER_B_ANCHORED
    tiers[teacher_c] = TIER_C_TEACHER
    provenance[teacher_b | teacher_c] = PROVENANCE_TEACHER
    scores[teacher_b | teacher_c] = quality[teacher_b | teacher_c]
    return tiers, provenance, scores


def _pairwise_scalar_edges(
    values: np.ndarray,
    valid: np.ndarray,
    tolerance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.bool_)
    scale = np.asarray(tolerance, dtype=np.float32)
    scores = np.zeros_like(field, dtype=np.float32)
    neighbors = np.zeros_like(mask, dtype=np.uint8)
    for first, second in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        pair_valid = mask[first] & mask[second]
        pair_tolerance = np.maximum(0.5 * (scale[first] + scale[second]), 1e-4)
        difference = np.abs(field[first] - field[second]) / pair_tolerance
        pair_score = (1.0 - np.exp(-difference)).astype(np.float32)
        pair_score[~pair_valid] = 0.0
        scores[first] = np.maximum(scores[first], pair_score)
        scores[second] = np.maximum(scores[second], pair_score)
        neighbors[first] += pair_valid.astype(np.uint8)
        neighbors[second] += pair_valid.astype(np.uint8)
    return scores, neighbors


def _pairwise_point_to_plane_edges(
    points: np.ndarray,
    normals: np.ndarray,
    normal_valid: np.ndarray,
    metric_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score 3D discontinuity without treating a continuous sloped plane as an edge."""
    point_values = np.asarray(points, dtype=np.float64)
    normal_values = np.asarray(normals, dtype=np.float64)
    normal_mask = np.asarray(normal_valid, dtype=np.bool_)
    metric_mask = np.asarray(metric_valid, dtype=np.bool_)
    require(point_values.shape == normal_values.shape, "point/normal shape mismatch")
    require(point_values.shape[:2] == normal_mask.shape == metric_mask.shape, "geometric edge mask shape mismatch")
    scores = np.zeros(metric_mask.shape, dtype=np.float32)
    normal_jump = np.zeros(metric_mask.shape, dtype=np.float32)
    neighbors = np.zeros(metric_mask.shape, dtype=np.uint8)
    for first, second in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        pair_valid = (
            metric_mask[first]
            & metric_mask[second]
            & normal_mask[first]
            & normal_mask[second]
        )
        delta = point_values[second] - point_values[first]
        first_residual = np.abs(np.sum(delta * normal_values[first], axis=-1))
        second_residual = np.abs(np.sum(delta * normal_values[second], axis=-1))
        depth_scale = np.minimum(point_values[first][..., 2], point_values[second][..., 2])
        tolerance = 0.01 + 0.015 * np.maximum(depth_scale, 0.0)
        pair_score = 1.0 - np.exp(
            -np.maximum(first_residual, second_residual) / np.maximum(tolerance, 1e-4)
        )
        dot = np.sum(normal_values[first] * normal_values[second], axis=-1)
        pair_normal_jump = np.clip(
            (1.0 - dot) / (1.0 - math.cos(math.radians(20.0))),
            0.0,
            1.0,
        )
        pair_score[~pair_valid] = 0.0
        pair_normal_jump[~pair_valid] = 0.0
        scores[first] = np.maximum(scores[first], pair_score.astype(np.float32))
        scores[second] = np.maximum(scores[second], pair_score.astype(np.float32))
        normal_jump[first] = np.maximum(normal_jump[first], pair_normal_jump.astype(np.float32))
        normal_jump[second] = np.maximum(normal_jump[second], pair_normal_jump.astype(np.float32))
        neighbors[first] += pair_valid.astype(np.uint8)
        neighbors[second] += pair_valid.astype(np.uint8)
    return scores, normal_jump, neighbors


def compute_geometric_factors(
    metric_depth_m: np.ndarray,
    metric_valid: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    quality_score: np.ndarray,
    quality_tier: np.ndarray,
    provenance: np.ndarray,
    depth_uncertainty_m: np.ndarray,
    policy: FactorLabelPolicy = FactorLabelPolicy(),
) -> dict[str, Any]:
    depth = np.asarray(metric_depth_m, dtype=np.float32)
    valid = np.asarray(metric_valid, dtype=np.bool_)
    quality = np.asarray(quality_score, dtype=np.float32)
    tiers = np.asarray(quality_tier, dtype=np.uint8)
    provenance_values = np.asarray(provenance, dtype=np.uint8)
    uncertainty = np.asarray(depth_uncertainty_m, dtype=np.float32)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    require(
        depth.shape
        == valid.shape
        == quality.shape
        == tiers.shape
        == provenance_values.shape
        == uncertainty.shape,
        "factor shape mismatch",
    )
    require(pose.shape == (4, 4), "camera pose invalid")
    up_camera = pose[:3, :3].T @ WORLD_UP
    up_camera /= np.linalg.norm(up_camera)
    normals, normal_valid = compute_dense_normals(depth, valid, intrinsics)
    normal_tiers = minimum_filter(tiers, size=3, mode="constant", cval=TIER_UNKNOWN)
    teacher_neighborhood = maximum_filter(
        (provenance_values == PROVENANCE_TEACHER).astype(np.uint8),
        size=3,
        mode="constant",
        cval=0,
    ) > 0
    normal_valid &= normal_tiers > TIER_UNKNOWN
    normal_tiers[~normal_valid] = TIER_UNKNOWN
    normal_provenance = np.zeros_like(provenance_values)
    normal_provenance[normal_valid & ~teacher_neighborhood] = PROVENANCE_SOURCE_NATIVE
    normal_provenance[normal_valid & teacher_neighborhood] = PROVENANCE_TEACHER

    source_fit = (
        valid
        & (tiers >= TIER_B_ANCHORED)
        & (provenance_values == PROVENANCE_SOURCE_NATIVE)
    )
    sampled_points, _ = unproject_depth(
        depth,
        intrinsics,
        source_fit,
        policy.point_stride,
    )
    plane_fit_mask = source_fit
    if len(sampled_points) < policy.minimum_plane_support_points:
        plane_fit_mask = valid & (tiers >= TIER_B_ANCHORED)
        sampled_points, _ = unproject_depth(
            depth,
            intrinsics,
            plane_fit_mask,
            policy.point_stride,
        )
    plane = (
        fit_gravity_support_plane(sampled_points, up_camera, policy)
        if len(sampled_points) >= policy.minimum_plane_support_points
        else None
    )
    plane_tier = (
        int(np.min(tiers[plane_fit_mask]))
        if plane is not None and np.any(plane_fit_mask)
        else TIER_UNKNOWN
    )
    plane_uses_teacher = bool(
        plane is not None
        and np.any(plane_fit_mask & (provenance_values == PROVENANCE_TEACHER))
    )
    plane_provenance = (
        PROVENANCE_TEACHER
        if plane_uses_teacher
        else PROVENANCE_SOURCE_NATIVE
        if plane is not None
        else PROVENANCE_UNKNOWN
    )
    if plane is not None:
        require(
            float(np.dot(plane["normal_camera"], up_camera)) > 0.999,
            "support-plane normal is not aligned with gravity up",
        )

    support_probability = np.zeros_like(depth, dtype=np.float32)
    support_valid = np.zeros_like(valid)
    obstacle_probability = np.zeros_like(depth, dtype=np.float32)
    obstacle_valid = np.zeros_like(valid)
    heights = np.full_like(depth, np.nan, dtype=np.float32)
    if plane is not None:
        points = backproject_depth_grid(depth, intrinsics)
        heights = (
            np.einsum("...i,i->...", points, plane["normal_camera"])
            + float(plane["camera_height_m"])
        ).astype(np.float32)
        height_sigma = max(0.06, float(plane["median_residual_m"]) + 0.04)
        height_score = np.exp(-0.5 * np.square(heights / height_sigma))
        normal_alignment = np.abs(
            np.einsum("...i,i->...", normals, plane["normal_camera"])
        )
        cosine_limit = math.cos(math.radians(policy.maximum_support_tilt_degrees))
        normal_score = np.clip(
            (normal_alignment - cosine_limit) / (1.0 - cosine_limit),
            0.0,
            1.0,
        )
        support_valid = valid & normal_valid
        support_probability[support_valid] = (
            height_score[support_valid] * normal_score[support_valid]
        ).astype(np.float32)
        lower = 1.0 / (
            1.0 + np.exp(-(heights - policy.minimum_obstacle_height_m) / 0.04)
        )
        upper = 1.0 / (
            1.0 + np.exp((heights - policy.maximum_obstacle_height_m) / 0.15)
        )
        obstacle_valid = valid
        obstacle_probability[obstacle_valid] = (
            lower[obstacle_valid]
            * upper[obstacle_valid]
            * (1.0 - support_probability[obstacle_valid])
        ).astype(np.float32)

    points = backproject_depth_grid(depth, intrinsics)
    point_plane_edge, normal_edge, neighbor_count = _pairwise_point_to_plane_edges(
        points,
        normals,
        normal_valid,
        valid,
    )
    support_edge, _ = _pairwise_scalar_edges(
        support_probability,
        support_valid,
        np.full_like(depth, 0.20, dtype=np.float32),
    )
    boundary_probability = np.maximum.reduce(
        (point_plane_edge, 0.85 * normal_edge, 0.75 * support_edge)
    ).astype(np.float32)
    boundary_tiers = minimum_filter(
        tiers,
        size=3,
        mode="constant",
        cval=TIER_UNKNOWN,
    )
    physical_boundary_valid = valid & (neighbor_count > 0) & (boundary_tiers > 0)
    boundary_probability[~physical_boundary_valid] = 0.0
    support_valid &= boundary_probability < 0.50
    support_probability[~support_valid] = 0.0
    support_tiers = np.minimum(normal_tiers, plane_tier).astype(np.uint8)
    support_tiers[~support_valid] = TIER_UNKNOWN
    support_provenance = np.zeros_like(provenance_values)
    support_teacher = (
        support_valid
        & (
            (normal_provenance == PROVENANCE_TEACHER)
            | (plane_provenance == PROVENANCE_TEACHER)
        )
    )
    support_provenance[support_valid & ~support_teacher] = PROVENANCE_SOURCE_NATIVE
    support_provenance[support_teacher] = PROVENANCE_TEACHER
    obstacle_tiers = np.minimum(tiers, plane_tier).astype(np.uint8)
    obstacle_tiers[~obstacle_valid] = TIER_UNKNOWN
    evidence_tiers = np.minimum(boundary_tiers, obstacle_tiers).astype(np.uint8)
    evidence_valid = physical_boundary_valid & obstacle_valid & (evidence_tiers > 0)
    obstacle_probability[~evidence_valid] = 0.0
    boundary_probability[~evidence_valid] = 0.0

    boundary_seed = evidence_valid & (boundary_probability >= 0.50)
    if np.any(boundary_seed):
        boundary_distance = np.minimum(
            distance_transform_edt(~boundary_seed),
            32.0,
        ).astype(np.float32)
    else:
        boundary_distance = np.full_like(depth, 32.0, dtype=np.float32)
    boundary_distance[~evidence_valid] = np.nan
    uncertainty_ratio = np.clip(
        uncertainty / np.maximum(0.03 + 0.05 * np.maximum(depth, 0.0), 1e-4),
        0.0,
        1.0,
    )
    boundary_uncertainty = (
        0.75 + 3.0 * uncertainty_ratio + 1.5 * (1.0 - quality)
    ).astype(np.float32)
    boundary_uncertainty[~evidence_valid] = np.nan
    support_truth_valid = support_valid & (
        (support_probability >= 0.70) | (support_probability <= 0.20)
    )
    support_truth = (support_probability >= 0.70).astype(np.float32)
    support_truth[~support_truth_valid] = 0.0
    evidence_provenance = np.zeros_like(provenance_values)
    evidence_teacher = evidence_valid & (
        teacher_neighborhood | (plane_provenance == PROVENANCE_TEACHER)
    )
    evidence_provenance[evidence_valid & ~evidence_teacher] = PROVENANCE_SOURCE_NATIVE
    evidence_provenance[evidence_teacher] = PROVENANCE_TEACHER
    return {
        "dense_normal_diagnostic_camera_xyz_hwc": normals,
        "normal_valid_hw": normal_valid,
        "normal_quality_tier_hw": normal_tiers,
        "normal_provenance_code_hw": normal_provenance,
        "support_probability_pseudo_hw": support_probability,
        "support_truth_hw": support_truth,
        "support_truth_valid_hw": support_truth_valid,
        "support_quality_tier_hw": support_tiers,
        "support_provenance_code_hw": support_provenance,
        "support_plane_normal_camera_xyz": (
            np.asarray(plane["normal_camera"], dtype=np.float32)
            if plane is not None
            else np.zeros(3, dtype=np.float32)
        ),
        "camera_height_m": (
            np.asarray(float(plane["camera_height_m"]), dtype=np.float32)
            if plane is not None
            else np.asarray(np.nan, dtype=np.float32)
        ),
        "support_plane_fit_residual_diagnostic_m": (
            np.asarray(max(float(plane["median_residual_m"]), 0.01), dtype=np.float32)
            if plane is not None
            else np.asarray(np.nan, dtype=np.float32)
        ),
        "support_plane_valid": np.asarray(plane is not None, dtype=np.bool_),
        "support_plane_quality_tier": np.asarray(plane_tier, dtype=np.uint8),
        "support_plane_provenance_code": np.asarray(plane_provenance, dtype=np.uint8),
        "support_plane_fit_source_pixel_count": np.asarray(
            int(np.sum(plane_fit_mask & (provenance_values == PROVENANCE_SOURCE_NATIVE))),
            dtype=np.int64,
        ),
        "support_plane_fit_teacher_pixel_count": np.asarray(
            int(np.sum(plane_fit_mask & (provenance_values == PROVENANCE_TEACHER))),
            dtype=np.int64,
        ),
        "height_above_support_m_hw": heights,
        "obstacle_evidence_truth_hw": obstacle_probability,
        "boundary_probability_pseudo_hw": boundary_probability,
        "boundary_distance_px_hw": boundary_distance,
        "boundary_uncertainty_proxy_px_hw": boundary_uncertainty,
        "evidence_truth_valid_hw": evidence_valid,
        "evidence_quality_tier_hw": evidence_tiers,
        "evidence_provenance_code_hw": evidence_provenance,
        "physical_boundary_valid_diagnostic_hw": physical_boundary_valid,
    }


def depth_uncertainty_proxy(
    depth_m: np.ndarray,
    tiers: np.ndarray,
    provenance: np.ndarray,
    quality_score: np.ndarray,
    anchor_residual_m: np.ndarray,
    multiview_residual_m: np.ndarray,
    multiview_valid: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    tier = np.asarray(tiers, dtype=np.uint8)
    provenance_values = np.asarray(provenance, dtype=np.uint8)
    quality = np.asarray(quality_score, dtype=np.float32)
    anchor = np.asarray(anchor_residual_m, dtype=np.float32)
    multiview = np.asarray(multiview_residual_m, dtype=np.float32)
    multiview_mask = np.asarray(multiview_valid, dtype=np.bool_)
    base = 0.015 + 0.02 * np.maximum(depth, 0.0)
    output = base * (1.0 + 2.5 * (1.0 - quality))
    require(provenance_values.shape == tier.shape, "uncertainty provenance shape mismatch")
    teacher = provenance_values == PROVENANCE_TEACHER
    output[teacher] += 0.50 * np.minimum(anchor[teacher], 0.50)
    teacher_multiview = teacher & multiview_mask
    output[teacher_multiview] += 0.50 * np.minimum(multiview[teacher_multiview], 0.50)
    source = provenance_values == PROVENANCE_SOURCE_NATIVE
    output[source] = 0.015 + 0.01 * depth[source]
    output[tier == TIER_UNKNOWN] = np.nan
    return output.astype(np.float32)


@dataclass
class FrameBundle:
    parent_id: str
    frame_index: int
    frame_stem: str
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    source_depth_m: np.ndarray
    source_valid: np.ndarray
    sensor_confidence: np.ndarray
    observed_depth_m: np.ndarray
    teacher_depth_m: np.ndarray
    teacher_confidence: np.ndarray
    teacher_valid: np.ndarray
    hidden_mask: np.ndarray
    baseline_depth_m: np.ndarray


def _preprocess_source_for_output(
    frame: dict[str, Any],
    output_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from mapanything.utils.cropping import crop_resize_if_necessary

    depth = np.asarray(frame["depth_m_upright"], dtype=np.float32)
    source_valid = np.asarray(frame["depth_valid_upright"], dtype=np.uint8)
    sensor_confidence = np.asarray(frame["confidence_upright"], dtype=np.uint8)
    processed = crop_resize_if_necessary(
        image=np.asarray(frame["rgb_upright"], dtype=np.uint8),
        resolution=output_wh,
        depthmap=depth,
        intrinsics=np.asarray(frame["intrinsics_upright"], dtype=np.float64),
        additional_quantities=[source_valid, sensor_confidence],
    )
    _, depth_output, intrinsics_output, quantities = processed
    return (
        np.asarray(depth_output, dtype=np.float32),
        np.asarray(quantities[0] > 0, dtype=np.bool_),
        np.asarray(np.rint(quantities[1]), dtype=np.uint8),
        np.asarray(intrinsics_output, dtype=np.float64),
    )


def _load_parent_frames(
    parent_run: dict[str, Any],
    video: dict[str, Any],
    stage0a_dir: Path,
) -> list[FrameBundle]:
    trajectory = parse_trajectory(Path(video["trajectory"]["path"]))
    frames: list[FrameBundle] = []
    seen: set[tuple[str, int]] = set()
    for summary in parent_run["frame_summaries"]:
        frame_index = int(summary["frame_index"])
        frame_stem = str(summary["frame_stem"])
        identity = (str(video["video_id"]), frame_index)
        require(identity not in seen, "duplicate parent/frame identity")
        seen.add(identity)
        npz_path = stage0a_dir / f"{video['video_id']}_{frame_stem}.npz"
        require(npz_path.is_file(), f"missing Stage 0A frame payload: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as payload:
            values = {key: np.asarray(payload[key]).copy() for key in payload.files}
        teacher_depth = np.asarray(values["prediction_depth_m"], dtype=np.float32)
        output_wh = (teacher_depth.shape[1], teacher_depth.shape[0])
        source_frame = load_factor_source_frame(video, frame_index, trajectory)
        source_depth, source_valid, sensor_confidence, intrinsics = _preprocess_source_for_output(
            source_frame,
            output_wh,
        )
        require(source_depth.shape == teacher_depth.shape, "source/Teacher output shape drift")
        reference = np.asarray(values["truth_depth_m"], dtype=np.float32)
        reference_valid = np.asarray(values["source_valid"], dtype=np.bool_)
        require(np.array_equal(source_valid, reference_valid), "Stage 0A source-valid replay drift")
        require(
            np.allclose(source_depth[source_valid], reference[source_valid], atol=1e-5, rtol=0.0),
            "Stage 0A source-depth replay drift",
        )
        teacher_valid = (
            np.asarray(values["teacher_non_ambiguous_mask"], dtype=np.bool_)
            & np.isfinite(teacher_depth)
            & (teacher_depth > 0)
        )
        frames.append(
            FrameBundle(
                parent_id=str(video["video_id"]),
                frame_index=frame_index,
                frame_stem=frame_stem,
                intrinsics=intrinsics,
                camera_to_world=np.asarray(source_frame["camera_to_world_upright"], dtype=np.float64),
                source_depth_m=source_depth,
                source_valid=source_valid,
                sensor_confidence=sensor_confidence,
                observed_depth_m=np.asarray(values["observed_depth_m"], dtype=np.float32),
                teacher_depth_m=teacher_depth,
                teacher_confidence=np.asarray(values["teacher_confidence"], dtype=np.float32),
                teacher_valid=teacher_valid,
                hidden_mask=np.asarray(values["hidden_mask"], dtype=np.bool_),
                baseline_depth_m=np.asarray(values["source_only_nearest_depth_m"], dtype=np.float32),
            )
        )
    require(len(frames) >= 2, "each parent needs at least two Stage 0A views")
    return frames


def _tier_counts(tiers: np.ndarray) -> dict[str, int]:
    values = np.asarray(tiers, dtype=np.uint8)
    return {
        "UNKNOWN": int(np.sum(values == TIER_UNKNOWN)),
        "C_TEACHER": int(np.sum(values == TIER_C_TEACHER)),
        "B_ANCHORED": int(np.sum(values == TIER_B_ANCHORED)),
        "A_SOURCE": int(np.sum(values == TIER_A_SOURCE)),
    }


def _compact_curve(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "coverage": row["coverage_of_hidden"],
            "mae_m": row["overall"]["mae_m"],
            "bad_0_10m_rate": row["overall"]["bad_0_10m_rate"],
            "parent_macro_mae_m": row["parent_macro_mae_m"],
            "parent_macro_evaluable": row["parent_macro_evaluable"],
        }
        for row in metrics["teacher_confidence_risk_coverage"]
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    require(args.stage0a_result.is_file(), f"missing Stage 0A result: {args.stage0a_result}")
    require(not args.output_dir.exists(), f"output directory already exists: {args.output_dir}")
    stage0a = json.loads(args.stage0a_result.read_text(encoding="utf-8"))
    require(stage0a.get("schema") == "blindassist_ag_st_stage0a_wild_lab_result_v1", "Stage 0A schema drift")
    require(stage0a.get("status") == "COMPLETED", "Stage 0A result is not completed")
    source_manifest_path = Path(stage0a["source"]["manifest_path"])
    require(source_manifest_path.is_file(), "Stage 0A source manifest missing")
    require(
        sha256_file(source_manifest_path) == stage0a["source"]["manifest_sha256"],
        "Stage 0A source manifest SHA drift",
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    parent_ids = [str(value) for value in stage0a["source"]["parents"]]
    videos = {str(row["video_id"]): row for row in select_train_videos(source_manifest, parent_ids)}
    parent_runs = {str(row["parent_id"]): row for row in stage0a["parent_runs"]}
    require(set(parent_runs) == set(parent_ids), "Stage 0A parent result roster drift")
    args.output_dir.mkdir(parents=True)

    evaluation_records: dict[str, list[dict[str, Any]]] = {
        "confidence_only": [],
        "confidence_plus_anchor": [],
        "confidence_plus_anchor_plus_multiview": [],
    }
    frame_receipts: list[dict[str, Any]] = []
    aggregate_tiers = np.zeros(4, dtype=np.int64)
    aggregate_pixels = 0
    aggregate_source_valid = 0
    aggregate_teacher_added = 0
    aggregate_metric_valid = 0
    aggregate_normal_valid = 0
    aggregate_support_valid = 0
    aggregate_evidence_valid = 0
    support_plane_valid_frames = 0

    for parent_id in parent_ids:
        frames = _load_parent_frames(parent_runs[parent_id], videos[parent_id], args.stage0a_result.parent)
        multiview = [aggregate_multiview_residual(index, frames) for index in range(len(frames))]
        for frame, (multiview_residual, multiview_valid) in zip(frames, multiview):
            signals = teacher_quality_signals(
                frame.teacher_depth_m,
                frame.teacher_confidence,
                frame.teacher_valid,
                frame.observed_depth_m,
                multiview_residual,
                multiview_valid,
            )
            tiers, provenance, quality_score = assign_quality_tiers(
                frame.source_valid,
                frame.sensor_confidence,
                frame.teacher_valid,
                signals["combined_quality"],
                signals["anchor_quality"],
                signals["multiview_valid"],
            )
            metric_depth = np.where(
                frame.source_valid,
                frame.source_depth_m,
                frame.teacher_depth_m,
            ).astype(np.float32)
            metric_valid = tiers > TIER_UNKNOWN
            metric_depth[~metric_valid] = np.nan
            uncertainty = depth_uncertainty_proxy(
                metric_depth,
                tiers,
                provenance,
                quality_score,
                signals["anchor_residual_m"],
                signals["multiview_residual_m"],
                signals["multiview_valid"],
            )
            factors = compute_geometric_factors(
                metric_depth,
                metric_valid,
                frame.intrinsics,
                frame.camera_to_world,
                quality_score,
                tiers,
                provenance,
                uncertainty,
            )
            label_payload = {
                "metric_depth_m_hw": metric_depth,
                "metric_depth_valid_hw": metric_valid,
                "depth_uncertainty_proxy_m_hw": uncertainty,
                "quality_score_hw": quality_score,
                "quality_tier_hw": tiers,
                "provenance_code_hw": provenance,
                "source_native_valid_hw": frame.source_valid,
                "teacher_candidate_valid_hw": frame.teacher_valid,
                "teacher_confidence_quality_hw": signals["confidence_quality"],
                "anchor_quality_hw": signals["anchor_quality"],
                "anchor_residual_m_hw": signals["anchor_residual_m"],
                "anchor_distance_px_hw": signals["anchor_distance_px"],
                "multiview_quality_hw": signals["multiview_quality"],
                "multiview_residual_m_hw": signals["multiview_residual_m"],
                "multiview_valid_hw": signals["multiview_valid"],
                "intrinsics_output": frame.intrinsics.astype(np.float64),
                "camera_to_world_output": frame.camera_to_world.astype(np.float64),
                **factors,
            }
            for key in label_payload:
                lower = key.lower()
                require(not any(token in lower for token in FORBIDDEN_TASK_TOKENS), f"forbidden task field in label payload: {key}")
            require(np.all(np.isfinite(metric_depth[metric_valid])), "non-finite valid metric depth")
            require(np.all((quality_score >= 0) & (quality_score <= 1)), "quality score outside [0,1]")
            normal_values = factors["dense_normal_diagnostic_camera_xyz_hwc"][factors["normal_valid_hw"]]
            if len(normal_values):
                normal_error = float(np.max(np.abs(np.linalg.norm(normal_values, axis=1) - 1.0)))
                require(normal_error <= 1e-4, "dense normal unit-norm drift")
            else:
                normal_error = None

            output_path = args.output_dir / f"{frame.frame_stem}.npz"
            np.savez_compressed(output_path, **label_payload)
            tier_counts = _tier_counts(tiers)
            teacher_added = (~frame.source_valid) & metric_valid
            frame_receipts.append(
                {
                    "parent_id": parent_id,
                    "frame_index": frame.frame_index,
                    "frame_stem": frame.frame_stem,
                    "output_path": str(output_path.resolve()),
                    "output_bytes": output_path.stat().st_size,
                    "tier_counts": tier_counts,
                    "source_native_coverage": float(np.mean(frame.source_valid)),
                    "teacher_added_coverage": float(np.mean(teacher_added)),
                    "metric_label_coverage": float(np.mean(metric_valid)),
                    "normal_coverage": float(np.mean(factors["normal_valid_hw"])),
                    "support_coverage": float(np.mean(factors["support_truth_valid_hw"])),
                    "evidence_coverage": float(np.mean(factors["evidence_truth_valid_hw"])),
                    "support_plane_valid": bool(factors["support_plane_valid"]),
                    "normal_max_unit_error": normal_error,
                }
            )
            counts = np.bincount(tiers.reshape(-1), minlength=4)
            aggregate_tiers += counts[:4]
            aggregate_pixels += tiers.size
            aggregate_source_valid += int(frame.source_valid.sum())
            aggregate_teacher_added += int(teacher_added.sum())
            aggregate_metric_valid += int(metric_valid.sum())
            aggregate_normal_valid += int(factors["normal_valid_hw"].sum())
            aggregate_support_valid += int(factors["support_truth_valid_hw"].sum())
            aggregate_evidence_valid += int(factors["evidence_truth_valid_hw"].sum())
            support_plane_valid_frames += int(bool(factors["support_plane_valid"]))

            signal_values = {
                "confidence_only": frame.teacher_confidence,
                "confidence_plus_anchor": signals["anchor_combined_quality"],
                "confidence_plus_anchor_plus_multiview": signals["combined_quality"],
            }
            for name, score in signal_values.items():
                evaluation_records[name].append(
                    {
                        "parent_id": parent_id,
                        "truth_depth_m": frame.source_depth_m,
                        "prediction_depth_m": frame.teacher_depth_m,
                        "confidence": score,
                        "hidden_mask": frame.hidden_mask,
                        "model_mask": frame.teacher_valid,
                        "baseline_depth_m": frame.baseline_depth_m,
                    }
                )

    risk_coverage = {
        name: compute_selective_metrics(records)
        for name, records in evaluation_records.items()
    }
    tier_names = ("UNKNOWN", "C_TEACHER", "B_ANCHORED", "A_SOURCE")
    result = {
        "schema": "blindassist_ag_st_superteacher_factor_label_factory_wild_lab_result_v1",
        "status": "COMPLETED",
        "mode": "WILD_LAB_REVERSIBLE_EXPLORATION",
        "question": "Can source-first fusion plus confidence, anchor residual, and multi-view consistency produce useful graded factor pseudo-labels without complete truth?",
        "input": {
            "stage0a_result_path": str(args.stage0a_result.resolve()),
            "stage0a_result_sha256": sha256_file(args.stage0a_result),
            "source_manifest_path": str(source_manifest_path.resolve()),
            "source_manifest_sha256": stage0a["source"]["manifest_sha256"],
            "parent_count": len(parent_ids),
            "frame_count": len(frame_receipts),
            "role": "CONSUMED_TRAIN_WILD_LAB",
        },
        "factory": {
            "source_priority": "SOURCE_NATIVE_VALID_DEPTH_OVERRIDES_TEACHER",
            "teacher_model_id": stage0a["teacher"]["model_id"],
            "teacher_checkpoint_sha256": stage0a["teacher"]["checkpoint_sha256"],
            "teacher_b_quality_threshold": TEACHER_B_QUALITY,
            "teacher_c_quality_threshold": TEACHER_C_QUALITY,
            "quality_signals": [
                "Teacher confidence",
                "observed-anchor residual propagated without hidden reference",
                "distance to observed anchor",
                "multi-view metric reprojection residual",
                "Teacher non-ambiguous mask",
            ],
            "derived_factors": [
                "metric depth",
                "dense normal diagnostic",
                "support probability and support plane",
                "obstacle evidence probability",
                "physical boundary probability",
                "depth and boundary uncertainty proxies",
            ],
            "forbidden_task_fields_written": False,
        },
        "coverage": {
            "total_pixels": aggregate_pixels,
            "tier_counts": {
                tier_names[index]: int(aggregate_tiers[index])
                for index in range(4)
            },
            "tier_rates": {
                tier_names[index]: float(aggregate_tiers[index] / aggregate_pixels)
                for index in range(4)
            },
            "source_native_coverage": aggregate_source_valid / aggregate_pixels,
            "teacher_added_coverage": aggregate_teacher_added / aggregate_pixels,
            "metric_label_coverage": aggregate_metric_valid / aggregate_pixels,
            "normal_coverage": aggregate_normal_valid / aggregate_pixels,
            "support_coverage": aggregate_support_valid / aggregate_pixels,
            "boundary_evidence_coverage": aggregate_evidence_valid / aggregate_pixels,
            "support_plane_valid_frames": support_plane_valid_frames,
            "support_plane_valid_frame_rate": support_plane_valid_frames / len(frame_receipts),
        },
        "risk_coverage": {
            name: {
                "compact_curve": _compact_curve(metrics),
                "full": metrics,
            }
            for name, metrics in risk_coverage.items()
        },
        "frame_receipts": frame_receipts,
        "elapsed_seconds": time.monotonic() - started,
        "next_decision": "Use the observed ablation and coverage to decide whether a second independent Teacher is worth its cost; do not require complete truth before training a masked student on A/B/C pseudo-labels.",
        "claim_boundary": "Consumed TRAIN-only source/Teacher pseudo-label factory diagnostic. Outputs are graded pseudo-labels, not objective truth, uncertainty truth, F1 authorization, cross-source generalization, deployment, product, or safety evidence.",
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0a-result", type=Path, default=DEFAULT_STAGE0A_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        _write_json_exclusive(args.output_dir / "result.json", result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "result": str(args.output_dir / "result.json"),
                    "coverage": result["coverage"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
