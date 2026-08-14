#!/usr/bin/env python3
"""Evaluate query-local inverse-depth ray-plane representation headroom.

Phase A fits geometry-only oracle parameters and seals a candidate plan before
any clearance field or tri-state task outcome is derived.  Phase B reopens that
plan, constructs per-query temporary depth in memory, calls the frozen
clearance reducer, and discards the temporary geometry.  No corrected dense
depth is persisted and no learned model is trained.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import cv2
import numpy as np

HFTF_DIR = Path(__file__).resolve().parents[1] / "hftf"
if str(HFTF_DIR) not in sys.path:
    sys.path.insert(0, str(HFTF_DIR))

from evaluate_metric3d_clearance_field_a0 import (  # noqa: E402
    BANDS,
    HORIZONS_M,
    clearance_field,
    depth_to_points,
    fit_gravity_guided_ground_plane,
    intrinsics_matrix,
    load_tum_camera_poses,
    quaternion_rotation_matrix_xyzw,
    tum_depth_metres,
)
from prepare_bonn_rgbd_metric_depth_manifest import (  # noqa: E402
    normalize_depth_image,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "docs/research/assistive-geometry/"
    "BLINDASSIST_BA_CLEAR_QPLANE_O0A_REPRESENTATION_HEADROOM_PROTOCOL_2026-08-14.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ba-clear-qplane-o0a-headroom-r0"
)
ARMS = (
    "A0_FROZEN_DEPTHART",
    "A1_GLOBAL_SCALE",
    "A2_GLOBAL_AFFINE",
    "A3_GLOBAL_RAY_PLANE",
    "A4_QUERY_LOCAL_RAY_PLANE",
    "A5_SOURCE_DEPTH_ORACLE",
    "NC_SHUFFLED_QUERY",
    "NC_WRONG_GRAVITY",
    "NC_WRONG_K",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def query_key(band: str, horizon: float) -> str:
    return f"{band}@{horizon:.1f}m"


def frozen_queries() -> list[tuple[str, float]]:
    return [
        (band, float(horizon))
        for band in BANDS
        for horizon in HORIZONS_M
    ]


def ray_grid(intrinsics: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width]
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    return np.stack(
        (
            (columns.astype(np.float64) - cx) / fx,
            (rows.astype(np.float64) - cy) / fy,
            np.ones((height, width), dtype=np.float64),
        ),
        axis=-1,
    )


def ground_axes(up_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = np.asarray(up_camera, dtype=np.float64)
    up /= np.linalg.norm(up)
    optical_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    forward = optical_forward - float(np.dot(optical_forward, up)) * up
    require(float(np.linalg.norm(forward)) > 1e-6, "gravity/forward degeneracy")
    forward /= np.linalg.norm(forward)
    lateral = np.cross(forward, up)
    lateral /= np.linalg.norm(lateral)
    return lateral, up, forward


def plane_basis(
    intrinsics: np.ndarray,
    shape: tuple[int, int],
    up_camera: np.ndarray,
) -> np.ndarray:
    rays = ray_grid(intrinsics, shape)
    lateral, up, _ = ground_axes(up_camera)
    return np.stack(
        (
            np.ones(shape, dtype=np.float64),
            np.einsum("...j,j->...", rays, lateral),
            np.einsum("...j,j->...", rays, up),
        ),
        axis=-1,
    )


def plane_for_depth(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    up_camera: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    points, pixels = depth_to_points(depth_m, intrinsics)
    value = fit_gravity_guided_ground_plane(
        points, pixels, up_camera, depth_m.shape[0]
    )
    require(value is not None, "gravity-guided support plane unavailable")
    return value


def geometry_coordinates(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    plane: tuple[np.ndarray, float, float],
) -> dict[str, np.ndarray]:
    rays = ray_grid(intrinsics, depth_m.shape)
    points = rays * np.asarray(depth_m, dtype=np.float64)[..., None]
    up, height, _ = plane
    lateral_axis, up_axis, forward_axis = ground_axes(up)
    return {
        "lateral": np.einsum("...j,j->...", points, lateral_axis),
        "height": np.einsum("...j,j->...", points, up_axis) + float(height),
        "forward": np.einsum("...j,j->...", points, forward_axis),
    }


def soft_interval(
    values: np.ndarray,
    low: float,
    high: float,
    low_margin: float,
    high_margin: float | None = None,
) -> np.ndarray:
    if high_margin is None:
        high_margin = low_margin
    require(
        high > low and low_margin > 0.0 and high_margin > 0.0,
        "invalid soft interval",
    )
    left = np.clip((values - (low - low_margin)) / low_margin, 0.0, 1.0)
    right = np.clip(
        ((high + high_margin) - values) / high_margin, 0.0, 1.0
    )
    return np.minimum(left, right)


def query_soft_mask(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    up_camera: np.ndarray,
    band: str,
    horizon_m: float,
    representation: dict[str, Any],
) -> np.ndarray:
    plane = plane_for_depth(depth_m, intrinsics, up_camera)
    coordinates = geometry_coordinates(depth_m, intrinsics, plane)
    low, high = (float(value) for value in BANDS[band])
    lateral = soft_interval(
        coordinates["lateral"],
        low,
        high,
        float(representation["query_mask_lateral_margin_m"]),
    )
    forward = soft_interval(
        coordinates["forward"],
        0.2,
        horizon_m,
        float(representation["query_mask_near_margin_m"]),
        float(representation["query_mask_far_margin_m"]),
    )
    valid = np.isfinite(depth_m) & (depth_m >= 0.05) & (depth_m <= 20.0)
    return np.where(valid, lateral * forward, 0.0).astype(np.float64)


def support_and_evaluation_masks(
    source_depth_m: np.ndarray,
    intrinsics: np.ndarray,
    up_camera: np.ndarray,
    band: str,
    horizon_m: float,
    representation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plane = plane_for_depth(source_depth_m, intrinsics, up_camera)
    coordinates = geometry_coordinates(source_depth_m, intrinsics, plane)
    rows = np.arange(source_depth_m.shape[0])[:, None]
    valid = (
        np.isfinite(source_depth_m)
        & (source_depth_m >= 0.25)
        & (source_depth_m <= 6.0)
    )
    support = (
        valid
        & (np.abs(coordinates["height"]) <= float(representation["support_plane_tolerance_m"]))
        & (coordinates["forward"] >= float(representation["support_fit_forward_range_m"][0]))
        & (coordinates["forward"] <= float(representation["support_fit_forward_range_m"][1]))
        & (rows >= int(round(0.45 * source_depth_m.shape[0])))
    )
    low, high = (float(value) for value in BANDS[band])
    lateral_weight = soft_interval(
        coordinates["lateral"],
        low,
        high,
        float(representation["query_mask_lateral_margin_m"]),
    )
    horizon_weight = np.exp(
        -np.abs(coordinates["forward"] - horizon_m)
        / float(representation["support_fit_horizon_decay_m"])
    )
    local_weight = np.where(valid, lateral_weight * horizon_weight, 0.0)
    local_support = support & (
        local_weight
        >= float(representation["support_fit_minimum_query_weight"])
    )
    evaluation = (
        valid
        & (coordinates["height"] >= 0.08)
        & (coordinates["height"] <= 2.0)
        & (coordinates["forward"] >= 0.2)
        & (coordinates["forward"] <= horizon_m)
        & (coordinates["lateral"] >= low)
        & (coordinates["lateral"] < high)
    )
    return support, local_support, evaluation, local_weight


def bounded_sample(mask: np.ndarray, maximum: int) -> np.ndarray:
    indices = np.flatnonzero(mask.reshape(-1))
    if len(indices) <= maximum:
        return indices
    selected = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
    return indices[selected]


def fit_ridge(
    basis: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    representation: dict[str, Any],
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    full_count = int(np.sum(mask))
    require(
        full_count >= int(representation["minimum_fit_pixels"]),
        "insufficient geometry-only fit pixels",
    )
    indices = bounded_sample(mask, int(representation["maximum_fit_pixels"]))
    x = basis.reshape(-1, basis.shape[-1])[indices]
    y = target.reshape(-1)[indices]
    selected_weights = (
        np.ones(len(indices), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)[indices]
    )
    finite = (
        np.isfinite(x).all(axis=1)
        & np.isfinite(y)
        & np.isfinite(selected_weights)
        & (selected_weights > 0.0)
    )
    x = x[finite]
    y = y[finite]
    selected_weights = selected_weights[finite]
    require(
        len(y) >= int(representation["minimum_fit_pixels"]),
        "insufficient finite fit pixels",
    )
    root_weight = np.sqrt(selected_weights)[:, None]
    weighted_x = x * root_weight
    weighted_y = y * root_weight[:, 0]
    ridge = float(representation["ridge_lambda"])
    normal = weighted_x.T @ weighted_x + ridge * np.eye(
        weighted_x.shape[1], dtype=np.float64
    )
    theta = np.linalg.solve(normal, weighted_x.T @ weighted_y)
    require(bool(np.isfinite(theta).all()), "non-finite fitted theta")
    return theta, full_count, len(y)


def inverse_depth_residual(
    base_depth_m: np.ndarray, source_depth_m: np.ndarray
) -> np.ndarray:
    base = np.asarray(base_depth_m, dtype=np.float64)
    source = np.asarray(source_depth_m, dtype=np.float64)
    output = np.full(base.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(base)
        & (base > 0.0)
        & np.isfinite(source)
        & (source > 0.0)
    )
    output[valid] = 1.0 / source[valid] - 1.0 / base[valid]
    return output


def fit_global_scale(
    base_depth_m: np.ndarray,
    source_depth_m: np.ndarray,
    support_mask: np.ndarray,
    representation: dict[str, Any],
) -> float:
    indices = bounded_sample(support_mask, int(representation["maximum_fit_pixels"]))
    base = base_depth_m.reshape(-1)[indices]
    source = source_depth_m.reshape(-1)[indices]
    ratio = source / base
    ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    require(
        len(ratio) >= int(representation["minimum_fit_pixels"]),
        "insufficient scale pixels",
    )
    low, high = (float(value) for value in representation["global_scale_clip"])
    return float(np.clip(np.median(ratio), low, high))


def fit_global_inverse_affine(
    base_depth_m: np.ndarray,
    source_depth_m: np.ndarray,
    support_mask: np.ndarray,
    representation: dict[str, Any],
) -> tuple[float, float]:
    indices = bounded_sample(support_mask, int(representation["maximum_fit_pixels"]))
    rho0 = 1.0 / np.asarray(base_depth_m.reshape(-1)[indices], dtype=np.float64)
    truth = 1.0 / np.asarray(source_depth_m.reshape(-1)[indices], dtype=np.float64)
    finite = np.isfinite(rho0) & np.isfinite(truth)
    rho0 = rho0[finite]
    truth = truth[finite]
    require(
        len(truth) >= int(representation["minimum_fit_pixels"]),
        "insufficient affine pixels",
    )
    design = np.stack((rho0, np.ones_like(rho0)), axis=1)
    coefficients, _, _, _ = np.linalg.lstsq(design, truth, rcond=None)
    scale_low, scale_high = (
        float(value)
        for value in representation["global_inverse_affine_scale_clip"]
    )
    shift_low, shift_high = (
        float(value)
        for value in representation["global_inverse_affine_shift_clip_m_inverse"]
    )
    return (
        float(np.clip(coefficients[0], scale_low, scale_high)),
        float(np.clip(coefficients[1], shift_low, shift_high)),
    )


def perturb_gravity(up_camera: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(float(degrees))
    axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    value = np.asarray(up_camera, dtype=np.float64)
    rotated = (
        value * math.cos(angle)
        + np.cross(axis, value) * math.sin(angle)
        + axis * float(np.dot(axis, value)) * (1.0 - math.cos(angle))
    )
    rotated /= np.linalg.norm(rotated)
    return rotated


def perturb_intrinsics(
    intrinsics: np.ndarray, control: dict[str, Any]
) -> np.ndarray:
    value = np.asarray(intrinsics, dtype=np.float64).copy()
    value[0, 0] *= float(control["focal_scale"])
    value[1, 1] *= float(control["focal_scale"])
    value[0, 2] += float(control["principal_point_delta_px"][0])
    value[1, 2] += float(control["principal_point_delta_px"][1])
    return value


def build_frame_plan(
    row: dict[str, Any],
    base_depth_m: np.ndarray,
    source_depth_m: np.ndarray,
    up_camera: np.ndarray,
    representation: dict[str, Any],
    controls: dict[str, Any],
) -> dict[str, Any]:
    intrinsics = intrinsics_matrix(row)
    support, _, _, _ = support_and_evaluation_masks(
        source_depth_m,
        intrinsics,
        up_camera,
        "center",
        2.0,
        representation,
    )
    residual = inverse_depth_residual(base_depth_m, source_depth_m)
    basis = plane_basis(intrinsics, base_depth_m.shape, up_camera)
    global_theta, global_fit_count, global_used_count = fit_ridge(
        basis, residual, support, representation
    )
    wrong_up = perturb_gravity(up_camera, float(controls["wrong_gravity_degrees"]))
    wrong_intrinsics = perturb_intrinsics(intrinsics, controls["wrong_intrinsics"])
    query_plans: dict[str, Any] = {}
    all_overlap_counts: list[int] = []
    for band, horizon in frozen_queries():
        _, local_support, evaluation, local_weight = support_and_evaluation_masks(
            source_depth_m,
            intrinsics,
            up_camera,
            band,
            horizon,
            representation,
        )
        overlap = int(np.sum(local_support & evaluation))
        all_overlap_counts.append(overlap)
        theta, full_count, used_count = fit_ridge(
            basis, residual, local_support, representation, local_weight
        )

        wrong_gravity_theta, wrong_gravity_count, _ = fit_ridge(
            plane_basis(intrinsics, base_depth_m.shape, wrong_up),
            residual,
            local_support,
            representation,
            local_weight,
        )
        wrong_k_theta, wrong_k_count, _ = fit_ridge(
            plane_basis(wrong_intrinsics, base_depth_m.shape, up_camera),
            residual,
            local_support,
            representation,
            local_weight,
        )
        query_plans[query_key(band, horizon)] = {
            "theta": theta.tolist(),
            "fit_pixel_count": full_count,
            "used_fit_pixel_count": used_count,
            "fit_evaluation_overlap_pixel_count": overlap,
            "wrong_gravity_theta": wrong_gravity_theta.tolist(),
            "wrong_gravity_fit_pixel_count": wrong_gravity_count,
            "wrong_k_theta": wrong_k_theta.tolist(),
            "wrong_k_fit_pixel_count": wrong_k_count,
        }
    require(max(all_overlap_counts) == 0, "fit/evaluation pixel leakage")
    return {
        "index": int(row["index"]),
        "frame_id": str(row["frame_id"]),
        "parent_id": str(row["sequence_id"]),
        "sequence_root": str(row["sequence_root"]),
        "timestamp": float(row["timestamp"]),
        "global_scale": fit_global_scale(
            base_depth_m, source_depth_m, support, representation
        ),
        "global_inverse_affine": list(
            fit_global_inverse_affine(
                base_depth_m, source_depth_m, support, representation
            )
        ),
        "global_ray_plane_theta": global_theta.tolist(),
        "global_fit_pixel_count": global_fit_count,
        "global_used_fit_pixel_count": global_used_count,
        "query_plans": query_plans,
    }


def apply_scale(
    base_depth_m: np.ndarray, scale: float, representation: dict[str, Any]
) -> np.ndarray:
    low, high = (float(value) for value in representation["corrected_depth_clip_m"])
    return np.clip(np.asarray(base_depth_m, dtype=np.float64) * scale, low, high)


def apply_inverse_affine(
    base_depth_m: np.ndarray,
    coefficients: Iterable[float],
    representation: dict[str, Any],
) -> np.ndarray:
    scale, shift = (float(value) for value in coefficients)
    rho = scale / np.asarray(base_depth_m, dtype=np.float64) + shift
    low, high = (float(value) for value in representation["corrected_depth_clip_m"])
    rho = np.clip(rho, 1.0 / high, 1.0 / low)
    return 1.0 / rho


def apply_plane_residual(
    base_depth_m: np.ndarray,
    intrinsics: np.ndarray,
    up_camera: np.ndarray,
    theta: Iterable[float],
    mask: np.ndarray,
    representation: dict[str, Any],
) -> np.ndarray:
    basis = plane_basis(intrinsics, base_depth_m.shape, up_camera)
    correction = np.einsum(
        "...j,j->...", basis, np.asarray(list(theta), dtype=np.float64)
    )
    epsilon = float(representation["epsilon_rho_m_inverse"])
    correction = np.clip(correction, -epsilon, epsilon)
    rho = 1.0 / np.asarray(base_depth_m, dtype=np.float64) + mask * correction
    low, high = (float(value) for value in representation["corrected_depth_clip_m"])
    rho = np.clip(rho, 1.0 / high, 1.0 / low)
    return 1.0 / rho


def extract_query(field: dict[str, Any], band: str, horizon: float) -> dict[str, Any]:
    if field.get("status") != "VALID":
        return {"known": False, "occupied": None, "clearance_m": None}
    band_field = field["bands"][band]
    occupied = band_field["occupied_by_horizon"].get(str(horizon))
    clearance = band_field.get("clearance_m")
    return {
        "known": occupied is not None,
        "occupied": bool(occupied) if occupied is not None else None,
        "clearance_m": float(clearance) if clearance is not None else None,
    }


def evaluate_frame(
    row: dict[str, Any],
    plan: dict[str, Any],
    base_depth_m: np.ndarray,
    source_depth_m: np.ndarray,
    up_camera: np.ndarray,
    representation: dict[str, Any],
    controls: dict[str, Any],
) -> list[dict[str, Any]]:
    intrinsics = intrinsics_matrix(row)
    wrong_up = perturb_gravity(up_camera, float(controls["wrong_gravity_degrees"]))
    wrong_intrinsics = perturb_intrinsics(intrinsics, controls["wrong_intrinsics"])
    global_fields = {
        "A0_FROZEN_DEPTHART": clearance_field(
            base_depth_m, intrinsics, up_camera=up_camera
        ),
        "A1_GLOBAL_SCALE": clearance_field(
            apply_scale(base_depth_m, float(plan["global_scale"]), representation),
            intrinsics,
            up_camera=up_camera,
        ),
        "A2_GLOBAL_AFFINE": clearance_field(
            apply_inverse_affine(
                base_depth_m, plan["global_inverse_affine"], representation
            ),
            intrinsics,
            up_camera=up_camera,
        ),
        "A3_GLOBAL_RAY_PLANE": clearance_field(
            apply_plane_residual(
                base_depth_m,
                intrinsics,
                up_camera,
                plan["global_ray_plane_theta"],
                np.ones(base_depth_m.shape, dtype=np.float64),
                representation,
            ),
            intrinsics,
            up_camera=up_camera,
        ),
        "A5_SOURCE_DEPTH_ORACLE": clearance_field(
            source_depth_m, intrinsics, up_camera=up_camera
        ),
    }
    queries = frozen_queries()
    shuffled_keys = [query_key(*value) for value in queries[1:] + queries[:1]]
    records: list[dict[str, Any]] = []
    for query_index, (band, horizon) in enumerate(queries):
        key = query_key(band, horizon)
        query_plan = plan["query_plans"][key]
        mask = query_soft_mask(
            base_depth_m,
            intrinsics,
            up_camera,
            band,
            horizon,
            representation,
        )
        qplane_depth = apply_plane_residual(
            base_depth_m,
            intrinsics,
            up_camera,
            query_plan["theta"],
            mask,
            representation,
        )
        qplane_field = clearance_field(
            qplane_depth, intrinsics, up_camera=up_camera
        )

        shuffled_plan = plan["query_plans"][shuffled_keys[query_index]]
        shuffled_depth = apply_plane_residual(
            base_depth_m,
            intrinsics,
            up_camera,
            shuffled_plan["theta"],
            mask,
            representation,
        )
        shuffled_field = clearance_field(
            shuffled_depth, intrinsics, up_camera=up_camera
        )

        wrong_gravity_mask = query_soft_mask(
            base_depth_m,
            intrinsics,
            wrong_up,
            band,
            horizon,
            representation,
        )
        wrong_gravity_depth = apply_plane_residual(
            base_depth_m,
            intrinsics,
            wrong_up,
            query_plan["wrong_gravity_theta"],
            wrong_gravity_mask,
            representation,
        )
        wrong_gravity_field = clearance_field(
            wrong_gravity_depth, intrinsics, up_camera=up_camera
        )

        wrong_k_mask = query_soft_mask(
            base_depth_m,
            wrong_intrinsics,
            up_camera,
            band,
            horizon,
            representation,
        )
        wrong_k_depth = apply_plane_residual(
            base_depth_m,
            wrong_intrinsics,
            up_camera,
            query_plan["wrong_k_theta"],
            wrong_k_mask,
            representation,
        )
        wrong_k_field = clearance_field(
            wrong_k_depth, intrinsics, up_camera=up_camera
        )
        arm_values = {
            name: extract_query(field, band, horizon)
            for name, field in global_fields.items()
        }
        arm_values.update(
            {
                "A4_QUERY_LOCAL_RAY_PLANE": extract_query(
                    qplane_field, band, horizon
                ),
                "NC_SHUFFLED_QUERY": extract_query(
                    shuffled_field, band, horizon
                ),
                "NC_WRONG_GRAVITY": extract_query(
                    wrong_gravity_field, band, horizon
                ),
                "NC_WRONG_K": extract_query(wrong_k_field, band, horizon),
            }
        )
        records.append(
            {
                "frame_id": str(row["frame_id"]),
                "parent_id": str(row["sequence_id"]),
                "band": band,
                "horizon_m": horizon,
                "arms": arm_values,
            }
        )
    return records


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_records(
    records: list[dict[str, Any]], arm: str
) -> dict[str, Any]:
    truth_known = 0
    known = 0
    false_block = 0
    false_clear = 0
    truth_clear = 0
    truth_occupied = 0
    clearance_errors: list[float] = []
    clearance_biases: list[float] = []
    for row in records:
        truth = row["arms"]["A5_SOURCE_DEPTH_ORACLE"]
        candidate = row["arms"][arm]
        if not truth["known"]:
            continue
        truth_known += 1
        if not candidate["known"]:
            continue
        known += 1
        truth_state = bool(truth["occupied"])
        candidate_state = bool(candidate["occupied"])
        truth_clear += int(not truth_state)
        truth_occupied += int(truth_state)
        false_block += int((not truth_state) and candidate_state)
        false_clear += int(truth_state and (not candidate_state))
        if truth["clearance_m"] is not None and candidate["clearance_m"] is not None:
            delta = float(candidate["clearance_m"]) - float(truth["clearance_m"])
            clearance_biases.append(delta)
            clearance_errors.append(abs(delta))
    return {
        "truth_known_decisions": truth_known,
        "known_decisions": known,
        "coverage": safe_rate(known, truth_known),
        "known_clearance_pairs": len(clearance_errors),
        "clearance_mae_m": (
            float(np.mean(clearance_errors)) if clearance_errors else None
        ),
        "clearance_bias_m": (
            float(np.mean(clearance_biases)) if clearance_biases else None
        ),
        "false_block_count": false_block,
        "false_block_rate_all_known": safe_rate(false_block, known),
        "false_block_rate_given_clear": safe_rate(false_block, truth_clear),
        "false_clear_count": false_clear,
        "false_clear_rate_all_known": safe_rate(false_clear, known),
        "false_clear_rate_given_occupied": safe_rate(
            false_clear, truth_occupied
        ),
    }


def grouped_records(
    records: list[dict[str, Any]], field: str
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        result[str(row[field])].append(row)
    return dict(sorted(result.items()))


def macro(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def summarize_all(records: list[dict[str, Any]]) -> dict[str, Any]:
    parents = grouped_records(records, "parent_id")
    bands = grouped_records(records, "band")
    horizons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        horizons[f"{float(row['horizon_m']):.1f}m"].append(row)
    output: dict[str, Any] = {
        "overall": {},
        "per_parent": {},
        "parent_macro": {},
        "per_band": {},
        "per_horizon": {},
    }
    for arm in ARMS:
        output["overall"][arm] = summarize_records(records, arm)
        output["per_parent"][arm] = {
            parent: summarize_records(rows, arm)
            for parent, rows in parents.items()
        }
        output["parent_macro"][arm] = {
            metric: macro(
                summary.get(metric)
                for summary in output["per_parent"][arm].values()
            )
            for metric in (
                "coverage",
                "clearance_mae_m",
                "clearance_bias_m",
                "false_block_rate_all_known",
                "false_clear_rate_all_known",
            )
        }
        output["per_band"][arm] = {
            band: summarize_records(rows, arm) for band, rows in bands.items()
        }
        output["per_horizon"][arm] = {
            horizon: summarize_records(rows, arm)
            for horizon, rows in sorted(horizons.items())
        }
    return output


def finite_float(value: Any) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        "required finite metric missing",
    )
    return float(value)


def decide_gates(
    metrics: dict[str, Any],
    isolation: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    macro_metrics = metrics["parent_macro"]
    primary = {
        arm: finite_float(macro_metrics[arm]["clearance_mae_m"])
        for arm in (
            "A1_GLOBAL_SCALE",
            "A2_GLOBAL_AFFINE",
            "A3_GLOBAL_RAY_PLANE",
            "A4_QUERY_LOCAL_RAY_PLANE",
            "A5_SOURCE_DEPTH_ORACLE",
            "NC_SHUFFLED_QUERY",
        )
    }
    a3 = primary["A3_GLOBAL_RAY_PLANE"]
    a4 = primary["A4_QUERY_LOCAL_RAY_PLANE"]
    a5 = primary["A5_SOURCE_DEPTH_ORACLE"]
    gap_denominator = a3 - a5
    gap_closure = (a3 - a4) / gap_denominator if gap_denominator > 0.0 else None
    parents_a3 = metrics["per_parent"]["A3_GLOBAL_RAY_PLANE"]
    parents_a4 = metrics["per_parent"]["A4_QUERY_LOCAL_RAY_PLANE"]
    false_block_improvements = sum(
        finite_float(parents_a4[parent]["false_block_rate_all_known"])
        < finite_float(parents_a3[parent]["false_block_rate_all_known"])
        for parent in parents_a3
    )
    false_clear_noninferior = all(
        finite_float(parents_a4[parent]["false_clear_rate_all_known"])
        <= finite_float(parents_a3[parent]["false_clear_rate_all_known"])
        + float(gates["per_parent_false_clear_rate_all_known_increase_max"])
        for parent in parents_a3
    )
    a0_parents = metrics["per_parent"]["A0_FROZEN_DEPTHART"]
    coverage_equal = all(
        parents_a4[parent]["known_decisions"]
        == a0_parents[parent]["known_decisions"]
        and parents_a4[parent]["truth_known_decisions"]
        == a0_parents[parent]["truth_known_decisions"]
        for parent in parents_a3
    )
    band_improvements = sum(
        finite_float(value["clearance_mae_m"])
        < finite_float(metrics["per_band"]["A3_GLOBAL_RAY_PLANE"][band]["clearance_mae_m"])
        for band, value in metrics["per_band"]["A4_QUERY_LOCAL_RAY_PLANE"].items()
    )
    horizon_improvements = sum(
        finite_float(value["clearance_mae_m"])
        < finite_float(metrics["per_horizon"]["A3_GLOBAL_RAY_PLANE"][horizon]["clearance_mae_m"])
        for horizon, value in metrics["per_horizon"]["A4_QUERY_LOCAL_RAY_PLANE"].items()
    )
    checks = {
        "a4_strictly_better_than_a1_a2_a3_primary_metric": all(
            a4 < primary[arm]
            for arm in (
                "A1_GLOBAL_SCALE",
                "A2_GLOBAL_AFFINE",
                "A3_GLOBAL_RAY_PLANE",
            )
        ),
        "minimum_parents_with_false_block_improvement_vs_a3": (
            false_block_improvements
            >= int(gates["minimum_parents_with_false_block_improvement_vs_a3"])
        ),
        "per_parent_false_clear_noninferiority_vs_a3": false_clear_noninferior,
        "coverage_exactly_equal_a0": coverage_equal,
        "minimum_global_plane_to_source_oracle_gap_closure": (
            gap_closure is not None
            and gap_closure
            >= float(gates["minimum_global_plane_to_source_oracle_gap_closure"])
        ),
        "minimum_advantage_over_shuffled_query": (
            primary["NC_SHUFFLED_QUERY"] - a4
            >= float(gates["minimum_a4_clearance_mae_advantage_over_shuffled_query_m"])
        ),
        "minimum_bands_with_improvement_vs_a3": (
            band_improvements
            >= int(gates["minimum_bands_with_clearance_mae_improvement_vs_a3"])
        ),
        "minimum_horizons_with_improvement_vs_a3": (
            horizon_improvements
            >= int(gates["minimum_horizons_with_clearance_mae_improvement_vs_a3"])
        ),
        "overall_false_block_improves_vs_a3": (
            finite_float(
                metrics["overall"]["A4_QUERY_LOCAL_RAY_PLANE"]["false_block_rate_all_known"]
            )
            < finite_float(
                metrics["overall"]["A3_GLOBAL_RAY_PLANE"]["false_block_rate_all_known"]
            )
        ),
        "all_fit_evaluation_overlap_counts_zero": (
            int(isolation["maximum_fit_evaluation_overlap_pixel_count"]) == 0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostics": {
            "primary_parent_macro_clearance_mae_m": primary,
            "global_plane_to_source_oracle_gap_closure": gap_closure,
            "parents_with_false_block_improvement_vs_a3": false_block_improvements,
            "bands_with_clearance_improvement_vs_a3": band_improvements,
            "horizons_with_clearance_improvement_vs_a3": horizon_improvements,
        },
    }


def verify_inputs(
    protocol_path: Path,
    protocol: dict[str, Any],
    roster_path: Path,
    source_root: Path,
    depthart_cache_path: Path,
) -> dict[str, Any]:
    require(
        protocol.get("schema") == "blindassist_ba_clear_qplane_o0a_protocol_v1"
        and protocol.get("status") == "FROZEN_REPRESENTATION_AUDIT_ONLY",
        "Q-Plane O0-A protocol not frozen",
    )
    require(
        not protocol["authority"]["training_authorized"]
        and not protocol["authority"]["fresh_outcome_authorized"]
        and not protocol["authority"]["android_qnn_htp_authorized"],
        "O0-A authority expanded",
    )
    data = protocol["data"]
    roster_hash = sha256_file(roster_path)
    cache_hash = sha256_file(depthart_cache_path)
    require(roster_hash == data["roster_sha256"], "roster SHA drift")
    require(cache_hash == data["depthart_cache_sha256"], "DepthART cache SHA drift")
    roster = read_json(roster_path)
    rows = roster.get("rows")
    require(
        isinstance(rows, list) and len(rows) == int(data["frame_count"]),
        "roster frame count drift",
    )
    require(
        len({str(row["sequence_id"]) for row in rows}) == int(data["parent_count"]),
        "roster parent count drift",
    )
    for sequence, expected_hash in data["groundtruth_sha256"].items():
        path = source_root / sequence / "groundtruth.txt"
        require(path.is_file(), f"missing groundtruth: {sequence}")
        require(sha256_file(path) == expected_hash, f"groundtruth SHA drift: {sequence}")
    cache = np.load(depthart_cache_path, mmap_mode="r")
    require(
        list(cache.shape) == data["depthart_cache_shape"]
        and str(cache.dtype) == data["depthart_cache_dtype"],
        "DepthART cache tensor drift",
    )
    return {
        "protocol_sha256": sha256_file(protocol_path),
        "roster_sha256": roster_hash,
        "depthart_cache_sha256": cache_hash,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "frame_count": len(rows),
        "parent_count": len({str(row["sequence_id"]) for row in rows}),
    }


def load_source_depth(source_root: Path, row: dict[str, Any]) -> np.ndarray:
    path = source_root / str(row["sequence_root"]) / str(row["depth_path"])
    require(sha256_file(path) == str(row["depth_sha256"]), "source depth SHA drift")
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    require(raw is not None, f"source depth unreadable: {path}")
    return tum_depth_metres(normalize_depth_image(raw, path))


def pose_for_row(
    source_root: Path,
    row: dict[str, Any],
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    maximum_delta_seconds: float,
) -> tuple[np.ndarray, float]:
    sequence = str(row["sequence_root"])
    if sequence not in cache:
        cache[sequence] = load_tum_camera_poses(
            source_root / sequence / "groundtruth.txt"
        )
    timestamps, _, quaternions = cache[sequence]
    timestamp = float(row["timestamp"])
    index = int(np.argmin(np.abs(timestamps - timestamp)))
    delta_seconds = abs(float(timestamps[index]) - timestamp)
    require(
        delta_seconds <= maximum_delta_seconds,
        f"pose unavailable within frozen delta: {row['frame_id']}",
    )
    camera_to_world = quaternion_rotation_matrix_xyzw(quaternions[index])
    up_camera = camera_to_world.T @ np.asarray([0.0, 0.0, 1.0])
    return up_camera, delta_seconds


def run(
    protocol_path: Path,
    roster_path: Path,
    source_root: Path,
    depthart_cache_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    require(not output_dir.exists(), f"output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    protocol = read_json(protocol_path)
    inputs = verify_inputs(
        protocol_path,
        protocol,
        roster_path,
        source_root,
        depthart_cache_path,
    )
    roster = read_json(roster_path)
    base_cache = np.load(depthart_cache_path, mmap_mode="r")
    representation = protocol["representation"]
    controls = protocol["negative_controls"]
    maximum_pose_delta = float(
        protocol["data"]["pose_maximum_association_delta_seconds"]
    )
    poses: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    phase_a_started = time.perf_counter()
    frame_plans: list[dict[str, Any]] = []
    for index, row in enumerate(roster["rows"]):
        require(int(row["index"]) == index, "roster index drift")
        source_depth = load_source_depth(source_root, row)
        up_camera, pose_delta_seconds = pose_for_row(
            source_root, row, poses, maximum_pose_delta
        )
        frame_plan = build_frame_plan(
            row,
            np.asarray(base_cache[index], dtype=np.float64),
            source_depth,
            up_camera,
            representation,
            controls,
        )
        frame_plan["pose_association_delta_seconds"] = pose_delta_seconds
        frame_plans.append(frame_plan)
    candidate_plan = {
        "schema": "blindassist_ba_clear_qplane_o0a_candidate_plan_v1",
        "status": "CANDIDATE_PARAMETERS_FROZEN_BEFORE_TASK_OUTCOMES",
        "protocol_sha256": inputs["protocol_sha256"],
        "fit_target": protocol["fit_evaluation_isolation"]["fit_target"],
        "clearance_reducer_called": False,
        "task_outcomes_derived": False,
        "unified_corrected_dense_depth_persisted": False,
        "frames": frame_plans,
    }
    candidate_plan_path = output_dir / "candidate-plan.json"
    write_json(candidate_plan_path, candidate_plan)
    candidate_plan_hash = sha256_file(candidate_plan_path)
    phase_a_seconds = time.perf_counter() - phase_a_started

    frozen_plan = read_json(candidate_plan_path)
    require(
        frozen_plan.get("status")
        == "CANDIDATE_PARAMETERS_FROZEN_BEFORE_TASK_OUTCOMES"
        and not frozen_plan.get("clearance_reducer_called")
        and not frozen_plan.get("task_outcomes_derived"),
        "candidate plan firewall drift",
    )
    phase_b_started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for index, row in enumerate(roster["rows"]):
        source_depth = load_source_depth(source_root, row)
        up_camera, pose_delta_seconds = pose_for_row(
            source_root, row, poses, maximum_pose_delta
        )
        require(
            abs(
                pose_delta_seconds
                - float(frozen_plan["frames"][index]["pose_association_delta_seconds"])
            )
            <= 1e-12,
            "pose association drift after candidate freeze",
        )
        records.extend(
            evaluate_frame(
                row,
                frozen_plan["frames"][index],
                np.asarray(base_cache[index], dtype=np.float64),
                source_depth,
                up_camera,
                representation,
                controls,
            )
        )
    metrics = summarize_all(records)
    overlap_counts = [
        int(query["fit_evaluation_overlap_pixel_count"])
        for frame in frame_plans
        for query in frame["query_plans"].values()
    ]
    fit_counts = [
        int(query["fit_pixel_count"])
        for frame in frame_plans
        for query in frame["query_plans"].values()
    ]
    isolation = {
        "candidate_plan_sha256_before_task_evaluation": candidate_plan_hash,
        "geometry_only_candidate_frame_count": len(frame_plans),
        "query_parameter_vector_count": len(fit_counts),
        "theta_dimensions_per_query": int(representation["theta_dimensions_per_query"]),
        "query_local_degrees_of_freedom_per_frame": (
            len(frozen_queries()) * int(representation["theta_dimensions_per_query"])
        ),
        "minimum_fit_pixel_count": min(fit_counts),
        "maximum_fit_pixel_count": max(fit_counts),
        "maximum_fit_evaluation_overlap_pixel_count": max(overlap_counts),
        "maximum_pose_association_delta_seconds": max(
            float(frame["pose_association_delta_seconds"])
            for frame in frame_plans
        ),
        "task_label_optimization_used": False,
        "unified_corrected_dense_depth_persisted": False,
    }
    gate = decide_gates(metrics, isolation, protocol["gates"])
    phase_b_seconds = time.perf_counter() - phase_b_started
    passed = bool(gate["passed"])
    result = {
        "schema": "blindassist_ba_clear_qplane_o0a_result_v1",
        "status": (
            "BA_CLEAR_QPLANE_O0A_REPRESENTATION_HEADROOM_PASS"
            if passed
            else "BA_CLEAR_QPLANE_O0A_REPRESENTATION_HEADROOM_FAIL_CLOSE_NO_TRAINING"
        ),
        "passed": passed,
        "question": protocol["question"],
        "inputs": inputs,
        "candidate_plan": {
            "path": str(candidate_plan_path.resolve()),
            "sha256": candidate_plan_hash,
            "status": frozen_plan["status"],
        },
        "representation": {
            "name": "BODY_QUERY_CONDITIONED_INVERSE_DEPTH_RAY_PLANE_RESIDUAL",
            "formula": protocol["representation"]["query_inverse_depth"],
            "query_count": len(frozen_queries()),
            "theta_dimensions_per_query": int(representation["theta_dimensions_per_query"]),
            "epsilon_rho_m_inverse": float(representation["epsilon_rho_m_inverse"]),
        },
        "metrics": metrics,
        "paired_parent_deltas_a4_minus_a3": {
            parent: {
                metric: (
                    finite_float(
                        metrics["per_parent"]["A4_QUERY_LOCAL_RAY_PLANE"][parent][metric]
                    )
                    - finite_float(
                        metrics["per_parent"]["A3_GLOBAL_RAY_PLANE"][parent][metric]
                    )
                )
                for metric in (
                    "clearance_mae_m",
                    "clearance_bias_m",
                    "false_block_rate_all_known",
                    "false_clear_rate_all_known",
                    "coverage",
                )
            }
            for parent in metrics["per_parent"]["A3_GLOBAL_RAY_PLANE"]
        },
        "negative_controls": {
            "shuffled_query": metrics["parent_macro"]["NC_SHUFFLED_QUERY"],
            "wrong_gravity": metrics["parent_macro"]["NC_WRONG_GRAVITY"],
            "wrong_intrinsics": metrics["parent_macro"]["NC_WRONG_K"],
            "globalized_qplane": {
                "alias": "A3_GLOBAL_RAY_PLANE",
                "metrics": metrics["parent_macro"]["A3_GLOBAL_RAY_PLANE"],
            },
        },
        "fit_evaluation_isolation": isolation,
        "gate": gate,
        "runtime_diagnostic": {
            "device": "CPU",
            "phase_a_geometry_fit_seconds": phase_a_seconds,
            "phase_b_task_evaluation_seconds": phase_b_seconds,
            "total_seconds": time.perf_counter() - started,
            "frame_count": len(frame_plans),
            "claim_ceiling": "HOST_CPU_DIAGNOSTIC_ONLY",
        },
        "authority": {
            "training_authorized": False,
            "fresh_outcome_read": False,
            "android_qnn_htp_authorized": False,
            "default_app_changed": False,
        },
        "decision": {
            "next_successor": (
                protocol["decision"]["pass_successor"]
                if passed
                else protocol["decision"]["fail_successor"]
            ),
            "automatic_successor_execution_authorized": False,
        },
        "claim_ceiling": protocol["claim_ceiling"],
    }
    write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    data = protocol["data"]
    result = run(
        protocol_path,
        (REPO_ROOT / data["roster"]).resolve(),
        (REPO_ROOT / data["source_root"]).resolve(),
        (REPO_ROOT / data["depthart_cache"]).resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "gate": result["gate"],
                "runtime_diagnostic": result["runtime_diagnostic"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
