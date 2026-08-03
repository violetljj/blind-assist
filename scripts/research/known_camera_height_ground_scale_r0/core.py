"""Scale relative depth from an independently known camera height.

This module deliberately contains no dataset or outcome reader.  It implements the
frozen, scale-equivariant geometry operator and fail-closed admission only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LOWER_ROI_START_FRACTION = 0.55
RANSAC_SEED = 1729
RANSAC_ITERATIONS = 240
MAXIMUM_CANDIDATES = 5000
MINIMUM_CANDIDATES = 100
MINIMUM_INLIERS = 80
MINIMUM_INLIER_FRACTION = 0.08
MINIMUM_ABS_NORMAL_Y = 0.55
MAXIMUM_NORMALIZED_PLANE_RESIDUAL = 0.035
CAMERA_HEIGHT_RANGE_M = (0.80, 2.20)
MAXIMUM_CAMERA_HEIGHT_UNCERTAINTY_M = 0.05
SCALE_RANGE = (0.25, 4.0)
TEMPORAL_SCALE_WINDOW = 9


@dataclass(frozen=True)
class CameraHeightReceipt:
    camera_profile_id: str
    mount_profile_id: str
    height_m: float
    uncertainty_m: float


@dataclass(frozen=True)
class RelativeGroundPlane:
    normal: np.ndarray
    relative_height: float
    normalized_median_residual: float
    candidate_count: int
    inlier_count: int
    inlier_fraction: float


def _unknown(reason: str, **diagnostics: object) -> dict[str, object]:
    return {"status": "UNKNOWN", "reason": reason, **diagnostics}


def validate_intrinsics(intrinsics: np.ndarray, width: int, height: int) -> bool:
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return False
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    return (
        fx > 0.0
        and fy > 0.0
        and 0.0 <= cx < width
        and 0.0 <= cy < height
        and np.allclose(matrix[2], [0.0, 0.0, 1.0])
    )


def validate_height_receipt(receipt: CameraHeightReceipt) -> bool:
    return (
        bool(receipt.camera_profile_id.strip())
        and bool(receipt.mount_profile_id.strip())
        and np.isfinite(receipt.height_m)
        and CAMERA_HEIGHT_RANGE_M[0] <= receipt.height_m <= CAMERA_HEIGHT_RANGE_M[1]
        and np.isfinite(receipt.uncertainty_m)
        and 0.0 <= receipt.uncertainty_m <= MAXIMUM_CAMERA_HEIGHT_UNCERTAINTY_M
    )


def causal_median_scale(
    valid_scale_history: list[float] | tuple[float, ...],
    window: int = TEMPORAL_SCALE_WINDOW,
) -> float:
    """Return a causal robust scale using only the current and prior valid values."""
    if window <= 0:
        raise ValueError("window must be positive")
    values = np.asarray(valid_scale_history, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("valid_scale_history must be a non-empty 1D sequence")
    values = values[-window:]
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("scale history must contain only finite positive values")
    return float(np.median(values))


def relative_depth_to_points(
    depth: np.ndarray, intrinsics: np.ndarray, stride: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(depth, dtype=np.float64)
    if values.ndim != 2 or stride <= 0:
        raise ValueError("depth must be a 2D array and stride must be positive")
    height, width = values.shape
    if not validate_intrinsics(intrinsics, width, height):
        raise ValueError("invalid intrinsics")
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    z = values[::stride, ::stride]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    x = (columns - cx) * z / fx
    y = (rows - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    pixels = np.stack((columns, rows), axis=-1).reshape(-1, 2)
    valid = np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0.0)
    return points[valid], pixels[valid]


def fit_relative_ground_plane(
    points: np.ndarray,
    pixels: np.ndarray,
    image_height: int,
    seed: int = RANSAC_SEED,
) -> tuple[RelativeGroundPlane | None, str | None]:
    point_values = np.asarray(points, dtype=np.float64)
    pixel_values = np.asarray(pixels, dtype=np.float64)
    candidates = point_values[
        pixel_values[:, 1] >= LOWER_ROI_START_FRACTION * image_height
    ]
    if len(candidates) < MINIMUM_CANDIDATES:
        return None, "INSUFFICIENT_GROUND_CANDIDATES"
    if len(candidates) > MAXIMUM_CANDIDATES:
        indices = np.linspace(
            0, len(candidates) - 1, MAXIMUM_CANDIDATES, dtype=int
        )
        candidates = candidates[indices]

    characteristic = float(np.median(np.linalg.norm(candidates, axis=1)))
    if not np.isfinite(characteristic) or characteristic <= 0.0:
        return None, "DEGENERATE_RELATIVE_DEPTH"
    minimum_height = max(np.finfo(np.float64).eps, characteristic * 1e-6)
    minimum_cross_norm = max(
        np.finfo(np.float64).eps, characteristic * characteristic * 1e-12
    )

    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_score: tuple[int, float] | None = None
    for _ in range(RANSAC_ITERATIONS):
        sample = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= minimum_cross_norm:
            continue
        normal /= norm
        if abs(float(normal[1])) < MINIMUM_ABS_NORMAL_Y:
            continue
        offset = -float(np.dot(normal, sample[0]))
        if offset < 0.0:
            normal = -normal
            offset = -offset
        if not np.isfinite(offset) or offset <= minimum_height:
            continue
        normalized = np.abs(candidates @ normal + offset) / offset
        inliers = normalized <= MAXIMUM_NORMALIZED_PLANE_RESIDUAL
        count = int(np.sum(inliers))
        residual = float(np.median(normalized[inliers])) if count else float("inf")
        score = (count, -residual)
        if best_score is None or score > best_score:
            best_score = score
            best_inliers = inliers

    required = max(MINIMUM_INLIERS, int(np.ceil(MINIMUM_INLIER_FRACTION * len(candidates))))
    if best_inliers is None or int(np.sum(best_inliers)) < required:
        return None, "NO_GROUND_CONSENSUS"

    ground = candidates[best_inliers]
    center = np.mean(ground, axis=0)
    _, _, right_vectors = np.linalg.svd(ground - center, full_matrices=False)
    normal = right_vectors[-1]
    offset = -float(np.dot(normal, center))
    if offset < 0.0:
        normal = -normal
        offset = -offset
    if not np.isfinite(offset) or offset <= minimum_height:
        return None, "DEGENERATE_RELATIVE_HEIGHT"
    if abs(float(normal[1])) < MINIMUM_ABS_NORMAL_Y:
        return None, "GROUND_ORIENTATION_REJECTED"

    normalized = np.abs(candidates @ normal + offset) / offset
    inliers = normalized <= MAXIMUM_NORMALIZED_PLANE_RESIDUAL
    count = int(np.sum(inliers))
    fraction = count / len(candidates)
    if count < required or fraction < MINIMUM_INLIER_FRACTION:
        return None, "GROUND_SUPPORT_REJECTED"
    median_residual = float(np.median(normalized[inliers]))
    if median_residual > MAXIMUM_NORMALIZED_PLANE_RESIDUAL:
        return None, "GROUND_RESIDUAL_REJECTED"
    return (
        RelativeGroundPlane(
            normal=normal,
            relative_height=offset,
            normalized_median_residual=median_residual,
            candidate_count=len(candidates),
            inlier_count=count,
            inlier_fraction=fraction,
        ),
        None,
    )


def recover_metric_scale(
    relative_depth: np.ndarray,
    intrinsics: np.ndarray,
    height_receipt: CameraHeightReceipt,
    expected_camera_profile_id: str,
    expected_mount_profile_id: str,
    stride: int = 4,
) -> dict[str, object]:
    depth = np.asarray(relative_depth, dtype=np.float64)
    if depth.ndim != 2:
        return _unknown("INVALID_DEPTH_SHAPE")
    height, width = depth.shape
    if not validate_intrinsics(intrinsics, width, height):
        return _unknown("INVALID_INTRINSICS")
    if not validate_height_receipt(height_receipt):
        return _unknown("INVALID_HEIGHT_RECEIPT")
    if (
        height_receipt.camera_profile_id != expected_camera_profile_id
        or height_receipt.mount_profile_id != expected_mount_profile_id
    ):
        return _unknown("HEIGHT_PROFILE_IDENTITY_MISMATCH")

    points, pixels = relative_depth_to_points(depth, intrinsics, stride=stride)
    plane, reason = fit_relative_ground_plane(points, pixels, height)
    if plane is None:
        return _unknown(reason or "UNKNOWN_GROUND")

    scale = height_receipt.height_m / plane.relative_height
    if not np.isfinite(scale) or not SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]:
        return _unknown(
            "SCALE_OUT_OF_RANGE",
            scale=float(scale),
            relative_height=plane.relative_height,
        )
    lower_scale = (height_receipt.height_m - height_receipt.uncertainty_m) / plane.relative_height
    upper_scale = (height_receipt.height_m + height_receipt.uncertainty_m) / plane.relative_height
    return {
        "status": "VALID",
        "scale": float(scale),
        "scale_interval": [float(lower_scale), float(upper_scale)],
        "metric_depth": depth * scale,
        "ground": plane,
        "camera_height_m": height_receipt.height_m,
        "camera_height_uncertainty_m": height_receipt.uncertainty_m,
    }
