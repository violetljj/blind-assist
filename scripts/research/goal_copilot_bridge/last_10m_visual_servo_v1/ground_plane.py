"""Deterministic current-frame ground-plane estimation from metric depth."""

from __future__ import annotations

import numpy as np


def depth_points(depth: np.ndarray, horizontal_fov_degrees: float = 90.0) -> np.ndarray:
    """Return camera-frame XYZ, with +Y down and +Z forward."""
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")
    height, width = depth.shape
    focal = width / (2.0 * np.tan(np.deg2rad(horizontal_fov_degrees) / 2.0))
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    x = (xx - (width - 1) / 2.0) * depth / focal
    y = (yy - (height - 1) / 2.0) * depth / focal
    return np.stack((x, y, depth), axis=-1)


def estimate_ground_plane(
    depth: np.ndarray,
    *,
    horizontal_fov_degrees: float = 90.0,
    valid_range_m: tuple[float, float] = (0.4, 8.0),
    residual_threshold_m: float = 0.06,
    iterations: int = 512,
    random_seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Fit a near-horizontal plane using deterministic constrained RANSAC."""
    points_image = depth_points(depth, horizontal_fov_degrees)
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth >= valid_range_m[0]) & (depth <= valid_range_m[1])
    sample_region = np.zeros_like(valid)
    sample_region[int(height * 0.55) :, int(width * 0.10) : int(width * 0.90)] = True
    sample_mask = valid & sample_region
    points = points_image[sample_mask][::4]
    if len(points) < 64:
        raise ValueError("insufficient public depth for ground-plane fit")
    rng = np.random.default_rng(random_seed)
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    best_count = -1
    for _ in range(iterations):
        triple = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(triple[1] - triple[0], triple[2] - triple[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            continue
        normal /= norm
        if normal[1] < 0:
            normal = -normal
        if normal[1] < 0.85:
            continue
        offset = -float(np.dot(normal, triple[0]))
        count = int((np.abs(points @ normal + offset) <= residual_threshold_m).sum())
        if count > best_count:
            best_normal, best_offset, best_count = normal, offset, count
    if best_normal is None or best_count < 32:
        raise ValueError("no supported near-horizontal ground plane")
    inliers = points[np.abs(points @ best_normal + best_offset) <= residual_threshold_m]
    center = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vh[-1]
    if normal[1] < 0:
        normal = -normal
    if normal[1] < 0.85:
        normal = best_normal
        offset = best_offset
    else:
        offset = -float(np.dot(normal, center))
    return normal.astype(np.float32), float(offset)


def ground_mask_from_depth(
    depth: np.ndarray,
    *,
    horizontal_fov_degrees: float = 90.0,
    distance_threshold_m: float = 0.09,
) -> tuple[np.ndarray, dict[str, float | list[float]]]:
    """Return pixels supported by the fitted current-frame ground plane."""
    normal, offset = estimate_ground_plane(depth, horizontal_fov_degrees=horizontal_fov_degrees)
    points = depth_points(depth, horizontal_fov_degrees)
    valid = np.isfinite(depth) & (depth >= 0.4) & (depth <= 8.0)
    distances = np.abs(points @ normal + offset)
    mask = valid & (distances <= distance_threshold_m) & (points[..., 1] > 0.0)
    return mask, {"normal_xyz": [float(value) for value in normal], "offset": offset, "inlier_fraction": float(mask.sum() / max(1, valid.sum()))}
