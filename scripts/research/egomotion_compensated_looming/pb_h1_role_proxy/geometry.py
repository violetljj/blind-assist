from __future__ import annotations

import math
from typing import Any

import numpy as np


def _validate_inputs(
    pixels_xy: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    rotation_current_from_previous: np.ndarray,
    translation_current_from_previous_m: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    k = np.asarray(intrinsic, dtype=np.float64)
    rotation = np.asarray(rotation_current_from_previous, dtype=np.float64)
    translation = np.asarray(
        translation_current_from_previous_m, dtype=np.float64
    )
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("PB_H1_PIXELS_SHAPE")
    if depth.shape != (pixels.shape[0],):
        raise ValueError("PB_H1_DEPTH_SHAPE")
    if k.shape != (3, 3) or rotation.shape != (3, 3):
        raise ValueError("PB_H1_MATRIX_SHAPE")
    if translation.shape != (3,):
        raise ValueError("PB_H1_TRANSLATION_SHAPE")
    if (
        not np.all(np.isfinite(pixels))
        or not np.all(np.isfinite(depth))
        or not np.all(np.isfinite(k))
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
        or not math.isfinite(dt_s)
        or dt_s <= 0.0
    ):
        raise ValueError("PB_H1_NONFINITE_OR_DT")
    return pixels, depth, k, rotation, translation


def translation_induced_geometry(
    pixels_xy: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    rotation_current_from_previous: np.ndarray,
    translation_current_from_previous_m: np.ndarray,
    dt_s: float,
    *,
    image_size_wh: tuple[int, int],
    minimum_radius_px: float = 8.0,
    zbuffer: bool = True,
) -> dict[str, np.ndarray | float | int]:
    """Compute the translation term relative to the rotation-only projection.

    For a previous-camera point X, the rotation-only and full predictions are
    X_r = R X and X_f = R X + t. Radial expansion is
    log(rho_f / rho_r) / dt around the calibrated principal point. Parallax is
    the angle between the unit bearings of X_r and X_f, divided by dt.
    """

    pixels, depth, k, rotation, translation = _validate_inputs(
        pixels_xy,
        depth_m,
        intrinsic,
        rotation_current_from_previous,
        translation_current_from_previous_m,
        dt_s,
    )
    width, height = image_size_wh
    if width <= 0 or height <= 0 or minimum_radius_px <= 0.0:
        raise ValueError("PB_H1_IMAGE_OR_RADIUS")

    valid_source = depth > 0.0
    homogeneous = np.column_stack(
        (pixels, np.ones(pixels.shape[0], dtype=np.float64))
    )
    points_previous = (
        np.linalg.inv(k) @ homogeneous.T
    ).T * depth[:, None]
    points_rotation = (rotation @ points_previous.T).T
    points_full = points_rotation + translation[None, :]

    def project(points: np.ndarray) -> np.ndarray:
        projected = (k @ points.T).T
        return projected[:, :2] / projected[:, 2:3]

    positive_z = (points_rotation[:, 2] > 0.0) & (
        points_full[:, 2] > 0.0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pixel_rotation = project(points_rotation)
        pixel_full = project(points_full)
    in_bounds = (
        (pixel_rotation[:, 0] >= 0.0)
        & (pixel_rotation[:, 0] < width)
        & (pixel_rotation[:, 1] >= 0.0)
        & (pixel_rotation[:, 1] < height)
        & (pixel_full[:, 0] >= 0.0)
        & (pixel_full[:, 0] < width)
        & (pixel_full[:, 1] >= 0.0)
        & (pixel_full[:, 1] < height)
    )
    center = np.asarray((k[0, 2], k[1, 2]), dtype=np.float64)
    radius_rotation = np.linalg.norm(pixel_rotation - center[None, :], axis=1)
    radius_full = np.linalg.norm(pixel_full - center[None, :], axis=1)
    valid = (
        valid_source
        & positive_z
        & in_bounds
        & np.isfinite(radius_rotation)
        & np.isfinite(radius_full)
        & (radius_rotation >= minimum_radius_px)
        & (radius_full > 0.0)
    )

    if zbuffer:
        candidate_indices = np.flatnonzero(valid)
        winners: dict[tuple[int, int], int] = {}
        for index in candidate_indices:
            destination = tuple(
                np.floor(pixel_full[index] + 0.5).astype(np.int64).tolist()
            )
            incumbent = winners.get(destination)
            if incumbent is None or points_full[index, 2] < points_full[
                incumbent, 2
            ]:
                winners[destination] = int(index)
        visible_indices = np.asarray(
            sorted(winners.values()), dtype=np.int64
        )
    else:
        visible_indices = np.flatnonzero(valid)

    rotation_points = points_rotation[visible_indices]
    full_points = points_full[visible_indices]
    rotation_bearings = rotation_points / np.linalg.norm(
        rotation_points, axis=1, keepdims=True
    )
    full_bearings = full_points / np.linalg.norm(
        full_points, axis=1, keepdims=True
    )
    bearing_dot = np.sum(rotation_bearings * full_bearings, axis=1)
    bearing_cross = np.linalg.norm(
        np.cross(rotation_bearings, full_bearings), axis=1
    )
    parallax_rad_s = np.arctan2(
        bearing_cross, np.clip(bearing_dot, -1.0, 1.0)
    ) / dt_s
    radial_expansion_s = np.log(
        radius_full[visible_indices] / radius_rotation[visible_indices]
    ) / dt_s
    return {
        "source_count": int(pixels.shape[0]),
        "valid_count": int(visible_indices.size),
        "valid_fraction": float(
            visible_indices.size / pixels.shape[0] if pixels.shape[0] else 0.0
        ),
        "raw_translation_speed_m_s": float(
            np.linalg.norm(translation) / dt_s
        ),
        "radial_expansion_per_s": radial_expansion_s,
        "parallax_rad_per_s": parallax_rad_s,
        "visible_source_indices": visible_indices,
    }


def summarize_translation_induced_geometry(
    result: dict[str, np.ndarray | float | int],
) -> dict[str, Any]:
    radial = np.asarray(result["radial_expansion_per_s"], dtype=np.float64)
    parallax = np.asarray(result["parallax_rad_per_s"], dtype=np.float64)
    if radial.size == 0 or parallax.size == 0:
        return {
            "evaluable": False,
            "source_count": int(result["source_count"]),
            "valid_count": int(result["valid_count"]),
            "valid_fraction": float(result["valid_fraction"]),
            "raw_translation_speed_m_s": float(
                result["raw_translation_speed_m_s"]
            ),
        }
    return {
        "evaluable": True,
        "source_count": int(result["source_count"]),
        "valid_count": int(result["valid_count"]),
        "valid_fraction": float(result["valid_fraction"]),
        "raw_translation_speed_m_s": float(
            result["raw_translation_speed_m_s"]
        ),
        "median_signed_radial_expansion_per_s": float(np.median(radial)),
        "median_absolute_radial_expansion_per_s": float(
            np.median(np.abs(radial))
        ),
        "radial_expansion_positive_fraction": float(np.mean(radial > 0.0)),
        "q90_time_normalized_parallax_rad_per_s": float(
            np.quantile(parallax, 0.90)
        ),
    }
