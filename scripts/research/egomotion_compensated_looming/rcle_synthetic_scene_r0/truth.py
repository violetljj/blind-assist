from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _project(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    projected = (intrinsics @ points_camera.T).T
    return projected[:, :2] / projected[:, 2:3]


def _fit_dense_expansion(
    pixels: np.ndarray,
    velocity: np.ndarray,
    width: int,
    height: int,
) -> tuple[list[float], list[int]]:
    expansions: list[float] = []
    counts: list[int] = []
    for row in range(3):
        for column in range(3):
            x0, x1 = width * column / 3.0, width * (column + 1) / 3.0
            y0, y1 = height * row / 3.0, height * (row + 1) / 3.0
            selected = (
                (pixels[:, 0] >= x0)
                & (pixels[:, 0] < x1)
                & (pixels[:, 1] >= y0)
                & (pixels[:, 1] < y1)
            )
            point_count = int(np.count_nonzero(selected))
            counts.append(point_count)
            if point_count < 24:
                continue
            local = pixels[selected].astype(np.float64)
            local[:, 0] -= (x0 + x1) * 0.5
            local[:, 1] -= (y0 + y1) * 0.5
            design = np.column_stack(
                [local[:, 0], local[:, 1], np.ones(point_count)]
            )
            coefficients, _, rank, _ = np.linalg.lstsq(
                design, velocity[selected], rcond=None
            )
            if rank != 3:
                continue
            # design @ coefficients -> [vx, vy], so the spatial Jacobian is
            # [[coef_x_for_vx, coef_y_for_vx], [coef_x_for_vy, coef_y_for_vy]].
            expansion = 0.5 * (
                float(coefficients[0, 0]) + float(coefficients[1, 1])
            )
            expansions.append(expansion)
    return expansions, counts


def evaluate_truth_pair(
    dataset_root: Path,
    previous: dict[str, Any],
    current: dict[str, Any],
    sample_step: int = 4,
) -> dict[str, Any]:
    depth_previous = np.load(
        dataset_root / previous["depth_npy_path"], allow_pickle=False
    ).astype(np.float64)
    depth_current = np.load(
        dataset_root / current["depth_npy_path"], allow_pickle=False
    ).astype(np.float64)
    height, width = depth_previous.shape
    intrinsics = np.asarray(previous["intrinsics"], dtype=np.float64)
    inverse_intrinsics = np.linalg.inv(intrinsics)
    t_world_previous = np.asarray(
        previous["t_world_camera"], dtype=np.float64
    )
    t_world_current = np.asarray(
        current["t_world_camera"], dtype=np.float64
    )
    dt = (
        float(current["timestamp"]["seconds"])
        - float(previous["timestamp"]["seconds"])
    )
    if not np.isfinite(dt) or dt <= 0:
        return {"evaluable": False, "reason": "NON_POSITIVE_DT"}

    ys, xs = np.mgrid[
        sample_step // 2 : height : sample_step,
        sample_step // 2 : width : sample_step,
    ]
    pixels = np.column_stack(
        [xs.reshape(-1), ys.reshape(-1)]
    ).astype(np.float64)
    depth = depth_previous[
        ys.reshape(-1), xs.reshape(-1)
    ]
    finite = np.isfinite(depth) & (depth > 0.0)
    pixels = pixels[finite]
    depth = depth[finite]
    homogeneous = np.column_stack(
        [pixels, np.ones(len(pixels), dtype=np.float64)]
    )
    points_previous = (
        (inverse_intrinsics @ homogeneous.T).T * depth[:, None]
    )
    rotation_world_previous = t_world_previous[:3, :3]
    translation_world_previous = t_world_previous[:3, 3]
    rotation_world_current = t_world_current[:3, :3]
    translation_world_current = t_world_current[:3, 3]
    points_world = (
        rotation_world_previous @ points_previous.T
    ).T + translation_world_previous
    points_current = (
        rotation_world_current.T
        @ (points_world - translation_world_current).T
    ).T
    in_front = points_current[:, 2] > 1e-6
    projected_current = np.full((len(points_current), 2), np.nan)
    projected_current[in_front] = _project(
        points_current[in_front], intrinsics
    )
    inside = (
        in_front
        & (projected_current[:, 0] >= 0)
        & (projected_current[:, 0] < width)
        & (projected_current[:, 1] >= 0)
        & (projected_current[:, 1] < height)
    )
    nearest_x = np.clip(
        np.rint(projected_current[:, 0]).astype(np.int64, casting="unsafe"),
        0,
        width - 1,
    )
    nearest_y = np.clip(
        np.rint(projected_current[:, 1]).astype(np.int64, casting="unsafe"),
        0,
        height - 1,
    )
    rendered_depth = depth_current[nearest_y, nearest_x]
    visible = inside & np.isfinite(rendered_depth)
    visible &= np.abs(rendered_depth - points_current[:, 2]) <= np.maximum(
        0.04, 0.025 * points_current[:, 2]
    )

    rotation_current_from_previous = (
        rotation_world_current.T @ rotation_world_previous
    )
    rotation_h = (
        intrinsics
        @ rotation_current_from_previous
        @ inverse_intrinsics
    )
    inverse_rotation_h = np.linalg.inv(rotation_h)
    current_homogeneous = np.column_stack(
        [
            projected_current[:, 0],
            projected_current[:, 1],
            np.ones(len(projected_current)),
        ]
    )
    compensated_homogeneous = (
        inverse_rotation_h @ current_homogeneous.T
    ).T
    compensated_pixels = (
        compensated_homogeneous[:, :2]
        / compensated_homogeneous[:, 2:3]
    )
    selected_pixels = pixels[visible]
    velocity = (
        compensated_pixels[visible] - selected_pixels
    ) / dt
    expansions, cell_counts = _fit_dense_expansion(
        selected_pixels, velocity, width, height
    )
    if len(expansions) < 5:
        return {
            "evaluable": False,
            "reason": "TRUTH_COMMON_GRID_SUPPORT_BELOW_5_OF_9",
            "visible_sample_count": int(np.count_nonzero(visible)),
            "cell_sample_counts": cell_counts,
        }
    return {
        "evaluable": True,
        "reason": None,
        "visible_sample_count": int(np.count_nonzero(visible)),
        "cell_sample_counts": cell_counts,
        "cell_expansion_per_s": expansions,
        "compensated_expansion_median_per_s": float(
            np.median(expansions)
        ),
        "compensated_abs_expansion_median_per_s": float(
            np.median(np.abs(expansions))
        ),
        "rotation_current_from_previous": (
            rotation_current_from_previous.tolist()
        ),
    }
