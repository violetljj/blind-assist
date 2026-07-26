from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .sparse_flow import SparseTrackResult


@dataclass(frozen=True)
class LocalExpansionResult:
    expansion: float | None
    confidence: float
    support_count: int
    tracked_support_count: int
    fit_residual_pixels_per_frame: float | None
    region: tuple[int, int, int, int]
    condition_number: float | None
    hull_fraction: float
    evaluable: bool
    abstention_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cell_bounds(
    width: int, height: int, row: int, column: int, rows: int, columns: int
) -> tuple[int, int, int, int]:
    x0 = int(round(column * width / columns))
    x1 = int(round((column + 1) * width / columns))
    y0 = int(round(row * height / rows))
    y1 = int(round((row + 1) * height / rows))
    return x0, y0, x1, y1


def _abstain(
    reason: str,
    support: int,
    tracked_support: int,
    region: tuple[int, int, int, int],
    condition: float | None = None,
    hull_fraction: float = 0.0,
    residual: float | None = None,
) -> LocalExpansionResult:
    return LocalExpansionResult(
        expansion=None,
        confidence=0.0,
        support_count=support,
        tracked_support_count=tracked_support,
        fit_residual_pixels_per_frame=residual,
        region=region,
        condition_number=condition,
        hull_fraction=hull_fraction,
        evaluable=False,
        abstention_reason=reason,
    )


def _consensus_mask(
    previous: np.ndarray,
    current: np.ndarray,
    residual_threshold: float,
) -> np.ndarray:
    if previous.shape[0] < 3:
        return np.zeros(previous.shape[0], dtype=bool)
    _, inliers = cv2.estimateAffine2D(
        previous.astype(np.float32),
        current.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=residual_threshold,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if inliers is None:
        return np.zeros(previous.shape[0], dtype=bool)
    return np.ascontiguousarray(inliers.reshape(-1) > 0)


def fit_fixed_grid_local_affine(
    tracks: SparseTrackResult,
    dt_seconds: float,
    image_shape: tuple[int, int],
    parameters: dict[str, Any],
) -> list[LocalExpansionResult]:
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("NON_POSITIVE_OR_MISSING_DT")
    height, width = image_shape
    rows = int(parameters["grid_rows"])
    columns = int(parameters["grid_cols"])
    minimum_support = int(parameters["minimum_tracks_per_cell"])
    residual_threshold = float(
        parameters["maximum_median_fit_residual_pixels_per_frame"]
    )
    results: list[LocalExpansionResult] = []
    previous = tracks.previous_points.astype(np.float64)
    current = tracks.current_points.astype(np.float64)
    for row in range(rows):
        for column in range(columns):
            region = _cell_bounds(width, height, row, column, rows, columns)
            x0, y0, x1, y1 = region
            in_cell = (
                (previous[:, 0] >= x0)
                & (previous[:, 0] < x1)
                & (previous[:, 1] >= y0)
                & (previous[:, 1] < y1)
            )
            tracked_previous = previous[in_cell]
            tracked_current = current[in_cell]
            tracked_support = int(tracked_previous.shape[0])
            consensus = _consensus_mask(
                tracked_previous, tracked_current, residual_threshold
            )
            points = tracked_previous[consensus]
            endpoints = tracked_current[consensus]
            support = int(points.shape[0])
            if support < minimum_support:
                results.append(
                    _abstain(
                        "LK_TRACK_SUPPORT_BELOW_12",
                        support,
                        tracked_support,
                        region,
                    )
                )
                continue
            hull = cv2.convexHull(points.astype(np.float32))
            hull_area = float(cv2.contourArea(hull))
            cell_area = float(max((x1 - x0) * (y1 - y0), 1))
            hull_fraction = hull_area / cell_area
            if hull_fraction < float(
                parameters["minimum_track_convex_hull_fraction"]
            ):
                results.append(
                    _abstain(
                        "TRACK_HULL_COVERAGE_BELOW_0_10",
                        support,
                        tracked_support,
                        region,
                        hull_fraction=hull_fraction,
                    )
                )
                continue
            center_x = 0.5 * (x0 + x1)
            center_y = 0.5 * (y0 + y1)
            half_width = max(0.5 * (x1 - x0), 1.0)
            half_height = max(0.5 * (y1 - y0), 1.0)
            design = np.column_stack(
                (
                    (points[:, 0] - center_x) / half_width,
                    (points[:, 1] - center_y) / half_height,
                    np.ones(support, dtype=np.float64),
                )
            )
            condition = float(np.linalg.cond(design))
            if (
                not np.isfinite(condition)
                or condition
                > float(parameters["maximum_design_condition_number"])
            ):
                results.append(
                    _abstain(
                        "AFFINE_DESIGN_CONDITION_ABOVE_1000",
                        support,
                        tracked_support,
                        region,
                        condition=condition,
                        hull_fraction=hull_fraction,
                    )
                )
                continue
            velocities = (endpoints - points) / dt_seconds
            coefficients, _, _, _ = np.linalg.lstsq(
                design, velocities, rcond=None
            )
            predicted = design @ coefficients
            residual_per_second = np.linalg.norm(
                predicted - velocities, axis=1
            )
            residual_per_frame = float(
                np.median(residual_per_second) * dt_seconds
            )
            if residual_per_frame > residual_threshold:
                results.append(
                    _abstain(
                        "AFFINE_MEDIAN_RESIDUAL_ABOVE_0_75PX_PER_FRAME",
                        support,
                        tracked_support,
                        region,
                        condition=condition,
                        hull_fraction=hull_fraction,
                        residual=residual_per_frame,
                    )
                )
                continue
            a_xx = coefficients[0, 0] / half_width
            a_yy = coefficients[1, 1] / half_height
            expansion = 0.5 * (a_xx + a_yy)
            if not np.isfinite(expansion):
                results.append(
                    _abstain(
                        "NON_FINITE_EXPANSION",
                        support,
                        tracked_support,
                        region,
                        condition=condition,
                        hull_fraction=hull_fraction,
                        residual=residual_per_frame,
                    )
                )
                continue
            support_quality = min(1.0, support / max(2 * minimum_support, 1))
            residual_quality = max(
                0.0, 1.0 - residual_per_frame / residual_threshold
            )
            spatial_quality = min(
                1.0,
                hull_fraction
                / max(
                    2.0
                    * float(
                        parameters["minimum_track_convex_hull_fraction"]
                    ),
                    1e-12,
                ),
            )
            confidence = support_quality * residual_quality * spatial_quality
            results.append(
                LocalExpansionResult(
                    expansion=float(expansion),
                    confidence=float(confidence),
                    support_count=support,
                    tracked_support_count=tracked_support,
                    fit_residual_pixels_per_frame=residual_per_frame,
                    region=region,
                    condition_number=condition,
                    hull_fraction=hull_fraction,
                    evaluable=True,
                    abstention_reason=None,
                )
            )
    return results
