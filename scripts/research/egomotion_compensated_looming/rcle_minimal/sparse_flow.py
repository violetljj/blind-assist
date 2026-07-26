from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class SparseTrackResult:
    previous_points: np.ndarray
    current_points: np.ndarray
    forward_backward_errors: np.ndarray
    requested_count: int

    @property
    def valid_count(self) -> int:
        return int(self.previous_points.shape[0])


def _cell_bounds(
    width: int, height: int, row: int, column: int, rows: int, columns: int
) -> tuple[int, int, int, int]:
    x0 = int(round(column * width / columns))
    x1 = int(round((column + 1) * width / columns))
    y0 = int(round(row * height / rows))
    y1 = int(round((row + 1) * height / rows))
    return x0, y0, x1, y1


def detect_fixed_grid_features(
    previous_image: np.ndarray,
    valid_mask: np.ndarray,
    parameters: dict[str, Any],
) -> np.ndarray:
    height, width = previous_image.shape
    rows = int(parameters["grid_rows"])
    columns = int(parameters["grid_cols"])
    all_points: list[np.ndarray] = []
    for row in range(rows):
        for column in range(columns):
            x0, y0, x1, y1 = _cell_bounds(
                width, height, row, column, rows, columns
            )
            cell_mask = np.zeros_like(valid_mask, dtype=np.uint8)
            cell_mask[y0:y1, x0:x1] = valid_mask[y0:y1, x0:x1]
            points = cv2.goodFeaturesToTrack(
                previous_image,
                maxCorners=int(parameters["max_features_per_cell"]),
                qualityLevel=float(parameters["quality_level"]),
                minDistance=float(parameters["min_distance_pixels"]),
                mask=cell_mask,
                blockSize=int(parameters["block_size"]),
                useHarrisDetector=False,
            )
            if points is not None:
                all_points.append(points.reshape(-1, 2))
    if not all_points:
        return np.empty((0, 2), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(all_points).astype(np.float32))


def _sample_mask(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    height, width = mask.shape
    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    valid = np.zeros(points.shape[0], dtype=bool)
    indices = np.flatnonzero(inside)
    valid[indices] = mask[y[indices], x[indices]] > 0
    return valid


def track_features(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    initial_points: np.ndarray,
    current_valid_mask: np.ndarray,
    parameters: dict[str, Any],
) -> SparseTrackResult:
    requested = int(initial_points.shape[0])
    if requested == 0:
        return SparseTrackResult(
            previous_points=np.empty((0, 2), dtype=np.float32),
            current_points=np.empty((0, 2), dtype=np.float32),
            forward_backward_errors=np.empty((0,), dtype=np.float32),
            requested_count=0,
        )
    points = initial_points.reshape(-1, 1, 2).astype(np.float32)
    window = tuple(int(value) for value in parameters["window_size"])
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(parameters["termination_count"]),
        float(parameters["termination_epsilon"]),
    )
    current, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_image,
        current_image,
        points,
        None,
        winSize=window,
        maxLevel=int(parameters["max_pyramid_level"]),
        criteria=criteria,
    )
    if current is None or forward_status is None:
        return SparseTrackResult(
            previous_points=np.empty((0, 2), dtype=np.float32),
            current_points=np.empty((0, 2), dtype=np.float32),
            forward_backward_errors=np.empty((0,), dtype=np.float32),
            requested_count=requested,
        )
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_image,
        previous_image,
        current,
        None,
        winSize=window,
        maxLevel=int(parameters["max_pyramid_level"]),
        criteria=criteria,
    )
    if backward is None or backward_status is None:
        return SparseTrackResult(
            previous_points=np.empty((0, 2), dtype=np.float32),
            current_points=np.empty((0, 2), dtype=np.float32),
            forward_backward_errors=np.empty((0,), dtype=np.float32),
            requested_count=requested,
        )
    previous_flat = points.reshape(-1, 2)
    current_flat = current.reshape(-1, 2)
    backward_flat = backward.reshape(-1, 2)
    fb_error = np.linalg.norm(backward_flat - previous_flat, axis=1)
    valid = (
        (forward_status.reshape(-1) > 0)
        & (backward_status.reshape(-1) > 0)
        & np.isfinite(current_flat).all(axis=1)
        & np.isfinite(backward_flat).all(axis=1)
        & (fb_error <= float(parameters["forward_backward_max_error_pixels"]))
        & _sample_mask(current_valid_mask, current_flat)
    )
    return SparseTrackResult(
        previous_points=np.ascontiguousarray(previous_flat[valid]),
        current_points=np.ascontiguousarray(current_flat[valid]),
        forward_backward_errors=np.ascontiguousarray(fb_error[valid]),
        requested_count=requested,
    )
