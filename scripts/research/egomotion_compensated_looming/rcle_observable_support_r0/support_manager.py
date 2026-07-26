from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.sparse_flow import (
    SparseTrackResult,
)


GEOMETRIC_FIELD_EXIT = "GEOMETRIC_FIELD_EXIT"
OBSERVABLE_OCCLUSION = "OBSERVABLE_OCCLUSION"
ORDINARY_NEW_TRACK_FAILURE = "ORDINARY_NEW_TRACK_FAILURE"
CURRENT_LEG_SURVIVOR = "CURRENT_LEG_SURVIVOR"

_PATCH_OFFSETS = np.asarray(
    [(x, y) for y in range(-3, 4) for x in range(-3, 4)],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ObservableTrackDiagnostics:
    """Per-candidate results derived only from two observable image frames."""

    initial_points: np.ndarray
    forward_points: np.ndarray
    forward_available: np.ndarray
    forward_backward_errors: np.ndarray
    forward_backward_pass: np.ndarray
    source_patch_valid: np.ndarray
    target_patch_valid: np.ndarray
    photometric_errors: np.ndarray
    photometric_pass: np.ndarray
    accepted: np.ndarray

    def __post_init__(self) -> None:
        count = int(self.initial_points.shape[0])
        if self.initial_points.shape != (count, 2):
            raise ValueError("INITIAL_POINTS_MUST_BE_N_BY_2")
        if self.forward_points.shape != (count, 2):
            raise ValueError("FORWARD_POINTS_MUST_BE_N_BY_2")
        for name in (
            "forward_available",
            "forward_backward_errors",
            "forward_backward_pass",
            "source_patch_valid",
            "target_patch_valid",
            "photometric_errors",
            "photometric_pass",
            "accepted",
        ):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name.upper()}_SHAPE_MISMATCH")

    @property
    def requested_count(self) -> int:
        return int(self.initial_points.shape[0])

    @property
    def accepted_count(self) -> int:
        return int(np.count_nonzero(self.accepted))

    def accepted_tracks(self) -> SparseTrackResult:
        selected = self.accepted
        return SparseTrackResult(
            previous_points=np.ascontiguousarray(
                self.initial_points[selected].astype(np.float32)
            ),
            current_points=np.ascontiguousarray(
                self.forward_points[selected].astype(np.float32)
            ),
            forward_backward_errors=np.ascontiguousarray(
                self.forward_backward_errors[selected].astype(np.float32)
            ),
            requested_count=self.requested_count,
        )


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("OBSERVABLE_IMAGE_MUST_BE_2D_UINT8")
    return np.ascontiguousarray(image)


def _as_valid_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape != shape:
        raise ValueError("OBSERVABLE_VALIDITY_SHAPE_MISMATCH")
    return np.ascontiguousarray(mask > 0)


def _empty_diagnostics() -> ObservableTrackDiagnostics:
    points = np.empty((0, 2), dtype=np.float32)
    boolean = np.empty((0,), dtype=bool)
    errors = np.empty((0,), dtype=np.float32)
    return ObservableTrackDiagnostics(
        initial_points=points,
        forward_points=points.copy(),
        forward_available=boolean,
        forward_backward_errors=errors,
        forward_backward_pass=boolean.copy(),
        source_patch_valid=boolean.copy(),
        target_patch_valid=boolean.copy(),
        photometric_errors=errors.copy(),
        photometric_pass=boolean.copy(),
        accepted=boolean.copy(),
    )


def _bilinear_patch(
    image: np.ndarray,
    valid_mask: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    coordinates = _PATCH_OFFSETS + center.astype(np.float64)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    height, width = image.shape
    inside = (
        np.isfinite(coordinates).all(axis=1)
        & (x >= 0.0)
        & (x <= width - 1)
        & (y >= 0.0)
        & (y <= height - 1)
    )
    if not bool(np.all(inside)):
        return None, False

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.ceil(x).astype(np.int64)
    y1 = np.ceil(y).astype(np.int64)
    validity = (
        valid_mask[y0, x0]
        & valid_mask[y0, x1]
        & valid_mask[y1, x0]
        & valid_mask[y1, x1]
    )
    if not bool(np.all(validity)):
        return None, False

    dx = x - x0
    dy = y - y0
    values = (
        image[y0, x0].astype(np.float64) * (1.0 - dx) * (1.0 - dy)
        + image[y0, x1].astype(np.float64) * dx * (1.0 - dy)
        + image[y1, x0].astype(np.float64) * (1.0 - dx) * dy
        + image[y1, x1].astype(np.float64) * dx * dy
    )
    return values, True


def median_centered_patch_errors(
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return error plus source/target full-patch validity for each pair."""

    source = _as_gray_u8(source_image)
    target = _as_gray_u8(target_image)
    if source.shape != target.shape:
        raise ValueError("OBSERVABLE_FRAME_SHAPE_MISMATCH")
    source_valid = _as_valid_mask(source_valid_mask, source.shape)
    target_valid = _as_valid_mask(target_valid_mask, target.shape)
    source_points = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    target_points = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    if source_points.shape != target_points.shape:
        raise ValueError("PATCH_POINT_COUNT_MISMATCH")

    count = source_points.shape[0]
    errors = np.full(count, np.inf, dtype=np.float32)
    source_patch_valid = np.zeros(count, dtype=bool)
    target_patch_valid = np.zeros(count, dtype=bool)
    for index in range(count):
        source_patch, source_ok = _bilinear_patch(
            source, source_valid, source_points[index]
        )
        target_patch, target_ok = _bilinear_patch(
            target, target_valid, target_points[index]
        )
        source_patch_valid[index] = source_ok
        target_patch_valid[index] = target_ok
        if not source_ok or not target_ok:
            continue
        assert source_patch is not None and target_patch is not None
        source_centered = source_patch - np.median(source_patch)
        target_centered = target_patch - np.median(target_patch)
        errors[index] = float(
            np.mean(np.abs(source_centered - target_centered))
        )
    return errors, source_patch_valid, target_patch_valid


def _lk_settings(parameters: dict[str, Any]) -> tuple[
    tuple[int, int], int, tuple[int, int, float]
]:
    window = tuple(int(value) for value in parameters["window_size"])
    if len(window) != 2:
        raise ValueError("LK_WINDOW_MUST_HAVE_TWO_DIMENSIONS")
    maximum_level = int(parameters["max_pyramid_level"])
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(parameters["termination_count"]),
        float(parameters["termination_epsilon"]),
    )
    return window, maximum_level, criteria


def track_observable_points(
    source_image: np.ndarray,
    target_image: np.ndarray,
    initial_points: np.ndarray,
    source_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
    parameters: dict[str, Any],
    *,
    forward_backward_threshold_pixels: float = 1.0,
    photometric_threshold_intensity: float = 20.0,
) -> ObservableTrackDiagnostics:
    """Track candidates and retain failures needed for observable classification."""

    source = _as_gray_u8(source_image)
    target = _as_gray_u8(target_image)
    if source.shape != target.shape:
        raise ValueError("OBSERVABLE_FRAME_SHAPE_MISMATCH")
    source_valid = _as_valid_mask(source_valid_mask, source.shape)
    target_valid = _as_valid_mask(target_valid_mask, target.shape)
    initial = np.ascontiguousarray(
        np.asarray(initial_points, dtype=np.float32).reshape(-1, 2)
    )
    count = initial.shape[0]
    if count == 0:
        return _empty_diagnostics()
    if forward_backward_threshold_pixels != 1.0:
        raise ValueError("FORWARD_BACKWARD_THRESHOLD_DRIFT")
    if photometric_threshold_intensity != 20.0:
        raise ValueError("PHOTOMETRIC_THRESHOLD_DRIFT")

    window, maximum_level, criteria = _lk_settings(parameters)
    points = initial.reshape(-1, 1, 2)
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        source,
        target,
        points,
        None,
        winSize=window,
        maxLevel=maximum_level,
        criteria=criteria,
    )
    if forward is None:
        forward_flat = np.full((count, 2), np.nan, dtype=np.float32)
    else:
        forward_flat = np.ascontiguousarray(
            forward.reshape(-1, 2).astype(np.float32)
        )
    status = (
        np.zeros(count, dtype=bool)
        if forward_status is None
        else forward_status.reshape(-1) > 0
    )
    forward_available = status & np.isfinite(forward_flat).all(axis=1)

    best_error = np.full(count, np.inf, dtype=np.float64)
    backward_available = np.zeros(count, dtype=bool)
    if forward is not None:
        for level in range(maximum_level, -1, -1):
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                target,
                source,
                forward,
                points.copy(),
                winSize=window,
                maxLevel=level,
                criteria=criteria,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            )
            if backward is None or backward_status is None:
                continue
            backward_flat = backward.reshape(-1, 2)
            errors = np.linalg.norm(backward_flat - initial, axis=1)
            finite = (
                (backward_status.reshape(-1) > 0)
                & np.isfinite(backward_flat).all(axis=1)
                & np.isfinite(errors)
            )
            improve = finite & (errors < best_error)
            best_error[improve] = errors[improve]
            backward_available |= finite
    forward_backward_pass = (
        forward_available
        & backward_available
        & (best_error <= forward_backward_threshold_pixels)
    )

    photometric_errors, source_patch_valid, target_patch_valid = (
        median_centered_patch_errors(
            source,
            target,
            initial,
            forward_flat,
            source_valid,
            target_valid,
        )
    )
    photometric_pass = (
        source_patch_valid
        & target_patch_valid
        & np.isfinite(photometric_errors)
        & (photometric_errors <= photometric_threshold_intensity)
    )
    accepted = forward_backward_pass & photometric_pass
    return ObservableTrackDiagnostics(
        initial_points=initial,
        forward_points=forward_flat,
        forward_available=np.ascontiguousarray(forward_available),
        forward_backward_errors=np.ascontiguousarray(
            best_error.astype(np.float32)
        ),
        forward_backward_pass=np.ascontiguousarray(forward_backward_pass),
        source_patch_valid=np.ascontiguousarray(source_patch_valid),
        target_patch_valid=np.ascontiguousarray(target_patch_valid),
        photometric_errors=np.ascontiguousarray(photometric_errors),
        photometric_pass=np.ascontiguousarray(photometric_pass),
        accepted=np.ascontiguousarray(accepted),
    )


def _full_patch_supported_points(
    valid_mask: np.ndarray, points: np.ndarray
) -> np.ndarray:
    mask = _as_valid_mask(valid_mask, valid_mask.shape)
    dummy = np.zeros(mask.shape, dtype=np.uint8)
    result = np.zeros(points.shape[0], dtype=bool)
    for index, point in enumerate(points):
        _, result[index] = _bilinear_patch(dummy, mask, point)
    return result


def classify_prior_survivors(
    prior_endpoints_at_t: np.ndarray,
    prior_displacements: np.ndarray,
    current_leg: ObservableTrackDiagnostics,
    current_target_valid_mask: np.ndarray,
    *,
    prior_dt_seconds: float,
    current_dt_seconds: float,
) -> np.ndarray:
    """Classify prior survivors with field exit taking strict precedence."""

    endpoints = np.asarray(prior_endpoints_at_t, dtype=np.float64).reshape(-1, 2)
    displacements = np.asarray(
        prior_displacements, dtype=np.float64
    ).reshape(-1, 2)
    count = endpoints.shape[0]
    if (
        displacements.shape != endpoints.shape
        or current_leg.requested_count != count
        or not np.allclose(current_leg.initial_points, endpoints)
    ):
        raise ValueError("PRIOR_SURVIVOR_CURRENT_LEG_ALIGNMENT_MISMATCH")
    if (
        not np.isfinite(prior_dt_seconds)
        or prior_dt_seconds <= 0.0
        or not np.isfinite(current_dt_seconds)
        or current_dt_seconds <= 0.0
    ):
        raise ValueError("NON_POSITIVE_OR_MISSING_DT")

    ratio = current_dt_seconds / prior_dt_seconds
    predictions = endpoints + displacements * ratio
    exit_check_points = predictions.copy()
    exit_check_points[current_leg.forward_available] = (
        current_leg.forward_points[current_leg.forward_available]
    )
    supports_patch = _full_patch_supported_points(
        current_target_valid_mask, exit_check_points
    )
    result = np.full(count, OBSERVABLE_OCCLUSION, dtype=object)
    result[~supports_patch] = GEOMETRIC_FIELD_EXIT
    result[supports_patch & current_leg.accepted] = CURRENT_LEG_SURVIVOR
    return result


def classify_new_track_failures(
    current_leg: ObservableTrackDiagnostics,
    current_target_valid_mask: np.ndarray,
) -> np.ndarray:
    """Classify failed new points without ever promoting them to occlusions."""

    count = current_leg.requested_count
    result = np.full(count, ORDINARY_NEW_TRACK_FAILURE, dtype=object)
    result[current_leg.accepted] = CURRENT_LEG_SURVIVOR
    if count:
        supported = _full_patch_supported_points(
            current_target_valid_mask, current_leg.forward_points
        )
        field_exit = current_leg.forward_available & ~supported
        result[field_exit] = GEOMETRIC_FIELD_EXIT
    return result


def observable_occlusion_centers(
    prior_endpoints_at_t: np.ndarray, classifications: Sequence[str]
) -> np.ndarray:
    endpoints = np.asarray(prior_endpoints_at_t, dtype=np.float32).reshape(-1, 2)
    classes = np.asarray(classifications, dtype=object).reshape(-1)
    if classes.shape[0] != endpoints.shape[0]:
        raise ValueError("CLASSIFICATION_COUNT_MISMATCH")
    return np.ascontiguousarray(endpoints[classes == OBSERVABLE_OCCLUSION])


def activated_cell_indices(
    raw_cells: Sequence[Any],
    compensated_cells: Sequence[Any],
    *,
    minimum_support: int = 12,
    minimum_hull_fraction: float = 0.10,
) -> tuple[int, ...]:
    if minimum_support != 12 or minimum_hull_fraction != 0.10:
        raise ValueError("SUPPORT_OR_HULL_GATE_DRIFT")
    if len(raw_cells) != 9 or len(compensated_cells) != 9:
        raise ValueError("EXPECTED_FIXED_3X3_GRID")
    activated: list[int] = []
    for index, (raw, compensated) in enumerate(
        zip(raw_cells, compensated_cells, strict=True)
    ):
        if (
            int(raw.support_count) < minimum_support
            or float(raw.hull_fraction) < minimum_hull_fraction
            or int(compensated.support_count) < minimum_support
            or float(compensated.hull_fraction) < minimum_hull_fraction
        ):
            activated.append(index)
    return tuple(activated)


def _distance_allowed(
    point: np.ndarray,
    existing_points: np.ndarray,
    minimum_distance_pixels: float,
) -> bool:
    if existing_points.size == 0:
        return True
    squared = np.sum(
        (existing_points.astype(np.float64) - point.astype(np.float64)) ** 2,
        axis=1,
    )
    return bool(np.all(squared >= minimum_distance_pixels**2))


def select_spatial_supplements(
    source_image: np.ndarray,
    source_valid_mask: np.ndarray,
    cell_region: tuple[int, int, int, int],
    existing_points: np.ndarray,
    exclusion_centers: np.ndarray,
    *,
    subdivision_rows: int = 4,
    subdivision_columns: int = 4,
    minimum_distance_pixels: float = 5.0,
    exclusion_radius_pixels: float = 10.0,
    maximum_total_features_per_cell: int = 80,
    shi_tomasi_block_size: int = 7,
) -> np.ndarray:
    """Select at most one deterministic Shi-Tomasi maximum per 4x4 subcell."""

    image = _as_gray_u8(source_image)
    valid = _as_valid_mask(source_valid_mask, image.shape)
    if (subdivision_rows, subdivision_columns) != (4, 4):
        raise ValueError("SPATIAL_SUBDIVISION_DRIFT")
    if (
        minimum_distance_pixels != 5.0
        or exclusion_radius_pixels != 10.0
        or maximum_total_features_per_cell != 80
    ):
        raise ValueError("SPATIAL_SUPPLEMENT_PARAMETER_DRIFT")
    x0, y0, x1, y1 = (int(value) for value in cell_region)
    height, width = image.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("INVALID_CELL_REGION")

    existing = np.asarray(existing_points, dtype=np.float32).reshape(-1, 2)
    exclusions = np.asarray(exclusion_centers, dtype=np.float32).reshape(-1, 2)
    in_cell = (
        (existing[:, 0] >= x0)
        & (existing[:, 0] < x1)
        & (existing[:, 1] >= y0)
        & (existing[:, 1] < y1)
    )
    selected: list[np.ndarray] = []
    occupied = np.ascontiguousarray(existing)
    capacity = maximum_total_features_per_cell - int(np.count_nonzero(in_cell))
    if capacity <= 0:
        return np.empty((0, 2), dtype=np.float32)

    response = cv2.cornerMinEigenVal(
        image,
        blockSize=int(shi_tomasi_block_size),
        ksize=3,
        borderType=cv2.BORDER_DEFAULT,
    )
    for sub_row in range(subdivision_rows):
        sy0 = int(round(y0 + sub_row * (y1 - y0) / subdivision_rows))
        sy1 = int(round(y0 + (sub_row + 1) * (y1 - y0) / subdivision_rows))
        for sub_column in range(subdivision_columns):
            if len(selected) >= capacity:
                break
            sx0 = int(
                round(x0 + sub_column * (x1 - x0) / subdivision_columns)
            )
            sx1 = int(
                round(x0 + (sub_column + 1) * (x1 - x0) / subdivision_columns)
            )
            ys, xs = np.nonzero(valid[sy0:sy1, sx0:sx1])
            if xs.size == 0:
                continue
            xs = xs.astype(np.int64) + sx0
            ys = ys.astype(np.int64) + sy0
            values = response[ys, xs].astype(np.float64)
            order = np.lexsort((xs, ys, -values))
            for candidate_index in order:
                point = np.asarray(
                    [xs[candidate_index], ys[candidate_index]],
                    dtype=np.float32,
                )
                if not _distance_allowed(
                    point, occupied, minimum_distance_pixels
                ):
                    continue
                if not _distance_allowed(
                    point, exclusions, exclusion_radius_pixels
                ):
                    continue
                selected.append(point)
                occupied = np.vstack((occupied, point.reshape(1, 2)))
                break
        if len(selected) >= capacity:
            break
    if not selected:
        return np.empty((0, 2), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(selected).astype(np.float32))


def merge_path_correspondences(
    baseline_tracks: SparseTrackResult,
    carried_survivors: ObservableTrackDiagnostics,
    spatial_supplements: ObservableTrackDiagnostics,
    *,
    duplicate_radius_pixels: float = 5.0,
) -> SparseTrackResult:
    """Merge one path in the frozen baseline, carry, supplement order."""

    if duplicate_radius_pixels != 5.0:
        raise ValueError("DUPLICATE_SUPPRESSION_RADIUS_DRIFT")
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    errors: list[float] = []

    def append(source: np.ndarray, target: np.ndarray, error: float) -> None:
        if sources:
            occupied = np.vstack(sources)
            if not _distance_allowed(
                source, occupied, duplicate_radius_pixels
            ):
                return
        sources.append(np.asarray(source, dtype=np.float32))
        targets.append(np.asarray(target, dtype=np.float32))
        errors.append(float(error))

    for source, target, error in zip(
        baseline_tracks.previous_points,
        baseline_tracks.current_points,
        baseline_tracks.forward_backward_errors,
        strict=True,
    ):
        append(source, target, float(error))
    for diagnostics in (carried_survivors, spatial_supplements):
        for source, target, error in zip(
            diagnostics.initial_points[diagnostics.accepted],
            diagnostics.forward_points[diagnostics.accepted],
            diagnostics.forward_backward_errors[diagnostics.accepted],
            strict=True,
        ):
            append(source, target, float(error))

    if not sources:
        previous = np.empty((0, 2), dtype=np.float32)
        current = np.empty((0, 2), dtype=np.float32)
        fb_errors = np.empty((0,), dtype=np.float32)
    else:
        previous = np.ascontiguousarray(np.vstack(sources).astype(np.float32))
        current = np.ascontiguousarray(np.vstack(targets).astype(np.float32))
        fb_errors = np.ascontiguousarray(np.asarray(errors, dtype=np.float32))
    return SparseTrackResult(
        previous_points=previous,
        current_points=current,
        forward_backward_errors=fb_errors,
        requested_count=(
            baseline_tracks.requested_count
            + carried_survivors.requested_count
            + spatial_supplements.requested_count
        ),
    )
