"""Deterministic scale-free primitives for SVRF-O0 mechanics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    valid: bool
    status: str
    scale: float | None
    shift: float | None
    support_count: int
    support_fraction: float
    normalized_residual_mad: float | None
    aligned_current_depth: np.ndarray | None
    support_mask: np.ndarray


def robust_scale_shift_align(
    previous_depth: np.ndarray,
    warped_current_depth: np.ndarray,
    static_background_mask: np.ndarray,
    *,
    minimum_support: int = 64,
    minimum_support_fraction: float = 0.25,
    maximum_normalized_residual_mad: float = 0.08,
) -> AlignmentResult:
    """Fit previous ~= scale * current + shift on static background only."""
    previous = np.asarray(previous_depth, dtype=np.float64)
    current = np.asarray(warped_current_depth, dtype=np.float64)
    mask = np.asarray(static_background_mask, dtype=bool)
    if previous.shape != current.shape or previous.shape != mask.shape or previous.ndim < 1:
        raise ValueError("SVRF depth and background-mask shapes must match")
    finite = mask & np.isfinite(previous) & np.isfinite(current) & (previous > 1e-6) & (current > 1e-6)
    available = int(np.count_nonzero(mask))
    empty = np.zeros_like(mask, dtype=bool)
    if available == 0 or np.count_nonzero(finite) < minimum_support:
        return AlignmentResult(False, "UNKNOWN_ALIGNMENT_SUPPORT", None, None, int(np.count_nonzero(finite)), 0.0, None, None, empty)
    support = finite.copy()
    scale, shift = 1.0, 0.0
    for _ in range(4):
        x = current[support]
        y = previous[support]
        design = np.column_stack((x, np.ones_like(x)))
        solution, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 2:
            return AlignmentResult(False, "UNKNOWN_ALIGNMENT_RANK", None, None, int(np.count_nonzero(support)), 0.0, None, None, empty)
        scale, shift = float(solution[0]), float(solution[1])
        residual = previous - (scale * current + shift)
        center = float(np.median(residual[support]))
        mad = float(np.median(np.abs(residual[support] - center)))
        threshold = max(2.5 * 1.4826 * mad, 1e-6)
        updated = finite & (np.abs(residual - center) <= threshold)
        if np.array_equal(updated, support):
            break
        support = updated
        if np.count_nonzero(support) < minimum_support:
            return AlignmentResult(False, "UNKNOWN_ALIGNMENT_ROBUST_SUPPORT", scale, shift, int(np.count_nonzero(support)), np.count_nonzero(support) / available, None, None, support)
    support_count = int(np.count_nonzero(support))
    support_fraction = support_count / available
    if support_fraction < minimum_support_fraction:
        return AlignmentResult(False, "UNKNOWN_ALIGNMENT_FRACTION", scale, shift, support_count, support_fraction, None, None, support)
    if not 0.2 <= scale <= 5.0:
        return AlignmentResult(False, "UNKNOWN_ALIGNMENT_SCALE", scale, shift, support_count, support_fraction, None, None, support)
    aligned = scale * current + shift
    if np.count_nonzero(aligned[support] > 1e-6) / support_count < 0.95:
        return AlignmentResult(False, "UNKNOWN_ALIGNMENT_POSITIVITY", scale, shift, support_count, support_fraction, None, None, support)
    residual = previous[support] - aligned[support]
    residual_mad = float(np.median(np.abs(residual - np.median(residual))))
    depth_iqr = float(np.quantile(previous[support], 0.75) - np.quantile(previous[support], 0.25))
    normalized = residual_mad / max(depth_iqr, 1e-6)
    if normalized > maximum_normalized_residual_mad:
        return AlignmentResult(False, "UNKNOWN_ALIGNMENT_RESIDUAL", scale, shift, support_count, support_fraction, normalized, None, support)
    return AlignmentResult(True, "VALID_SCALE_SHIFT_ALIGNMENT", scale, shift, support_count, support_fraction, normalized, aligned, support)


def relative_depth_approach_rate(
    previous_depth: np.ndarray,
    aligned_current_depth: np.ndarray,
    delta_seconds: float,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Positive values mean depth decreased after scale-shift alignment."""
    if not math.isfinite(delta_seconds) or not 0 < delta_seconds <= 0.5:
        raise ValueError("SVRF frame delta must be in (0, 0.5] seconds")
    previous = np.asarray(previous_depth, dtype=np.float64)
    current = np.asarray(aligned_current_depth, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if previous.shape != current.shape or previous.shape != mask.shape:
        raise ValueError("SVRF approach-rate shapes must match")
    output = np.full(previous.shape, np.nan, dtype=np.float64)
    admitted = mask & np.isfinite(previous) & np.isfinite(current) & (previous > 1e-6) & (current > 1e-6)
    output[admitted] = -(np.log(current[admitted]) - np.log(previous[admitted])) / delta_seconds
    return output


@dataclass(frozen=True)
class ExpansionResult:
    valid: bool
    status: str
    expansion_per_second: float | None
    residual_mad_pixels: float | None
    support_count: int


def rotation_compensated_local_expansion(
    rotation_warped_previous_xy: np.ndarray,
    current_xy: np.ndarray,
    delta_seconds: float,
    *,
    minimum_support: int = 12,
    maximum_residual_mad_pixels: float = 1.5,
) -> ExpansionResult:
    """Fit residual local affine flow after a separately estimated rotation warp."""
    previous = np.asarray(rotation_warped_previous_xy, dtype=np.float64)
    current = np.asarray(current_xy, dtype=np.float64)
    if previous.shape != current.shape or previous.ndim != 2 or previous.shape[1] != 2:
        raise ValueError("SVRF local-flow coordinates must be Nx2 and shape-matched")
    if not math.isfinite(delta_seconds) or not 0 < delta_seconds <= 0.5:
        raise ValueError("SVRF frame delta must be in (0, 0.5] seconds")
    finite = np.all(np.isfinite(previous), axis=1) & np.all(np.isfinite(current), axis=1)
    previous, current = previous[finite], current[finite]
    if len(previous) < minimum_support:
        return ExpansionResult(False, "UNKNOWN_EXPANSION_SUPPORT", None, None, len(previous))
    center = np.median(previous, axis=0)
    design = np.column_stack((previous - center, np.ones(len(previous))))
    displacement = current - previous
    coefficients, _, rank, _ = np.linalg.lstsq(design, displacement, rcond=None)
    if rank < 3:
        return ExpansionResult(False, "UNKNOWN_EXPANSION_RANK", None, None, len(previous))
    predicted = design @ coefficients
    residual_norm = np.linalg.norm(displacement - predicted, axis=1)
    residual_mad = float(np.median(np.abs(residual_norm - np.median(residual_norm))))
    if residual_mad > maximum_residual_mad_pixels:
        return ExpansionResult(False, "UNKNOWN_EXPANSION_RESIDUAL", None, residual_mad, len(previous))
    spatial = coefficients[:2, :]
    expansion = float(0.5 * (spatial[0, 0] + spatial[1, 1]) / delta_seconds)
    return ExpansionResult(True, "VALID_ROTATION_COMPENSATED_EXPANSION", expansion, residual_mad, len(previous))


def fuse_region_risk(
    relative_nearness: float,
    depth_approach_rate: float,
    expansion_per_second: float,
    path_intrusion: float,
    observation_quality: float,
) -> dict[str, float | str]:
    values = (relative_nearness, depth_approach_rate, expansion_per_second, path_intrusion, observation_quality)
    if not all(math.isfinite(value) for value in values):
        return {"state": "UNKNOWN", "risk_score": 0.0, "consistency": 0.0}
    if not 0 <= relative_nearness <= 1 or not 0 <= path_intrusion <= 1 or not 0 <= observation_quality <= 1:
        raise ValueError("SVRF bounded region features must lie in [0,1]")
    if observation_quality < 0.60:
        return {"state": "UNKNOWN", "risk_score": 0.0, "consistency": 0.0}
    depth_signal = math.tanh(max(depth_approach_rate, 0.0) / 0.50)
    expansion_signal = math.tanh(max(expansion_per_second, 0.0) / 0.50)
    consistency = math.sqrt(depth_signal * expansion_signal)
    risk = (
        0.15 * relative_nearness
        + 0.25 * depth_signal
        + 0.20 * expansion_signal
        + 0.25 * path_intrusion
        + 0.15 * consistency
    ) * observation_quality
    if risk >= 0.70 and consistency >= 0.35 and path_intrusion >= 0.50:
        state = "HIGH_RELATIVE_RISK"
    elif path_intrusion >= 0.50 and (depth_signal >= 0.35 or expansion_signal >= 0.35):
        state = "PATH_INTRUSION"
    elif depth_signal >= 0.35 and expansion_signal >= 0.20:
        state = "APPROACHING"
    else:
        state = "NO_HIGH_RISK_EVIDENCE"
    return {"state": state, "risk_score": risk, "consistency": consistency}
