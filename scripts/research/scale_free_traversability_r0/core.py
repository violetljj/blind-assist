"""Frozen scale-free three-band relative traversability mechanics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


BANDS = {
    "left": (0.05, 0.35),
    "center": (0.35, 0.65),
    "right": (0.65, 0.95),
}
ROI_Y = (0.30, 0.90)
MINIMUM_VALID_FRACTION = 0.90
ROW_BASELINE_PERCENTILE = 25.0
BAND_SCORE_PERCENTILE = 85.0
CAUSAL_WINDOW = 5
MINIMUM_MARGIN_LOG_UNITS = 0.08
MINIMUM_WINNER_COUNT = 4


def _bounds(length: int, start: float, end: float) -> tuple[int, int]:
    left = max(0, min(length, int(round(length * start))))
    right = max(0, min(length, int(round(length * end))))
    if right <= left:
        raise ValueError("normalized bounds collapse at this resolution")
    return left, right


def score_relative_intrusion(depth: np.ndarray) -> dict[str, Any]:
    """Return scale-invariant per-band intrusion scores from positive depth."""

    values = np.asarray(depth, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 16:
        return {"status": "UNKNOWN", "reason": "INVALID_DEPTH_SHAPE"}
    height, width = values.shape
    y0, y1 = _bounds(height, *ROI_Y)
    x0, x1 = _bounds(width, 0.05, 0.95)
    roi = values[y0:y1, x0:x1]
    valid = np.isfinite(roi) & (roi > 0.0)
    valid_rows = np.mean(valid, axis=1) >= MINIMUM_VALID_FRACTION
    if float(np.mean(valid_rows)) < MINIMUM_VALID_FRACTION:
        return {"status": "UNKNOWN", "reason": "INSUFFICIENT_VALID_ROWS"}

    log_inverse = np.full(roi.shape, np.nan, dtype=np.float64)
    log_inverse[valid] = -np.log(roi[valid])
    baselines = np.full(roi.shape[0], np.nan, dtype=np.float64)
    for row_index in np.flatnonzero(valid_rows):
        baselines[row_index] = np.percentile(
            log_inverse[row_index, valid[row_index]], ROW_BASELINE_PERCENTILE
        )
    intrusion = np.maximum(0.0, log_inverse - baselines[:, None])

    scores: dict[str, float] = {}
    coverage: dict[str, float] = {}
    for name, (start, end) in BANDS.items():
        band_x0, band_x1 = _bounds(width, start, end)
        local_x0, local_x1 = band_x0 - x0, band_x1 - x0
        band_valid = valid[:, local_x0:local_x1] & valid_rows[:, None]
        fraction = float(np.mean(band_valid))
        coverage[name] = fraction
        if fraction < MINIMUM_VALID_FRACTION:
            return {
                "status": "UNKNOWN",
                "reason": f"INSUFFICIENT_{name.upper()}_COVERAGE",
                "coverage": coverage,
            }
        band_values = intrusion[:, local_x0:local_x1][band_valid]
        scores[name] = float(np.percentile(band_values, BAND_SCORE_PERCENTILE))
    return {"status": "VALID", "scores": scores, "coverage": coverage}


def decide_relative_open(history: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Apply the frozen five-frame causal stability and separation rule."""

    if len(history) < CAUSAL_WINDOW:
        return {"status": "UNKNOWN", "reason": "UNKNOWN_WARMUP"}
    window = list(history[-CAUSAL_WINDOW:])
    if any(set(row) != set(BANDS) for row in window):
        return {"status": "UNKNOWN", "reason": "INVALID_SCORE_HISTORY"}
    matrix = np.asarray([[row[name] for name in BANDS] for row in window], dtype=np.float64)
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        return {"status": "UNKNOWN", "reason": "INVALID_SCORE_HISTORY"}
    names = list(BANDS)
    smoothed = np.median(matrix, axis=0)
    order = np.argsort(smoothed, kind="stable")
    winner = names[int(order[0])]
    margin = float(smoothed[order[1]] - smoothed[order[0]])
    raw_winners = [names[int(index)] for index in np.argmin(matrix, axis=1)]
    winner_count = raw_winners.count(winner)
    label = (
        f"RELATIVELY_OPEN_{winner.upper()}"
        if margin >= MINIMUM_MARGIN_LOG_UNITS and winner_count >= MINIMUM_WINNER_COUNT
        else "AMBIGUOUS"
    )
    return {
        "status": "VALID",
        "label": label,
        "smoothed_scores": dict(zip(names, map(float, smoothed))),
        "margin_log_units": margin,
        "winner_count": winner_count,
    }
