"""Evaluator-only interval truth and censored IVCA metrics.

Truth comes exclusively from source-native evaluator rows.  Prediction
receipts are never consulted while constructing contact intervals.  Because
CARLA evaluator truth is sampled, entry and exit remain interval-censored
between the last negative and first positive (and vice versa) instead of being
invented by interpolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import dtr_carla_ivca_interval_authority as ivca


EPSILON = 1e-9


@dataclass(frozen=True)
class TruthIntervalBounds:
    entry_lower_s: float
    entry_upper_s: float
    exit_lower_s: float
    exit_upper_s: float

    @property
    def midpoint_entry_s(self) -> float:
        return (self.entry_lower_s + self.entry_upper_s) * 0.5

    @property
    def midpoint_exit_s(self) -> float:
        return (self.exit_lower_s + self.exit_upper_s) * 0.5


@dataclass(frozen=True)
class RealizedIntervalTruth:
    components: tuple[TruthIntervalBounds, ...]
    sampled_minimum_clearance_m: float | None
    sampled_minimum_clearance_time_s: float | None
    temporal_resolution_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": [
                {
                    **asdict(component),
                    "midpoint_entry_s": component.midpoint_entry_s,
                    "midpoint_exit_s": component.midpoint_exit_s,
                }
                for component in self.components
            ],
            "sampled_minimum_clearance_m": self.sampled_minimum_clearance_m,
            "sampled_minimum_clearance_time_s": self.sampled_minimum_clearance_time_s,
            "temporal_resolution_s": self.temporal_resolution_s,
            "entry_exit_authority": "INTERVAL_CENSORED_SOURCE_NATIVE_SAMPLES",
        }


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _uniform_resolution(rows: Sequence[Mapping[str, Any]]) -> float:
    if len(rows) < 2:
        raise ValueError("at least two evaluator rows are required")
    deltas = [
        _finite(right["time_s"], "time_s") - _finite(left["time_s"], "time_s")
        for left, right in zip(rows, rows[1:])
    ]
    if min(deltas) <= 0.0:
        raise ValueError("evaluator times must be strictly increasing")
    resolution = sum(deltas) / len(deltas)
    if max(abs(value - resolution) for value in deltas) > 1e-6:
        raise ValueError("evaluator sampling must be uniform")
    return resolution


def realized_interval_truth(
    evaluator_rows: Sequence[Mapping[str, Any]],
    *,
    current_index: int,
    horizon_s: float,
    wearer_radius_m: float,
) -> RealizedIntervalTruth:
    """Construct future realized-contact bounds without prediction leakage."""

    rows = list(evaluator_rows)
    resolution = _uniform_resolution(rows)
    if current_index < 0 or current_index >= len(rows):
        raise IndexError("current_index is outside evaluator rows")
    horizon = _finite(horizon_s, "horizon_s")
    radius = _finite(wearer_radius_m, "wearer_radius_m")
    if horizon < 0.0 or radius <= 0.0:
        raise ValueError("horizon must be non-negative and wearer radius positive")
    current_time = _finite(rows[current_index]["time_s"], "current_time")
    window = [
        row
        for row in rows[current_index:]
        if _finite(row["time_s"], "time_s") - current_time <= horizon + EPSILON
    ]
    contacts = [bool(row["truth"]["current_contact"]) for row in window]
    components: list[TruthIntervalBounds] = []
    index = 0
    while index < len(window):
        if not contacts[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(window) and contacts[index + 1]:
            index += 1
        end = index
        first_positive = _finite(window[start]["time_s"], "entry_time") - current_time
        last_positive = _finite(window[end]["time_s"], "exit_time") - current_time
        if start == 0:
            entry_lower = entry_upper = 0.0
        else:
            entry_lower = _finite(window[start - 1]["time_s"], "entry_lower") - current_time
            entry_upper = first_positive
        if end + 1 < len(window):
            exit_lower = last_positive
            exit_upper = _finite(window[end + 1]["time_s"], "exit_upper") - current_time
        else:
            exit_lower = last_positive
            exit_upper = horizon
        components.append(
            TruthIntervalBounds(
                entry_lower_s=max(0.0, entry_lower),
                entry_upper_s=max(0.0, entry_upper),
                exit_lower_s=min(horizon, exit_lower),
                exit_upper_s=min(horizon, max(exit_lower, exit_upper)),
            )
        )
        index += 1

    clearance_values = [
        (
            _finite(row["truth"]["minimum_distance_m"], "minimum_distance_m") - radius,
            _finite(row["time_s"], "clearance_time") - current_time,
        )
        for row in window
        if row["truth"].get("minimum_distance_m") is not None
        and math.isfinite(float(row["truth"]["minimum_distance_m"]))
    ]
    minimum = min(clearance_values, default=None, key=lambda value: value[0])
    return RealizedIntervalTruth(
        components=tuple(components),
        sampled_minimum_clearance_m=None if minimum is None else minimum[0],
        sampled_minimum_clearance_time_s=None if minimum is None else minimum[1],
        temporal_resolution_s=resolution,
    )


def _interval_measure(values: Sequence[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end + EPSILON:
            end = max(end, next_end)
        else:
            total += max(0.0, end - start)
            start, end = next_start, next_end
    return total + max(0.0, end - start)


def _intersection_measure(
    first: Sequence[tuple[float, float]], second: Sequence[tuple[float, float]]
) -> float:
    return sum(
        max(0.0, min(a_end, b_end) - max(a_start, b_start))
        for a_start, a_end in first
        for b_start, b_end in second
    )


def _censored_error(value: float | None, lower: float, upper: float) -> float | None:
    if value is None:
        return None
    if lower - EPSILON <= value <= upper + EPSILON:
        return 0.0
    return min(abs(value - lower), abs(value - upper))


def score_interval_prediction(
    prediction: ivca.CollisionIntervalSet,
    truth: RealizedIntervalTruth,
) -> dict[str, Any]:
    predicted = [(value.entry_s, value.exit_s) for value in prediction.components]
    truth_midpoint = [
        (value.midpoint_entry_s, value.midpoint_exit_s) for value in truth.components
    ]
    intersection = _intersection_measure(predicted, truth_midpoint)
    union = _interval_measure((*predicted, *truth_midpoint))
    first_prediction = prediction.components[0] if prediction.components else None
    first_truth = truth.components[0] if truth.components else None
    predicted_minimum = min(
        (value.minimum_clearance_m for value in prediction.components), default=None
    )
    return {
        "midpoint_interval_iou": intersection / union if union > EPSILON else None,
        "false_onset_birth": bool(prediction.components) and not bool(truth.components),
        "missed_truth_interval": bool(truth.components) and not bool(prediction.components),
        "entry_censored_error_s": (
            None
            if first_truth is None
            else _censored_error(
                None if first_prediction is None else first_prediction.entry_s,
                first_truth.entry_lower_s,
                first_truth.entry_upper_s,
            )
        ),
        "exit_censored_error_s": (
            None
            if first_truth is None
            else _censored_error(
                None if first_prediction is None else first_prediction.exit_s,
                first_truth.exit_lower_s,
                first_truth.exit_upper_s,
            )
        ),
        "minimum_clearance_error_m": (
            None
            if predicted_minimum is None or truth.sampled_minimum_clearance_m is None
            else abs(predicted_minimum - truth.sampled_minimum_clearance_m)
        ),
        "predicted_component_count": len(prediction.components),
        "truth_component_count": len(truth.components),
        "truth_temporal_resolution_s": truth.temporal_resolution_s,
    }
