#!/usr/bin/env python3
"""Compare metric-depth sources on explicit target observations.

The input is JSONL with one row per model/target/frame.  This evaluator is
deliberately detector-agnostic: a caller supplies the already selected target
depth so that depth-source error is not conflated with tracking error.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_hftf_external_rgb_metric_depth_source_r0"
HISTORY_COUNT = 7
DIRECTION_EPSILON_MPS = 0.05


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _ols_slope(timestamps_ns: list[int], values: list[float]) -> float:
    times = [(value - timestamps_ns[0]) / 1_000_000_000.0 for value in timestamps_ns]
    mean_time = statistics.fmean(times)
    mean_value = statistics.fmean(values)
    denominator = sum((value - mean_time) ** 2 for value in times)
    if denominator <= 0:
        raise ValueError("timestamps must span positive time")
    return sum(
        (time - mean_time) * (value - mean_value)
        for time, value in zip(times, values, strict=True)
    ) / denominator


def _direction(slope_mps: float) -> str:
    if slope_mps < -DIRECTION_EPSILON_MPS:
        return "approach"
    if slope_mps > DIRECTION_EPSILON_MPS:
        return "recede"
    return "stable_or_lateral"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {
            "model_id",
            "sequence_id",
            "frame_index",
            "timestamp_ns",
            "scenario",
            "camera_motion",
            "predicted_depth_m",
        }
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"line {line_number} missing fields: {missing}")
        truth_depth = row.get("truth_depth_m")
        truth_direction = row.get("truth_direction")
        if truth_depth is not None and (
            not _finite(truth_depth) or float(truth_depth) <= 0
        ):
            raise ValueError(
                f"line {line_number} truth_depth_m must be null or positive"
            )
        if truth_depth is None and truth_direction not in {
            "approach",
            "recede",
            "stable_or_lateral",
        }:
            raise ValueError(
                f"line {line_number} needs truth_depth_m or truth_direction"
            )
        rows.append(row)
    if not rows:
        raise ValueError("input contains no observations")
    return rows


def _sequence_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (int(row["timestamp_ns"]), int(row["frame_index"])))
    valid = [row for row in ordered if _finite(row["predicted_depth_m"]) and float(row["predicted_depth_m"]) > 0]
    total_windows = max(0, len(ordered) - HISTORY_COUNT + 1)
    valid_by_frame = {
        int(row["frame_index"])
        for row in valid
    }
    usable_windows = 0
    for start in range(total_windows):
        window = ordered[start : start + HISTORY_COUNT]
        frame_indices = [int(row["frame_index"]) for row in window]
        contiguous = all(
            right == left + 1
            for left, right in zip(frame_indices, frame_indices[1:])
        )
        if contiguous and all(index in valid_by_frame for index in frame_indices):
            usable_windows += 1

    result: dict[str, Any] = {
        "observations": len(ordered),
        "valid_observations": len(valid),
        "seven_frame_windows": total_windows,
        "usable_seven_frame_windows": usable_windows,
    }
    if len(valid) >= 2:
        timestamps = [int(row["timestamp_ns"]) for row in valid]
        predicted = [float(row["predicted_depth_m"]) for row in valid]
        predicted_slope = _ols_slope(timestamps, predicted)
        result["predicted_depth_slope_mps"] = predicted_slope
        result["predicted_direction"] = _direction(predicted_slope)
        explicit_directions = {
            str(row["truth_direction"])
            for row in valid
            if row.get("truth_direction") is not None
        }
        if len(explicit_directions) > 1:
            raise ValueError("sequence has conflicting truth_direction values")
        if explicit_directions:
            truth_direction = explicit_directions.pop()
            result["truth_direction"] = truth_direction
            result["direction_correct"] = (
                _direction(predicted_slope) == truth_direction
            )
        elif all(_finite(row.get("truth_depth_m")) for row in valid):
            truth = [float(row["truth_depth_m"]) for row in valid]
            truth_slope = _ols_slope(timestamps, truth)
            result["truth_depth_slope_mps"] = truth_slope
            result["truth_direction"] = _direction(truth_slope)
            result["direction_correct"] = (
                _direction(predicted_slope) == _direction(truth_slope)
            )
    if str(ordered[0]["scenario"]) == "static" and valid:
        predicted = [float(row["predicted_depth_m"]) for row in valid]
        median = statistics.median(predicted)
        result["static_median_depth_m"] = median
        result["static_mad_jitter_m"] = statistics.median(abs(value - median) for value in predicted)
        result["static_peak_to_peak_m"] = max(predicted) - min(predicted)
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model_id"])].append(row)

    models = {}
    for model_id, model_rows in sorted(by_model.items()):
        valid = [
            row
            for row in model_rows
            if _finite(row["predicted_depth_m"])
            and float(row["predicted_depth_m"]) > 0
        ]
        error_rows = [
            row for row in valid if _finite(row.get("truth_depth_m"))
        ]
        absolute_errors = [
            abs(
                float(row["predicted_depth_m"])
                - float(row["truth_depth_m"])
            )
            for row in error_rows
        ]
        relative_errors = [
            error / float(row["truth_depth_m"])
            for error, row in zip(absolute_errors, error_rows, strict=True)
        ]
        latency_ms = [float(row["latency_ms"]) for row in valid if _finite(row.get("latency_ms"))]
        sequence_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in model_rows:
            key = (str(row["sequence_id"]), str(row["scenario"]), str(row["camera_motion"]))
            sequence_groups[key].append(row)
        sequences = {
            "|".join(key): _sequence_metrics(sequence_rows)
            for key, sequence_rows in sorted(sequence_groups.items())
        }
        cold_start_latency_ms = []
        steady_state_latency_ms = []
        for sequence_rows in sequence_groups.values():
            ordered_valid = [
                row
                for row in sorted(
                    sequence_rows,
                    key=lambda value: int(value["timestamp_ns"]),
                )
                if _finite(row.get("predicted_depth_m"))
                and float(row["predicted_depth_m"]) > 0
                and _finite(row.get("latency_ms"))
            ]
            if ordered_valid:
                cold_start_latency_ms.append(
                    float(ordered_valid[0]["latency_ms"])
                )
                steady_state_latency_ms.extend(
                    float(row["latency_ms"])
                    for row in ordered_valid[1:]
                )
        process_rss_mib = [
            float(row["process_rss_mib"])
            for row in valid
            if _finite(row.get("process_rss_mib"))
        ]
        cuda_peak_mib = [
            float(row["cuda_peak_allocated_mib"])
            for row in valid
            if _finite(row.get("cuda_peak_allocated_mib"))
        ]
        direction_rows = [value for value in sequences.values() if "direction_correct" in value]
        total_windows = sum(int(value["seven_frame_windows"]) for value in sequences.values())
        usable_windows = sum(int(value["usable_seven_frame_windows"]) for value in sequences.values())
        static_jitter = [
            float(value["static_mad_jitter_m"])
            for value in sequences.values()
            if "static_mad_jitter_m" in value
        ]
        models[model_id] = {
            "observations": len(model_rows),
            "valid_observations": len(valid),
            "metric_truth_observations": len(error_rows),
            "valid_fraction": len(valid) / len(model_rows),
            "mean_absolute_error_m": statistics.fmean(absolute_errors) if absolute_errors else None,
            "median_absolute_error_m": statistics.median(absolute_errors) if absolute_errors else None,
            "mean_relative_error": statistics.fmean(relative_errors) if relative_errors else None,
            "median_relative_error": statistics.median(relative_errors) if relative_errors else None,
            "seven_frame_availability": usable_windows / total_windows if total_windows else None,
            "direction_accuracy": (
                sum(bool(value["direction_correct"]) for value in direction_rows) / len(direction_rows)
                if direction_rows else None
            ),
            "median_static_mad_jitter_m": statistics.median(static_jitter) if static_jitter else None,
            "mean_latency_ms": statistics.fmean(latency_ms) if latency_ms else None,
            "p95_latency_ms": _percentile(latency_ms, 0.95),
            "mean_cold_start_latency_ms": (
                statistics.fmean(cold_start_latency_ms)
                if cold_start_latency_ms
                else None
            ),
            "mean_steady_state_latency_ms": (
                statistics.fmean(steady_state_latency_ms)
                if steady_state_latency_ms
                else None
            ),
            "p95_steady_state_latency_ms": _percentile(
                steady_state_latency_ms, 0.95
            ),
            "max_process_rss_mib": (
                max(process_rss_mib) if process_rss_mib else None
            ),
            "max_cuda_peak_allocated_mib": (
                max(cuda_peak_mib) if cuda_peak_mib else None
            ),
            "sequences": sequences,
        }
    return {"schema": SCHEMA, "models": models}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observations", type=Path, nargs="+", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        row
        for observations in args.observations
        for row in load_rows(observations)
    ]
    report = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
