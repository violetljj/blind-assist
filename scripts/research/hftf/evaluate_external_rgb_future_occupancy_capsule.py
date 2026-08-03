#!/usr/bin/env python3
"""Calibrate and evaluate a set-valued one-second person forecast.

The candidate is the horizontal capsule joining the current relative position
to the seven-frame OLS endpoint. Its radius and both point-baseline radii are
fixed by split-conformal calibration before evaluation rows are read.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_external_rgb_metric_track_d44 import (
    FUTURE_FRAME_OFFSET,
    HISTORY_COUNT,
    load_jsonl,
    ols_predict,
    relative_position,
)


SCHEMA = "blindassist_hftf_external_rgb_future_occupancy_capsule_r0"
STATIC_DISK = "current_static_disk"
OLS_DISK = "ols_endpoint_disk"
MOTION_CAPSULE = "current_to_ols_motion_capsule"
ARMS = (STATIC_DISK, OLS_DISK, MOTION_CAPSULE)
SUPPORTED = "FUTURE_OCCUPANCY_CAPSULE_SUPPORTED_DEVELOPMENT_ONLY"
NOT_SUPPORTED = "FUTURE_OCCUPANCY_CAPSULE_NOT_SUPPORTED"


def point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, delta) / denominator)
    fraction = min(1.0, max(0.0, fraction))
    closest = start + fraction * delta
    return float(np.linalg.norm(point - closest))


def conformal_radius(scores: list[float], alpha: float) -> tuple[float, int]:
    if not scores:
        raise ValueError("conformal calibration requires scores")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not all(math.isfinite(score) and score >= 0 for score in scores):
        raise ValueError("conformal scores must be finite and non-negative")
    rank = min(len(scores), math.ceil((len(scores) + 1) * (1.0 - alpha)))
    return sorted(scores)[rank - 1], rank


def _observation_index(
    observations: list[dict[str, Any]] | None,
) -> tuple[dict[tuple[str, int], dict[str, Any]] | None, str]:
    if observations is None:
        return None, "registered_sensor_depth_oracle"
    index: dict[tuple[str, int], dict[str, Any]] = {}
    model_ids = set()
    for row in observations:
        key = (str(row["sequence_id"]), int(row["frame_index"]))
        if key in index:
            raise ValueError(f"duplicate observation: {key}")
        index[key] = row
        model_ids.add(str(row["model_id"]))
    if len(model_ids) != 1:
        raise ValueError("observations must contain exactly one model_id")
    return index, model_ids.pop()


def build_opportunities(
    manifests: list[dict[str, Any]],
    observations: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    observation_by_key, source = _observation_index(observations)
    sequences: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in manifests:
        sequence = str(row["sequence_id"])
        frame = int(row["frame_index"])
        if frame in sequences[sequence]:
            raise ValueError(f"duplicate manifest row: {(sequence, frame)}")
        sequences[sequence][frame] = row
    rows = []
    for sequence, frames in sorted(sequences.items()):
        for current_frame in sorted(frames):
            history_indices = list(
                range(current_frame - HISTORY_COUNT + 1, current_frame + 1)
            )
            future_frame = current_frame + FUTURE_FRAME_OFFSET
            if any(index not in frames for index in history_indices):
                continue
            if future_frame not in frames:
                continue
            history_rows = [frames[index] for index in history_indices]
            history_positions = []
            for index, manifest in zip(
                history_indices, history_rows, strict=True
            ):
                if observation_by_key is None:
                    depth = float(manifest["truth_depth_m"])
                else:
                    observation = observation_by_key.get((sequence, index))
                    if observation is None:
                        raise ValueError(
                            f"missing observation for {(sequence, index)}"
                        )
                    depth = float(observation["predicted_depth_m"])
                history_positions.append(relative_position(manifest, depth))
            future_manifest = frames[future_frame]
            truth = relative_position(
                future_manifest, float(future_manifest["truth_depth_m"])
            )[:2]
            current = history_positions[-1][:2]
            endpoint = ols_predict(
                [int(row["timestamp_ns"]) for row in history_rows],
                history_positions,
                int(future_manifest["timestamp_ns"]),
            )[:2]
            rows.append(
                {
                    "sequence_id": sequence,
                    "distances": {
                        STATIC_DISK: float(np.linalg.norm(truth - current)),
                        OLS_DISK: float(np.linalg.norm(truth - endpoint)),
                        MOTION_CAPSULE: point_segment_distance(
                            truth, current, endpoint
                        ),
                    },
                    "capsule_segment_length_m": float(
                        np.linalg.norm(endpoint - current)
                    ),
                }
            )
    if not rows:
        raise ValueError("no seven-frame plus future opportunities")
    return rows, source


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), percentile))


def summarize_arm(
    rows: list[dict[str, Any]], arm: str, radius: float
) -> dict[str, Any]:
    areas = []
    covered = []
    excess = []
    by_sequence: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        distance = float(row["distances"][arm])
        length = (
            float(row["capsule_segment_length_m"])
            if arm == MOTION_CAPSULE
            else 0.0
        )
        areas.append(math.pi * radius * radius + 2.0 * radius * length)
        is_covered = distance <= radius + 1e-12
        covered.append(is_covered)
        excess.append(max(0.0, distance - radius))
        by_sequence[str(row["sequence_id"])].append(is_covered)
    return {
        "radius_m": radius,
        "coverage": statistics.fmean(covered),
        "mean_area_m2": statistics.fmean(areas),
        "median_area_m2": statistics.median(areas),
        "p95_area_m2": _percentile(areas, 0.95),
        "mean_excess_distance_m": statistics.fmean(excess),
        "max_excess_distance_m": max(excess),
        "sequence_coverage": {
            sequence: statistics.fmean(values)
            for sequence, values in sorted(by_sequence.items())
        },
    }


def _reduction(baseline: float, candidate: float) -> float | None:
    return (baseline - candidate) / baseline if baseline > 0 else None


def calibrate_and_evaluate(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    radii = {}
    ranks = {}
    for arm in ARMS:
        radii[arm], ranks[arm] = conformal_radius(
            [float(row["distances"][arm]) for row in calibration], alpha
        )
    arms = {
        arm: summarize_arm(evaluation, arm, radii[arm]) for arm in ARMS
    }
    candidate = arms[MOTION_CAPSULE]
    static = arms[STATIC_DISK]
    ols = arms[OLS_DISK]
    effects = {
        "mean_area_reduction_vs_static": _reduction(
            static["mean_area_m2"], candidate["mean_area_m2"]
        ),
        "mean_area_reduction_vs_ols": _reduction(
            ols["mean_area_m2"], candidate["mean_area_m2"]
        ),
        "median_area_reduction_vs_static": _reduction(
            static["median_area_m2"], candidate["median_area_m2"]
        ),
        "median_area_reduction_vs_ols": _reduction(
            ols["median_area_m2"], candidate["median_area_m2"]
        ),
    }
    gates = {
        "coverage_at_least_0_85": candidate["coverage"] >= 0.85,
        "mean_area_20pct_below_static": (
            effects["mean_area_reduction_vs_static"] is not None
            and effects["mean_area_reduction_vs_static"] >= 0.20
        ),
        "mean_area_20pct_below_ols": (
            effects["mean_area_reduction_vs_ols"] is not None
            and effects["mean_area_reduction_vs_ols"] >= 0.20
        ),
        "median_area_20pct_below_static": (
            effects["median_area_reduction_vs_static"] is not None
            and effects["median_area_reduction_vs_static"] >= 0.20
        ),
        "median_area_20pct_below_ols": (
            effects["median_area_reduction_vs_ols"] is not None
            and effects["median_area_reduction_vs_ols"] >= 0.20
        ),
        "mean_excess_no_worse_than_static": (
            candidate["mean_excess_distance_m"]
            <= static["mean_excess_distance_m"] + 1e-12
        ),
        "mean_excess_no_worse_than_ols": (
            candidate["mean_excess_distance_m"]
            <= ols["mean_excess_distance_m"] + 1e-12
        ),
    }
    return {
        "calibration_opportunities": len(calibration),
        "evaluation_opportunities": len(evaluation),
        "target_coverage": 1.0 - alpha,
        "conformal_ranks": ranks,
        "arms": arms,
        "effects": effects,
        "gates": gates,
        "status": SUPPORTED if all(gates.values()) else NOT_SUPPORTED,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-manifest", type=Path, nargs="+", required=True
    )
    parser.add_argument(
        "--evaluation-manifest", type=Path, nargs="+", required=True
    )
    parser.add_argument("--calibration-observations", type=Path, nargs="+")
    parser.add_argument("--evaluation-observations", type=Path, nargs="+")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration, calibration_source = build_opportunities(
        load_jsonl(args.calibration_manifest),
        load_jsonl(args.calibration_observations)
        if args.calibration_observations
        else None,
    )
    evaluation, evaluation_source = build_opportunities(
        load_jsonl(args.evaluation_manifest),
        load_jsonl(args.evaluation_observations)
        if args.evaluation_observations
        else None,
    )
    if calibration_source != evaluation_source:
        raise ValueError(
            "calibration and evaluation must use the same track source"
        )
    result = calibrate_and_evaluate(calibration, evaluation, args.alpha)
    report = {
        "schema": SCHEMA,
        "history_count": HISTORY_COUNT,
        "future_frame_offset": FUTURE_FRAME_OFFSET,
        "source": calibration_source,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
