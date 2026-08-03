#!/usr/bin/env python3
"""Evaluate a seven-frame D44 forecast from RGB-derived metric target tracks."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HISTORY_COUNT = 7
FUTURE_FRAME_OFFSET = 10
SCHEMA = "blindassist_hftf_external_rgb_metric_track_d44_r0"


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def relative_position(
    row: dict[str, Any], depth_m: float
) -> np.ndarray:
    if not math.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth must be finite and positive")
    roi = row.get("torso_roi_xyxy_px")
    intrinsics = row.get("intrinsics_fx_fy_cx_cy")
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError("manifest row needs four-value torso ROI")
    if not isinstance(intrinsics, list) or len(intrinsics) != 4:
        raise ValueError("manifest row needs four-value intrinsics")
    left, top, right, bottom = (float(value) for value in roi)
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if right <= left or bottom <= top or fx <= 0 or fy <= 0:
        raise ValueError("invalid ROI or intrinsics")
    u = (left + right) / 2.0
    v = (top + bottom) / 2.0
    lateral = (u - cx) * depth_m / fx
    vertical_up = -(v - cy) * depth_m / fy
    return np.asarray([depth_m, lateral, vertical_up], dtype=np.float64)


def ols_predict(
    timestamps_ns: list[int],
    positions: list[np.ndarray],
    target_timestamp_ns: int,
) -> np.ndarray:
    if len(timestamps_ns) != HISTORY_COUNT or len(positions) != HISTORY_COUNT:
        raise ValueError("D44 requires seven history observations")
    times = np.asarray(timestamps_ns, dtype=np.float64) / 1e9
    if np.any(np.diff(times) <= 0):
        raise ValueError("history timestamps must increase")
    target = target_timestamp_ns / 1e9
    if target <= times[-1]:
        raise ValueError("target timestamp must be in the future")
    values = np.stack(positions)
    centered_times = times - float(np.mean(times))
    denominator = float(np.dot(centered_times, centered_times))
    if denominator <= 0:
        raise ValueError("degenerate history timestamps")
    slopes = centered_times @ values / denominator
    return np.mean(values, axis=0) + slopes * (target - float(np.mean(times)))


def bearing_error_deg(prediction: np.ndarray, truth: np.ndarray) -> float:
    predicted = math.atan2(float(prediction[1]), float(prediction[0]))
    actual = math.atan2(float(truth[1]), float(truth[0]))
    delta = math.atan2(math.sin(predicted - actual), math.cos(predicted - actual))
    return abs(math.degrees(delta))


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "horizontal_error_m": float(np.linalg.norm(prediction[:2] - truth[:2])),
        "three_dimensional_error_m": float(np.linalg.norm(prediction - truth)),
        "absolute_range_error_m": abs(
            float(np.linalg.norm(prediction)) - float(np.linalg.norm(truth))
        ),
        "absolute_bearing_error_deg": bearing_error_deg(prediction, truth),
    }


def reduction(baseline: float, candidate: float) -> float | None:
    return (baseline - candidate) / baseline if baseline > 0 else None


def summarize_opportunities(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = ("current_static", "d44_ols", "oracle_depth_d44_ols")
    summary: dict[str, Any] = {"opportunities": len(rows), "arms": {}}
    for arm in arms:
        arm_rows = [row[arm] for row in rows]
        summary["arms"][arm] = {
            key: statistics.fmean(float(row[key]) for row in arm_rows)
            for key in (
                "horizontal_error_m",
                "three_dimensional_error_m",
                "absolute_range_error_m",
                "absolute_bearing_error_deg",
            )
        }
        summary["arms"][arm]["median_horizontal_error_m"] = statistics.median(
            float(row["horizontal_error_m"]) for row in arm_rows
        )
    baseline = summary["arms"]["current_static"]
    candidate = summary["arms"]["d44_ols"]
    summary["effect"] = {
        "mean_horizontal_error_relative_reduction": reduction(
            baseline["horizontal_error_m"], candidate["horizontal_error_m"]
        ),
        "median_horizontal_error_relative_reduction": reduction(
            baseline["median_horizontal_error_m"],
            candidate["median_horizontal_error_m"],
        ),
        "mean_three_dimensional_error_relative_reduction": reduction(
            baseline["three_dimensional_error_m"],
            candidate["three_dimensional_error_m"],
        ),
        "mean_range_error_relative_reduction": reduction(
            baseline["absolute_range_error_m"],
            candidate["absolute_range_error_m"],
        ),
        "mean_bearing_error_relative_reduction": reduction(
            baseline["absolute_bearing_error_deg"],
            candidate["absolute_bearing_error_deg"],
        ),
        "horizontal_error_better_fraction": statistics.fmean(
            row["d44_ols"]["horizontal_error_m"]
            < row["current_static"]["horizontal_error_m"]
            for row in rows
        ),
    }
    return summary


def evaluate(
    manifests: list[dict[str, Any]], observations: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manifests:
        key = (str(row["sequence_id"]), int(row["frame_index"]))
        if key in manifest_by_key:
            raise ValueError(f"duplicate manifest key: {key}")
        manifest_by_key[key] = row

    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in observations:
        key = (str(row["sequence_id"]), int(row["frame_index"]))
        if key not in manifest_by_key:
            raise ValueError(f"observation has no manifest row: {key}")
        group = (str(row["model_id"]), key[0])
        if key[1] in grouped[group]:
            raise ValueError(f"duplicate observation: {group} frame {key[1]}")
        grouped[group][key[1]] = row

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_sequence: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (model_id, sequence_id), frames in sorted(grouped.items()):
        for current_frame in sorted(frames):
            history_indices = list(
                range(current_frame - HISTORY_COUNT + 1, current_frame + 1)
            )
            future_frame = current_frame + FUTURE_FRAME_OFFSET
            if any(index not in frames for index in history_indices):
                continue
            if future_frame not in frames:
                continue
            history_manifest = [
                manifest_by_key[(sequence_id, index)] for index in history_indices
            ]
            future_manifest = manifest_by_key[(sequence_id, future_frame)]
            history_observations = [frames[index] for index in history_indices]
            future_truth = relative_position(
                future_manifest, float(future_manifest["truth_depth_m"])
            )
            predicted_history = [
                relative_position(manifest, float(observation["predicted_depth_m"]))
                for manifest, observation in zip(
                    history_manifest, history_observations, strict=True
                )
            ]
            truth_history = [
                relative_position(manifest, float(manifest["truth_depth_m"]))
                for manifest in history_manifest
            ]
            timestamps = [int(row["timestamp_ns"]) for row in history_manifest]
            target_timestamp = int(future_manifest["timestamp_ns"])
            opportunity = {
                "current_static": metrics(predicted_history[-1], future_truth),
                "d44_ols": metrics(
                    ols_predict(timestamps, predicted_history, target_timestamp),
                    future_truth,
                ),
                "oracle_depth_d44_ols": metrics(
                    ols_predict(timestamps, truth_history, target_timestamp),
                    future_truth,
                ),
            }
            by_model[model_id].append(opportunity)
            by_model_sequence[model_id][sequence_id].append(opportunity)

    models = {}
    for model_id, rows in sorted(by_model.items()):
        models[model_id] = summarize_opportunities(rows)
        models[model_id]["sequences"] = {
            sequence: summarize_opportunities(sequence_rows)
            for sequence, sequence_rows in sorted(
                by_model_sequence[model_id].items()
            )
        }
    return {
        "schema": SCHEMA,
        "history_count": HISTORY_COUNT,
        "future_frame_offset": FUTURE_FRAME_OFFSET,
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load_jsonl(args.manifest), load_jsonl(args.observations))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
