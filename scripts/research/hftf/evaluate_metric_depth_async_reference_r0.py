#!/usr/bin/env python3
"""Evaluate causal sparse metric anchors against a consumed full-rate source."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


HISTORY_COUNT = 7
FUTURE_SECONDS = 1.0
SERVICE_TIME_S = 1.500794
MIN_AVAILABLE_FRAMES = 80
MAX_MEAN_DEPTH_DIFFERENCE_M = 0.35
MAX_MEAN_D44_POSITION_DIFFERENCE_M = 0.50
METHODS = (
    "latest_anchor_hold",
    "two_anchor_linear",
    "torso_height_ratio",
    "rebased_torso_history",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def position(row: dict[str, Any], depth_m: float) -> np.ndarray:
    left, top, right, bottom = (float(value) for value in row["torso_roi_xyxy_px"])
    fx, fy, cx, cy = (float(value) for value in row["intrinsics_fx_fy_cx_cy"])
    u, v = (left + right) / 2.0, (top + bottom) / 2.0
    return np.asarray(
        [depth_m, (u - cx) * depth_m / fx, -(v - cy) * depth_m / fy]
    )


def ols_predict(rows: list[dict[str, Any]], key: str, target_ns: int) -> np.ndarray:
    return ols_predict_depths(
        rows, [float(row[key]) for row in rows], target_ns
    )


def ols_predict_depths(
    rows: list[dict[str, Any]], depths: list[float], target_ns: int
) -> np.ndarray:
    times = np.asarray([int(row["timestamp_ns"]) for row in rows], np.float64) / 1e9
    values = np.stack(
        [position(row, depth) for row, depth in zip(rows, depths, strict=True)]
    )
    centered = times - float(np.mean(times))
    slopes = centered @ values / float(np.dot(centered, centered))
    return np.mean(values, axis=0) + slopes * (
        target_ns / 1e9 - float(np.mean(times))
    )


def build_schedule(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    origin = int(rows[0]["timestamp_ns"]) / 1e9
    times = [int(row["timestamp_ns"]) / 1e9 - origin for row in rows]
    anchors = []
    source = 0
    start = times[0]
    while True:
        completion = start + SERVICE_TIME_S
        anchors.append(
            {
                "source_frame": source,
                "source_time_s": times[source],
                "completion_time_s": completion,
                "depth_m": float(rows[source]["full_rate_depth_m"]),
            }
        )
        if completion > times[-1]:
            break
        source = max(index for index, timestamp in enumerate(times) if timestamp <= completion)
        start = completion
    return anchors


def propagate(rows: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> None:
    origin = int(rows[0]["timestamp_ns"]) / 1e9
    for row in rows:
        timestamp = int(row["timestamp_ns"]) / 1e9 - origin
        available = [anchor for anchor in anchors if anchor["completion_time_s"] <= timestamp]
        for method in METHODS:
            row[method] = None
        row["latest_anchor_source_frame"] = None
        if not available:
            continue
        latest = available[-1]
        row["latest_anchor_source_frame"] = int(latest["source_frame"])
        row["latest_anchor_hold"] = float(latest["depth_m"])
        if len(available) == 1:
            row["two_anchor_linear"] = float(latest["depth_m"])
        else:
            previous = available[-2]
            elapsed = float(latest["source_time_s"] - previous["source_time_s"])
            slope = (float(latest["depth_m"]) - float(previous["depth_m"])) / elapsed
            row["two_anchor_linear"] = float(latest["depth_m"]) + slope * (
                timestamp - float(latest["source_time_s"])
            )
        anchor_row = rows[int(latest["source_frame"])]
        anchor_height = float(anchor_row["torso_roi_xyxy_px"][3]) - float(
            anchor_row["torso_roi_xyxy_px"][1]
        )
        current_height = float(row["torso_roi_xyxy_px"][3]) - float(
            row["torso_roi_xyxy_px"][1]
        )
        row["torso_height_ratio"] = float(latest["depth_m"]) * anchor_height / current_height
        row["rebased_torso_history"] = row["torso_height_ratio"]


def evaluate_arm(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    available = [row for row in rows if row[key] is not None]
    depth_differences = [
        abs(float(row[key]) - float(row["full_rate_depth_m"])) for row in available
    ]
    forecast_differences = []
    opportunities = []
    timestamps = np.asarray([int(row["timestamp_ns"]) for row in rows], np.int64)
    for current in range(HISTORY_COUNT - 1, len(rows)):
        history = rows[current - HISTORY_COUNT + 1 : current + 1]
        if key == "rebased_torso_history":
            if rows[current]["latest_anchor_source_frame"] is None:
                continue
        elif any(row[key] is None for row in history):
            continue
        target_ns = int(rows[current]["timestamp_ns"] + FUTURE_SECONDS * 1e9)
        later = np.arange(current + 1, len(rows))
        if not len(later):
            continue
        target_index = int(later[np.argmin(np.abs(timestamps[later] - target_ns))])
        tolerance_ns = int(np.median(np.diff(timestamps)) / 2 + 1)
        if abs(int(timestamps[target_index]) - target_ns) > tolerance_ns:
            continue
        full = ols_predict(history, "full_rate_depth_m", int(timestamps[target_index]))
        if key == "rebased_torso_history":
            anchor_index = rows[current]["latest_anchor_source_frame"]
            if anchor_index is None:
                continue
            anchor_row = rows[int(anchor_index)]
            anchor_height = float(anchor_row["torso_roi_xyxy_px"][3]) - float(
                anchor_row["torso_roi_xyxy_px"][1]
            )
            anchor_depth = float(anchor_row["full_rate_depth_m"])
            rebased_depths = []
            for history_row in history:
                history_height = float(history_row["torso_roi_xyxy_px"][3]) - float(
                    history_row["torso_roi_xyxy_px"][1]
                )
                rebased_depths.append(anchor_depth * anchor_height / history_height)
            candidate = ols_predict_depths(
                history, rebased_depths, int(timestamps[target_index])
            )
        else:
            candidate = ols_predict(history, key, int(timestamps[target_index]))
        difference = float(np.linalg.norm(candidate - full))
        forecast_differences.append(difference)
        opportunities.append(
            {
                "current_frame": int(rows[current]["frame_index"]),
                "target_frame": int(rows[target_index]["frame_index"]),
                "position_difference_m": difference,
            }
        )
    summary = {
        "available_frames": len(available),
        "startup_unavailable_frames": len(rows) - len(available),
        "mean_depth_difference_m": statistics.fmean(depth_differences),
        "median_depth_difference_m": statistics.median(depth_differences),
        "max_depth_difference_m": max(depth_differences),
        "d44_opportunities": len(forecast_differences),
        "mean_d44_position_difference_m": statistics.fmean(forecast_differences),
        "median_d44_position_difference_m": statistics.median(forecast_differences),
        "max_d44_position_difference_m": max(forecast_differences),
        "opportunities": opportunities,
    }
    gates = {
        "available_frames": summary["available_frames"] >= MIN_AVAILABLE_FRAMES,
        "mean_depth_difference": summary["mean_depth_difference_m"]
        <= MAX_MEAN_DEPTH_DIFFERENCE_M,
        "mean_d44_position_difference": summary["mean_d44_position_difference_m"]
        <= MAX_MEAN_D44_POSITION_DIFFERENCE_M,
    }
    return {"summary": summary, "gates": gates, "all_gates_pass": all(gates.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests = sorted(load_jsonl(args.manifest), key=lambda row: int(row["frame_index"]))
    observations = {
        int(row["frame_index"]): row for row in load_jsonl(args.observations)
    }
    if len(manifests) != 130 or set(observations) != set(range(130)):
        raise ValueError("expected frozen 130-frame consumed sequence")
    rows = []
    for row in manifests:
        merged = dict(row)
        merged["full_rate_depth_m"] = float(
            observations[int(row["frame_index"])]["predicted_depth_m"]
        )
        rows.append(merged)
    anchors = build_schedule(rows)
    propagate(rows, anchors)
    arms = {method: evaluate_arm(rows, method) for method in METHODS}
    passing = [method for method in METHODS if arms[method]["all_gates_pass"]]
    selected = min(
        passing,
        key=lambda method: arms[method]["summary"]["mean_d44_position_difference_m"],
        default=None,
    )
    report = {
        "schema": "hftf_metric_depth_async_reference_result_r0",
        "protocol": {
            "source_role": "already_consumed_technical_sequence_no_outcome_truth_read",
            "fresh_data_opened": False,
            "tuning_performed": False,
            "service_time_s": SERVICE_TIME_S,
            "future_seconds": FUTURE_SECONDS,
            "history_count": HISTORY_COUNT,
            "gates": {
                "min_available_frames": MIN_AVAILABLE_FRAMES,
                "max_mean_depth_difference_m": MAX_MEAN_DEPTH_DIFFERENCE_M,
                "max_mean_d44_position_difference_m": MAX_MEAN_D44_POSITION_DIFFERENCE_M,
            },
        },
        "anchors": anchors,
        "arms": arms,
        "selected_arm": selected,
        "terminal": "SUPPORTED_CONSUMED_CONTINUITY_ONLY" if selected else "NOT_SUPPORTED",
        "claim_ceiling": "full-rate source continuity on one consumed sequence only; no metric truth, preprocessing latency, final camera, alert, safety, or mainline authority",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_arm": selected,
                "terminal": report["terminal"],
                "arms": {
                    method: {
                        key: item
                        for key, item in value["summary"].items()
                        if key != "opportunities"
                    }
                    | {"gates": value["gates"]}
                    for method, value in arms.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
