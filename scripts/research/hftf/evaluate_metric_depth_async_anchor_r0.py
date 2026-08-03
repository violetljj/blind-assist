#!/usr/bin/env python3
"""Evaluate a frozen low-rate HTP depth-anchor scheduler on consumed RGB-D truth.

The three propagation arms are parameter-free and causal. This diagnostic does
not search cadence, thresholds, history length, or an alert operating point.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HEIGHT = 392
WIDTH = 672
HISTORY_COUNT = 7
FUTURE_FRAME_OFFSET = 10
DEFAULT_SERVICE_TIME_S = 0.428058
MAX_SOURCE_DEPTH_MAE_M = 0.25
MAX_SOURCE_MEAN_RELATIVE_AE = 0.15
MIN_ASYNC_D44_OPPORTUNITIES = 8
MAX_ASYNC_DEPTH_MAE_INCREMENT_M = 0.10
MAX_ASYNC_D44_3D_ERROR_INCREMENT_M = 0.10
METHODS = ("latest_anchor_hold", "two_anchor_linear", "torso_height_ratio")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def robust_roi_median(depth: np.ndarray, roi: list[Any]) -> float:
    x0, y0, x1, y1 = (int(round(float(value))) for value in roi)
    values = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
    valid = values[np.isfinite(values) & (values > 0)]
    if not len(valid):
        raise ValueError("no valid torso depth")
    if len(valid) >= 10:
        lower, upper = np.quantile(valid, [0.1, 0.9])
        trimmed = valid[(valid >= lower) & (valid <= upper)]
        if len(trimmed):
            valid = trimmed
    return float(np.median(valid))


def restore_depth(
    canonical: np.ndarray, transform: dict[str, Any], fx: float
) -> np.ndarray:
    top = int(transform["pad_top"])
    left = int(transform["pad_left"])
    resized_height = int(transform["resized_height"])
    resized_width = int(transform["resized_width"])
    source_height = int(transform["source_height"])
    source_width = int(transform["source_width"])
    cropped = canonical[top : top + resized_height, left : left + resized_width]
    restored = cv2.resize(
        cropped, (source_width, source_height), interpolation=cv2.INTER_LINEAR
    )
    return restored * (fx * float(transform["scale"]) / 1000.0)


def relative_position(row: dict[str, Any], depth_m: float) -> np.ndarray:
    left, top, right, bottom = (float(value) for value in row["torso_roi_xyxy_px"])
    fx, fy, cx, cy = (float(value) for value in row["intrinsics_fx_fy_cx_cy"])
    u = (left + right) / 2.0
    v = (top + bottom) / 2.0
    return np.asarray(
        [depth_m, (u - cx) * depth_m / fx, -(v - cy) * depth_m / fy],
        dtype=np.float64,
    )


def ols_predict(
    timestamps_ns: list[int], positions: list[np.ndarray], target_timestamp_ns: int
) -> np.ndarray:
    times = np.asarray(timestamps_ns, dtype=np.float64) / 1e9
    values = np.stack(positions)
    centered = times - float(np.mean(times))
    denominator = float(np.dot(centered, centered))
    if len(times) != HISTORY_COUNT or denominator <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("invalid seven-frame history")
    slopes = centered @ values / denominator
    target = float(target_timestamp_ns) / 1e9
    return np.mean(values, axis=0) + slopes * (target - float(np.mean(times)))


def position_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "horizontal_error_m": float(np.linalg.norm(predicted[:2] - truth[:2])),
        "three_dimensional_error_m": float(np.linalg.norm(predicted - truth)),
        "absolute_range_error_m": abs(
            float(np.linalg.norm(predicted)) - float(np.linalg.norm(truth))
        ),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {"opportunities": 0}
    return {
        "opportunities": len(rows),
        **{
            f"mean_{key}": statistics.fmean(row[key] for row in rows)
            for key in rows[0]
        },
    }


def depth_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    paired = [row for row in rows if row.get(key) is not None]
    errors = [abs(float(row[key]) - float(row["truth_depth_m"])) for row in paired]
    relative = [error / float(row["truth_depth_m"]) for error, row in zip(errors, paired)]
    return {
        "frames": len(paired),
        "mae_m": statistics.fmean(errors),
        "median_ae_m": statistics.median(errors),
        "mean_relative_ae": statistics.fmean(relative),
        "max_ae_m": max(errors),
    }


def schedule(
    rows: list[dict[str, Any]], service_time_s: float
) -> list[dict[str, Any]]:
    times = [float(row["timestamp_ns"]) / 1e9 for row in rows]
    origin = times[0]
    relative_times = [value - origin for value in times]
    anchors = []
    source_index = 0
    start_time = relative_times[0]
    while True:
        completion = start_time + service_time_s
        anchors.append(
            {
                "source_frame": source_index,
                "source_time_s": relative_times[source_index],
                "completion_time_s": completion,
                "depth_m": float(rows[source_index]["full_rate_htp_depth_m"]),
            }
        )
        if completion > relative_times[-1]:
            break
        available = [
            index
            for index, timestamp in enumerate(relative_times)
            if timestamp <= completion
        ]
        next_source = max(available)
        if next_source <= source_index:
            later = [index for index in range(source_index + 1, len(rows))]
            if not later:
                break
            next_source = later[0]
            start_time = max(completion, relative_times[next_source])
        else:
            start_time = completion
        source_index = next_source
    return anchors


def propagate(rows: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> None:
    origin = float(rows[0]["timestamp_ns"]) / 1e9
    for row in rows:
        current_time = float(row["timestamp_ns"]) / 1e9 - origin
        available = [anchor for anchor in anchors if anchor["completion_time_s"] <= current_time]
        for method in METHODS:
            row[method] = None
        if not available:
            continue
        latest = available[-1]
        row["latest_anchor_hold"] = float(latest["depth_m"])
        if len(available) == 1:
            row["two_anchor_linear"] = float(latest["depth_m"])
        else:
            previous = available[-2]
            delta = float(latest["source_time_s"] - previous["source_time_s"])
            slope = (
                float(latest["depth_m"]) - float(previous["depth_m"])
            ) / delta
            row["two_anchor_linear"] = float(latest["depth_m"]) + slope * (
                current_time - float(latest["source_time_s"])
            )
        anchor_row = rows[int(latest["source_frame"])]
        anchor_height = float(anchor_row["torso_roi_xyxy_px"][3]) - float(
            anchor_row["torso_roi_xyxy_px"][1]
        )
        current_height = float(row["torso_roi_xyxy_px"][3]) - float(
            row["torso_roi_xyxy_px"][1]
        )
        row["torso_height_ratio"] = float(latest["depth_m"]) * anchor_height / current_height


def d44(rows: list[dict[str, Any]], depth_key: str) -> dict[str, Any]:
    predicted_rows = []
    static_rows = []
    opportunities = []
    by_frame = {int(row["frame_index"]): row for row in rows}
    for current in range(HISTORY_COUNT - 1, len(rows) - FUTURE_FRAME_OFFSET):
        history = [by_frame[index] for index in range(current - 6, current + 1)]
        future = by_frame[current + FUTURE_FRAME_OFFSET]
        if any(row.get(depth_key) is None for row in history):
            continue
        timestamps = [int(row["timestamp_ns"]) for row in history]
        positions = [relative_position(row, float(row[depth_key])) for row in history]
        truth = relative_position(future, float(future["truth_depth_m"]))
        prediction = ols_predict(timestamps, positions, int(future["timestamp_ns"]))
        predicted_metric = position_metrics(prediction, truth)
        static_metric = position_metrics(positions[-1], truth)
        predicted_rows.append(predicted_metric)
        static_rows.append(static_metric)
        opportunities.append(
            {
                "current_frame": current,
                "future_frame": current + FUTURE_FRAME_OFFSET,
                "d44": predicted_metric,
                "current_static": static_metric,
            }
        )
    d44_summary = aggregate(predicted_rows)
    static_summary = aggregate(static_rows)
    if predicted_rows:
        d44_summary["better_than_current_static_fraction_3d"] = statistics.fmean(
            predicted["three_dimensional_error_m"] < static["three_dimensional_error_m"]
            for predicted, static in zip(predicted_rows, static_rows)
        )
    return {"d44": d44_summary, "current_static": static_summary, "rows": opportunities}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--device-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--service-time-s", type=float, default=DEFAULT_SERVICE_TIME_S
    )
    args = parser.parse_args()
    if not math.isfinite(args.service_time_s) or args.service_time_s <= 0:
        raise ValueError("service time must be finite and positive")

    rows = sorted(load_jsonl(args.manifest), key=lambda row: int(row["frame_index"]))
    input_manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    transforms = {
        int(row["frame_index"]): row["transform"] for row in input_manifest["rows"]
    }
    if len(rows) != 30 or set(transforms) != set(range(30)):
        raise ValueError("expected frozen 30-frame Bonn cohort")
    if any(row.get("truth_source") != "registered_rgbd_sensor_depth_not_model_input" for row in rows):
        raise ValueError("truth firewall mismatch")

    for result_index, row in enumerate(rows):
        raw_path = args.device_output / f"Result_{result_index}" / "predicted_depth.raw"
        canonical = np.fromfile(raw_path, dtype=np.float32).reshape(HEIGHT, WIDTH)
        fx = float(row["intrinsics_fx_fy_cx_cy"][0])
        depth = restore_depth(canonical, transforms[int(row["frame_index"])], fx)
        row["full_rate_htp_depth_m"] = robust_roi_median(depth, row["torso_roi_xyxy_px"])

    anchors = schedule(rows, args.service_time_s)
    propagate(rows, anchors)
    full_depth = depth_metrics(rows, "full_rate_htp_depth_m")
    full_d44 = d44(rows, "full_rate_htp_depth_m")
    arms = {}
    for method in METHODS:
        method_depth = depth_metrics(rows, method)
        method_d44 = d44(rows, method)
        gates = {
            "minimum_d44_opportunities": method_d44["d44"]["opportunities"]
            >= MIN_ASYNC_D44_OPPORTUNITIES,
            "depth_mae_increment": method_depth["mae_m"] - full_depth["mae_m"]
            <= MAX_ASYNC_DEPTH_MAE_INCREMENT_M,
            "d44_3d_error_increment": method_d44["d44"]["mean_three_dimensional_error_m"]
            - full_d44["d44"]["mean_three_dimensional_error_m"]
            <= MAX_ASYNC_D44_3D_ERROR_INCREMENT_M,
        }
        arms[method] = {
            "depth": method_depth,
            "d44": method_d44,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        }
    passing = [method for method in METHODS if arms[method]["all_gates_pass"]]
    selected = min(
        passing,
        key=lambda method: arms[method]["d44"]["d44"]["mean_three_dimensional_error_m"],
        default=None,
    )
    source_gates = {
        "source_depth_mae": full_depth["mae_m"] <= MAX_SOURCE_DEPTH_MAE_M,
        "source_mean_relative_ae": full_depth["mean_relative_ae"]
        <= MAX_SOURCE_MEAN_RELATIVE_AE,
    }
    report = {
        "schema": "hftf_metric_depth_async_anchor_result_r0",
        "protocol": {
            "source_role": "already_consumed_bonn_rgbd_truth_diagnostic",
            "fresh_data_opened": False,
            "tuning_performed": False,
            "service_time_s": args.service_time_s,
            "service_time_scope": "mean cached RAFT-2 HTP execute only; preprocessing and transfer excluded; optimistic lower bound",
            "history_count": HISTORY_COUNT,
            "future_frame_offset": FUTURE_FRAME_OFFSET,
            "gates": {
                "max_source_depth_mae_m": MAX_SOURCE_DEPTH_MAE_M,
                "max_source_mean_relative_ae": MAX_SOURCE_MEAN_RELATIVE_AE,
                "min_async_d44_opportunities": MIN_ASYNC_D44_OPPORTUNITIES,
                "max_async_depth_mae_increment_m": MAX_ASYNC_DEPTH_MAE_INCREMENT_M,
                "max_async_d44_3d_error_increment_m": MAX_ASYNC_D44_3D_ERROR_INCREMENT_M,
            },
        },
        "source": {
            "full_rate_htp_depth": full_depth,
            "full_rate_htp_d44": full_d44,
            "gates": source_gates,
            "all_gates_pass": all(source_gates.values()),
        },
        "scheduler": {
            "anchors": anchors,
            "arms": arms,
            "selected_arm": selected,
            "supported": bool(selected is not None and all(source_gates.values())),
        },
        "rows": rows,
        "claim_ceiling": "single already-consumed Bonn sequence and execute-only scheduler simulation; no fresh, preprocessing-latency, multi-person, final-camera, alert, safety, or mainline authority",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source": report["source"],
                "selected_arm": selected,
                "scheduler_supported": report["scheduler"]["supported"],
                "arm_summaries": {
                    method: {
                        "depth": arms[method]["depth"],
                        "d44": arms[method]["d44"]["d44"],
                        "gates": arms[method]["gates"],
                    }
                    for method in METHODS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
