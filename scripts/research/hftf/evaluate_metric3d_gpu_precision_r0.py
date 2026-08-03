#!/usr/bin/env python3
"""Evaluate a frozen Metric3D GPU precision candidate on consumed RGB-D truth."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from run_external_rgb_metric_track_sidecar import relative_position

MAX_MEAN_DEPTH_DIFFERENCE_M = 0.05
MAX_MAX_DEPTH_DIFFERENCE_M = 0.10
MAX_MEAN_D44_DIFFERENCE_M = 0.10
MAX_STEADY_LATENCY_RATIO = 0.90
MAX_TRUTH_DEPTH_MAE_INCREMENT_M = 0.02
MAX_TRUTH_D44_ERROR_INCREMENT_M = 0.05
MIN_TRUTH_D44_WINDOWS = 8


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def one_track(row: dict[str, Any]) -> dict[str, Any]:
    tracks = row.get("tracks", [])
    if len(tracks) != 1:
        raise ValueError("precision canary requires exactly one frozen track per frame")
    return tracks[0]


def truth_position(row: dict[str, Any]) -> np.ndarray:
    return relative_position(
        row["torso_roi_xyxy_px"],
        row["intrinsics_fx_fy_cx_cy"],
        float(row["truth_depth_m"]),
    )


def summarize_truth(
    manifest: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> dict[str, Any]:
    by_key = {
        (str(row["sequence_id"]), int(row["timestamp_ns"])): row for row in manifest
    }
    timestamps_by_sequence = {
        sequence_id: np.asarray(
            sorted(
                int(row["timestamp_ns"])
                for row in manifest
                if str(row["sequence_id"]) == sequence_id
            ),
            dtype=np.int64,
        )
        for sequence_id in {str(row["sequence_id"]) for row in manifest}
    }
    depth_errors = []
    relative_errors = []
    d44_errors = []
    for source, output in zip(manifest, outputs, strict=True):
        track = one_track(output)
        truth_depth = float(source["truth_depth_m"])
        error = abs(float(track["depth_m"]) - truth_depth)
        depth_errors.append(error)
        relative_errors.append(error / truth_depth)
        forecast = track.get("d44_future_relative_position_m")
        if forecast is None:
            continue
        sequence_id = str(source["sequence_id"])
        timestamps = timestamps_by_sequence[sequence_id]
        tolerance_ns = int(statistics.median(np.diff(timestamps)) / 2 + 1)
        target_ns = int(track["d44_future_timestamp_ns"])
        nearest_index = int(np.argmin(np.abs(timestamps - target_ns)))
        nearest_ns = int(timestamps[nearest_index])
        if abs(nearest_ns - target_ns) > tolerance_ns:
            continue
        d44_errors.append(
            float(
                np.linalg.norm(
                    np.asarray(forecast, dtype=np.float64)
                    - truth_position(by_key[(sequence_id, nearest_ns)])
                )
            )
        )
    return {
        "frames": len(depth_errors),
        "depth_mae_m": statistics.fmean(depth_errors),
        "depth_mean_relative_absolute_error": statistics.fmean(relative_errors),
        "d44_windows": len(d44_errors),
        "d44_future_mean_3d_error_m": statistics.fmean(d44_errors),
        "d44_future_median_3d_error_m": statistics.median(d44_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_jsonl(args.manifest)
    baseline = load_jsonl(args.baseline)
    candidate = load_jsonl(args.candidate)
    if not (len(manifest) == len(baseline) == len(candidate)) or len(manifest) < 12:
        raise ValueError("manifest and precision arms must align with at least 12 frames")
    if any(
        not (
            int(source["frame_index"])
            == int(base["frame_index"])
            == int(test["frame_index"])
        )
        or not (
            str(source["sequence_id"])
            == str(base["sequence_id"])
            == str(test["sequence_id"])
        )
        for source, base, test in zip(manifest, baseline, candidate, strict=True)
    ):
        raise ValueError("frame identity mismatch")
    if any(
        row.get("truth_source") != "registered_rgbd_sensor_depth_not_model_input"
        for row in manifest
    ):
        raise ValueError("registered RGB-D truth firewall mismatch")

    baseline_depths = np.asarray(
        [float(one_track(row)["depth_m"]) for row in baseline], dtype=np.float64
    )
    candidate_depths = np.asarray(
        [float(one_track(row)["depth_m"]) for row in candidate], dtype=np.float64
    )
    if not np.all(np.isfinite(candidate_depths)):
        raise ValueError("candidate produced non-finite torso depth")
    depth_differences = np.abs(candidate_depths - baseline_depths)
    d44_differences = []
    for base, test in zip(baseline, candidate, strict=True):
        base_forecast = one_track(base).get("d44_future_relative_position_m")
        test_forecast = one_track(test).get("d44_future_relative_position_m")
        if base_forecast is None and test_forecast is None:
            continue
        if base_forecast is None or test_forecast is None:
            raise ValueError("D44 opportunity mismatch")
        difference = float(
            np.linalg.norm(
                np.asarray(test_forecast, dtype=np.float64)
                - np.asarray(base_forecast, dtype=np.float64)
            )
        )
        if not math.isfinite(difference):
            raise ValueError("candidate produced non-finite D44 difference")
        d44_differences.append(difference)

    baseline_latency = statistics.median(
        float(row["metric_depth_latency_ms"]) for row in baseline[1:]
    )
    candidate_latency = statistics.median(
        float(row["metric_depth_latency_ms"]) for row in candidate[1:]
    )
    baseline_truth = summarize_truth(manifest, baseline)
    candidate_truth = summarize_truth(manifest, candidate)
    summary = {
        "frames": len(manifest),
        "sequences": len({str(row["sequence_id"]) for row in manifest}),
        "baseline_steady_median_ms": baseline_latency,
        "candidate_steady_median_ms": candidate_latency,
        "steady_latency_ratio": candidate_latency / baseline_latency,
        "mean_depth_difference_m": float(np.mean(depth_differences)),
        "max_depth_difference_m": float(np.max(depth_differences)),
        "d44_opportunities": len(d44_differences),
        "mean_d44_difference_m": statistics.fmean(d44_differences),
        "max_d44_difference_m": max(d44_differences),
        "baseline_truth": baseline_truth,
        "candidate_truth": candidate_truth,
    }
    gates = {
        "mean_depth_difference": summary["mean_depth_difference_m"]
        <= MAX_MEAN_DEPTH_DIFFERENCE_M,
        "max_depth_difference": summary["max_depth_difference_m"]
        <= MAX_MAX_DEPTH_DIFFERENCE_M,
        "mean_d44_difference": summary["mean_d44_difference_m"]
        <= MAX_MEAN_D44_DIFFERENCE_M,
        "steady_latency_reduction": summary["steady_latency_ratio"]
        <= MAX_STEADY_LATENCY_RATIO,
        "minimum_truth_d44_windows": candidate_truth["d44_windows"]
        >= MIN_TRUTH_D44_WINDOWS,
        "truth_depth_mae_noninferiority": candidate_truth["depth_mae_m"]
        - baseline_truth["depth_mae_m"]
        <= MAX_TRUTH_DEPTH_MAE_INCREMENT_M,
        "truth_d44_error_noninferiority": candidate_truth[
            "d44_future_mean_3d_error_m"
        ]
        - baseline_truth["d44_future_mean_3d_error_m"]
        <= MAX_TRUTH_D44_ERROR_INCREMENT_M,
    }
    supported = all(gates.values())
    report = {
        "schema": "hftf_metric3d_gpu_precision_result_r0",
        "protocol": {
            "data_role": "already_consumed_bonn_rgbd_truth_diagnostic",
            "fresh_data_opened": False,
            "tuning_performed": False,
            "gates": {
                "max_mean_depth_difference_m": MAX_MEAN_DEPTH_DIFFERENCE_M,
                "max_max_depth_difference_m": MAX_MAX_DEPTH_DIFFERENCE_M,
                "max_mean_d44_difference_m": MAX_MEAN_D44_DIFFERENCE_M,
                "max_steady_latency_ratio": MAX_STEADY_LATENCY_RATIO,
                "max_truth_depth_mae_increment_m": MAX_TRUTH_DEPTH_MAE_INCREMENT_M,
                "max_truth_d44_error_increment_m": MAX_TRUTH_D44_ERROR_INCREMENT_M,
                "min_truth_d44_windows": MIN_TRUTH_D44_WINDOWS,
            },
        },
        "summary": summary,
        "gates": gates,
        "all_gates_pass": supported,
        "terminal": (
            "METRIC3D_VITS_CUDA_FP16_PRECISION_AND_LATENCY_SUPPORTED_CONSUMED_BONN_RGBD"
            if supported
            else "METRIC3D_VITS_CUDA_FP16_PRECISION_OR_LATENCY_NOT_SUPPORTED"
        ),
        "claim_ceiling": "already-consumed Bonn sequences only; no fresh, final-camera, alert, safety, mainline, or default-App authority",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
