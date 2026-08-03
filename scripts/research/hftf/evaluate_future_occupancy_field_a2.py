#!/usr/bin/env python3
"""Window-LOSO evaluation of causal 0.5-second future occupancy fields."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_collision_risk_field_a1 import binary_metrics
from evaluate_metric3d_clearance_field_a0 import BANDS, HORIZONS_M
from evaluate_metric3d_probabilistic_occupancy_a0 import (
    _log_loss,
    expected_calibration_error,
)
from evaluate_motion_conditioned_occupancy_a0 import (
    FEATURE_NAMES as CURRENT_FEATURE_NAMES,
    extract_motion,
    fit_logistic,
    predict_logistic,
)


SCHEMA = "blindassist_hftf_future_occupancy_field_a2"
LEAD_FRAMES = 5
HISTORY_FRAMES = 7
LEAD_SECONDS = 0.5
FUTURE_FEATURE_NAMES = (
    "history_fraction",
    "clearance_slope_mps",
    "clearance_acceleration_mps2",
    "cv_residual_rms_m",
    "ca_residual_rms_m",
    "hold_future_margin_m",
    "cv_future_margin_m",
    "ca_future_margin_m",
    "imm_occupancy_probability",
)
FEATURE_NAMES = tuple(CURRENT_FEATURE_NAMES) + FUTURE_FEATURE_NAMES


def sigmoid_margin(horizon: float, clearance: float, scale: float) -> float:
    value = (horizon - clearance) / scale
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40, 40))))


def motion_modes(times: np.ndarray, values: np.ndarray, horizon: float) -> dict[str, float]:
    if len(values) == 0:
        raise ValueError("at least one history value is required")
    future_time = float(times[-1] + LEAD_SECONDS)
    hold = float(values[-1])
    hold_rmse = float(np.sqrt(np.mean((values - hold) ** 2)))

    if len(values) >= 2:
        cv_coefficients = np.polyfit(times, values, 1)
        cv_fitted = np.polyval(cv_coefficients, times)
        cv = float(np.polyval(cv_coefficients, future_time))
        slope = float(cv_coefficients[0])
        cv_rmse = float(np.sqrt(np.mean((values - cv_fitted) ** 2)))
    else:
        cv, slope, cv_rmse = hold, 0.0, hold_rmse

    if len(values) >= 5:
        ca_coefficients = np.polyfit(times, values, 2)
        ca_fitted = np.polyval(ca_coefficients, times)
        ca = float(np.polyval(ca_coefficients, future_time))
        acceleration = float(2.0 * ca_coefficients[0])
        ca_rmse = float(np.sqrt(np.mean((values - ca_fitted) ** 2)))
    else:
        ca, acceleration, ca_rmse = cv, 0.0, cv_rmse

    mode_probabilities = np.asarray(
        [
            sigmoid_margin(horizon, hold, 0.20),
            sigmoid_margin(horizon, cv, 0.25),
            sigmoid_margin(horizon, ca, 0.35),
        ]
    )
    priors = np.asarray([0.30, 0.50, 0.20])
    residuals = np.asarray([hold_rmse, cv_rmse, ca_rmse])
    weights = priors * np.exp(-residuals / 0.15)
    weights /= np.sum(weights)
    return {
        "hold": hold,
        "cv": cv,
        "ca": ca,
        "slope": slope,
        "acceleration": acceleration,
        "cv_rmse": cv_rmse,
        "ca_rmse": ca_rmse,
        "hold_probability": float(mode_probabilities[0]),
        "cv_probability": float(mode_probabilities[1]),
        "ca_probability": float(mode_probabilities[2]),
        "imm_probability": float(weights @ mode_probabilities),
    }


def build_future_rows(
    reports: list[dict[str, Any]], motion: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        for frame in report["frames"]:
            by_sequence[str(frame["sequence_id"])].append(frame)

    features = []
    labels = []
    groups = []
    baselines: dict[str, list[float]] = {
        "hold": [],
        "cv": [],
        "ca": [],
        "imm": [],
    }
    for sequence, frames in sorted(by_sequence.items()):
        frames.sort(key=lambda value: float(value["timestamp"]))
        for index in range(0, len(frames) - LEAD_FRAMES):
            current = frames[index]
            future = frames[index + LEAD_FRAMES]
            if current["metric3d"]["status"] != "VALID" or future["sensor"]["status"] != "VALID":
                continue
            for band in BANDS:
                current_band = current["metric3d"]["bands"][band]
                predicted = current_band["clearance_m"]
                confidence = current_band.get("clearance_log1p_confidence")
                truth = future["sensor"]["bands"][band]["clearance_m"]
                if predicted is None or confidence is None or truth is None:
                    continue
                history_times = []
                history_values = []
                for prior in frames[max(0, index - HISTORY_FRAMES + 1) : index + 1]:
                    if prior["metric3d"]["status"] != "VALID":
                        continue
                    value = prior["metric3d"]["bands"][band]["clearance_m"]
                    if value is not None:
                        history_times.append(float(prior["timestamp"]))
                        history_values.append(float(value))
                times = np.asarray(history_times, dtype=np.float64)
                times -= times[-1]
                values = np.asarray(history_values, dtype=np.float64)
                for horizon in HORIZONS_M:
                    modes = motion_modes(times, values, horizon)
                    current_features = np.asarray(
                        [
                            float(predicted) - horizon,
                            float(predicted),
                            horizon,
                            float(confidence),
                            float(current["metric3d"]["ground_plane_median_residual_m"]),
                            math.log1p(int(current_band["obstacle_points"])),
                            float(band == "left"),
                            float(band == "center"),
                        ],
                        dtype=np.float64,
                    )
                    future_features = np.asarray(
                        [
                            len(values) / HISTORY_FRAMES,
                            modes["slope"],
                            modes["acceleration"],
                            modes["cv_rmse"],
                            modes["ca_rmse"],
                            modes["hold"] - horizon,
                            modes["cv"] - horizon,
                            modes["ca"] - horizon,
                            modes["imm_probability"],
                        ],
                        dtype=np.float64,
                    )
                    features.append(
                        np.concatenate(
                            (current_features, motion[current["frame_path"]], future_features)
                        )
                    )
                    labels.append(float(truth) <= horizon)
                    groups.append(sequence)
                    baselines["hold"].append(modes["hold_probability"])
                    baselines["cv"].append(modes["cv_probability"])
                    baselines["ca"].append(modes["ca_probability"])
                    baselines["imm"].append(modes["imm_probability"])
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.float64),
        np.asarray(groups),
        {name: np.asarray(values) for name, values in baselines.items()},
    )


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    output = binary_metrics(labels, probabilities, 0.50)
    output["log_loss"] = _log_loss(probabilities.tolist(), labels.astype(bool).tolist())
    output["expected_calibration_error"] = expected_calibration_error(
        probabilities.tolist(), labels.astype(bool).tolist()
    )
    return output


def evaluate(reports: list[dict[str, Any]], raft_weights: Path) -> dict[str, Any]:
    unique_frames = {}
    for report in reports:
        for frame in report["frames"]:
            unique_frames.setdefault(frame["frame_path"], frame)
    motion = extract_motion(list(unique_frames.values()), raft_weights)
    x, y, groups, baseline_scores = build_future_rows(reports, motion)
    probabilities = np.zeros(len(y), dtype=np.float64)
    folds = {}
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        fitted = fit_logistic(x[train], y[train], l2=0.01, positive_weight=1.25)
        probabilities[test] = predict_logistic(x[test], fitted)
        folds[group] = {"train_rows": int(np.sum(train)), "test_rows": int(np.sum(test))}

    arms = {name: probability_metrics(y, values) for name, values in baseline_scores.items()}
    arms["learned_future_field"] = probability_metrics(y, probabilities)
    candidate = arms["learned_future_field"]
    best_brier = min(arms[name]["brier_score"] for name in baseline_scores)
    best_log_loss = min(arms[name]["log_loss"] for name in baseline_scores)
    best_mcc = max(arms[name]["mcc"] for name in baseline_scores)
    effects = {
        "brier_reduction_vs_best_geometric": (best_brier - candidate["brier_score"]) / best_brier,
        "log_loss_reduction_vs_best_geometric": (best_log_loss - candidate["log_loss"]) / best_log_loss,
    }
    gates = {
        "known_future_opportunities_at_least_3000": len(y) >= 3000,
        "brier_reduction_at_least_0_15": effects["brier_reduction_vs_best_geometric"] >= 0.15,
        "log_loss_reduction_at_least_0_20": effects["log_loss_reduction_vs_best_geometric"] >= 0.20,
        "ece_at_most_0_10": candidate["expected_calibration_error"] <= 0.10,
        "recall_at_least_0_85_and_fpr_at_most_0_15": candidate["recall"] >= 0.85 and candidate["false_positive_rate"] <= 0.15,
        "mcc_strictly_best_geometric": candidate["mcc"] > best_mcc,
    }
    final_fit = fit_logistic(x, y, l2=0.01, positive_weight=1.25)
    return {
        "schema": SCHEMA,
        "features": list(FEATURE_NAMES),
        "groups": len(set(groups.tolist())),
        "opportunities": len(y),
        "lead_frames": LEAD_FRAMES,
        "lead_seconds": LEAD_SECONDS,
        "folds": folds,
        "arms": arms,
        "effects": effects,
        "gates": gates,
        "final_model": {
            "weights_intercept_then_features": final_fit[0].tolist(),
            "feature_mean": final_fit[1].tolist(),
            "feature_scale": final_fit[2].tolist(),
            "l2": 0.01,
            "positive_weight": 1.25,
        },
        "status": (
            "FUTURE_OCCUPANCY_FIELD_A2_WINDOW_LOSO_PASS"
            if all(gates.values())
            else "FUTURE_OCCUPANCY_FIELD_A2_WINDOW_LOSO_FAIL"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.report]
    result = evaluate(reports, args.raft_weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    main()
