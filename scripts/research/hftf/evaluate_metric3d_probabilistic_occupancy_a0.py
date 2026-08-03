#!/usr/bin/env python3
"""Calibrate and evaluate band-wise probabilistic collision occupancy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_metric3d_clearance_field_a0 import BANDS, HORIZONS_M


SCHEMA = "blindassist_hftf_metric3d_probabilistic_occupancy_a0"


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report.get("frames"), list) or not report["frames"]:
        raise ValueError(f"{path}: report has no frames")
    return report


def calibration_residuals(report: dict[str, Any]) -> dict[str, list[float]]:
    residuals = {band: [] for band in BANDS}
    for frame in report["frames"]:
        if frame["sensor"]["status"] != "VALID" or frame["metric3d"]["status"] != "VALID":
            continue
        for band in BANDS:
            truth = frame["sensor"]["bands"][band]["clearance_m"]
            prediction = frame["metric3d"]["bands"][band]["clearance_m"]
            if truth is not None and prediction is not None:
                residuals[band].append(float(truth) - float(prediction))
    if any(len(values) < 20 for values in residuals.values()):
        raise ValueError("each band requires at least 20 calibration residuals")
    return residuals


def empirical_occupied_probability(
    predicted_clearance: float,
    horizon: float,
    residuals: list[float],
) -> float:
    threshold = horizon - predicted_clearance
    count = sum(residual <= threshold for residual in residuals)
    return (count + 0.5) / (len(residuals) + 1.0)


def _log_loss(probabilities: list[float], labels: list[bool]) -> float:
    values = []
    for probability, label in zip(probabilities, labels, strict=True):
        clipped = min(0.999, max(0.001, probability))
        values.append(-(math.log(clipped) if label else math.log(1.0 - clipped)))
    return float(np.mean(values))


def expected_calibration_error(
    probabilities: list[float], labels: list[bool], bins: int = 10
) -> float:
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    total = len(probabilities_array)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (probabilities_array >= lower) & (
            probabilities_array <= upper if index == bins - 1 else probabilities_array < upper
        )
        count = int(np.sum(selected))
        if count:
            error += count / total * abs(
                float(np.mean(probabilities_array[selected]))
                - float(np.mean(labels_array[selected]))
            )
    return error


def evaluate(
    calibration: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, Any]:
    residuals = calibration_residuals(calibration)
    probabilities = []
    baseline_probabilities = []
    labels = []
    bands = []
    for frame in evaluation["frames"]:
        if frame["sensor"]["status"] != "VALID" or frame["metric3d"]["status"] != "VALID":
            continue
        for band in BANDS:
            truth_clearance = frame["sensor"]["bands"][band]["clearance_m"]
            predicted_clearance = frame["metric3d"]["bands"][band]["clearance_m"]
            if truth_clearance is None or predicted_clearance is None:
                continue
            for horizon in HORIZONS_M:
                labels.append(float(truth_clearance) <= horizon)
                probabilities.append(
                    empirical_occupied_probability(
                        float(predicted_clearance), horizon, residuals[band]
                    )
                )
                baseline_probabilities.append(0.999 if float(predicted_clearance) <= horizon else 0.001)
                bands.append(band)
    if not labels:
        raise ValueError("evaluation has no known occupancy opportunities")
    labels_array = np.asarray(labels, dtype=np.float64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    baseline_array = np.asarray(baseline_probabilities, dtype=np.float64)
    brier = float(np.mean((probabilities_array - labels_array) ** 2))
    baseline_brier = float(np.mean((baseline_array - labels_array) ** 2))
    log_loss = _log_loss(probabilities, labels)
    baseline_log_loss = _log_loss(baseline_probabilities, labels)
    high_clear = probabilities_array <= 0.05
    predicted_occupied = probabilities_array >= 0.50
    positives = labels_array == 1.0
    high_clear_count = int(np.sum(high_clear))
    false_clear_rate = (
        float(np.mean(labels_array[high_clear])) if high_clear_count else None
    )
    occupied_recall = float(np.mean(predicted_occupied[positives])) if int(np.sum(positives)) else None
    effects = {
        "brier_reduction_vs_deterministic": (baseline_brier - brier) / baseline_brier if baseline_brier > 0 else None,
        "log_loss_reduction_vs_deterministic": (baseline_log_loss - log_loss) / baseline_log_loss if baseline_log_loss > 0 else None,
    }
    gates = {
        "brier_reduction_at_least_0_15": effects["brier_reduction_vs_deterministic"] is not None and effects["brier_reduction_vs_deterministic"] >= 0.15,
        "log_loss_reduction_at_least_0_20": effects["log_loss_reduction_vs_deterministic"] is not None and effects["log_loss_reduction_vs_deterministic"] >= 0.20,
        "ece_at_most_0_10": expected_calibration_error(probabilities, labels) <= 0.10,
        "high_clear_false_clear_at_most_0_05": false_clear_rate is not None and false_clear_rate <= 0.05,
        "high_clear_coverage_at_least_0_10": high_clear_count / len(labels) >= 0.10,
        "occupied_recall_at_least_0_85": occupied_recall is not None and occupied_recall >= 0.85,
    }
    per_band = {}
    for band in BANDS:
        selected = np.asarray([value == band for value in bands])
        per_band[band] = {
            "opportunities": int(np.sum(selected)),
            "brier_score": float(np.mean((probabilities_array[selected] - labels_array[selected]) ** 2)),
            "ece": expected_calibration_error(
                probabilities_array[selected].tolist(), labels_array[selected].astype(bool).tolist()
            ),
        }
    return {
        "schema": SCHEMA,
        "calibration_residuals": {band: len(values) for band, values in residuals.items()},
        "evaluation_opportunities": len(labels),
        "brier_score": brier,
        "deterministic_brier_score": baseline_brier,
        "log_loss": log_loss,
        "deterministic_log_loss": baseline_log_loss,
        "expected_calibration_error": expected_calibration_error(probabilities, labels),
        "high_clear_opportunities": high_clear_count,
        "high_clear_coverage": high_clear_count / len(labels),
        "high_clear_false_clear_rate": false_clear_rate,
        "occupied_recall_at_probability_0_50": occupied_recall,
        "effects": effects,
        "per_band": per_band,
        "gates": gates,
        "status": "METRIC3D_PROBABILISTIC_OCCUPANCY_A0_DEVELOPMENT_PASS" if all(gates.values()) else "METRIC3D_PROBABILISTIC_OCCUPANCY_A0_DEVELOPMENT_FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(load_report(args.calibration_report), load_report(args.evaluation_report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
