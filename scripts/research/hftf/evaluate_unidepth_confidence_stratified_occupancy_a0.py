#!/usr/bin/env python3
"""Evaluate confidence-conditioned empirical occupancy probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_metric3d_clearance_field_a0 import BANDS, HORIZONS_M
from evaluate_metric3d_probabilistic_occupancy_a0 import (
    _log_loss,
    expected_calibration_error,
    load_report,
)


SCHEMA = "blindassist_hftf_unidepth_confidence_stratified_occupancy_a0"


def confidence_bin(value: float, internal_edges: list[float]) -> int:
    return int(np.searchsorted(np.asarray(internal_edges), value, side="right"))


def calibrate(report: dict[str, Any]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    entries: dict[str, list[tuple[float, float]]] = {band: [] for band in BANDS}
    for frame in report["frames"]:
        if frame["sensor"]["status"] != "VALID" or frame["metric3d"]["status"] != "VALID":
            continue
        for band in BANDS:
            truth = frame["sensor"]["bands"][band]["clearance_m"]
            predicted = frame["metric3d"]["bands"][band]["clearance_m"]
            confidence = frame["metric3d"]["bands"][band].get("clearance_log1p_confidence")
            if truth is not None and predicted is not None and confidence is not None:
                entries[band].append((float(confidence), float(truth) - float(predicted)))
    edges = {}
    residuals = {}
    for band, values in entries.items():
        if len(values) < 80:
            raise ValueError(f"{band} needs at least 80 confidence calibration rows")
        confidence_values = np.asarray([value[0] for value in values])
        edges[band] = np.quantile(confidence_values, [0.25, 0.50, 0.75]).tolist()
        for index in range(4):
            residuals[f"{band}:{index}"] = []
        for confidence, residual in values:
            residuals[f"{band}:{confidence_bin(confidence, edges[band])}"].append(residual)
    if any(len(values) < 20 for values in residuals.values()):
        raise ValueError("every confidence stratum needs at least 20 residuals")
    return residuals, edges


def empirical_probability(clearance: float, horizon: float, residuals: list[float]) -> float:
    count = sum(residual <= horizon - clearance for residual in residuals)
    return (count + 0.5) / (len(residuals) + 1.0)


def evaluate(calibration: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    residuals, edges = calibrate(calibration)
    probabilities = []
    baseline = []
    labels = []
    for frame in evaluation["frames"]:
        if frame["sensor"]["status"] != "VALID" or frame["metric3d"]["status"] != "VALID":
            continue
        for band in BANDS:
            truth = frame["sensor"]["bands"][band]["clearance_m"]
            predicted = frame["metric3d"]["bands"][band]["clearance_m"]
            confidence = frame["metric3d"]["bands"][band].get("clearance_log1p_confidence")
            if truth is None or predicted is None or confidence is None:
                continue
            stratum = residuals[f"{band}:{confidence_bin(float(confidence), edges[band])}"]
            for horizon in HORIZONS_M:
                labels.append(float(truth) <= horizon)
                probabilities.append(empirical_probability(float(predicted), horizon, stratum))
                baseline.append(0.999 if float(predicted) <= horizon else 0.001)
    p = np.asarray(probabilities, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if not len(y):
        raise ValueError("no confidence-known evaluation opportunities")
    brier = float(np.mean((p - y) ** 2))
    baseline_brier = float(np.mean((b - y) ** 2))
    log_loss = _log_loss(probabilities, labels)
    baseline_log_loss = _log_loss(baseline, labels)
    clear = p <= 0.05
    positive = y == 1
    predicted_positive = p >= 0.50
    false_clear = float(np.mean(y[clear])) if int(np.sum(clear)) else None
    recall = float(np.mean(predicted_positive[positive])) if int(np.sum(positive)) else None
    effects = {
        "brier_reduction_vs_deterministic": (baseline_brier - brier) / baseline_brier,
        "log_loss_reduction_vs_deterministic": (baseline_log_loss - log_loss) / baseline_log_loss,
    }
    ece = expected_calibration_error(probabilities, labels)
    gates = {
        "brier_reduction_at_least_0_15": effects["brier_reduction_vs_deterministic"] >= 0.15,
        "log_loss_reduction_at_least_0_20": effects["log_loss_reduction_vs_deterministic"] >= 0.20,
        "ece_at_most_0_10": ece <= 0.10,
        "high_clear_false_clear_at_most_0_05": false_clear is not None and false_clear <= 0.05,
        "high_clear_coverage_at_least_0_10": float(np.mean(clear)) >= 0.10,
        "occupied_recall_at_least_0_85": recall is not None and recall >= 0.85,
    }
    return {
        "schema": SCHEMA,
        "calibration_edges": edges,
        "calibration_strata": {key: len(value) for key, value in residuals.items()},
        "evaluation_opportunities": len(labels),
        "brier_score": brier,
        "deterministic_brier_score": baseline_brier,
        "log_loss": log_loss,
        "deterministic_log_loss": baseline_log_loss,
        "expected_calibration_error": ece,
        "high_clear_coverage": float(np.mean(clear)),
        "high_clear_false_clear_rate": false_clear,
        "occupied_recall_at_probability_0_50": recall,
        "effects": effects,
        "gates": gates,
        "status": "UNIDEPTH_CONFIDENCE_STRATIFIED_OCCUPANCY_A0_DEVELOPMENT_PASS" if all(gates.values()) else "UNIDEPTH_CONFIDENCE_STRATIFIED_OCCUPANCY_A0_DEVELOPMENT_FAIL",
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
