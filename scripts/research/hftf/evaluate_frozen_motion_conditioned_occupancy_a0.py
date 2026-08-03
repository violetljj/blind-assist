#!/usr/bin/env python3
"""Evaluate the frozen motion-conditioned occupancy model on one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_metric3d_probabilistic_occupancy_a0 import (
    _log_loss,
    expected_calibration_error,
    load_report,
)
from evaluate_motion_conditioned_occupancy_a0 import (
    EXPECTED_RAFT_SHA256,
    FEATURE_NAMES,
    build_rows,
    extract_motion,
)


SCHEMA = "blindassist_hftf_frozen_motion_conditioned_occupancy_a0"


def evaluate(report: dict, model: dict, raft_weights: Path) -> dict:
    if model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("frozen feature order mismatch")
    if model.get("raft_sha256") != EXPECTED_RAFT_SHA256:
        raise ValueError("frozen RAFT identity mismatch")
    frames = report["frames"]
    motion = extract_motion(frames, raft_weights)
    x, y, _ = build_rows([report], motion)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights_intercept_then_features"], dtype=np.float64)
    if x.shape[1] != len(mean) or len(scale) != len(mean) or len(weights) != len(mean) + 1:
        raise ValueError("frozen model dimension mismatch")
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -40, 40)))
    baseline = np.where(x[:, 0] <= 0, 0.999, 0.001)
    brier = float(np.mean((probabilities - y) ** 2))
    baseline_brier = float(np.mean((baseline - y) ** 2))
    log_loss = _log_loss(probabilities.tolist(), y.astype(bool).tolist())
    baseline_log_loss = _log_loss(baseline.tolist(), y.astype(bool).tolist())
    high_clear = probabilities <= 0.05
    positive = y == 1
    false_clear = float(np.mean(y[high_clear])) if int(np.sum(high_clear)) else None
    recall = float(np.mean(probabilities[positive] >= 0.50)) if int(np.sum(positive)) else None
    effects = {
        "brier_reduction_vs_deterministic": (baseline_brier - brier) / baseline_brier,
        "log_loss_reduction_vs_deterministic": (baseline_log_loss - log_loss) / baseline_log_loss,
    }
    ece = expected_calibration_error(probabilities.tolist(), y.astype(bool).tolist())
    gates = {
        "brier_reduction_at_least_0_15": effects["brier_reduction_vs_deterministic"] >= 0.15,
        "log_loss_reduction_at_least_0_20": effects["log_loss_reduction_vs_deterministic"] >= 0.20,
        "ece_at_most_0_10": ece <= 0.10,
        "high_clear_false_clear_at_most_0_05": false_clear is not None and false_clear <= 0.05,
        "high_clear_coverage_at_least_0_10": float(np.mean(high_clear)) >= 0.10,
        "occupied_recall_at_least_0_85": recall is not None and recall >= 0.85,
    }
    return {
        "schema": SCHEMA,
        "opportunities": len(y),
        "brier_score": brier,
        "deterministic_brier_score": baseline_brier,
        "log_loss": log_loss,
        "deterministic_log_loss": baseline_log_loss,
        "expected_calibration_error": ece,
        "high_clear_coverage": float(np.mean(high_clear)),
        "high_clear_false_clear_rate": false_clear,
        "occupied_recall_at_probability_0_50": recall,
        "effects": effects,
        "gates": gates,
        "status": "MOTION_CONDITIONED_OCCUPANCY_A0_1_FRESH_SUPPORTED_DEVELOPMENT_ONLY" if all(gates.values()) else "MOTION_CONDITIONED_OCCUPANCY_A0_1_FRESH_NOT_SUPPORTED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        load_report(args.report),
        json.loads(args.model.read_text(encoding="utf-8")),
        args.raft_weights,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
