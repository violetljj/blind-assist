#!/usr/bin/env python3
"""Consumed-only fixed geometry versus geometry-plus-motion A1 ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_collision_risk_field_a1 import binary_metrics
from evaluate_metric3d_probabilistic_occupancy_a0 import (
    _log_loss,
    expected_calibration_error,
)
from evaluate_motion_conditioned_occupancy_a0 import (
    FEATURE_NAMES,
    build_rows,
    extract_motion,
    fit_logistic,
    predict_logistic,
    sha256,
)


SCHEMA = "blindassist_hftf_collision_risk_field_a1_consumed_incremental_ablation_r0"
ARMS = {
    "geometry_only": tuple(range(8)),
    "geometry_plus_motion": tuple(range(len(FEATURE_NAMES))),
}


def _arm_predictions(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, indices: tuple[int, ...]
) -> np.ndarray:
    selected = x[:, indices]
    probabilities = np.zeros(len(y), dtype=np.float64)
    for group in sorted(set(groups.tolist())):
        test = groups == group
        fitted = fit_logistic(
            selected[~test], y[~test], l2=0.01, positive_weight=1.25
        )
        probabilities[test] = predict_logistic(selected[test], fitted)
    return probabilities


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    metrics = binary_metrics(labels, probabilities, 0.50)
    metrics["log_loss"] = _log_loss(probabilities.tolist(), labels.astype(bool).tolist())
    metrics["expected_calibration_error"] = expected_calibration_error(
        probabilities.tolist(), labels.astype(bool).tolist()
    )
    return metrics


def evaluate_arrays(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("unexpected feature matrix shape")
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError("row count mismatch")
    if len(set(groups.tolist())) < 2:
        raise ValueError("at least two complete windows are required")

    probabilities = {
        name: _arm_predictions(x, y, groups, indices)
        for name, indices in ARMS.items()
    }
    arms = {name: _metrics(y, values) for name, values in probabilities.items()}
    geometry = arms["geometry_only"]
    full = arms["geometry_plus_motion"]
    per_window = {}
    positive_windows = 0
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        geometry_brier = float(np.mean((probabilities["geometry_only"][mask] - y[mask]) ** 2))
        full_brier = float(np.mean((probabilities["geometry_plus_motion"][mask] - y[mask]) ** 2))
        improved = full_brier < geometry_brier
        positive_windows += int(improved)
        per_window[group] = {
            "rows": int(np.sum(mask)),
            "geometry_only_brier": geometry_brier,
            "geometry_plus_motion_brier": full_brier,
            "motion_brier_delta": full_brier - geometry_brier,
            "motion_improves_brier": improved,
        }

    gates = {
        "pooled_brier_lower": full["brier_score"] < geometry["brier_score"],
        "pooled_log_loss_lower": full["log_loss"] < geometry["log_loss"],
        "pooled_f1_higher": full["f1"] > geometry["f1"],
        "pooled_recall_not_lower": full["recall"] >= geometry["recall"],
        "strict_majority_windows_brier_better": positive_windows > len(per_window) / 2,
    }
    supported = all(gates.values())
    return {
        "schema": SCHEMA,
        "data_role": "CONSUMED_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "opportunities": len(y),
        "windows": len(per_window),
        "feature_names": list(FEATURE_NAMES),
        "arms": arms,
        "motion_increment": {
            "brier_delta": full["brier_score"] - geometry["brier_score"],
            "brier_relative_reduction": (
                geometry["brier_score"] - full["brier_score"]
            ) / geometry["brier_score"],
            "log_loss_delta": full["log_loss"] - geometry["log_loss"],
            "f1_delta": full["f1"] - geometry["f1"],
            "recall_delta": full["recall"] - geometry["recall"],
            "mcc_delta": full["mcc"] - geometry["mcc"],
            "brier_positive_windows": positive_windows,
        },
        "per_window": per_window,
        "gates": gates,
        "status": (
            "A1_CONSUMED_MOTION_INCREMENT_SUPPORTED_DIAGNOSTIC_ONLY"
            if supported
            else "A1_CONSUMED_MOTION_INCREMENT_NOT_SUPPORTED"
        ),
        "preserved_terminal": "COLLISION_RISK_FIELD_A1_DEVELOPMENT_FAIL",
    }


def evaluate(report: dict[str, Any], raft_weights: Path) -> dict[str, Any]:
    frames = report["frames"]
    motion = extract_motion(frames, raft_weights)
    x, y, groups = build_rows([report], motion)
    result = evaluate_arrays(x, y, groups)
    result["input_report_sha256"] = None
    result["raft_weights_sha256"] = sha256(raft_weights).upper()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = evaluate(report, args.raft_weights)
    result["input_report_sha256"] = sha256(args.report).upper()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
