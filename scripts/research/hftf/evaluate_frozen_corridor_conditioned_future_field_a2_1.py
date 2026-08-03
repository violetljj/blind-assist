#!/usr/bin/env python3
"""Evaluate the frozen A2.1 future field on one fresh report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_corridor_conditioned_future_field_a2_1 import (
    FEATURE_NAMES,
    build_corridor_rows,
    infer_corridors,
)
from evaluate_future_occupancy_field_a2 import build_future_rows, probability_metrics
from evaluate_motion_conditioned_occupancy_a0 import EXPECTED_RAFT_SHA256, extract_motion
from produce_external_rgb_metric_depth_observations import UniDepthSource


SCHEMA = "blindassist_hftf_frozen_corridor_conditioned_future_field_a2_1"


def evaluate(
    report: dict[str, Any],
    model: dict[str, Any],
    raft_weights: Path,
    corridors: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    if model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("frozen feature order mismatch")
    if model.get("raft_sha256") != EXPECTED_RAFT_SHA256:
        raise ValueError("frozen RAFT identity mismatch")
    motion = extract_motion(report["frames"], raft_weights)
    x, y, groups, baseline_scores = build_future_rows([report], motion)
    extra, corridor_scores = build_corridor_rows([report], corridors)
    if len(extra) != len(x):
        raise RuntimeError("corridor/A2 row mismatch")
    x = np.column_stack((x, extra))
    baseline_scores["corridor_2d_hold"] = corridor_scores

    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights_intercept_then_features"], dtype=np.float64)
    if x.shape[1] != len(mean) or len(scale) != len(mean) or len(weights) != len(mean) + 1:
        raise ValueError("frozen model dimension mismatch")
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -40, 40)))

    arms = {name: probability_metrics(y, values) for name, values in baseline_scores.items()}
    arms["corridor_conditioned_future_field"] = probability_metrics(y, probabilities)
    candidate = arms["corridor_conditioned_future_field"]
    best_brier = min(arms[name]["brier_score"] for name in baseline_scores)
    best_log_loss = min(arms[name]["log_loss"] for name in baseline_scores)
    best_mcc = max(arms[name]["mcc"] for name in baseline_scores)
    effects = {
        "brier_reduction_vs_best_fixed": (best_brier - candidate["brier_score"]) / best_brier,
        "log_loss_reduction_vs_best_fixed": (best_log_loss - candidate["log_loss"]) / best_log_loss,
    }
    gates = {
        "known_future_opportunities_at_least_1200": len(y) >= 1200,
        "brier_reduction_at_least_0_15": effects["brier_reduction_vs_best_fixed"] >= 0.15,
        "log_loss_reduction_at_least_0_20": effects["log_loss_reduction_vs_best_fixed"] >= 0.20,
        "ece_at_most_0_10": candidate["expected_calibration_error"] <= 0.10,
        "recall_at_least_0_85_and_fpr_at_most_0_15": candidate["recall"] >= 0.85 and candidate["false_positive_rate"] <= 0.15,
        "mcc_strictly_best_fixed": candidate["mcc"] > best_mcc,
    }
    return {
        "schema": SCHEMA,
        "groups": len(set(groups.tolist())),
        "opportunities": len(y),
        "arms": arms,
        "effects": effects,
        "gates": gates,
        "status": (
            "CORRIDOR_CONDITIONED_FUTURE_FIELD_A2_1_FRESH_SUPPORTED_DEVELOPMENT_ONLY"
            if all(gates.values())
            else "CORRIDOR_CONDITIONED_FUTURE_FIELD_A2_1_FRESH_NOT_SUPPORTED"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--unidepth-repo", type=Path, required=True)
    parser.add_argument("--unidepth-model-name", default="lpiccinelli/unidepth-v2-vits14")
    parser.add_argument("--unidepth-resolution-level", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--intrinsics-fx-fy-cx-cy", nargs=4, type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    model = json.loads(args.model.read_text(encoding="utf-8"))
    source = UniDepthSource(
        args.unidepth_repo,
        args.unidepth_model_name,
        args.unidepth_resolution_level,
        args.device,
    )
    corridors = infer_corridors([report], source, *args.intrinsics_fx_fy_cx_cy)
    result = evaluate(report, model, args.raft_weights, corridors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
