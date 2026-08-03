#!/usr/bin/env python3
"""Evaluate A2.1 with the independently supported 2D corridor evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_collision_risk_field_a1 import two_d_clearances
from evaluate_future_occupancy_field_a2 import (
    FEATURE_NAMES as A2_FEATURE_NAMES,
    HISTORY_FRAMES,
    LEAD_FRAMES,
    build_future_rows,
    probability_metrics,
    sigmoid_margin,
)
from evaluate_metric3d_clearance_field_a0 import BANDS, HORIZONS_M
from evaluate_motion_conditioned_occupancy_a0 import (
    extract_motion,
    fit_logistic,
    predict_logistic,
)
from produce_external_rgb_metric_depth_observations import UniDepthSource


SCHEMA = "blindassist_hftf_corridor_conditioned_future_field_a2_1"
EXTRA_FEATURE_NAMES = (
    "corridor_2d_clearance_m",
    "corridor_2d_margin_m",
    "corridor_2d_hold_probability",
)
FEATURE_NAMES = tuple(A2_FEATURE_NAMES) + EXTRA_FEATURE_NAMES


def corridor_extra_features(clearance: float, horizon: float) -> np.ndarray:
    return np.asarray(
        [
            clearance,
            clearance - horizon,
            sigmoid_margin(horizon, clearance, 0.20),
        ],
        dtype=np.float64,
    )


def infer_corridors(
    reports: list[dict[str, Any]], source: UniDepthSource, fx: float, fy: float, cx: float, cy: float
) -> dict[str, dict[str, float | None]]:
    frames = {}
    for report in reports:
        for frame in report["frames"]:
            frames.setdefault(frame["frame_path"], frame)
    output = {}
    row = {"intrinsics_fx_fy_cx_cy": [fx, fy, cx, cy]}
    for path in sorted(frames):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth, _ = source.infer(rgb, row)
        output[path] = two_d_clearances(depth, fx, cx)
    return output


def build_corridor_rows(
    reports: list[dict[str, Any]], corridors: dict[str, dict[str, float | None]]
) -> tuple[np.ndarray, np.ndarray]:
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        for frame in report["frames"]:
            by_sequence[str(frame["sequence_id"])].append(frame)
    features = []
    probabilities = []
    for _, frames in sorted(by_sequence.items()):
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
                corridor = corridors[current["frame_path"]][band]
                if predicted is None or confidence is None or truth is None or corridor is None:
                    continue
                # Mirror A2's requirement that at least the current history value exists.
                history_values = [
                    prior["metric3d"]["bands"][band]["clearance_m"]
                    for prior in frames[max(0, index - HISTORY_FRAMES + 1) : index + 1]
                    if prior["metric3d"]["status"] == "VALID"
                    and prior["metric3d"]["bands"][band]["clearance_m"] is not None
                ]
                if not history_values:
                    continue
                for horizon in HORIZONS_M:
                    extra = corridor_extra_features(float(corridor), horizon)
                    features.append(extra)
                    probabilities.append(extra[2])
    return np.stack(features), np.asarray(probabilities)


def evaluate(
    reports: list[dict[str, Any]],
    raft_weights: Path,
    corridors: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    unique_frames = {}
    for report in reports:
        for frame in report["frames"]:
            unique_frames.setdefault(frame["frame_path"], frame)
    motion = extract_motion(list(unique_frames.values()), raft_weights)
    x, y, groups, baseline_scores = build_future_rows(reports, motion)
    extra, corridor_scores = build_corridor_rows(reports, corridors)
    if len(extra) != len(x):
        raise RuntimeError(f"corridor/A2 row mismatch: {len(extra)} != {len(x)}")
    x = np.column_stack((x, extra))
    baseline_scores["corridor_2d_hold"] = corridor_scores

    probabilities = np.zeros(len(y), dtype=np.float64)
    folds = {}
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        fitted = fit_logistic(x[train], y[train], l2=0.01, positive_weight=1.25)
        probabilities[test] = predict_logistic(x[test], fitted)
        folds[group] = {"train_rows": int(np.sum(train)), "test_rows": int(np.sum(test))}

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
        "known_future_opportunities_at_least_3000": len(y) >= 3000,
        "brier_reduction_at_least_0_15": effects["brier_reduction_vs_best_fixed"] >= 0.15,
        "log_loss_reduction_at_least_0_20": effects["log_loss_reduction_vs_best_fixed"] >= 0.20,
        "ece_at_most_0_10": candidate["expected_calibration_error"] <= 0.10,
        "recall_at_least_0_85_and_fpr_at_most_0_15": candidate["recall"] >= 0.85 and candidate["false_positive_rate"] <= 0.15,
        "mcc_strictly_best_fixed": candidate["mcc"] > best_mcc,
    }
    final_fit = fit_logistic(x, y, l2=0.01, positive_weight=1.25)
    return {
        "schema": SCHEMA,
        "features": list(FEATURE_NAMES),
        "groups": len(set(groups.tolist())),
        "opportunities": len(y),
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
            "CORRIDOR_CONDITIONED_FUTURE_FIELD_A2_1_WINDOW_LOSO_PASS"
            if all(gates.values())
            else "CORRIDOR_CONDITIONED_FUTURE_FIELD_A2_1_WINDOW_LOSO_FAIL"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--unidepth-repo", type=Path, required=True)
    parser.add_argument("--unidepth-model-name", default="lpiccinelli/unidepth-v2-vits14")
    parser.add_argument("--unidepth-resolution-level", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--intrinsics-fx-fy-cx-cy", nargs=4, type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.report]
    source = UniDepthSource(
        args.unidepth_repo,
        args.unidepth_model_name,
        args.unidepth_resolution_level,
        args.device,
    )
    corridors = infer_corridors(reports, source, *args.intrinsics_fx_fy_cx_cy)
    result = evaluate(reports, args.raft_weights, corridors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    main()
