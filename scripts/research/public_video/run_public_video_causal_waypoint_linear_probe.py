#!/usr/bin/env python3
"""Run a source-heldout deterministic ridge probe for future ego waypoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_causal_waypoint_linear_probe_v1"


def pooled_features(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 4 or x.shape[2:] != (16, 16):
        raise ValueError("expected NxCx16x16 causal feature tensor")
    pooled = x.reshape(len(x), x.shape[1], 4, 4, 4, 4).mean(axis=(3, 5))
    global_std = x.std(axis=(2, 3))
    return np.concatenate([pooled.reshape(len(x), -1), global_std], axis=1)


def waypoint_targets(fields: np.ndarray) -> np.ndarray:
    y = np.asarray(fields)
    if y.ndim != 4 or y.shape[1:] != (3, 16, 16):
        raise ValueError("expected Nx3x16x16 route target tensor")
    result = np.empty((len(y), 6), dtype=np.float64)
    for row in range(len(y)):
        for horizon in range(3):
            yy, xx = np.unravel_index(int(np.argmax(y[row, horizon])), (16, 16))
            result[row, horizon * 2:horizon * 2 + 2] = ((xx + 0.5) / 16.0, (yy + 0.5) / 16.0)
    return result


def source_balanced_weights(sources: np.ndarray) -> np.ndarray:
    source_values = np.asarray(sources).astype(str)
    unique = sorted(set(source_values.tolist()))
    weights = np.zeros(len(source_values), dtype=np.float64)
    for source in unique:
        indices = np.flatnonzero(source_values == source)
        weights[indices] = len(source_values) / (len(unique) * len(indices))
    if not np.isclose(weights.sum(), len(source_values)):
        raise ValueError("source-balanced weights have wrong total mass")
    return weights


def fit_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), normalized])
    weighted = design * weights[:, None]
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ weighted + penalty, design.T @ (weights[:, None] * y))
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    normalized = (x - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(len(x)), normalized])
    return np.clip(design @ model["coefficients"], 0.0, 1.0)


def event_predictions(
    coordinates: np.ndarray, events: np.ndarray, labels: np.ndarray, obstacles: np.ndarray
) -> list[dict[str, Any]]:
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, set[int]] = defaultdict(set)
    for points, event, label, obstacle in zip(coordinates, events, labels, obstacles):
        hits = []
        for horizon in range(3):
            x, y = points[horizon * 2:horizon * 2 + 2]
            xx = int(np.clip(np.floor(x * 16), 0, 15))
            yy = int(np.clip(np.floor(y * 16), 0, 15))
            hits.append(bool(obstacle[yy, xx]))
        grouped_scores[str(event)].append(sum(hits) / 3.0)
        grouped_labels[str(event)].add(int(label))
    rows = []
    for event in sorted(grouped_scores):
        if len(grouped_labels[event]) != 1:
            raise ValueError(f"event has inconsistent labels: {event}")
        rows.append({"event_id": event, "label": next(iter(grouped_labels[event])),
                     "frame_count": len(grouped_scores[event]),
                     "predicted_horizon_hit_fraction": float(np.mean(grouped_scores[event]))})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.training_contract, args.cache_report, args.cache, args.head_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    for path, key in ((args.training_contract, "r764_training_contract_sha256"),
                      (args.cache_report, "r764_feature_cache_report_sha256"),
                      (args.cache, "r764_feature_cache_sha256"),
                      (args.head_report, "r764_head_report_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    cache_report = lifecycle.verify_json_sidecar(args.cache_report)
    if cache_report["cache"]["sha256"] != common.sha256_file(args.cache):
        raise ValueError("cache report does not bind cache")
    cache = np.load(args.cache)
    train_x = pooled_features(cache["train_x"])
    train_y = waypoint_targets(cache["train_y"])
    train_sources = cache["train_sources"].astype(str)
    eval_x = pooled_features(cache["eval_x"])
    eval_sources = cache["eval_sources"].astype(str)
    sources = sorted(set(train_sources.tolist()))
    oof = np.zeros_like(train_y)
    eval_pred = np.zeros((len(eval_x), 6), dtype=np.float64)
    eval_seen = np.zeros(len(eval_x), dtype=bool)
    folds = []
    alpha = float(contract["probe"]["alpha"])
    for source in sources:
        fit_indices = np.flatnonzero(train_sources != source)
        test_indices = np.flatnonzero(train_sources == source)
        weights = source_balanced_weights(train_sources[fit_indices])
        model = fit_ridge(train_x[fit_indices], train_y[fit_indices], weights, alpha)
        oof[test_indices] = predict(model, train_x[test_indices])
        source_eval = np.flatnonzero(eval_sources == source)
        if len(source_eval):
            eval_pred[source_eval] = predict(model, eval_x[source_eval])
            eval_seen[source_eval] = True
        fold_error = np.linalg.norm(
            oof[test_indices].reshape(-1, 3, 2) - train_y[test_indices].reshape(-1, 3, 2), axis=2
        ).mean()
        folds.append({"held_out_source_id": source, "train_count": len(fit_indices),
                      "test_count": len(test_indices), "eval_count": len(source_eval),
                      "mean_waypoint_localization_error_norm": float(fold_error),
                      "finite": bool(np.isfinite(fold_error))})
    if not eval_seen.all():
        raise ValueError(f"evaluation source missing held-out model: {sorted(set(eval_sources[~eval_seen]))}")
    errors = np.linalg.norm(oof.reshape(-1, 3, 2) - train_y.reshape(-1, 3, 2), axis=2)
    mean_error = float(errors.mean())
    events = event_predictions(eval_pred, cache["eval_events"], cache["eval_labels"], cache["eval_obstacles"])
    positive = [row["predicted_horizon_hit_fraction"] for row in events if row["label"] == 1]
    negative = [row["predicted_horizon_hit_fraction"] for row in events if row["label"] == 0]
    gate = contract["gate"]
    improvement = float(gate["r764_dense_head_localization_error_norm"]) - mean_error
    checks = {
        "mean_waypoint_localization_error": mean_error <= float(gate["mean_waypoint_localization_error_norm_at_most"]),
        "improvement_over_r764_dense_head": improvement >= float(gate["improvement_over_r764_dense_head_at_least"]),
        "strict_event_label_separation": min(positive) > max(negative),
        "all_folds_finite": all(row["finite"] for row in folds),
    }
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "cache_sha256": common.sha256_file(args.cache)},
        "source_count": len(sources), "train_frame_count": len(train_x),
        "eval_frame_count": len(eval_x), "input_feature_dimension": train_x.shape[1],
        "output_dimension": train_y.shape[1], "optimizer_steps": 0, "weights_saved": False,
        "folds": folds,
        "mean_waypoint_localization_error_norm": mean_error,
        "median_waypoint_localization_error_norm": float(np.median(errors)),
        "improvement_over_r764_dense_head": improvement,
        "event_predictions": events,
        "checks": checks,
        "diagnostic_gate_passed": all(checks.values()),
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--head-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({"ok": True, "mean_error": value["mean_waypoint_localization_error_norm"],
                      "passed": value["diagnostic_gate_passed"],
                      "output_sha256": common.sha256_file(parsed.output)}))
