#!/usr/bin/env python3
"""Run five r7.72 source-bootstrap same-source pair-ranking short heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_marker_relation_bootstrap_short_runs as bootstrap
import run_public_video_marker_relation_pair_ranking_probe as pair_probe


SCHEMA = "blindassist_public_video_marker_relation_pair_bootstrap_short_runs_v1"


def bootstrap_pair_rows(pair_sources: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sources = np.asarray(pair_sources).astype(str)
    unique = sorted(set(sources.tolist()))
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(unique, size=len(unique), replace=True).tolist()
    indices: list[int] = []
    weights: list[float] = []
    for source in draws:
        rows = np.flatnonzero(sources == source).tolist()
        indices.extend(rows)
        weights.extend([1.0 / len(draws) / len(rows)] * len(rows))
    return np.asarray(indices, dtype=np.int64), np.asarray(weights), draws


def fit_pair_head(
    deltas: np.ndarray, pair_sources: np.ndarray, seed: int, optimizer: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    sampled, weights, draws = bootstrap_pair_rows(pair_sources, seed)
    values = np.asarray(deltas, dtype=np.float64)
    variance = np.average(values[sampled] ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    normalized = values[sampled] / scale
    initial = np.average(normalized, axis=0, weights=weights)
    norm = float(np.linalg.norm(initial))
    if norm <= 1e-12:
        raise ValueError("bootstrap pair prototype is degenerate")
    initial /= norm
    weight = initial.copy()
    m = np.zeros_like(weight)
    v = np.zeros_like(weight)
    losses: list[float] = []
    learning_rate = float(optimizer["learning_rate"])
    weight_decay = float(optimizer["weight_decay"])
    margin = float(optimizer["pair_margin"])
    for step in range(1, int(optimizer["steps"]) + 1):
        projection = normalized @ weight
        softplus = np.logaddexp(0.0, margin - projection)
        loss = float(weights @ softplus + 0.5 * weight_decay * (weight @ weight))
        sigmoid_negative = 1.0 / (1.0 + np.exp(np.clip(projection - margin, -40.0, 40.0)))
        gradient = -(normalized.T @ (weights * sigmoid_negative)) + weight_decay * weight
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient * gradient
        weight -= learning_rate * (m / (1.0 - 0.9**step)) / (np.sqrt(v / (1.0 - 0.999**step)) + 1e-8)
        if step in {1, int(optimizer["steps"])}:
            losses.append(loss)
    prototype = {"scale": scale, "weight": initial}
    optimized = {"scale": scale, "weight": weight}
    audit = {"sampled_source_ids": draws, "sampled_unique_source_count": len(set(draws)),
             "loss_first_last": losses,
             "coefficient_sha256": hashlib.sha256(np.asarray(weight, dtype="<f8").tobytes()).hexdigest()}
    return optimized, prototype, audit


def projection(model: dict[str, np.ndarray], values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) / model["scale"]) @ model["weight"]


def evaluate_seed(
    x: np.ndarray, active: np.ndarray, sources: np.ndarray, deltas: np.ndarray,
    pair_sources: np.ndarray, seed: int, optimizer: dict[str, Any]
) -> dict[str, Any]:
    folds = []
    for fold_index, held_source in enumerate(sorted(set(pair_sources.tolist()))):
        train = pair_sources != held_source
        test = ~train
        model, prototype, audit = fit_pair_head(
            deltas[train], pair_sources[train], seed + 1009 * fold_index, optimizer
        )
        selected_frames = sources == held_source
        optimized_frame_scores = projection(model, x[selected_frames])
        prototype_frame_scores = projection(prototype, x[selected_frames])
        frame_labels = active[selected_frames]
        optimized_pair = projection(model, deltas[test])
        prototype_pair = projection(prototype, deltas[test])
        folds.append({
            "held_out_source_id": held_source,
            "held_out_pair_count": int(test.sum()),
            "prototype_pair_ordering_rate": float((prototype_pair > 0.0).mean()),
            "optimized_pair_ordering_rate": float((optimized_pair > 0.0).mean()),
            "prototype_frame_auroc": pair_probe.linear.roc_auc(frame_labels.astype(np.int64), prototype_frame_scores),
            "optimized_frame_auroc": pair_probe.linear.roc_auc(frame_labels.astype(np.int64), optimized_frame_scores),
            **audit,
        })
    optimized_aurocs = [row["optimized_frame_auroc"] for row in folds]
    prototype_aurocs = [row["prototype_frame_auroc"] for row in folds]
    return {
        "seed": seed,
        "folds": folds,
        "prototype": {
            "source_auroc_median": float(np.median(prototype_aurocs)),
            "source_auroc_minimum": float(min(prototype_aurocs)),
            "source_macro_pair_ordering": float(np.mean([row["prototype_pair_ordering_rate"] for row in folds])),
        },
        "optimized": {
            "source_auroc_median": float(np.median(optimized_aurocs)),
            "source_auroc_minimum": float(min(optimized_aurocs)),
            "source_macro_pair_ordering": float(np.mean([row["optimized_pair_ordering_rate"] for row in folds])),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.linear_contract, args.pair_contract, args.pair_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    for path, key in ((args.linear_contract, "r767a_contract_sha256"),
                      (args.pair_contract, "r771_contract_sha256"),
                      (args.pair_report, "r771_report_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    if lifecycle.verify_json_sidecar(args.pair_report).get("diagnostic_gate_passed") is not True:
        raise ValueError("r7.71 pair gate did not pass")
    linear_contract = common.load_json(args.linear_contract)
    x, y, sources, timestamps = pair_probe.load_data(linear_contract)
    active = y > 0.0
    pairs = pair_probe.nearest_time_pairs(active, sources, timestamps)
    deltas = np.stack([x[row["positive_index"]] - x[row["negative_index"]] for row in pairs])
    pair_sources = np.asarray([row["source_id"] for row in pairs])
    runs = [evaluate_seed(x, active, sources, deltas, pair_sources, int(seed), contract["optimizer"])
            for seed in contract["seeds"]]
    gate = contract["stability_gate"]
    for row in runs:
        metrics = row["optimized"]
        row["checks"] = {
            "source_auroc_median": metrics["source_auroc_median"] >= float(gate["passing_run_source_auroc_median_at_least"]),
            "source_auroc_minimum": metrics["source_auroc_minimum"] >= float(gate["passing_run_source_auroc_minimum_at_least"]),
            "source_macro_pair_ordering": metrics["source_macro_pair_ordering"] >= float(gate["passing_run_source_macro_pair_ordering_at_least"]),
            "bootstrap_unique_source_floor": all(row_fold["sampled_unique_source_count"] >= int(gate["each_fold_unique_training_sources_at_least"]) for row_fold in row["folds"]),
        }
        row["run_gate_passed"] = all(row["checks"].values())
    passing = sum(row["run_gate_passed"] for row in runs)
    optimized_median = float(np.median([row["optimized"]["source_auroc_median"] for row in runs]))
    prototype_median = float(np.median([row["prototype"]["source_auroc_median"] for row in runs]))
    no_regression = optimized_median >= prototype_median - float(gate["optimized_median_max_drop_from_prototype"])
    passed = passing >= int(gate["runs_passing_at_least"]) and no_regression
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "r771_report_sha256": common.sha256_file(args.pair_report)},
        "data": {"pair_count": len(pairs), "mixed_source_count": len(set(pair_sources.tolist()))},
        "head_contract": contract["head"], "optimizer": contract["optimizer"], "runs": runs,
        "summary": {"runs_passing": passing,
                    "optimized_source_auroc_medians": [row["optimized"]["source_auroc_median"] for row in runs],
                    "optimized_source_auroc_minima": [row["optimized"]["source_auroc_minimum"] for row in runs],
                    "optimized_pair_ordering": [row["optimized"]["source_macro_pair_ordering"] for row in runs],
                    "prototype_source_auroc_median": prototype_median,
                    "optimized_source_auroc_median": optimized_median,
                    "optimized_not_materially_worse_than_prototype": no_regression},
        "bootstrap_stability_gate": {"passed": passed, "requirements": gate},
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--linear-contract", type=Path, required=True)
    parser.add_argument("--pair-contract", type=Path, required=True)
    parser.add_argument("--pair-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(json.dumps({"ok": True, **result["summary"], "stable": result["bootstrap_stability_gate"]["passed"],
                      "output_sha256": common.sha256_file(parsed.output)}))
