#!/usr/bin/env python3
"""Run the deterministic r7.71 same-source marker-relation pair-ranking probe."""

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
import run_public_video_marker_relation_linear_probe as linear


SCHEMA = "blindassist_public_video_marker_relation_pair_ranking_probe_v1"


def nearest_time_pairs(
    active: np.ndarray, sources: np.ndarray, timestamps: np.ndarray
) -> list[dict[str, Any]]:
    labels = np.asarray(active, dtype=bool)
    source_values = np.asarray(sources).astype(str)
    time_values = np.asarray(timestamps, dtype=np.int64)
    if labels.shape != source_values.shape or labels.shape != time_values.shape:
        raise ValueError("pair arrays differ")
    pairs: list[dict[str, Any]] = []
    for source in sorted(set(source_values.tolist())):
        source_indices = np.flatnonzero(source_values == source)
        positive = source_indices[labels[source_indices]]
        negative = source_indices[~labels[source_indices]]
        if not len(positive) or not len(negative):
            continue
        for positive_index in positive:
            negative_index = min(
                negative.tolist(),
                key=lambda index: (abs(int(time_values[index]) - int(time_values[positive_index])),
                                   int(time_values[index]), int(index)),
            )
            pairs.append({
                "source_id": source,
                "positive_index": int(positive_index),
                "negative_index": int(negative_index),
                "positive_timestamp_ms": int(time_values[positive_index]),
                "negative_timestamp_ms": int(time_values[negative_index]),
                "absolute_time_gap_ms": abs(int(time_values[positive_index]) - int(time_values[negative_index])),
            })
    return pairs


def fit_signed_pair_ridge(deltas: np.ndarray, pair_sources: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    values = np.asarray(deltas, dtype=np.float64)
    sources = np.asarray(pair_sources).astype(str)
    if values.ndim != 2 or len(values) != len(sources) or not len(values):
        raise ValueError("pair ridge requires aligned deltas")
    unique_sources = sorted(set(sources.tolist()))
    pair_weights = np.zeros(len(values), dtype=np.float64)
    for source in unique_sources:
        selected = sources == source
        pair_weights[selected] = 1.0 / len(unique_sources) / int(selected.sum())
    signed_x = np.concatenate([values, -values], axis=0)
    signed_y = np.concatenate([np.ones(len(values)), -np.ones(len(values))])
    weights = np.concatenate([pair_weights * 0.5, pair_weights * 0.5])
    variance = np.average(signed_x * signed_x, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    normalized = signed_x / scale
    coefficients = np.linalg.solve(
        normalized.T @ (normalized * weights[:, None]) + float(alpha) * np.eye(values.shape[1]),
        normalized.T @ (weights * signed_y),
    )
    return {"scale": scale, "coefficients": coefficients}


def pair_projection(model: dict[str, np.ndarray], deltas: np.ndarray) -> np.ndarray:
    return (np.asarray(deltas, dtype=np.float64) / model["scale"]) @ model["coefficients"]


def load_data(contract: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, sources = bootstrap.load_relation_data(contract)
    manifest = linear._load_manifest(linear._resolve(contract["bound_inputs"]["r763_manifest_path"]))
    marker_rows = [row for row in manifest if int(row["marker_detection_count"]) > 0]
    timestamps = np.asarray([int(row["timestamp_ms"]) for row in marker_rows], dtype=np.int64)
    manifest_sources = np.asarray([str(row["source_id"]) for row in marker_rows])
    if len(timestamps) != len(x) or not np.array_equal(manifest_sources, sources):
        raise ValueError("marker manifest metadata differs from relation vectors")
    return x, y, sources, timestamps


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.linear_contract, args.linear_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    for path, key in ((args.linear_contract, "r767a_contract_sha256"),
                      (args.linear_report, "r767a_report_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    if lifecycle.verify_json_sidecar(args.linear_report).get("diagnostic_gate_passed") is not True:
        raise ValueError("r7.67a gate did not pass")
    linear_contract = common.load_json(args.linear_contract)
    x, y, sources, timestamps = load_data(linear_contract)
    active = y > 0.0
    pairs = nearest_time_pairs(active, sources, timestamps)
    deltas = np.stack([x[row["positive_index"]] - x[row["negative_index"]] for row in pairs])
    pair_sources = np.asarray([row["source_id"] for row in pairs])
    alpha = float(contract["probe"]["ridge_alpha"])
    folds = []
    for held_source in sorted(set(pair_sources.tolist())):
        train = pair_sources != held_source
        test = ~train
        model = fit_signed_pair_ridge(deltas[train], pair_sources[train], alpha)
        projection = pair_projection(model, deltas[test])
        frame_selected = sources == held_source
        frame_scores = (x[frame_selected] / model["scale"]) @ model["coefficients"]
        frame_labels = active[frame_selected]
        coefficient_sha = hashlib.sha256(np.asarray(model["coefficients"], dtype="<f8").tobytes()).hexdigest()
        folds.append({
            "held_out_source_id": held_source,
            "held_out_pair_count": int(test.sum()),
            "pair_ordering_rate": float((projection > 0.0).mean()),
            "frame_teacher_active_auroc": linear.roc_auc(frame_labels.astype(np.int64), frame_scores),
            "minimum_pair_projection": float(projection.min()),
            "mean_pair_projection": float(projection.mean()),
            "coefficient_sha256": coefficient_sha,
        })
    source_aurocs = [row["frame_teacher_active_auroc"] for row in folds]
    ordering_rates = [row["pair_ordering_rate"] for row in folds]
    gate = contract["gate"]
    checks = {
        "source_auroc_median": float(np.median(source_aurocs)) >= float(gate["source_auroc_median_at_least"]),
        "source_auroc_minimum": min(source_aurocs) >= float(gate["source_auroc_minimum_at_least"]),
        "source_macro_pair_ordering": float(np.mean(ordering_rates)) >= float(gate["source_macro_pair_ordering_at_least"]),
        "all_folds_finite": all(np.isfinite([
            row["pair_ordering_rate"], row["frame_teacher_active_auroc"],
            row["minimum_pair_projection"], row["mean_pair_projection"],
        ]).all() for row in folds),
    }
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "r767a_contract_sha256": common.sha256_file(args.linear_contract),
                   "r767a_report_sha256": common.sha256_file(args.linear_report)},
        "data": {"marker_frame_count": len(x), "active_frame_count": int(active.sum()),
                 "mixed_source_count": len(set(pair_sources.tolist())), "pair_count": len(pairs)},
        "pair_contract": contract["pair_construction"],
        "folds": folds,
        "summary": {"source_aurocs": source_aurocs, "source_auroc_median": float(np.median(source_aurocs)),
                    "source_auroc_minimum": float(min(source_aurocs)),
                    "source_macro_pair_ordering": float(np.mean(ordering_rates)),
                    "pair_time_gap_ms_median": float(np.median([row["absolute_time_gap_ms"] for row in pairs]))},
        "checks": checks, "diagnostic_gate_passed": all(checks.values()),
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
    parser.add_argument("--linear-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(json.dumps({"ok": True, **result["summary"], "passed": result["diagnostic_gate_passed"],
                      "output_sha256": common.sha256_file(parsed.output)}))
