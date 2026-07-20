#!/usr/bin/env python3
"""Run r7.75 nearest-time pair probe after training-fold geometry residualization."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_marker_relation_pair_ranking_probe as pair_probe


SCHEMA = "blindassist_public_video_marker_relation_geometry_residual_probe_v1"


def source_equal_pair_weights(pair_sources: np.ndarray) -> np.ndarray:
    sources = np.asarray(pair_sources).astype(str)
    unique = sorted(set(sources.tolist()))
    weights = np.zeros(len(sources), dtype=np.float64)
    for source in unique:
        selected = sources == source
        weights[selected] = 1.0 / len(unique) / int(selected.sum())
    return weights


def fit_geometry_residualizer(
    x: np.ndarray, pairs: list[dict[str, Any]], selected: np.ndarray, alpha: float
) -> np.ndarray:
    pair_indices = np.flatnonzero(selected)
    semantic_delta = np.stack([
        x[pairs[index]["positive_index"], :-3] - x[pairs[index]["negative_index"], :-3]
        for index in pair_indices
    ])
    geometry_delta = np.stack([
        x[pairs[index]["positive_index"], -3:] - x[pairs[index]["negative_index"], -3:]
        for index in pair_indices
    ])
    sources = np.asarray([pairs[index]["source_id"] for index in pair_indices])
    weights = source_equal_pair_weights(sources)
    return np.linalg.solve(
        geometry_delta.T @ (geometry_delta * weights[:, None]) + float(alpha) * np.eye(3),
        geometry_delta.T @ (semantic_delta * weights[:, None]),
    )


def residualize_frames(x: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    semantic = values[:, :-3] - values[:, -3:] @ np.asarray(mapping, dtype=np.float64)
    return np.concatenate([semantic, np.zeros((len(values), 3), dtype=np.float64)], axis=1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.linear_contract, args.r774_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    for path, key in ((args.linear_contract, "r767a_contract_sha256"),
                      (args.r774_report, "r774_report_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    if lifecycle.verify_json_sidecar(args.r774_report).get("diagnostic_gate_passed") is not False:
        raise ValueError("r7.74 failure evidence is missing")
    linear_contract = common.load_json(args.linear_contract)
    x, y, sources, timestamps = pair_probe.load_data(linear_contract)
    active = y > 0.0
    pairs = pair_probe.nearest_time_pairs(active, sources, timestamps)
    pair_sources = np.asarray([row["source_id"] for row in pairs])
    folds = []
    spec = contract["geometry_residualization"]
    for held_source in sorted(set(pair_sources.tolist())):
        train = pair_sources != held_source
        test = ~train
        mapping = fit_geometry_residualizer(x, pairs, train, float(spec["ridge_alpha"]))
        residual_x = residualize_frames(x, mapping)
        residual_deltas = np.stack([
            residual_x[row["positive_index"]] - residual_x[row["negative_index"]] for row in pairs
        ])
        model = pair_probe.fit_signed_pair_ridge(
            residual_deltas[train], pair_sources[train], float(contract["probe"]["ridge_alpha"])
        )
        frame_selected = sources == held_source
        frame_scores = (residual_x[frame_selected] / model["scale"]) @ model["coefficients"]
        projection = pair_probe.pair_projection(model, residual_deltas[test])
        raw_test_deltas = np.stack([x[row["positive_index"]] - x[row["negative_index"]] for index, row in enumerate(pairs) if test[index]])
        residual_test_deltas = residual_deltas[test]
        folds.append({"held_out_source_id": held_source, "held_out_pair_count": int(test.sum()),
                      "pair_ordering_rate": float((projection > 0.0).mean()),
                      "frame_teacher_active_auroc": pair_probe.linear.roc_auc(active[frame_selected].astype(np.int64), frame_scores),
                      "semantic_delta_norm_ratio_after_before": float(np.linalg.norm(residual_test_deltas[:, :-3]) / max(np.linalg.norm(raw_test_deltas[:, :-3]), 1e-12)),
                      "mean_pair_projection": float(projection.mean())})
    aurocs = [row["frame_teacher_active_auroc"] for row in folds]
    ordering = [row["pair_ordering_rate"] for row in folds]
    gate = contract["gate"]
    checks = {"source_auroc_median": float(np.median(aurocs)) >= float(gate["source_auroc_median_at_least"]),
              "source_auroc_minimum": min(aurocs) >= float(gate["source_auroc_minimum_at_least"]),
              "source_macro_pair_ordering": float(np.mean(ordering)) >= float(gate["source_macro_pair_ordering_at_least"]),
              "all_folds_finite": all(np.isfinite([row["frame_teacher_active_auroc"], row["pair_ordering_rate"], row["semantic_delta_norm_ratio_after_before"], row["mean_pair_projection"]]).all() for row in folds)}
    report = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"contract_sha256": common.sha256_file(args.contract), "r774_report_sha256": common.sha256_file(args.r774_report)},
              "data": {"pair_count": len(pairs), "mixed_source_count": len(set(pair_sources.tolist()))},
              "geometry_residualization_contract": spec, "folds": folds,
              "summary": {"source_aurocs": aurocs, "source_auroc_median": float(np.median(aurocs)),
                          "source_auroc_minimum": float(min(aurocs)), "source_macro_pair_ordering": float(np.mean(ordering)),
                          "semantic_delta_norm_ratio_median": float(np.median([row["semantic_delta_norm_ratio_after_before"] for row in folds]))},
              "checks": checks, "diagnostic_gate_passed": all(checks.values()), "authorization": contract["authorization"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--linear-contract", type=Path, required=True)
    parser.add_argument("--r774-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(json.dumps({"ok": True, **result["summary"], "passed": result["diagnostic_gate_passed"],
                      "output_sha256": common.sha256_file(parsed.output)}))
