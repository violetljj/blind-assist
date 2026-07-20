#!/usr/bin/env python3
"""Run r7.74 same-source geometry-matched marker-relation probe."""

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


SCHEMA = "blindassist_public_video_marker_relation_geometry_matched_probe_v1"


def geometry_matched_pairs(
    x: np.ndarray, active: np.ndarray, sources: np.ndarray, timestamps: np.ndarray,
    log_area_weight: float, centroid_weight: float,
) -> list[dict[str, Any]]:
    values = np.asarray(x, dtype=np.float64)
    labels = np.asarray(active, dtype=bool)
    source_values = np.asarray(sources).astype(str)
    time_values = np.asarray(timestamps, dtype=np.int64)
    geometry = values[:, -3:]
    if values.ndim != 2 or len(values) != len(labels) or np.any(geometry[:, 0] <= 0.0):
        raise ValueError("geometry matcher requires aligned positive-area relation vectors")
    pairs = []
    for source in sorted(set(source_values.tolist())):
        indices = np.flatnonzero(source_values == source)
        positive = indices[labels[indices]]
        negative = indices[~labels[indices]]
        if not len(positive) or not len(negative):
            continue
        for positive_index in positive:
            candidates = []
            for negative_index in negative:
                area_distance = abs(np.log(geometry[positive_index, 0]) - np.log(geometry[negative_index, 0]))
                centroid_distance = abs(geometry[positive_index, 1] - geometry[negative_index, 1]) + abs(geometry[positive_index, 2] - geometry[negative_index, 2])
                distance = float(log_area_weight * area_distance + centroid_weight * centroid_distance)
                candidates.append((distance, abs(int(time_values[positive_index]) - int(time_values[negative_index])),
                                   int(time_values[negative_index]), int(negative_index)))
            distance, gap, _, negative_index = min(candidates)
            pairs.append({"source_id": source, "positive_index": int(positive_index),
                          "negative_index": int(negative_index), "geometry_distance": distance,
                          "absolute_time_gap_ms": int(gap)})
    return pairs


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.linear_contract, args.r773_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    for path, key in ((args.linear_contract, "r767a_contract_sha256"),
                      (args.r773_report, "r773_report_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    if lifecycle.verify_json_sidecar(args.r773_report).get("posthoc_diagnostic_passed") is not False:
        raise ValueError("r7.73 failure evidence is missing")
    linear_contract = common.load_json(args.linear_contract)
    x, y, sources, timestamps = pair_probe.load_data(linear_contract)
    active = y > 0.0
    matcher = contract["geometry_match"]
    pairs = geometry_matched_pairs(x, active, sources, timestamps,
                                   float(matcher["log_area_weight"]), float(matcher["centroid_l1_weight"]))
    deltas = np.stack([x[row["positive_index"]] - x[row["negative_index"]] for row in pairs])
    pair_sources = np.asarray([row["source_id"] for row in pairs])
    folds = []
    for held_source in sorted(set(pair_sources.tolist())):
        train = pair_sources != held_source
        test = ~train
        model = pair_probe.fit_signed_pair_ridge(deltas[train], pair_sources[train], float(contract["probe"]["ridge_alpha"]))
        frame_selected = sources == held_source
        frame_scores = (x[frame_selected] / model["scale"]) @ model["coefficients"]
        projection = pair_probe.pair_projection(model, deltas[test])
        folds.append({"held_out_source_id": held_source, "held_out_pair_count": int(test.sum()),
                      "pair_ordering_rate": float((projection > 0.0).mean()),
                      "frame_teacher_active_auroc": pair_probe.linear.roc_auc(active[frame_selected].astype(np.int64), frame_scores),
                      "mean_geometry_distance": float(np.mean([pairs[index]["geometry_distance"] for index in np.flatnonzero(test)])),
                      "mean_pair_projection": float(projection.mean())})
    aurocs = [row["frame_teacher_active_auroc"] for row in folds]
    ordering = [row["pair_ordering_rate"] for row in folds]
    gate = contract["gate"]
    checks = {"source_auroc_median": float(np.median(aurocs)) >= float(gate["source_auroc_median_at_least"]),
              "source_auroc_minimum": min(aurocs) >= float(gate["source_auroc_minimum_at_least"]),
              "source_macro_pair_ordering": float(np.mean(ordering)) >= float(gate["source_macro_pair_ordering_at_least"]),
              "all_folds_finite": all(np.isfinite([row["frame_teacher_active_auroc"], row["pair_ordering_rate"], row["mean_geometry_distance"], row["mean_pair_projection"]]).all() for row in folds)}
    report = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"contract_sha256": common.sha256_file(args.contract), "r773_report_sha256": common.sha256_file(args.r773_report)},
              "data": {"pair_count": len(pairs), "mixed_source_count": len(set(pair_sources.tolist()))},
              "geometry_match_contract": matcher, "folds": folds,
              "summary": {"source_aurocs": aurocs, "source_auroc_median": float(np.median(aurocs)),
                          "source_auroc_minimum": float(min(aurocs)), "source_macro_pair_ordering": float(np.mean(ordering)),
                          "geometry_distance_median": float(np.median([row["geometry_distance"] for row in pairs])),
                          "time_gap_ms_median_diagnostic": float(np.median([row["absolute_time_gap_ms"] for row in pairs]))},
              "checks": checks, "diagnostic_gate_passed": all(checks.values()), "authorization": contract["authorization"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--linear-contract", type=Path, required=True)
    parser.add_argument("--r773-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(json.dumps({"ok": True, **result["summary"], "passed": result["diagnostic_gate_passed"],
                      "output_sha256": common.sha256_file(parsed.output)}))
