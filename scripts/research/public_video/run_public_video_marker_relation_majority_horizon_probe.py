#!/usr/bin/env python3
"""Run r7.76 pair probe for majority-horizon future-route intrusion."""

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


SCHEMA = "blindassist_public_video_marker_relation_majority_horizon_probe_v1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.linear_contract, args.r775_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    for path, key in ((args.linear_contract, "r767a_contract_sha256"),
                      (args.r775_report, "r775_report_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    if lifecycle.verify_json_sidecar(args.r775_report).get("diagnostic_gate_passed") is not False:
        raise ValueError("r7.75 failure evidence is missing")
    linear_contract = common.load_json(args.linear_contract)
    x, y, sources, timestamps = pair_probe.load_data(linear_contract)
    threshold = float(contract["target"]["strong_intrusion_fraction_at_least"])
    strong = y >= threshold
    pairs = pair_probe.nearest_time_pairs(strong, sources, timestamps)
    deltas = np.stack([x[row["positive_index"]] - x[row["negative_index"]] for row in pairs])
    pair_sources = np.asarray([row["source_id"] for row in pairs])
    folds = []
    for held_source in sorted(set(pair_sources.tolist())):
        train = pair_sources != held_source
        test = ~train
        model = pair_probe.fit_signed_pair_ridge(deltas[train], pair_sources[train], float(contract["probe"]["ridge_alpha"]))
        frame_selected = sources == held_source
        scores = (x[frame_selected] / model["scale"]) @ model["coefficients"]
        projection = pair_probe.pair_projection(model, deltas[test])
        folds.append({"held_out_source_id": held_source, "held_out_pair_count": int(test.sum()),
                      "pair_ordering_rate": float((projection > 0.0).mean()),
                      "frame_strong_intrusion_auroc": pair_probe.linear.roc_auc(strong[frame_selected].astype(np.int64), scores),
                      "minimum_pair_projection": float(projection.min()), "mean_pair_projection": float(projection.mean())})
    aurocs = [row["frame_strong_intrusion_auroc"] for row in folds]
    ordering = [row["pair_ordering_rate"] for row in folds]
    gate = contract["gate"]
    checks = {"source_auroc_median": float(np.median(aurocs)) >= float(gate["source_auroc_median_at_least"]),
              "source_auroc_minimum": min(aurocs) >= float(gate["source_auroc_minimum_at_least"]),
              "source_macro_pair_ordering": float(np.mean(ordering)) >= float(gate["source_macro_pair_ordering_at_least"]),
              "all_folds_finite": all(np.isfinite([row["frame_strong_intrusion_auroc"], row["pair_ordering_rate"], row["minimum_pair_projection"], row["mean_pair_projection"]]).all() for row in folds)}
    levels, counts = np.unique(np.rint(y * 3).astype(int), return_counts=True)
    report = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"contract_sha256": common.sha256_file(args.contract), "r775_report_sha256": common.sha256_file(args.r775_report)},
              "data": {"marker_frame_count": len(x), "strong_frame_count": int(strong.sum()),
                       "mixed_source_count": len(set(pair_sources.tolist())), "pair_count": len(pairs),
                       "teacher_hit_count_distribution": {str(level): int(count) for level, count in zip(levels, counts)}},
              "target_contract": contract["target"], "folds": folds,
              "summary": {"source_aurocs": aurocs, "source_auroc_median": float(np.median(aurocs)),
                          "source_auroc_minimum": float(min(aurocs)), "source_macro_pair_ordering": float(np.mean(ordering)),
                          "pair_time_gap_ms_median": float(np.median([row["absolute_time_gap_ms"] for row in pairs]))},
              "checks": checks, "diagnostic_gate_passed": all(checks.values()), "authorization": contract["authorization"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--linear-contract", type=Path, required=True)
    parser.add_argument("--r775-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(json.dumps({"ok": True, **result["summary"], "passed": result["diagnostic_gate_passed"],
                      "output_sha256": common.sha256_file(parsed.output)}))
