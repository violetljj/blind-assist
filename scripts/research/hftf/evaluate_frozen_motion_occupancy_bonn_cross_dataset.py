#!/usr/bin/env python3
"""Evaluate the frozen current-occupancy model across two Bonn sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_frozen_motion_conditioned_occupancy_a0 import evaluate
from evaluate_metric3d_probabilistic_occupancy_a0 import load_report


SCHEMA = "blindassist_hftf_frozen_motion_occupancy_bonn_cross_dataset"


def admission(
    reports: list[dict[str, Any]],
    per_source: list[dict[str, Any]],
    pooled: dict[str, Any],
) -> dict[str, bool]:
    return {
        "exactly_two_sources": len(reports) == 2,
        "each_paired_valid_fraction_at_least_0_90": all(
            float(report["paired_valid_fraction"]) >= 0.90 for report in reports
        ),
        "each_source_opportunities_at_least_900": all(
            int(result["opportunities"]) >= 900 for result in per_source
        ),
        "all_pooled_a0_1_probability_gates": all(pooled["gates"].values()),
    }


def evaluate_cross_dataset(
    reports: list[dict[str, Any]], model: dict[str, Any], raft_weights: Path
) -> dict[str, Any]:
    per_source = []
    for report in reports:
        result = evaluate(report, model, raft_weights)
        per_source.append(
            {
                "candidate_model_id": report.get("candidate_model_id"),
                "paired_valid_fraction": report["paired_valid_fraction"],
                **result,
            }
        )
    merged = {"frames": [frame for report in reports for frame in report["frames"]]}
    pooled = evaluate(merged, model, raft_weights)
    gates = admission(reports, per_source, pooled)
    return {
        "schema": SCHEMA,
        "sources": len(reports),
        "per_source": per_source,
        "pooled": pooled,
        "cross_dataset_gates": gates,
        "status": (
            "MOTION_OCCUPANCY_A0_1_BONN_CROSS_DATASET_SUPPORTED_DEVELOPMENT_ONLY"
            if all(gates.values())
            else "MOTION_OCCUPANCY_A0_1_BONN_CROSS_DATASET_NOT_SUPPORTED"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [load_report(path) for path in args.report]
    model = json.loads(args.model.read_text(encoding="utf-8"))
    result = evaluate_cross_dataset(reports, model, args.raft_weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
