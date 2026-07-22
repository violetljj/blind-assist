#!/usr/bin/env python3
"""Summarize paired multi-seed R1.2d validation and event evidence."""

from __future__ import annotations

import argparse
import collections
import statistics
from pathlib import Path
from typing import Any

from r12d_contract import load_json, require, sha256_file, validate_matrix, write_json


METRICS = {
    "validation_recall": lambda row: row["validation"]["aggregate"]["recall"],
    "validation_small_recall": lambda row: row["validation"]["aggregate"]["small_recall"],
    "validation_london_like_recall": lambda row: row["validation"]["aggregate"]["london_like_recall"],
    "validation_precision": lambda row: row["validation"]["aggregate"]["precision"],
    "validation_false_detections_per_image": lambda row: row["validation"]["aggregate"]["false_detections_per_image"],
    "validation_worst_source_recall": lambda row: row["validation"]["worst_source_recall"],
    "positive_event_recall": lambda row: row["event_evaluation"]["aggregate"]["positive_event_recall"],
    "critical_event_miss_count": lambda row: row["event_evaluation"]["aggregate"]["critical_event_miss_count"],
    "target_conditioned_false_alert_count": lambda row: row["event_evaluation"]["aggregate"]["target_conditioned_false_alert_count"],
    "target_conditioned_false_alerts_per_minute": lambda row: row["event_evaluation"]["aggregate"]["target_conditioned_false_alerts_per_minute"],
    "delivered_repeated_alert_count": lambda row: row["event_evaluation"]["aggregate"]["delivered_repeated_alert_count"],
    "unassigned_route_inside_pressure_count": lambda row: row["event_evaluation"]["aggregate"]["unassigned_route_inside_pressure_count"],
    "observable_clearance_count": lambda row: row["event_evaluation"]["aggregate"]["observable_clearance_count"],
    "censored_clearance_count": lambda row: row["event_evaluation"]["aggregate"]["censored_clearance_count"],
    "identity_switch_count": lambda row: row["event_evaluation"]["aggregate"]["identity_switch_count"],
    "association_coverage": lambda row: row["event_evaluation"]["aggregate"]["association_coverage"],
    "association_ambiguous_frame_rate": lambda row: row["event_evaluation"]["aggregate"]["association_ambiguous_frame_rate"],
    "worst_source_association_coverage": lambda row: row["event_evaluation"]["aggregate"]["worst_source_association_coverage"],
    "worst_source_ambiguity_rate": lambda row: row["event_evaluation"]["aggregate"]["worst_source_ambiguity_rate"],
    "london_event_recall": lambda row: row["event_evaluation"]["aggregate"]["london"]["event_recall"],
    "london_visible_anchor_recall": lambda row: row["event_evaluation"]["aggregate"]["london"]["visible_anchor_recall"],
    "london_frame_recall": lambda row: row["event_evaluation"]["aggregate"]["london"]["frame_recall"],
}


LOWER_IS_BETTER = {
    "validation_false_detections_per_image", "critical_event_miss_count",
    "target_conditioned_false_alert_count", "target_conditioned_false_alerts_per_minute",
    "delivered_repeated_alert_count", "censored_clearance_count", "identity_switch_count",
    "unassigned_route_inside_pressure_count",
    "association_ambiguous_frame_rate", "worst_source_ambiguity_rate",
}


def stats(values: list[float], lower_is_better: bool) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "standard_deviation": statistics.pstdev(values), "range": max(values) - min(values),
        "worst": max(values) if lower_is_better else min(values), "best": min(values) if lower_is_better else max(values),
    }


def optional_stats(values: list[float | None]) -> dict[str, Any]:
    observed = [float(value) for value in values if value is not None]
    return {
        "observed_seed_count": len(observed), "total_seed_count": len(values),
        "mean": statistics.fmean(observed) if observed else None,
        "median": statistics.median(observed) if observed else None,
        "standard_deviation": statistics.pstdev(observed) if observed else None,
        "range": max(observed) - min(observed) if observed else None,
        "worst": max(observed) if observed else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve(); matrix_path = args.matrix.resolve(); matrix = validate_matrix(matrix_path, repo)
    reports = [load_json(path.resolve()) for path in args.evaluation]
    for path, report in zip(args.evaluation, reports):
        require(report["matrix_sha256"] == sha256_file(matrix_path), f"matrix mismatch: {path}")
    trained = [row for row in reports if row["training_run"] is not None]
    external = [row for row in reports if row["training_run"] is None]
    expected_seeds = set(matrix["training"]["seeds"])
    arms: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trained:
        arms[row["training_run"]["arm_id"]].append(row)
    require(set(arms) == {row["arm_id"] for row in matrix["paired_arms"]}, "paired arm coverage incomplete")
    for arm_id, rows in arms.items():
        require({row["training_run"]["seed"] for row in rows} == expected_seeds, f"seed coverage incomplete: {arm_id}")
    p2_id = next(row["arm_id"] for row in matrix["paired_arms"] if row["p2"])
    control_id = next(row["arm_id"] for row in matrix["paired_arms"] if not row["p2"])
    p2_by_seed = {row["training_run"]["seed"]: row for row in arms[p2_id]}
    control_by_seed = {row["training_run"]["seed"]: row for row in arms[control_id]}
    for seed in expected_seeds:
        require(p2_by_seed[seed]["training_run"]["shared_backbone_tensor_sha256"] == control_by_seed[seed]["training_run"]["shared_backbone_tensor_sha256"],
                f"shared backbone parity failed: {seed}")
    arm_summary = {}
    for arm_id, rows in sorted(arms.items()):
        arm_summary[arm_id] = {name: stats([float(function(row)) for row in rows], name in LOWER_IS_BETTER) for name, function in METRICS.items()}
        arm_summary[arm_id]["clearance_delay_ms"] = {
            "p50_by_seed": {str(row["training_run"]["seed"]): row["event_evaluation"]["aggregate"]["clearance_p50_ms"] for row in rows},
            "p95_by_seed": {str(row["training_run"]["seed"]): row["event_evaluation"]["aggregate"]["clearance_p95_ms"] for row in rows},
            "p50_stability": optional_stats([row["event_evaluation"]["aggregate"]["clearance_p50_ms"] for row in rows]),
            "p95_stability": optional_stats([row["event_evaluation"]["aggregate"]["clearance_p95_ms"] for row in rows]),
        }
        source_success: dict[str, list[int]] = collections.defaultdict(list)
        for row in rows:
            for event in row["event_evaluation"]["events"]:
                if not event["gate_eligible"]:
                    continue
                success = event["event_hit"] if event["expected_class"] == "positive" else not event["target_conditioned_false_alert"]
                source_success[event["source_id"]].append(int(success))
        arm_summary[arm_id]["source_success_seed_fraction"] = {source: sum(values) / len(values) for source, values in sorted(source_success.items())}
    paired = {}
    for name, function in METRICS.items():
        paired[name] = {str(seed): float(function(p2_by_seed[seed])) - float(function(control_by_seed[seed])) for seed in sorted(expected_seeds)}
        paired[name]["mean"] = statistics.fmean(paired[name].values())
    p2_supported = (
        arm_summary[p2_id]["london_event_recall"]["worst"] > arm_summary[control_id]["london_event_recall"]["worst"]
        and arm_summary[p2_id]["positive_event_recall"]["worst"] >= arm_summary[control_id]["positive_event_recall"]["worst"]
        and arm_summary[p2_id]["target_conditioned_false_alert_count"]["worst"] <= arm_summary[control_id]["target_conditioned_false_alert_count"]["worst"]
        and arm_summary[p2_id]["validation_worst_source_recall"]["worst"] >= arm_summary[control_id]["validation_worst_source_recall"]["worst"]
    )
    result = {
        "schema": "blindassist_ustrf_r12d_multiseed_summary_v1", "matrix_sha256": sha256_file(matrix_path),
        "seeds": sorted(expected_seeds), "arms": arm_summary, "paired_p2_minus_control": paired,
        "external_references": [{"model_id": row["model_id"], "weights_sha256": row["weights_sha256"],
                                 "event": row["event_evaluation"]["aggregate"], "validation": row["validation"]} for row in external],
        "decision": {
            "p2_hypothesis_supported_under_preregistered_rule": p2_supported,
            "production_model_replacement_authorized": False,
            "r13_inventory_read_authorized": False,
            "note": "Positive research evidence does not grant human truth, device geometry, INT8, Android, or production authority.",
        },
    }
    write_json(args.output.resolve(), result)
    print("USTRF_R12D_SUMMARY_OK", p2_supported, arm_summary[p2_id]["london_event_recall"]["worst"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
