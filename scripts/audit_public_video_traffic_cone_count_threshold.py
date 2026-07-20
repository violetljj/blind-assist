#!/usr/bin/env python3
"""Post-failure diagnostic for a mechanism-specific multi-cone corridor expert."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import evaluate_public_video_tristate_negative_controls as negative
import public_video_tristate_contract as prospective
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_traffic_cone_count_threshold_audit_v1"


def cone_expert_samples(
    samples: Sequence[dict[str, Any]], minimum_count: int
) -> list[dict[str, Any]]:
    if minimum_count <= 0:
        raise ValueError("minimum traffic-cone count must be positive")
    result: list[dict[str, Any]] = []
    for sample in samples:
        row = copy.deepcopy(sample)
        class_counts = sample.get("semantic_class_counts", {})
        count = int(class_counts.get("traffic cone", 0)) if isinstance(class_counts, dict) else 0
        row["semantic_group_counts"] = (
            {"barrier_structure": 1} if count >= minimum_count else {}
        )
        row["semantic_class_counts"] = (
            {"traffic cone": count} if count >= minimum_count else {}
        )
        result.append(row)
    return result


def evaluate_positive(
    source: dict[str, Any],
    review_body: dict[str, Any],
    contract: dict[str, Any],
    minimum_count: int,
) -> dict[str, Any]:
    samples = cone_expert_samples(source["samples"], minimum_count)
    settings = contract["lifecycle"]
    result = lifecycle.tristate_exit_intervals(
        samples,
        ["barrier_structure"],
        entry_window_samples=settings["entry_window_samples"],
        entry_min_active_samples=settings["entry_min_active_samples"],
        clear_absent_samples=settings["clear_absent_samples"],
    )
    score = lifecycle.score_intervals(result["intervals"], review_body["candidate_boundary"])
    risk = lifecycle.activity_window_diagnostics(
        samples, ["barrier_structure"], review_body["risk_present_window_ms"]
    )
    clear = lifecycle.activity_window_diagnostics(
        samples, ["barrier_structure"], review_body["stable_clear_window_ms"]
    )
    acceptance = contract["acceptance"]
    passed = bool(
        score["passed"]
        and risk["active_fraction"] is not None
        and risk["active_fraction"] >= acceptance["minimum_risk_present_active_fraction"]
        and clear["active_fraction"] is not None
        and clear["active_fraction"] <= acceptance["maximum_stable_clear_active_fraction"]
    )
    return {
        "lifecycle": result,
        "score": score,
        "risk_present": risk,
        "stable_clear": clear,
        "passed": passed,
    }


def evaluate_controls(
    sources: Sequence[dict[str, Any]], contract: dict[str, Any], minimum_count: int
) -> dict[str, Any]:
    results = []
    for source in sources:
        filtered = dict(source)
        filtered["samples"] = cone_expert_samples(source["samples"], minimum_count)
        results.append(negative.evaluate_source(filtered, contract))
    return {
        "passed_source_count": sum(result["passed"] for result in results),
        "source_count": len(results),
        "failed_source_ids": [result["source_id"] for result in results if not result["passed"]],
        "passed": all(result["passed"] for result in results),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    positive_scan = lifecycle.verify_json_sidecar(args.positive_scan)
    negative_scan = lifecycle.verify_json_sidecar(args.negative_scan)
    review = lifecycle.verify_json_sidecar(args.positive_review)
    contract, contract_attestation = prospective.load_contract(args.contract)
    review_body = review.get("review")
    if not isinstance(review_body, dict):
        raise ValueError("positive review body is missing")
    positive_sources = [
        source for source in positive_scan.get("sources", [])
        if source.get("source_id") == review_body.get("source_id")
    ]
    if len(positive_sources) != 1:
        raise ValueError("positive scan does not bind exactly one reviewed source")
    negative_sources = negative_scan.get("sources")
    if not isinstance(negative_sources, list) or not negative_sources:
        raise ValueError("negative scan has no sources")
    variants: dict[str, Any] = {}
    for minimum_count in range(1, 5):
        positive = evaluate_positive(
            positive_sources[0], review_body, contract, minimum_count
        )
        controls = evaluate_controls(negative_sources, contract, minimum_count)
        variants[str(minimum_count)] = {
            "minimum_traffic_cone_count_per_frame": minimum_count,
            "positive": positive,
            "negative_controls": controls,
            "passed_both": positive["passed"] and controls["passed"],
        }
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "positive_scan": {
                "path": str(args.positive_scan.resolve()),
                "sha256": common.sha256_file(args.positive_scan),
            },
            "positive_review": {
                "path": str(args.positive_review.resolve()),
                "sha256": common.sha256_file(args.positive_review),
            },
            "negative_scan": {
                "path": str(args.negative_scan.resolve()),
                "sha256": common.sha256_file(args.negative_scan),
            },
            "reference_contract": contract_attestation,
        },
        "audit_contract": {
            "kind": "post-r7.9-failure traffic-cone count threshold diagnosis",
            "thresholds_tested": [1, 2, 3, 4],
            "other_semantic_classes_used": [],
            "lifecycle_parameters_changed": False,
            "acceptance_thresholds_changed": False,
        },
        "variants": variants,
        "summary": {
            "thresholds_passing_positive": [
                int(key) for key, value in variants.items() if value["positive"]["passed"]
            ],
            "thresholds_passing_controls": [
                int(key) for key, value in variants.items() if value["negative_controls"]["passed"]
            ],
            "thresholds_passing_both": [
                int(key) for key, value in variants.items() if value["passed_both"]
            ],
        },
        "evidence_limit": "Post-failure mechanism-specific threshold audit only. It may nominate a future frozen expert but cannot retroactively pass r7.9 or authorize training, calibration, blind evaluation, Android runtime, or production.",
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(
        common.sha256_file(args.output) + "\n", encoding="ascii"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-scan", type=Path, required=True)
    parser.add_argument("--positive-review", type=Path, required=True)
    parser.add_argument("--negative-scan", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
