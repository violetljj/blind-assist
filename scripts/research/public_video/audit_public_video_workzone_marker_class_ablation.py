#!/usr/bin/env python3
"""Diagnose which exploratory work-zone marker classes cause control failures.

This is a post-failure leave-one-class-out audit. It may nominate a future
contract but cannot retroactively make r7.9 prospective evidence pass.
"""

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
import run_public_silver_prompt_free_semantic_probe as semantic


SCHEMA = "blindassist_public_video_workzone_marker_class_ablation_v1"
BASELINE_BARRIER_CLASSES = frozenset(semantic.SEMANTIC_GROUPS["barrier_structure"])
SURFACE_CLASSES = frozenset(semantic.SEMANTIC_GROUPS["surface_material"])


def filtered_samples(
    samples: Sequence[dict[str, Any]], marker_classes: set[str] | frozenset[str]
) -> list[dict[str, Any]]:
    allowed_barrier = BASELINE_BARRIER_CLASSES | frozenset(marker_classes)
    result: list[dict[str, Any]] = []
    for sample in samples:
        row = copy.deepcopy(sample)
        class_counts = sample.get("semantic_class_counts", {})
        if not isinstance(class_counts, dict):
            raise ValueError("sample semantic_class_counts must be an object")
        surface_count = sum(
            int(count) for name, count in class_counts.items() if name in SURFACE_CLASSES
        )
        barrier_count = sum(
            int(count) for name, count in class_counts.items() if name in allowed_barrier
        )
        group_counts: dict[str, int] = {}
        if surface_count:
            group_counts["surface_material"] = surface_count
        if barrier_count:
            group_counts["barrier_structure"] = barrier_count
        row["semantic_group_counts"] = group_counts
        row["semantic_class_counts"] = {
            name: int(count)
            for name, count in class_counts.items()
            if name in SURFACE_CLASSES or name in allowed_barrier
        }
        result.append(row)
    return result


def positive_result(
    source: dict[str, Any],
    review_body: dict[str, Any],
    contract: dict[str, Any],
    marker_classes: set[str] | frozenset[str],
) -> dict[str, Any]:
    samples = filtered_samples(source["samples"], marker_classes)
    settings = contract["lifecycle"]
    result = lifecycle.tristate_exit_intervals(
        samples,
        settings["selected_groups"],
        entry_window_samples=settings["entry_window_samples"],
        entry_min_active_samples=settings["entry_min_active_samples"],
        clear_absent_samples=settings["clear_absent_samples"],
    )
    score = lifecycle.score_intervals(result["intervals"], review_body["candidate_boundary"])
    risk = lifecycle.activity_window_diagnostics(
        samples, settings["selected_groups"], review_body["risk_present_window_ms"]
    )
    clear = lifecycle.activity_window_diagnostics(
        samples, settings["selected_groups"], review_body["stable_clear_window_ms"]
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


def negative_results(
    sources: Sequence[dict[str, Any]],
    contract: dict[str, Any],
    marker_classes: set[str] | frozenset[str],
) -> dict[str, Any]:
    results = []
    for source in sources:
        filtered = dict(source)
        filtered["samples"] = filtered_samples(source["samples"], marker_classes)
        results.append(negative.evaluate_source(filtered, contract))
    return {
        "source_count": len(results),
        "passed_source_count": sum(result["passed"] for result in results),
        "failed_source_ids": [result["source_id"] for result in results if not result["passed"]],
        "passed": all(result["passed"] for result in results),
        "sources": results,
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

    full = set(prospective.WORKZONE_MARKER_ADDITIONS)
    variants = {"full_r79": full}
    for class_name in sorted(full):
        variants[f"drop_{class_name.replace(' ', '_')}"] = full - {class_name}
    reports: dict[str, Any] = {}
    for name, marker_classes in variants.items():
        positive = positive_result(
            positive_sources[0], review_body, contract, marker_classes
        )
        controls = negative_results(negative_sources, contract, marker_classes)
        reports[name] = {
            "workzone_marker_additions": sorted(marker_classes),
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
            "contract": contract_attestation,
        },
        "audit_contract": {
            "kind": "post-r7.9-failure leave-one-marker-class-out diagnosis",
            "baseline_barrier_classes_always_retained": sorted(BASELINE_BARRIER_CLASSES),
            "surface_classes_always_retained": sorted(SURFACE_CLASSES),
            "lifecycle_parameters_changed": False,
            "thresholds_changed": False,
        },
        "variants": reports,
        "summary": {
            "variant_count": len(reports),
            "variants_passing_positive": [
                name for name, value in reports.items() if value["positive"]["passed"]
            ],
            "variants_passing_controls": [
                name for name, value in reports.items() if value["negative_controls"]["passed"]
            ],
            "variants_passing_both": [
                name for name, value in reports.items() if value["passed_both"]
            ],
        },
        "evidence_limit": "Post-failure exploratory ablation only. A passing variant may define a future frozen contract but cannot retroactively pass r7.9 or authorize training, calibration, blind evaluation, Android runtime, or production.",
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
