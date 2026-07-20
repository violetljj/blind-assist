#!/usr/bin/env python3
"""Audit a chromatic cone expert with mechanism-specific clear persistence."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import evaluate_public_video_multicone_persistent_entry_control as persistent
import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_chromatic_cone_lifecycle_audit_v1"


def chromatic_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for sample in samples:
        row = copy.deepcopy(sample)
        detections = sample.get("detections", [])
        if not isinstance(detections, list):
            raise ValueError("feature sample detections must be a list")
        accepted = [
            detection for detection in detections
            if float(detection["features"]["high_saturation_fraction"])
            > float(detection["features"]["dark_fraction"])
        ]
        row["semantic_group_counts"] = (
            {"barrier_structure": len(accepted)} if accepted else {}
        )
        row["semantic_class_counts"] = (
            {"chromatic traffic cone": len(accepted)} if accepted else {}
        )
        result.append(row)
    return result


def source_by_role(report: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [
        source for source in report.get("sources", [])
        if source.get("diagnostic_role") == role
    ]
    if len(matches) != 1:
        raise ValueError(f"feature report does not bind exactly one role: {role}")
    return matches[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    feature_report = lifecycle.verify_json_sidecar(args.features)
    dense_review = lifecycle.verify_json_sidecar(args.dense_review)
    sparse_review = lifecycle.verify_json_sidecar(args.sparse_review)
    extra_negative_reports = [
        lifecycle.verify_json_sidecar(path) for path in args.extra_negative_features
    ]
    if feature_report.get("schema") != "blindassist_public_video_traffic_cone_detection_features_v1":
        raise ValueError("unexpected traffic-cone feature schema")
    dense_risk = source_by_role(feature_report, "dense traffic-cone work-zone risk")
    dense_clear = source_by_role(feature_report, "post-exit ordinary road clear")
    gate_control = source_by_role(feature_report, "fixed gate false traffic-cone detections")
    sparse_risk = source_by_role(feature_report, "sparse red-white delineator work-zone risk")
    dense_review_body = dense_review["review"]
    sparse_review_body = sparse_review["review"]
    dense_samples = sorted(
        chromatic_samples(dense_risk["samples"]) + chromatic_samples(dense_clear["samples"]),
        key=lambda sample: int(sample["timestamp_ms"]),
    )
    gate_samples = chromatic_samples(gate_control["samples"])
    sparse_samples = chromatic_samples(sparse_risk["samples"])
    extra_negative_sources = [
        source
        for report in extra_negative_reports
        for source in report.get("sources", [])
    ]
    variants: dict[str, Any] = {}
    for clear_absent_samples in range(3, 7):
        dense_result = lifecycle.tristate_exit_intervals(
            dense_samples,
            ["barrier_structure"],
            entry_window_samples=3,
            entry_min_active_samples=2,
            clear_absent_samples=clear_absent_samples,
        )
        dense_score = lifecycle.score_intervals(
            dense_result["intervals"], dense_review_body["candidate_boundary"]
        )
        gate_result = lifecycle.tristate_exit_intervals(
            gate_samples,
            ["barrier_structure"],
            entry_window_samples=3,
            entry_min_active_samples=2,
            clear_absent_samples=clear_absent_samples,
        )
        gate_passed = not gate_result["intervals"] and gate_result["terminal_state"] == "clear"
        sparse_result = lifecycle.tristate_exit_intervals(
            sparse_samples,
            ["barrier_structure"],
            entry_window_samples=3,
            entry_min_active_samples=2,
            clear_absent_samples=clear_absent_samples,
        )
        sparse_score = persistent.score_persistent_entry(
            sparse_result, sparse_review_body["visual_entry_transition_window_ms"]
        )
        extra_negative_results = []
        for source in extra_negative_sources:
            result = lifecycle.tristate_exit_intervals(
                chromatic_samples(source["samples"]),
                ["barrier_structure"],
                entry_window_samples=3,
                entry_min_active_samples=2,
                clear_absent_samples=clear_absent_samples,
            )
            extra_negative_results.append({
                "source_id": source["source_id"],
                "lifecycle": result,
                "passed": not result["intervals"] and result["terminal_state"] == "clear",
            })
        extra_negative_passed = all(
            result["passed"] for result in extra_negative_results
        )
        variants[str(clear_absent_samples)] = {
            "clear_absent_samples": clear_absent_samples,
            "dense_exit": {"lifecycle": dense_result, "score": dense_score},
            "fixed_gate_control": {"lifecycle": gate_result, "passed": gate_passed},
            "sparse_persistent_entry": {
                "lifecycle": sparse_result,
                "score": sparse_score,
            },
            "extra_negative_controls": {
                "sources": extra_negative_results,
                "passed": extra_negative_passed,
            },
            "passed_all_three": (
                dense_score["passed"]
                and gate_passed
                and sparse_score["passed"]
                and extra_negative_passed
            ),
        }
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "features": {"path": str(args.features.resolve()), "sha256": common.sha256_file(args.features)},
            "dense_review": {"path": str(args.dense_review.resolve()), "sha256": common.sha256_file(args.dense_review)},
            "sparse_review": {"path": str(args.sparse_review.resolve()), "sha256": common.sha256_file(args.sparse_review)},
            "extra_negative_features": [
                {"path": str(path.resolve()), "sha256": common.sha256_file(path)}
                for path in args.extra_negative_features
            ],
        },
        "expert_contract": {
            "target_class": "traffic cone",
            "detection_acceptance": "high_saturation_fraction > dark_fraction",
            "minimum_detections_per_active_frame": 1,
            "entry_window_samples": 3,
            "entry_min_active_samples": 2,
            "clear_absent_samples_tested": [3, 4, 5, 6],
            "learned_parameters": 0,
        },
        "coverage": {
            "dense_risk": {
                "active_samples": sum(bool(sample["semantic_group_counts"]) for sample in dense_samples[:29]),
                "sample_count": 29,
            },
            "dense_clear": {
                "active_samples": sum(bool(sample["semantic_group_counts"]) for sample in dense_samples[29:]),
                "sample_count": 17,
            },
            "fixed_gate": {
                "active_samples": sum(bool(sample["semantic_group_counts"]) for sample in gate_samples),
                "sample_count": len(gate_samples),
            },
            "sparse_risk": {
                "active_samples": sum(bool(sample["semantic_group_counts"]) for sample in sparse_samples),
                "sample_count": len(sparse_samples),
            },
        },
        "variants": variants,
        "summary": {
            "clear_absent_values_passing_all_three": [
                int(key) for key, value in variants.items() if value["passed_all_three"]
            ]
        },
        "evidence_limit": "Post-r7.10 exploratory chromatic-cone lifecycle audit only. Passing values may define a future prospective contract but cannot authorize training, calibration, blind evaluation, Android runtime, or production.",
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
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dense-review", type=Path, required=True)
    parser.add_argument("--sparse-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-negative-features", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
