#!/usr/bin/env python3
"""Evaluate one held-out positive exit against the frozen chromatic marker contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_chromatic_marker_policy as chromatic
import public_video_tristate_contract as prospective
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_chromatic_marker_prospective_positive_v1"
REVIEW_SCHEMA = "blindassist_public_video_prospective_positive_review_v1"
FEATURE_SCHEMA = "blindassist_public_video_traffic_cone_detection_features_v1"
INVENTORY_AUDIT_SCHEMA = "blindassist_public_video_prospective_source_inventory_audit_v1"
ACCEPT_DECISION = "accept_as_prospective_discovery_only_exit_boundary"


def one_source(rows: Any, source_id: str, label: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} sources must be a list")
    matches = [row for row in rows if row.get("source_id") == source_id]
    if len(matches) != 1:
        raise ValueError(f"{label} must bind exactly one source: {source_id}")
    return matches[0]


def evaluate(
    *,
    features: dict[str, Any],
    review: dict[str, Any],
    inventory_audit: dict[str, Any],
    contract: dict[str, Any],
    contract_attestation: dict[str, str],
    feature_sha256: str,
    inventory_audit_sha256: str,
) -> dict[str, Any]:
    policy = chromatic.validate_policy(contract)
    if features.get("schema") != FEATURE_SCHEMA:
        raise ValueError("unexpected construction-marker feature schema")
    if features.get("prospective_contract") != contract_attestation:
        raise ValueError("feature report is not bound to the supplied frozen contract")
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("unexpected prospective positive review schema")
    if inventory_audit.get("schema") != INVENTORY_AUDIT_SCHEMA:
        raise ValueError("unexpected source inventory audit schema")
    if inventory_audit.get("frozen_contract") != contract_attestation:
        raise ValueError("source inventory audit is not bound to the supplied contract")

    attestation = review.get("prospective_attestation")
    if not isinstance(attestation, dict):
        raise ValueError("review prospective attestation is missing")
    if attestation.get("frozen_contract_sha256") != contract_attestation["sha256"]:
        raise ValueError("review frozen contract hash mismatch")
    if attestation.get("feature_report_sha256") != feature_sha256:
        raise ValueError("review does not attest the supplied frozen feature report")
    if attestation.get("source_inventory_audit_sha256") != inventory_audit_sha256:
        raise ValueError("review does not attest the supplied source inventory audit")
    if attestation.get("policy_frozen_before_visual_review") is not True:
        raise ValueError("review does not attest that policy was frozen before visual review")
    if attestation.get("original_temporal_order_reviewed") is not True:
        raise ValueError("review does not attest original temporal order")
    if attestation.get("hard_cut_detected") is not False:
        raise ValueError("hard-cut or montage source cannot be a prospective positive")

    review_body = review.get("review")
    if not isinstance(review_body, dict):
        raise ValueError("prospective review body is missing")
    source_id = review_body.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("prospective review source_id is missing")
    reference = review_body.get("candidate_boundary")
    risk_window = review_body.get("risk_present_window_ms")
    clear_window = review_body.get("stable_clear_window_ms")
    if not isinstance(reference, dict):
        raise ValueError("prospective review candidate boundary is missing")
    if not isinstance(risk_window, list) or not isinstance(clear_window, list):
        raise ValueError("prospective review diagnostic windows are missing")

    feature_source = one_source(features.get("sources"), source_id, "feature report")
    inventory_source = one_source(
        inventory_audit.get("sources"), source_id, "source inventory audit"
    )
    video_sha = str(feature_source.get("video_sha256", ""))
    if not video_sha or video_sha != inventory_source.get("video_sha256"):
        raise ValueError("feature and source inventory video hashes differ")
    evidence = review.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("video_sha256") != video_sha:
        raise ValueError("review video hash differs from frozen feature source")
    samples = feature_source.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("prospective feature source has no samples")

    filtered_samples = chromatic.apply_policy(samples, policy)
    lifecycle_contract = contract["lifecycle"]
    lifecycle_result = lifecycle.tristate_exit_intervals(
        filtered_samples,
        lifecycle_contract["selected_groups"],
        entry_window_samples=int(lifecycle_contract["entry_window_samples"]),
        entry_min_active_samples=int(lifecycle_contract["entry_min_active_samples"]),
        clear_absent_samples=int(lifecycle_contract["clear_absent_samples"]),
    )
    score = lifecycle.score_intervals(lifecycle_result["intervals"], reference)
    risk_diagnostics = lifecycle.activity_window_diagnostics(
        filtered_samples, lifecycle_contract["selected_groups"], risk_window
    )
    clear_diagnostics = lifecycle.activity_window_diagnostics(
        filtered_samples, lifecycle_contract["selected_groups"], clear_window
    )
    risk_fraction = risk_diagnostics["active_fraction"]
    clear_fraction = clear_diagnostics["active_fraction"]
    if risk_fraction is None or clear_fraction is None:
        raise ValueError("prospective diagnostic window has no frozen samples")
    acceptance = contract["acceptance"]
    lineage_passed = inventory_source.get("eligible_prospective_positive_exit") is True
    review_passed = review_body.get("decision") == ACCEPT_DECISION
    risk_passed = risk_fraction >= float(
        acceptance["minimum_risk_present_active_fraction"]
    )
    clear_passed = clear_fraction <= float(
        acceptance["maximum_stable_clear_active_fraction"]
    )
    terminal_clear_passed = (
        lifecycle_result["terminal_state"] == "clear"
        and lifecycle_result["open_event"] is None
    )
    checks = {
        "source_lineage_gate_passed": lineage_passed,
        "large_model_review_acceptance_passed": review_passed,
        "exact_single_reference_containing_exit_passed": score["passed"],
        "risk_present_coverage_passed": risk_passed,
        "stable_clear_false_activation_passed": clear_passed,
        "terminal_clear_without_open_event_passed": terminal_clear_passed,
    }
    return {
        "source_id": source_id,
        "video_sha256": video_sha,
        "viewpoint": inventory_source.get("viewpoint"),
        "eligible_prospective_pedestrian_exit": inventory_source.get(
            "eligible_prospective_pedestrian_exit"
        ) is True,
        "risk_evidence_policy": policy,
        "reference": {
            "kind": "large-model timestamped multiframe review; not human truth",
            "present_timestamp_ms": int(reference["present_timestamp_ms"]),
            "absent_timestamp_ms": int(reference["absent_timestamp_ms"]),
        },
        "lifecycle": lifecycle_result,
        "diagnostics": {
            "risk_present_window": risk_diagnostics,
            "stable_clear_window": clear_diagnostics,
        },
        "score": score,
        "acceptance": {
            **checks,
            "minimum_risk_present_active_fraction": acceptance[
                "minimum_risk_present_active_fraction"
            ],
            "maximum_stable_clear_active_fraction": acceptance[
                "maximum_stable_clear_active_fraction"
            ],
            "passed": all(checks.values()),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    features = lifecycle.verify_json_sidecar(args.features)
    review = lifecycle.verify_json_sidecar(args.review)
    inventory_audit = lifecycle.verify_json_sidecar(args.inventory_audit)
    contract, contract_attestation = prospective.load_contract(args.contract)
    result = evaluate(
        features=features,
        review=review,
        inventory_audit=inventory_audit,
        contract=contract,
        contract_attestation=contract_attestation,
        feature_sha256=common.sha256_file(args.features),
        inventory_audit_sha256=common.sha256_file(args.inventory_audit),
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "features": {
                "path": str(args.features.resolve()),
                "sha256": common.sha256_file(args.features),
            },
            "review": {
                "path": str(args.review.resolve()),
                "sha256": common.sha256_file(args.review),
            },
            "inventory_audit": {
                "path": str(args.inventory_audit.resolve()),
                "sha256": common.sha256_file(args.inventory_audit),
            },
            "contract": contract_attestation,
        },
        **result,
        "evidence_limit": "Large-model prospective discovery evidence only; not human truth, training truth, calibration, blind evaluation, Android runtime authorization, or production evidence.",
        "human_event_truth_present": False,
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(str(args.output) + ".sha256").write_text(
        common.sha256_file(args.output) + "\n", encoding="ascii"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--inventory-audit", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "source_id": report["source_id"],
        "passed": report["acceptance"]["passed"],
        "pedestrian": report["eligible_prospective_pedestrian_exit"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
