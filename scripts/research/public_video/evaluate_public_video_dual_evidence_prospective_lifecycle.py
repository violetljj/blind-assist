"""Evaluate one new source against the frozen r7.17 lifecycle contract.

The full-video feature report must be frozen before original-order visual
review. The reviewer selects only three timestamp windows and a visual boundary;
all channel scores, transition decisions and fail-closed controls are computed
from the immutable feature report.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle_io
import public_video_dual_evidence_lifecycle_contract as prospective
import run_public_silver_dual_evidence_lifecycle_fusion as fusion
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_dual_evidence_prospective_lifecycle_v1"
FEATURE_SCHEMA = "blindassist_public_video_dual_evidence_feature_report_v1"
REVIEW_SCHEMA = "blindassist_public_video_dual_evidence_prospective_review_v1"
INVENTORY_SCHEMA = "blindassist_public_video_prospective_source_inventory_audit_v1"
ACCEPT_DECISION = "accept_as_prospective_dual_evidence_lifecycle_challenge"
MECHANISMS = {"dynamic_agent_approach", "static_corridor_narrowing"}


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value.lower()
    ):
        raise ValueError(f"{label} SHA256 is invalid")
    return value.lower()


def one_source(rows: Any, source_id: str, label: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} sources must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("source_id") == source_id]
    if len(matches) != 1:
        raise ValueError(f"{label} must bind exactly one source: {source_id}")
    return matches[0]


def _window(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list) or len(value) != 2
        or not all(isinstance(item, int) for item in value)
        or value[0] > value[1]
    ):
        raise ValueError(f"{label} must be an ordered integer [start,end] window")
    return int(value[0]), int(value[1])


def _validate_samples(source: dict[str, Any], sample_interval_ms: int) -> list[dict[str, Any]]:
    samples = source.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("feature source contains no samples")
    ordered = sorted(samples, key=lambda row: int(row.get("timestamp_ms", -1)))
    timestamps: list[int] = []
    for sample in ordered:
        timestamp = sample.get("timestamp_ms")
        if not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError("feature sample timestamp must be a non-negative integer")
        dynamic = sample.get("dynamic_occupancy")
        residual = sample.get("static_residual")
        if not isinstance(dynamic, (int, float)) or not math.isfinite(float(dynamic)) or float(dynamic) < 0:
            raise ValueError("dynamic occupancy must be finite and non-negative")
        if residual is not None and (
            not isinstance(residual, (int, float)) or not math.isfinite(float(residual)) or float(residual) < 0
        ):
            raise ValueError("static residual must be null or finite and non-negative")
        if not isinstance(sample.get("static_residual_reliable"), bool):
            raise ValueError("static residual reliability must be boolean")
        if sample.get("static_residual_reliable") and residual is None:
            raise ValueError("reliable static residual cannot be null")
        count = sample.get("semantic_risk_count")
        if not isinstance(count, int) or count < 0:
            raise ValueError("semantic risk count must be a non-negative integer")
        timestamps.append(timestamp)
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("feature sample timestamps must be unique")
    if any(current - previous != sample_interval_ms for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("full feature report has missing or irregular samples")
    return ordered


def _select(samples: Sequence[dict[str, Any]], window: tuple[int, int]) -> list[dict[str, Any]]:
    return [row for row in samples if window[0] <= int(row["timestamp_ms"]) <= window[1]]


def _score(samples: Sequence[dict[str, Any]], mechanism: str, minimum_reliable: int) -> float:
    if mechanism == "dynamic_agent_approach":
        values = [float(row["dynamic_occupancy"]) for row in samples]
        if len(values) < 2:
            raise ValueError("dynamic window needs at least two samples")
    else:
        values = [
            float(row["static_residual"])
            for row in samples
            if row["static_residual_reliable"] and row["static_residual"] is not None
        ]
        if len(values) < minimum_reliable:
            raise ValueError("static window lacks reliable registered transitions")
    return max(values) - min(values)


def _normalized_change(previous: float, current: float) -> float:
    return (current - previous) / max(abs(previous), abs(current), 1e-12)


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
    prospective.validate_contract(contract)
    if features.get("schema") != FEATURE_SCHEMA:
        raise ValueError("unexpected dual-evidence feature schema")
    if features.get("prospective_contract") != contract_attestation:
        raise ValueError("feature report is not bound to the supplied frozen contract")
    generation = features.get("feature_generation")
    feature_contract = contract["feature_contract"]
    if not isinstance(generation, dict):
        raise ValueError("feature generation attestation is missing")
    if generation.get("complete_video_processed") is not True:
        raise ValueError("feature report does not cover the complete video")
    if generation.get("review_windows_known_during_feature_generation") is not False:
        raise ValueError("feature report was generated after review windows were known")
    if generation.get("feature_values_immutable") is not True:
        raise ValueError("feature values are not attested immutable")
    if generation.get("sample_interval_ms") != feature_contract["sample_interval_ms"]:
        raise ValueError("feature sample interval differs from the frozen contract")
    if generation.get("dynamic_weights_sha256") != feature_contract["dynamic_channel"]["weights_sha256"]:
        raise ValueError("dynamic feature weights differ from the frozen contract")
    if generation.get("semantic_weights_sha256") != feature_contract["semantic_exit_channel"]["weights_sha256"]:
        raise ValueError("semantic feature weights differ from the frozen contract")

    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("unexpected dual-evidence prospective review schema")
    attestation = review.get("prospective_attestation")
    if not isinstance(attestation, dict):
        raise ValueError("prospective review attestation is missing")
    if attestation.get("frozen_contract_sha256") != contract_attestation["sha256"]:
        raise ValueError("review frozen contract hash mismatch")
    _require_sha256(feature_sha256, "feature report")
    _require_sha256(inventory_audit_sha256, "source inventory audit")
    if attestation.get("full_feature_report_sha256") != feature_sha256:
        raise ValueError("review does not attest the supplied full feature report")
    if attestation.get("source_inventory_audit_sha256") != inventory_audit_sha256:
        raise ValueError("review does not attest the supplied source inventory audit")
    required_true = (
        "contract_frozen_before_source_visual_review",
        "full_feature_report_frozen_before_visual_review",
        "reviewer_did_not_edit_feature_values",
        "original_temporal_order_reviewed",
    )
    if any(attestation.get(key) is not True for key in required_true):
        raise ValueError("review chronology or feature immutability attestation failed")
    if attestation.get("hard_cut_detected") is not False:
        raise ValueError("hard-cut or montage source cannot enter prospective evaluation")
    for key in prospective.REQUIRED_FALSE_AUTHORIZATIONS:
        if review.get(key) is not False:
            raise ValueError("review contains an unauthorized promotion flag")

    if inventory_audit.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("unexpected source inventory audit schema")
    review_body = review.get("review")
    if not isinstance(review_body, dict) or review_body.get("decision") != ACCEPT_DECISION:
        raise ValueError("prospective review decision is missing or not accepted")
    source_id = review_body.get("source_id")
    mechanism = review_body.get("mechanism")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("prospective review source_id is missing")
    if mechanism not in MECHANISMS:
        raise ValueError("prospective review mechanism is invalid")
    feature_source = one_source(features.get("sources"), source_id, "feature report")
    inventory_source = one_source(inventory_audit.get("sources"), source_id, "source inventory audit")
    video_sha = _require_sha256(feature_source.get("video_sha256"), "feature video")
    if video_sha != inventory_source.get("video_sha256"):
        raise ValueError("feature and inventory video hashes differ")
    evidence = review.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("video_sha256") != video_sha:
        raise ValueError("review video hash differs from the frozen feature source")
    _require_sha256(evidence.get("contact_sheet_manifest_sha256"), "contact sheet manifest")

    samples = _validate_samples(feature_source, int(feature_contract["sample_interval_ms"]))
    pre_window = _window(review_body.get("pre_risk_clear_window_ms"), "pre-risk clear")
    risk_window = _window(review_body.get("risk_present_window_ms"), "risk present")
    post_window = _window(review_body.get("stable_post_clear_window_ms"), "stable post-clear")
    if not (pre_window[1] < risk_window[0] and risk_window[1] < post_window[0]):
        raise ValueError("review windows must be non-overlapping and chronological")
    pre_samples = _select(samples, pre_window)
    risk_samples = _select(samples, risk_window)
    post_samples = _select(samples, post_window)
    protocol = contract["review_protocol"]
    if len(pre_samples) < protocol["minimum_pre_risk_clear_samples"]:
        raise ValueError("pre-risk clear window has too few frozen samples")
    if len(risk_samples) < protocol["minimum_risk_present_samples"]:
        raise ValueError("risk-present window has too few frozen samples")
    if len(post_samples) < protocol["minimum_stable_post_clear_samples"]:
        raise ValueError("stable post-clear window has too few frozen samples")
    confirm_count = int(protocol["post_clear_confirmation_samples"])
    immediate_clear = post_samples[:confirm_count]
    stable_clear = post_samples[-confirm_count:]
    minimum_reliable = int(feature_contract["static_channel"]["minimum_reliable_transitions_per_window"])
    pre_score = _score(pre_samples, str(mechanism), minimum_reliable)
    risk_score = _score(risk_samples, str(mechanism), minimum_reliable)
    immediate_clear_score = _score(immediate_clear, str(mechanism), minimum_reliable)
    stable_clear_score = _score(stable_clear, str(mechanism), minimum_reliable)

    semantic_contract = feature_contract["semantic_exit_channel"]
    gap_ms = int(immediate_clear[0]["timestamp_ms"]) - int(risk_samples[-1]["timestamp_ms"])
    semantic_exit = (
        any(int(row["semantic_risk_count"]) > 0 for row in risk_samples)
        and all(int(row["semantic_risk_count"]) == 0 for row in immediate_clear)
        and max(float(row["dynamic_occupancy"]) for row in immediate_clear)
        < max(float(row["dynamic_occupancy"]) for row in risk_samples)
        and 0 < gap_ms <= int(semantic_contract["maximum_risk_to_clear_gap_ms"])
    )
    open_change = _normalized_change(pre_score, risk_score)
    close_change = _normalized_change(risk_score, immediate_clear_score)
    stability_change = _normalized_change(immediate_clear_score, stable_clear_score)
    open_decision = fusion.decide_transition(
        previous_state="clear", normalized_signed_change=open_change,
        semantic_exit=False, trusted_reference=True,
    )
    close_decision = fusion.decide_transition(
        previous_state="risk", normalized_signed_change=close_change,
        semantic_exit=semantic_exit, trusted_reference=True,
    )
    stability_decision = fusion.decide_transition(
        previous_state="clear", normalized_signed_change=stability_change,
        semantic_exit=False, trusted_reference=True,
    )

    boundary = review_body.get("visual_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("visual lifecycle boundary is missing")
    present_ms = boundary.get("risk_present_timestamp_ms")
    clear_ms = boundary.get("clear_timestamp_ms")
    if not isinstance(present_ms, int) or not isinstance(clear_ms, int):
        raise ValueError("visual boundary timestamps must be integers")
    boundary_contained = (
        risk_window[0] <= present_ms <= risk_window[1]
        and post_window[0] <= clear_ms <= post_window[1]
        and present_ms < clear_ms
    )
    controls = {
        "cold_start_uncertain": fusion.decide_transition(
            previous_state="risk", normalized_signed_change=-0.8,
            semantic_exit=False, trusted_reference=False,
        )["predicted_transition"] == "uncertain",
        "semantic_absence_only_uncertain": fusion.decide_transition(
            previous_state="risk", normalized_signed_change=0.0,
            semantic_exit=True, trusted_reference=True,
        )["predicted_transition"] == "uncertain",
        "conflicting_rise_and_exit_uncertain": fusion.decide_transition(
            previous_state="risk", normalized_signed_change=0.2,
            semantic_exit=True, trusted_reference=True,
        )["predicted_transition"] == "uncertain",
    }
    checks = {
        "source_lineage_gate_passed": inventory_source.get("eligible_prospective_positive_exit") is True,
        "large_model_review_acceptance_passed": True,
        "visual_boundary_contained": boundary_contained,
        "strong_open_passed": open_decision["predicted_transition"] == "open_event",
        "close_passed": close_decision["predicted_transition"] == "close_event",
        "stable_post_clear_no_reopen_passed": stability_decision["predicted_transition"] != "open_event",
        "cold_start_control_passed": controls["cold_start_uncertain"],
        "semantic_absence_only_control_passed": controls["semantic_absence_only_uncertain"],
        "conflicting_rise_and_exit_control_passed": controls["conflicting_rise_and_exit_uncertain"],
    }
    return {
        "source_id": source_id,
        "video_sha256": video_sha,
        "viewpoint": inventory_source.get("viewpoint"),
        "eligible_prospective_pedestrian_exit": inventory_source.get("eligible_prospective_pedestrian_exit") is True,
        "mechanism": mechanism,
        "windows": {
            "pre_risk_clear": {"range_ms": list(pre_window), "sample_count": len(pre_samples), "score": pre_score},
            "risk_present": {"range_ms": list(risk_window), "sample_count": len(risk_samples), "score": risk_score},
            "immediate_post_clear": {"sample_count": len(immediate_clear), "score": immediate_clear_score},
            "stable_post_clear": {"range_ms": list(post_window), "sample_count": len(post_samples), "score": stable_clear_score},
        },
        "semantic_exit": {
            "passed": semantic_exit,
            "risk_to_clear_gap_ms": gap_ms,
            "risk_dynamic_occupancy_peak": max(
                float(row["dynamic_occupancy"]) for row in risk_samples
            ),
            "post_clear_dynamic_occupancy_peak": max(
                float(row["dynamic_occupancy"]) for row in immediate_clear
            ),
        },
        "transitions": {
            "open": {"normalized_signed_change": open_change, **open_decision},
            "close": {"normalized_signed_change": close_change, **close_decision},
            "post_clear_stability": {"normalized_signed_change": stability_change, **stability_decision},
        },
        "controls": controls,
        "acceptance": {**checks, "passed": all(checks.values())},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    features = lifecycle_io.verify_json_sidecar(args.features)
    review = lifecycle_io.verify_json_sidecar(args.review)
    inventory_audit = lifecycle_io.verify_json_sidecar(args.inventory_audit)
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
            "features": {"path": str(args.features.resolve()), "sha256": common.sha256_file(args.features)},
            "review": {"path": str(args.review.resolve()), "sha256": common.sha256_file(args.review)},
            "inventory_audit": {"path": str(args.inventory_audit.resolve()), "sha256": common.sha256_file(args.inventory_audit)},
            "contract": contract_attestation,
        },
        **result,
        "evidence_limit": "Large-model prospective lifecycle challenge only; not human truth, training truth, calibration, blind evaluation, Android runtime authorization, or production evidence.",
        "human_event_truth_present": False,
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output or sidecar: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
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
