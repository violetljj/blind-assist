#!/usr/bin/env python3
"""Score a frozen r7.25 radial entry against a post-freeze positive lifecycle review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_chromatic_marker_policy as chromatic
import public_video_marker_radial_approach_contract as approach_contract
import public_video_tristate_contract as tristate_contract
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_marker_radial_lifecycle_positive_result_v1"
REVIEW_SCHEMA = "blindassist_public_video_marker_radial_lifecycle_review_v1"


def interval_overlaps(event: dict[str, Any], window: Sequence[int]) -> bool:
    start = int(event["event_entry_timestamp_ms"])
    end = int(event.get("confirmed_clear_timestamp_ms", event["last_active_timestamp_ms"]))
    return start <= int(window[1]) and end >= int(window[0])


def reconstruct_base_lifecycle(
    source: dict[str, Any], base_contract: dict[str, Any]
) -> dict[str, Any]:
    policy = chromatic.validate_policy(base_contract)
    configured = base_contract["lifecycle"]
    samples = chromatic.apply_policy(source["samples"], policy)
    return lifecycle.tristate_exit_intervals(
        samples,
        configured["selected_groups"],
        entry_window_samples=int(configured["entry_window_samples"]),
        entry_min_active_samples=int(configured["entry_min_active_samples"]),
        clear_absent_samples=int(configured["clear_absent_samples"]),
    )


def matching_base_interval(
    candidate: dict[str, Any], intervals: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    matches = [
        event for event in intervals
        if int(event["event_entry_timestamp_ms"]) == int(candidate["event_entry_timestamp_ms"])
        and int(event["last_active_timestamp_ms"]) == int(candidate["last_active_timestamp_ms"])
    ]
    if len(matches) != 1:
        raise ValueError("radial candidate does not bind exactly one base lifecycle interval")
    return matches[0]


def score_positive_source(
    candidate_events: Sequence[dict[str, Any]],
    base_intervals: Sequence[dict[str, Any]],
    review_item: dict[str, Any],
) -> dict[str, Any]:
    risk_window = review_item["visual_risk_present_window_ms"]
    stable_clear_window = review_item["stable_route_clear_window_ms"]
    latest_open_ms = int(review_item["latest_acceptable_open_timestamp_ms"])
    relevant = [
        event for event in candidate_events
        if int(risk_window[0]) <= int(event["event_entry_timestamp_ms"]) <= latest_open_ms
    ]
    if len(relevant) != 1:
        return {
            "prospective_positive_passed": False,
            "failure_reason": "expected exactly one frozen radial candidate overlapping the reviewed risk window",
            "overlapping_candidate_event_count": len(relevant),
        }

    candidate = relevant[0]
    base = matching_base_interval(candidate, base_intervals)
    entry_ms = int(candidate["event_entry_timestamp_ms"])
    clear_ms = int(base["confirmed_clear_timestamp_ms"])
    risk_end_ms = int(risk_window[1])
    stable_clear_end_ms = int(stable_clear_window[1])
    later_candidates = [
        event for event in candidate_events
        if int(event["event_entry_timestamp_ms"]) > entry_ms
        and int(event["event_entry_timestamp_ms"]) <= stable_clear_end_ms
    ]
    checks = {
        "radial_entry_opens_in_time": int(risk_window[0]) <= entry_ms <= latest_open_ms,
        "event_not_cleared_before_visual_risk_ends": clear_ms >= risk_end_ms,
        "event_clear_by_stable_route_clear_window_end": clear_ms <= stable_clear_end_ms,
        "same_event_not_reopened": not later_candidates,
    }
    return {
        "candidate_event": candidate,
        "bound_base_interval": base,
        "predicted_open_timestamp_ms": entry_ms,
        "predicted_clear_timestamp_ms": clear_ms,
        "visual_risk_present_window_ms": list(map(int, risk_window)),
        "stable_route_clear_window_ms": list(map(int, stable_clear_window)),
        "false_clear_gap_ms": max(0, risk_end_ms - clear_ms),
        "later_candidate_event_count": len(later_candidates),
        "checks": checks,
        "prospective_positive_passed": all(checks.values()),
    }


def verify_review_manifest(path: Path, expected_sha256: str) -> None:
    lifecycle.verify_json_sidecar(path)
    if common.sha256_file(path) != expected_sha256:
        raise ValueError(f"review manifest hash mismatch: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract, contract_meta = approach_contract.load_contract(args.contract)
    base_contract, base_meta = tristate_contract.load_contract(args.base_contract)
    features = lifecycle.verify_json_sidecar(args.features)
    candidates = lifecycle.verify_json_sidecar(args.candidates)
    review = lifecycle.verify_json_sidecar(args.review)
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("unexpected lifecycle review schema")
    feature_sha = common.sha256_file(args.features)
    candidate_sha = common.sha256_file(args.candidates)
    if base_meta["sha256"] != contract["bound_inputs"]["chromatic_marker_contract_sha256"]:
        raise ValueError("base chromatic contract hash mismatch")
    if features.get("prospective_contract", {}).get("sha256") != base_meta["sha256"]:
        raise ValueError("feature report base contract mismatch")
    if candidates.get("approach_contract", {}).get("sha256") != contract_meta["sha256"]:
        raise ValueError("candidate approach contract mismatch")
    if candidates.get("feature_report_sha256") != feature_sha:
        raise ValueError("candidate feature report mismatch")
    bindings = review.get("frozen_inputs", {})
    expected_bindings = {
        "approach_contract_sha256": contract_meta["sha256"],
        "base_contract_sha256": base_meta["sha256"],
        "feature_report_sha256": feature_sha,
        "candidate_report_sha256": candidate_sha,
    }
    for key, expected in expected_bindings.items():
        if bindings.get(key) != expected:
            raise ValueError(f"review binding mismatch: {key}")
    if review.get("reviewed_after_features_and_candidates_frozen") is not True:
        raise ValueError("review chronology is invalid")
    if review.get("reviewer_type") != "large_model_visual_review_not_human_truth":
        raise ValueError("review evidence role is not explicit")
    manifest_args = {
        "overview_manifest_sha256": args.overview_manifest,
        "primary_review_manifest_sha256": args.primary_review_manifest,
        "post_review_manifest_sha256": args.post_review_manifest,
    }
    for key, path in manifest_args.items():
        verify_review_manifest(path, bindings[key])

    feature_by_source = {row["source_id"]: row for row in features.get("sources", [])}
    candidate_by_source = {row["source_id"]: row.get("events", []) for row in candidates.get("sources", [])}
    rows: list[dict[str, Any]] = []
    scored = 0
    passed = 0
    context_only = 0
    for item in review["sources"]:
        source_id = item["source_id"]
        if source_id not in feature_by_source or source_id not in candidate_by_source:
            raise ValueError(f"review source is not frozen in both inputs: {source_id}")
        if feature_by_source[source_id].get("video_sha256") != item.get("video_sha256"):
            raise ValueError(f"video lineage mismatch: {source_id}")
        role = item["evaluation_role"]
        if role == "context_only":
            context_only += 1
            rows.append({
                "source_id": source_id,
                "evaluation_role": role,
                "gate_credit": False,
                "candidate_event_count": len(candidate_by_source[source_id]),
                "reason": item["reason"],
            })
            continue
        if role != "prospective_positive_lifecycle":
            raise ValueError(f"unsupported evaluation role: {role}")
        if item.get("continuity") != {
            "continuous_ego_pedestrian_capture": True,
            "original_temporal_order": True,
            "hard_cut_or_montage_observed": False,
        }:
            raise ValueError(f"positive continuity gate failed: {source_id}")
        finding = item.get("visual_finding", {})
        if finding.get("pedestrian_corridor_risk_present") is not True:
            raise ValueError("positive review does not claim corridor risk")
        if finding.get("same_risk_episode_across_detector_gap") is not True:
            raise ValueError("positive review does not claim a continuous risk episode")
        base = reconstruct_base_lifecycle(feature_by_source[source_id], base_contract)
        score = score_positive_source(candidate_by_source[source_id], base["intervals"], item)
        scored += 1
        passed += int(score["prospective_positive_passed"])
        rows.append({
            "source_id": source_id,
            "evaluation_role": role,
            "gate_credit": True,
            "visual_finding": finding,
            "base_lifecycle_interval_count": len(base["intervals"]),
            **score,
        })

    gate_passed = scored > 0 and scored == passed
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "approach_contract": contract_meta,
            "base_contract": base_meta,
            "feature_report_sha256": feature_sha,
            "candidate_report_sha256": candidate_sha,
            "review_sha256": common.sha256_file(args.review),
            **{key: common.sha256_file(path) for key, path in manifest_args.items()},
        },
        "sources": rows,
        "summary": {
            "scored_prospective_positive": scored,
            "passed_prospective_positive": passed,
            "context_only": context_only,
            "prospective_positive_lifecycle_gate_passed": gate_passed,
        },
        "diagnosis": (
            "The radial gate is an entry mechanism, not sufficient exit evidence. "
            "A false clear while the reviewed corridor episode remains active is a lifecycle fragmentation failure."
        ),
        "evidence_limit": (
            "Large-model multiframe review is provisional silver evidence, not human event truth. "
            "This report preserves a contract-after-freeze failure and does not authorize threshold repair on the same source."
        ),
        "authorizations": {
            "training": False,
            "calibration": False,
            "blind": False,
            "android_runtime_change": False,
            "production_model_replacement": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--overview-manifest", type=Path, required=True)
    parser.add_argument("--primary-review-manifest", type=Path, required=True)
    parser.add_argument("--post-review-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({"ok": True, **value["summary"], "output_sha256": common.sha256_file(parsed.output)}, ensure_ascii=False))
