#!/usr/bin/env python3
"""Evaluate radial context notices plus past-only committed-route escalation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_temporal_risk_profile_prospective as prospective
import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_chromatic_marker_policy as chromatic
import public_video_tristate_contract as tristate_contract
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_causal_past_ego_trace_probe as causal


SCHEMA = "blindassist_public_video_tiered_causal_alerts_v1"


def first_upgrade(frames: list[dict[str, Any]], threshold: float, consecutive: int) -> int | None:
    run = 0
    previous: int | None = None
    for row in frames:
        timestamp = int(row["timestamp_ms"])
        score = row.get("trace_intrusion_score")
        if previous is None or timestamp - previous != 1000:
            run = 0
        if score is not None and float(score) >= threshold:
            run += 1
            if run >= consecutive:
                return timestamp
        else:
            run = 0
        previous = timestamp
    return None


def select_event(candidates: dict[str, Any], source_id: str, entry: int, end: int) -> dict[str, Any]:
    source_rows = [row for row in candidates["sources"] if row["source_id"] == source_id]
    if len(source_rows) != 1:
        raise ValueError("candidate source missing")
    events = [row for row in source_rows[0]["events"]
              if int(row["event_entry_timestamp_ms"]) == entry and int(row["last_active_timestamp_ms"]) == end]
    if len(events) != 1:
        raise ValueError("review does not bind exactly one candidate")
    return events[0]


def confirmed_clear_timestamp(intervals: list[dict[str, Any]]) -> int | None:
    if len(intervals) != 1:
        return None
    value = intervals[0].get("confirmed_clear_timestamp_ms")
    return int(value) if value is not None else None


def evaluate_bound_event(
    features: dict[str, Any], candidates: dict[str, Any], review: dict[str, Any],
    causal_policy: dict[str, Any], chromatic_policy: dict[str, Any], alert_policy: dict[str, Any]
) -> dict[str, Any]:
    role = str(review["role"])
    section = review[role]
    source_id = str(review["source"]["source_id"])
    entry = int(section["frozen_event_entry_timestamp_ms"])
    end = int(section["frozen_event_last_active_timestamp_ms"])
    event = select_event(candidates, source_id, entry, end)
    trace = causal.evaluate_event(features, source_id, (entry, end + 1000), causal_policy)
    tier2 = alert_policy["tier_2"]
    upgrade = first_upgrade(trace["frames"], float(tier2["frame_trace_intrusion_at_least"]),
                            int(tier2["consecutive_one_second_samples"]))
    source = next(row for row in features["sources"] if row["source_id"] == source_id)
    state = prospective.replay_selected_event_lifecycle(source["samples"], chromatic_policy, event)
    intervals = [row for row in state["intervals"] if int(row["event_entry_timestamp_ms"]) == entry]
    clear_timestamp = confirmed_clear_timestamp(intervals)
    latest = section.get("latest_acceptable_open_timestamp_ms")
    return {
        "source_id": source_id, "role": role, "event_entry_timestamp_ms": entry,
        "event_last_active_timestamp_ms": end,
        "tier_1_context_notice_timestamp_ms": entry,
        "tier_2_route_blocking_upgrade_timestamp_ms": upgrade,
        "latest_useful_reminder_timestamp_ms": latest,
        "tier_1_is_early_enough": latest is None or entry <= int(latest),
        "uncertain_route_without_directional_claim": upgrade is None,
        "lifecycle_clear_timestamp_ms": clear_timestamp,
        "lifecycle_interval_count": len(intervals),
        "causal_trace": trace,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = (args.contract, args.r756_contract, args.r779_report, args.chromatic_contract,
             args.bangkok_features, args.bangkok_candidates, args.bangkok_safe_review,
             args.bangkok_positive_review, args.duesseldorf_features, args.duesseldorf_candidates,
             args.duesseldorf_safe_review, args.output)
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    checks = ((args.r756_contract, "r756_contract_sha256"), (args.r779_report, "r779_report_sha256"),
              (args.chromatic_contract, "chromatic_contract_sha256"),
              (args.bangkok_features, "bangkok_features_sha256"),
              (args.bangkok_candidates, "bangkok_candidates_sha256"),
              (args.bangkok_safe_review, "bangkok_safe_review_sha256"),
              (args.bangkok_positive_review, "bangkok_positive_review_sha256"),
              (args.duesseldorf_features, "duesseldorf_features_sha256"),
              (args.duesseldorf_candidates, "duesseldorf_candidates_sha256"),
              (args.duesseldorf_safe_review, "duesseldorf_safe_review_sha256"))
    for path, key in checks:
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    r756 = common.load_json(args.r756_contract)
    causal_policy = r756["causal_teacher"]
    _, chromatic_meta = tristate_contract.load_contract(args.chromatic_contract)
    if chromatic_meta["sha256"] != bound["chromatic_contract_sha256"]:
        raise ValueError("chromatic contract mismatch")
    chromatic_policy = chromatic.validate_policy(common.load_json(args.chromatic_contract))
    bangkok_features = lifecycle.verify_json_sidecar(args.bangkok_features)
    bangkok_candidates = lifecycle.verify_json_sidecar(args.bangkok_candidates)
    duesseldorf_features = lifecycle.verify_json_sidecar(args.duesseldorf_features)
    duesseldorf_candidates = lifecycle.verify_json_sidecar(args.duesseldorf_candidates)
    rows = [
        evaluate_bound_event(bangkok_features, bangkok_candidates,
                             lifecycle.verify_json_sidecar(args.bangkok_safe_review), causal_policy,
                             chromatic_policy, contract["alert_policy"]),
        evaluate_bound_event(bangkok_features, bangkok_candidates,
                             lifecycle.verify_json_sidecar(args.bangkok_positive_review), causal_policy,
                             chromatic_policy, contract["alert_policy"]),
        evaluate_bound_event(duesseldorf_features, duesseldorf_candidates,
                             lifecycle.verify_json_sidecar(args.duesseldorf_safe_review), causal_policy,
                             chromatic_policy, contract["alert_policy"]),
    ]
    safe = [row for row in rows if row["role"] == "true_radial_safe_lateral_negative"]
    positive = [row for row in rows if row["role"] == "prospective_positive_event"]
    gate_checks = {
        "positive_context_notice_by_latest_useful_reminder": all(row["tier_1_is_early_enough"] for row in positive),
        "all_safe_lateral_events_have_no_tier_2_upgrade": all(row["tier_2_route_blocking_upgrade_timestamp_ms"] is None for row in safe),
        "all_events_clear_under_frozen_lifecycle": all(row["lifecycle_interval_count"] == 1 and row["lifecycle_clear_timestamp_ms"] is not None for row in rows),
    }
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract)},
        "policy": contract["alert_policy"], "events": rows,
        "tier_2_positive_coverage": sum(row["tier_2_route_blocking_upgrade_timestamp_ms"] is not None for row in positive) / max(1, len(positive)),
        "checks": gate_checks, "tiered_context_safety_gate_passed": all(gate_checks.values()),
        "evidence_limit": "Offline tiered policy prototype on provisional silver; no runtime, training, calibration, blind, or production authority.",
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--r756-contract", type=Path, required=True)
    parser.add_argument("--r779-report", type=Path, required=True)
    parser.add_argument("--chromatic-contract", type=Path, required=True)
    parser.add_argument("--bangkok-features", type=Path, required=True)
    parser.add_argument("--bangkok-candidates", type=Path, required=True)
    parser.add_argument("--bangkok-safe-review", type=Path, required=True)
    parser.add_argument("--bangkok-positive-review", type=Path, required=True)
    parser.add_argument("--duesseldorf-features", type=Path, required=True)
    parser.add_argument("--duesseldorf-candidates", type=Path, required=True)
    parser.add_argument("--duesseldorf-safe-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({"ok": True, "passed": value["tiered_context_safety_gate_passed"],
                      "tier_2_positive_coverage": value["tier_2_positive_coverage"],
                      "output_sha256": common.sha256_file(parsed.output)}))
