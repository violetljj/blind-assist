#!/usr/bin/env python3
"""Evaluate one prospective r7.50 positive with frozen route and lifecycle rules."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_chromatic_marker_policy as chromatic
import public_video_event_timing_contract as timing_contract
import public_video_tristate_contract as tristate_contract
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_explicit_ego_route_relation_probe as route_relation
import run_public_video_obstacle_aware_route_width_probe as route_width
import run_public_video_radial_lifecycle_gap_bridge_probe as gap


SCHEMA = "blindassist_public_video_event_timing_positive_evaluation_v1"


def timing_checks(
    *,
    reminder_timestamp_ms: int,
    route_delta: float,
    event: dict[str, Any] | None,
    reminder_count: int,
    review: dict[str, Any],
    maximum_early_warning_lead_ms: int,
) -> dict[str, bool]:
    onset = int(review["material_risk_onset_ms"])
    latest = int(review["latest_useful_reminder_ms"])
    risk_end = int(review["reviewed_risk_end_ms"])
    stable_clear = list(map(int, review["stable_post_clear_window_ms"]))
    clear_ms = int(event["confirmed_clear_timestamp_ms"]) if event is not None else -1
    return {
        "route_relation_supports_entry": route_delta > 0.0,
        "reminder_inside_fixed_early_warning_band": onset - maximum_early_warning_lead_ms <= reminder_timestamp_ms <= latest,
        "event_interval_present": event is not None,
        "no_clear_before_reviewed_risk_end": event is not None and clear_ms >= risk_end,
        "clear_inside_stable_post_clear_window": event is not None and stable_clear[0] <= clear_ms <= stable_clear[1],
        "same_visual_episode_reminder_once": reminder_count == 1,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.contract, args.base_contract, args.features, args.candidates, args.review_plan, args.review, args.model_dir, args.output]
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract, contract_meta = timing_contract.load_contract(args.contract)
    base_contract, base_meta = tristate_contract.load_contract(args.base_contract)
    policy = chromatic.validate_policy(base_contract)
    features = lifecycle.verify_json_sidecar(args.features)
    candidates = lifecycle.verify_json_sidecar(args.candidates)
    review_plan = lifecycle.verify_json_sidecar(args.review_plan)
    review_report = lifecycle.verify_json_sidecar(args.review)
    if features.get("prospective_contract", {}).get("sha256") != base_meta["sha256"]:
        raise ValueError("feature report base contract mismatch")
    if candidates.get("feature_report_sha256") != common.sha256_file(args.features):
        raise ValueError("candidate feature report mismatch")
    if review_plan.get("full_feature_report_sha256") != common.sha256_file(args.features):
        raise ValueError("review plan feature report mismatch")
    if review_plan.get("frozen_radial_candidate_report_sha256") != common.sha256_file(args.candidates):
        raise ValueError("review plan candidate report mismatch")
    if review_report.get("contract_sha256") != contract_meta["sha256"]:
        raise ValueError("review report contract mismatch")
    if review_report.get("role") != "prospective_positive_event":
        raise ValueError("review report is not a prospective positive")
    positive_review = review_report.get("prospective_positive_event", {})
    if positive_review.get("applicable") is not True or positive_review.get("hard_cut_or_montage_present") is not False:
        raise ValueError("positive review is ineligible")
    source_id = review_report["source"]["source_id"]
    source_rows = [row for row in features.get("sources", []) if row.get("source_id") == source_id]
    candidate_rows = [row for row in candidates.get("sources", []) if row.get("source_id") == source_id]
    if len(source_rows) != 1 or len(candidate_rows) != 1:
        raise ValueError("positive source must bind one feature and candidate row")
    events = [event for event in candidate_rows[0].get("events", []) if event.get("radial_approach_passed") is True]
    if len(events) != 1:
        raise ValueError("positive source must have exactly one frozen radial event")
    frozen_event = events[0]
    plan_event = review_plan.get("candidate", {})
    for key in ("event_entry_timestamp_ms", "last_active_timestamp_ms", "accepted_sample_count"):
        if int(plan_event.get(key, -1)) != int(frozen_event[key]):
            raise ValueError(f"review plan candidate differs: {key}")
    entry_ms = int(frozen_event["event_entry_timestamp_ms"])
    last_active_ms = int(frozen_event["last_active_timestamp_ms"])
    review_window = review_plan["review_window_rule"]
    teacher = route_width.FrozenWalkableTeacher(args.model_dir)
    route = route_relation.real_pressure(
        teacher,
        features,
        sample_id=f"{source_id}_prospective_positive",
        source_id=source_id,
        label=1,
        clear_window=(int(review_window["review_start_ms"]), entry_ms),
        marker_window=(entry_ms, last_active_ms + 1000),
        batch_size=args.batch_size,
    )
    state = gap.radial_entry_lifecycle(
        source_rows[0]["samples"],
        policy,
        events,
        clear_absent_samples=9,
    )
    matching = [event for event in state["intervals"] if int(event["event_entry_timestamp_ms"]) == entry_ms]
    event = matching[0] if len(matching) == 1 else None
    checks = timing_checks(
        reminder_timestamp_ms=entry_ms,
        route_delta=float(route["marker_minus_clear_intrusion"]),
        event=event,
        reminder_count=len(state["reminder_timestamps_ms"]),
        review=positive_review,
        maximum_early_warning_lead_ms=int(contract["review_timing_fields"]["maximum_early_warning_lead_ms"]),
    )
    passed = all(checks.values())
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "prospective_positive_after_feature_and_candidate_freeze",
        "inputs": {
            "event_timing_contract_sha256": contract_meta["sha256"],
            "base_chromatic_contract_sha256": base_meta["sha256"],
            "feature_report_sha256": common.sha256_file(args.features),
            "candidate_report_sha256": common.sha256_file(args.candidates),
            "review_plan_sha256": common.sha256_file(args.review_plan),
            "provisional_review_sha256": common.sha256_file(args.review),
            "segformer_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin"),
        },
        "source_id": source_id,
        "source_video_sha256": source_rows[0]["video_sha256"],
        "frozen_radial_event": frozen_event,
        "explicit_route_relation": route,
        "lifecycle": state,
        "provisional_review": positive_review,
        "checks": checks,
        "prospective_positive_passed": passed,
        "full_r750_closure_passed": False,
        "remaining_hard_requirement": "An independent real true-radial safe-lateral negative must be vetoed by the frozen route relation.",
        "evidence_limit": "Large-model review is provisional silver, not human truth. A positive pass cannot substitute for the missing independent true-radial safe-lateral negative or authorize training/runtime changes.",
        "authorization": {
            "five_prototype_bootstrap_short_runs": False,
            "training_execution_authorized": False,
            "human_event_truth_present": False,
            "calibration_authorized": False,
            "blind_evaluation_authorized": False,
            "android_runtime_change_authorized": False,
            "production_model_replacement_authorized": False,
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
    parser.add_argument("--review-plan", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "prospective_positive_passed": value["prospective_positive_passed"],
        "full_r750_closure_passed": value["full_r750_closure_passed"],
        "output_sha256": common.sha256_file(parsed.output),
    }, ensure_ascii=False))
