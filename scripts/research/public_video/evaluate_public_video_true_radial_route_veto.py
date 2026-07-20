#!/usr/bin/env python3
"""Evaluate one prospective true-radial safe-lateral negative with frozen r7.47 route geometry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_event_timing_contract as timing_contract
import public_video_tristate_contract as tristate_contract
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_explicit_ego_route_relation_probe as route_relation
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_true_radial_route_veto_evaluation_v1"


def veto_checks(*, frozen_event: dict[str, Any], negative: dict[str, Any], route_delta: float) -> dict[str, bool]:
    return {
        "frozen_radial_entry_present": frozen_event.get("radial_approach_passed") is True,
        "visual_review_confirms_safe_lateral": negative.get("obstacle_remains_safely_lateral_to_ego_route") is True,
        "visual_review_expects_route_veto": negative.get("route_relation_should_veto_event_entry") is True,
        "no_hard_cut_or_montage": negative.get("hard_cut_or_montage_present") is False,
        "frozen_route_relation_vetoes_entry": route_delta <= 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.contract, args.base_contract, args.features, args.candidates, args.review_plan,
             args.review, args.model_dir, args.output]
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")

    contract, contract_meta = timing_contract.load_contract(args.contract)
    _, base_meta = tristate_contract.load_contract(args.base_contract)
    features = lifecycle.verify_json_sidecar(args.features)
    candidates = lifecycle.verify_json_sidecar(args.candidates)
    review_plan = lifecycle.verify_json_sidecar(args.review_plan)
    review_report = lifecycle.verify_json_sidecar(args.review)

    if features.get("prospective_contract", {}).get("sha256") != base_meta["sha256"]:
        raise ValueError("feature report base contract mismatch")
    if candidates.get("feature_report_sha256") != common.sha256_file(args.features):
        raise ValueError("candidate feature report mismatch")
    if review_plan.get("contract_sha256") != contract_meta["sha256"]:
        raise ValueError("review plan contract mismatch")
    if review_plan.get("full_feature_report_sha256") != common.sha256_file(args.features):
        raise ValueError("review plan feature report mismatch")
    if review_plan.get("frozen_radial_candidate_report_sha256") != common.sha256_file(args.candidates):
        raise ValueError("review plan candidate report mismatch")
    if review_report.get("contract_sha256") != contract_meta["sha256"]:
        raise ValueError("review report contract mismatch")
    if review_report.get("role") != "true_radial_safe_lateral_negative":
        raise ValueError("review report is not a true-radial safe-lateral negative")

    negative = review_report.get("true_radial_safe_lateral_negative", {})
    if negative.get("applicable") is not True:
        raise ValueError("negative review is ineligible")
    candidate_id = negative.get("candidate_id")
    plan_rows = [row for row in review_plan.get("candidates", []) if row.get("candidate_id") == candidate_id]
    if len(plan_rows) != 1:
        raise ValueError("negative review must bind exactly one planned candidate")
    plan_event = plan_rows[0]

    source_id = review_report["source"]["source_id"]
    source_rows = [row for row in features.get("sources", []) if row.get("source_id") == source_id]
    candidate_rows = [row for row in candidates.get("sources", []) if row.get("source_id") == source_id]
    if len(source_rows) != 1 or len(candidate_rows) != 1:
        raise ValueError("negative source must bind one feature and candidate row")
    events = [event for event in candidate_rows[0].get("events", [])
              if event.get("radial_approach_passed") is True
              and int(event["event_entry_timestamp_ms"]) == int(plan_event["event_entry_timestamp_ms"])
              and int(event["last_active_timestamp_ms"]) == int(plan_event["last_active_timestamp_ms"])]
    if len(events) != 1:
        raise ValueError("planned negative does not match exactly one frozen radial event")
    frozen_event = events[0]

    entry_ms = int(frozen_event["event_entry_timestamp_ms"])
    last_active_ms = int(frozen_event["last_active_timestamp_ms"])
    review_window = list(map(int, plan_event["review_window_ms"]))
    if list(map(int, negative.get("review_window_ms", []))) != review_window:
        raise ValueError("negative review window differs from frozen plan")

    teacher = route_width.FrozenWalkableTeacher(args.model_dir)
    route = route_relation.real_pressure(
        teacher,
        features,
        sample_id=f"{source_id}_{candidate_id}_prospective_negative",
        source_id=source_id,
        label=0,
        clear_window=(review_window[0], entry_ms),
        marker_window=(entry_ms, last_active_ms + 1000),
        batch_size=args.batch_size,
    )
    route_delta = float(route["marker_minus_clear_intrusion"])
    checks = veto_checks(frozen_event=frozen_event, negative=negative, route_delta=route_delta)
    passed = all(checks.values())
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "prospective_true_radial_safe_lateral_negative_after_feature_and_candidate_freeze",
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
        "candidate_id": candidate_id,
        "frozen_radial_event": frozen_event,
        "explicit_route_relation": route,
        "provisional_review": negative,
        "checks": checks,
        "prospective_true_radial_safe_lateral_negative_passed": passed,
        "full_r750_closure_passed": False,
        "evidence_limit": "Large-model review is provisional silver, not human truth. This report cannot alone authorize training, calibration, Android, or deployment changes.",
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
        "prospective_true_radial_safe_lateral_negative_passed": value["prospective_true_radial_safe_lateral_negative_passed"],
        "full_r750_closure_passed": value["full_r750_closure_passed"],
        "output_sha256": common.sha256_file(parsed.output),
    }, ensure_ascii=False))
