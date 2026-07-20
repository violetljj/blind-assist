#!/usr/bin/env python3
"""Test frozen causal actionability on Ulm route-turn construction events."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import audit_public_video_actionability_relabels as relabel
import run_public_silver_frozen_feature_probe as common
import run_public_video_causal_past_ego_trace_probe as causal


SCHEMA = "blindassist_public_video_ulm_route_turn_actionability_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_route_turn_rejection(review: dict[str, Any], candidate_id: str, reason_fragment: str) -> dict[str, Any]:
    rejected = review["true_radial_safe_lateral_negative"]["rejected_frozen_candidates"]
    matches = [row for row in rejected if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"review must bind one rejected safe-lateral candidate: {candidate_id}")
    row = matches[0]
    if reason_fragment not in row["reason"]:
        raise ValueError(f"route-turn review reason mismatch: {candidate_id}")
    if review["source"].get("continuous_ego_pedestrian_capture") is not True:
        raise ValueError("source is not continuous ego pedestrian capture")
    return row


def select_candidate(candidates: dict[str, Any], source_id: str, entry: int, end_inclusive: int) -> dict[str, Any]:
    source_rows = [row for row in candidates["sources"] if row["source_id"] == source_id]
    if len(source_rows) != 1:
        raise ValueError(f"candidate source missing: {source_id}")
    matches = [row for row in source_rows[0]["events"]
               if int(row["event_entry_timestamp_ms"]) == entry and int(row["last_active_timestamp_ms"]) == end_inclusive]
    if len(matches) != 1 or matches[0].get("radial_approach_passed") is not True:
        raise ValueError("frozen radial candidate mismatch")
    return matches[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = load_json(args.contract)
    bindings = {
        "r756_contract_sha256": args.r756_contract,
        "ulm_review_sha256": args.ulm_review,
        "ulm_features_sha256": args.ulm_features,
        "ulm_candidates_sha256": args.ulm_candidates,
    }
    for key, path in bindings.items():
        if common.sha256_file(path) != contract["bound_inputs"][key]:
            raise ValueError(f"bound input mismatch: {key}")
    r756 = load_json(args.r756_contract)
    review = load_json(args.ulm_review)
    features = load_json(args.ulm_features)
    candidates = load_json(args.ulm_candidates)
    rows = []
    for event in contract["events"]:
        window = tuple(map(int, event["causal_window_ms"]))
        review_row = select_route_turn_rejection(
            review, event["candidate_id"], event["required_review_reason_fragment"]
        )
        candidate = select_candidate(candidates, event["source_id"], window[0], window[1] - 1000)
        diagnostic = causal.evaluate_event(features, event["source_id"], window, r756["causal_teacher"])
        state = relabel.replay(diagnostic["frames"], contract["state_policy"])
        rows.append({
            "candidate_id": event["candidate_id"],
            "source_id": event["source_id"],
            "review_reason": review_row["reason"],
            "accepted_sample_count": candidate["accepted_sample_count"],
            "causal_window_ms": list(window),
            "actionability_class": state["actionability_class"],
            "intervention_episode_count": state["intervention_episode_count"],
            "invalid_causal_score_count": state["invalid_causal_score_count"],
            "transitions": state["transitions"],
            "causal_trace": diagnostic,
        })
    intervention_rows = [row for row in rows if row["actionability_class"] != "context_only"]
    checks = {
        "all_frames_have_valid_causal_scores": all(row["invalid_causal_score_count"] == 0 for row in rows),
        "at_least_one_ulm_route_turn_event_is_intervention_positive": bool(intervention_rows),
        "minimum_intervention_episode_count": sum(row["intervention_episode_count"] for row in rows) >= int(contract["gate"]["minimum_intervention_episode_count"]),
        "threshold_or_window_search": contract["gate"]["threshold_or_window_search"] is False,
    }
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract)},
        "events": rows,
        "summary": {
            "event_count": len(rows),
            "intervention_positive_event_count": len(intervention_rows),
            "intervention_episode_count": sum(row["intervention_episode_count"] for row in rows),
            "new_independent_intervention_source_count": 1 if intervention_rows else 0,
        },
        "checks": checks,
        "ulm_third_source_gate_passed": all(checks.values()),
        "evidence_limit": "Frozen past-only scoring on previously reviewed real route-turn construction candidates. Passing adds provisional source coverage only; it is not calibration, blind, human truth, or production evidence.",
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--r756-contract", type=Path, required=True)
    parser.add_argument("--ulm-review", type=Path, required=True)
    parser.add_argument("--ulm-features", type=Path, required=True)
    parser.add_argument("--ulm-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "ulm_third_source_gate_passed": value["ulm_third_source_gate_passed"],
        "intervention_positive_event_count": value["summary"]["intervention_positive_event_count"],
        "output_sha256": common.sha256_file(parsed.output),
    }))
