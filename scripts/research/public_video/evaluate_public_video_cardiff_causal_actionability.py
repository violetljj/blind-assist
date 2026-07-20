#!/usr/bin/env python3
"""Test frozen past-only actionability on the two retained Cardiff events."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import audit_public_video_actionability_relabels as relabel
import run_public_silver_frozen_feature_probe as common
import run_public_video_causal_past_ego_trace_probe as causal


SCHEMA = "blindassist_public_video_cardiff_causal_actionability_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_retained_review(review: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [row for row in review["reviews"] if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"review must bind one candidate: {candidate_id}")
    row = matches[0]
    if row["decision"] != "retain_as_provisional_positive_event_role_candidate":
        raise ValueError(f"candidate is not retained: {candidate_id}")
    if row.get("real_target_object") is not True or row.get("same_object_track") is not True:
        raise ValueError(f"candidate lacks real coherent target evidence: {candidate_id}")
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = load_json(args.contract)
    paths = {
        "r756_contract_sha256": args.r756_contract,
        "cardiff_review_sha256": args.cardiff_review,
        "cardiff_0_20_features_sha256": args.cardiff_0_20_features,
        "cardiff_remainder_features_sha256": args.cardiff_remainder_features,
    }
    for key, path in paths.items():
        if common.sha256_file(path) != contract["bound_inputs"][key]:
            raise ValueError(f"bound input mismatch: {key}")
    r756 = load_json(args.r756_contract)
    review = load_json(args.cardiff_review)
    features = {
        "cardiff_0_20": load_json(args.cardiff_0_20_features),
        "cardiff_remainder": load_json(args.cardiff_remainder_features),
    }
    rows = []
    for event in contract["events"]:
        review_row = select_retained_review(review, event["candidate_id"])
        diagnostic = causal.evaluate_event(
            features[event["feature_key"]],
            event["source_id"],
            tuple(map(int, event["causal_window_ms"])),
            r756["causal_teacher"],
        )
        state = relabel.replay(diagnostic["frames"], contract["state_policy"])
        rows.append({
            "candidate_id": event["candidate_id"],
            "source_id": event["source_id"],
            "review_confidence": review_row["confidence"],
            "causal_window_ms": event["causal_window_ms"],
            "actionability_class": state["actionability_class"],
            "intervention_episode_count": state["intervention_episode_count"],
            "invalid_causal_score_count": state["invalid_causal_score_count"],
            "transitions": state["transitions"],
            "causal_trace": diagnostic,
        })
    intervention_rows = [row for row in rows if row["actionability_class"] != "context_only"]
    checks = {
        "all_frames_have_valid_causal_scores": all(row["invalid_causal_score_count"] == 0 for row in rows),
        "at_least_one_cardiff_event_is_intervention_positive": bool(intervention_rows),
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
        "cardiff_third_source_gate_passed": all(checks.values()),
        "evidence_limit": "Frozen past-only scoring on previously reviewed provisional Cardiff candidates. This adds source-level diagnostic coverage only; it is not event truth, calibration, blind, or production evidence.",
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
    parser.add_argument("--cardiff-review", type=Path, required=True)
    parser.add_argument("--cardiff-0-20-features", type=Path, required=True)
    parser.add_argument("--cardiff-remainder-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "cardiff_third_source_gate_passed": value["cardiff_third_source_gate_passed"],
        "intervention_positive_event_count": value["summary"]["intervention_positive_event_count"],
        "output_sha256": common.sha256_file(parsed.output),
    }))
