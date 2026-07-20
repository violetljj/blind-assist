#!/usr/bin/env python3
"""Evaluate frozen r7.25 marker-approach candidates against a post-freeze review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_marker_radial_approach_contract as approach_contract
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_marker_radial_approach_review_result_v1"


def event_overlaps(event: dict[str, Any], window: Sequence[int]) -> bool:
    start = int(event["event_entry_timestamp_ms"])
    end = int(event.get("confirmed_clear_timestamp_ms", event["last_active_timestamp_ms"]))
    return start < int(window[1]) and end >= int(window[0])


def score_reviewed_sources(
    candidates: dict[str, Any], review: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_source = {row["source_id"]: row.get("events", []) for row in candidates.get("sources", [])}
    rows: list[dict[str, Any]] = []
    counts = {"scored_negative": 0, "passed_negative": 0, "context_only": 0}
    for item in review["sources"]:
        source_id = item["source_id"]
        if source_id not in by_source:
            raise ValueError(f"review source missing from candidates: {source_id}")
        role = item["evaluation_role"]
        if role == "context_only":
            counts["context_only"] += 1
            rows.append({
                "source_id": source_id,
                "evaluation_role": role,
                "gate_credit": False,
                "candidate_event_count": len(by_source[source_id]),
                "reason": item["reason"],
            })
            continue
        if role != "prospective_negative_control":
            raise ValueError(f"unsupported evaluation role: {role}")
        finding = item["visual_finding"]
        if finding.get("pedestrian_corridor_risk_present") is not False:
            raise ValueError("negative control claims corridor risk")
        if finding.get("should_open_risk_event") is not False:
            raise ValueError("negative control claims event should open")
        window = item["review_window_ms"]
        overlapping = [event for event in by_source[source_id] if event_overlaps(event, window)]
        passed = not overlapping
        counts["scored_negative"] += 1
        counts["passed_negative"] += int(passed)
        rows.append({
            "source_id": source_id,
            "evaluation_role": role,
            "gate_credit": True,
            "review_window_ms": window,
            "overlapping_candidate_events": overlapping,
            "negative_control_passed": passed,
            "visual_finding": finding,
        })
    return rows, counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract, contract_meta = approach_contract.load_contract(args.contract)
    features = lifecycle.verify_json_sidecar(args.features)
    candidates = lifecycle.verify_json_sidecar(args.candidates)
    review = lifecycle.verify_json_sidecar(args.review)

    feature_sha = common.sha256_file(args.features)
    candidate_sha = common.sha256_file(args.candidates)
    if candidates.get("approach_contract", {}).get("sha256") != contract_meta["sha256"]:
        raise ValueError("candidate contract hash mismatch")
    if candidates.get("feature_report_sha256") != feature_sha:
        raise ValueError("candidate feature hash mismatch")
    if review.get("approach_contract_sha256") != contract_meta["sha256"]:
        raise ValueError("review contract hash mismatch")
    if review.get("feature_report_sha256") != feature_sha:
        raise ValueError("review feature hash mismatch")
    if review.get("candidate_report_sha256") != candidate_sha:
        raise ValueError("review candidate hash mismatch")
    if review.get("reviewed_after_features_and_candidates_frozen") is not True:
        raise ValueError("review chronology is invalid")
    if review.get("reviewer_type") != "large_model_visual_review_not_human_truth":
        raise ValueError("review evidence role is not explicit")

    feature_video_sha = {
        row["source_id"]: row["video_sha256"] for row in features.get("sources", [])
    }
    for item in review["sources"]:
        if feature_video_sha.get(item["source_id"]) != item.get("video_sha256"):
            raise ValueError(f"video lineage mismatch: {item['source_id']}")
        continuity = item.get("continuity", {})
        if continuity != {
            "continuous_ego_pedestrian_capture": True,
            "original_temporal_order": True,
            "hard_cut_or_montage_observed": False,
        }:
            raise ValueError(f"continuity gate failed: {item['source_id']}")

    rows, counts = score_reviewed_sources(candidates, review)
    passed = counts["scored_negative"] > 0 and counts["scored_negative"] == counts["passed_negative"]
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "approach_contract": contract_meta,
            "feature_report_sha256": feature_sha,
            "candidate_report_sha256": candidate_sha,
            "review_sha256": common.sha256_file(args.review),
        },
        "sources": rows,
        "summary": {**counts, "prospective_negative_gate_passed": passed},
        "evidence_limit": (
            "Large-model visual review is provisional silver evidence, not human truth. "
            "Context-only sources receive no gate credit; no independent prospective positive is established."
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
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({"ok": True, **value["summary"], "output_sha256": common.sha256_file(parsed.output)}))
