#!/usr/bin/env python3
"""Validate three isolated Codex reviews per role and emit provisional consensus."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .contract import (
        BUNDLE_SCHEMA, CONSENSUS_SCHEMA, CONTRACT_ID, canonical_sha256, load_json, majority,
        require_false_flags, sha256_file, validate_review, write_json,
    )
except ImportError:  # Direct execution through scripts/run_research_tool.py.
    from contract import (
        BUNDLE_SCHEMA, CONSENSUS_SCHEMA, CONTRACT_ID, canonical_sha256, load_json, majority,
        require_false_flags, sha256_file, validate_review, write_json,
    )


def frame_time_lookup(bundle: Mapping[str, Any], role: str) -> dict[str, int]:
    return {frame["frame_id"]: int(frame["relative_ms"]) for frame in bundle["review_artifacts"][role]["frames"]}


def consensus_events(reviews: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    times = frame_time_lookup(bundle, role)
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for review in reviews:
        for event in review["events"]:
            groups[(event["category"], event["route_relation"], event["required_action"])].append(event)
    results: list[dict[str, Any]] = []
    for key, rows in groups.items():
        contributing_rounds = sum(
            1 for review in reviews if any(
                (event["category"], event["route_relation"], event["required_action"]) == key
                for event in review["events"]
            )
        )
        if contributing_rounds < 2:
            continue
        start_ms = round(statistics.median(times[row["start_frame"]] for row in rows))
        end_ms = round(statistics.median(times[row["end_frame"]] for row in rows))
        peak_ms = round(statistics.median(times[row["peak_frame"]] for row in rows))
        distance, _ = majority(row["distance_band"] for row in rows)
        ttc, _ = majority(row["ttc_band"] for row in rows)
        confidence, _ = majority(row["confidence"] for row in rows)
        results.append({
            "category": key[0],
            "route_relation": key[1],
            "required_action": key[2],
            "distance_band": distance or "unknown",
            "ttc_band": ttc or "unknown",
            "confidence": confidence or "low",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "peak_ms": peak_ms,
            "contributing_rounds": contributing_rounds,
            "event_vote_count": len(rows),
        })
    return sorted(results, key=lambda row: (row["start_ms"], row["category"], row["route_relation"]))


def build_role_consensus(reviews: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any], role: str) -> dict[str, Any]:
    risk, risk_count = majority(review["overall_risk"] for review in reviews)
    route, route_count = majority(review["route_valid"] for review in reviews)
    events = consensus_events(reviews, bundle, role)
    abstain = sorted({reason for review in reviews for reason in review["abstain_reasons"]})
    decision = "CONSENSUS_AVAILABLE" if risk is not None and route is not None else "NO_TWO_OF_THREE_CONSENSUS"
    if decision != "CONSENSUS_AVAILABLE":
        risk = "unknown"
        route = "uncertain"
        events = []
        if "other" not in abstain:
            abstain.append("other")
    return {
        "role": role,
        "decision": decision,
        "route_valid": route,
        "overall_risk": risk,
        "events": events,
        "abstain_reasons": abstain,
        "risk_agreement_count": risk_count,
        "route_agreement_count": route_count,
        "review_count": len(reviews),
        "review_ids": [review["review_id"] for review in reviews],
        "review_payload_sha256": [canonical_sha256(review) for review in reviews],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    bundle = load_json(args.bundle_manifest)
    if not isinstance(bundle, dict) or bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("contract_id") != CONTRACT_ID:
        raise ValueError("bundle manifest schema/contract mismatch")
    require_false_flags(bundle, "bundle manifest")
    bundle_sha = sha256_file(args.bundle_manifest)
    reviews_by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    paths = sorted(path for path in args.reviews_dir.glob("*.json") if path.is_file())
    if not paths:
        raise ValueError("reviews directory is empty")
    seen_ids: set[str] = set()
    seen_rounds: set[tuple[str, int]] = set()
    review_receipts: list[dict[str, str]] = []
    for path in paths:
        review = validate_review(load_json(path), bundle=bundle, bundle_sha256=bundle_sha)
        if review["review_id"] in seen_ids:
            raise ValueError("duplicate review_id")
        round_key = (review["role"], review["round"])
        if round_key in seen_rounds:
            raise ValueError("duplicate role/round review")
        seen_ids.add(review["review_id"])
        seen_rounds.add(round_key)
        reviews_by_role[review["role"]].append(review)
        review_receipts.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    for role in ("full_context_teacher", "causal_codex_baseline"):
        rounds = sorted(review["round"] for review in reviews_by_role.get(role, []))
        if rounds != [1, 2, 3]:
            raise ValueError(f"{role} requires exactly isolated rounds 1,2,3")
    roles = {
        role: build_role_consensus(reviews_by_role[role], bundle, role)
        for role in ("full_context_teacher", "causal_codex_baseline")
    }
    report = {
        "schema": CONSENSUS_SCHEMA,
        "contract_id": CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest": str(args.bundle_manifest.resolve()),
        "bundle_manifest_sha256": bundle_sha,
        "review_receipts": review_receipts,
        "roles": roles,
        "teacher_reference_authority": "provisional_silver_only",
        "causal_baseline_authority": "provisional_comparator_only",
        "human_event_truth_present": False,
        "metric_geometry_present": False,
        "training_authorized": False,
        "u0_authority_granted": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    write_json(args.output, report)
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--reviews-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(parse_args(argv))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "teacher_decision": report["roles"]["full_context_teacher"]["decision"],
        "causal_decision": report["roles"]["causal_codex_baseline"]["decision"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
