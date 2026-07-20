#!/usr/bin/env python3
"""Audit legacy event roles against a current/past-only actionability state machine."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_public_video_actionability_relabel_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def replay(frames: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    intervention_threshold = float(policy["intervention_threshold"])
    clear_threshold = float(policy["route_clear_threshold"])
    enter_required = int(policy["intervention_consecutive_one_second_samples"])
    clear_required = int(policy["route_clear_consecutive_one_second_samples"])
    state = "context_attention"
    high_run = 0
    low_run = 0
    previous: int | None = None
    transitions: list[dict[str, Any]] = []
    invalid_scores = 0
    for frame in frames:
        timestamp = int(frame["timestamp_ms"])
        score = frame.get("trace_intrusion_score")
        if previous is None or timestamp - previous != 1000:
            high_run = 0
            low_run = 0
        previous = timestamp
        if score is None:
            invalid_scores += 1
            high_run = 0
            low_run = 0
            continue
        value = float(score)
        if state == "context_attention":
            high_run = high_run + 1 if value >= intervention_threshold else 0
            if high_run >= enter_required:
                state = "intervention_needed"
                transitions.append({"state": state, "timestamp_ms": timestamp})
                high_run = 0
                low_run = 0
        else:
            low_run = low_run + 1 if value < clear_threshold else 0
            if low_run >= clear_required:
                state = "context_attention"
                transitions.append({"state": "route_clear", "timestamp_ms": timestamp})
                high_run = 0
                low_run = 0
    intervention_count = sum(row["state"] == "intervention_needed" for row in transitions)
    if intervention_count == 0:
        actionability_class = "context_only"
    elif state == "intervention_needed":
        actionability_class = "persistent_intervention"
    else:
        actionability_class = "intervention_then_route_clear"
    return {
        "actionability_class": actionability_class,
        "intervention_episode_count": intervention_count,
        "final_state": state,
        "invalid_causal_score_count": invalid_scores,
        "transitions": transitions,
    }


def normalize_r756(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "source_id": event["source_id"],
        "legacy_route_role_positive": bool(int(event["label"])),
        "legacy_role": "route_intrusion_positive" if int(event["label"]) else "route_relation_negative",
        "frames": event["event_diagnostics"]["frames"],
    }


def normalize_r780a(event: dict[str, Any]) -> dict[str, Any]:
    role = str(event["role"])
    return {
        "event_id": f"{event['source_id']}:{int(event['event_entry_timestamp_ms'])}",
        "source_id": event["source_id"],
        "legacy_route_role_positive": role == "prospective_positive_event",
        "legacy_role": role,
        "frames": event["causal_trace"]["frames"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = load_json(args.contract)
    hashes = {
        "r756_report_sha256": sha256_file(args.r756_report),
        "r780a_report_sha256": sha256_file(args.r780a_report),
    }
    if hashes != contract["bound_inputs"]:
        raise ValueError("bound input hash mismatch")
    r756 = load_json(args.r756_report)
    r780a = load_json(args.r780a_report)
    normalized = [normalize_r756(row) for row in r756["events"]] + [normalize_r780a(row) for row in r780a["events"]]
    rows = []
    for event in normalized:
        actionability = replay(event.pop("frames"), contract["state_policy"])
        online_intervention_positive = actionability["actionability_class"] != "context_only"
        rows.append({
            **event,
            **actionability,
            "online_intervention_positive": online_intervention_positive,
            "legacy_route_role_disagrees_with_online_intervention": event["legacy_route_role_positive"] != online_intervention_positive,
        })
    intervention_sources = {row["source_id"] for row in rows if row["online_intervention_positive"]}
    context_sources = {row["source_id"] for row in rows if not row["online_intervention_positive"]}
    disagreements = sum(row["legacy_route_role_disagrees_with_online_intervention"] for row in rows)
    invalid = sum(row["invalid_causal_score_count"] for row in rows)
    gate = contract["readiness_gate"]
    checks = {
        "minimum_independent_intervention_sources": len(intervention_sources) >= int(gate["minimum_independent_intervention_sources"]),
        "minimum_independent_context_only_sources": len(context_sources) >= int(gate["minimum_independent_context_only_sources"]),
        "maximum_legacy_route_role_disagreement_fraction": disagreements / len(rows) <= float(gate["maximum_legacy_route_role_disagreement_fraction"]),
        "all_event_frames_have_valid_causal_scores": invalid == 0,
    }
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": sha256_file(args.contract), **hashes},
        "events": rows,
        "summary": {
            "event_count": len(rows),
            "source_count": len({row["source_id"] for row in rows}),
            "actionability_class_counts": {
                name: sum(row["actionability_class"] == name for row in rows)
                for name in ("context_only", "intervention_then_route_clear", "persistent_intervention")
            },
            "independent_intervention_source_count": len(intervention_sources),
            "independent_context_only_source_count": len(context_sources),
            "legacy_route_role_disagreement_count": disagreements,
            "legacy_route_role_disagreement_fraction": disagreements / len(rows),
            "intervention_sources": sorted(intervention_sources),
            "context_only_sources": sorted(context_sources),
        },
        "checks": checks,
        "actionability_probe_ready": all(checks.values()),
        "decision": "collect_more_independent_intervention_events_before_training" if not checks["minimum_independent_intervention_sources"] else "coverage_floor_met_subject_to_all_gates",
        "evidence_limit": "Post-hoc causal relabel audit. It may identify target-contract pollution but cannot establish event truth, calibration, blind validity, or deployment safety.",
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--r756-report", type=Path, required=True)
    parser.add_argument("--r780a-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "actionability_probe_ready": value["actionability_probe_ready"],
        "decision": value["decision"],
        "output_sha256": sha256_file(parsed.output),
    }))
