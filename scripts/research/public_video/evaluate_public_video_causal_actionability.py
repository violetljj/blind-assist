#!/usr/bin/env python3
"""Reinterpret tiered alerts using causal no-change intervention states."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA = "blindassist_public_video_causal_actionability_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def consecutive_endpoint(
    frames: list[dict[str, Any]],
    predicate: Callable[[float], bool],
    required: int,
    after_timestamp_ms: int | None = None,
) -> int | None:
    run = 0
    previous: int | None = None
    for frame in frames:
        timestamp = int(frame["timestamp_ms"])
        if after_timestamp_ms is not None and timestamp <= after_timestamp_ms:
            continue
        score = frame.get("trace_intrusion_score")
        if previous is None or timestamp - previous != 1000:
            run = 0
        if score is not None and predicate(float(score)):
            run += 1
            if run >= required:
                return timestamp
        else:
            run = 0
        previous = timestamp
    return None


def classify_event(event: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    frames = event["causal_trace"]["frames"]
    threshold = float(policy["intervention_threshold"])
    intervention = consecutive_endpoint(
        frames,
        lambda score: score >= threshold,
        int(policy["intervention_consecutive_one_second_samples"]),
    )
    route_clear = None
    if intervention is not None:
        clear_threshold = float(policy["route_clear_threshold"])
        route_clear = consecutive_endpoint(
            frames,
            lambda score: score < clear_threshold,
            int(policy["route_clear_consecutive_one_second_samples"]),
            after_timestamp_ms=intervention,
        )
    event_end = int(event["event_last_active_timestamp_ms"])
    if intervention is None:
        actionability_class = "context_only"
    elif route_clear is not None and route_clear <= event_end:
        actionability_class = "intervention_then_route_clear"
    else:
        actionability_class = "persistent_intervention"
    transitions = [{"state": "context_attention", "timestamp_ms": int(event["tier_1_context_notice_timestamp_ms"])}]
    if intervention is not None:
        transitions.append({"state": "intervention_needed", "timestamp_ms": intervention})
    if route_clear is not None and route_clear <= event_end:
        transitions.append({"state": "route_clear", "timestamp_ms": route_clear})
    transitions.append({"state": "marker_clear", "timestamp_ms": int(event["lifecycle_clear_timestamp_ms"])})
    original_role = str(event["role"])
    return {
        "source_id": event["source_id"],
        "event_entry_timestamp_ms": int(event["event_entry_timestamp_ms"]),
        "event_last_active_timestamp_ms": event_end,
        "original_review_role": original_role,
        "actionability_class": actionability_class,
        "intervention_timestamp_ms": intervention,
        "route_clear_timestamp_ms": route_clear if route_clear is not None and route_clear <= event_end else None,
        "intervention_duration_until_route_clear_ms": None if intervention is None or route_clear is None else route_clear - intervention,
        "eventual_safe_label_conflicts_with_causal_intervention": original_role == "true_radial_safe_lateral_negative" and intervention is not None,
        "transitions": transitions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = load_json(args.contract)
    report_hash = sha256_file(args.r780a_report)
    if report_hash != contract["bound_input"]["r780a_report_sha256"]:
        raise ValueError("r7.80a report hash mismatch")
    source = load_json(args.r780a_report)
    rows = [classify_event(event, contract["state_policy"]) for event in source["events"]]
    actual = {(row["source_id"], row["event_entry_timestamp_ms"]): row for row in rows}
    expectation_checks = []
    for expected in contract["frozen_event_expectations"]:
        key = (expected["source_id"], int(expected["event_entry_timestamp_ms"]))
        if key not in actual:
            raise ValueError(f"missing expected event: {key}")
        observed = actual[key]["actionability_class"]
        expectation_checks.append({
            "source_id": key[0],
            "event_entry_timestamp_ms": key[1],
            "expected": expected["expected_actionability_class"],
            "observed": observed,
            "passed": observed == expected["expected_actionability_class"],
        })
    contradiction_count = sum(row["eventual_safe_label_conflicts_with_causal_intervention"] for row in rows)
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "contract_sha256": sha256_file(args.contract),
            "r780a_report_sha256": report_hash,
        },
        "events": rows,
        "expectation_checks": expectation_checks,
        "summary": {
            "event_count": len(rows),
            "actionability_class_counts": {
                name: sum(row["actionability_class"] == name for row in rows)
                for name in ("context_only", "intervention_then_route_clear", "persistent_intervention")
            },
            "eventual_safe_label_causal_contradiction_count": contradiction_count,
            "frozen_expectations_passed": all(row["passed"] for row in expectation_checks),
        },
        "conclusion": "An eventual safe pass cannot serve as a no-warning label when the causal no-change corridor first requires intervention and later clears. Use online actionability states for alert policy; keep eventual route outcome only as a lifecycle/response attribute.",
        "evidence_limit": "Post-hoc offline semantics audit on three provisional silver events. Passing supports a target-contract change, not model accuracy, calibration, blind validity, or deployment.",
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--r780a-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "frozen_expectations_passed": value["summary"]["frozen_expectations_passed"],
        "causal_contradictions": value["summary"]["eventual_safe_label_causal_contradiction_count"],
        "output_sha256": sha256_file(parsed.output),
    }))
