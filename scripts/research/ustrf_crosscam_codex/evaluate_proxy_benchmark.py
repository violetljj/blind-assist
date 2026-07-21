#!/usr/bin/env python3
"""Compare causal candidate traces with provisional full-context Codex silver events."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .contract import (
        ACTIONS, CANDIDATE_SCHEMA, CATEGORIES, CONSENSUS_SCHEMA, CONTRACT_ID, REPORT_SCHEMA,
        ROUTE_RELATIONS, TTC_BANDS, load_json, require_false_flags, sha256_file, write_json,
    )
except ImportError:  # Direct execution through scripts/run_research_tool.py.
    from contract import (
        ACTIONS, CANDIDATE_SCHEMA, CATEGORIES, CONSENSUS_SCHEMA, CONTRACT_ID, REPORT_SCHEMA,
        ROUTE_RELATIONS, TTC_BANDS, load_json, require_false_flags, sha256_file, write_json,
    )


def event_match(reference: Mapping[str, Any], candidate: Mapping[str, Any], tolerance_ms: int) -> bool:
    if reference["route_relation"] in ("inside", "entering") and candidate["route_relation"] not in ("inside", "entering"):
        return False
    if reference["route_relation"] == "outside" and candidate["route_relation"] != "outside":
        return False
    if reference["category"] != "other" and candidate["category"] not in (reference["category"], "other"):
        return False
    return not (
        int(candidate["end_ms"]) < int(reference["start_ms"]) - tolerance_ms
        or int(candidate["start_ms"]) > int(reference["end_ms"]) + tolerance_ms
    )


def metrics(reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], duration_ms: int, tolerance_ms: int) -> dict[str, Any]:
    matched_candidate: set[int] = set()
    matched_reference = 0
    critical_misses = 0
    onset_errors: list[int] = []
    clearance_errors: list[int] = []
    for ref in reference:
        matches = [
            index for index, row in enumerate(candidate)
            if index not in matched_candidate and event_match(ref, row, tolerance_ms)
        ]
        if not matches:
            if ref["required_action"] in ("stop", "detour") or ref["ttc_band"] == "0-1.5s":
                critical_misses += 1
            continue
        index = min(matches, key=lambda item: abs(int(candidate[item]["peak_ms"]) - int(ref["peak_ms"])))
        matched_candidate.add(index)
        matched_reference += 1
        onset_errors.append(int(candidate[index]["start_ms"]) - int(ref["start_ms"]))
        clearance_errors.append(int(candidate[index]["end_ms"]) - int(ref["end_ms"]))
    false_alerts = len(candidate) - len(matched_candidate)
    minutes = duration_ms / 60_000.0
    return {
        "reference_event_count": len(reference),
        "candidate_event_count": len(candidate),
        "matched_event_count": matched_reference,
        "event_recall": matched_reference / len(reference) if reference else None,
        "critical_miss_count": critical_misses,
        "false_alert_count": false_alerts,
        "false_alerts_per_minute": false_alerts / minutes if minutes > 0 else None,
        "mean_onset_error_ms": sum(onset_errors) / len(onset_errors) if onset_errors else None,
        "mean_clearance_error_ms": sum(clearance_errors) / len(clearance_errors) if clearance_errors else None,
    }


def validate_candidate(value: Any, *, bundle_sha: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != CANDIDATE_SCHEMA or value.get("contract_id") != CONTRACT_ID:
        raise ValueError("candidate schema/contract mismatch")
    if value.get("bundle_manifest_sha256") != bundle_sha:
        raise ValueError("candidate is bound to another bundle")
    if not isinstance(value.get("candidate_id"), str) or not value["candidate_id"]:
        raise ValueError("candidate_id is missing")
    events = value.get("events")
    if not isinstance(events, list):
        raise ValueError("candidate events must be an array")
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"candidate event {index} must be an object")
        enums = {
            "category": CATEGORIES,
            "route_relation": ROUTE_RELATIONS,
            "required_action": ACTIONS,
            "ttc_band": TTC_BANDS,
        }
        for key, allowed in enums.items():
            if event.get(key) not in allowed:
                raise ValueError(f"candidate event {index} has invalid {key}")
        for key in ("start_ms", "end_ms", "peak_ms"):
            if not isinstance(event.get(key), int) or event[key] < 0:
                raise ValueError(f"candidate event {index} lacks valid {key}")
    require_false_flags(value, "candidate")
    return value


def causal_consensus_candidate(consensus: Mapping[str, Any]) -> dict[str, Any]:
    role = consensus["roles"]["causal_codex_baseline"]
    return {
        "schema": CANDIDATE_SCHEMA,
        "contract_id": CONTRACT_ID,
        "candidate_id": "codex_causal_visual_baseline_v1",
        "candidate_kind": "codex_causal_provisional_comparator",
        "bundle_manifest_sha256": consensus["bundle_manifest_sha256"],
        "overall_risk": role["overall_risk"],
        "events": role["events"],
        "human_event_truth_present": False,
        "metric_geometry_present": False,
        "training_authorized": False,
        "u0_authority_granted": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    consensus = load_json(args.consensus)
    if not isinstance(consensus, dict) or consensus.get("schema") != CONSENSUS_SCHEMA or consensus.get("contract_id") != CONTRACT_ID:
        raise ValueError("consensus schema/contract mismatch")
    require_false_flags(consensus, "consensus")
    teacher = consensus["roles"]["full_context_teacher"]
    bundle = load_json(Path(consensus["bundle_manifest"]))
    duration_ms = int(bundle["window"]["duration_ms"])
    candidates: list[tuple[Mapping[str, Any], str | None]] = [(causal_consensus_candidate(consensus), None)]
    for path in args.candidate:
        candidates.append((validate_candidate(load_json(path), bundle_sha=consensus["bundle_manifest_sha256"]), str(path.resolve())))
    rows = []
    for candidate, path in candidates:
        scored = teacher["decision"] == "CONSENSUS_AVAILABLE"
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "candidate_kind": candidate.get("candidate_kind"),
            "candidate_path": path,
            "candidate_sha256": sha256_file(Path(path)) if path else None,
            "scored": scored,
            "metrics": metrics(teacher["events"], candidate["events"], duration_ms, args.tolerance_ms) if scored else None,
        })
    external_ids = [row["candidate_id"] for row in rows if row["candidate_id"] != "codex_causal_visual_baseline_v1"]
    if teacher["decision"] != "CONSENSUS_AVAILABLE":
        decision = "BLOCKED_ON_TEACHER_CONSENSUS"
    elif not external_ids:
        decision = "BLOCKED_ON_EXTERNAL_CANDIDATE_TRACE"
    else:
        decision = "PROXY_COMPARISON_AVAILABLE"
    report = {
        "schema": REPORT_SCHEMA,
        "contract_id": CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "consensus_path": str(args.consensus.resolve()),
        "consensus_sha256": sha256_file(args.consensus),
        "reference": {
            "role": "full_context_teacher",
            "authority": "provisional_silver_only",
            "decision": teacher["decision"],
            "overall_risk": teacher["overall_risk"],
            "event_count": len(teacher["events"]),
        },
        "candidates": rows,
        "decision": decision,
        "claim_limit": "Proxy ranking only. Codex teacher is not human truth and assumed geometry is not device metric geometry.",
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
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--tolerance-ms", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(parse_args(argv))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "decision": report["decision"], "candidate_count": len(report["candidates"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
