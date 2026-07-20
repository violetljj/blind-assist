#!/usr/bin/env python3
"""Build a source-isolated event manifest from causal actionability audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_public_video_actionability_manifest_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_item(event_id: str, source_id: str, window_ms: list[int], action: dict[str, Any], origin: str) -> dict[str, Any]:
    actionability_class = action["actionability_class"]
    if actionability_class not in {"context_only", "intervention_then_route_clear", "persistent_intervention"}:
        raise ValueError(f"unsupported actionability class: {actionability_class}")
    if int(action.get("invalid_causal_score_count", 0)) != 0:
        raise ValueError(f"invalid causal scores: {event_id}")
    return {
        "item_id": event_id,
        "parent_source_id": source_id,
        "window_ms": list(map(int, window_ms)),
        "actionability_class": actionability_class,
        "intervention_required": actionability_class != "context_only",
        "intervention_episode_count": int(action["intervention_episode_count"]),
        "transitions": action["transitions"],
        "label_basis": "frozen_current_or_past_only_causal_trace",
        "origin_report": origin,
    }


def coverage_checks(items: list[dict[str, Any]], gate: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    positives = [row for row in items if row["intervention_required"]]
    negatives = [row for row in items if not row["intervention_required"]]
    positive_sources = {row["parent_source_id"] for row in positives}
    negative_sources = {row["parent_source_id"] for row in negatives}
    ratio = max(len(positives), len(negatives)) / max(1, min(len(positives), len(negatives)))
    identifiers = [row["item_id"] for row in items]
    checks = {
        "minimum_intervention_events": len(positives) >= int(gate["minimum_intervention_events"]),
        "minimum_context_only_events": len(negatives) >= int(gate["minimum_context_only_events"]),
        "minimum_independent_intervention_sources": len(positive_sources) >= int(gate["minimum_independent_intervention_sources"]),
        "minimum_independent_context_only_sources": len(negative_sources) >= int(gate["minimum_independent_context_only_sources"]),
        "maximum_event_class_ratio": ratio <= float(gate["maximum_event_class_ratio"]),
        "all_causal_scores_valid": all(row["label_basis"] == "frozen_current_or_past_only_causal_trace" for row in items),
        "all_item_ids_unique": len(identifiers) == len(set(identifiers)),
    }
    summary = {
        "event_count": len(items),
        "intervention_event_count": len(positives),
        "context_only_event_count": len(negatives),
        "independent_intervention_source_count": len(positive_sources),
        "independent_context_only_source_count": len(negative_sources),
        "event_class_ratio": ratio,
        "intervention_sources": sorted(positive_sources),
        "context_only_sources": sorted(negative_sources),
    }
    return checks, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = load_json(args.contract)
    paths = {
        "r756_report_sha256": args.r756_report,
        "r780a_report_sha256": args.r780a_report,
        "r786_report_sha256": args.r786_report,
        "r787_report_sha256": args.r787_report,
        "r788_report_sha256": args.r788_report,
    }
    for key, path in paths.items():
        if sha256_file(path) != contract["bound_inputs"][key]:
            raise ValueError(f"bound input mismatch: {key}")
    r756 = load_json(args.r756_report)
    r780a = load_json(args.r780a_report)
    r786 = load_json(args.r786_report)
    r787 = load_json(args.r787_report)
    r788 = load_json(args.r788_report)
    audited = {row["event_id"]: row for row in r786["events"]}
    items = []
    for event in r756["events"]:
        action = audited[event["event_id"]]
        items.append(make_item(
            event["event_id"], event["source_id"], event["event_diagnostics"]["window_ms"], action, "r786:r756"
        ))
    for event in r780a["events"]:
        event_id = f"{event['source_id']}:{int(event['event_entry_timestamp_ms'])}"
        action = audited[event_id]
        items.append(make_item(
            event_id,
            event["source_id"],
            [int(event["event_entry_timestamp_ms"]), int(event["event_last_active_timestamp_ms"]) + 1000],
            action,
            "r786:r780a",
        ))
    for report_name, report in (("r787", r787), ("r788", r788)):
        for event in report["events"]:
            items.append(make_item(
                event["candidate_id"], event["source_id"], event["causal_window_ms"], event, report_name
            ))
    forbidden = {"label", "legacy_role", "legacy_route_role_positive", "original_review_role"}
    if any(forbidden.intersection(row) for row in items):
        raise ValueError("legacy route-role field leaked into manifest")
    checks, summary = coverage_checks(items, contract["coverage_gate"])
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": sha256_file(args.contract), **{key: sha256_file(path) for key, path in paths.items()}},
        "target_contract": contract["target_contract"],
        "items": items,
        "summary": summary,
        "checks": checks,
        "deterministic_actionability_probe_ready": all(checks.values()),
        "evidence_limit": "Causal provisional event manifest for a deterministic source-heldout representation probe only. It is not event truth, calibration, blind evidence, or deployment authority.",
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
    parser.add_argument("--r786-report", type=Path, required=True)
    parser.add_argument("--r787-report", type=Path, required=True)
    parser.add_argument("--r788-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "deterministic_actionability_probe_ready": value["deterministic_actionability_probe_ready"],
        "summary": value["summary"],
        "output_sha256": sha256_file(parsed.output),
    }))
