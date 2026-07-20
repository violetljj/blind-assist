#!/usr/bin/env python3
"""Run the frozen r7.54 future-ego-trace teacher across multiple real sources."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_future_ego_trace_probe as trace


SCHEMA = "blindassist_public_video_future_ego_trace_multisource_probe_v1"


def separation_checks(rows: list[dict[str, Any]], minimum_valid_fraction: float) -> dict[str, bool]:
    positives = [float(row["score"]) for row in rows if int(row["label"]) == 1]
    negatives = [float(row["score"]) for row in rows if int(row["label"]) == 0]
    return {
        "all_events_have_sufficient_valid_frames": all(float(row["valid_frame_fraction"]) >= minimum_valid_fraction for row in rows),
        "both_classes_present": bool(positives and negatives),
        "strict_complete_separation": bool(positives and negatives) and min(positives) > max(negatives),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.teacher_contract, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    teacher_contract = json.loads(args.teacher_contract.read_text(encoding="utf-8"))
    if common.sha256_file(args.teacher_contract) != contract["teacher_contract_sha256"]:
        raise ValueError("teacher contract hash mismatch")
    reports = {}
    for key, binding in contract["feature_reports"].items():
        path = Path(binding["path"])
        if common.sha256_file(path) != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {key}")
        reports[key] = lifecycle.verify_json_sidecar(path)
    rows = []
    for event in contract["events"]:
        result = trace.evaluate_event(
            reports[event["feature_key"]], event["source_id"], tuple(map(int, event["window_ms"])), teacher_contract["teacher"]
        )
        score = result["mean_trace_intrusion_score"]
        if score is None:
            score = 0.0
        rows.append({
            "event_id": event["event_id"],
            "source_id": event["source_id"],
            "label": int(event["label"]),
            "score": float(score),
            "valid_frame_fraction": float(result["valid_frame_fraction"]),
            "event_diagnostics": result,
        })
    checks = separation_checks(rows, float(contract["gate"]["minimum_valid_frame_fraction"]))
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "contract_sha256": common.sha256_file(args.contract),
            "teacher_contract_sha256": common.sha256_file(args.teacher_contract),
            "feature_report_sha256": {key: value["sha256"] for key, value in contract["feature_reports"].items()},
        },
        "aggregation": contract["aggregation"],
        "events": rows,
        "checks": checks,
        "diagnostic_gate_passed": all(checks.values()),
        "evidence_limit": contract["evidence_role"],
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--teacher-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, "diagnostic_gate_passed": value["diagnostic_gate_passed"],
                      "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
