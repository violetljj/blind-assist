#!/usr/bin/env python3
"""Audit per-direction class coverage behind an aggregate explicit-choice score."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_explicit_choice_direction_coverage_v1"


def summarize(events: list[dict[str, Any]], directions: list[str]) -> dict[str, dict[str, int]]:
    return {
        direction: {
            "intervention_event_count": sum(
                row["explicit_choice"] == direction and row["reference_intervention_required"] for row in events
            ),
            "context_event_count": sum(
                row["explicit_choice"] == direction and not row["reference_intervention_required"] for row in events
            ),
        }
        for direction in directions
    }


def run(contract_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or Path(str(output_path) + ".sha256").exists():
        raise ValueError("refusing to overwrite direction-coverage audit")
    contract = common.load_json(contract_path)
    input_path = Path(contract["bound_input"]["path"])
    if common.sha256_file(input_path) != contract["bound_input"]["sha256"]:
        raise ValueError("r799 report hash mismatch")
    report = common.load_json(input_path)
    gate = contract["coverage_gate"]
    directions = list(gate["required_directions"])
    coverage = summarize(report["events"], directions)
    checks = {
        "aggregate_r799_gate_passed": report.get("three_state_choice_provider_supported") is True,
        "minimum_intervention_events_per_direction": all(
            coverage[direction]["intervention_event_count"] >= int(gate["minimum_intervention_events_per_direction"])
            for direction in directions
        ),
        "minimum_context_events_per_direction": all(
            coverage[direction]["context_event_count"] >= int(gate["minimum_context_events_per_direction"])
            for direction in directions
        ),
    }
    straight = coverage["STRAIGHT"]
    result = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(contract_path),
                   "r799_report_sha256": common.sha256_file(input_path)},
        "direction_coverage": coverage, "checks": checks,
        "straight_choice_diagnostic_supported": straight["intervention_event_count"] >= 1 and
                                                straight["context_event_count"] >= 1,
        "full_three_state_provider_supported": bool(all(checks.values())),
        "authorization": contract["authorization"],
        "evidence_limit": "Perfect aggregate accuracy cannot authorize LEFT/RIGHT without positive events in those directions."
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(output_path) + ".sha256").write_text(common.sha256_file(output_path) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract, args.output)
    print(json.dumps({"direction_coverage": result["direction_coverage"],
                      "full_three_state_provider_supported": result["full_three_state_provider_supported"]}))


if __name__ == "__main__":
    main()
