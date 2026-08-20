"""BA-only held-out evaluation after the GC2-B winner is irrevocably locked."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_2a.evaluator import evaluate_condition
from scripts.research.goal_copilot_2b.heldout_crypto import unseal

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE.parent / "goal_copilot_bridge" / "pilot" / "dev_scenarios.json"
FROZEN_WINNER = HERE.parent / "goal_copilot_2a" / "frozen_gc1_winner.py"
CONDITIONS = ("CLEAN", "COMBINED_MILD", "COMBINED_MODERATE")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def evaluate_locked(
    candidate: Path, envelope_path: Path, key_path: Path, output_path: Path
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"held-out receipt already exists: {output_path}")
    payload = unseal(
        json.loads(envelope_path.read_text(encoding="utf-8")),
        key_path.read_text(encoding="ascii").strip(),
    )
    base_scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]
    schedules = []
    all_pass = True
    for schedule in payload["schedules"]:
        reordered = [base_scenarios[index] for index in schedule["scenario_order"]]
        with tempfile.TemporaryDirectory() as temporary:
            scenario_path = Path(temporary) / "scenarios.json"
            scenario_path.write_bytes(canonical({"scenarios": reordered}))
            cells = {
                policy_name: {
                    condition: evaluate_condition(policy_path, scenario_path, condition)
                    for condition in CONDITIONS
                }
                for policy_name, policy_path in {
                    "FROZEN_WINNER": FROZEN_WINNER,
                    "LOCKED_CANDIDATE": candidate,
                }.items()
            }
        cm = cells["LOCKED_CANDIDATE"]["COMBINED_MODERATE"]["metrics"]
        wm = cells["FROZEN_WINNER"]["COMBINED_MODERATE"]["metrics"]
        candidate_metrics = [
            cells["LOCKED_CANDIDATE"][condition]["metrics"] for condition in CONDITIONS
        ]
        improved_families = sum(
            cm["family_completion_counts"][family] > wm["family_completion_counts"][family]
            for family in cm["family_completion_counts"]
        )
        passed = (
            all(item["semantic_validity"] for item in candidate_metrics)
            and sum(item["unsafe_guidance"] for item in candidate_metrics) == 0
            and sum(item["premature_completion"] for item in candidate_metrics) == 0
            and cells["LOCKED_CANDIDATE"]["CLEAN"]["metrics"]["completion_count"] == 12
            and cells["LOCKED_CANDIDATE"]["COMBINED_MILD"]["metrics"]["completion_count"] >= 10
            and cm["completion_count"] >= 8
            and cm["completion_count"] - wm["completion_count"] >= 4
            and improved_families >= 2
            and cm["eligible_reacquisition_rate"] >= 2 / 3
        )
        all_pass = all_pass and passed
        schedules.append(
            {
                "seed": schedule["seed"],
                "passed": passed,
                "improved_family_count": improved_families,
                "cells": cells,
            }
        )
    result = {
        "schema_version": 1,
        "protocol_id": "GOAL-COPILOT-2B",
        "status": "ACCEPT" if all_pass else "REJECT",
        "evidence_role": payload["evidence_role"],
        "schedule_results": schedules,
        "claim_ceiling": "symbolic_consumed_task_noise_robust_search_signal_only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(canonical(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_locked(
        args.candidate.resolve(),
        args.envelope.resolve(),
        args.key.resolve(),
        args.output.resolve(),
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
