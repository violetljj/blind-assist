"""Apply the single frozen causal three-pair confirmation revision."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .attribution import load_object, read_jsonl, sha256, write_exclusive


THRESHOLD = 0.01
REQUIRED_CONSECUTIVE_PAIRS = 3


def apply_confirmation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    revised: list[dict[str, Any]] = []
    streak = 0
    current_window: str | None = None
    for row in rows:
        window_id = str(row["window_id"])
        if window_id != current_window:
            current_window = window_id
            streak = 0
        above = bool(
            row.get("evaluable") is True
            and float(row["compensated_expansion_median_per_s"]) > THRESHOLD
        )
        streak = streak + 1 if above else 0
        revised_trigger = bool(streak >= REQUIRED_CONSECUTIVE_PAIRS)
        revised.append(
            {
                "window_id": window_id,
                "role": row["role"],
                "pair_index": row["pair_index"],
                "previous_timestamp_s": row["previous_timestamp_s"],
                "current_timestamp_s": row["current_timestamp_s"],
                "dt_s": row["dt_s"],
                "evaluable": row["evaluable"],
                "reason": row["reason"],
                "compensated_expansion_median_per_s": row.get(
                    "compensated_expansion_median_per_s"
                ),
                "threshold_per_s": THRESHOLD,
                "old_trigger": row["trigger"],
                "consecutive_above_threshold_pair_count": streak,
                "revised_trigger": revised_trigger,
            }
        )
    return revised


def longest_trigger(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    best_count = count = 0
    best_duration = 0.0
    start: float | None = None
    for row in rows:
        if row[field] is True:
            if count == 0:
                start = float(row["previous_timestamp_s"])
            count += 1
            duration = float(row["current_timestamp_s"]) - float(start)
            if count > best_count or (count == best_count and duration > best_duration):
                best_count, best_duration = count, duration
        else:
            count = 0
            start = None
    return {"pair_count": best_count, "duration_s": best_duration}


def summarize_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    old_count = sum(row["old_trigger"] is True for row in rows)
    revised_count = sum(row["revised_trigger"] is True for row in rows)
    old_first = next(
        (float(row["current_timestamp_s"]) for row in rows if row["old_trigger"] is True),
        None,
    )
    revised_first = next(
        (
            float(row["current_timestamp_s"])
            for row in rows
            if row["revised_trigger"] is True
        ),
        None,
    )
    return {
        "window_id": rows[0]["window_id"],
        "role": rows[0]["role"],
        "candidate_pair_count": len(rows),
        "evaluable_pair_count": sum(row["evaluable"] is True for row in rows),
        "old_trigger_count": old_count,
        "revised_trigger_count": revised_count,
        "old_trigger_coverage_fixed_denominator": old_count / len(rows),
        "revised_trigger_coverage_fixed_denominator": revised_count / len(rows),
        "trigger_retention": revised_count / old_count if old_count else None,
        "first_trigger_delay_s": (
            revised_first - old_first
            if old_first is not None and revised_first is not None
            else None
        ),
        "old_longest_trigger_run": longest_trigger(rows, "old_trigger"),
        "revised_longest_trigger_run": longest_trigger(rows, "revised_trigger"),
    }


def aggregate_roles(window_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_summaries:
        grouped[row["role"]].append(row)
    result: dict[str, Any] = {}
    for role, selected in grouped.items():
        result[role] = {
            "window_count": len(selected),
            "old_trigger_coverage_fixed_denominator": float(
                np.median(
                    [
                        row["old_trigger_coverage_fixed_denominator"]
                        for row in selected
                    ]
                )
            ),
            "revised_trigger_coverage_fixed_denominator": float(
                np.median(
                    [
                        row["revised_trigger_coverage_fixed_denominator"]
                        for row in selected
                    ]
                )
            ),
        }
    return result


def build_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_window[row["window_id"]].append(row)
    windows = [summarize_window(selected) for selected in by_window.values()]
    roles = aggregate_roles(windows)
    positive = roles["POSITIVE_APPROACH_WINDOW"]
    below = roles["BELOW_TRIGGER_REFERENCE_WINDOW"]
    positive_retention = (
        positive["revised_trigger_coverage_fixed_denominator"]
        / positive["old_trigger_coverage_fixed_denominator"]
    )
    below_reduction = 1.0 - (
        below["revised_trigger_coverage_fixed_denominator"]
        / below["old_trigger_coverage_fixed_denominator"]
    )
    positive_delays = [
        row["first_trigger_delay_s"]
        for row in windows
        if row["role"] == "POSITIVE_APPROACH_WINDOW"
    ]
    maximum_delay = (
        max(float(value) for value in positive_delays if value is not None)
        if all(value is not None for value in positive_delays)
        else None
    )
    direction = bool(
        positive["revised_trigger_coverage_fixed_denominator"]
        > below["revised_trigger_coverage_fixed_denominator"]
    )
    gates = {
        "below_relative_trigger_reduction": {
            "value": below_reduction,
            "operator": ">=",
            "threshold": 0.3,
            "pass": below_reduction >= 0.3,
        },
        "positive_trigger_retention": {
            "value": positive_retention,
            "operator": ">=",
            "threshold": 0.9,
            "pass": positive_retention >= 0.9,
        },
        "maximum_positive_first_trigger_delay_s": {
            "value": maximum_delay,
            "operator": "<=",
            "threshold": 0.25,
            "pass": maximum_delay is not None and maximum_delay <= 0.25,
        },
        "positive_direction_pass": {
            "value": 1 if direction else 0,
            "operator": "==",
            "threshold": 1,
            "pass": direction,
        },
    }
    ready = all(gate["pass"] is True for gate in gates.values())
    return {
        "window_summaries": windows,
        "role_aggregates": roles,
        "gates": gates,
        "all_gates_pass": ready,
        "scientific_outcome": (
            "IMPLEMENTATION_READY_FOR_CONFIRMATION"
            if ready
            else "IMPLEMENTATION_NOT_READY"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    contract = load_object(args.contract.resolve())
    if contract["protocol_id"] != "RCLE_LOW_REFERENCE_TEMPORAL_CONFIRMATION_R1":
        raise ValueError("PROTOCOL_ID")
    if contract["single_revision"] != {
        "id": "CAUSAL_THREE_PAIR_CONFIRMATION_R1",
        "candidate_count": 1,
        "input_signal": "compensated_expansion_median_per_s",
        "threshold_per_s": 0.01,
        "confirmation_rule": "revised_trigger is true only when the current pair and the two immediately preceding pairs in the same window are evaluable and each exceeds 0.01/s",
        "reset_rule": "Any abstention, window boundary, or pair at or below threshold resets the consecutive-pair count to zero.",
        "causal": True,
        "lookahead_pairs": 0,
    }:
        raise ValueError("SINGLE_REVISION_DRIFT")
    bound: dict[str, Path] = {}
    for name, entry in contract["inputs"].items():
        path = repo / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"BOUND_INPUT_IDENTITY:{entry['path']}")
        bound[name] = path
    attribution_validation = load_object(bound["attribution_validation"])
    if (
        attribution_validation.get("status") != "VALID"
        or attribution_validation.get("errors") != []
    ):
        raise ValueError("ATTRIBUTION_NOT_VALID")
    old_rows = read_jsonl(bound["old_pair_ledger"])
    old_result = load_object(bound["old_result"])
    if len(old_rows) != 967 or old_result["rgb_pair_ledger_sha256"] != sha256(
        bound["old_pair_ledger"]
    ):
        raise ValueError("OLD_LEDGER_IDENTITY")
    revised_rows = apply_confirmation(old_rows)
    if len(revised_rows) != 967:
        raise ValueError("REVISED_PAIR_DENOMINATOR")
    computed = build_result(revised_rows)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("OUTPUT_DIRECTORY_EXISTS")
    ledger_sha = write_exclusive(
        output / "revised_pair_ledger.jsonl", revised_rows, jsonl=True
    )
    result = {
        "schema": "rcle.low_reference.temporal_confirmation_result.v1",
        "protocol_id": contract["protocol_id"],
        "execution_validity": "VALID",
        "revision_id": contract["single_revision"]["id"],
        "candidate_count": 1,
        "old_pair_ledger_sha256": sha256(bound["old_pair_ledger"]),
        "revised_pair_ledger_sha256": ledger_sha,
        "threshold_changed": False,
        "window_changed": False,
        "pair_identity_changed": False,
        "lookahead_pairs": 0,
        **computed,
        "authority": {
            "development": True,
            "confirmation": False,
            "external_validation_decision_only": computed["all_gates_pass"],
            "android": False,
            "product": False,
            "safety": False,
        },
    }
    write_exclusive(output / "result.json", result)
    print(
        json.dumps(
            {
                "scientific_outcome": result["scientific_outcome"],
                "gates": result["gates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
