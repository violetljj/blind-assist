"""Independent validator for the single temporal-confirmation revision."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .attribution import load_object, read_jsonl, sha256


THRESHOLD = 0.01


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


def recompute_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["window_id"]].append(row)
    windows = [summarize_window(selected) for selected in grouped.values()]
    role_windows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        role_windows[row["role"]].append(row)
    roles: dict[str, Any] = {}
    for role, selected in role_windows.items():
        roles[role] = {
            "window_count": len(selected),
            "old_trigger_coverage_fixed_denominator": float(
                median(
                    row["old_trigger_coverage_fixed_denominator"] for row in selected
                )
            ),
            "revised_trigger_coverage_fixed_denominator": float(
                median(
                    row["revised_trigger_coverage_fixed_denominator"]
                    for row in selected
                )
            ),
        }
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
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    contract = load_object(args.contract.resolve())
    evidence = args.evidence_dir.resolve()
    result = load_object(evidence / "result.json")
    revised_path = evidence / "revised_pair_ledger.jsonl"
    revised = read_jsonl(revised_path)
    old_entry = contract["inputs"]["old_pair_ledger"]
    old_path = repo / old_entry["path"]
    old = read_jsonl(old_path)
    errors: list[str] = []
    for entry in contract["inputs"].values():
        path = repo / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            errors.append(f"BOUND_INPUT_IDENTITY:{entry['path']}")
    if len(old) != 967 or len(revised) != 967:
        errors.append("PAIR_DENOMINATOR")
    if result.get("revised_pair_ledger_sha256") != sha256(revised_path):
        errors.append("REVISED_LEDGER_SHA256")
    streak = 0
    window_id: str | None = None
    for old_row, new_row in zip(old, revised):
        identity = (
            old_row["window_id"],
            old_row["role"],
            old_row["pair_index"],
            old_row["previous_timestamp_s"],
            old_row["current_timestamp_s"],
            old_row["evaluable"],
            old_row.get("compensated_expansion_median_per_s"),
        )
        revised_identity = (
            new_row["window_id"],
            new_row["role"],
            new_row["pair_index"],
            new_row["previous_timestamp_s"],
            new_row["current_timestamp_s"],
            new_row["evaluable"],
            new_row.get("compensated_expansion_median_per_s"),
        )
        if identity != revised_identity:
            errors.append(f"PAIR_IDENTITY:{old_row['window_id']}:{old_row['pair_index']}")
            continue
        if new_row["window_id"] != window_id:
            window_id = new_row["window_id"]
            streak = 0
        above = bool(
            new_row["evaluable"] is True
            and float(new_row["compensated_expansion_median_per_s"]) > THRESHOLD
        )
        streak = streak + 1 if above else 0
        expected = streak >= 3
        if (
            new_row["consecutive_above_threshold_pair_count"] != streak
            or new_row["revised_trigger"] is not expected
        ):
            errors.append(f"REVISION_PARITY:{window_id}:{new_row['pair_index']}")
    recomputed = recompute_result(revised)
    for key in (
        "window_summaries",
        "role_aggregates",
        "gates",
        "all_gates_pass",
        "scientific_outcome",
    ):
        if result.get(key) != recomputed[key]:
            errors.append(f"RESULT_MISMATCH:{key}")
    if result.get("candidate_count") != 1:
        errors.append("CANDIDATE_COUNT")
    receipt = {
        "schema": "rcle.low_reference.temporal_confirmation_validation.v1",
        "protocol_id": contract["protocol_id"],
        "status": "VALID" if not errors else "INVALID",
        "errors": sorted(set(errors)),
        "pair_count": len(revised),
        "recomputed_scientific_outcome": recomputed["scientific_outcome"],
        "recomputed_gates": recomputed["gates"],
    }
    if args.output.exists():
        raise FileExistsError("VALIDATION_OUTPUT_EXISTS")
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "errors": receipt["errors"]}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
