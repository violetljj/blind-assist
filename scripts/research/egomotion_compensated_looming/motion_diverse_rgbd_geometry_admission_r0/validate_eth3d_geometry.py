"""Independently validate the frozen ETH3D geometry admission aggregates."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


REL = Decimal("1e-12")
ABS = Decimal("1e-15")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def close(actual: Any, expected: Decimal) -> bool:
    observed = Decimal(str(actual))
    return abs(observed - expected) <= max(ABS, REL * abs(expected))


def longest(rows: list[dict[str, Any]], band: str) -> tuple[int, Decimal]:
    best_count, count = 0, 0
    best_duration = Decimal("0")
    start: Decimal | None = None
    previous_index: int | None = None
    for row in rows:
        contiguous = previous_index is not None and int(row["pair_index"]) == previous_index + 1
        if row.get("geometry_band") == band:
            if not contiguous or count == 0:
                count = 0
                start = Decimal(str(row["previous_timestamp_s"]))
            count += 1
            duration = Decimal(str(row["current_timestamp_s"])) - start
            if count > best_count or (count == best_count and duration > best_duration):
                best_count, best_duration = count, duration
        else:
            count, start = 0, None
        previous_index = int(row["pair_index"])
    return best_count, best_duration


def recompute_role(window: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = int(window["pair_count"])
    if len(rows) != denominator or [int(row["pair_index"]) for row in rows] != list(range(denominator)):
        raise ValueError(f"PAIR_IDENTITY:{window['window_index']}")
    evaluable = [row for row in rows if row.get("geometry_evaluable") is True]
    for row in evaluable:
        signed = Decimal(str(row["geometry_signed_radial_expansion_per_s"]))
        expected_band = (
            "BELOW_TRIGGER_REFERENCE"
            if signed < Decimal("0.01")
            else "WEAK_POSITIVE_RADIAL"
            if signed < Decimal("0.05")
            else "POSITIVE_APPROACH_GEOMETRY"
        )
        if row.get("geometry_band") != expected_band:
            raise ValueError(f"BAND_MISMATCH:{window['window_index']}:{row['pair_index']}")
    counts = Counter(row["geometry_band"] for row in evaluable)
    coverage = Decimal(len(evaluable)) / Decimal(denominator)
    positive = Decimal(counts["POSITIVE_APPROACH_GEOMETRY"]) / Decimal(denominator)
    below = Decimal(counts["BELOW_TRIGGER_REFERENCE"]) / Decimal(denominator)
    positive_count, positive_duration = longest(rows, "POSITIVE_APPROACH_GEOMETRY")
    below_count, below_duration = longest(rows, "BELOW_TRIGGER_REFERENCE")
    positive_ok = coverage >= Decimal("0.8") and positive >= Decimal("0.8") and positive_duration >= Decimal("5")
    below_ok = coverage >= Decimal("0.8") and below >= Decimal("0.8") and below_duration >= Decimal("5")
    if positive_ok and below_ok:
        raise ValueError(f"ROLE_OVERLAP:{window['window_index']}")
    role = (
        "POSITIVE_APPROACH_WINDOW"
        if positive_ok
        else "BELOW_TRIGGER_REFERENCE_WINDOW"
        if below_ok
        else "AMBIGUOUS_OR_INELIGIBLE"
    )
    return {
        "role": role,
        "coverage": coverage,
        "positive": positive,
        "below": below,
        "positive_count": positive_count,
        "positive_duration": positive_duration,
        "below_count": below_count,
        "below_duration": below_duration,
    }


def expected_selection(summaries: list[dict[str, Any]]) -> list[int]:
    positive = [row for row in summaries if row["role"] == "POSITIVE_APPROACH_WINDOW"]
    below = [row for row in summaries if row["role"] == "BELOW_TRIGGER_REFERENCE_WINDOW"]
    feasible: list[list[dict[str, Any]]] = []
    for positive_rows in itertools.combinations(positive, 2):
        for below_rows in itertools.combinations(below, 2):
            rows = sorted((*positive_rows, *below_rows), key=lambda row: int(row["window_index"]))
            starts = [Decimal(str(row["start_timestamp_s"])) for row in rows]
            if all(right - left >= Decimal("20") for left, right in zip(starts, starts[1:])):
                feasible.append(rows)
    chosen = min(feasible, key=lambda rows: tuple(int(row["window_index"]) for row in rows), default=[])
    return [int(row["window_index"]) for row in chosen]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze = load(args.window_freeze.resolve())
    output = args.output_dir.resolve()
    ledger_path = output / "geometry_pair_ledger.jsonl"
    ledger_bytes = ledger_path.read_bytes()
    ledger = [json.loads(line) for line in ledger_bytes.splitlines()]
    selection = load(output / "geometry_selection.json")
    result = load(output / "result.json")
    if selection["geometry_pair_ledger_sha256"] != hashlib.sha256(ledger_bytes).hexdigest():
        raise ValueError("LEDGER_SHA256")
    summaries = {int(row["window_index"]): row for row in selection["window_summaries"]}
    recomputed = []
    for window in freeze["windows"]:
        index = int(window["window_index"])
        rows = [row for row in ledger if int(row["window_index"]) == index]
        check = recompute_role(window, rows)
        summary = summaries[index]
        exact = {
            "role": check["role"],
            "longest_positive_run_pair_count": check["positive_count"],
            "longest_below_run_pair_count": check["below_count"],
        }
        if any(summary[key] != value for key, value in exact.items()):
            raise ValueError(f"SUMMARY_EXACT:{index}")
        numeric = {
            "geometry_pair_coverage_fixed_denominator": check["coverage"],
            "positive_fraction_fixed_denominator": check["positive"],
            "below_fraction_fixed_denominator": check["below"],
            "longest_positive_run_duration_s": check["positive_duration"],
            "longest_below_run_duration_s": check["below_duration"],
        }
        if any(not close(summary[key], value) for key, value in numeric.items()):
            raise ValueError(f"SUMMARY_NUMERIC:{index}")
        recomputed.append(summary)
    selected = expected_selection(recomputed)
    reported = [int(row["window_index"]) for row in selection["selected_windows"]]
    terminal = "GEOMETRY_ADMITTED_FOUR_WINDOWS_FROZEN / VALID" if selected else "NOT_EVALUABLE_NO_RGB_NO_REPLACEMENT / VALID"
    if (
        selected != reported
        or result["selected_window_indices"] != selected
        or selection["terminal"] != terminal
        or result["terminal"] != terminal
        or selection["workers"] != 8
        or selection["rgb_bytes_accessed"] != 0
        or result["rgb_bytes_accessed"] != 0
        or selection["candidate_replacement"] is not False
        or result["candidate_replacement"] is not False
        or selection["post_outcome_windows_added"] != 0
    ):
        raise ValueError("TERMINAL_OR_BOUNDARY")
    print(json.dumps({"status": "PASS", "terminal": terminal, "selected_window_indices": selected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
