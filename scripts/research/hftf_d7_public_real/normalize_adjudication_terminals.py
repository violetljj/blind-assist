#!/usr/bin/env python3
"""Rebind a conservative non-admission terminal to the frozen enum.

Some adjudicators use the readable value ``NOT_ADMIT``.  The D7 frozen
adjudication contract represents a non-admitted, unevaluable terminal as
``NOT_EVALUABLE``.  This utility changes only that exact safe combination and
refuses any ambiguous value; the caller should preserve the raw output first.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pipeline import ContractError


def normalize(
    path: Path,
    *,
    expected_count: int,
    canonicalize_legacy_final: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"adjudication output is missing: {path}")
    rows: list[dict[str, Any]] = []
    rebound = 0
    rebound_not_admitted_decision = 0
    rebound_swapped_terminal = 0
    decision_aliases = 0
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"adjudication row is not an object: {path}:{line_number}")
            if canonicalize_legacy_final:
                if "decision" in row and "adjudication_decision" in row:
                    raise ContractError(f"both decision keys are present: {path}:{line_number}")
                if "decision" in row:
                    row["adjudication_decision"] = row.pop("decision")
                    decision_aliases += 1
                row["schema"] = "hftf_d7_public_real_completed_adjudication_v1"
                row["record_kind"] = "COMPLETED_ADJUDICATION"
            decision = row.get("adjudication_decision")
            if decision == "NOT_ADMIT":
                if row.get("admission_status") != "NOT_ADMITTED" or row.get("event_bucket") != "NOT_EVALUABLE":
                    raise ContractError(f"unsafe NOT_ADMIT terminal at {path}:{line_number}")
                row["adjudication_decision"] = "NOT_EVALUABLE"
                row["normalization_note"] = "NOT_ADMIT_REBOUND_TO_FROZEN_NOT_EVALUABLE_TERMINAL"
                rebound += 1
            elif (
                decision == "NOT_ADMITTED"
                and row.get("admission_status") == "NOT_EVALUABLE"
                and row.get("event_bucket") == "NOT_EVALUABLE"
            ):
                # A legacy writer occasionally swapped the terminal enum and
                # admission-status fields.  This is safe only for the fully
                # unevaluable, non-admitted terminal; never repair a bucket
                # carrying a class or phase evidence here.
                row["adjudication_decision"] = "NOT_EVALUABLE"
                row["admission_status"] = "NOT_ADMITTED"
                row["normalization_note"] = "SWAPPED_NOT_ADMITTED_NOT_EVALUABLE_TERMINAL_REBOUND"
                rebound_swapped_terminal += 1
            elif decision == "NOT_ADMITTED":
                if row.get("event_bucket") != "NOT_EVALUABLE" or row.get("admission_status") not in (None, "NOT_ADMITTED"):
                    raise ContractError(f"unsafe NOT_ADMITTED decision at {path}:{line_number}")
                row["adjudication_decision"] = "NOT_EVALUABLE"
                row["admission_status"] = "NOT_ADMITTED"
                row["normalization_note"] = "NOT_ADMITTED_DECISION_REBOUND_TO_FROZEN_NOT_EVALUABLE_TERMINAL"
                rebound_not_admitted_decision += 1
            if row.get("model_output_visible") is not False:
                raise ContractError(f"adjudication model visibility is not false: {path}:{line_number}")
            rows.append(row)
    if len(rows) != expected_count:
        raise ContractError(f"expected {expected_count} rows, got {len(rows)}: {path}")
    values = [str(row.get("candidate_id") or "") for row in rows]
    if not all(values) or len(set(values)) != len(values):
        raise ContractError(f"candidate_id is not unique and complete: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return {
        "path": str(path),
        "rows": len(rows),
        "rebound_not_admit": rebound,
        "rebound_not_admitted_decision": rebound_not_admitted_decision,
        "rebound_swapped_terminal": rebound_swapped_terminal,
        "decision_aliases": decision_aliases,
        "status": "NORMALIZED",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument(
        "--canonicalize-legacy-final",
        action="store_true",
        help="map legacy decision/schema/record-kind fields to the frozen adjudication contract",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(
        normalize(
            Path(args.path),
            expected_count=args.expected_count,
            canonicalize_legacy_final=args.canonicalize_legacy_final,
        ),
        ensure_ascii=False,
        sort_keys=True,
    ))
