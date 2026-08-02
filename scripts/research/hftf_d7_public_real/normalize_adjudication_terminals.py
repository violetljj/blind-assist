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


def normalize(path: Path, *, expected_count: int) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"adjudication output is missing: {path}")
    rows: list[dict[str, Any]] = []
    rebound = 0
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
            decision = row.get("adjudication_decision")
            if decision == "NOT_ADMIT":
                if row.get("admission_status") != "NOT_ADMITTED" or row.get("event_bucket") != "NOT_EVALUABLE":
                    raise ContractError(f"unsafe NOT_ADMIT terminal at {path}:{line_number}")
                row["adjudication_decision"] = "NOT_EVALUABLE"
                row["normalization_note"] = "NOT_ADMIT_REBOUND_TO_FROZEN_NOT_EVALUABLE_TERMINAL"
                rebound += 1
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
    return {"path": str(path), "rows": len(rows), "rebound_not_admit": rebound, "status": "NORMALIZED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--expected-count", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(normalize(Path(args.path), expected_count=args.expected_count), ensure_ascii=False, sort_keys=True))
