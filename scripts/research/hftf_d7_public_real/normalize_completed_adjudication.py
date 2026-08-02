#!/usr/bin/env python3
"""Normalize the final adjudicator's legacy ``decision`` key.

This changes only the frozen field name from ``decision`` to
``adjudication_decision``.  It never changes the decision value, admission
status, event bucket, phase intervals, or notes.
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
            if "decision" in row:
                if "adjudication_decision" in row:
                    raise ContractError(f"both adjudication decision keys are present: {path}:{line_number}")
                row["adjudication_decision"] = row.pop("decision")
                row["normalization_note"] = "ADJUDICATION_DECISION_KEY_REBOUND"
            if row.get("model_output_visible") is not False:
                raise ContractError(f"adjudication model visibility is not false: {path}:{line_number}")
            rows.append(row)
    if len(rows) != expected_count:
        raise ContractError(f"expected {expected_count} rows, got {len(rows)}: {path}")
    for key in ("candidate_id", "event_id"):
        values = [str(row.get(key) or "") for row in rows]
        if not all(values) or len(set(values)) != len(values):
            raise ContractError(f"{key} is not unique and complete: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return {"path": str(path), "rows": len(rows), "status": "NORMALIZED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--expected-count", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(normalize(Path(args.path), expected_count=args.expected_count), ensure_ascii=False, sort_keys=True))
