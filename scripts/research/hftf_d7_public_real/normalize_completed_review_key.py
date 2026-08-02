#!/usr/bin/env python3
"""Normalize legacy decision-key names in an already completed review file."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pipeline import ContractError


DECISIONS = {"SUPPORT", "REJECT", "NOT_EVALUABLE", "ESCALATE"}


def normalize(path: Path, old_key: str, expected_count: int) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"review output is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ContractError(f"review row is not an object: {path}:{line_number}")
            if old_key not in row or "decision" in row:
                raise ContractError(f"unexpected decision keys at {path}:{line_number}")
            row["decision"] = row.pop(old_key)
            if row["decision"] not in DECISIONS:
                raise ContractError(f"invalid decision at {path}:{line_number}")
            if row["decision"] == "NOT_EVALUABLE" and row.get("event_bucket") != "NOT_EVALUABLE":
                raise ContractError(f"NOT_EVALUABLE row has a non-terminal bucket: {path}:{line_number}")
            if row.get("model_output_visible") is not False:
                raise ContractError(f"model visibility is not false: {path}:{line_number}")
            rows.append(row)
    if len(rows) != expected_count:
        raise ContractError(f"expected {expected_count} rows, got {len(rows)}: {path}")
    for key in ("candidate_id", "review_input_id"):
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
    return {"path": str(path), "rows": len(rows), "old_key": old_key, "status": "NORMALIZED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--old-key", required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(normalize(Path(args.path), args.old_key, args.expected_count), ensure_ascii=False, sort_keys=True))
