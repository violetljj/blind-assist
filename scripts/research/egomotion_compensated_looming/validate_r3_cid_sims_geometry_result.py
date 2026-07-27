"""Independently validate R3 outcome aggregation without reopening the payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "RCLE_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R3_CID_SIMS"
ADMITTED = "REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID"
HOLD = "HOLD_NO_QUALIFYING_APPROACH_WINDOW / VALID"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(rows: list[dict[str, Any]], field: str) -> float | None:
    return float(np.median([float(row[field]) for row in rows])) if rows else None


def same(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("R3_VALIDATION_OUTPUT_EXISTS")
    result = json.loads(args.result.read_text(encoding="utf-8"))
    claim = json.loads(args.claim.read_text(encoding="utf-8"))
    ledger_bytes = args.ledger.read_bytes()
    rows = [
        json.loads(line)
        for line in ledger_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    if result.get("protocol_id") != PROTOCOL_ID or claim.get("protocol_id") != PROTOCOL_ID:
        errors.append("PROTOCOL")
    if result.get("pair_ledger_sha256") != hashlib.sha256(ledger_bytes).hexdigest():
        errors.append("LEDGER_HASH")
    recomputed: list[dict[str, Any]] = []
    for window in result.get("windows", []):
        index = int(window["window_index"])
        selected = [row for row in rows if int(row["window_index"]) == index]
        evaluable = [row for row in selected if row.get("evaluable") is True]
        coverage = len(evaluable) / len(selected) if selected else 0.0
        signed = median(evaluable, "median_signed_radial_expansion_per_s")
        positive = median(evaluable, "radial_expansion_positive_fraction")
        admitted = bool(
            coverage >= 0.80
            and len(evaluable) >= 12
            and signed is not None
            and signed >= 0.05
            and positive is not None
            and positive >= 0.75
        )
        if (
            window.get("sampled_pair_count") != len(selected)
            or window.get("evaluable_pair_count") != len(evaluable)
            or not same(window.get("coverage"), coverage)
            or not same(window.get("median_signed_radial_expansion_per_s"), signed)
            or not same(window.get("median_radial_expansion_positive_fraction"), positive)
            or window.get("admitted") is not admitted
        ):
            errors.append(f"WINDOW_{index}")
        recomputed.append({**window, "admitted": admitted})
    passing = [window for window in recomputed if window["admitted"]]
    selected = passing[0] if passing else None
    expected_terminal = ADMITTED if selected else HOLD
    if result.get("terminal") != expected_terminal:
        errors.append("TERMINAL")
    if result.get("admitted_window_count") != len(passing):
        errors.append("ADMITTED_COUNT")
    if result.get("selected_earliest_admitted_window") != selected:
        errors.append("EARLIEST_SELECTION")
    validation = {
        "schema_version": "rcle.r3.geometry_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "valid": not errors,
        "errors": errors,
        "terminal_recomputed": expected_terminal,
        "admitted_window_count_recomputed": len(passing),
        "earliest_admitted_window_index": selected["window_index"] if selected else None,
        "result_sha256": sha256(args.result),
        "pair_ledger_sha256": sha256(args.ledger),
        "claim_sha256": sha256(args.claim),
        "payload_reopened": False,
        "rgb_pixels_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
