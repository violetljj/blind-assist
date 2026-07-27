"""Revalidate strict below-reference anchors from the burned TUM audit."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def longest_below(rows: list[dict[str, Any]]) -> Decimal:
    best = Decimal("0")
    start: Decimal | None = None
    for row in rows:
        if row.get("evaluable") is True and Decimal(str(row["median_signed_radial_expansion_per_s"])) < Decimal("0.01"):
            if start is None:
                start = Decimal(str(row["previous_rgb_timestamp"]))
            best = max(best, Decimal(str(row["current_rgb_timestamp"])) - start)
        else:
            start = None
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    amendment = load(args.amendment.resolve())
    audit_path = args.audit_result.resolve()
    audit = load(audit_path)
    required = amendment["burned_below_anchor_source"]["required_window_indices"]
    summaries = []
    for index in required:
        window = next(row for row in audit["windows"] if int(row["window_index"]) == int(index))
        rows = window["pairs"]
        denominator = int(window["candidate_pair_count"])
        evaluable = [row for row in rows if row.get("evaluable") is True]
        below = [
            row
            for row in evaluable
            if Decimal(str(row["median_signed_radial_expansion_per_s"])) < Decimal("0.01")
        ]
        coverage = Decimal(len(evaluable)) / Decimal(denominator)
        fraction = Decimal(len(below)) / Decimal(denominator)
        duration = longest_below(rows)
        admitted = coverage >= Decimal("0.8") and fraction >= Decimal("0.8") and duration >= Decimal("5")
        summaries.append(
            {
                "window_id": f"TUM_RGBD_FR2_RPY@{index}",
                "window_index": int(index),
                "candidate_pair_count": denominator,
                "evaluable_pair_count": len(evaluable),
                "coverage": float(coverage),
                "below_fraction_fixed_denominator": float(fraction),
                "longest_below_duration_s": float(duration),
                "role": "BELOW_TRIGGER_REFERENCE_WINDOW" if admitted else "NOT_ADMITTED",
            }
        )
    if any(row["role"] != "BELOW_TRIGGER_REFERENCE_WINDOW" for row in summaries):
        raise ValueError("REQUIRED_TUM_BELOW_ANCHOR_NOT_ADMITTED")
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.tum_below_anchor_receipt.v1",
        "protocol_id": amendment["protocol_id"],
        "amendment_sha256": sha(args.amendment.resolve()),
        "audit_result_sha256": sha(audit_path),
        "source_id": "TUM_RGBD_FR2_RPY",
        "window_summaries": summaries,
        "below_anchor_count": len(summaries),
        "new_holdout_authority": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"below_anchor_count": len(summaries), "windows": [row["window_id"] for row in summaries]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
