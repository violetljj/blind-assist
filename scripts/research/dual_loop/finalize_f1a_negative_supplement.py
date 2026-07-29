#!/usr/bin/env python3
"""Merge an isolated F-1A negative-category supplement with immutable R0 labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from compare_f1a_label_reviews import ReviewError
from finalize_f1a_label_repair import (
    assign_records,
    materialize_items,
    sha256_file,
    validate_cross_item_consistency,
    write_json,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_combined_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in records if row["item_kind"] == "positive_event"]
    negatives = [row for row in records if row["item_kind"] == "negative_window"]
    category_counts = Counter(row["negative_type"] for row in negatives)
    decision_ids = {
        row["input_id"] for row in records if row.get("role") == "DECISION"
    }
    decision_with_positive = {
        row["input_id"]
        for row in positives
        if row.get("role") == "DECISION"
    }
    decision_with_negative = {
        row["input_id"]
        for row in negatives
        if row.get("role") == "DECISION"
    }
    development_ids = {
        row["input_id"]
        for row in records
        if str(row.get("role", "")).startswith("DEVELOPMENT")
    }
    checks = {
        "independent_capture_sessions": len(
            {row["parent_capture_id"] for row in records}
        )
        >= 3,
        "positive_events": len(positives) >= 6,
        "positive_sessions": len({row["session_id"] for row in positives}) >= 2,
        "negative_windows": len(negatives) >= 12,
        "negative_categories": sum(count >= 2 for count in category_counts.values())
        >= 4,
        "development_sessions": len(development_ids) >= 1,
        "decision_sessions": len(decision_ids) >= 2,
        "each_decision_has_positive": decision_with_positive == decision_ids,
        "each_decision_has_negative": decision_with_negative == decision_ids,
    }
    return {
        "terminal": "READY" if all(checks.values()) else "HOLD_DATA",
        "checks": checks,
        "counts": {
            "independent_capture_sessions": len(
                {row["parent_capture_id"] for row in records}
            ),
            "positive_events": len(positives),
            "positive_sessions": len({row["session_id"] for row in positives}),
            "negative_windows": len(negatives),
            "negative_category_counts": dict(sorted(category_counts.items())),
            "development_sessions": len(development_ids),
            "decision_sessions": len(decision_ids),
            "decision_with_positive": sorted(decision_with_positive),
            "decision_with_negative": sorted(decision_with_negative),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--r0-ledger", type=Path, required=True)
    parser.add_argument("--r0-validation", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ledger.exists() or args.validation.exists():
        raise FileExistsError("formal supplement output already exists")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    adjudication = (
        json.loads(args.adjudication.read_text(encoding="utf-8"))
        if args.adjudication
        else None
    )
    r0_validation = json.loads(args.r0_validation.read_text(encoding="utf-8"))
    if sha256_file(args.r0_validation) != spec["predecessor_result"]["sha256"]:
        raise ReviewError("R0 validation hash differs from frozen supplement spec")
    if sha256_file(args.r0_ledger) != r0_validation["ledger_sha256"]:
        raise ReviewError("R0 ledger no longer matches its signed validation")

    comparison_sha256 = sha256_file(args.comparison)
    items, quarantines = materialize_items(
        comparison,
        adjudication=adjudication,
        spec=spec,
        manifest=manifest,
        comparison_sha256=comparison_sha256,
    )
    if any(item["item_kind"] != "negative_window" for item in items):
        raise ReviewError("supplement attempted to add non-negative labels")
    validate_cross_item_consistency(items)
    supplement_records = assign_records(items, manifest=manifest)
    for index, record in enumerate(supplement_records, start=1):
        record.pop("negative_window_id", None)
        record["negative_window_id"] = f"F1A-R1-N-{index:03d}"
        record["supplement_protocol_id"] = spec["protocol_id"]

    r0_records = load_jsonl(args.r0_ledger)
    combined = r0_records + supplement_records
    gate = evaluate_combined_gate(combined)
    supplement_pass = (
        len(supplement_records)
        >= int(spec["supplement_gate"]["accepted_new_negative_windows_min"])
        and gate["checks"]["negative_categories"]
    )
    if not supplement_pass:
        gate["terminal"] = "HOLD_DATA"

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in combined
        ),
        encoding="utf-8",
    )
    adjudication_sha256 = (
        sha256_file(args.adjudication) if args.adjudication else None
    )
    validation = {
        "schema": "blindassist_dual_loop_f1a_negative_category_supplement_validation_v1",
        "protocol_id": spec["protocol_id"],
        "spec_sha256": sha256_file(args.spec),
        "review_bundle_subject_sha256": manifest["bundle_subject_sha256"],
        "comparison_sha256": comparison_sha256,
        "adjudication_sha256": adjudication_sha256,
        "r0_validation_sha256": sha256_file(args.r0_validation),
        "r0_ledger_sha256": sha256_file(args.r0_ledger),
        "combined_ledger_sha256": sha256_file(args.ledger),
        "accepted_supplement_count": len(supplement_records),
        "quarantine_count": len(quarantines),
        "quarantines": quarantines,
        "candidate_output_visibility": False,
        "data_protocol_status": "VALID",
        **gate,
    }
    write_json(args.validation, validation)
    print(
        json.dumps(
            {
                "status": "PASS",
                "terminal": validation["terminal"],
                "accepted_supplement_count": len(supplement_records),
                "counts": validation["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
