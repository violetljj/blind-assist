"""Independent structural and scoring validator for a RISKSEG-R1 P0 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .core import (
    MODEL_ARMS,
    ORACLE_ARM,
    canonical_sha256,
    fold_assignments,
    nested_oof_score,
    read_object,
    sha256_file,
    timing_against_yolo,
    write_object,
)
from .run_audit import decide, old_adapter_metrics, verify_bound_file, yolo_event_scores


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    report_path = args.report.resolve()
    contract = read_object(contract_path)
    report = read_object(report_path)
    root = contract_path.parents[3]
    if report["contract_sha256"] != sha256_file(contract_path):
        raise ValueError("contract SHA mismatch")
    manifest_path = verify_bound_file(root, contract["frozen_manifest"])
    manifest = read_object(manifest_path)
    assignments = fold_assignments(manifest, contract)
    if canonical_sha256(assignments) != report["fold_assignment_sha256"]:
        raise ValueError("fold assignment mismatch")
    trace_path = report_path.parent / report["feature_trace"]["path"]
    if sha256_file(trace_path) != report["feature_trace"]["sha256"]:
        raise ValueError("feature trace SHA mismatch")
    rows = read_jsonl(trace_path)
    if len(rows) != 1920 * 4:
        raise ValueError(f"feature trace row count {len(rows)} != 7680")
    identities = {
        (row["arm"], row["parent_event_id"], int(row["frame_index"]))
        for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError("duplicate frame identity")
    yolo_path = verify_bound_file(root, contract["yolo_reference_report"])
    yolo = yolo_event_scores(read_object(yolo_path))
    old = old_adapter_metrics(root, contract)
    nested = {
        arm: nested_oof_score(
            arm=arm,
            manifest=manifest,
            frame_rows=rows,
            yolo_events=yolo,
            contract=contract,
        )
        for arm in (*MODEL_ARMS, ORACLE_ARM)
    }
    for arm, scored in nested.items():
        scored["timing_against_yolo"] = timing_against_yolo(
            scored["oof_event_scores"], yolo
        )
    if nested != report["nested_oof"]:
        raise ValueError("independent nested OOF recomputation mismatch")
    decision = decide(nested, old, contract)
    if decision != report["decision"]:
        raise ValueError("independent terminal recomputation mismatch")
    validation = {
        "schema_version": "blindassist.riskseg_r1.p0_validation.v1",
        "status": "PASS",
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "trace_sha256": sha256_file(trace_path),
        "row_count": len(rows),
        "terminal": decision["terminal"],
    }
    write_object(args.output.resolve(), validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

