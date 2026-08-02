#!/usr/bin/env python3
"""Normalize legacy bucket/decision keys in a completed review artifact."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from pipeline import ContractError, load_jsonl


DECISIONS = {"SUPPORT", "REJECT", "NOT_EVALUABLE", "ESCALATE"}
REVIEW_ROLES = {
    "RGB_REVIEWER_A",
    "RGB_REVIEWER_B",
    "RGB_REVIEWER_C",
    "GEOMETRY_EVIDENCE_REVIEWER",
    "COUNTEREXAMPLE_REVIEWER",
}
POSITIVE_BUCKETS = {
    "BLOCKING_BODY_POSITIVE",
    "BOUNDARY_LEVEL_CHANGE_POSITIVE",
    "HEAD_HAZARD_POSITIVE",
    "DYNAMIC_INTRUSION_POSITIVE",
}
INTERVAL_KEYS = {"pre_interval", "alertable_interval", "passed_clearance_interval", "continuous_negative_interval"}


def _interval_valid(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        return int(value.get("start_timestamp_ns")) < int(value.get("end_timestamp_ns"))
    except (TypeError, ValueError):
        return False


def _phase_valid(bucket: str, phases: object) -> bool:
    if not isinstance(phases, dict) or not set(phases).issubset(INTERVAL_KEYS):
        return False
    if bucket in POSITIVE_BUCKETS:
        return all(_interval_valid(phases.get(key)) for key in (
            "pre_interval", "alertable_interval", "passed_clearance_interval"
        ))
    return _interval_valid(phases.get("continuous_negative_interval"))


def _negative_interval_from_positive_phase_span(phases: object) -> dict[str, int] | None:
    """Re-key a contiguous negative SUPPORT phase without changing its span."""

    if not isinstance(phases, dict):
        return None
    ordered = [phases.get(key) for key in (
        "pre_interval", "alertable_interval", "passed_clearance_interval"
    )]
    if not all(_interval_valid(value) for value in ordered):
        return None
    first, second, third = ordered
    if int(first["end_timestamp_ns"]) != int(second["start_timestamp_ns"]):
        return None
    if int(second["end_timestamp_ns"]) != int(third["start_timestamp_ns"]):
        return None
    return {
        "start_timestamp_ns": int(first["start_timestamp_ns"]),
        "end_timestamp_ns": int(third["end_timestamp_ns"]),
    }


def normalize(
    path: Path,
    expected_count: int,
    default_decision: str | None,
    role: str | None = None,
    downgrade_incomplete_support: bool = False,
    normalize_negative_support_phase: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    if not path.is_file():
        raise ContractError(f"review output is missing: {path}")
    rows: list[dict[str, object]] = []
    input_ids_by_candidate: dict[str, str] = {}
    if manifest_path is not None:
        manifest_rows = load_jsonl(manifest_path)
        for manifest_row in manifest_rows:
            candidate_id = str(manifest_row.get("candidate_id") or "")
            review_input_id = str(manifest_row.get("review_input_id") or "")
            if not candidate_id or not review_input_id or candidate_id in input_ids_by_candidate:
                raise ContractError(f"invalid or duplicate review manifest identity: {manifest_path}")
            input_ids_by_candidate[candidate_id] = review_input_id
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ContractError(f"review row is not an object: {path}:{line_number}")
            if "bucket" in row:
                if "event_bucket" in row:
                    raise ContractError(f"both bucket keys are present: {path}:{line_number}")
                row["event_bucket"] = row.pop("bucket")
            if "decision" not in row:
                if default_decision is None:
                    raise ContractError(f"missing decision: {path}:{line_number}")
                row["decision"] = default_decision
            if not row.get("review_input_id") and manifest_path is not None:
                candidate_id = str(row.get("candidate_id") or "")
                review_input_id = input_ids_by_candidate.get(candidate_id)
                if review_input_id is None:
                    raise ContractError(f"candidate missing from review manifest: {candidate_id}")
                row["review_input_id"] = review_input_id
                row["normalization_note"] = "REVIEW_INPUT_ID_BOUND_FROM_IMMUTABLE_MANIFEST"
            if downgrade_incomplete_support and row.get("decision") == "SUPPORT":
                bucket = str(row.get("event_bucket") or "")
                if normalize_negative_support_phase and bucket not in POSITIVE_BUCKETS:
                    continuous = _negative_interval_from_positive_phase_span(row.get("phase_intervals"))
                    if continuous is not None:
                        row["phase_intervals"] = {"continuous_negative_interval": continuous}
                        row["normalization_note"] = "NEGATIVE_SUPPORT_PHASE_REKEYED_CONTIGUOUS_SPAN"
                if not _phase_valid(bucket, row.get("phase_intervals")):
                    row["decision"] = "NOT_EVALUABLE"
                    row["event_bucket"] = "NOT_EVALUABLE"
                    row["phase_intervals"] = None
                    row["normalization_note"] = "SUPPORT_DOWNGRADED_MISSING_COMPLETE_PHASE_CONTRACT"
            if row.get("decision") not in DECISIONS:
                raise ContractError(f"invalid decision: {path}:{line_number}")
            if row.get("event_bucket") not in {
                "BLOCKING_BODY_POSITIVE",
                "BOUNDARY_LEVEL_CHANGE_POSITIVE",
                "HEAD_HAZARD_POSITIVE",
                "DYNAMIC_INTRUSION_POSITIVE",
                "PARALLEL_STRUCTURE_NEGATIVE",
                "SIDE_OBJECT_NONBLOCKING_NEGATIVE",
                "NORMAL_WALKABLE_NEGATIVE",
                "EGOMOTION_VISUAL_HARD_NEGATIVE",
                "HEAD_NONACTIONABLE_NEGATIVE",
                "NOT_EVALUABLE",
            }:
                raise ContractError(f"invalid event_bucket: {path}:{line_number}")
            if row.get("decision") == "NOT_EVALUABLE" and row.get("event_bucket") != "NOT_EVALUABLE":
                raise ContractError(f"NOT_EVALUABLE bucket mismatch: {path}:{line_number}")
            if row.get("model_output_visible") is not False:
                raise ContractError(f"model visibility is not false: {path}:{line_number}")
            if role is not None:
                if role not in REVIEW_ROLES:
                    raise ContractError(f"unknown review role: {role}")
                expected_geometry_only = role == "GEOMETRY_EVIDENCE_REVIEWER"
                geometry_only = row.get("source_native_geometry_only")
                if geometry_only is None:
                    row["source_native_geometry_only"] = expected_geometry_only
                elif geometry_only is not expected_geometry_only:
                    raise ContractError(f"source-native role flag conflicts with role: {path}:{line_number}")
                expected_counterexample = role == "COUNTEREXAMPLE_REVIEWER"
                counterexample_done = row.get("counterexample_search_completed")
                if counterexample_done is None:
                    row["counterexample_search_completed"] = expected_counterexample
                elif counterexample_done is not expected_counterexample:
                    raise ContractError(f"counterexample role flag conflicts with role: {path}:{line_number}")
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
    return {"path": str(path), "rows": len(rows), "status": "NORMALIZED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument("--default-decision")
    parser.add_argument("--role", choices=sorted(REVIEW_ROLES))
    parser.add_argument("--downgrade-incomplete-support", action="store_true")
    parser.add_argument("--normalize-negative-support-phase", action="store_true")
    parser.add_argument("--manifest-path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(normalize(
        Path(args.path),
        args.expected_count,
        args.default_decision,
        args.role,
        args.downgrade_incomplete_support,
        args.normalize_negative_support_phase,
        Path(args.manifest_path) if args.manifest_path else None,
    ), ensure_ascii=False, sort_keys=True))
