#!/usr/bin/env python3
"""Fail-closed validator for the HFTF D7 event package."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from pipeline import ContractError, load_json, sha256_file, utc_now, write_json


REVIEW_FILES = {
    "RGB_REVIEWER_A": "reviews/review_a.jsonl",
    "RGB_REVIEWER_B": "reviews/review_b.jsonl",
    "RGB_REVIEWER_C": "reviews/review_c.jsonl",
    "GEOMETRY_EVIDENCE_REVIEWER": "reviews/geometry_review.jsonl",
    "COUNTEREXAMPLE_REVIEWER": "reviews/counterexample_review.jsonl",
}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise ContractError(f"missing JSONL: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"JSONL object required: {path}:{line_number}")
            yield row


def _count_rows(path: Path) -> int:
    return sum(1 for _ in _iter_jsonl(path))


def _phase_valid(event: dict[str, Any]) -> tuple[bool, str]:
    bucket = str(event.get("event_bucket", ""))
    if bucket.endswith("POSITIVE"):
        for key in ("pre_interval", "alertable_interval", "passed_clearance_interval"):
            value = event.get(key)
            if not isinstance(value, dict) or value.get("start_timestamp_ns") is None or value.get("end_timestamp_ns") is None:
                return False, f"positive event missing {key}"
            if int(value["start_timestamp_ns"]) >= int(value["end_timestamp_ns"]):
                return False, f"invalid {key} interval"
        return True, "ok"
    if bucket.endswith("NEGATIVE"):
        value = event.get("continuous_negative_interval")
        if not isinstance(value, dict) or value.get("start_timestamp_ns") is None or value.get("end_timestamp_ns") is None:
            return False, "negative event missing continuous_negative_interval"
        if int(value["start_timestamp_ns"]) >= int(value["end_timestamp_ns"]):
            return False, "invalid continuous_negative_interval"
        return True, "ok"
    return False, "event bucket is not an admitted positive or negative bucket"


def _validate_adjudicated(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for row in _iter_jsonl(path):
        event_id = str(row.get("event_id", ""))
        if not event_id:
            errors.append("adjudicated event missing event_id")
            continue
        if event_id in seen:
            errors.append(f"duplicate adjudicated event_id: {event_id}")
        seen.add(event_id)
        if row.get("record_kind") == "ASSIGNMENT_ONLY" or row.get("decision") in {"PENDING", None}:
            errors.append(f"assignment-only row found in adjudicated_events.jsonl: {event_id}")
        if row.get("admission_status") != "ADMITTED":
            errors.append(f"event not marked ADMITTED: {event_id}")
        if row.get("model_output_visible") is True or row.get("review_model_output_visible") is True:
            errors.append(f"model output visibility drift: {event_id}")
        phase_ok, reason = _phase_valid(row)
        if not phase_ok:
            errors.append(f"{event_id}: {reason}")
        rows.append(row)
    return rows, errors


def _validate_split(path: Path, *, event_by_id: dict[str, dict[str, Any]]) -> tuple[set[str], list[str]]:
    payload = load_json(path)
    errors: list[str] = []
    if not isinstance(payload, dict):
        return set(), [f"split is not an object: {path}"]
    ids: set[str] = set()
    event_ids = payload.get("event_ids", {})
    if not isinstance(event_ids, dict):
        return set(), [f"split event_ids is not an object: {path}"]
    for role, values in event_ids.items():
        if not isinstance(values, list):
            errors.append(f"{path}: {role} is not a list")
            continue
        for event_id in values:
            event_id = str(event_id)
            if event_id in ids:
                errors.append(f"duplicate split event within {path}: {event_id}")
            ids.add(event_id)
            if event_id not in event_by_id:
                errors.append(f"split references unknown event: {path}: {event_id}")
    return ids, errors


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    required = {
        "candidate_index": root / "candidates" / "candidate_index.jsonl",
        "event_manifest": root / "manifests" / "event_manifest.jsonl",
        "review_a": root / "reviews" / "review_a.jsonl",
        "review_b": root / "reviews" / "review_b.jsonl",
        "review_c": root / "reviews" / "review_c.jsonl",
        "geometry_review": root / "reviews" / "geometry_review.jsonl",
        "counterexample_review": root / "reviews" / "counterexample_review.jsonl",
        "adjudicated_events": root / "adjudication" / "adjudicated_events.jsonl",
        "rejected_events": root / "adjudication" / "rejected_events.jsonl",
        "development_split": root / "splits" / "development_split.json",
        "confirmation_split": root / "splits" / "confirmation_split.json",
        "leave_one_dataset_out": root / "splits" / "leave_one_dataset_out_splits.json",
        "dataset_quality_report": root / "reports" / "dataset_quality_report.md",
        "class_balance_report": root / "reports" / "class_balance_report.md",
        "source_coverage_report": root / "reports" / "source_coverage_report.md",
        "duplicate_audit_report": root / "reports" / "duplicate_audit_report.md",
        "label_agreement_report": root / "reports" / "label_agreement_report.md",
        "role_isolation_report": root / "reports" / "role_isolation_report.md",
        "role_isolation_receipt": root / "manifests" / "role_isolation_receipt.json",
        "final_report": root / "reports" / "d7_final_report.md",
        "final_report_receipt": root / "manifests" / "final_report_receipt.json",
    }
    errors: list[str] = []
    for name, path in required.items():
        if not path.is_file():
            errors.append(f"missing required output: {name} -> {path}")
    if errors:
        raise ContractError("; ".join(errors))

    candidate_ids: set[str] = set()
    candidate_count = 0
    candidate_duplicate_count = 0
    session_by_candidate: dict[str, str] = {}
    dataset_counts: Counter[str] = Counter()
    for row in _iter_jsonl(required["candidate_index"]):
        candidate_count += 1
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            errors.append("candidate missing candidate_id")
            continue
        if candidate_id in candidate_ids:
            candidate_duplicate_count += 1
            errors.append(f"duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        session_by_candidate[candidate_id] = str(row.get("source_session_id", ""))
        dataset_counts[str(row.get("dataset_id", "UNKNOWN"))] += 1

    event_manifest_count = _count_rows(required["event_manifest"])
    review_counts = {role: _count_rows(root / path) for role, path in REVIEW_FILES.items()}
    assignment_only_counts: dict[str, int] = {}
    for role, relative in REVIEW_FILES.items():
        assignment_only_counts[role] = sum(1 for row in _iter_jsonl(root / relative) if row.get("record_kind") == "ASSIGNMENT_ONLY")
    adjudicated, adjudication_errors = _validate_adjudicated(required["adjudicated_events"])
    errors.extend(adjudication_errors)
    event_by_id = {str(row.get("event_id")): row for row in adjudicated}
    development_ids, split_errors_a = _validate_split(required["development_split"], event_by_id=event_by_id)
    confirmation_ids, split_errors_b = _validate_split(required["confirmation_split"], event_by_id=event_by_id)
    errors.extend(split_errors_a + split_errors_b)
    if development_ids & confirmation_ids:
        errors.append("development/confirmation event overlap")
    source_sessions_by_split: dict[str, set[str]] = {}
    for name, ids in (("development", development_ids), ("confirmation", confirmation_ids)):
        source_sessions_by_split[name] = {str(event_by_id[event_id].get("source_session_id")) for event_id in ids}
    if source_sessions_by_split["development"] & source_sessions_by_split["confirmation"]:
        errors.append("development/confirmation source-session overlap")
    source_receipts = root / "receipts" / "source_receipts.jsonl"
    receipt_count = _count_rows(source_receipts) if source_receipts.is_file() else 0
    receipt_hash_missing = 0
    for row in _iter_jsonl(source_receipts) if source_receipts.is_file() else []:
        if not row.get("source_hash"):
            receipt_hash_missing += 1
    role_isolation_path = required["role_isolation_receipt"]
    role_isolation = load_json(role_isolation_path)

    admitted_count = len(adjudicated)
    completion_checks = {
        "candidate_target_reached": candidate_count >= 50000,
        "parent_event_target_reached": admitted_count >= 10000,
        "required_review_outputs_non_assignment": all(assignment_only_counts[role] < review_counts[role] for role in REVIEW_FILES) if admitted_count else False,
        "adjudication_present": admitted_count > 0,
        "required_phase_contract": not adjudication_errors,
        "splits_session_disjoint": not (source_sessions_by_split["development"] & source_sessions_by_split["confirmation"]),
        "source_receipts_hash_complete": receipt_count > 0 and receipt_hash_missing == 0,
        "role_isolation_clear": isinstance(role_isolation, dict) and role_isolation.get("status") == "PASS_WITHOUT_SPLIT_ASSIGNMENT",
    }
    complete = all(completion_checks.values()) and not errors
    status = "COMPLETE" if complete else "NOT_COMPLETE"
    report = {
        "schema": "hftf_d7_public_real_validation_report_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "status": status,
        "counts": {
            "candidate_windows": candidate_count,
            "adjudicated_parent_events": admitted_count,
            "event_manifest_rows": event_manifest_count,
            "review_rows": review_counts,
            "candidate_duplicate_ids": candidate_duplicate_count,
            "source_receipt_rows": receipt_count,
            "source_receipt_missing_hash_rows": receipt_hash_missing,
        },
        "dataset_counts": dict(dataset_counts),
        "completion_checks": completion_checks,
        "errors": errors,
        "authority": {
            "training_authorized": complete,
            "confirmation_authorized": complete,
            "production_authorized": False,
        },
        "role_isolation": role_isolation,
        "artifacts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in required.items() if path.is_file()},
    }
    write_json(root / "reports" / "d7_validation_report.json", report)
    if args.require_complete and not complete:
        raise ContractError(f"D7 package is not complete; see {root / 'reports' / 'd7_validation_report.json'}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
