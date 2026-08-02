#!/usr/bin/env python3
"""Validate final adjudication outputs and update only event terminals."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

from materialize_review_bundle import ALLOWED_EVENT_BUCKETS
from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json


DECISIONS = {"ADMIT", "REJECT", "NOT_EVALUABLE", "ESCALATE"}
INTERVAL_KEYS = {
    "pre_interval",
    "alertable_interval",
    "passed_clearance_interval",
    "continuous_negative_interval",
}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise ContractError(f"required JSONL missing: {path}")
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


def _interval_valid(value: object) -> bool:
    if not isinstance(value, dict) or value.get("start_timestamp_ns") is None or value.get("end_timestamp_ns") is None:
        return False
    try:
        return int(value["start_timestamp_ns"]) < int(value["end_timestamp_ns"])
    except (TypeError, ValueError):
        return False


def _phase_valid(bucket: str, phases: object) -> bool:
    if not isinstance(phases, dict) or not set(phases).issubset(INTERVAL_KEYS):
        return False
    if bucket.endswith("POSITIVE"):
        return all(_interval_valid(phases.get(key)) for key in (
            "pre_interval", "alertable_interval", "passed_clearance_interval"
        ))
    if bucket.endswith("NEGATIVE"):
        return _interval_valid(phases.get("continuous_negative_interval"))
    return False


def _validate_row(row: dict[str, Any], *, batch_id: str, event_by_candidate: dict[str, dict[str, Any]]) -> None:
    candidate_id = str(row.get("candidate_id") or "")
    if row.get("schema") != "hftf_d7_public_real_completed_adjudication_v1" or row.get("record_kind") != "COMPLETED_ADJUDICATION":
        raise ContractError(f"invalid adjudication schema/record kind: {candidate_id}")
    if row.get("batch_id") != batch_id or candidate_id not in event_by_candidate:
        raise ContractError(f"adjudication batch/candidate mismatch: {candidate_id}")
    if row.get("event_id") != event_by_candidate[candidate_id].get("event_id"):
        raise ContractError(f"adjudication event_id mismatch: {candidate_id}")
    if row.get("adjudication_decision") not in DECISIONS:
        raise ContractError(f"invalid adjudication decision: {candidate_id}")
    if row.get("admission_status") not in {"ADMITTED", "NOT_ADMITTED"}:
        raise ContractError(f"invalid admission_status: {candidate_id}")
    if row.get("event_bucket") not in ALLOWED_EVENT_BUCKETS:
        raise ContractError(f"invalid adjudication event_bucket: {candidate_id}")
    if row.get("model_output_visible") is not False:
        raise ContractError(f"adjudication model visibility drift: {candidate_id}")
    if row.get("adjudication_decision") == "ADMIT":
        if row.get("admission_status") != "ADMITTED" or not _phase_valid(str(row["event_bucket"]), row.get("phase_intervals")):
            raise ContractError(f"ADMIT lacks admission/phase contract: {candidate_id}")
    else:
        if row.get("admission_status") != "NOT_ADMITTED":
            raise ContractError(f"non-ADMIT adjudication marked admitted: {candidate_id}")
        if row.get("event_bucket") != "NOT_EVALUABLE" and row.get("phase_intervals") is not None and not _phase_valid(str(row["event_bucket"]), row.get("phase_intervals")):
            raise ContractError(f"non-ADMIT has invalid phase intervals: {candidate_id}")


def _write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    bundle_root = root / "reviews" / "adjudication_bundles" / args.batch_id
    bundle_manifest = load_json(bundle_root / "bundle_manifest.json")
    if not isinstance(bundle_manifest, dict) or bundle_manifest.get("status") != "READY_FOR_FINAL_ADJUDICATION":
        raise ContractError("adjudication bundle is not ready")
    output_path = bundle_root / "FINAL_ADJUDICATOR" / "final_adjudication.jsonl"
    outputs = list(_iter_jsonl(output_path))
    selected_ids = {str(value) for value in bundle_manifest.get("candidate_ids", [])}
    if len(outputs) != len(selected_ids):
        raise ContractError(f"adjudication output count mismatch: expected {len(selected_ids)}, got {len(outputs)}")
    event_path = root / "manifests" / "event_manifest.jsonl"
    rejected_path = root / "adjudication" / "rejected_events.jsonl"
    adjudicated_path = root / "adjudication" / "adjudicated_events.jsonl"
    event_rows = load_jsonl(event_path)
    event_by_candidate = {str(row.get("candidate_id")): row for row in event_rows}
    seen: set[str] = set()
    for row in outputs:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in seen or candidate_id not in selected_ids:
            raise ContractError(f"duplicate/unknown adjudication candidate: {candidate_id}")
        seen.add(candidate_id)
        _validate_row(row, batch_id=args.batch_id, event_by_candidate=event_by_candidate)
    if seen != selected_ids:
        raise ContractError("adjudication output candidate set mismatch")

    receipt_path = root / "receipts" / f"adjudication_ingest_receipt_{args.batch_id}.json"
    if receipt_path.exists():
        raise ContractError(f"adjudication receipt already exists; refusing overwrite: {receipt_path}")
    backup_root = root / "adjudication" / "backups" / args.batch_id
    backup_root.mkdir(parents=True, exist_ok=False)
    for path in (event_path, rejected_path, adjudicated_path):
        shutil.copy2(path, backup_root / path.name)

    existing_adjudicated = load_jsonl(adjudicated_path)
    existing_event_ids = {str(row.get("event_id")) for row in existing_adjudicated}
    admitted_rows: list[dict[str, Any]] = []
    final_by_candidate = {str(row["candidate_id"]): row for row in outputs}
    for row in outputs:
        if row.get("admission_status") != "ADMITTED":
            continue
        candidate = event_by_candidate[str(row["candidate_id"])]
        event_id = str(candidate.get("event_id"))
        if event_id in existing_event_ids:
            raise ContractError(f"adjudicated event already exists: {event_id}")
        phases = row.get("phase_intervals") if isinstance(row.get("phase_intervals"), dict) else {}
        admitted_rows.append({
            "schema": "hftf_d7_public_real_adjudicated_event_v1",
            "record_kind": "ADJUDICATED_EVENT",
            "event_id": event_id,
            "parent_event_id": event_id,
            "candidate_id": candidate.get("candidate_id"),
            "dataset_id": candidate.get("dataset_id"),
            "source_session_id": candidate.get("source_session_id"),
            "ancestry_group": candidate.get("ancestry_group"),
            "frame_ids": candidate.get("frame_ids", []),
            "start_timestamp_ns": candidate.get("start_timestamp_ns"),
            "end_timestamp_ns": candidate.get("end_timestamp_ns"),
            "event_bucket": row.get("event_bucket"),
            "truth_status": "ADJUDICATED",
            "admission_status": "ADMITTED",
            "pre_interval": phases.get("pre_interval"),
            "alertable_interval": phases.get("alertable_interval"),
            "passed_clearance_interval": phases.get("passed_clearance_interval"),
            "continuous_negative_interval": phases.get("continuous_negative_interval"),
            "review_model_output_visible": False,
            "geometry_model_output_visible": False,
            "adjudication_batch_id": args.batch_id,
            "adjudication_reason": row.get("reason_code"),
            "adjudication_notes": row.get("notes"),
        })
    _write_atomic(adjudicated_path, existing_adjudicated + admitted_rows)

    rejected_rows = load_jsonl(rejected_path)
    rejected_by_event = {str(row.get("event_id")): row for row in rejected_rows}
    for candidate_id, row in final_by_candidate.items():
        candidate = event_by_candidate[candidate_id]
        event_id = str(candidate.get("event_id"))
        if row.get("admission_status") == "ADMITTED":
            rejected_by_event.pop(event_id, None)
            continue
        rejected_by_event[event_id] = {
            "schema": "hftf_d7_public_real_rejected_event_v1",
            "record_kind": "FINAL_ADJUDICATION_TERMINAL",
            "event_id": event_id,
            "candidate_id": candidate_id,
            "dataset_id": candidate.get("dataset_id"),
            "source_session_id": candidate.get("source_session_id"),
            "terminal_state": "NOT_EVALUABLE" if row.get("event_bucket") == "NOT_EVALUABLE" else "REJECTED",
            "negative_evidence": False,
            "training_eligible": False,
            "confirmation_eligible": False,
            "adjudication_batch_id": args.batch_id,
            "reason": row.get("reason_code"),
            "notes": row.get("notes"),
        }
    _write_atomic(rejected_path, [rejected_by_event[key] for key in sorted(rejected_by_event)])

    event_merged: list[dict[str, Any]] = []
    for event in event_rows:
        candidate_id = str(event.get("candidate_id"))
        adjudication = final_by_candidate.get(candidate_id)
        if adjudication is None:
            event_merged.append(event)
            continue
        updated = dict(event)
        phases = adjudication.get("phase_intervals") if isinstance(adjudication.get("phase_intervals"), dict) else {}
        updated["review_state"] = "FINAL_ADJUDICATED"
        updated["review_batch_id"] = args.batch_id
        updated["admission_status"] = adjudication.get("admission_status")
        updated["review_model_output_visible"] = False
        updated["geometry_model_output_visible"] = False
        if adjudication.get("admission_status") == "ADMITTED":
            updated["truth_status"] = "ADJUDICATED"
            updated["event_bucket"] = adjudication.get("event_bucket")
            updated["pre_interval"] = phases.get("pre_interval")
            updated["alertable_interval"] = phases.get("alertable_interval")
            updated["passed_clearance_interval"] = phases.get("passed_clearance_interval")
            updated["continuous_negative_interval"] = phases.get("continuous_negative_interval")
            updated["not_evaluable_reason"] = None
        else:
            updated["truth_status"] = "NOT_EVALUABLE"
            updated["event_bucket"] = "NOT_EVALUABLE"
            updated["pre_interval"] = None
            updated["alertable_interval"] = None
            updated["passed_clearance_interval"] = None
            updated["continuous_negative_interval"] = None
            updated["not_evaluable_reason"] = adjudication.get("reason_code")
        event_merged.append(updated)
    _write_atomic(event_path, event_merged)

    receipt = {
        "schema": "hftf_d7_public_real_adjudication_ingest_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": utc_now(),
        "status": "ADJUDICATION_INGESTED",
        "candidate_count": len(outputs),
        "admitted_parent_events_added": len(admitted_rows),
        "adjudication_decisions": {
            decision: sum(1 for row in outputs if row.get("adjudication_decision") == decision)
            for decision in sorted(DECISIONS)
        },
        "admission_statuses": {
            status: sum(1 for row in outputs if row.get("admission_status") == status)
            for status in ("ADMITTED", "NOT_ADMITTED")
        },
        "backup_root": str(backup_root),
        "adjudicated_events_sha256": sha256_file(adjudicated_path),
        "rejected_events_sha256": sha256_file(rejected_path),
        "event_manifest_sha256": sha256_file(event_path),
        "final_adjudication_output_sha256": sha256_file(output_path),
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
        "notes": [
            "Non-admitted outputs remain terminal evidence and are never negative evidence by default.",
            "Only ADMITTED outputs enter adjudicated_events.jsonl; splits remain blocked until all gates pass.",
        ],
    }
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
