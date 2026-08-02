#!/usr/bin/env python3
"""Validate and merge isolated D7 review outputs into the review surface.

The command accepts only completed records that correspond to one immutable
review-bundle manifest.  It replaces assignment-only rows for that batch and
leaves all other candidate assignments unchanged.  It never writes an
adjudicated event or a split assignment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

from materialize_review_bundle import (
    ALLOWED_EVENT_BUCKETS,
    COUNTEREXAMPLE_ROLE,
    GEOMETRY_ROLE,
    REVIEW_ROLES,
)
from pipeline import ContractError, load_json, load_jsonl, sha256_file, stable_id, utc_now, write_json


ROLE_TO_PRIMARY = {
    "RGB_REVIEWER_A": "reviews/review_a.jsonl",
    "RGB_REVIEWER_B": "reviews/review_b.jsonl",
    "RGB_REVIEWER_C": "reviews/review_c.jsonl",
    "GEOMETRY_EVIDENCE_REVIEWER": "reviews/geometry_review.jsonl",
    "COUNTEREXAMPLE_REVIEWER": "reviews/counterexample_review.jsonl",
}

DECISIONS = {"SUPPORT", "REJECT", "NOT_EVALUABLE", "ESCALATE"}
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


def _interval_valid(interval: object) -> bool:
    if not isinstance(interval, dict):
        return False
    if interval.get("start_timestamp_ns") is None or interval.get("end_timestamp_ns") is None:
        return False
    try:
        return int(interval["start_timestamp_ns"]) < int(interval["end_timestamp_ns"])
    except (TypeError, ValueError):
        return False


def _phase_valid(event_bucket: str, phase_intervals: object) -> bool:
    if not isinstance(phase_intervals, dict):
        return False
    if event_bucket.endswith("POSITIVE"):
        return all(_interval_valid(phase_intervals.get(key)) for key in (
            "pre_interval", "alertable_interval", "passed_clearance_interval"
        ))
    if event_bucket.endswith("NEGATIVE"):
        return _interval_valid(phase_intervals.get("continuous_negative_interval"))
    return False


def _load_bundle(root: Path, batch_id: str, roles: list[str]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    batch_root = root / "reviews" / "input_bundles" / batch_id
    bundle_manifest_path = batch_root / "bundle_manifest.json"
    bundle_manifest = load_json(bundle_manifest_path)
    if not isinstance(bundle_manifest, dict) or bundle_manifest.get("status") != "READY_FOR_ISOLATED_REVIEW":
        raise ContractError(f"review bundle is not ready: {bundle_manifest_path}")
    if bundle_manifest.get("model_output_visible_in_any_input") is not False:
        raise ContractError("review bundle model-output firewall is not closed")
    selected_ids = {str(value) for value in bundle_manifest.get("candidate_ids", [])}
    if not selected_ids:
        raise ContractError("review bundle has no candidate_ids")
    outputs: dict[str, list[dict[str, Any]]] = {}
    for role in roles:
        manifest_path = batch_root / "manifests" / f"{role}.jsonl"
        output_path = batch_root / role / "completed_review.jsonl"
        manifest_rows = list(_iter_jsonl(manifest_path))
        manifest_ids = {str(row.get("candidate_id")) for row in manifest_rows}
        if manifest_ids != selected_ids:
            raise ContractError(f"{role} manifest candidate set differs from bundle manifest")
        if not output_path.is_file():
            raise ContractError(f"completed review output missing: {output_path}")
        rows = list(_iter_jsonl(output_path))
        seen: set[str] = set()
        for row in rows:
            candidate_id = str(row.get("candidate_id") or "")
            if candidate_id not in selected_ids:
                raise ContractError(f"{role} output has unknown candidate_id: {candidate_id}")
            if candidate_id in seen:
                raise ContractError(f"{role} output has duplicate candidate_id: {candidate_id}")
            seen.add(candidate_id)
            _validate_review_row(row, role=role, batch_id=batch_id)
        if seen != selected_ids:
            raise ContractError(f"{role} output is incomplete: expected {len(selected_ids)}, got {len(seen)}")
        outputs[role] = rows
    return bundle_manifest, outputs


def _validate_review_row(row: dict[str, Any], *, role: str, batch_id: str) -> None:
    if row.get("schema") != "hftf_d7_public_real_completed_review_v1":
        raise ContractError(f"{role} output has wrong schema: {row.get('candidate_id')}")
    if row.get("record_kind") != "COMPLETED_REVIEW":
        raise ContractError(f"{role} output is not completed: {row.get('candidate_id')}")
    if row.get("review_role") != role or row.get("batch_id") != batch_id:
        raise ContractError(f"{role} output role/batch mismatch: {row.get('candidate_id')}")
    if row.get("review_completed") is not True:
        raise ContractError(f"{role} output is not marked review_completed: {row.get('candidate_id')}")
    if row.get("decision") not in DECISIONS:
        raise ContractError(f"{role} output has invalid decision: {row.get('candidate_id')}")
    if row.get("event_bucket") not in ALLOWED_EVENT_BUCKETS:
        raise ContractError(f"{role} output has invalid event_bucket: {row.get('candidate_id')}")
    if row.get("model_output_visible") is not False:
        raise ContractError(f"{role} output model visibility drift: {row.get('candidate_id')}")
    phase_intervals = row.get("phase_intervals")
    if phase_intervals is not None and not isinstance(phase_intervals, dict):
        raise ContractError(f"{role} output phase_intervals is not an object/null: {row.get('candidate_id')}")
    if isinstance(phase_intervals, dict) and not set(phase_intervals).issubset(INTERVAL_KEYS):
        raise ContractError(f"{role} output has unknown phase interval keys: {row.get('candidate_id')}")
    if row.get("decision") == "NOT_EVALUABLE" and row.get("event_bucket") != "NOT_EVALUABLE":
        raise ContractError(f"{role} NOT_EVALUABLE decision has non-NOT_EVALUABLE bucket: {row.get('candidate_id')}")
    if row.get("decision") == "SUPPORT" and not _phase_valid(str(row["event_bucket"]), phase_intervals):
        raise ContractError(f"{role} SUPPORT record lacks a complete phase contract: {row.get('candidate_id')}")
    if role == GEOMETRY_ROLE and row.get("source_native_geometry_only") is not True:
        raise ContractError(f"geometry output is not source-native-only: {row.get('candidate_id')}")
    if role != GEOMETRY_ROLE and row.get("source_native_geometry_only") is not False:
        raise ContractError(f"non-geometry output claims source-native-only: {row.get('candidate_id')}")
    if role == COUNTEREXAMPLE_ROLE and row.get("counterexample_search_completed") is not True:
        raise ContractError(f"counterexample search not completed: {row.get('candidate_id')}")
    if role != COUNTEREXAMPLE_ROLE and row.get("counterexample_search_completed") is not False:
        raise ContractError(f"non-counterexample output has counterexample search flag: {row.get('candidate_id')}")


def _write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    roles = [item.strip() for item in args.roles.split(",") if item.strip()]
    if not roles or any(role not in ROLE_TO_PRIMARY for role in roles) or len(set(roles)) != len(roles):
        raise ContractError(f"invalid --roles: {roles}")
    bundle_manifest, output_by_role = _load_bundle(root, args.batch_id, roles)
    selected_ids = {str(value) for value in bundle_manifest["candidate_ids"]}

    event_rows = load_jsonl(root / "manifests" / "event_manifest.jsonl")
    event_by_candidate = {str(row.get("candidate_id")): row for row in event_rows}
    if not selected_ids.issubset(event_by_candidate):
        raise ContractError("review bundle references candidate absent from event manifest")

    receipt_path = root / "receipts" / f"review_ingest_receipt_{args.batch_id}.json"
    if receipt_path.exists():
        raise ContractError(f"review ingest receipt already exists; refusing overwrite: {receipt_path}")
    backup_root = root / "reviews" / "backups" / args.batch_id
    backup_root.mkdir(parents=True, exist_ok=False)
    primary_hashes_before: dict[str, str] = {}
    primary_hashes_after: dict[str, str] = {}
    replacement_counts: dict[str, int] = {}
    for role in roles:
        primary_path = root / ROLE_TO_PRIMARY[role]
        if not primary_path.is_file():
            raise ContractError(f"primary review file missing: {primary_path}")
        shutil.copy2(primary_path, backup_root / primary_path.name)
        primary_hashes_before[role] = sha256_file(primary_path)
        replacements = {
            str(row["candidate_id"]): row
            for row in output_by_role[role]
        }
        primary_rows = list(_iter_jsonl(primary_path))
        seen_primary: set[str] = set()
        merged: list[dict[str, Any]] = []
        replacements_applied = 0
        for row in primary_rows:
            candidate_id = str(row.get("candidate_id") or "")
            if row.get("review_role") != role:
                raise ContractError(f"primary review role drift in {primary_path}: {candidate_id}")
            if candidate_id in seen_primary:
                raise ContractError(f"duplicate primary review candidate: {role} {candidate_id}")
            seen_primary.add(candidate_id)
            if candidate_id not in replacements:
                merged.append(row)
                continue
            if row.get("record_kind") != "ASSIGNMENT_ONLY" or row.get("review_completed") is True:
                raise ContractError(f"refusing to overwrite an existing completed review: {role} {candidate_id}")
            source_event = event_by_candidate[candidate_id]
            source_review = replacements[candidate_id]
            merged.append({
                "schema": "hftf_d7_public_real_review_record_v1",
                "record_kind": "COMPLETED_REVIEW",
                "review_role": role,
                "event_id": source_event.get("event_id"),
                "candidate_id": candidate_id,
                "dataset_id": source_event.get("dataset_id"),
                "source_session_id": source_event.get("source_session_id"),
                "review_completed": True,
                "decision": source_review.get("decision"),
                "event_bucket": source_review.get("event_bucket"),
                "phase_intervals": source_review.get("phase_intervals"),
                "model_output_visible": False,
                "source_native_geometry_only": source_review.get("source_native_geometry_only"),
                "counterexample_search_required": role == COUNTEREXAMPLE_ROLE,
                "counterexample_search_completed": source_review.get("counterexample_search_completed"),
                "evidence_basis": source_review.get("evidence_basis"),
                "review_batch_id": args.batch_id,
                "review_input_id": source_review.get("review_input_id"),
                "review_notes": source_review.get("notes"),
                "reason_code": source_review.get("reason_code"),
                "rgb_local_path": source_event.get("rgb_local_path"),
            })
            replacements_applied += 1
        if seen_primary != set(event_by_candidate):
            # The primary files should have one row for every candidate; this
            # check catches partial packages before any replacement is atomic.
            raise ContractError(f"primary review candidate set mismatch: {role}")
        _write_atomic(primary_path, merged)
        primary_hashes_after[role] = sha256_file(primary_path)
        replacement_counts[role] = replacements_applied

    # Mark the event shells as having completed raw review roles, without
    # changing their event bucket, truth status, or admission status.
    event_path = root / "manifests" / "event_manifest.jsonl"
    event_merged: list[dict[str, Any]] = []
    for row in event_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in selected_ids:
            updated = dict(row)
            previous = updated.get("completed_review_roles")
            completed_roles = set(previous if isinstance(previous, list) else [])
            completed_roles.update(roles)
            updated["completed_review_roles"] = sorted(completed_roles)
            updated["review_state"] = "INDEPENDENT_REVIEW_COMPLETED"
            updated["review_batch_id"] = args.batch_id
            updated["admission_status"] = "PENDING_REVIEW"
            updated["truth_status"] = "NOT_EVALUABLE"
            event_merged.append(updated)
        else:
            event_merged.append(row)
    _write_atomic(event_path, event_merged)

    receipt = {
        "schema": "hftf_d7_public_real_review_ingest_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": utc_now(),
        "status": "REVIEWS_INGESTED_NO_ADJUDICATION",
        "candidate_count": len(selected_ids),
        "review_roles": roles,
        "replacement_counts": replacement_counts,
        "primary_review_hashes_before": primary_hashes_before,
        "primary_review_hashes_after": primary_hashes_after,
        "event_manifest_sha256": sha256_file(event_path),
        "backup_root": str(backup_root),
        "backup_files": {role: str((backup_root / Path(ROLE_TO_PRIMARY[role]).name).resolve()) for role in roles},
        "completed_review_output_hashes": {
            role: sha256_file(root / "reviews" / "input_bundles" / args.batch_id / role / "completed_review.jsonl")
            for role in roles
        },
        "adjudicated_parent_events": 0,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
        "notes": [
            "Only assignment-only rows for this batch were replaced.",
            "No event bucket was promoted to adjudicated truth.",
            "A final adjudicator must consume all independent reviews before any admission or split assignment.",
        ],
    }
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--roles", default=",".join(REVIEW_ROLES))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
