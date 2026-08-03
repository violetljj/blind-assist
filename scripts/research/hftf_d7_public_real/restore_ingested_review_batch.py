#!/usr/bin/env python3
"""Restore a completed-review input after an untracked concurrent overwrite.

The primary role file is the authoritative artifact after ingest.  This
command preserves the current input as a conflict copy, restores the exact
candidate rows already present in the primary role file, and writes a receipt.
It does not infer, relabel, or synthesize review decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


ROLE_TO_PRIMARY = {
    "RGB_REVIEWER_A": "reviews/review_a.jsonl",
    "RGB_REVIEWER_B": "reviews/review_b.jsonl",
    "RGB_REVIEWER_C": "reviews/review_c.jsonl",
    "GEOMETRY_EVIDENCE_REVIEWER": "reviews/geometry_review.jsonl",
    "COUNTEREXAMPLE_REVIEWER": "reviews/counterexample_review.jsonl",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def restore(root: Path, batch_id: str, role: str) -> dict[str, Any]:
    if role not in ROLE_TO_PRIMARY:
        raise ContractError(f"unknown review role: {role}")
    batch_root = root / "reviews" / "input_bundles" / batch_id
    input_path = batch_root / role / "completed_review.jsonl"
    manifest_path = batch_root / "manifests" / f"{role}.jsonl"
    primary_path = root / ROLE_TO_PRIMARY[role]
    if not input_path.is_file() or not manifest_path.is_file() or not primary_path.is_file():
        raise ContractError("input, manifest, and primary review files are required")

    manifest_rows = load_jsonl(manifest_path)
    candidate_ids = [str(row.get("candidate_id") or "") for row in manifest_rows]
    if len(candidate_ids) != len(set(candidate_ids)) or not all(candidate_ids):
        raise ContractError(f"invalid candidate identities in manifest: {manifest_path}")
    primary_by_id: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(primary_path):
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in candidate_ids:
            if candidate_id in primary_by_id:
                raise ContractError(f"duplicate candidate in primary review file: {candidate_id}")
            primary_by_id[candidate_id] = row
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in primary_by_id]
    if missing:
        raise ContractError(f"primary review file lacks batch candidates: {missing[:5]}")

    previous_sha = sha256_file(input_path)
    conflict_dir = root / "reviews" / "conflicts" / batch_id / role
    conflict_dir.mkdir(parents=True, exist_ok=False)
    conflict_path = conflict_dir / f"completed_review_overwritten_{previous_sha}.jsonl"
    shutil.copy2(input_path, conflict_path)

    restored = [primary_by_id[candidate_id] for candidate_id in candidate_ids]
    write_jsonl(input_path, restored)
    restored_sha = sha256_file(input_path)
    receipt = {
        "schema": "hftf_d7_public_real_review_batch_restore_receipt_v1",
        "record_kind": "REVIEW_BATCH_RESTORE",
        "generated_at_utc": utc_now(),
        "status": "RESTORED_FROM_INGESTED_PRIMARY",
        "batch_id": batch_id,
        "review_role": role,
        "candidate_count": len(restored),
        "candidate_ids": candidate_ids,
        "primary_review": {"path": str(primary_path), "sha256": sha256_file(primary_path)},
        "previous_input": {"path": str(conflict_path), "sha256": previous_sha},
        "restored_input": {"path": str(input_path), "sha256": restored_sha},
        "reason": "Concurrent untracked output overwrote the input after ingest; restored from authoritative primary rows.",
        "decision_inferred": False,
        "event_truth_assigned": False,
        "admission_assigned": False,
    }
    receipt_path = root / "receipts" / f"review_batch_restore_{batch_id}_{role}.json"
    if receipt_path.exists():
        raise ContractError(f"refusing to overwrite restore receipt: {receipt_path}")
    write_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--role", choices=sorted(ROLE_TO_PRIMARY), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    import json

    args = parse_args()
    print(json.dumps(restore(Path(args.output_root).resolve(), args.batch_id, args.role), ensure_ascii=False, sort_keys=True))
