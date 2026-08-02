#!/usr/bin/env python3
"""Rebind already-completed reviews to a non-overlapping subset bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, stable_id, utc_now, write_json, write_jsonl


ROLES = (
    "RGB_REVIEWER_A",
    "RGB_REVIEWER_B",
    "RGB_REVIEWER_C",
    "GEOMETRY_EVIDENCE_REVIEWER",
    "COUNTEREXAMPLE_REVIEWER",
)


def _row_digest(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_batch_root).resolve()
    target_root = Path(args.target_batch_root).resolve()
    source_manifest = load_json(source_root / "bundle_manifest.json")
    target_manifest = load_json(target_root / "bundle_manifest.json")
    source_ids = {str(value) for value in source_manifest.get("candidate_ids", [])}
    target_ids = {str(value) for value in target_manifest.get("candidate_ids", [])}
    if not target_ids or not target_ids.issubset(source_ids):
        raise ContractError("target subset is not contained in source review batch")
    if target_manifest.get("batch_id") != args.target_batch_id:
        raise ContractError("target bundle batch_id does not match --target-batch-id")
    source_output_hashes: dict[str, str] = {}
    row_digests: dict[str, dict[str, str]] = {}
    for role in ROLES:
        source_output = source_root / role / "completed_review.jsonl"
        source_rows = load_jsonl(source_output)
        by_candidate = {str(row.get("candidate_id")): row for row in source_rows}
        if len(by_candidate) != len(source_rows):
            raise ContractError(f"source review has duplicate candidate IDs: {role}")
        target_input_rows = load_jsonl(target_root / "manifests" / f"{role}.jsonl")
        target_inputs = {str(row["candidate_id"]): row for row in target_input_rows}
        if set(target_inputs) != target_ids:
            raise ContractError(f"target manifest candidate set mismatch: {role}")
        selected: list[dict[str, Any]] = []
        row_digests[role] = {}
        for candidate_id in sorted(target_ids):
            source_row = by_candidate.get(candidate_id)
            if source_row is None:
                raise ContractError(f"source review lacks target candidate: {role}/{candidate_id}")
            row = copy.deepcopy(source_row)
            original_review_input_id = row.get("review_input_id")
            row["batch_id"] = args.target_batch_id
            row["review_input_id"] = target_inputs[candidate_id]["review_input_id"]
            row["review_reuse"] = {
                "source_batch_id": source_row.get("batch_id"),
                "source_review_input_id": original_review_input_id,
                "source_completed_review_file_sha256": sha256_file(source_output),
                "reason": "NON_OVERLAP_SUBSET_FOR_APPEND_ONLY_INGEST",
                "new_independent_review": False,
            }
            selected.append(row)
            row_digests[role][candidate_id] = _row_digest(source_row)
        output_path = target_root / role / "completed_review.jsonl"
        write_jsonl(output_path, selected)
        source_output_hashes[role] = sha256_file(source_output)
    receipt = {
        "schema": "hftf_d7_public_real_review_reuse_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "status": "COMPLETED_REVIEW_SUBSET_REBOUND",
        "source_batch_id": source_manifest.get("batch_id"),
        "target_batch_id": args.target_batch_id,
        "source_bundle_manifest_sha256": sha256_file(source_root / "bundle_manifest.json"),
        "target_bundle_manifest_sha256": sha256_file(target_root / "bundle_manifest.json"),
        "candidate_count": len(target_ids),
        "review_roles": list(ROLES),
        "source_output_hashes": source_output_hashes,
        "row_digests": row_digests,
        "new_independent_review": False,
        "authority": {"event_truth": False, "training": False, "confirmation": False, "production": False},
        "notes": [
            "Only the non-overlapping subset was rebound; source evidence and reviewer decisions were not changed.",
            "The original completed review batch and input IDs remain recorded per row.",
            "A final adjudicator must consume this subset before any admission or split assignment.",
        ],
    }
    receipt_path = target_root.parent.parent.parent / "receipts" / f"review_reuse_receipt_{args.target_batch_id}.json"
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-batch-root", required=True)
    parser.add_argument("--target-batch-root", required=True)
    parser.add_argument("--target-batch-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
