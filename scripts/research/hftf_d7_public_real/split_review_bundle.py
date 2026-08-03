#!/usr/bin/env python3
"""Split a frozen D7 review bundle into smaller model-blind input bundles.

This is an input-only throughput helper.  It preserves the parent bundle as
the audit source, hard-links its evidence files when possible, and creates
fresh immutable review identities for each child bundle.  It never writes a
review decision, event bucket, adjudication, admission, or split role.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, stable_id, utc_now, write_json, write_jsonl


REVIEW_ROLES = (
    "RGB_REVIEWER_A",
    "RGB_REVIEWER_B",
    "RGB_REVIEWER_C",
    "GEOMETRY_EVIDENCE_REVIEWER",
    "COUNTEREXAMPLE_REVIEWER",
)


def _link_or_copy(source: Path, target: Path) -> str:
    if not source.is_file():
        raise ContractError(f"evidence file missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ContractError(f"split target already exists: {target}")
    try:
        os.link(source, target)
        mode = "HARDLINK"
    except OSError:
        shutil.copy2(source, target)
        mode = "COPY_FALLBACK"
    if sha256_file(source) != sha256_file(target):
        raise ContractError(f"evidence hash changed while splitting: {source} -> {target}")
    return mode


def _rewrite_row(row: dict[str, Any], role: str, batch_id: str, index: int, batch_root: Path) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id") or "")
    if not candidate_id:
        raise ContractError(f"review input has no candidate_id: role={role}")
    out = dict(row)
    out["batch_id"] = batch_id
    out["review_index"] = index
    out["review_input_id"] = stable_id("d7review-input", batch_id, role, candidate_id)

    contact_sheet = row.get("contact_sheet_path")
    if contact_sheet:
        source = Path(str(contact_sheet))
        target = batch_root / role / "contact_sheets" / f"{candidate_id}{source.suffix or '.jpg'}"
        _link_or_copy(source, target)
        out["contact_sheet_path"] = str(target.resolve())
        out["contact_sheet_sha256"] = sha256_file(target)

    native_geometry = row.get("native_geometry_path")
    if native_geometry:
        source = Path(str(native_geometry))
        target = batch_root / role / "native_geometry" / f"{candidate_id}{source.suffix or '.json'}"
        _link_or_copy(source, target)
        out["native_geometry_path"] = str(target.resolve())
        out["native_geometry_sha256"] = sha256_file(target)
    return out


def _manifest_files(batch_root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for role in REVIEW_ROLES:
        path = batch_root / "manifests" / f"{role}.jsonl"
        result[role] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    source_root = output_root / "reviews" / "input_bundles" / args.source_batch_id
    source_manifest_path = source_root / "bundle_manifest.json"
    source_manifest = load_json(source_manifest_path)
    if not isinstance(source_manifest, dict) or source_manifest.get("status") != "READY_FOR_ISOLATED_REVIEW":
        raise ContractError(f"source review bundle is not ready: {source_manifest_path}")
    if source_manifest.get("model_output_visible_in_any_input") is not False:
        raise ContractError("source review bundle is not model-blind")

    source_ids = [str(value) for value in source_manifest.get("candidate_ids", [])]
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise ContractError("source bundle candidate_ids must be non-empty and unique")
    parent_sha = sha256_file(source_manifest_path)
    role_rows: dict[str, list[dict[str, Any]]] = {}
    for role in REVIEW_ROLES:
        path = source_root / "manifests" / f"{role}.jsonl"
        rows = load_jsonl(path)
        by_id = {str(row.get("candidate_id")): row for row in rows}
        if set(by_id) != set(source_ids) or len(rows) != len(source_ids):
            raise ContractError(f"role manifest candidate mismatch: {role}")
        role_rows[role] = [by_id[candidate_id] for candidate_id in source_ids]

    if args.chunk_size <= 0:
        raise ContractError("chunk_size must be positive")
    chunks = [source_ids[offset : offset + args.chunk_size] for offset in range(0, len(source_ids), args.chunk_size)]
    child_batches: list[dict[str, Any]] = []
    split_mode_counts = {"HARDLINK": 0, "COPY_FALLBACK": 0}
    for chunk_index, chunk_ids in enumerate(chunks, 1):
        child_batch_id = f"{args.source_batch_id}-chunk-{chunk_index:02d}"
        child_root = output_root / "reviews" / "input_bundles" / child_batch_id
        if child_root.exists():
            raise ContractError(f"child bundle already exists: {child_root}")
        child_root.mkdir(parents=True)
        chunk_set = set(chunk_ids)
        for role in REVIEW_ROLES:
            child_rows = []
            for index, row in enumerate(role_rows[role]):
                if str(row["candidate_id"]) not in chunk_set:
                    continue
                child_rows.append(_rewrite_row(row, role, child_batch_id, index, child_root))
            # _rewrite_row intentionally handles the evidence link before the
            # row is persisted; every path in the manifest is self-contained.
            write_jsonl(child_root / "manifests" / f"{role}.jsonl", child_rows)

        child_manifest = {
            "schema": "hftf_d7_public_real_review_bundle_v1",
            "run_id": f"{args.source_batch_id}-split-{chunk_index:02d}",
            "batch_id": child_batch_id,
            "status": "READY_FOR_ISOLATED_REVIEW",
            "contract_id": source_manifest.get("contract_id", "HFTF_D7_PUBLIC_REAL_R1"),
            "candidate_ids": chunk_ids,
            "candidate_count": len(chunk_ids),
            "review_roles": list(REVIEW_ROLES),
            "manifest_files": _manifest_files(child_root),
            "model_output_visible_in_any_input": False,
            "review_assignments_are_not_labels": True,
            "confirmation_authorized": False,
            "production_authorized": False,
            "training_authorized": False,
            "parent_batch_id": args.source_batch_id,
            "parent_bundle_manifest_sha256": parent_sha,
            "split_chunk_index": chunk_index,
            "split_chunk_count": len(chunks),
            "evidence_files_are_hardlinked_when_possible": True,
        }
        child_manifest_path = child_root / "bundle_manifest.json"
        write_json(child_manifest_path, child_manifest)
        receipt = {
            "schema": "hftf_d7_public_real_review_bundle_split_receipt_v1",
            "record_kind": "REVIEW_BUNDLE_SPLIT_RECEIPT",
            "status": "READY_FOR_ISOLATED_REVIEW",
            "generated_at_utc": utc_now(),
            "source_batch_id": args.source_batch_id,
            "source_bundle_manifest": {"path": str(source_manifest_path), "sha256": parent_sha},
            "child_batch_id": child_batch_id,
            "child_bundle_manifest": {"path": str(child_manifest_path), "sha256": sha256_file(child_manifest_path)},
            "candidate_count": len(chunk_ids),
            "candidate_ids": chunk_ids,
            "split_chunk_index": chunk_index,
            "split_chunk_count": len(chunks),
            "review_roles": list(REVIEW_ROLES),
            "admission_assigned": False,
            "event_truth_inferred": False,
            "split_assigned": False,
        }
        receipt_path = output_root / "receipts" / f"review_bundle_split_receipt_{child_batch_id}.json"
        write_json(receipt_path, receipt)
        child_batches.append({
            "batch_id": child_batch_id,
            "candidate_count": len(chunk_ids),
            "bundle_manifest": {"path": str(child_manifest_path), "sha256": sha256_file(child_manifest_path)},
            "receipt": str(receipt_path),
        })

    return {
        "schema": "hftf_d7_public_real_review_bundle_split_v1",
        "record_kind": "REVIEW_BUNDLE_SPLIT",
        "status": "READY_FOR_ISOLATED_REVIEW",
        "generated_at_utc": utc_now(),
        "source_batch_id": args.source_batch_id,
        "source_bundle_manifest_sha256": parent_sha,
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "candidate_count": len(source_ids),
        "child_batches": child_batches,
        "admission_assigned": False,
        "event_truth_inferred": False,
        "split_assigned": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--source-batch-id", required=True)
    parser.add_argument("--chunk-size", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    import json

    try:
        print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
    except ContractError as exc:
        raise SystemExit(str(exc))
