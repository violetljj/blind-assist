#!/usr/bin/env python3
"""Audit candidate and source-session overlap between two review bundles.

This is a manifest-level identity audit only.  A PASS proves that the two
review bundles do not reuse the same candidate IDs or source-session identity
keys recorded in the role manifests.  It does not prove absence of
near-duplicate media, shared ancestry, or independent parent events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json


ROLES = (
    "RGB_REVIEWER_A",
    "RGB_REVIEWER_B",
    "RGB_REVIEWER_C",
    "GEOMETRY_EVIDENCE_REVIEWER",
    "COUNTEREXAMPLE_REVIEWER",
)
IDENTITY_FIELDS = ("source_session_token", "source_session_id")


def _identity_key(row: dict[str, Any]) -> str | None:
    for field in IDENTITY_FIELDS:
        value = row.get(field)
        if value is not None and str(value):
            return str(value)
    return None


def _read_batch(batch_root: Path) -> tuple[dict[str, Any], set[str], set[str], list[str]]:
    manifest_path = batch_root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"review bundle manifest missing: {manifest_path}")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ContractError(f"review bundle manifest is not an object: {manifest_path}")
    candidate_ids = [str(value) for value in manifest.get("candidate_ids", [])]
    errors: list[str] = []
    if len(candidate_ids) != int(manifest.get("candidate_count", -1)):
        errors.append("bundle candidate_count does not match candidate_ids length")
    if len(set(candidate_ids)) != len(candidate_ids):
        errors.append("bundle candidate_ids contain duplicates")

    role_candidate_sets: list[set[str]] = []
    role_identity_maps: list[dict[str, str | None]] = []
    for role in ROLES:
        role_path = batch_root / "manifests" / f"{role}.jsonl"
        if not role_path.is_file():
            errors.append(f"missing role manifest: {role}")
            continue
        rows = load_jsonl(role_path)
        role_map = {str(row.get("candidate_id")): _identity_key(row) for row in rows}
        role_candidate_sets.append(set(role_map))
        role_identity_maps.append(role_map)
        if set(role_map) != set(candidate_ids):
            errors.append(f"role candidate set differs from bundle: {role}")
        if any(value is None for value in role_map.values()):
            errors.append(f"role has missing source-session identity key: {role}")

    if role_candidate_sets and any(values != role_candidate_sets[0] for values in role_candidate_sets[1:]):
        errors.append("role candidate sets disagree")
    if role_identity_maps:
        first_map = role_identity_maps[0]
        for role_map in role_identity_maps[1:]:
            if role_map != first_map:
                errors.append("role source-session identity mappings disagree")

    identity_keys = {
        value
        for role_map in role_identity_maps
        for value in role_map.values()
        if value is not None
    }
    return manifest, set(candidate_ids), identity_keys, sorted(set(errors))


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference_root = Path(args.reference_batch_root).resolve()
    candidate_root = Path(args.candidate_batch_root).resolve()
    output_root = Path(args.output_root).resolve()
    reference_manifest, reference_ids, reference_sessions, reference_errors = _read_batch(reference_root)
    candidate_manifest, candidate_ids, candidate_sessions, candidate_errors = _read_batch(candidate_root)

    candidate_overlap = sorted(reference_ids & candidate_ids)
    session_overlap = sorted(reference_sessions & candidate_sessions)
    errors = sorted(set(reference_errors + candidate_errors))
    if errors:
        status = "NOT_EVALUABLE"
    elif candidate_overlap or session_overlap:
        status = "FAIL_OVERLAP"
    else:
        status = "PASS_NO_MANIFEST_OVERLAP"

    report = {
        "schema": "hftf_d7_public_real_review_bundle_overlap_audit_v1",
        "record_kind": "REVIEW_BUNDLE_OVERLAP_AUDIT",
        "audit_id": args.audit_id,
        "generated_at_utc": utc_now(),
        "status": status,
        "reference_batch": {
            "batch_root": str(reference_root),
            "batch_id": reference_manifest.get("batch_id"),
            "candidate_count": len(reference_ids),
            "source_session_identity_count": len(reference_sessions),
            "bundle_manifest_sha256": sha256_file(reference_root / "bundle_manifest.json"),
        },
        "candidate_batch": {
            "batch_root": str(candidate_root),
            "batch_id": candidate_manifest.get("batch_id"),
            "candidate_count": len(candidate_ids),
            "source_session_identity_count": len(candidate_sessions),
            "bundle_manifest_sha256": sha256_file(candidate_root / "bundle_manifest.json"),
        },
        "candidate_overlap_count": len(candidate_overlap),
        "candidate_overlap_ids": candidate_overlap,
        "source_session_identity_overlap_count": len(session_overlap),
        "source_session_identity_overlap_keys": session_overlap,
        "errors": errors,
        "isolated_review_authorized_by_this_audit": status == "PASS_NO_MANIFEST_OVERLAP",
        "event_truth_inferred": False,
        "parent_event_independence_established": False,
        "limitations": [
            "The identity key is the source_session_token or source_session_id recorded in review manifests.",
            "This audit does not prove absence of near-duplicate media, shared ancestry, stereo/view collapse, or parent-event adjacency.",
            "A PASS is necessary for isolated review packaging but is not an admission or event-truth result.",
        ],
    }
    report_dir = output_root / "reports" / "review_bundle_overlap" / args.audit_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "overlap_audit.json"
    write_json(report_path, report)
    receipt = {
        "schema": "hftf_d7_public_real_review_bundle_overlap_receipt_v1",
        "record_kind": "REVIEW_BUNDLE_OVERLAP_RECEIPT",
        "audit_id": args.audit_id,
        "generated_at_utc": report["generated_at_utc"],
        "status": status,
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "reference_batch_root": str(reference_root),
        "candidate_batch_root": str(candidate_root),
        "candidate_overlap_count": len(candidate_overlap),
        "source_session_identity_overlap_count": len(session_overlap),
        "isolated_review_authorized_by_this_audit": report["isolated_review_authorized_by_this_audit"],
        "event_truth_inferred": False,
    }
    receipt_path = output_root / "receipts" / f"review_bundle_overlap_receipt_{args.audit_id}.json"
    write_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reference-batch-root", required=True)
    parser.add_argument("--candidate-batch-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
