#!/usr/bin/env python3
"""Non-bypassable preflight for the benchmark-only SANPO segmentation candidate.

This module is intentionally the only preflight component allowed to inspect the
blind holdout.  The trainer consumes only the canonical train/dev manifest after
this module has produced a green, SHA256-attested report; it never receives a
blind path, manifest, image or mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import validate_sanpo_v3_dataset as validator


REPORT_SCHEMA = "blindassist_sanpo_training_gate_v1"
CANONICAL_TRAINING_MANIFEST = "training_manifest.jsonl"
CANONICAL_BLIND_MANIFEST = "blind_holdout/manifest.jsonl"
ALLOWED_PRIVACY_STATUSES = {
    "approved_no_identifiable_subjects",
    "public_dataset_no_personal_data",
    "automated_privacy_clear",
}
ATTESTATION_SCHEMA = "blindassist_v3_source_attestation_v1"
ALLOWED_SOURCE_ADAPTERS = {
    "sanpo_v0",
    "bdd100k_v1",
    "guidetwsi_v1",
    "teacher_consensus_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "green" if passed else "red", "detail": detail}


def privacy_and_source_errors(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        sample_id = str(row.get("id", "<unknown>"))
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        missing = [field for field in ("dataset", "license", "license_url") if not str(source.get(field, "")).strip()]
        if missing:
            errors.append(f"{sample_id}: missing source fields {missing}")
        status = str(source.get("privacy_review_status", "")).strip()
        if status not in ALLOWED_PRIVACY_STATUSES:
            errors.append(f"{sample_id}: privacy_review_status must be one of {sorted(ALLOWED_PRIVACY_STATUSES)}")
    return errors


def asset_hashes(root: Path, rows: list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    errors: list[str] = []
    for row in rows:
        sample_id = str(row.get("id", "<unknown>"))
        for field, expected_field in (("image_path", "image_sha256"), ("semantic_mask_path", "semantic_mask_sha256")):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{sample_id}: missing {field}")
                continue
            path = (root / value).resolve()
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                errors.append(f"{sample_id}: {field} escapes dataset root")
                continue
            if not path.is_file():
                errors.append(f"{sample_id}: missing {relative}")
                continue
            actual = sha256_file(path)
            hashes[relative] = actual
            if actual != row.get(expected_field):
                errors.append(f"{sample_id}: {expected_field} mismatch")
    return dict(sorted(hashes.items())), errors


def verify_bound_file(root: Path, item: dict[str, Any], prefix: str, errors: list[str]) -> None:
    relative = str(item.get(f"{prefix}_path", "")).strip()
    expected = str(item.get(f"{prefix}_sha256", "")).strip()
    if not relative or len(expected) != 64:
        errors.append(f"source attestation missing {prefix}_path/{prefix}_sha256")
        return
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        errors.append(f"source attestation {prefix} escapes dataset root")
        return
    if not path.is_file() or sha256_file(path) != expected:
        errors.append(f"source attestation {prefix} file is missing or SHA256 differs")


def source_attestation_errors(root: Path, rows: list[dict[str, Any]], payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != ATTESTATION_SCHEMA:
        errors.append(f"source_attestation.json schema must be {ATTESTATION_SCHEMA}")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        return errors + ["source_attestation.json requires a non-empty sources list"]
    by_id: dict[str, dict[str, Any]] = {}
    for item in sources:
        if not isinstance(item, dict):
            errors.append("source attestation entries must be objects")
            continue
        source_id = str(item.get("source_id", "")).strip()
        adapter_id = str(item.get("adapter_id", "")).strip()
        if not source_id or source_id in by_id:
            errors.append(f"duplicate or missing source attestation id: {source_id!r}")
            continue
        by_id[source_id] = item
        if adapter_id not in ALLOWED_SOURCE_ADAPTERS:
            errors.append(f"{source_id}: adapter_id {adapter_id!r} is not allow-listed")
        for field in ("dataset", "dataset_version", "license", "license_url", "privacy_review_status"):
            if not str(item.get(field, "")).strip():
                errors.append(f"{source_id}: missing attestation {field}")
        if item.get("privacy_review_status") not in ALLOWED_PRIVACY_STATUSES:
            errors.append(f"{source_id}: attestation privacy status is not allowed")
        for prefix in ("license_evidence", "privacy_evidence", "inventory"):
            verify_bound_file(root, item, prefix, errors)
    for row in rows:
        sample_id = str(row.get("id", "<unknown>"))
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        source_id = str(source.get("source_id", "")).strip()
        receipt = by_id.get(source_id)
        if receipt is None:
            errors.append(f"{sample_id}: source.source_id is missing from source attestation")
            continue
        for field in ("dataset", "license", "license_url", "privacy_review_status"):
            if source.get(field) != receipt.get(field):
                errors.append(f"{sample_id}: source.{field} differs from attested receipt")
    return errors


def policy_errors(policy: dict[str, Any], train_rows: list[dict[str, Any]], blind_rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if policy.get("format") != "blindassist_sanpo_v3_access_policy_v2":
        errors.append("access policy format must be blindassist_sanpo_v3_access_policy_v2")
    if policy.get("training_manifest") != CANONICAL_TRAINING_MANIFEST:
        errors.append("access policy must pin the canonical training manifest")
    if policy.get("blind_manifest") != CANONICAL_BLIND_MANIFEST:
        errors.append("access policy must pin the canonical blind manifest")
    if policy.get("blind_label_access") != "benchmark_only":
        errors.append("blind label access must be benchmark_only")
    if set(policy.get("forbidden_training_paths", [])) != {"blind_holdout"}:
        errors.append("access policy must forbid exactly blind_holdout for training")
    if set(policy.get("forbidden_threshold_selection_paths", [])) != {"blind_holdout"}:
        errors.append("access policy must forbid exactly blind_holdout for threshold selection")
    actual_blind_sessions = sorted({validator.row_session_id(row) for row in blind_rows})
    locked_sessions = policy.get("benchmark_only_sessions")
    if not isinstance(locked_sessions, list) or sorted(locked_sessions) != actual_blind_sessions or len(actual_blind_sessions) != 2:
        errors.append("policy must lock exactly the two blind sessions as benchmark_only")
    protected_sessions = set(policy.get("forbidden_training_sessions", [])) | set(policy.get("forbidden_threshold_selection_sessions", []))
    if protected_sessions != set(actual_blind_sessions):
        errors.append("training and threshold selection must forbid both blind sessions")
    train_sessions = {validator.row_session_id(row) for row in train_rows}
    if train_sessions & set(actual_blind_sessions):
        errors.append("training manifest contains a benchmark_only session")
    return errors


def build_report(dataset_root: Path) -> dict[str, Any]:
    root = dataset_root.resolve()
    train_path = root / CANONICAL_TRAINING_MANIFEST
    blind_path = root / CANONICAL_BLIND_MANIFEST
    policy_path = root / "access_policy.json"
    attestation_path = root / "source_attestation.json"
    errors: list[str] = []
    checks: list[dict[str, Any]] = []
    required = [train_path, blind_path, policy_path, attestation_path]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        errors.extend(f"missing required gate input: {item}" for item in missing)
        checks.append(check("canonical_inputs", False, "; ".join(errors)))
        return {
            "schema": REPORT_SCHEMA, "overall_status": "red", "dataset_root": str(root),
            "checks": checks, "errors": errors, "input_sha256": {}, "asset_sha256": {},
            "benchmark_only": True, "training_authorized": False,
        }

    try:
        train_rows = validator.load_jsonl(train_path)
        blind_rows = validator.load_jsonl(blind_path)
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"failed to load canonical gate inputs: {error}")
        train_rows, blind_rows, policy, attestation = [], [], {}, {}
    checks.append(check("canonical_inputs", not errors, "canonical manifests and access policy loaded" if not errors else errors[-1]))
    if not errors:
        row_errors, train_summary = validator.validate_rows(train_rows, root, {"train", "dev"})
        blind_errors, blind_summary = validator.validate_rows(blind_rows, root, {"blind"})
        coverage_errors = validator.validate_v3_coverage(train_summary, blind_summary)
        isolation_errors = validator.validate_access_lock(root, train_rows, blind_rows)
        policy_lock_errors = policy_errors(policy, train_rows, blind_rows)
        provenance_errors = privacy_and_source_errors(train_rows + blind_rows)
        attestation_errors = source_attestation_errors(root, train_rows + blind_rows, attestation)
        hashes, hash_errors = asset_hashes(root, train_rows + blind_rows)
        errors.extend(row_errors + blind_errors + coverage_errors + isolation_errors + policy_lock_errors + provenance_errors + attestation_errors + hash_errors)
        checks.extend([
            check("300_train_dev_plus_120_blind", not coverage_errors, "requires six 50-frame train/dev sequences and two 60-frame blind sequences"),
            check("four_class_semantic_masks", not (row_errors + blind_errors) and not coverage_errors, "all masks are dimension-matched 0..3 IDs and all four classes occur in train/dev"),
            check("asset_sha256", not hash_errors, f"{len(hashes)} image/mask hashes match manifest declarations"),
            check("source_and_privacy", not provenance_errors, "source license fields and allowed privacy status are present"),
            check("source_attestation", not attestation_errors, "source receipts, evidence and inventories are SHA256-bound"),
            check("label_authority", not (row_errors + blind_errors), "train authority is explicit; dev/blind are source-ground-truth only"),
            check("session_isolation", not isolation_errors and not policy_lock_errors, "train/dev and exactly two benchmark_only blind sessions are disjoint"),
        ])
    else:
        train_summary = {"row_count": 0, "sequence_count": 0}
        blind_summary = {"row_count": 0, "sequence_count": 0}
        hashes = {}
    input_hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in required if path.is_file()}
    return {
        "schema": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "overall_status": "green" if not errors else "red",
        "training_authorized": not errors,
        "benchmark_only": True,
        "canonical_training_manifest": CANONICAL_TRAINING_MANIFEST,
        "blind_holdout": {
            "manifest": CANONICAL_BLIND_MANIFEST,
            "access": "benchmark_only",
            "session_count": len({str(row.get("session_id") or row.get("source", {}).get("session_id") or "").strip() for row in blind_rows if str(row.get("session_id") or row.get("source", {}).get("session_id") or "").strip()}),
        },
        "training": train_summary,
        "blind": blind_summary,
        "input_sha256": input_hashes,
        "asset_sha256": hashes,
        "checks": checks,
        "errors": errors,
    }


def write_report(report: dict[str, Any], report_path: Path) -> tuple[Path, str]:
    report_path = report_path.resolve()
    if "blind_holdout" in report_path.parts:
        raise ValueError("gate report must not be written inside blind_holdout")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256_file(report_path)
    report_path.with_suffix(report_path.suffix + ".sha256").write_text(f"{digest}  {report_path.name}\n", encoding="ascii")
    return report_path, digest


def run_gate(dataset_root: Path, report_path: Path) -> dict[str, Any]:
    report = build_report(dataset_root)
    _, digest = write_report(report, report_path)
    report["report_sha256"] = digest
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run_gate(args.dataset_root, args.report)
    print(json.dumps({"overall_status": report["overall_status"], "report_sha256": report["report_sha256"]}, ensure_ascii=False))
    return 0 if report["overall_status"] == "green" else 1


if __name__ == "__main__":
    raise SystemExit(main())
