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
import zlib
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
ASSET_INVENTORY_SCHEMA = "blindassist_source_asset_inventory_v1"
ASSEMBLY_RECIPE = "assembly_recipe.json"
ASSET_INVENTORY = "source_asset_inventory.json"
ALLOWED_SOURCE_ADAPTERS = {
    "sanpo_v0",
    "bdd100k_v1",
    "guidetwsi_v1",
    "procedural_tactile_v1",
    "teacher_consensus_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crc32_file(path: Path) -> str:
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return f"{checksum & 0xffffffff:08x}"


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
    for prefix in ("recipe", "asset_inventory"):
        verify_bound_file(root, payload, prefix, errors)
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
        if row.get("label_authority") == "procedural_ground_truth":
            provenance = row.get("label_provenance") if isinstance(row.get("label_provenance"), dict) else {}
            inputs = provenance.get("source_masks") if isinstance(provenance.get("source_masks"), list) else []
            for item in inputs:
                input_source_id = str(item.get("source_id", "")).strip() if isinstance(item, dict) else ""
                if input_source_id not in by_id:
                    errors.append(
                        f"{sample_id}: procedural input source_id {input_source_id!r} is missing from source attestation"
                    )
    return errors


def source_asset_inventory_errors(
    root: Path, rows: list[dict[str, Any]], payload: dict[str, Any], attestation: dict[str, Any],
) -> list[str]:
    """Require a one-to-one, SHA-bound raw-evidence inventory for every sample."""
    errors: list[str] = []
    if payload.get("schema") != ASSET_INVENTORY_SCHEMA:
        errors.append(f"source asset inventory schema must be {ASSET_INVENTORY_SCHEMA}")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        return errors + ["source asset inventory requires an assets list"]
    source_ids = {
        str(item.get("source_id", "")).strip()
        for item in attestation.get("sources", []) if isinstance(item, dict)
    }
    remote_inventories: dict[str, dict[str, Any]] = {}
    for receipt in attestation.get("sources", []):
        if not isinstance(receipt, dict):
            continue
        source_id = str(receipt.get("source_id", "")).strip()
        relative = str(receipt.get("inventory_path", "")).strip()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_file():
            try:
                remote_inventories[source_id] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    row_by_id = {str(row.get("id", "")): row for row in rows}
    by_id: dict[str, dict[str, Any]] = {}
    for item in assets:
        if not isinstance(item, dict):
            errors.append("source asset inventory entries must be objects")
            continue
        entry_id = str(item.get("entry_id", "")).strip()
        if not entry_id or entry_id in by_id:
            errors.append(f"duplicate or missing source asset inventory id: {entry_id!r}")
            continue
        by_id[entry_id] = item
        sample_id = str(item.get("sample_id", "")).strip()
        row = row_by_id.get(sample_id)
        if row is None:
            errors.append(f"{entry_id}: inventory references unknown sample {sample_id!r}")
            continue
        if str(item.get("source_id", "")).strip() not in source_ids:
            errors.append(f"{entry_id}: inventory source_id is not attested")
        if item.get("session_id") != row.get("session_id") or item.get("frame_index") != row.get("frame_index"):
            errors.append(f"{entry_id}: inventory session/frame differs from sample")
        relative = str(item.get("path", "")).strip()
        expected = str(item.get("sha256", "")).strip()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"{entry_id}: raw evidence path escapes dataset root")
            continue
        if len(expected) != 64 or not path.is_file() or sha256_file(path) != expected:
            errors.append(f"{entry_id}: raw evidence file is missing or SHA256 differs")
            continue
        role = str(item.get("role", ""))
        if role in {"guide_rgb", "guide_polygon"}:
            receipt = item.get("remote_receipt") if isinstance(item.get("remote_receipt"), dict) else {}
            inventory = remote_inventories.get(str(item.get("source_id", "")), {})
            members = inventory.get("members") if isinstance(inventory.get("members"), list) else []
            member_path = str(receipt.get("origin_member_path", ""))
            member = next(
                (candidate for candidate in members if isinstance(candidate, dict) and candidate.get("path") == member_path),
                None,
            )
            if member is None:
                errors.append(f"{entry_id}: Guide origin member is absent from attested remote inventory")
                continue
            if int(receipt.get("size", -1)) != path.stat().st_size or int(member.get("size", -2)) != path.stat().st_size:
                errors.append(f"{entry_id}: Guide origin member size differs from raw asset")
            actual_crc = crc32_file(path)
            if str(receipt.get("crc32", "")).lower() != actual_crc or str(member.get("crc32", "")).lower() != actual_crc:
                errors.append(f"{entry_id}: Guide origin member CRC32 differs from raw asset")
            archive = receipt.get("archive") if isinstance(receipt.get("archive"), dict) else {}
            source = inventory.get("source") if isinstance(inventory.get("source"), dict) else {}
            for field in ("etag", "generation", "md5_base64"):
                if not str(archive.get(field, "")).strip() or archive.get(field) != source.get(field):
                    errors.append(f"{entry_id}: Guide archive {field} differs from attested remote inventory")

    referenced: list[str] = []
    for row in rows:
        sample_id = str(row.get("id", "<unknown>"))
        ids = row.get("source_asset_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(value, str) for value in ids):
            errors.append(f"{sample_id}: source_asset_ids must be a non-empty string list")
            continue
        if len(ids) != len(set(ids)):
            errors.append(f"{sample_id}: source_asset_ids contains duplicates")
        referenced.extend(ids)
        selected = [by_id[value] for value in ids if value in by_id]
        if len(selected) != len(ids):
            errors.append(f"{sample_id}: source_asset_ids references missing inventory entries")
            continue
        if any(str(item.get("sample_id", "")) != sample_id for item in selected):
            errors.append(f"{sample_id}: source asset belongs to another sample")
        authority = row.get("label_authority")
        required_roles = (
            {"guide_rgb", "guide_polygon", "sanpo_rgb", "sanpo_raw_mask"}
            if authority == "procedural_ground_truth" else {"sanpo_rgb", "sanpo_raw_mask"}
        )
        actual_roles = {str(item.get("role", "")) for item in selected}
        if actual_roles != required_roles:
            errors.append(f"{sample_id}: raw evidence roles must be {sorted(required_roles)}")
        provenance = row.get("label_provenance") if isinstance(row.get("label_provenance"), dict) else {}
        declared_assets = provenance.get("source_assets")
        if not isinstance(declared_assets, list):
            errors.append(f"{sample_id}: label provenance lacks raw source_assets")
            continue
        declared = {
            (str(item.get("role", "")), str(item.get("source_id", "")), str(item.get("path", "")), str(item.get("sha256", "")))
            for item in declared_assets if isinstance(item, dict)
        }
        inventoried = {
            (str(item.get("role", "")), str(item.get("source_id", "")), str(item.get("path", "")), str(item.get("sha256", "")))
            for item in selected
        }
        if declared != inventoried:
            errors.append(f"{sample_id}: provenance raw assets differ from source inventory")
        declared_by_role = {
            str(item.get("role", "")): item for item in declared_assets if isinstance(item, dict)
        }
        inventoried_by_role = {str(item.get("role", "")): item for item in selected}
        for role in {"guide_rgb", "guide_polygon"} & actual_roles:
            if declared_by_role[role].get("remote_receipt") != inventoried_by_role[role].get("remote_receipt"):
                errors.append(f"{sample_id}: {role} remote receipt differs from source inventory")
        if authority == "source_ground_truth":
            by_role = {str(item.get("role", "")): item for item in selected}
            raw_image = by_role.get("sanpo_rgb", {})
            raw_mask = by_role.get("sanpo_raw_mask", {})
            if raw_image.get("path") != row.get("image_path") or raw_image.get("sha256") != row.get("image_sha256"):
                errors.append(f"{sample_id}: source-ground-truth RGB is not the canonical input image")
            if raw_mask.get("sha256") != provenance.get("source_mask_sha256"):
                errors.append(f"{sample_id}: source-ground-truth raw mask is not bound to label provenance")
    if len(referenced) != len(set(referenced)):
        errors.append("source asset inventory entries must be referenced by exactly one sample")
    if set(referenced) != set(by_id):
        errors.append("source asset inventory has missing or unreferenced entries")
    splits_by_asset: dict[tuple[str, str, str], set[str]] = {}
    for item in assets:
        if not isinstance(item, dict):
            continue
        row = row_by_id.get(str(item.get("sample_id", "")))
        if row is None:
            continue
        identity = (
            str(item.get("source_id", "")), str(item.get("role", "")), str(item.get("sha256", "")),
        )
        splits_by_asset.setdefault(identity, set()).add(str(row.get("split", "")))
    for (source_id, role, digest), splits in splits_by_asset.items():
        if len(splits) > 1:
            errors.append(
                f"raw source asset crosses splits: {source_id}/{role}/{digest} appears in {sorted(splits)}"
            )
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
    recipe_path = root / ASSEMBLY_RECIPE
    inventory_path = root / ASSET_INVENTORY
    errors: list[str] = []
    checks: list[dict[str, Any]] = []
    required = [train_path, blind_path, policy_path, attestation_path, recipe_path, inventory_path]
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
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"failed to load canonical gate inputs: {error}")
        train_rows, blind_rows, policy, attestation, recipe, inventory = [], [], {}, {}, {}, {}
    checks.append(check("canonical_inputs", not errors, "canonical manifests and access policy loaded" if not errors else errors[-1]))
    if not errors:
        row_errors, train_summary = validator.validate_rows(train_rows, root, {"train", "dev"})
        blind_errors, blind_summary = validator.validate_rows(blind_rows, root, {"blind"})
        coverage_policy = recipe.get("coverage_policy") if isinstance(recipe, dict) else None
        if not isinstance(coverage_policy, dict):
            coverage_policy = None
        coverage_errors = validator.validate_v3_coverage(
            train_summary, blind_summary, coverage_policy,
        )
        isolation_errors = validator.validate_access_lock(root, train_rows, blind_rows)
        policy_lock_errors = policy_errors(policy, train_rows, blind_rows)
        provenance_errors = privacy_and_source_errors(train_rows + blind_rows)
        attestation_errors = source_attestation_errors(root, train_rows + blind_rows, attestation)
        inventory_errors = source_asset_inventory_errors(root, train_rows + blind_rows, inventory, attestation)
        training_hashes, training_hash_errors = asset_hashes(root, train_rows)
        blind_hashes, blind_hash_errors = asset_hashes(root, blind_rows)
        hashes = dict(sorted({**training_hashes, **blind_hashes}.items()))
        hash_errors = training_hash_errors + blind_hash_errors
        errors.extend(row_errors + blind_errors + coverage_errors + isolation_errors + policy_lock_errors + provenance_errors + attestation_errors + inventory_errors + hash_errors)
        checks.extend([
            check(
                "expanded_session_scene_coverage" if coverage_policy else "300_train_dev_plus_120_blind",
                not coverage_errors,
                (
                    "recipe-bound minimum train/dev sessions, per-scene session coverage, official split separation, and two real blind sessions"
                    if coverage_policy
                    else "requires six 50-frame train/dev sequences and two 60-frame blind sequences"
                ),
            ),
            check(
                "duplicate_mask_observation",
                True,
                "train/dev=" + json.dumps(train_summary.get("duplicate_mask_observation", {}), sort_keys=True)
                + "; blind=" + json.dumps(blind_summary.get("duplicate_mask_observation", {}), sort_keys=True),
            ),
            check("four_class_semantic_masks", not (row_errors + blind_errors) and not coverage_errors, "all masks are dimension-matched 0..3 IDs and all four classes occur in train/dev"),
            check("asset_sha256", not hash_errors, f"{len(hashes)} image/mask hashes match manifest declarations"),
            check("source_and_privacy", not provenance_errors, "source license fields and allowed privacy status are present"),
            check("source_attestation", not attestation_errors, "source receipts, evidence and inventories are SHA256-bound"),
            check("raw_asset_closure", not inventory_errors, "recipe and every raw source asset are inventoried, copied and SHA256-bound"),
            check("label_authority", not (row_errors + blind_errors), "authority is explicit; dev/blind allow only source GT or fully SHA256-bound procedural GT and always reject pseudo labels"),
            check("session_isolation", not isolation_errors and not policy_lock_errors, "train/dev and exactly two benchmark_only blind sessions are disjoint"),
        ])
    else:
        train_summary = {"row_count": 0, "sequence_count": 0}
        blind_summary = {"row_count": 0, "sequence_count": 0}
        hashes = {}
        training_hashes = {}
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
        "training_asset_sha256": training_hashes,
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


def consume_training_authorization(dataset_root: Path, report_path: Path) -> dict[str, Any]:
    """Verify a precomputed authorization without reading any blind asset."""
    root = dataset_root.resolve()
    report_path = report_path.resolve()
    sidecar_path = report_path.with_suffix(report_path.suffix + ".sha256")
    if not report_path.is_file() or not sidecar_path.is_file():
        raise ValueError("precomputed training-gate report and SHA256 sidecar are required")
    digest = sha256_file(report_path)
    sidecar_digest = sidecar_path.read_text(encoding="ascii").strip().split()[0]
    if digest != sidecar_digest:
        raise ValueError("training-gate report SHA256 sidecar mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("training-gate report schema mismatch")
    if Path(str(report.get("dataset_root", ""))).resolve() != root:
        raise ValueError("training-gate report is bound to a different dataset root")
    if (
        report.get("overall_status") != "green"
        or report.get("training_authorized") is not True
        or report.get("benchmark_only") is not True
        or report.get("errors")
        or any(item.get("status") != "green" for item in report.get("checks", []))
    ):
        raise ValueError("training-gate report does not authorize benchmark-only training")
    if report.get("canonical_training_manifest") != CANONICAL_TRAINING_MANIFEST:
        raise ValueError("training-gate report does not pin the canonical training manifest")
    manifest = root / CANONICAL_TRAINING_MANIFEST
    expected_manifest = report.get("input_sha256", {}).get(CANONICAL_TRAINING_MANIFEST)
    if not manifest.is_file() or sha256_file(manifest) != expected_manifest:
        raise ValueError("canonical training manifest differs from the authorized SHA256")
    training_assets = report.get("training_asset_sha256")
    if not isinstance(training_assets, dict) or not training_assets:
        raise ValueError("training-gate report lacks bound train/dev assets")
    for relative, expected in training_assets.items():
        parts = Path(relative).parts
        if not parts or parts[0] == "blind_holdout":
            raise ValueError("training authorization contains a blind asset")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("authorized training asset escapes dataset root") from error
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"authorized training asset changed: {relative}")
    verified = dict(report)
    verified["report_sha256"] = digest
    return verified


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
