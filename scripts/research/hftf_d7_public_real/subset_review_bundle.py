#!/usr/bin/env python3
"""Create a fresh review batch from an existing batch without re-downloading evidence."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
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


def _copy_artifact(source_value: str | None, destination: Path) -> tuple[str | None, str | None]:
    if not source_value:
        return None, None
    source = Path(source_value).resolve()
    if not source.is_file():
        raise ContractError(f"source review evidence is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination.resolve()), sha256_file(destination)


def _rematerialize_row(row: dict[str, Any], *, source_root: Path, output_root: Path, role: str, batch_id: str, index: int) -> dict[str, Any]:
    result = copy.deepcopy(row)
    candidate_id = str(result["candidate_id"])
    result["batch_id"] = batch_id
    result["review_input_id"] = stable_id("d7review-input", batch_id, role, candidate_id)
    for field, subdirectory, suffix in (
        ("contact_sheet_path", "contact_sheets", ".jpg"),
        ("temporal_manifest_path", "temporal_manifests", ".jsonl"),
        ("native_geometry_path", "native_geometry", ".json"),
    ):
        source_value = result.get(field)
        if not source_value:
            result[field] = None
            hash_field = field.replace("_path", "_sha256")
            result[hash_field] = None
            continue
        destination = output_root / role / subdirectory / f"{candidate_id}{suffix}"
        copied_path, copied_hash = _copy_artifact(str(source_value), destination)
        result[field] = copied_path
        result[field.replace("_path", "_sha256")] = copied_hash
    result["review_index"] = index
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_batch_root).resolve()
    output_root = Path(args.output_batch_root).resolve()
    if output_root.exists():
        raise ContractError(f"refusing to overwrite subset review batch: {output_root}")
    source_manifest = load_json(source_root / "bundle_manifest.json")
    source_ids = {str(value) for value in source_manifest.get("candidate_ids", [])}
    excluded = {str(value) for value in args.exclude_candidate_id}
    included = {str(value) for value in args.include_candidate_id}
    if not excluded.issubset(source_ids):
        raise ContractError(f"excluded candidate is not in source batch: {sorted(excluded - source_ids)}")
    if not included.issubset(source_ids):
        raise ContractError(f"included candidate is not in source batch: {sorted(included - source_ids)}")
    if included and excluded.intersection(included):
        raise ContractError("a candidate cannot be both included and excluded")
    selected_ids = included if included else source_ids - excluded
    if not selected_ids:
        raise ContractError("subset selection is empty")
    output_root.mkdir(parents=True, exist_ok=False)
    selected_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in ROLES:
        source_manifest_path = source_root / "manifests" / f"{role}.jsonl"
        rows = load_jsonl(source_manifest_path)
        role_ids = {str(row.get("candidate_id")) for row in rows}
        if role_ids != source_ids:
            raise ContractError(f"source role candidate set differs from bundle: {role}")
        selected = [row for row in rows if str(row["candidate_id"]) in selected_ids]
        selected.sort(key=lambda row: str(row["candidate_id"]))
        selected_by_role[role] = [
            _rematerialize_row(row, source_root=source_root, output_root=output_root, role=role, batch_id=args.batch_id, index=index)
            for index, row in enumerate(selected)
        ]
        manifest_path = output_root / "manifests" / f"{role}.jsonl"
        write_jsonl(manifest_path, selected_by_role[role])
    candidate_ids = sorted(selected_ids)
    bundle_manifest = {
        "schema": "hftf_d7_public_real_review_bundle_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_ISOLATED_REVIEW",
        "dataset_id": source_manifest.get("dataset_id"),
        "candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "roles": {
            role: {
                "manifest_path": str((output_root / "manifests" / f"{role}.jsonl").resolve()),
                "manifest_sha256": sha256_file(output_root / "manifests" / f"{role}.jsonl"),
                "row_count": len(selected_by_role[role]),
                "input_scope": "SOURCE_NATIVE_GEOMETRY_ONLY" if role == "GEOMETRY_EVIDENCE_REVIEWER" else "RGB_ONLY",
            }
            for role in ROLES
        },
        "source_batch_root": str(source_root),
        "source_bundle_manifest_sha256": sha256_file(source_root / "bundle_manifest.json"),
        "source_media_reused_read_only": True,
        "model_output_visible_in_any_input": False,
        "review_assignments_are_not_labels": True,
        "final_adjudication_written": False,
        "notes": [
            "This batch is a non-overlapping subset of an existing downloaded evidence bundle.",
            "Provider media/depth files remain read-only references to the source batch; role evidence paths are copied and re-hashed.",
            "No review decision, event bucket, phase interval, or admission result is created by this command.",
        ],
    }
    write_json(output_root / "bundle_manifest.json", bundle_manifest)
    receipt = {
        "schema": "hftf_d7_public_real_review_bundle_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": bundle_manifest["generated_at_utc"],
        "status": "READY_FOR_ISOLATED_REVIEW",
        "output_root": str(output_root.parent.parent.parent.resolve()),
        "batch_root": str(output_root),
        "dataset_id": bundle_manifest.get("dataset_id"),
        "candidate_count": len(candidate_ids),
        "review_roles": list(ROLES),
        "bundle_manifest": {"path": str((output_root / "bundle_manifest.json").resolve()), "sha256": sha256_file(output_root / "bundle_manifest.json")},
        "manifest_files": {
            role: {
                "path": str((output_root / "manifests" / f"{role}.jsonl").resolve()),
                "sha256": sha256_file(output_root / "manifests" / f"{role}.jsonl"),
            }
            for role in ROLES
        },
        "source_media_reused_read_only": True,
        "model_output_visible_in_any_input": False,
        "review_assignments_are_not_labels": True,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
    }
    receipt_path = output_root.parent.parent.parent / "receipts" / f"review_bundle_receipt_{args.batch_id}.json"
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-batch-root", required=True)
    parser.add_argument("--output-batch-root", required=True)
    parser.add_argument("--exclude-candidate-id", action="append", default=[])
    parser.add_argument(
        "--include-candidate-id",
        action="append",
        default=[],
        help="select exactly these candidate IDs; repeat for a deterministic re-review sample",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
