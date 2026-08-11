#!/usr/bin/env python3
"""Validate the sealed D2 Phase-C exact-eight source materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d2_phase_c_sources import (
    MANIFEST_SCHEMA,
    load_json,
    read_checkpoint,
    role_rows,
    sha256_file,
)


RESULT_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_c_source_governed_result_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest_shape(manifest: dict[str, Any], expected_roles: list[dict[str, Any]]) -> None:
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema drift")
    require(manifest["terminal"] == "D2_PHASE_C_SOURCE_MATERIALIZATION_PASS_EXACT_EIGHT_SEALED", "terminal drift")
    require(manifest["identity_count"] == 8 and len(manifest["roles"]) == 8, "identity count drift")
    require(manifest["train_identity_count"] == 4 and manifest["development_sealed_identity_count"] == 4, "role count drift")
    require(manifest["source_asset_count"] == 32 and manifest["extracted_file_count"] == 9600, "source count drift")
    require(manifest["exact_total_body_bytes"] == 5281655713, "body total drift")
    require(manifest["image_decode"] is False and manifest["truth_derivation"] is False, "source decode drift")
    require(manifest["model_output_read"] is False and manifest["training_executed"] is False, "model authority drift")
    require(manifest["development_outcome_opened"] is False and manifest["r2_cohort_access"] == "NONE", "sealed authority drift")
    for actual, expected in zip(manifest["roles"], expected_roles, strict=True):
        for key in ("role", "role_order", "phase_a_order", "pool_order", "visit_id", "video_id", "selected_frame_stems"):
            require(actual[key] == expected[key], f"role binding drift: {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scope-protocol", type=Path, required=True)
    parser.add_argument("--d2r1-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")
    protocol = load_json(args.protocol)
    d2r1 = load_json(args.d2r1_manifest)
    manifest = load_json(args.source_manifest)
    expected_bindings = {
        "protocol_sha256": sha256_file(args.protocol),
        "scope_protocol_sha256": sha256_file(args.scope_protocol),
        "d2r1_manifest_sha256": sha256_file(args.d2r1_manifest),
    }
    for key, value in expected_bindings.items():
        require(manifest[key] == value, f"manifest binding drift: {key}")
    expected_roles = role_rows(d2r1)
    validate_manifest_shape(manifest, expected_roles)
    source_root = (args.source_manifest.parent / "source").resolve()
    all_paths: set[Path] = set()
    extracted_bytes = 0
    checkpoint_rows = []
    for index, (manifest_role, expected_role) in enumerate(zip(manifest["roles"], expected_roles, strict=True), start=1):
        checkpoint_path = args.receipt_root / f"{index:02d}-{expected_role['video_id']}.json"
        require(sha256_file(checkpoint_path) == manifest_role["checkpoint_sha256"], "manifest checkpoint SHA drift")
        checkpoint = read_checkpoint(checkpoint_path, expected_role)
        expected_folder = "train" if expected_role["role"] == "D2_TRAIN" else "development_sealed"
        for modality, entries in checkpoint["extracted"].items():
            stems = [Path(entry["path"]).stem for entry in entries]
            require(stems == expected_role["selected_frame_stems"], f"extracted stem order drift: {modality}")
            for entry in entries:
                path = Path(entry["path"]).resolve()
                require(source_root in path.parents, f"source path escaped root: {path}")
                require(expected_folder in path.parts, f"role folder drift: {path}")
                require(path not in all_paths, f"duplicate extracted path: {path}")
                all_paths.add(path)
                extracted_bytes += int(entry["bytes"])
        checkpoint_rows.append(
            {
                "role": expected_role["role"],
                "role_order": expected_role["role_order"],
                "visit_id": expected_role["visit_id"],
                "video_id": expected_role["video_id"],
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "extracted_file_count": checkpoint["extracted_file_count"],
            }
        )
    require(len(all_paths) == 9600, "unique extracted path count drift")
    value = {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "terminal": "D2_PHASE_C_GOVERNED_SOURCE_MATERIALIZATION_PASS_EXACT_EIGHT_SEALED",
        "bindings": expected_bindings,
        "source_manifest": {
            "path": str(args.source_manifest),
            "bytes": args.source_manifest.stat().st_size,
            "sha256": sha256_file(args.source_manifest),
        },
        "identity_count": 8,
        "train_identity_count": 4,
        "development_sealed_identity_count": 4,
        "source_asset_count": 32,
        "extracted_file_count": len(all_paths),
        "extracted_file_bytes": extracted_bytes,
        "checkpoints": checkpoint_rows,
        "image_decode": False,
        "truth_derivation": False,
        "model_output_read": False,
        "training_executed": False,
        "development_outcome_opened": False,
        "r2_cohort_access": "NONE",
        "authority": "D2_PHASE_C_EXACT_SOURCE_MATERIALIZATION_ONLY",
        "next_gate": "EXPLICIT_D2_TRAIN_ONLY_BASE_OUTPUT_AND_HEAD_TRAINING_ACTIVATION",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in value.items() if key != "checkpoints"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
