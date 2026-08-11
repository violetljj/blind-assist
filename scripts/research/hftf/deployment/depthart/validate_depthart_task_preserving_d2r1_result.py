#!/usr/bin/env python3
"""Validate the governed D2R1 aggregate result and checkpoint seals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d2_phase_b import qualifies
from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d2r1 import (
    CHECKPOINT_SCHEMA,
    MANIFEST_SCHEMA,
    role_assignments,
)
from scripts.research.hftf.deployment.depthart.reseal_depthart_task_preserving_d2r1_checkpoints import (
    RESULT_SCHEMA as RESEAL_SCHEMA,
)
from scripts.research.spatial_calibration_head_r1.materialize_cache import timestamp_from_stem


RESULT_SCHEMA = "blindassist_depthart_task_preserving_d2r1_governed_result_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object required")
    return value


def validate_manifest(
    manifest: dict[str, Any],
    phase_a: dict[str, Any],
    reseal: dict[str, Any] | None = None,
    receipt_root: Path | None = None,
) -> dict[str, Any]:
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema drift")
    require(manifest["terminal"] == "D2R1_SOURCE_SUPPORT_PASS_4_TRAIN_4_DEVELOPMENT_ROLES_LOCKED", "terminal drift")
    require(manifest["identity_count"] == 16 and len(manifest["videos"]) == 16, "identity count drift")
    require(manifest["qualified_identity_count"] == 16, "qualified identity count drift")
    require(manifest["per_frame_truth_rows_saved"] is False, "per-frame truth retention drift")
    require(manifest["rgb_read"] is False and manifest["model_output_read"] is False, "outcome access drift")
    require(manifest["r2_cohort_access"] == "NONE", "R2 access drift")
    selected = [
        {
            "phase_a_order": index + 1,
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "fold": str(row["fold"]),
        }
        for index, row in enumerate(phase_a["selected_phase_b"])
    ]
    require(len(selected) == 16, "Phase-A selected count drift")
    total_windows = 0
    total_decoded = 0
    for expected, video in zip(selected, manifest["videos"], strict=True):
        for key, value in expected.items():
            require(str(video[key]) == str(value), f"video identity drift: {key}")
        require(video.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema drift in manifest")
        require(video["qualified"] is True and not video["qualification_failures"], "qualification drift")
        stems = [str(value) for value in video["selected_frame_stems"]]
        require(len(stems) == 300 and len(set(stems)) == 300, "selected stem count drift")
        timestamps = [timestamp_from_stem(stem) for stem in stems]
        require(all(0 < right - left <= 0.5 for left, right in zip(timestamps, timestamps[1:])), "selected continuity drift")
        passed, failures = qualifies(video["truth_support"], manifest["truth_support_thresholds"])
        require(passed and not failures, "truth support threshold drift")
        require(video["per_frame_truth_rows_saved"] is False, "checkpoint per-frame truth drift")
        require(video["rgb_read"] is False and video["model_output_read"] is False, "checkpoint outcome drift")
        require(video["r2_cohort_access"] == "NONE", "checkpoint R2 drift")
        total_windows += int(video["windows_tested"])
        total_decoded += int(video["decoded_frame_count"])
    expected_roles = role_assignments(manifest["videos"])
    require(manifest["role_assignments"] == expected_roles, "role assignment drift")
    require([row["role"] for row in expected_roles] == ["D2_TRAIN"] * 4 + ["D2_DEVELOPMENT_SEALED"] * 4, "role order drift")
    if receipt_root is not None:
        require(reseal is not None and reseal.get("schema") == RESEAL_SCHEMA, "checkpoint reseal missing")
        require(reseal["status"] == "PASS" and reseal["receipt_count"] == 16, "checkpoint reseal status drift")
        by_name = {row["receipt_name"]: row for row in reseal["receipts"]}
        require(len(by_name) == 16, "checkpoint reseal row drift")
        for expected, video in zip(selected, manifest["videos"], strict=True):
            name = f"{expected['phase_a_order']:02d}-{expected['video_id']}.json"
            receipt = receipt_root / name
            row = by_name[name]
            require(receipt.stat().st_size == int(row["actual_crlf_bytes"]), f"resealed bytes drift: {receipt}")
            require(sha256_file(receipt) == row["actual_crlf_sha256"], f"resealed SHA drift: {receipt}")
            require(load_json(receipt) == video, f"manifest/checkpoint semantic drift: {receipt}")
    return {
        "qualified_identity_count": 16,
        "role_count": 8,
        "train_identity_count": 4,
        "development_sealed_identity_count": 4,
        "total_windows_tested_until_earliest_pass": total_windows,
        "total_frames_decoded_for_support": total_decoded,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--head-result", type=Path, required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--checkpoint-reseal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")
    protocol = load_json(args.protocol)
    phase_a = load_json(args.phase_a_manifest)
    manifest = load_json(args.manifest)
    reseal = load_json(args.checkpoint_reseal)
    expected_hashes = {
        "protocol_sha256": sha256_file(args.protocol),
        "recovery_protocol_sha256": sha256_file(args.recovery_protocol),
        "phase_a_manifest_sha256": sha256_file(args.phase_a_manifest),
        "head_result_sha256": sha256_file(args.head_result),
        "license_receipt_sha256": sha256_file(args.license_receipt),
    }
    for key, value in expected_hashes.items():
        require(manifest[key] == value, f"manifest binding drift: {key}")
    require(protocol["truth_support_thresholds"] == manifest["truth_support_thresholds"], "protocol threshold drift")
    require(reseal["manifest_sha256"] == sha256_file(args.manifest), "reseal manifest binding drift")
    require(reseal["body_protocol_sha256"] == sha256_file(args.protocol), "reseal protocol binding drift")
    summary = validate_manifest(manifest, phase_a, reseal, args.receipt_root)
    expected_hashes["checkpoint_reseal_sha256"] = sha256_file(args.checkpoint_reseal)
    result = {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "terminal": "D2R1_GOVERNED_SOURCE_SUPPORT_PASS_4_TRAIN_4_DEVELOPMENT_ROLES_LOCKED",
        "bindings": expected_hashes,
        "manifest": {
            "path": str(args.manifest),
            "bytes": args.manifest.stat().st_size,
            "sha256": sha256_file(args.manifest),
        },
        "summary": summary,
        "roles": [
            {key: row[key] for key in ("role", "role_order", "phase_a_order", "pool_order", "visit_id", "video_id")}
            for row in manifest["role_assignments"]
        ],
        "per_frame_truth_rows_saved": False,
        "rgb_read": False,
        "model_output_read": False,
        "training_executed": False,
        "development_outcome_opened": False,
        "r2_cohort_access": "NONE",
        "authority": "D2R1_SOURCE_SUPPORT_AND_ROLE_LOCK_ONLY",
        "next_gate": "EXPLICIT_D2_PHASE_C_RGB_HEAD_SCOPE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
