#!/usr/bin/env python3
"""Metadata-only audit of the immutable D3R1 Phase-B incomplete attempt root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_execution_stop_inventory_audit_v1"
PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_body_protocol_v1"
ACTIVATION_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_body_activation_v1"
HEAD_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_asset_header_preflight_v1"
ATTEMPT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_attempt_v1"
CHECKPOINT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_identity_checkpoint_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def metadata_entry(path: Path, root: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing file: {path}")
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}


def sealed_entry(path: Path, root: Path) -> dict[str, Any]:
    result = metadata_entry(path, root)
    result["sha256"] = sha256_file(path)
    return result


def verify_frozen(entry: dict[str, Any], label: str) -> Path:
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = REPO_ROOT / path
    require(path.is_file(), f"{label} missing")
    require(path.stat().st_size == int(entry["bytes"]), f"{label} bytes drift")
    require(sha256_file(path) == entry["sha256"], f"{label} SHA drift")
    return path


def read_checkpoint(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema drift")
    payload = path.read_bytes()
    sidecar = load_json(path.with_suffix(".sha256.json"))
    require(
        sidecar
        == {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest().upper()},
        "checkpoint sidecar drift",
    )
    return value


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            require(written > 0, "audit output write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def audit_inventory(
    protocol_path: Path,
    activation_path: Path,
    head_path: Path,
    attempt_root: Path,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    activation = load_json(activation_path)
    head = load_json(head_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(activation.get("schema") == ACTIVATION_SCHEMA, "activation schema drift")
    require(head.get("schema") == HEAD_SCHEMA, "HEAD schema drift")
    require(activation["protocol_sha256"] == sha256_file(protocol_path), "activation/protocol drift")
    for label, entry in protocol["frozen_files"].items():
        verify_frozen(entry, label)
    require(
        head_path.resolve()
        == verify_frozen(protocol["frozen_files"]["head_machine_result"], "HEAD result").resolve(),
        "passed HEAD path drift",
    )
    root = attempt_root.resolve()
    require(root == (REPO_ROOT / protocol["output_root"]).resolve(), "attempt root drift")
    require(root.is_dir(), "attempt root missing")
    require(not (root / "manifest.json").exists(), "scientific manifest unexpectedly exists")
    require(not (root / "validation.json").exists(), "scientific validation unexpectedly exists")
    require(not (root / "_temporary_downloads").exists(), "temporary download inventory remains")
    require({path.name for path in root.iterdir()} == {"attempt.json", "receipts", "source"}, "attempt-root inventory drift")
    require({path.name for path in (root / "source").iterdir()} == {"Training"}, "source-fold inventory drift")
    training = root / "source" / "Training"
    require({path.name for path in training.iterdir()} == {"42447264", "42898216"}, "source identity inventory drift")
    expected_assets = {"confidence.zip", "lowres_depth.zip"}
    require({path.name for path in (training / "42447264").iterdir()} == expected_assets, "identity-1 asset inventory drift")
    require({path.name for path in (training / "42898216").iterdir()} == expected_assets, "identity-2 asset inventory drift")
    receipts = root / "receipts"
    require(
        {path.name for path in receipts.iterdir()}
        == {"001-42447264.json", "001-42447264.sha256.json"},
        "checkpoint inventory drift",
    )
    attempt = load_json(root / "attempt.json")
    require(attempt.get("schema") == ATTEMPT_SCHEMA, "attempt schema drift")
    require(attempt["bindings"]["protocol_sha256"] == sha256_file(protocol_path), "attempt protocol binding drift")
    require(attempt["bindings"]["activation_sha256"] == sha256_file(activation_path), "attempt activation binding drift")
    require(len(attempt["identity_plan"]) == 32, "attempt identity plan drift")
    first = read_checkpoint(receipts / "001-42447264.json")
    require(first["selection_order"] == 1 and first["video_id"] == "42447264", "checkpoint identity drift")
    require(first["source_truth_support_qualified"] is False, "checkpoint qualification drift")
    require(first["role_assigned"] is False and first["r2_access"] == "NONE", "checkpoint authority drift")
    heads = {(str(row["video_id"]), str(row["asset"])): row for row in head["assets"]}
    require(len(heads) == 64, "HEAD inventory drift")
    body_files: list[dict[str, Any]] = []
    for video_id in ("42447264", "42898216"):
        for asset in ("lowres_depth.zip", "confidence.zip"):
            path = training / video_id / asset
            expected_bytes = int(heads[(video_id, asset)]["content_length_bytes"])
            require(path.stat().st_size == expected_bytes, f"{video_id}/{asset} byte length drift")
            body_files.append(metadata_entry(path, root) | {"head_content_length_bytes": expected_bytes})
    non_body_files = [
        sealed_entry(root / "attempt.json", root),
        sealed_entry(receipts / "001-42447264.json", root),
        sealed_entry(receipts / "001-42447264.sha256.json", root),
    ]
    require(len(list(path for path in root.rglob("*") if path.is_file())) == 7, "attempt file count drift")
    return {
        "schema": SCHEMA,
        "status": "D3R1_PHASE_B_EXECUTION_STOP_INVENTORY_AUDIT_PASS",
        "execution_status": "INVALID_INCOMPLETE",
        "scientific_terminal": None,
        "bindings": {
            "protocol": {"path": str(protocol_path.resolve()), "bytes": protocol_path.stat().st_size, "sha256": sha256_file(protocol_path)},
            "activation": {"path": str(activation_path.resolve()), "bytes": activation_path.stat().st_size, "sha256": sha256_file(activation_path)},
            "head_machine_result": {"path": str(head_path.resolve()), "bytes": head_path.stat().st_size, "sha256": sha256_file(head_path)},
        },
        "attempt_root": str(root),
        "planned_identity_count": 32,
        "checkpointed_identity_count": 1,
        "partial_source_identity_count": 2,
        "attempt_file_count": 7,
        "sealed_non_body_files": non_body_files,
        "body_file_metadata_only": body_files,
        "body_bytes_read_by_auditor": 0,
        "body_hashes_computed_by_auditor": False,
        "archive_members_opened_by_auditor": False,
        "scientific_manifest_present": False,
        "scientific_validation_present": False,
        "temporary_download_inventory_present": False,
        "selection_evaluated": False,
        "phase_b_selection_locked": False,
        "selected_phase_b": None,
        "attempt_resumable": False,
        "attempt_root_immutable": True,
        "rgb_read": False,
        "model_output_read": False,
        "role_assignment_made": False,
        "training": False,
        "development_outcome_read": False,
        "r2_access": "NONE",
        "next_gate": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--head-machine-result", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_inventory(
        args.protocol, args.activation, args.head_machine_result, args.attempt_root
    )
    write_json_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
