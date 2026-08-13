#!/usr/bin/env python3
"""Metadata-only auditor for the consumed D3R2 coverage-census stop.

The auditor verifies JSON seals, the continuous checkpoint prefix, source file
names and lengths, the failure receipt, and the non-resumable marker.  It never
opens or hashes a retained source archive and never reads a ZIP directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d3r2_phase_b_source_coverage_recovery_protocol_v1"
ACTIVATION_SCHEMA = "blindassist_depthart_task_preserving_d3r2_phase_b_source_coverage_activation_v1"
ATTEMPT_SCHEMA = "blindassist_depthart_task_preserving_d3r2_phase_b_source_coverage_attempt_v1"
CHECKPOINT_SCHEMA = "blindassist_depthart_task_preserving_d3r2_phase_b_source_coverage_asset_checkpoint_v1"
FAILURE_SCHEMA = "blindassist_depthart_task_preserving_d3r2_phase_b_source_coverage_asset_failure_v1"
AUDIT_SCHEMA = "blindassist_depthart_task_preserving_d3r2_phase_b_source_coverage_execution_stop_inventory_audit_v1"

EXPECTED_PROTOCOL_BYTES = 15234
EXPECTED_PROTOCOL_SHA256 = "4BF9023F76C8933B08CEB70D2FC498DA4CE6B892D0A0FD5D4B063E8C60B3F041"
EXPECTED_ACTIVATION_BYTES = 4853
EXPECTED_ACTIVATION_SHA256 = "601BD550DEDB986456A02F753247CF3E32CCB3496A968E2F55123B802B7F8A52"
EXPECTED_COMPLETED_ASSETS = 44
EXPECTED_FAILURE_ORDER = 45
EXPECTED_FAILURE_SELECTION_ORDER = 23
EXPECTED_FAILURE_POOL_ORDER = 58
EXPECTED_FAILURE_VISIT_ID = "466238"
EXPECTED_FAILURE_VIDEO_ID = "44796744"
EXPECTED_FAILURE_ASSET = "lowres_depth.zip"
EXPECTED_OUTPUT = REPO_ROOT / "artifacts.local/evidence/hftf/depthart-task-preserving-d3r2-phase-b-source-coverage-execution-stop-audit-20260813-r0/inventory-audit.json"


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
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def load_sealed_metadata(path: Path, schema: str) -> dict[str, Any]:
    require(path.is_file(), f"sealed metadata missing: {path}")
    payload = path.read_bytes()
    sidecar_path = path.with_suffix(".sha256.json")
    require(sidecar_path.is_file(), f"metadata sidecar missing: {sidecar_path}")
    sidecar = load_json(sidecar_path)
    require(
        sidecar == {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest().upper(),
        },
        f"metadata seal mismatch: {path}",
    )
    value = json.loads(payload.decode("utf-8"))
    require(value.get("schema") == schema, f"metadata schema drift: {path}")
    return value


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def checkpoint_name(row: dict[str, Any]) -> str:
    token = str(row["asset"]).replace(".zip", "")
    return f"{int(row['request_order']):03d}-{row['video_id']}-{token}.json"


def validate_success_history(receipt: dict[str, Any]) -> None:
    history = receipt["attempt_history"]
    require(isinstance(history, list) and len(history) == int(receipt["attempts"]), "transport history count drift")
    require(1 <= len(history) <= 3, "transport attempt budget drift")
    for index, event in enumerate(history, start=1):
        require(event["attempt"] == index and event["method"] == "GET", "transport history order/method drift")
        if index < len(history):
            require(event["retry_class"] in {"TRANSIENT_HTTP", "TRANSIENT_TRANSPORT"}, "terminal transport event was retried")
    final = history[-1]
    require(final["http_status"] == 200 and final["error"] is None and final["retry_class"] is None, "completed asset lacks final HTTP 200")


def validate_attempt_root(
    *,
    root: Path,
    protocol: dict[str, Any],
    activation_sha256: str,
    head_machine: dict[str, Any],
    expected_plan_count: int,
    expected_completed: int,
    expected_failure_order: int,
    expected_failure_selection_order: int,
    expected_failure_pool_order: int,
    expected_failure_visit_id: str,
    expected_failure_video_id: str,
    expected_failure_asset: str,
) -> dict[str, Any]:
    require(root.is_dir(), "attempt root missing")
    require(
        {path.name for path in root.iterdir()}
        == {"_temporary_downloads", "attempt.json", "receipts", "source"},
        "attempt root inventory drift",
    )
    require(not (root / "manifest.json").exists(), "scientific/census manifest unexpectedly present")
    require(not (root / "validation.json").exists(), "offline validation unexpectedly present")

    attempt_path = root / "attempt.json"
    attempt_payload = attempt_path.read_bytes()
    attempt = json.loads(attempt_payload.decode("utf-8"))
    require(attempt.get("schema") == ATTEMPT_SCHEMA, "attempt schema drift")
    attempt_sha256 = hashlib.sha256(attempt_payload).hexdigest().upper()
    require(Path(attempt["output_root"]).resolve() == root.resolve(), "attempt root binding drift")
    require(attempt["bindings"]["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256, "attempt protocol binding drift")
    require(attempt["bindings"]["activation_sha256"] == activation_sha256, "attempt activation binding drift")
    require(attempt["scientific_terminal"] is None and attempt["selection_evaluated"] is False, "attempt scientific boundary drift")
    require(attempt["policy"]["max_attempts"] == 3, "attempt retry budget drift")
    require(attempt["policy"]["range_get"] is False and attempt["policy"]["redirect_following"] is False, "attempt transport boundary drift")
    require(attempt["policy"]["pixel_decode"] is False and attempt["policy"]["source_truth"] is False and attempt["policy"]["selection"] is False, "attempt scientific authority widened")
    plan = attempt["asset_plan"]
    require(isinstance(plan, list) and len(plan) == expected_plan_count, "attempt exact plan drift")

    rows = head_machine["assets"]
    require(isinstance(rows, list) and len(rows) == expected_plan_count, "frozen HEAD row count drift")
    heads = {(str(row["video_id"]), str(row["asset"])): row for row in rows}
    require(len(heads) == expected_plan_count, "frozen HEAD plan uniqueness drift")

    receipts_root = root / "receipts"
    checkpoints = sorted(
        path
        for path in receipts_root.glob("[0-9][0-9][0-9]-*.json")
        if not path.name.endswith(".sha256.json")
    )
    require(len(checkpoints) == expected_completed, "checkpoint count drift")
    expected_receipt_names: set[str] = {"failures"}
    expected_source_files: set[Path] = set()
    source_bytes = 0
    partial_missing_asset_observations = 0
    partial_missing_videos: set[str] = set()

    for index, checkpoint_path in enumerate(checkpoints, start=1):
        planned = plan[index - 1]
        require(checkpoint_path.name == checkpoint_name(planned), "checkpoint prefix/name drift")
        expected_receipt_names |= {
            checkpoint_path.name,
            checkpoint_path.with_suffix(".sha256.json").name,
        }
        checkpoint = load_sealed_metadata(checkpoint_path, CHECKPOINT_SCHEMA)
        require(checkpoint["request_order"] == index, "checkpoint request-order drift")
        for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold", "asset"):
            require(checkpoint[key] == planned[key], f"checkpoint plan drift: {key}")
        require(checkpoint["attempt_sha256"] == attempt_sha256, "checkpoint attempt binding drift")
        require(checkpoint["scientific_terminal"] is None and checkpoint["selection_evaluated"] is False, "checkpoint scientific boundary drift")
        require(checkpoint["archive_member_payload_bytes_read"] == 0 and checkpoint["zip_crc_verified"] is False, "checkpoint member/CRC boundary drift")
        require(checkpoint["pixel_decode"] is False and checkpoint["source_truth"] is False, "checkpoint truth boundary drift")
        require(checkpoint["role_assigned"] is False and checkpoint["training"] is False and checkpoint["development_outcome_read"] is False and checkpoint["r2_access"] == "NONE", "checkpoint downstream authority widened")
        receipt = checkpoint["source_asset"]
        validate_success_history(receipt)
        head = heads[(str(planned["video_id"]), str(planned["asset"]))]
        require(receipt["url"] == head["url"] and receipt["response_final_url"] == head["url"], "checkpoint URL drift")
        require(receipt["response_http_status"] == 200, "checkpoint HTTP status drift")
        require(int(receipt["bytes"]) == int(head["content_length_bytes"]), "checkpoint frozen length drift")
        require(receipt["response_etag"] == head["etag"] and receipt["response_last_modified"] == head["last_modified"], "checkpoint frozen header drift")
        require(receipt["range_request_used"] is False and receipt["redirect_followed"] is False, "checkpoint transport boundary drift")
        expected_source = root / "source" / "Training" / str(planned["video_id"]) / str(planned["asset"])
        require(Path(receipt["path"]).resolve() == expected_source.resolve(), "checkpoint source path drift")
        require(expected_source.is_file(), "checkpoint source body missing")
        require(expected_source.stat().st_size == int(receipt["bytes"]), "checkpoint source length drift")
        expected_source_files.add(expected_source.resolve())
        source_bytes += int(receipt["bytes"])
        partial_missing_asset_observations += int(checkpoint["selected_missing_count"])
        if int(checkpoint["selected_missing_count"]) > 0:
            partial_missing_videos.add(str(planned["video_id"]))

    require({path.name for path in receipts_root.iterdir()} == expected_receipt_names, "receipt inventory drift")
    failures_root = receipts_root / "failures"
    failure_files = [
        path for path in failures_root.glob("*.json") if not path.name.endswith(".sha256.json")
    ]
    require(len(failure_files) == 1, "failure receipt count drift")
    failure_path = failure_files[0]
    require(
        {path.name for path in failures_root.iterdir()}
        == {failure_path.name, failure_path.with_suffix(".sha256.json").name},
        "failure receipt inventory drift",
    )
    failure = load_sealed_metadata(failure_path, FAILURE_SCHEMA)
    planned_failure = plan[expected_failure_order - 1]
    require(failure["request_order"] == expected_failure_order, "failure request order drift")
    for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold", "asset"):
        require(failure[key] == planned_failure[key], f"failure plan drift: {key}")
    require(
        (
            failure["selection_order"],
            failure["pool_order"],
            str(failure["visit_id"]),
            str(failure["video_id"]),
            failure["asset"],
        )
        == (
            expected_failure_selection_order,
            expected_failure_pool_order,
            expected_failure_visit_id,
            expected_failure_video_id,
            expected_failure_asset,
        ),
        "frozen failure identity drift",
    )
    require(failure["attempt_sha256"] == attempt_sha256, "failure attempt binding drift")
    require(failure["failure_stage"] == "DOWNLOAD" and failure["error_type"] == "DownloadFailure", "failure class drift")
    require(failure["source_body_retained"] is False, "failed source body retained unexpectedly")
    require(failure["scientific_terminal"] is None and failure["selection_evaluated"] is False and failure["selected_phase_b"] is None and failure["next_gate"] is None, "failure scientific boundary drift")
    require(failure["archive_member_payload_bytes_read"] == 0 and failure["pixel_decode"] is False and failure["source_truth"] is False, "failure media/truth boundary drift")
    require(len(failure["attempt_history"]) == 1, "failure transport history drift")
    event = failure["attempt_history"][0]
    require(
        event == {
            "attempt": 1,
            "method": "GET",
            "http_status": 200,
            "error": "ValueError: download length mismatch",
            "error_type": "ValueError",
            "retry_class": "TERMINAL",
        },
        "failure transport event drift",
    )
    failed_source = root / "source" / "Training" / expected_failure_video_id / expected_failure_asset
    require(not failed_source.exists(), "failed source body unexpectedly present")

    source_parent = root / "source"
    require({path.name for path in source_parent.iterdir()} == {"Training"}, "source fold inventory drift")
    actual_source_files = {
        path.resolve() for path in (source_parent / "Training").rglob("*") if path.is_file()
    }
    require(actual_source_files == expected_source_files, "source file inventory drift")
    expected_identity_dirs = {
        str(row["video_id"]) for row in plan[:expected_completed]
    } | {expected_failure_video_id}
    actual_identity_dirs = {
        path.name for path in (source_parent / "Training").iterdir() if path.is_dir()
    }
    require(actual_identity_dirs == expected_identity_dirs, "source identity inventory drift")
    require(not any((source_parent / "Training" / expected_failure_video_id).iterdir()), "failed identity directory not empty")

    temporary = root / "_temporary_downloads"
    require(temporary.is_dir(), "non-resumable temporary marker missing")
    temporary_entries = list(temporary.rglob("*"))
    require(any(path.is_dir() for path in temporary_entries), "temporary marker directory missing")
    require(not any(path.is_file() for path in temporary_entries), "unexpected temporary payload retained")

    failure_payload = failure_path.read_bytes()
    return {
        "attempt_path": str(attempt_path.resolve()),
        "attempt_bytes": len(attempt_payload),
        "attempt_sha256": attempt_sha256,
        "checkpoint_count": len(checkpoints),
        "checkpoint_sidecar_count": len(checkpoints),
        "retained_source_identity_count": len(actual_identity_dirs),
        "checkpointed_identity_count": len({str(row["video_id"]) for row in plan[:expected_completed]}),
        "uncheckpointed_source_identity_count": 1,
        "retained_source_asset_count": len(actual_source_files),
        "retained_source_body_bytes_from_receipts_and_stat": source_bytes,
        "partial_selected_missing_asset_observation_count": partial_missing_asset_observations,
        "partial_identity_video_ids_with_missing_observation": sorted(partial_missing_videos),
        "failure_receipt_path": str(failure_path.resolve()),
        "failure_receipt_bytes": len(failure_payload),
        "failure_receipt_sha256": hashlib.sha256(failure_payload).hexdigest().upper(),
        "failed_request_order": expected_failure_order,
        "failed_selection_order": expected_failure_selection_order,
        "failed_pool_order": expected_failure_pool_order,
        "failed_visit_id": expected_failure_visit_id,
        "failed_video_id": expected_failure_video_id,
        "failed_asset": expected_failure_asset,
        "temporary_marker_present": True,
        "manifest_present": False,
        "validation_present": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(args.protocol.resolve() == (REPO_ROOT / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D3R2_PHASE_B_SOURCE_COVERAGE_RECOVERY_PROTOCOL_2026-08-12.json").resolve(), "protocol path drift")
    require(args.activation.resolve() == (REPO_ROOT / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D3R2_PHASE_B_SOURCE_COVERAGE_ACTIVATION_2026-08-13.json").resolve(), "activation path drift")
    require(args.output.resolve() == EXPECTED_OUTPUT.resolve(), "audit output path drift")
    require(not args.output.exists() and not args.output.parent.exists(), "audit output root already exists")
    require(args.protocol.stat().st_size == EXPECTED_PROTOCOL_BYTES and sha256_file(args.protocol) == EXPECTED_PROTOCOL_SHA256, "protocol binding drift")
    require(args.activation.stat().st_size == EXPECTED_ACTIVATION_BYTES and sha256_file(args.activation) == EXPECTED_ACTIVATION_SHA256, "activation binding drift")
    protocol = load_json(args.protocol)
    activation = load_json(args.activation)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(activation.get("schema") == ACTIVATION_SCHEMA, "activation schema drift")
    require(activation.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "activation/protocol binding drift")
    expected_root = REPO_ROOT / protocol["output_root"]
    require(args.attempt_root.resolve() == expected_root.resolve(), "attempt-root path drift")
    head_entry = protocol["frozen_files"]["head_machine_result"]
    head_path = REPO_ROOT / head_entry["path"]
    require(head_path.stat().st_size == int(head_entry["bytes"]) and sha256_file(head_path) == head_entry["sha256"], "HEAD machine binding drift")
    head_machine = load_json(head_path)

    summary = validate_attempt_root(
        root=args.attempt_root,
        protocol=protocol,
        activation_sha256=EXPECTED_ACTIVATION_SHA256,
        head_machine=head_machine,
        expected_plan_count=64,
        expected_completed=EXPECTED_COMPLETED_ASSETS,
        expected_failure_order=EXPECTED_FAILURE_ORDER,
        expected_failure_selection_order=EXPECTED_FAILURE_SELECTION_ORDER,
        expected_failure_pool_order=EXPECTED_FAILURE_POOL_ORDER,
        expected_failure_visit_id=EXPECTED_FAILURE_VISIT_ID,
        expected_failure_video_id=EXPECTED_FAILURE_VIDEO_ID,
        expected_failure_asset=EXPECTED_FAILURE_ASSET,
    )
    result = {
        "schema": AUDIT_SCHEMA,
        "status": "D3R2_PHASE_B_SOURCE_COVERAGE_EXECUTION_STOP_INVENTORY_AUDIT_PASS",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "bindings": {
            "protocol": {"path": str(args.protocol), "bytes": EXPECTED_PROTOCOL_BYTES, "sha256": EXPECTED_PROTOCOL_SHA256},
            "activation": {"path": str(args.activation), "bytes": EXPECTED_ACTIVATION_BYTES, "sha256": EXPECTED_ACTIVATION_SHA256},
            "attempt": {"path": summary["attempt_path"], "bytes": summary["attempt_bytes"], "sha256": summary["attempt_sha256"]},
            "failure_receipt": {"path": summary["failure_receipt_path"], "bytes": summary["failure_receipt_bytes"], "sha256": summary["failure_receipt_sha256"]},
        },
        "execution_status": "INVALID_INCOMPLETE",
        "execution_terminal": "D3R2_PHASE_B_COVERAGE_CENSUS_EXECUTION_INVALID_INCOMPLETE",
        "scientific_terminal": None,
        "selection_evaluated": False,
        "selected_phase_b": None,
        "next_gate": None,
        "inventory": summary,
        "audit_scope": {
            "attempt_checkpoint_and_failure_metadata_seals_verified": True,
            "continuous_checkpoint_prefix_verified": True,
            "frozen_head_urls_headers_and_lengths_replayed": True,
            "source_paths_names_and_file_lengths_verified_by_stat": True,
            "source_body_bytes_read": 0,
            "source_body_hashes_computed": False,
            "zip_directories_read": False,
            "archive_member_payload_bytes_read": 0,
            "pixel_decode": False,
            "source_truth_derived": False,
            "partial_coverage_promoted_to_result": False,
        },
        "governance": {
            "attempt_root_immutable": True,
            "attempt_resumable": False,
            "attempt_repair_authorized": False,
            "same_version_rerun_authorized": False,
            "new_recovery_version_automatically_authorized": False,
        },
        "authority": "METADATA_ONLY_EXECUTION_STOP_INVENTORY_AUDIT",
    }
    write_json_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
