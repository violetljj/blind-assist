#!/usr/bin/env python3
"""Fail-closed validator for the TARO O1R reducer-integration protocol lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / "docs/research/taro/TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_PROTOCOL_LOCK_2026-08-11.json"
SCHEMA = "blindassist.taro.o1r.r6_prospective_factor_reducer_integration_protocol_lock.v1"
LOCK_ID = "TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_PROTOCOL_LOCK"
SUCCESSOR = "TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_IMPLEMENTATION_LOCK"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _check_bound_file(errors: list[str], record: Any, prefix: str) -> None:
    if not isinstance(record, dict):
        errors.append(f"{prefix}_RECORD_INVALID")
        return
    relative = record.get("path") or record.get("artifact_path")
    expected_bytes = record.get("bytes") if "bytes" in record else record.get("artifact_bytes")
    expected_hash = record.get("sha256") if "sha256" in record else record.get("artifact_sha256")
    path = REPO_ROOT / str(relative)
    if not path.is_file():
        errors.append(f"{prefix}_MISSING")
        return
    if path.stat().st_size != expected_bytes:
        errors.append(f"{prefix}_BYTES_DRIFT")
    if _sha256(path) != expected_hash:
        errors.append(f"{prefix}_SHA256_DRIFT")


def validate(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"passed": False, "errors": [f"LOCK_READ_ERROR:{type(error).__name__}"]}
    if lock.get("schema") != SCHEMA or lock.get("lock_id") != LOCK_ID or lock.get("status") != "FROZEN":
        errors.append("LOCK_IDENTITY_DRIFT")
    authority = lock.get("authority", {})
    _check_bound_file(errors, authority.get("r6_result"), "R6_RESULT")
    _check_bound_file(errors, authority.get("r6_runtime"), "R6_RUNTIME")
    _check_bound_file(errors, authority.get("fit_only_uncertainty"), "UNCERTAINTY_ARTIFACT")
    _check_bound_file(errors, authority.get("taro_geometry_constants"), "TARO_CONSTANTS")
    uncertainty = authority.get("fit_only_uncertainty", {})
    receipt = REPO_ROOT / str(uncertainty.get("receipt_path"))
    if not receipt.is_file() or receipt.stat().st_size != uncertainty.get("receipt_bytes") or _sha256(receipt) != uncertainty.get("receipt_sha256"):
        errors.append("UNCERTAINTY_RECEIPT_DRIFT")
    contract = lock.get("integration_contract", {})
    if contract.get("public_builder") != "integrate_prospective_factor_bundle" or contract.get("exact_query_slots") != 9:
        errors.append("INTEGRATION_API_DRIFT")
    if contract.get("forbidden_public_input_tokens") != ["faro", "truth", "outcome", "task_metric"]:
        errors.append("RESULT_SIDE_FIREWALL_DRIFT")
    if contract.get("query_value") != "R6_QUERY_CLEARANCE_R1_BASELINE_CLEARANCE_M" or contract.get("sole_final_state_producer") != "TARO_O1R_R6_SOURCE_ONLY_INTERVAL_REDUCER_V1":
        errors.append("REDUCER_AUTHORITY_DRIFT")
    fail_closed = lock.get("fail_closed_contract", {})
    if any(fail_closed.get(key) != "UNKNOWN" for key in ("query_frame_missing", "factor_not_evaluable", "uncertainty_resolution_invalid", "query_support_insufficient")):
        errors.append("FAIL_CLOSED_DRIFT")
    firewalls = lock.get("firewalls", {})
    if any(firewalls.get(key) != 0 for key in ("faro_reads_during_integration", "training_steps_during_integration", "network_requests_during_integration")):
        errors.append("EXECUTION_FIREWALL_DRIFT")
    if lock.get("unique_successor") != SUCCESSOR:
        errors.append("SUCCESSOR_DRIFT")
    return {
        "schema": "blindassist.taro.o1r.r6_prospective_factor_reducer_integration_protocol_lock_validation.v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "terminal": "TARO_O1R_R6_REDUCER_INTEGRATION_PROTOCOL_VALID" if not errors else "TARO_O1R_R6_REDUCER_INTEGRATION_PROTOCOL_INVALID",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    result = validate(args.lock.resolve())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
