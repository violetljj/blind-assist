#!/usr/bin/env python3
"""Validate exact TARO O1R reducer-integration implementation bytes and API."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from scripts.research.taro_o1r_reducer_integration_runtime import locked_uncertainty
from scripts.research.taro_o1r_reducer_integration_runtime import reducer_integration as runtime
from scripts.research.taro_o1r_reducer_integration_runtime.validate_protocol_lock import validate as validate_protocol


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / "docs/research/taro/TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_IMPLEMENTATION_LOCK_2026-08-12.json"
SCHEMA = "blindassist.taro.o1r.r6_prospective_factor_reducer_integration_implementation_lock.v1"
LOCK_ID = "TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_IMPLEMENTATION_LOCK"
SUCCESSOR = "TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_ONE_SHOT_EXECUTION_LOCK"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"passed": False, "errors": [f"LOCK_READ_ERROR:{type(error).__name__}"]}
    if lock.get("schema") != SCHEMA or lock.get("lock_id") != LOCK_ID or lock.get("status") != "FROZEN":
        errors.append("IMPLEMENTATION_LOCK_IDENTITY_DRIFT")
    protocol = lock.get("protocol_binding", {})
    protocol_path = REPO_ROOT / str(protocol.get("path"))
    if not protocol_path.is_file() or protocol_path.stat().st_size != protocol.get("bytes") or _sha(protocol_path) != protocol.get("sha256"):
        errors.append("PROTOCOL_BINDING_DRIFT")
    elif not validate_protocol(protocol_path).get("passed"):
        errors.append("PROTOCOL_VALIDATION_FAILED")
    files = lock.get("implementation_files", {})
    for name, record in files.items() if isinstance(files, dict) else []:
        target = REPO_ROOT / str(record.get("path"))
        if not target.is_file():
            errors.append(f"{name.upper()}_MISSING")
        elif target.stat().st_size != record.get("bytes") or _sha(target) != record.get("sha256"):
            errors.append(f"{name.upper()}_BINDING_DRIFT")
    public = lock.get("public_api", {})
    parameters = inspect.signature(runtime.integrate_prospective_factor_bundle).parameters
    forbidden = ("faro", "truth", "outcome", "task_metric")
    if public.get("builder") != "integrate_prospective_factor_bundle" or any(token in name.lower() for name in parameters for token in forbidden):
        errors.append("PUBLIC_API_FIREWALL_DRIFT")
    if public.get("validator") != "validate_reducer_bundle" or public.get("locked_model_loader") != "load_locked_uncertainty_model":
        errors.append("PUBLIC_API_IDENTITY_DRIFT")
    invariants = lock.get("implemented_invariants", {})
    if invariants.get("query_clearance_owner") != "R1_BASELINE" or invariants.get("sole_final_state_producer") != runtime.REDUCER_VERSION:
        errors.append("FINAL_STATE_AUTHORITY_DRIFT")
    verification = lock.get("verification", {})
    if verification.get("focused_unittest_count") != 13 or verification.get("focused_unittest_passed") != 13 or verification.get("real_locked_uncertainty_model_sha256") != locked_uncertainty.MODEL_SHA256:
        errors.append("VERIFICATION_RECEIPT_DRIFT")
    if lock.get("execution_not_authorized_by_this_lock") is not True or lock.get("unique_successor") != SUCCESSOR:
        errors.append("AUTHORITY_OR_SUCCESSOR_DRIFT")
    return {
        "schema": "blindassist.taro.o1r.r6_prospective_factor_reducer_integration_implementation_lock_validation.v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "terminal": "TARO_O1R_R6_REDUCER_INTEGRATION_IMPLEMENTATION_VALID" if not errors else "TARO_O1R_R6_REDUCER_INTEGRATION_IMPLEMENTATION_INVALID",
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
