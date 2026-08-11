#!/usr/bin/env python3
"""Validate the one-shot TARO O1R source-only eval replay lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.taro_o1r_reducer_integration_runtime.validate_implementation_lock import validate as validate_implementation


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / "docs/research/taro/TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json"
SCHEMA = "blindassist.taro.o1r.r6_prospective_factor_reducer_integration_one_shot_execution_lock.v1"
LOCK_ID = "TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_ONE_SHOT_EXECUTION_LOCK"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate(path: Path, *, require_output_absent: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"passed": False, "errors": [f"LOCK_READ_ERROR:{type(error).__name__}"], "lock": {}}
    if lock.get("schema") != SCHEMA or lock.get("lock_id") != LOCK_ID or lock.get("status") != "FROZEN":
        errors.append("EXECUTION_LOCK_IDENTITY_DRIFT")
    implementation = lock.get("implementation_lock", {})
    implementation_path = REPO_ROOT / str(implementation.get("path"))
    if not implementation_path.is_file() or implementation_path.stat().st_size != implementation.get("bytes") or _sha(implementation_path) != implementation.get("sha256"):
        errors.append("IMPLEMENTATION_LOCK_BINDING_DRIFT")
    elif not validate_implementation(implementation_path).get("passed"):
        errors.append("IMPLEMENTATION_LOCK_INVALID")
    runner = lock.get("runner", {})
    runner_path = REPO_ROOT / str(runner.get("path"))
    if not runner_path.is_file() or runner_path.stat().st_size != runner.get("bytes") or _sha(runner_path) != runner.get("sha256"):
        errors.append("RUNNER_BINDING_DRIFT")
    for name, record in (lock.get("input_bindings") or {}).items():
        target = REPO_ROOT / str(record.get("path"))
        if not target.is_file():
            errors.append(f"INPUT_{name}_MISSING")
        elif target.stat().st_size != record.get("bytes") or _sha(target) != record.get("sha256"):
            errors.append(f"INPUT_{name}_BINDING_DRIFT")
    roots = lock.get("roots", {})
    if roots.get("repo_root") != REPO_ROOT.as_posix():
        errors.append("REPO_ROOT_DRIFT")
    output_root = Path(str(roots.get("output_root", "")))
    if require_output_absent and output_root.exists():
        errors.append("OUTPUT_ROOT_ALREADY_EXISTS_ONE_SHOT_CONSUMED")
    authority = lock.get("authority", {})
    if authority.get("eval_parent_count") != 16 or authority.get("eval_frame_count") != 239 or authority.get("query_count") != 2151:
        errors.append("EXECUTION_COHORT_DRIFT")
    if authority.get("allowed_source_payload_roles") != ["confidence"] or authority.get("faro_reads") != 0 or authority.get("training_steps") != 0 or authority.get("network_requests") != 0:
        errors.append("EXECUTION_FIREWALL_DRIFT")
    if lock.get("one_shot", {}).get("overwrite_allowed") is not False or lock.get("one_shot", {}).get("rerun_allowed") is not False:
        errors.append("ONE_SHOT_POLICY_DRIFT")
    result = {
        "schema": "blindassist.taro.o1r.r6_reducer_integration_execution_lock_validation.v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "terminal": "TARO_O1R_R6_REDUCER_INTEGRATION_EXECUTION_LOCK_VALID" if not errors else "TARO_O1R_R6_REDUCER_INTEGRATION_EXECUTION_LOCK_INVALID",
        "lock": lock,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--allow-consumed", action="store_true")
    args = parser.parse_args()
    result = validate(args.lock.resolve(), require_output_absent=not args.allow_consumed)
    printable = {key: value for key, value in result.items() if key != "lock"}
    print(json.dumps(printable, sort_keys=True, separators=(",", ":")))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
