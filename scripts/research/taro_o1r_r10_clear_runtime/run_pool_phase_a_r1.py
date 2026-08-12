#!/usr/bin/env python3
"""Recover R10 Phase A in a fresh root after the dependency-only R0 stop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r10_clear_runtime import run_pool_phase_a as base


LOCK_SCHEMA = "blindassist.taro.o1r.r10_fresh_pool_phase_a_r1_execution_lock.v1"
LOCK_ID = "TARO_O1R_R10_FRESH_POOL_SOURCE_ONLY_PHASE_A_R1_ONE_SHOT_EXECUTION_LOCK"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r10-fresh-pool-phase-a-r1"
PASS_TERMINAL = "TARO_O1R_R10_FRESH_POOL_PHASE_A_R1_SOURCE_ONLY_SEALED_PASS"
FAIL_TERMINAL = "TARO_O1R_R10_FRESH_POOL_PHASE_A_R1_EXECUTION_INVALID"
R0_ROOT = "artifacts.local/evidence/taro/o1r-r10-fresh-pool-phase-a-r0"
R0_FAIL_TERMINAL = "TARO_O1R_R10_FRESH_POOL_PHASE_A_EXECUTION_INVALID"

EXPECTED_BINDINGS = {
    role: path
    for role, path in base.EXPECTED_BINDINGS.items()
    if role not in {"R10_PHASE_A_RUNNER", "R10_PHASE_A_TEST"}
}
EXPECTED_BINDINGS.update(
    {
        "R10_PHASE_A_BASE_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_phase_a.py",
        "R10_PHASE_A_R1_RUNNER": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_phase_a_r1.py",
        "R10_PHASE_A_R1_TEST": "scripts/research/taro_o1r_r10_clear_runtime/test_run_pool_phase_a_r1.py",
        "R10_PHASE_A_R0_EXECUTION_LOCK": "docs/research/taro/TARO_O1R_R10_FRESH_POOL_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json",
        "R10_PHASE_A_R0_EXECUTION_RECEIPT": f"{R0_ROOT}/execution-receipt.json",
        "R10_PHASE_A_R0_FAILURE": f"{R0_ROOT}/failure.json",
        "R10_PHASE_A_R0_MANIFEST": f"{R0_ROOT}/manifest.json",
    }
)

_BASE_VALIDATE = base.validate_execution_lock
_BASE_CONFIG = {
    "LOCK_SCHEMA": base.LOCK_SCHEMA,
    "LOCK_ID": base.LOCK_ID,
    "OUTPUT_ROOT": base.OUTPUT_ROOT,
    "PASS_TERMINAL": base.PASS_TERMINAL,
    "FAIL_TERMINAL": base.FAIL_TERMINAL,
    "EXPECTED_BINDINGS": base.EXPECTED_BINDINGS,
    "validate_execution_lock": base.validate_execution_lock,
}


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise base.FreshPhaseAError(code, message)


def _configure() -> None:
    base.LOCK_SCHEMA = LOCK_SCHEMA
    base.LOCK_ID = LOCK_ID
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.PASS_TERMINAL = PASS_TERMINAL
    base.FAIL_TERMINAL = FAIL_TERMINAL
    base.EXPECTED_BINDINGS = EXPECTED_BINDINGS


def _restore() -> None:
    for name, value in _BASE_CONFIG.items():
        setattr(base, name, value)


def _verify_r0_failure() -> None:
    root = base._repo_path(R0_ROOT)
    require(root.is_dir(), "R10_PHASE_A_R1_R0_ROOT_MISSING", "R0 failure root is missing")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    require(
        observed == {"execution-receipt.json", "failure.json", "manifest.json"},
        "R10_PHASE_A_R1_R0_OUTPUT_DRIFT",
        "R0 produced files beyond the dependency-only failure receipt",
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    require(
        manifest.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_phase_a_manifest.v1"
        and manifest.get("terminal") == R0_FAIL_TERMINAL
        and manifest.get("file_count_before_manifest") == 2
        and isinstance(files, dict)
        and set(files) == {"execution-receipt.json", "failure.json"},
        "R10_PHASE_A_R1_R0_MANIFEST_DRIFT",
        "R0 failure manifest drift",
    )
    for relative, receipt in files.items():
        target = root / relative
        require(
            target.stat().st_size == receipt.get("bytes")
            and materializer.sha256_file(target) == receipt.get("sha256"),
            "R10_PHASE_A_R1_R0_FILE_DRIFT",
            f"R0 failure artifact drift: {relative}",
        )
    execution = json.loads((root / "execution-receipt.json").read_text(encoding="utf-8"))
    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    require(
        execution.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_phase_a_execution_receipt.v1"
        and execution.get("expected_parent_count") == base.PARENT_COUNT
        and execution.get("expected_frame_count") == base.FRAME_COUNT
        and execution.get("expected_query_count") == base.QUERY_COUNT
        and execution.get("faro_payload_read") is False
        and execution.get("training_steps") == 0
        and execution.get("network_requests") == 0,
        "R10_PHASE_A_R1_R0_EXECUTION_DRIFT",
        "R0 execution receipt drift",
    )
    require(
        failure.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_phase_a_failure.v1"
        and failure.get("terminal") == R0_FAIL_TERMINAL
        and failure.get("execution_valid") is False
        and failure.get("failure_code") == "ModuleNotFoundError"
        and failure.get("message") == "No module named 'timm'"
        and failure.get("one_shot_consumed") is True,
        "R10_PHASE_A_R1_R0_FAILURE_DRIFT",
        "R0 failure is not the exact pre-inference dependency stop",
    )


def _validate_configured(lock_path: Path) -> dict[str, Any]:
    lock = _BASE_VALIDATE(lock_path)
    _verify_r0_failure()
    require(
        lock.get("recovery_policy")
        == {
            "predecessor_root": R0_ROOT,
            "predecessor_candidate_outputs": 0,
            "predecessor_faro_reads": 0,
            "resume": False,
            "fresh_full_rerun": True,
        },
        "R10_PHASE_A_R1_RECOVERY_POLICY_DRIFT",
        "R1 recovery policy drift",
    )
    return lock


def validate_execution_lock(lock_path: Path) -> dict[str, Any]:
    _configure()
    try:
        return _validate_configured(lock_path)
    finally:
        _restore()


def execute(lock_path: Path) -> dict[str, Any]:
    _configure()
    base.validate_execution_lock = _validate_configured
    try:
        return base.execute(lock_path)
    finally:
        _restore()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        print(
            json.dumps(
                {
                    "terminal": FAIL_TERMINAL,
                    "failure_code": str(getattr(error, "code", type(error).__name__)),
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "passed": result["passed"],
                "execution_valid": result["execution_valid"],
                "parent_count": result["parent_count"],
                "frame_count": result["frame_count"],
                "query_count": result["query_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
