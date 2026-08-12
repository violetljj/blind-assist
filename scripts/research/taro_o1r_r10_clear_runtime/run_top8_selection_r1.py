#!/usr/bin/env python3
"""Recover R10 top-eight selection after the R0 canonical-float stop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r10_clear_runtime import run_top8_selection as base


LOCK_SCHEMA = "blindassist.taro.o1r.r10_fresh_pool_top8_selection_r1_execution_lock.v1"
LOCK_ID = "TARO_O1R_R10_FRESH_POOL_TOP8_SOURCE_ONLY_SELECTION_R1_ONE_SHOT_EXECUTION_LOCK"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r10-fresh-pool-top8-selection-r1"
PASS_TERMINAL = "TARO_O1R_R10_FRESH_POOL_TOP8_SOURCE_ONLY_SELECTION_R1_SEALED_PASS"
FAIL_TERMINAL = "TARO_O1R_R10_FRESH_POOL_TOP8_SELECTION_R1_EXECUTION_INVALID"
R0_ROOT = "artifacts.local/evidence/taro/o1r-r10-fresh-pool-top8-selection-r0"
R0_FAIL_TERMINAL = "TARO_O1R_R10_FRESH_POOL_TOP8_SELECTION_EXECUTION_INVALID"

EXPECTED_BINDINGS = {
    role: path
    for role, path in base.EXPECTED_BINDINGS.items()
    if role != "R10_SELECTION_RUNNER"
}
EXPECTED_BINDINGS.update(
    {
        "R10_SELECTION_BASE_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_top8_selection.py",
        "R10_SELECTION_BASE_TEST": "scripts/research/taro_o1r_r10_clear_runtime/test_run_top8_selection.py",
        "R10_SELECTION_R1_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_top8_selection_r1.py",
        "R10_SELECTION_R1_TEST": "scripts/research/taro_o1r_r10_clear_runtime/test_run_top8_selection_r1.py",
        "R10_SELECTION_R0_EXECUTION_LOCK": "docs/research/taro/TARO_O1R_R10_FRESH_POOL_TOP8_SOURCE_ONLY_SELECTION_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json",
        "R10_SELECTION_R0_EXECUTION_RECEIPT": f"{R0_ROOT}/execution-receipt.json",
        "R10_SELECTION_R0_FAILURE": f"{R0_ROOT}/failure.json",
        "R10_SELECTION_R0_MANIFEST": f"{R0_ROOT}/manifest.json",
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
        raise base.FreshTop8SelectionError(code, message)


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
    require(root.is_dir(), "R10_SELECTION_R1_R0_ROOT_MISSING", "R0 selection failure root is missing")
    observed = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(
        observed == {"execution-receipt.json", "failure.json", "manifest.json"},
        "R10_SELECTION_R1_R0_OUTPUT_DRIFT",
        "R0 selection produced files beyond its pre-selection failure receipt",
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    require(
        manifest.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_top8_selection_manifest.v1"
        and manifest.get("terminal") == R0_FAIL_TERMINAL
        and manifest.get("file_count_before_manifest") == 2
        and isinstance(files, dict)
        and set(files) == {"execution-receipt.json", "failure.json"},
        "R10_SELECTION_R1_R0_MANIFEST_DRIFT",
        "R0 selection failure manifest drift",
    )
    for relative, receipt in files.items():
        target = root / relative
        require(
            target.stat().st_size == receipt.get("bytes")
            and materializer.sha256_file(target) == receipt.get("sha256"),
            "R10_SELECTION_R1_R0_FILE_DRIFT",
            f"R0 selection artifact drift: {relative}",
        )
    execution = json.loads((root / "execution-receipt.json").read_text(encoding="utf-8"))
    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    require(
        execution.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_top8_selection_execution_receipt.v1"
        and execution.get("phase_a_root") == base.PHASE_A_ROOT
        and execution.get("expected_phase_a_file_count_before_manifest") == base.PHASE_A_FILE_COUNT
        and execution.get("frozen_selector") == base.FROZEN_LOCK_SELECTOR
        and execution.get("faro_reads") == execution.get("truth_reads") == execution.get("label_reads") == execution.get("outcome_reads") == 0
        and execution.get("training_steps") == execution.get("network_requests") == 0
        and execution.get("one_shot_consumed_on_root_creation") is True,
        "R10_SELECTION_R1_R0_EXECUTION_DRIFT",
        "R0 selection execution receipt drift",
    )
    require(
        failure.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_top8_selection_failure.v1"
        and failure.get("terminal") == R0_FAIL_TERMINAL
        and failure.get("execution_valid") is False
        and failure.get("failure_code") == "R10_SELECTION_PARENT_SCORE_INVALID"
        and failure.get("message") == "R10 parent score fields, selector binding, or firewall drift"
        and failure.get("faro_reads") == failure.get("truth_reads") == failure.get("label_reads") == failure.get("outcome_reads") == 0
        and failure.get("one_shot_consumed") is True,
        "R10_SELECTION_R1_R0_FAILURE_DRIFT",
        "R0 is not the exact pre-selection canonical-float stop",
    )


def _validate_configured(lock_path: Path) -> dict[str, Any]:
    lock = _BASE_VALIDATE(lock_path)
    _verify_r0_failure()
    require(
        lock.get("recovery_policy")
        == {
            "predecessor_root": R0_ROOT,
            "predecessor_parent_score_outputs": 0,
            "predecessor_selection_outputs": 0,
            "predecessor_faro_reads": 0,
            "resume": False,
            "fresh_full_recompute": True,
            "repair": "CANONICAL_FLOAT_ROUND12_VALIDATION_PARITY",
        },
        "R10_SELECTION_R1_RECOVERY_POLICY_DRIFT",
        "R1 selection recovery policy drift",
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
                "selected_parent_count": result["selected_parent_count"],
                "faro_reads": result["faro_reads"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
