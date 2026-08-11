#!/usr/bin/env python3
"""Validate the TARO R5 pre-activation argv-repair one-shot lock."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from scripts.research.taro_o0r_candidate_scale_runtime import validate_r5_execution_lock as base


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "blindassist.taro.o0r.r5_confirmation_one_shot_execution_lock_r1.v1"
LOCK_ID = "TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_ONE_SHOT_EXECUTION_LOCK_R1"
MODULE = "scripts.research.taro_o0r_candidate_scale_runtime.run_direct_apple_hybrid_adapter_fit_confirmation_r1"
DEFAULT_LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_ONE_SHOT_EXECUTION_LOCK_R1_2026-08-11.json"
BASE_LOCK_RELATIVE = "docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_ONE_SHOT_EXECUTION_LOCK_2026-08-11.json"
REPAIR_RELATIVE = "docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_PRE_ACTIVATION_ARGV_REPAIR_2026-08-11.json"
AUTHORIZATION_SHA256 = "6CF2531AB1119B67AD2010040AE1AC73F817785684FCCC26111B3B70EF5FCBE5"
_SHA = re.compile(r"^[0-9A-F]{64}$")


class R5ExecutionLockR1Error(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R5ExecutionLockR1Error(code, message, **context)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _binding(value: Any, expected_path: str, code: str) -> tuple[Path, dict[str, Any]]:
    require(isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"} and value.get("path") == expected_path, code, "R5 R1 binding fields/path drift")
    require(isinstance(value.get("bytes"), int) and value["bytes"] > 0 and isinstance(value.get("sha256"), str) and bool(_SHA.fullmatch(value["sha256"])), code, "R5 R1 binding metadata drift")
    path = (REPO_ROOT / expected_path).resolve()
    require(path.is_file() and path.stat().st_size == value["bytes"] and _sha256(path) == value["sha256"], code, "R5 R1 bound file drift", path=expected_path)
    return path, dict(value)


def _validate_repair(path: Path) -> dict[str, Any]:
    try:
        repair = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R5ExecutionLockR1Error("R5_R1_REPAIR_READ_FAILED", "R5 pre-activation repair cannot be read") from error
    expected = {
        "schema", "repair_id", "date", "status", "base_implementation_lock_binding",
        "base_execution_lock_binding", "defect", "repair_files", "evidence_state",
        "authority_carry_forward", "claim_ceiling", "unique_successor",
    }
    require(isinstance(repair, dict) and set(repair) == expected, "R5_R1_REPAIR_FIELD_DRIFT", "R5 pre-activation repair fields drift")
    require(repair["schema"] == "blindassist.taro.o0r.r5.pre_activation_argv_repair.v1" and repair["repair_id"] == "TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_PRE_ACTIVATION_ARGV_REPAIR", "R5_R1_REPAIR_ID_DRIFT", "R5 pre-activation repair identity drift")
    require(repair["status"] == "PRE_ACTIVATION_ARGV_VALIDATOR_REPAIR_EXECUTION_NOT_CONSUMED", "R5_R1_REPAIR_STATUS_DRIFT", "R5 pre-activation repair status drift")
    defect = repair["defect"]
    require(defect.get("failure_code") == "R5_EXEC_ACTUAL_ARGV_DRIFT" and defect.get("python_process_started") is True and defect.get("model_loaded") is False and defect.get("evidence_root_created") is False, "R5_R1_REPAIR_DEFECT_DRIFT", "R5 argv defect evidence drift")
    evidence = repair["evidence_state"]
    require(evidence == {"r5_evidence_root_absent": True, "r5_inference_count": 0, "r5_source_decision_count": 0, "r5_task_metric_count": 0, "one_shot_consumed": False}, "R5_R1_REPAIR_EVIDENCE_DRIFT", "R5 pre-activation evidence state drift")
    authority = repair["authority_carry_forward"]
    require(authority == {"authorization_sha256": AUTHORIZATION_SHA256, "scope_unchanged": True, "model_or_data_identity_changed": False, "algorithm_or_gate_changed": False}, "R5_R1_REPAIR_AUTHORITY_DRIFT", "R5 user authority carry-forward drift")
    files = repair["repair_files"]
    require(isinstance(files, list) and len(files) == 2, "R5_R1_REPAIR_FILE_SET_DRIFT", "R5 repair file set drift")
    observed = set()
    for row in files:
        require(isinstance(row, dict) and set(row) == {"role", "path", "bytes", "sha256"}, "R5_R1_REPAIR_FILE_SET_DRIFT", "R5 repair file binding drift")
        file_path = (REPO_ROOT / row["path"]).resolve()
        require(file_path.is_file() and file_path.stat().st_size == row["bytes"] and _sha256(file_path) == row["sha256"], "R5_R1_REPAIR_FILE_HASH_DRIFT", "R5 repair file bytes drift", role=row["role"])
        observed.add(row["role"])
    require(observed == {"R1_EXECUTION_LOCK_VALIDATOR", "R1_EXECUTION_ENTRYPOINT"}, "R5_R1_REPAIR_FILE_SET_DRIFT", "R5 repair roles drift")
    require(repair["unique_successor"] == LOCK_ID, "R5_R1_REPAIR_SUCCESSOR_DRIFT", "R5 repair successor drift")
    return repair


def validate_execution_lock(path: Path = DEFAULT_LOCK_PATH, *, enforce_argv: bool = True) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R5ExecutionLockR1Error("R5_R1_LOCK_READ_FAILED", "R5 R1 execution lock cannot be read") from error
    expected = {"schema", "lock_id", "date", "status", "base_execution_lock_binding", "pre_activation_repair_binding", "unique_argv", "argv_alternatives", "activation", "authority_carry_forward", "claim_ceiling"}
    require(isinstance(payload, dict) and set(payload) == expected, "R5_R1_LOCK_FIELD_DRIFT", "R5 R1 execution lock fields drift")
    require(payload["schema"] == SCHEMA and payload["lock_id"] == LOCK_ID and payload["date"] == "2026-08-11", "R5_R1_LOCK_ID_DRIFT", "R5 R1 execution lock identity drift")
    require(payload["status"] == "ONE_SHOT_EXECUTION_AUTHORIZED_NOT_YET_CONSUMED", "R5_R1_LOCK_STATUS_DRIFT", "R5 R1 execution status drift")
    base_path, base_binding = _binding(payload["base_execution_lock_binding"], BASE_LOCK_RELATIVE, "R5_R1_BASE_LOCK_BINDING_DRIFT")
    repair_path, repair_binding = _binding(payload["pre_activation_repair_binding"], REPAIR_RELATIVE, "R5_R1_REPAIR_BINDING_DRIFT")
    _validate_repair(repair_path)
    normalized = base.validate_execution_lock(base_path, verify_files=True, enforce_argv=False)
    relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    expected_argv = ["E:/codex-tools/tools/venvs/blindassist-venv-export312/Scripts/python.exe", "-m", MODULE, "--execution-lock", relative]
    require(payload["unique_argv"] == expected_argv and payload["argv_alternatives"] == [], "R5_R1_ARGV_DRIFT", "R5 R1 unique argv drift")
    if enforce_argv:
        actual_argv = [Path(sys.executable).resolve().as_posix(), "-m", MODULE, *sys.argv[1:]]
        require(actual_argv == expected_argv, "R5_R1_ACTUAL_ARGV_DRIFT", "actual R5 R1 argv differs from repaired lock", expected=expected_argv, actual=actual_argv)
    require(payload["activation"] == {"root_must_be_absent": True, "one_shot_consumed_on_root_creation": True, "overwrite": False, "rerun": False}, "R5_R1_ACTIVATION_DRIFT", "R5 R1 activation semantics drift")
    require(payload["authority_carry_forward"] == {"authorization_sha256": AUTHORIZATION_SHA256, "scope_unchanged": True}, "R5_R1_AUTHORITY_DRIFT", "R5 R1 authority carry-forward drift")
    normalized["unique_argv"] = expected_argv
    normalized["_verified_bindings"] = {"BASE_EXECUTION_LOCK": base_binding, "PRE_ACTIVATION_REPAIR": repair_binding}
    return normalized


__all__ = ["DEFAULT_LOCK_PATH", "R5ExecutionLockR1Error", "validate_execution_lock"]
