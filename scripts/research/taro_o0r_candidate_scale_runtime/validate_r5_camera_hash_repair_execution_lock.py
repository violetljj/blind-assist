#!/usr/bin/env python3
"""Validate the TARO R5 normalized-camera-hash Phase-B replay lock."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "blindassist.taro.o0r.r5_camera_hash_repair_one_shot_execution_lock.v1"
LOCK_ID = "TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_CAMERA_HASH_REPAIR_ONE_SHOT_EXECUTION_LOCK"
MODULE = "scripts.research.taro_o0r_candidate_scale_runtime.run_direct_apple_hybrid_adapter_fit_confirmation_r3"
DEFAULT_LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_CAMERA_HASH_REPAIR_ONE_SHOT_EXECUTION_LOCK_2026-08-11.json"
AUTHORIZATION_SHA256 = "6CF2531AB1119B67AD2010040AE1AC73F817785684FCCC26111B3B70EF5FCBE5"
_SHA = re.compile(r"^[0-9A-F]{64}$")


class R5CameraHashRepairLockError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R5CameraHashRepairLockError(code, message, **context)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _binding(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}, "R5_R3_BINDING_FIELDS_DRIFT", "R3 binding fields drift")
    path = _path(value["path"])
    require(
        isinstance(value["bytes"], int) and value["bytes"] > 0
        and isinstance(value["sha256"], str) and bool(_SHA.fullmatch(value["sha256"]))
        and path.is_file() and path.stat().st_size == value["bytes"] and _sha256(path) == value["sha256"],
        "R5_R3_BINDING_HASH_DRIFT",
        "R3 bound file differs",
        path=str(path),
    )
    return dict(value)


def validate_execution_lock(path: Path = DEFAULT_LOCK_PATH, *, enforce_argv: bool = True) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R5CameraHashRepairLockError("R5_R3_LOCK_READ_FAILED", "R3 execution lock cannot be read") from error
    expected = {
        "schema", "lock_id", "date", "status", "repair_authority_binding", "implementation_lock_binding",
        "code_bindings", "predecessor_evidence_binding", "r2_diagnostic_evidence_bindings", "frame_plan_path",
        "roots", "resource_budget", "unique_argv", "argv_alternatives", "activation",
        "authority_carry_forward", "frozen_repair_semantics", "claim_ceiling",
    }
    require(isinstance(payload, dict) and set(payload) == expected, "R5_R3_LOCK_FIELD_DRIFT", "R3 lock fields drift")
    require(payload["schema"] == SCHEMA and payload["lock_id"] == LOCK_ID and payload["date"] == "2026-08-11", "R5_R3_LOCK_ID_DRIFT", "R3 lock identity drift")
    require(payload["status"] == "ONE_SHOT_EXECUTION_AUTHORIZED_NOT_YET_CONSUMED", "R5_R3_LOCK_STATUS_DRIFT", "R3 lock status drift")
    _binding(payload["repair_authority_binding"])
    _binding(payload["implementation_lock_binding"])
    roles = set()
    require(isinstance(payload["code_bindings"], list), "R5_R3_CODE_BINDING_SET_DRIFT", "R3 code bindings are missing")
    for row in payload["code_bindings"]:
        require(isinstance(row, dict) and set(row) == {"role", "path", "bytes", "sha256"}, "R5_R3_CODE_BINDING_FIELDS_DRIFT", "R3 code binding fields drift")
        roles.add(row["role"])
        _binding({key: row[key] for key in ("path", "bytes", "sha256")})
    require(roles == {"R5_CONFIRMATION", "R5_SHARED_PHASE_B_RUNNER", "R5_R3_ENTRYPOINT", "R5_R3_LOCK_VALIDATOR", "R5_CONFIRMATION_TEST"}, "R5_R3_CODE_BINDING_SET_DRIFT", "R3 code binding roles drift")
    predecessor = payload["predecessor_evidence_binding"]
    require(isinstance(predecessor, dict) and set(predecessor) == {"manifest_sha256", "failure_sha256", "candidate_phase_completion_sha256", "phase_a_completion_sha256"} and all(bool(_SHA.fullmatch(value)) for value in predecessor.values()), "R5_R3_PREDECESSOR_BINDING_DRIFT", "R3 R5 predecessor bindings drift")
    diagnostic = payload["r2_diagnostic_evidence_bindings"]
    require(isinstance(diagnostic, list) and {row.get("role") for row in diagnostic} == {"R2_MANIFEST", "R2_RESULT", "R2_SUMMARY"}, "R5_R3_DIAGNOSTIC_BINDING_SET_DRIFT", "R3 R2 diagnostic bindings drift")
    for row in diagnostic:
        _binding({key: row[key] for key in ("path", "bytes", "sha256")})
    result_row = next(row for row in diagnostic if row["role"] == "R2_RESULT")
    result = json.loads(_path(result_row["path"]).read_text(encoding="utf-8"))
    require(result.get("execution_valid") is True and result.get("terminal") == "TARO_O0R_DIRECT_APPLE_HYBRID_R5_NOT_EVALUABLE", "R5_R3_R2_TERMINAL_DRIFT", "R3 diagnostic predecessor terminal differs")
    roots = payload["roots"]
    require(isinstance(roots, dict) and set(roots) == {"source_root", "r3_evidence_root", "r5_predecessor_root", "r5_r2_diagnostic_root", "r5_r3_evidence_root"}, "R5_R3_ROOT_FIELD_DRIFT", "R3 roots drift")
    require(all(_path(roots[key]).is_dir() for key in ("source_root", "r3_evidence_root", "r5_predecessor_root", "r5_r2_diagnostic_root")) and not _path(roots["r5_r3_evidence_root"]).exists(), "R5_R3_ROOT_PREFLIGHT_INVALID", "R3 roots do not satisfy preflight")
    require(_path(payload["frame_plan_path"]).is_file(), "R5_R3_FRAME_PLAN_MISSING", "R3 frame plan is missing")
    budget = payload["resource_budget"]
    require(isinstance(budget, dict) and set(budget) == {"maximum_wall_seconds", "maximum_peak_rss_bytes", "maximum_evidence_bytes"} and 0 < budget["maximum_wall_seconds"] <= 28800 and 0 < budget["maximum_peak_rss_bytes"] <= 16 * 1024**3 and 0 < budget["maximum_evidence_bytes"] <= 2 * 1024**3, "R5_R3_RESOURCE_BUDGET_DRIFT", "R3 resource budget drift")
    relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    expected_argv = ["E:/codex-tools/tools/venvs/blindassist-venv-export312/Scripts/python.exe", "-m", MODULE, "--execution-lock", relative]
    require(payload["unique_argv"] == expected_argv and payload["argv_alternatives"] == [], "R5_R3_ARGV_DRIFT", "R3 unique argv drift")
    if enforce_argv:
        actual = [Path(sys.executable).resolve().as_posix(), "-m", MODULE, *sys.argv[1:]]
        require(actual == expected_argv, "R5_R3_ACTUAL_ARGV_DRIFT", "actual R3 argv differs", expected=expected_argv, actual=actual)
    require(payload["activation"] == {"root_must_be_absent": True, "one_shot_consumed_on_root_creation": True, "overwrite": False, "rerun": False}, "R5_R3_ACTIVATION_DRIFT", "R3 activation drift")
    require(payload["authority_carry_forward"] == {"authorization_sha256": AUTHORIZATION_SHA256, "scope_unchanged": True}, "R5_R3_AUTHORITY_DRIFT", "R3 authority drift")
    require(payload["frozen_repair_semantics"] == {"legacy_phase_a_k_hash": "CANONICAL_JSON_LIST", "extractor_k_hash": "CANONICAL_FLOAT64_ARRAY", "numeric_k_change": False, "phase_a_branch_change": False, "model_inference_new": 0, "threshold_or_gate_change": False}, "R5_R3_REPAIR_SEMANTICS_DRIFT", "R3 repair semantics drift")
    return payload


__all__ = ["DEFAULT_LOCK_PATH", "R5CameraHashRepairLockError", "validate_execution_lock"]
