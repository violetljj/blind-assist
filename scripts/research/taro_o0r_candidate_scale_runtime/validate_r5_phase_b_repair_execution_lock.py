#!/usr/bin/env python3
"""Validate the one-shot TARO R5 Phase-B support-unobservable repair lock."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "blindassist.taro.o0r.r5_phase_b_repair_one_shot_execution_lock.v1"
LOCK_ID = "TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_PHASE_B_REPAIR_ONE_SHOT_EXECUTION_LOCK"
MODULE = "scripts.research.taro_o0r_candidate_scale_runtime.run_direct_apple_hybrid_adapter_fit_confirmation_r2"
DEFAULT_LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_PHASE_B_REPAIR_ONE_SHOT_EXECUTION_LOCK_2026-08-11.json"
AUTHORIZATION_SHA256 = "6CF2531AB1119B67AD2010040AE1AC73F817785684FCCC26111B3B70EF5FCBE5"
_SHA = re.compile(r"^[0-9A-F]{64}$")


class R5PhaseBRepairLockError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R5PhaseBRepairLockError(code, message, **context)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _verify_binding(value: Any, *, external_root: Path | None = None) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}, "R5_R2_BINDING_FIELDS_DRIFT", "R2 file binding fields drift")
    require(isinstance(value["path"], str) and isinstance(value["bytes"], int) and value["bytes"] > 0 and isinstance(value["sha256"], str) and bool(_SHA.fullmatch(value["sha256"])), "R5_R2_BINDING_VALUE_DRIFT", "R2 file binding values drift")
    path = Path(value["path"])
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    elif external_root is not None:
        require(path.resolve().is_relative_to(external_root.resolve()), "R5_R2_BINDING_SCOPE_DRIFT", "R2 external binding escapes its frozen root")
    require(path.is_file() and path.stat().st_size == value["bytes"] and _sha256(path) == value["sha256"], "R5_R2_BINDING_HASH_DRIFT", "R2 bound file differs", path=str(path))
    return dict(value)


def validate_execution_lock(path: Path = DEFAULT_LOCK_PATH, *, enforce_argv: bool = True) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R5PhaseBRepairLockError("R5_R2_LOCK_READ_FAILED", "R2 execution lock cannot be read") from error
    expected = {
        "schema", "lock_id", "date", "status", "repair_authority_binding", "implementation_lock_binding",
        "code_bindings", "predecessor_evidence_binding", "frame_plan_path", "roots", "resource_budget",
        "unique_argv", "argv_alternatives", "activation", "authority_carry_forward", "frozen_repair_semantics",
        "claim_ceiling",
    }
    require(isinstance(payload, dict) and set(payload) == expected, "R5_R2_LOCK_FIELD_DRIFT", "R2 execution lock fields drift")
    require(payload["schema"] == SCHEMA and payload["lock_id"] == LOCK_ID and payload["date"] == "2026-08-11", "R5_R2_LOCK_ID_DRIFT", "R2 execution lock identity drift")
    require(payload["status"] == "ONE_SHOT_EXECUTION_AUTHORIZED_NOT_YET_CONSUMED", "R5_R2_LOCK_STATUS_DRIFT", "R2 execution lock status drift")
    _verify_binding(payload["repair_authority_binding"])
    _verify_binding(payload["implementation_lock_binding"])
    require(isinstance(payload["code_bindings"], list) and len(payload["code_bindings"]) >= 4, "R5_R2_CODE_BINDING_SET_DRIFT", "R2 code binding set is incomplete")
    roles = set()
    for row in payload["code_bindings"]:
        require(isinstance(row, dict) and set(row) == {"role", "path", "bytes", "sha256"}, "R5_R2_CODE_BINDING_FIELDS_DRIFT", "R2 code binding fields drift")
        roles.add(row["role"])
        _verify_binding({key: row[key] for key in ("path", "bytes", "sha256")})
    require(roles == {"R5_CONFIRMATION", "R5_R2_RUNNER", "R5_R2_LOCK_VALIDATOR", "R5_CONFIRMATION_TEST"}, "R5_R2_CODE_BINDING_SET_DRIFT", "R2 code binding roles drift")
    roots = payload["roots"]
    require(isinstance(roots, dict) and set(roots) == {"source_root", "r3_evidence_root", "r5_predecessor_root", "r5_r2_evidence_root"}, "R5_R2_ROOT_FIELD_DRIFT", "R2 root fields drift")
    source_root = Path(roots["source_root"]).resolve()
    r3_root = Path(roots["r3_evidence_root"]).resolve()
    predecessor_root = Path(roots["r5_predecessor_root"]).resolve()
    output_root = Path(roots["r5_r2_evidence_root"]).resolve()
    require(source_root.is_dir() and r3_root.is_dir() and predecessor_root.is_dir() and not output_root.exists(), "R5_R2_ROOT_PREFLIGHT_INVALID", "R2 roots do not satisfy one-shot preflight")
    predecessor = payload["predecessor_evidence_binding"]
    require(
        isinstance(predecessor, dict)
        and set(predecessor) == {"manifest_sha256", "failure_sha256", "candidate_phase_completion_sha256", "phase_a_completion_sha256"}
        and all(isinstance(value, str) and bool(_SHA.fullmatch(value)) for value in predecessor.values()),
        "R5_R2_PREDECESSOR_BINDING_FIELDS_DRIFT",
        "R2 predecessor evidence binding fields drift",
    )
    plan = Path(payload["frame_plan_path"]).resolve()
    require(plan.is_file(), "R5_R2_FRAME_PLAN_MISSING", "R2 frame plan is missing")
    budget = payload["resource_budget"]
    require(
        isinstance(budget, dict)
        and set(budget) == {"maximum_wall_seconds", "maximum_peak_rss_bytes", "maximum_evidence_bytes"}
        and 0 < budget["maximum_wall_seconds"] <= 28800
        and 0 < budget["maximum_peak_rss_bytes"] <= 16 * 1024**3
        and 0 < budget["maximum_evidence_bytes"] <= 2 * 1024**3,
        "R5_R2_RESOURCE_BUDGET_DRIFT",
        "R2 resource budget differs",
    )
    relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    expected_argv = ["E:/codex-tools/tools/venvs/blindassist-venv-export312/Scripts/python.exe", "-m", MODULE, "--execution-lock", relative]
    require(payload["unique_argv"] == expected_argv and payload["argv_alternatives"] == [], "R5_R2_ARGV_DRIFT", "R2 unique argv differs")
    if enforce_argv:
        actual = [Path(sys.executable).resolve().as_posix(), "-m", MODULE, *sys.argv[1:]]
        require(actual == expected_argv, "R5_R2_ACTUAL_ARGV_DRIFT", "actual R2 argv differs from the lock", expected=expected_argv, actual=actual)
    require(payload["activation"] == {"root_must_be_absent": True, "one_shot_consumed_on_root_creation": True, "overwrite": False, "rerun": False}, "R5_R2_ACTIVATION_DRIFT", "R2 activation semantics drift")
    require(payload["authority_carry_forward"] == {"authorization_sha256": AUTHORIZATION_SHA256, "scope_unchanged": True}, "R5_R2_AUTHORITY_DRIFT", "R2 user authority carry-forward drift")
    semantics = payload["frozen_repair_semantics"]
    require(
        semantics == {
            "predecessor_failure_code": "SUPPORT_PLAUSIBLE_INSUFFICIENT",
            "support_unobservable_codes": sorted([
                "SUPPORT_GATE_FAILED", "SUPPORT_HISTOGRAM_EMPTY", "SUPPORT_NORMAL_INVALID",
                "SUPPORT_PLAUSIBLE_INSUFFICIENT", "SUPPORT_POINTS_INSUFFICIENT",
                "SUPPORT_REFINED_GATE_FAILED", "SUPPORT_SLOPE_EXCEEDED",
            ]),
            "mapping": "NINE_RETAINED_UNKNOWN_QUERY_SLOTS",
            "model_inference_new": 0,
            "source_decision_recomputed": 0,
            "branch_reselection": False,
            "threshold_or_gate_change": False,
        },
        "R5_R2_REPAIR_SEMANTICS_DRIFT",
        "R2 repair semantics drift",
    )
    return payload


__all__ = ["DEFAULT_LOCK_PATH", "R5PhaseBRepairLockError", "validate_execution_lock"]
