#!/usr/bin/env python3
"""Validate and normalize a future one-shot TARO R5 execution lock."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import validate_r5_implementation_lock


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "blindassist.taro.o0r.r5_confirmation_one_shot_execution_lock.v1"
LOCK_ID = "TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_ONE_SHOT_EXECUTION_LOCK"
MODULE = "scripts.research.taro_o0r_candidate_scale_runtime.run_direct_apple_hybrid_adapter_fit_confirmation"
DEFAULT_LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_ONE_SHOT_EXECUTION_LOCK_2026-08-11.json"
IMPLEMENTATION_LOCK_RELATIVE = "docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_IMPLEMENTATION_LOCK_2026-08-11.json"
SOURCE_ROOT_RELATIVE = "artifacts.local/datasets/taro/o0r-arkitscenes-source-adapter-r3"
R3_ROOT_RELATIVE = "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3"
R5_ROOT_RELATIVE = "artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-adapter-fit-r5"
FRAME_PLAN_RELATIVE = f"{R3_ROOT_RELATIVE}/exact-frame-plan.json.gz"
DEPTHART_SOURCE = "F:/ba-data/blindassist-artifacts-20260805/models/depthart/source"
CHECKPOINT = f"{DEPTHART_SOURCE}/checkpoints/metric/depthart_metric_indoor_s_448.pth"
_SHA = re.compile(r"^[0-9A-F]{64}$")


class R5ExecutionLockError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R5ExecutionLockError(code, message, **context)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _repo_path(relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative) and "\\" not in relative, "R5_EXEC_PATH_INVALID", "execution lock path is not canonical", path=relative)
    lexical = Path(os.path.abspath(REPO_ROOT / relative))
    require(lexical == REPO_ROOT or REPO_ROOT in lexical.parents, "R5_EXEC_PATH_INVALID", "execution lock path escapes repository", path=relative)
    return lexical.resolve()


def validate_payload(payload: Mapping[str, Any], *, lock_path: Path, verify_files: bool = True, enforce_argv: bool = True) -> dict[str, Any]:
    expected_keys = {
        "schema", "lock_id", "date", "research_mode", "status", "implementation_lock_binding",
        "roots", "candidate_identity", "exact_cohort", "phase_firewall", "unique_argv",
        "argv_alternatives", "resource_budget", "side_effects", "activation", "user_authority",
        "execution_authority", "claim_ceiling",
    }
    require(set(payload) == expected_keys, "R5_EXEC_TOP_LEVEL_KEY_SET_DRIFT", "R5 execution lock fields drift")
    require(payload.get("schema") == SCHEMA and payload.get("lock_id") == LOCK_ID, "R5_EXEC_SCHEMA_DRIFT", "R5 execution lock identity drift")
    require(payload.get("date") == "2026-08-11" and payload.get("research_mode") == "WILD_LAB", "R5_EXEC_CONTEXT_DRIFT", "R5 execution context drift")
    require(payload.get("status") == "ONE_SHOT_EXECUTION_AUTHORIZED_NOT_YET_CONSUMED", "R5_EXEC_STATUS_DRIFT", "R5 execution lock is not activated authority")

    binding = payload.get("implementation_lock_binding", {})
    require(isinstance(binding, dict) and set(binding) == {"path", "bytes", "sha256"} and binding.get("path") == IMPLEMENTATION_LOCK_RELATIVE, "R5_EXEC_IMPLEMENTATION_BINDING_DRIFT", "R5 implementation binding fields/path drift")
    require(isinstance(binding.get("bytes"), int) and binding["bytes"] > 0 and isinstance(binding.get("sha256"), str) and bool(_SHA.fullmatch(binding["sha256"])), "R5_EXEC_IMPLEMENTATION_BINDING_DRIFT", "R5 implementation binding metadata drift")

    roots = payload.get("roots", {})
    require(roots == {"source_root": SOURCE_ROOT_RELATIVE, "r3_evidence_root": R3_ROOT_RELATIVE, "r5_evidence_root": R5_ROOT_RELATIVE}, "R5_EXEC_ROOT_DRIFT", "R5 execution roots drift")
    candidate = payload.get("candidate_identity", {})
    require(candidate.get("model_id") == "depthart-s-metric-indoor-448-official-fp32", "R5_EXEC_MODEL_DRIFT", "R5 model identity drift")
    require(candidate.get("source_root") == DEPTHART_SOURCE and candidate.get("source_commit") == "0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c", "R5_EXEC_MODEL_DRIFT", "R5 model source drift")
    require(candidate.get("checkpoint_path") == CHECKPOINT and candidate.get("checkpoint_bytes") == 32871942 and candidate.get("checkpoint_sha256") == "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65", "R5_EXEC_MODEL_DRIFT", "R5 checkpoint drift")
    require(candidate.get("preprocess_id") == "DEPTHART_OFFICIAL_LOWER_BOUND_448_RGB_CUBIC_IMAGENET_V1" and candidate.get("postprocess_id") == "TARO_TORCH_CPU_BILINEAR_ALIGN_CORNERS_TRUE_FLOAT32_448X608_TO_1440X1920_V1", "R5_EXEC_TRANSFORM_DRIFT", "R5 transform identity drift")

    cohort = payload.get("exact_cohort", {})
    require(cohort == {"parent_count": 8, "physical_frame_count": r5.EXPECTED_FRAME_COUNT, "query_slot_count": r5.EXPECTED_QUERY_COUNT, "canonical_frame_identity_sequence_sha256": r5.EXPECTED_IDENTITY_SEQUENCE_SHA256}, "R5_EXEC_COHORT_DRIFT", "R5 execution cohort drift")
    firewall = payload.get("phase_firewall", {})
    require(firewall.get("all_candidates_before_decisions") is True and firewall.get("all_decisions_before_faro") is True and firewall.get("phase_a_completion_reload_before_faro") is True, "R5_EXEC_PHASE_FIREWALL_DRIFT", "R5 execution phase order drift")
    require(firewall.get("prior_eval_truth_or_outcome_roots_allowed") is False and firewall.get("branch_reselection_after_truth_allowed") is False, "R5_EXEC_PHASE_FIREWALL_DRIFT", "R5 execution truth firewall drift")

    argv = payload.get("unique_argv")
    relative_lock = lock_path.resolve().relative_to(REPO_ROOT).as_posix()
    expected_argv = [
        "E:/codex-tools/tools/venvs/blindassist-venv-export312/Scripts/python.exe",
        "-m", MODULE, "--execution-lock", relative_lock,
    ]
    require(argv == expected_argv and payload.get("argv_alternatives") == [], "R5_EXEC_ARGV_DRIFT", "R5 unique argv drift")
    if enforce_argv:
        actual = [Path(sys.executable).resolve().as_posix(), *sys.argv[1:]]
        require(actual == expected_argv, "R5_EXEC_ACTUAL_ARGV_DRIFT", "actual R5 argv differs from one-shot lock", expected=expected_argv, actual=actual)

    budget = payload.get("resource_budget", {})
    require(budget == {"maximum_wall_seconds": 28800, "maximum_peak_rss_bytes": 17179869184, "maximum_cuda_allocated_bytes": 8500000000, "maximum_evidence_bytes": 2147483648}, "R5_EXEC_RESOURCE_BUDGET_DRIFT", "R5 execution budget drift")
    require(payload.get("side_effects") == {"network_requests": 0, "training_steps": 0, "device_actions": 0}, "R5_EXEC_SIDE_EFFECT_DRIFT", "R5 execution side-effect budget drift")
    activation = payload.get("activation", {})
    require(activation == {"root_must_be_absent": True, "one_shot_consumed_on_root_creation": True, "overwrite": False, "rerun": False}, "R5_EXEC_ACTIVATION_DRIFT", "R5 one-shot activation semantics drift")

    user = payload.get("user_authority", {})
    require(user.get("explicit_model_execution_authority") is True and user.get("scope") == "EXACT_R5_211_DEPTHART_INFERENCES_PHASE_A_SOURCE_DECISIONS_AND_SAME_FRAME_PHASE_B_FARO_SCORING", "R5_EXEC_USER_AUTHORITY_MISSING", "explicit user model/task authority is absent")
    require(isinstance(user.get("authorization_sha256"), str) and bool(_SHA.fullmatch(user["authorization_sha256"])) and isinstance(user.get("recorded_at_utc"), str) and bool(user["recorded_at_utc"]), "R5_EXEC_USER_AUTHORITY_INVALID", "user authority receipt is malformed")
    authority = payload.get("execution_authority", {})
    expected_authority = {
        "one_shot_execution_lock": True, "user_model_execution_authority": True,
        "depthart_inference": True, "phase_a_source_decisions": True,
        "phase_b_truth_scoring": True, "training": False, "network": False,
        "device": False, "product": False, "safety": False,
    }
    require(authority == expected_authority, "R5_EXEC_AUTHORITY_DRIFT", "R5 execution authority drift")

    implementation_path = _repo_path(binding["path"])
    source_root = _repo_path(roots["source_root"])
    r3_root = _repo_path(roots["r3_evidence_root"])
    r5_root = _repo_path(roots["r5_evidence_root"])
    if verify_files:
        require(implementation_path.is_file() and implementation_path.stat().st_size == binding["bytes"] and _sha256(implementation_path) == binding["sha256"], "R5_EXEC_IMPLEMENTATION_BINDING_DRIFT", "bound R5 implementation lock drift")
        implementation_errors = validate_r5_implementation_lock.validate_file(implementation_path, verify_files=True)
        require(not implementation_errors, "R5_EXEC_IMPLEMENTATION_LOCK_INVALID", "bound R5 implementation lock is invalid", errors=implementation_errors)
        require(source_root.is_dir() and r3_root.is_dir() and not r5_root.exists(), "R5_EXEC_ROOT_PREFLIGHT_INVALID", "R5 execution roots are not in pre-activation state")
        source_model = Path(DEPTHART_SOURCE)
        checkpoint = Path(CHECKPOINT)
        require(source_model.is_dir() and checkpoint.is_file() and checkpoint.stat().st_size == candidate["checkpoint_bytes"] and _sha256(checkpoint) == candidate["checkpoint_sha256"], "R5_EXEC_MODEL_ASSET_DRIFT", "R5 model asset drift")
        frame_plan = _repo_path(FRAME_PLAN_RELATIVE)
        require(frame_plan.is_file(), "R5_EXEC_FRAME_PLAN_MISSING", "R5 exact frame plan missing")
    output = json.loads(json.dumps(payload))
    output["roots"] = {"source_root": str(source_root), "r3_evidence_root": str(r3_root), "r5_evidence_root": str(r5_root)}
    output["frame_plan_path"] = str(_repo_path(FRAME_PLAN_RELATIVE))
    output["depthart_source_root"] = DEPTHART_SOURCE
    output["checkpoint_path"] = CHECKPOINT
    output["_verified_bindings"] = {"R5_IMPLEMENTATION_LOCK": dict(binding)}
    return output


def validate_execution_lock(path: Path = DEFAULT_LOCK_PATH, *, verify_files: bool = True, enforce_argv: bool = True) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R5ExecutionLockError("R5_EXEC_LOCK_READ_FAILED", "R5 execution lock cannot be read", path=str(path)) from error
    require(isinstance(payload, Mapping), "R5_EXEC_LOCK_NOT_OBJECT", "R5 execution lock must be an object")
    return validate_payload(payload, lock_path=path, verify_files=verify_files, enforce_argv=enforce_argv)


__all__ = ["DEFAULT_LOCK_PATH", "R5ExecutionLockError", "validate_execution_lock", "validate_payload"]
