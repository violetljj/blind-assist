#!/usr/bin/env python3
"""Validate the unique TARO R6 untouched confirmation one-shot lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation as r6
from scripts.research.taro_o0r_candidate_scale_runtime import validate_r6_confirmation_implementation_lock as implementation_validator
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_R6_UNTOUCHED_CONFIRMATION_ONE_SHOT_EXECUTION_LOCK_2026-08-11.json"
SCHEMA = "blindassist.taro.o0r.r6_untouched_confirmation_one_shot_execution_lock.v1"
LOCK_ID = "TARO_O0R_R6_UNTOUCHED_CONFIRMATION_ONE_SHOT_EXECUTION_LOCK"
IMPLEMENTATION_LOCK = ("docs/research/taro/TARO_O0R_R6_UNTOUCHED_CONFIRMATION_EXECUTOR_IMPLEMENTATION_LOCK_2026-08-11.json", 5959, "651769E9D360CA84C81704D35BC57C78B948BC5ADC45500222457ED801B1A0BF")
INVENTORY = ("artifacts.local/evidence/taro/o0r-r6-untouched-inventory-r0/exact-frame-plan.json", 12340, "69352D6A940111E738488AA25CFAB8A924658B8C5720D9CE7A50AC612558D6A8")
PYTHON_PATH = "E:/codex-tools/tools/venvs/blindassist-venv-export312/Scripts/python.exe"
EVIDENCE_ROOT = "artifacts.local/evidence/taro/o0r-r6-untouched-confirmation-r0"
SOURCE_ROOT = "F:/ba-data/blindassist-artifacts-20260805/models/depthart/source"
CHECKPOINT_PATH = f"{SOURCE_ROOT}/checkpoints/metric/depthart_metric_indoor_s_448.pth"
CHECKPOINT_BYTES = 32871942
CHECKPOINT_SHA256 = "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65"
SOURCE_COMMIT = "0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c"


class R6ExecutionLockError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R6ExecutionLockError(code, message, **context)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _verify_binding(row: Any, expected: tuple[str, int, str], code: str) -> None:
    require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"} and (row["path"], row["bytes"], row["sha256"]) == expected, code, "R6 execution binding fields drift")
    path = materializer.safe_join(REPO_ROOT, expected[0])
    require(path.is_file() and path.stat().st_size == expected[1] and _sha(path) == expected[2], code, "R6 execution bound file drift", path=expected[0])


def validate_execution_lock(path: Path = DEFAULT_LOCK_PATH) -> dict[str, Any]:
    lock_path = path.resolve()
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R6ExecutionLockError("R6_EXEC_LOCK_READ_FAILED", "R6 execution lock cannot be read") from error
    expected_keys = {"schema", "lock_id", "date", "research_mode", "status", "implementation_lock_binding", "inventory_binding", "execution_validator_binding", "roots", "candidate_identity", "exact_cohort", "phase_firewall", "unique_argv", "argv_alternatives", "resource_budget", "activation", "user_authority", "execution_authority", "claim_ceiling"}
    require(isinstance(payload, dict) and set(payload) == expected_keys and payload["schema"] == SCHEMA and payload["lock_id"] == LOCK_ID and payload["status"] == "ONE_SHOT_EXECUTION_AUTHORIZED_NOT_YET_CONSUMED", "R6_EXEC_LOCK_IDENTITY_DRIFT", "R6 execution lock identity/status drift")
    _verify_binding(payload["implementation_lock_binding"], IMPLEMENTATION_LOCK, "R6_EXEC_IMPLEMENTATION_BINDING_DRIFT")
    _verify_binding(payload["inventory_binding"], INVENTORY, "R6_EXEC_INVENTORY_BINDING_DRIFT")
    require(implementation_validator.validate_file() == [], "R6_EXEC_IMPLEMENTATION_LOCK_INVALID", "R6 implementation lock no longer validates")

    validator = payload["execution_validator_binding"]
    relative_validator = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    require(isinstance(validator, dict) and set(validator) == {"path", "bytes", "sha256"} and validator["path"] == relative_validator and validator["bytes"] == Path(__file__).stat().st_size and validator["sha256"] == _sha(Path(__file__)), "R6_EXEC_VALIDATOR_BINDING_DRIFT", "R6 execution validator binding drift")

    roots = payload["roots"]
    require(roots == {"repo_root": REPO_ROOT.as_posix(), "inventory_path": (REPO_ROOT / INVENTORY[0]).as_posix(), "evidence_root": (REPO_ROOT / EVIDENCE_ROOT).as_posix()}, "R6_EXEC_ROOT_DRIFT", "R6 execution roots drift")
    require(not Path(roots["evidence_root"]).exists(), "R6_EXEC_ROOT_ALREADY_CONSUMED", "R6 one-shot evidence root already exists")

    identity = payload["candidate_identity"]
    require(identity == {"model_id": "depthart-s-metric-indoor-448-official-fp32", "source_root": SOURCE_ROOT, "source_commit": SOURCE_COMMIT, "checkpoint_path": CHECKPOINT_PATH, "checkpoint_bytes": CHECKPOINT_BYTES, "checkpoint_sha256": CHECKPOINT_SHA256, "preprocess_id": "DEPTHART_OFFICIAL_LOWER_BOUND_448_RGB_CUBIC_IMAGENET_V1", "postprocess_id": "TARO_TORCH_CPU_BILINEAR_ALIGN_CORNERS_TRUE_FLOAT32_448X608_TO_1440X1920_V1"}, "R6_EXEC_CANDIDATE_IDENTITY_DRIFT", "R6 candidate identity drift")
    checkpoint = Path(CHECKPOINT_PATH)
    require(checkpoint.is_file() and checkpoint.stat().st_size == CHECKPOINT_BYTES and _sha(checkpoint) == CHECKPOINT_SHA256, "R6_EXEC_CHECKPOINT_DRIFT", "R6 checkpoint drift")
    source_root = Path(SOURCE_ROOT)
    require(source_root.is_dir(), "R6_EXEC_SOURCE_ROOT_MISSING", "R6 DepthART source root is missing")
    try:
        observed_commit = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise R6ExecutionLockError("R6_EXEC_SOURCE_COMMIT_UNVERIFIED", "R6 DepthART source commit cannot be verified") from error
    require(observed_commit == SOURCE_COMMIT, "R6_EXEC_SOURCE_COMMIT_DRIFT", "R6 DepthART source commit drift")

    cohort = payload["exact_cohort"]
    require(cohort == {"parent_count": 8, "physical_frame_count": 120, "query_slot_count": 1080, "parent_frame_counts": r6.expected_parent_frame_counts(), "formation_parent_overlap_count": 0}, "R6_EXEC_COHORT_DRIFT", "R6 execution cohort drift")
    firewall = payload["phase_firewall"]
    require(firewall == {"all_candidates_before_decisions": True, "all_decisions_before_faro": True, "phase_a_completion_reload_before_faro": True, "phase_a_payload_allowlist": list(r6.PHASE_A_ASSET_ROLES), "phase_b_payload_allowlist": ["highres_depth"], "prior_outcome_roots_allowed": False, "branch_reselection_after_truth_allowed": False}, "R6_EXEC_PHASE_FIREWALL_DRIFT", "R6 execution firewall drift")

    expected_argv = [PYTHON_PATH, "-m", "scripts.research.taro_o0r_candidate_scale_runtime.run_r6_untouched_confirmation", "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(payload["unique_argv"] == expected_argv and payload["argv_alternatives"] == [], "R6_EXEC_ARGV_DRIFT", "R6 unique argv drift")
    budget = payload["resource_budget"]
    require(budget == {"maximum_wall_seconds": 28800, "maximum_peak_rss_bytes": 17179869184, "maximum_cuda_allocated_bytes": 8500000000, "maximum_evidence_bytes": 2147483648}, "R6_EXEC_RESOURCE_BUDGET_DRIFT", "R6 execution budget drift")
    require(payload["activation"] == {"root_must_be_absent": True, "one_shot_consumed_on_root_creation": True, "overwrite": False, "rerun": False}, "R6_EXEC_ACTIVATION_DRIFT", "R6 one-shot activation drift")
    require(payload["user_authority"].get("explicit_model_and_truth_execution_authority") is True and payload["user_authority"].get("authorization_sha256") == "6CF2531AB1119B67AD2010040AE1AC73F817785684FCCC26111B3B70EF5FCBE5", "R6_EXEC_USER_AUTHORITY_DRIFT", "R6 user execution authority drift")
    require(payload["execution_authority"] == {"one_shot_execution_lock": True, "depthart_inference": True, "phase_a_source_decisions": True, "phase_b_truth_scoring": True, "training": False, "network": False, "device": False, "product": False, "safety": False}, "R6_EXEC_AUTHORITY_DRIFT", "R6 execution authority drift")
    payload["_lock_path"] = lock_path
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    args = parser.parse_args(argv)
    try:
        validate_execution_lock(args.lock)
    except Exception as error:
        print(json.dumps({"passed": False, "error_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"passed": True, "terminal": "TARO_O0R_R6_CONFIRMATION_EXECUTION_LOCK_VALID"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
