#!/usr/bin/env python3
"""Validate the one-shot formation Phase-B repair execution lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "blindassist.taro.o0r.r6_formation_phase_b_repair_execution_lock.v1"
LOCK_ID = "TARO_O0R_R6_FORMATION_PHASE_B_REPAIR_R6_ONE_SHOT_EXECUTION_LOCK"
EXPECTED_ROOTS = {
    "repo_root": "E:/linnan/linnan",
    "frame_plan_path": "E:/linnan/linnan/artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/exact-frame-plan.json.gz",
    "source_root": "E:/linnan/linnan/artifacts.local/datasets/taro/o0r-arkitscenes-source-adapter-r3",
    "source_evidence_root": "E:/linnan/linnan/artifacts.local/evidence/taro/o0r-arkitscenes-formation-source-r0",
    "fit_candidate_root": "E:/linnan/linnan/artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-adapter-fit-r5",
    "eval_candidate_root": "E:/linnan/linnan/artifacts.local/evidence/taro/o0r-arkitscenes-factor-headroom-r3",
    "phase_a_root": "E:/linnan/linnan/artifacts.local/evidence/taro/o0r-r6-formation-replay-r4",
    "output_root": "E:/linnan/linnan/artifacts.local/evidence/taro/o0r-r6-formation-replay-r6",
}
EXPECTED_COHORT = {
    "frame_key_sequence_sha256": "CFAEA2236C5E1EA9A2DC811ECE432F955B680658E8D5D7B94468FEAEECE77653",
    "source_receipt_hash_sequence_sha256": "E2E6478B3819FF42B158516D8D5E7D862C0069C8172B874B535ED8D4C5DC1A43",
    "candidate_record_hash_sequence_sha256": "DA22AA6BC8B015B2587A26EB05FA842E6A513A28841838078B89301A3B303E01",
    "candidate_native_hash_sequence_sha256": "C6A41722142BCCCBC0CAE33BD1BF8FCA2BBA16FC5349C83D57273EABA3E71089",
}


class PhaseBRepairLockError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise PhaseBRepairLockError(code, message, **context)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_execution_lock(path: Path, *, enforce_argv: bool = True) -> dict[str, Any]:
    lock_path = path.resolve()
    repo_root = Path(EXPECTED_ROOTS["repo_root"]).resolve()
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhaseBRepairLockError("PHASE_B_REPAIR_LOCK_READ_FAILED", "Phase-B repair execution lock cannot be read") from error
    require(lock.get("schema") == SCHEMA and lock.get("lock_id") == LOCK_ID, "PHASE_B_REPAIR_LOCK_IDENTITY_DRIFT", "Phase-B repair lock identity drift")
    require(lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "PHASE_B_REPAIR_ALREADY_CONSUMED", "Phase-B repair lock is not authorized/unconsumed")
    require(lock.get("roots") == EXPECTED_ROOTS and lock.get("cohort_bindings") == EXPECTED_COHORT, "PHASE_B_REPAIR_ROOT_OR_COHORT_DRIFT", "Phase-B repair roots/cohort drift")
    require(lock.get("phase_a_completion_sha256") == "C3351E6745110850E0CA80267A8A0421E9548CFFF14F3EAED45F2B1F11311F77", "PHASE_B_REPAIR_COMPLETION_DRIFT", "Phase-B repair predecessor completion drift")
    require(lock.get("worker_count") == 4 and lock.get("output_root_must_be_absent") is True and lock.get("overwrite") is False and lock.get("rerun") is False, "PHASE_B_REPAIR_ONE_SHOT_DRIFT", "Phase-B repair one-shot policy drift")
    require(not Path(EXPECTED_ROOTS["output_root"]).exists(), "PHASE_B_REPAIR_OUTPUT_COLLISION", "Phase-B repair output root already exists")
    for name, raw in EXPECTED_ROOTS.items():
        if name == "output_root":
            continue
        target = Path(raw)
        require(target.is_dir() if name.endswith("root") else target.is_file(), "PHASE_B_REPAIR_INPUT_MISSING", "Phase-B repair input is missing", name=name)
    environment = lock.get("required_environment")
    require(isinstance(environment, dict) and bool(environment), "PHASE_B_REPAIR_ENVIRONMENT_MISSING", "Phase-B repair environment is missing")
    for key, expected in environment.items():
        require(os.environ.get(key) == str(expected), "PHASE_B_REPAIR_ENVIRONMENT_DRIFT", "Phase-B repair environment drift", key=key)
    budget = lock.get("resource_budget")
    require(isinstance(budget, dict) and budget.get("maximum_wall_seconds") == 14400 and budget.get("maximum_peak_rss_bytes") == 17179869184 and budget.get("maximum_evidence_bytes") == 536870912 and budget.get("training_steps") == budget.get("network_requests") == 0, "PHASE_B_REPAIR_BUDGET_DRIFT", "Phase-B repair budget drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and bool(bindings), "PHASE_B_REPAIR_BINDINGS_MISSING", "Phase-B repair bindings are missing")
    roles: set[str] = set()
    for binding in bindings:
        require(isinstance(binding, dict) and set(binding) == {"role", "path", "bytes", "sha256"}, "PHASE_B_REPAIR_BINDING_FIELDS", "Phase-B repair binding fields drift")
        role = str(binding["role"])
        relative = Path(str(binding["path"]))
        require(role not in roles and not relative.is_absolute() and ".." not in relative.parts, "PHASE_B_REPAIR_BINDING_INVALID", "Phase-B repair binding role/path invalid", role=role)
        roles.add(role)
        target = repo_root / relative
        require(target.is_file() and target.stat().st_size == binding["bytes"] and _sha(target) == binding["sha256"], "PHASE_B_REPAIR_BINDING_DRIFT", "Phase-B repair binding drift", role=role)
    require(roles == set(lock.get("required_binding_roles", [])), "PHASE_B_REPAIR_BINDING_ROLE_SET", "Phase-B repair binding role set drift")
    if enforce_argv:
        unique = lock.get("unique_argv")
        require(isinstance(unique, list) and len(unique) == 5 and Path(sys.executable).resolve() == Path(unique[0]).resolve(), "PHASE_B_REPAIR_ARGV_DRIFT", "Phase-B repair executable/argv drift")
        require(Path(sys.argv[0]).resolve() == (repo_root / unique[2]).resolve() and sys.argv[1:] == unique[3:], "PHASE_B_REPAIR_ARGV_DRIFT", "Phase-B repair argv drift")
    require("non-promotable" in str(lock.get("claim_ceiling", "")).lower(), "PHASE_B_REPAIR_CLAIM_DRIFT", "Phase-B repair claim ceiling drift")
    return lock


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        validate_execution_lock(args.execution_lock, enforce_argv=False)
    except Exception as error:
        print(json.dumps({"valid": False, "error_code": getattr(error, "code", type(error).__name__), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"valid": True, "terminal": "TARO_O0R_R6_FORMATION_PHASE_B_REPAIR_LOCK_VALID"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
