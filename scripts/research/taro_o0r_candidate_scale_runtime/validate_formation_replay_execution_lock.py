#!/usr/bin/env python3
"""Validate the one-shot TARO R6 formation replay execution lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "blindassist.taro.o0r.r6_formation_replay_execution_lock.v1"
LOCK_ID = "TARO_O0R_R6_FORMATION_REPLAY_R2_ONE_SHOT_EXECUTION_LOCK"
EXPECTED_ROOTS = {
    "repo_root": "E:/linnan/linnan",
    "frame_plan_path": "E:/linnan/linnan/artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/exact-frame-plan.json.gz",
    "source_root": "E:/linnan/linnan/artifacts.local/datasets/taro/o0r-arkitscenes-source-adapter-r3",
    "source_evidence_root": "E:/linnan/linnan/artifacts.local/evidence/taro/o0r-arkitscenes-formation-source-r0",
    "fit_candidate_root": "E:/linnan/linnan/artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-adapter-fit-r5",
    "eval_candidate_root": "E:/linnan/linnan/artifacts.local/evidence/taro/o0r-arkitscenes-factor-headroom-r3",
    "output_root": "E:/linnan/linnan/artifacts.local/evidence/taro/o0r-r6-formation-replay-r2",
}
EXPECTED_COHORT_BINDINGS = {
    "frame_key_sequence_sha256": "CFAEA2236C5E1EA9A2DC811ECE432F955B680658E8D5D7B94468FEAEECE77653",
    "source_receipt_hash_sequence_sha256": "E2E6478B3819FF42B158516D8D5E7D862C0069C8172B874B535ED8D4C5DC1A43",
    "candidate_record_hash_sequence_sha256": "DA22AA6BC8B015B2587A26EB05FA842E6A513A28841838078B89301A3B303E01",
    "candidate_native_hash_sequence_sha256": "C6A41722142BCCCBC0CAE33BD1BF8FCA2BBA16FC5349C83D57273EABA3E71089",
}


class FormationReplayLockError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FormationReplayLockError(code, message, **context)


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
        raise FormationReplayLockError("FORMATION_EXECUTION_LOCK_READ_FAILED", "formation execution lock cannot be read") from error
    require(isinstance(lock, dict) and lock.get("schema") == SCHEMA and lock.get("lock_id") == LOCK_ID, "FORMATION_EXECUTION_LOCK_IDENTITY_DRIFT", "formation execution lock schema/identity drift")
    require(lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "FORMATION_ONE_SHOT_ALREADY_CONSUMED", "formation replay lock is not authorized and unconsumed")
    require(lock.get("roots") == EXPECTED_ROOTS, "FORMATION_EXECUTION_ROOT_DRIFT", "formation replay roots drift")
    require(lock.get("cohort_bindings") == EXPECTED_COHORT_BINDINGS, "FORMATION_COHORT_BINDING_DRIFT", "formation replay cohort bindings drift")
    require(lock.get("worker_count") == 4, "FORMATION_WORKER_COUNT_DRIFT", "formation replay worker count must remain four")
    require(lock.get("output_root_must_be_absent") is True and lock.get("overwrite") is False and lock.get("rerun") is False, "FORMATION_ONE_SHOT_POLICY_DRIFT", "formation replay one-shot policy drift")
    output_root = Path(EXPECTED_ROOTS["output_root"])
    require(not output_root.exists(), "FORMATION_OUTPUT_ROOT_COLLISION", "formation replay output root already exists")
    for name, raw in EXPECTED_ROOTS.items():
        if name == "output_root":
            continue
        target = Path(raw)
        require(target.is_dir() if name.endswith("root") else target.is_file(), "FORMATION_REQUIRED_ROOT_MISSING", "formation replay input root/path is missing", name=name)
    authority = lock.get("execution_authority")
    require(
        authority == {
            "non_promotable_formation_replay": True,
            "sealed_candidate_reuse": True,
            "source_and_faro_local_reads": True,
            "candidate_inference": False,
            "training": False,
            "network": False,
            "device": False,
            "deployment": False,
            "product": False,
            "safety": False,
        },
        "FORMATION_EXECUTION_AUTHORITY_DRIFT",
        "formation replay execution authority drift",
    )
    budget = lock.get("resource_budget")
    require(
        isinstance(budget, dict)
        and budget.get("maximum_wall_seconds") == 14400
        and budget.get("maximum_peak_rss_bytes") == 17179869184
        and budget.get("maximum_evidence_bytes") == 536870912
        and budget.get("maximum_cuda_allocated_bytes") == 0
        and budget.get("training_steps") == 0
        and budget.get("network_requests") == 0,
        "FORMATION_RESOURCE_BUDGET_DRIFT",
        "formation replay resource budget drift",
    )
    environment = lock.get("required_environment")
    require(isinstance(environment, dict) and bool(environment), "FORMATION_ENVIRONMENT_MISSING", "formation replay required environment is missing")
    for key, expected in environment.items():
        require(os.environ.get(key) == str(expected), "FORMATION_ENVIRONMENT_DRIFT", "formation replay environment drift", key=key, expected=expected, actual=os.environ.get(key))
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and bool(bindings), "FORMATION_EXECUTION_BINDINGS_MISSING", "formation replay bindings are missing")
    roles: set[str] = set()
    for binding in bindings:
        require(isinstance(binding, dict) and set(binding) == {"role", "path", "bytes", "sha256"}, "FORMATION_EXECUTION_BINDING_FIELDS", "formation replay binding fields drift")
        role = str(binding["role"])
        require(role not in roles, "FORMATION_EXECUTION_BINDING_DUPLICATE", "formation replay binding role duplicated", role=role)
        roles.add(role)
        relative = Path(str(binding["path"]))
        require(not relative.is_absolute() and ".." not in relative.parts, "FORMATION_EXECUTION_BINDING_ESCAPE", "formation replay binding path is unsafe", role=role)
        target = repo_root / relative
        if not (relative.parts and relative.parts[0] == "artifacts.local"):
            resolved = target.resolve()
            require(repo_root == resolved or repo_root in resolved.parents, "FORMATION_EXECUTION_BINDING_ESCAPE", "formation replay binding escapes repository", role=role)
        require(target.is_file() and target.stat().st_size == binding["bytes"] and _sha(target) == binding["sha256"], "FORMATION_EXECUTION_BINDING_DRIFT", "formation replay binding hash/size drift", role=role)
    require(roles == set(lock.get("required_binding_roles", [])), "FORMATION_EXECUTION_BINDING_ROLE_SET", "formation replay binding role set drift")
    if enforce_argv:
        unique = lock.get("unique_argv")
        require(isinstance(unique, list) and len(unique) == 5, "FORMATION_EXECUTION_ARGV_DRIFT", "formation replay unique argv is malformed")
        require(Path(sys.executable).resolve() == Path(unique[0]).resolve(), "FORMATION_EXECUTION_ARGV_DRIFT", "formation replay Python executable drift")
        require(Path(sys.argv[0]).resolve() == (repo_root / unique[2]).resolve() and sys.argv[1:] == unique[3:], "FORMATION_EXECUTION_ARGV_DRIFT", "formation replay argv drift", actual=[sys.argv[0], *sys.argv[1:]])
    require(isinstance(lock.get("claim_ceiling"), str) and "non-promotable" in lock["claim_ceiling"].lower(), "FORMATION_CLAIM_CEILING_DRIFT", "formation replay claim ceiling drift")
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
    print(json.dumps({"valid": True, "terminal": "TARO_O0R_R6_FORMATION_REPLAY_EXECUTION_LOCK_VALID"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
