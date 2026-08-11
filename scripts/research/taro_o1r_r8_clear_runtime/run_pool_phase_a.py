#!/usr/bin/env python3
"""Run R8 candidate inference and source-only Phase A on the exact 24-parent pool."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation_io as r6io
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_phase_a as base
from scripts.research.taro_o1r_r8_clear_runtime import pool_cohort
from scripts.research.taro_o1r_r8_clear_runtime import run_pool_inventory


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r8_clear_pool_phase_a_execution_lock.v1"
LOCK_ID = "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_POOL_PHASE_A_SOURCE_AND_MODEL_ONE_SHOT_EXECUTION_LOCK"
INVENTORY_PATH = "artifacts.local/evidence/taro/o1r-r8-clear-pool-inventory-r0/exact-frame-plan.json"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r8-clear-pool-phase-a-r0"
PASS_TERMINAL = "TARO_O1R_R8_CLEAR_POOL_PHASE_A_SOURCE_ONLY_SEALED_PASS"
FAIL_TERMINAL = "TARO_O1R_R8_CLEAR_POOL_PHASE_A_EXECUTION_INVALID"
PARENT_COUNT = 24
FRAME_COUNT = 402
QUERY_COUNT = FRAME_COUNT * 9
FROZEN_FRAME_COUNTS = [6, 18, 13, 16, 10, 19, 14, 16, 34, 15, 8, 15, 6, 13, 21, 11, 25, 12, 22, 37, 17, 25, 15, 14]
FROZEN_MATERIALIZED_BYTES = 1_353_307_907

EXPECTED_BINDINGS = {
    "R8_PROTOCOL": "docs/research/taro/TARO_O1R_R8_SOURCE_ONLY_CLEAR_NEGATIVE_CONTROL_COHORT_ENRICHMENT_PROTOCOL_LOCK_2026-08-12.json",
    "R8_DOWNLOAD_RECEIPTS": "artifacts.local/evidence/taro/o1r-r8-clear-pool-source-r0/download-receipts.json",
    "R8_DOWNLOAD_RESULT": "artifacts.local/evidence/taro/o1r-r8-clear-pool-source-r0/result.json",
    "R8_INVENTORY_PLAN": INVENTORY_PATH,
    "R8_INVENTORY_RESULT": "artifacts.local/evidence/taro/o1r-r8-clear-pool-inventory-r0/result.json",
    "R8_INVENTORY_MANIFEST": "artifacts.local/evidence/taro/o1r-r8-clear-pool-inventory-r0/manifest.json",
    "R8_INVENTORY_RUNTIME": "scripts/research/taro_o1r_r8_clear_runtime/run_pool_inventory.py",
    "DEPTHART_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/depthart_runner.py",
    "CANDIDATE_INPUT_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/candidate_inputs.py",
    "PROSPECTIVE_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/prospective_factor_runtime.py",
    "R6_REDUCER_RUNTIME": "scripts/research/taro_o1r_reducer_integration_runtime/reducer_integration.py",
    "LOCKED_UNCERTAINTY_LOADER": "scripts/research/taro_o1r_reducer_integration_runtime/locked_uncertainty.py",
    "LOCKED_UNCERTAINTY_ARTIFACT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-artifact.json.gz",
    "LOCKED_UNCERTAINTY_RECEIPT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-receipt.json",
    "R7_CANARY_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "SHARED_PHASE_A_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_phase_a.py",
    "R8_PHASE_A_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_pool_phase_a.py",
}
EXPECTED_AUTHORITY = {
    "source_frame_decode": True,
    "depthart_inference": True,
    "candidate_inference_count": FRAME_COUNT,
    "source_only_phase_a": True,
    "source_only_parent_scoring": False,
    "faro_payload_read": False,
    "truth_scoring": False,
    "threshold_fit": False,
    "training": False,
    "network": False,
    "device": False,
    "product": False,
    "safety": False,
}
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "授权",
    "scope": "Exact frozen R8 24-parent/402-frame source-only Phase A: registered RGB/intrinsics DepthART inference followed by Apple depth/confidence source features; zero FARO/truth reads and no training.",
}


class PoolPhaseAError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PoolPhaseAError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frames(inventory_path: Path) -> list[r6io.R6FrameRef]:
    inventory = run_pool_inventory.validate_inventory(
        _load_json(inventory_path),
        expected_frame_counts=FROZEN_FRAME_COUNTS,
        expected_materialized_bytes=FROZEN_MATERIALIZED_BYTES,
    )
    frames: list[r6io.R6FrameRef] = []
    for parent, expected, expected_count in zip(inventory["parents"], pool_cohort.EXPECTED_POOL, FROZEN_FRAME_COUNTS, strict=True):
        identity = (str(parent["visit_id"]), str(parent["video_id"]))
        require(identity == expected[:2], "R8_PHASE_A_ROSTER_DRIFT", "R8 inventory roster drift")
        bindings = parent["container_bindings"]
        up_path = _repo_path(bindings["upsampling"]["path"])
        intr_path = _repo_path(bindings["intrinsics"]["path"])
        traj_path = _repo_path(bindings["trajectory"]["path"])
        r6io._verify_container(up_path, bindings["upsampling"])
        r6io._verify_container(intr_path, bindings["intrinsics"])
        r6io._verify_container(traj_path, bindings["trajectory"])
        up_index = r6io._index_upsampling(up_path, identity[1])
        intr_index = r6io._index_intrinsics(intr_path, identity[1])
        trajectory = tuple(materializer.parse_trajectory_payload(traj_path.read_bytes()))
        tokens = parent["frame_plan"]["exact_timestamp_tokens"]
        require(len(tokens) == expected_count, "R8_PHASE_A_FRAME_COUNT_DRIFT", "R8 parent frame count drift")
        for token in tokens:
            require(all(token in up_index[role] for role in up_index) and token in intr_index, "R8_PHASE_A_MEMBER_MISSING", "R8 exact member is absent")
            members = {role: up_index[role][token] for role in up_index}
            members["intrinsics"] = intr_index[token]
            frames.append(
                r6io.R6FrameRef(
                    identity[0], identity[1], token, f"{identity[1]}:{token}", up_path, intr_path, traj_path,
                    {key: dict(value) for key, value in bindings.items()}, trajectory, members,
                )
            )
    require(len(frames) == FRAME_COUNT, "R8_PHASE_A_COHORT_DRIFT", "R8 cohort is not exact 24/402")
    require(len({(row.parent_id, row.video_id, row.timestamp_token) for row in frames}) == FRAME_COUNT, "R8_PHASE_A_KEY_DUPLICATE", "R8 frame key duplicated")
    return frames


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = _load_json(lock_path)
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R8_PHASE_A_LOCK_IDENTITY", "R8 Phase-A lock identity drift")
    require(lock.get("user_authority") == EXPECTED_USER_AUTHORITY, "R8_PHASE_A_USER_AUTHORITY", "R8 Phase-A user authority drift")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv, "R8_PHASE_A_ARGV_DRIFT", "R8 Phase-A argv drift")
    require(lock.get("inventory_path") == INVENTORY_PATH and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R8_PHASE_A_ROOT_DRIFT", "R8 Phase-A root policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R8_PHASE_A_BINDINGS", "R8 Phase-A binding count drift")
    verified: dict[str, dict[str, Any]] = {}
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in verified and EXPECTED_BINDINGS.get(role) == relative, "R8_PHASE_A_BINDING_ROW", "R8 Phase-A binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R8_PHASE_A_BINDING_HASH", f"R8 Phase-A binding drift: {relative}")
        verified[role] = row
    inventory = run_pool_inventory.validate_inventory(_load_json(_repo_path(INVENTORY_PATH)), expected_frame_counts=FROZEN_FRAME_COUNTS, expected_materialized_bytes=FROZEN_MATERIALIZED_BYTES)
    require(inventory["exact_pose_bounded_frame_count"] == FRAME_COUNT, "R8_PHASE_A_INVENTORY_INVALID", "R8 inventory count drift")
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "R8_PHASE_A_AUTHORITY_DRIFT", "R8 Phase-A authority drift")
    require(lock.get("resource_budget") == {"maximum_wall_seconds": 28800, "maximum_peak_rss_bytes": 17179869184, "maximum_cuda_allocated_bytes": 8500000000, "maximum_evidence_bytes": 2147483648}, "R8_PHASE_A_BUDGET_DRIFT", "R8 Phase-A budget drift")
    identity = lock.get("candidate_identity", {})
    source = Path(identity.get("source_root", "")).resolve()
    checkpoint = Path(identity.get("checkpoint_path", "")).resolve()
    require(identity.get("model_id") == adapter.BASELINE_MODEL_ID and identity.get("source_commit") == depthart_runner.EXPECTED_SOURCE_GIT_COMMIT and identity.get("checkpoint_sha256") == adapter.BASELINE_CHECKPOINT_SHA256 and identity.get("preprocess_id") == depthart_runner.PREPROCESS_ID and identity.get("postprocess_id") == depthart_runner.POSTPROCESS_ID, "R8_PHASE_A_CANDIDATE_IDENTITY", "R8 candidate identity drift")
    require(source.is_dir() and checkpoint.is_file() and checkpoint.stat().st_size == identity.get("checkpoint_bytes") and materializer.sha256_file(checkpoint) == identity.get("checkpoint_sha256"), "R8_PHASE_A_CANDIDATE_ASSET", "R8 candidate asset drift")
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(source), "status", "--short"], capture_output=True, text=True, check=True).stdout.strip()
    require(commit == identity["source_commit"] and not dirty, "R8_PHASE_A_CANDIDATE_SOURCE_DRIFT", "R8 candidate source tree drift")
    require(not _repo_path(OUTPUT_ROOT).exists(), "R8_PHASE_A_ROOT_COLLISION", "R8 Phase-A evidence root exists")
    lock["_lock_path"], lock["_source_root"], lock["_checkpoint_path"] = lock_path, source, checkpoint
    return lock


def _configure_shared_runtime() -> None:
    base.LOCK_SCHEMA = LOCK_SCHEMA
    base.LOCK_ID = LOCK_ID
    base.INVENTORY_PATH = INVENTORY_PATH
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.PASS_TERMINAL = PASS_TERMINAL
    base.FAIL_TERMINAL = FAIL_TERMINAL
    base.FRAME_COUNT = FRAME_COUNT
    base.QUERY_COUNT = QUERY_COUNT
    base.PARENT_COUNT = PARENT_COUNT
    base.EXPECTED_PARENT_IDENTITIES = pool_cohort.EXPECTED_POOL
    base.EXPECTED_BINDINGS = EXPECTED_BINDINGS
    base.EXPECTED_AUTHORITY = EXPECTED_AUTHORITY
    base._load_frames = load_frames
    base.validate_execution_lock = validate_execution_lock


def execute(lock_path: Path) -> dict[str, Any]:
    _configure_shared_runtime()
    return base.execute(lock_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        print(json.dumps({"terminal": FAIL_TERMINAL, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "execution_valid": result["execution_valid"], "parent_count": result["parent_count"], "frame_count": result["frame_count"], "query_count": result["query_count"]}, sort_keys=True))
    return 0


_configure_shared_runtime()


if __name__ == "__main__":
    raise SystemExit(main())
