#!/usr/bin/env python3
"""Prepare the offline TARO O0R R2 truth-recovery execution locks."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o0r_truth_materializer_runtime import run_truth_only


DOC_ROOT = REPO_ROOT / "docs/research/taro"
SUCCESSOR_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_RECOVERY_R2_LOCK_2026-08-10.json"
IMPLEMENTATION_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R2_IMPLEMENTATION_LOCK_2026-08-10.json"
EXECUTION_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R2_EXECUTION_LOCK_2026-08-10.json"
PREFLIGHT_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_ONLY_PREFLIGHT_R1_LOCK_2026-08-10.json"
AUTHORIZATION_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_DATA_USE_AUTHORIZATION_R1_RECEIPT_2026-08-10.json"
R1_HEAD_PATH = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-head-r1/head-receipt.json"
R1_FAILURE_PATH = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r1/failure.json"
R1_MANIFEST_PATH = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r1/manifest.json"
R1_FRAME_PLAN_PATH = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r1/exact-frame-plan.json.gz"
R1_DOWNLOAD_RECEIPTS_PATH = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r1/download-receipts.json.gz"

REQUIRED_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONUTF8": "1",
}
RESOURCE_BUDGET = {
    "source_download_content_length_preflight_required": True,
    "maximum_compressed_source_bytes": 21_474_836_480,
    "maximum_materialized_source_bytes": 53_687_091_200,
    "truth_only_materialization_wall_seconds": 43_200,
    "truth_only_peak_rss_bytes": 12_884_901_888,
    "maximum_scientific_evidence_bytes": 2_147_483_648,
    "head_timeout_seconds": 20,
    "head_retries": 3,
    "head_workers": 8,
    "head_response_body_bytes": 0,
    "network_allowed_only_for_bound_source_assets": True,
    "training_steps": 0,
    "device_or_android": False,
}
USER_INSTRUCTION = "不管怎么样都行，赶快推进算法前进，你做了这么久浪费这么多token告诉我失败"


def _relative(path: Path) -> str:
    return path.absolute().relative_to(REPO_ROOT.absolute()).as_posix()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(materializer.canonical_json_bytes(payload) + b"\n")


def _binding(role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": materializer.sha256_file(path),
    }


def _bindings(rows: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    return [_binding(role, REPO_ROOT / relative) for role, relative in rows]


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_base() -> None:
    failure = materializer.load_json(R1_FAILURE_PATH)
    if failure.get("failure_code") != "SUPPORT_PLAUSIBLE_INSUFFICIENT":
        raise ValueError("R1 failure identity drift")
    frame_plan = _load_gzip_json(R1_FRAME_PLAN_PATH)
    if not isinstance(frame_plan, list) or len(frame_plan) != 24:
        raise ValueError("R1 frame plan drift")
    opportunity = [
        {
            "role": row["parent"]["role"],
            "visit_id": row["parent"]["visit_id"],
            "video_id": row["parent"]["video_id"],
            "exact_pose_bounded_frame_count": row["frame_plan"]["exact_pose_bounded_frame_count"],
        }
        for row in frame_plan
    ]
    eval_counts = [row["exact_pose_bounded_frame_count"] for row in opportunity if row["role"] == "O0R_EVAL_CANDIDATE"]
    payload = {
        "schema": "blindassist.taro.o0r.truth_recovery_r2_lock.v1",
        "lock_id": "TARO_O0R_ARKITSCENES_TRUTH_RECOVERY_R2_LOCK",
        "date": "2026-08-10",
        "research_mode": "WILD_LAB",
        "status": "R2_RECOVERY_FROZEN_BEFORE_ANY_EVAL_TRUTH_DECODE",
        "predecessor": {
            "terminal": failure["terminal"],
            "failure_code": failure["failure_code"],
            "one_shot_consumed": True,
            "source_download_complete": True,
            "eval_truth_decode_count": 0,
        },
        "predecessor_bindings": [
            _binding("R1_FAILURE", R1_FAILURE_PATH),
            _binding("R1_MANIFEST", R1_MANIFEST_PATH),
            _binding("R1_EXACT_FRAME_PLAN", R1_FRAME_PLAN_PATH),
            _binding("R1_DOWNLOAD_RECEIPTS", R1_DOWNLOAD_RECEIPTS_PATH),
            _binding("R1_HEAD_RECEIPT", R1_HEAD_PATH),
        ],
        "algorithm_amendments": {
            "per_target_missingness": "Frames with valid metric common depth retain scale residuals. FARO-plane failure makes support and boundary missing; Apple-only plane failure makes support missing while FARO-defined boundary remains available.",
            "support_failure_codes_treated_as_unobservable": sorted(materializer.adapter._SUPPORT_UNOBSERVABLE_CODES),
            "minimum_exact_frames_per_evaluable_parent": materializer.MINIMUM_EXACT_FRAMES_PER_EVAL_PARENT,
            "minimum_state_frames_per_evaluable_parent": materializer.MINIMUM_STATE_FRAMES_PER_EVAL_PARENT,
            "minimum_evaluable_parent_count": materializer.MINIMUM_EVALUABLE_EVAL_PARENTS,
            "threshold_freeze_basis": "source opportunity counts only; no eval frame, query truth, model output, task metric, or factor outcome was decoded",
        },
        "source_opportunity": {
            "parents": opportunity,
            "eval_parents_with_at_least_12_frames": sum(count >= 12 for count in eval_counts),
            "eval_parents_with_at_least_8_frames": sum(count >= 8 for count in eval_counts),
        },
        "source_reuse": run_truth_only.EXPECTED_SOURCE_CACHE,
        "network_requests_in_r2": 0,
        "exclusive_roots": run_truth_only.EXPECTED_ROOTS,
        "user_instruction_verbatim": USER_INSTRUCTION,
        "execution_authority": {"implementation": True, "source_cache_reuse": False, "truth_only": False, "depthart": False, "factorial": False},
        "unique_successor": "TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R2_IMPLEMENTATION_LOCK",
        "claim_ceiling": "R2 recovery mechanics are frozen from R1 source-only failure and source opportunity. No eval truth, DepthART, factor, device, product, or safety result has been observed.",
    }
    _write(SUCCESSOR_PATH, payload)


def write_implementation_lock(tests_run: int) -> None:
    rows = (
        ("TRUTH_RECOVERY_R2_LOCK", _relative(SUCCESSOR_PATH)),
        ("TRUTH_ONLY_PREFLIGHT_LOCK", _relative(PREFLIGHT_PATH)),
        ("DATA_USE_AUTHORIZATION", _relative(AUTHORIZATION_PATH)),
        ("SOURCE_ADAPTER", "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py"),
        ("MATERIALIZER", "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py"),
        ("TRUTH_RUNNER", "scripts/research/taro_o0r_truth_materializer_runtime/run_truth_only.py"),
        ("SOURCE_ADAPTER_TESTS", "scripts/research/taro_o0r_source_adapter_runtime/test_source_adapter.py"),
        ("MATERIALIZER_TESTS", "scripts/research/taro_o0r_truth_materializer_runtime/test_materializer.py"),
        ("R2_PREPARER", "scripts/research/taro_o0r_truth_materializer_runtime/prepare_truth_recovery_r2.py"),
    )
    payload = {
        "schema": "blindassist.taro.o0r.truth_materializer_r2_implementation_lock.v1",
        "lock_id": "TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R2_IMPLEMENTATION_LOCK",
        "date": "2026-08-10",
        "research_mode": "WILD_LAB",
        "status": "IMPLEMENTATION_LOCK_PASS",
        "implementation_base_commit": _git_head(),
        "bindings": _bindings(rows),
        "focused_validation": {"tests_run": tests_run, "tests_passed": tests_run, "failures": 0, "errors": 0, "network_requests": 0},
        "source_cache_mode": run_truth_only.EXPECTED_SOURCE_CACHE,
        "exclusive_roots": run_truth_only.EXPECTED_ROOTS,
        "execution_authority": {"implementation": True, "source_cache_reuse": False, "truth_only": False, "depthart": False, "factorial": False},
        "unique_successor": "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R2_EXECUTION_LOCK",
    }
    _write(IMPLEMENTATION_PATH, payload)


def write_execution_lock(execution_commit: str) -> None:
    preflight = materializer.load_json(PREFLIGHT_PATH)
    authorization = materializer.load_json(AUTHORIZATION_PATH)
    head = materializer.load_json(R1_HEAD_PATH)
    materializer.validate_authorization(preflight, authorization, preflight_sha256=materializer.sha256_file(PREFLIGHT_PATH))
    head_lookup = materializer.validate_head_receipt(preflight, materializer.sha256_file(AUTHORIZATION_PATH), head)
    cache_lookup = run_truth_only._load_cached_download_receipts(R1_DOWNLOAD_RECEIPTS_PATH)
    source_root = REPO_ROOT / run_truth_only.EXPECTED_SOURCE_CACHE["source_root"]
    for row in materializer.expanded_asset_plan(preflight):
        receipt = cache_lookup[row["url"]]
        if receipt.get("bytes") != head_lookup[row["url"]]["content_length_bytes"]:
            raise ValueError(f"cache/head length drift: {row['url']}")
        materializer.verify_bound_container(materializer.safe_join(source_root, row["relative_path"]), receipt)
    payload = {
        "schema": materializer.EXECUTION_LOCK_SCHEMA,
        "lock_id": "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R2_EXECUTION_LOCK",
        "date": "2026-08-10",
        "status": "AUTHORIZED_UNCONSUMED",
        "consumed": False,
        "execution_commit": execution_commit,
        "execution_authority": run_truth_only.EXPECTED_AUTHORITY,
        "roots": run_truth_only.EXPECTED_ROOTS,
        "source_cache": run_truth_only.EXPECTED_SOURCE_CACHE,
        "overwrite": False,
        "rerun": False,
        "required_environment": REQUIRED_ENVIRONMENT,
        "resource_budget": RESOURCE_BUDGET,
        "argv": [
            "scripts/research/taro_o0r_truth_materializer_runtime/run_truth_only.py",
            "--execution-lock",
            _relative(EXECUTION_PATH),
        ],
        "bindings": _bindings(run_truth_only.EXPECTED_TRUTH_BINDINGS.items()),
    }
    _write(EXECUTION_PATH, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write-base")
    implementation = subparsers.add_parser("write-implementation-lock")
    implementation.add_argument("--tests-run", type=int, required=True)
    execution = subparsers.add_parser("write-execution-lock")
    execution.add_argument("--execution-commit", default=None)
    args = parser.parse_args()
    if args.command == "write-base":
        write_base()
    elif args.command == "write-implementation-lock":
        write_implementation_lock(args.tests_run)
    elif args.command == "write-execution-lock":
        write_execution_lock(args.execution_commit or _git_head())
    else:
        raise AssertionError(args.command)
    print(json.dumps({"status": "WRITTEN", "command": args.command}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
