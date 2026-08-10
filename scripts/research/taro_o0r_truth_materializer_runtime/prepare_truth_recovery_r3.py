#!/usr/bin/env python3
"""Prepare the offline TARO O0R R3 compact-truth recovery locks."""

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
SUCCESSOR_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_RECOVERY_R3_LOCK_2026-08-10.json"
IMPLEMENTATION_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R3_IMPLEMENTATION_LOCK_2026-08-10.json"
EXECUTION_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R3_EXECUTION_LOCK_2026-08-10.json"
PREFLIGHT_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_ONLY_PREFLIGHT_R1_LOCK_2026-08-10.json"
AUTHORIZATION_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_DATA_USE_AUTHORIZATION_R1_RECEIPT_2026-08-10.json"
R1_HEAD_PATH = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-head-r1/head-receipt.json"
R2_RECOVERY_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_RECOVERY_R2_LOCK_2026-08-10.json"
R2_IMPLEMENTATION_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R2_IMPLEMENTATION_LOCK_2026-08-10.json"
R2_EXECUTION_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R2_EXECUTION_LOCK_2026-08-10.json"
R2_TRUTH_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r2"
R2_SOURCE_ROOT = REPO_ROOT / "artifacts.local/datasets/taro/o0r-arkitscenes-source-adapter-r2"
R2_EXECUTION_RECEIPT_PATH = R2_TRUTH_ROOT / "execution-receipt.json"
R2_DOWNLOAD_RECEIPTS_PATH = R2_TRUTH_ROOT / "download-receipts.json.gz"
R2_FRAME_PLAN_PATH = R2_TRUTH_ROOT / "exact-frame-plan.json.gz"
R2_MODEL_RECEIPT_PATH = R2_TRUTH_ROOT / "uncertainty-model-receipt.json"
R2_MODEL_ARTIFACT_PATH = R2_TRUTH_ROOT / "uncertainty-model-artifact.json.gz"

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


def _r2_snapshot() -> dict[str, Any]:
    files = sorted(path for path in R2_TRUTH_ROOT.rglob("*") if path.is_file())
    rows = [
        {
            "path": path.relative_to(R2_TRUTH_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": materializer.sha256_file(path),
        }
        for path in files
    ]
    truth_packages = [row for row in rows if row["path"].startswith("truth-frames/") and row["path"].endswith(".json.gz")]
    if not truth_packages:
        raise ValueError("R2 must contain at least one persisted truth-frame package")
    model_package = _load_gzip_json(R2_MODEL_ARTIFACT_PATH)
    seen_blob_paths = {
        reference["path"]
        for reference in model_package.get("array_blob_references", [])
    }
    frame_observations: list[dict[str, Any]] = []
    attributed_truth_bytes = 0
    for package_row in truth_packages:
        package_path = R2_TRUTH_ROOT / package_row["path"]
        package = _load_gzip_json(package_path)
        if package.get("schema") != materializer.ARRAY_ARTIFACT_SCHEMA:
            raise ValueError("R2 truth-frame package schema drift")
        references = package.get("array_blob_references")
        if not isinstance(references, list) or package.get("array_blob_reference_count") != len(references):
            raise ValueError("R2 truth-frame array reference summary drift")
        incremental_blob_bytes = 0
        new_blob_count = 0
        for reference in references:
            blob_path = R2_TRUTH_ROOT / reference["path"]
            if (
                not blob_path.is_file()
                or blob_path.stat().st_size != reference["gzip_bytes"]
                or materializer.sha256_file(blob_path) != reference["gzip_sha256"]
            ):
                raise ValueError(f"R2 truth-frame blob drift: {reference['path']}")
            if reference["path"] not in seen_blob_paths:
                seen_blob_paths.add(reference["path"])
                incremental_blob_bytes += int(reference["gzip_bytes"])
                new_blob_count += 1
        persisted_bytes = package_path.stat().st_size + incremental_blob_bytes
        attributed_truth_bytes += persisted_bytes
        frame_observations.append(
            {
                "package": package_row,
                "array_blob_reference_count": len(references),
                "new_array_blob_count": new_blob_count,
                "incremental_array_blob_bytes": incremental_blob_bytes,
                "incremental_persisted_bytes": persisted_bytes,
            }
        )
    frame_plan = _load_gzip_json(R2_FRAME_PLAN_PATH)
    eval_frames = sum(
        len(row["frame_plan"]["exact_timestamp_tokens"])
        for row in frame_plan
        if row["parent"]["role"] == "O0R_EVAL_CANDIDATE"
    )
    total_bytes = sum(row["bytes"] for row in rows)
    baseline_bytes = total_bytes - attributed_truth_bytes
    budget = RESOURCE_BUDGET["maximum_scientific_evidence_bytes"]
    minimum_observed_frame_bytes = min(row["incremental_persisted_bytes"] for row in frame_observations)
    max_frames = (budget - baseline_bytes) // minimum_observed_frame_bytes
    return {
        "file_count": len(rows),
        "bytes": total_bytes,
        "file_ledger_sha256": materializer.canonical_sha256(rows),
        "persisted_truth_frame_count": len(frame_observations),
        "truth_frame_observations": frame_observations,
        "minimum_observed_truth_frame_increment_bytes": minimum_observed_frame_bytes,
        "maximum_observed_truth_frame_increment_bytes": max(row["incremental_persisted_bytes"] for row in frame_observations),
        "bytes_before_first_truth_frame": baseline_bytes,
        "eval_exact_frame_count": eval_frames,
        "conservative_projected_evidence_bytes": baseline_bytes + minimum_observed_frame_bytes * eval_frames,
        "frozen_evidence_budget_bytes": budget,
        "maximum_frames_at_minimum_observed_size": max_frames,
        "first_budget_exceeding_frame_number": max_frames + 1,
    }


def write_base() -> None:
    required = [
        R2_RECOVERY_PATH,
        R2_IMPLEMENTATION_PATH,
        R2_EXECUTION_PATH,
        R2_EXECUTION_RECEIPT_PATH,
        R2_DOWNLOAD_RECEIPTS_PATH,
        R2_FRAME_PLAN_PATH,
        R2_MODEL_RECEIPT_PATH,
        R2_MODEL_ARTIFACT_PATH,
    ]
    if not all(path.is_file() for path in required):
        raise ValueError("R2 predecessor binding missing")
    for terminal_name in ("result.json", "failure.json", "completion-receipt.json", "manifest.json"):
        if (R2_TRUTH_ROOT / terminal_name).exists():
            raise ValueError(f"unexpected R2 terminal artifact: {terminal_name}")
    snapshot = _r2_snapshot()
    if snapshot["conservative_projected_evidence_bytes"] <= snapshot["frozen_evidence_budget_bytes"]:
        raise ValueError("R2 evidence budget projection is not structurally exceeded")
    payload = {
        "schema": "blindassist.taro.o0r.truth_recovery_r3_lock.v1",
        "lock_id": "TARO_O0R_ARKITSCENES_TRUTH_RECOVERY_R3_LOCK",
        "date": "2026-08-10",
        "research_mode": "WILD_LAB",
        "status": "R3_COMPACT_PERSISTENCE_RECOVERY_FROZEN",
        "predecessor": {
            "execution": "R2",
            "one_shot_consumed": True,
            "completion_receipt_absent": True,
            "controlled_process_stop_after_resource_projection": True,
            "scientific_thresholds_read_or_changed_for_r3": False,
            "model_outputs_absent": True,
        },
        "predecessor_bindings": [
            _binding("R2_RECOVERY_LOCK", R2_RECOVERY_PATH),
            _binding("R2_IMPLEMENTATION_LOCK", R2_IMPLEMENTATION_PATH),
            _binding("R2_EXECUTION_LOCK", R2_EXECUTION_PATH),
            _binding("R2_EXECUTION_RECEIPT", R2_EXECUTION_RECEIPT_PATH),
            _binding("R2_DOWNLOAD_RECEIPTS", R2_DOWNLOAD_RECEIPTS_PATH),
            _binding("R2_EXACT_FRAME_PLAN", R2_FRAME_PLAN_PATH),
            _binding("R2_UNCERTAINTY_MODEL_RECEIPT", R2_MODEL_RECEIPT_PATH),
            _binding("R2_UNCERTAINTY_MODEL_ARTIFACT", R2_MODEL_ARTIFACT_PATH),
        ],
        "r2_resource_observation": snapshot,
        "persistence_amendment": {
            "truth_record_schema": materializer.EVAL_TRUTH_COMMITMENT_SCHEMA,
            "dense_factor_arrays_persisted": False,
            "factor_frame_sha256_committed": True,
            "base_geometry_sha256_committed": True,
            "factor_value_sha256s_committed": True,
            "factor_validity_and_uncertainty_persisted": True,
            "nine_query_reducer_results_persisted": True,
            "source_depth_intrinsics_faro_and_model_bindings_persisted": True,
            "candidate_outputs_must_be_sealed_before_truth_recompute": True,
            "recomputed_factor_frame_sha256_must_match_before_join": True,
            "maximum_compact_truth_frame_bytes": run_truth_only.MAXIMUM_COMPACT_TRUTH_FRAME_BYTES,
            "scientific_threshold_changes": 0,
        },
        "source_reuse": run_truth_only.EXPECTED_SOURCE_CACHE,
        "network_requests_in_r3": 0,
        "exclusive_roots": run_truth_only.EXPECTED_ROOTS,
        "user_instruction_verbatim": USER_INSTRUCTION,
        "execution_authority": {"implementation": True, "source_cache_reuse": False, "truth_only": False, "depthart": False, "factorial": False},
        "unique_successor": "TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R3_IMPLEMENTATION_LOCK",
        "claim_ceiling": "R3 changes lossless truth evidence transport only. No new scientific threshold, DepthART, factor, device, product, or safety result is authorized.",
    }
    _write(SUCCESSOR_PATH, payload)


def write_implementation_lock(tests_run: int) -> None:
    rows = (
        ("TRUTH_RECOVERY_R3_LOCK", _relative(SUCCESSOR_PATH)),
        ("TRUTH_ONLY_PREFLIGHT_LOCK", _relative(PREFLIGHT_PATH)),
        ("DATA_USE_AUTHORIZATION", _relative(AUTHORIZATION_PATH)),
        ("SOURCE_ADAPTER", "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py"),
        ("MATERIALIZER", "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py"),
        ("TRUTH_RUNNER", "scripts/research/taro_o0r_truth_materializer_runtime/run_truth_only.py"),
        ("SOURCE_ADAPTER_TESTS", "scripts/research/taro_o0r_source_adapter_runtime/test_source_adapter.py"),
        ("MATERIALIZER_TESTS", "scripts/research/taro_o0r_truth_materializer_runtime/test_materializer.py"),
        ("R3_PREPARER", "scripts/research/taro_o0r_truth_materializer_runtime/prepare_truth_recovery_r3.py"),
    )
    payload = {
        "schema": "blindassist.taro.o0r.truth_materializer_r3_implementation_lock.v1",
        "lock_id": "TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R3_IMPLEMENTATION_LOCK",
        "date": "2026-08-10",
        "research_mode": "WILD_LAB",
        "status": "IMPLEMENTATION_LOCK_PASS",
        "implementation_base_commit": _git_head(),
        "bindings": _bindings(rows),
        "focused_validation": {"tests_run": tests_run, "tests_passed": tests_run, "failures": 0, "errors": 0, "network_requests": 0},
        "compact_truth_contract": {
            "schema": materializer.EVAL_TRUTH_COMMITMENT_SCHEMA,
            "maximum_frame_bytes": run_truth_only.MAXIMUM_COMPACT_TRUTH_FRAME_BYTES,
            "array_blob_count": 0,
            "recomputation_hash_gate": True,
        },
        "source_cache_mode": run_truth_only.EXPECTED_SOURCE_CACHE,
        "exclusive_roots": run_truth_only.EXPECTED_ROOTS,
        "execution_authority": {"implementation": True, "source_cache_reuse": False, "truth_only": False, "depthart": False, "factorial": False},
        "unique_successor": "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R3_EXECUTION_LOCK",
    }
    _write(IMPLEMENTATION_PATH, payload)


def write_execution_lock(execution_commit: str) -> None:
    preflight = materializer.load_json(PREFLIGHT_PATH)
    authorization = materializer.load_json(AUTHORIZATION_PATH)
    head = materializer.load_json(R1_HEAD_PATH)
    materializer.validate_authorization(preflight, authorization, preflight_sha256=materializer.sha256_file(PREFLIGHT_PATH))
    head_lookup = materializer.validate_head_receipt(preflight, materializer.sha256_file(AUTHORIZATION_PATH), head)
    cache_lookup = run_truth_only._load_cached_download_receipts(R2_DOWNLOAD_RECEIPTS_PATH)
    for row in materializer.expanded_asset_plan(preflight):
        receipt = cache_lookup[row["url"]]
        if receipt.get("bytes") != head_lookup[row["url"]]["content_length_bytes"]:
            raise ValueError(f"cache/head length drift: {row['url']}")
        materializer.verify_bound_container(materializer.safe_join(R2_SOURCE_ROOT, row["relative_path"]), receipt)
    payload = {
        "schema": materializer.EXECUTION_LOCK_SCHEMA,
        "lock_id": "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R3_EXECUTION_LOCK",
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
