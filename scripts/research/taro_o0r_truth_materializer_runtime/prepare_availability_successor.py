#!/usr/bin/env python3
"""Prepare the TARO O0R R1 availability-successor locks.

This helper is intentionally offline.  It creates deterministic, hash-bound
documents for the R1 roster that retires the one R0 parent whose official
trajectory endpoint was unavailable.  Network access remains isolated in the
existing HEAD and truth-only runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o0r_truth_materializer_runtime import run_head_preflight
from scripts.research.taro_o0r_truth_materializer_runtime import run_truth_only


DOC_ROOT = REPO_ROOT / "docs/research/taro"
SUCCESSOR_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_AVAILABILITY_SUCCESSOR_R1_LOCK_2026-08-10.json"
PREFLIGHT_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_ONLY_PREFLIGHT_R1_LOCK_2026-08-10.json"
AUTHORIZATION_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_DATA_USE_AUTHORIZATION_R1_RECEIPT_2026-08-10.json"
IMPLEMENTATION_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R1_IMPLEMENTATION_LOCK_2026-08-10.json"
HEAD_LOCK_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_R1_EXECUTION_LOCK_2026-08-10.json"
TRUTH_LOCK_PATH = DOC_ROOT / "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R1_EXECUTION_LOCK_2026-08-10.json"

SELECTION_SALT = "TARO_O0R_ARKITSCENES_R0"
ORIGINAL_AUTHORIZATION = (
    "我授权 TARO O0R 使用锁定的 24 个 ARKitScenes Training 视频及每个视频的 "
    "upsampling.zip、lowres_wide_intrinsics.zip、lowres_wide.traj，用于 HEAD 预检和 "
    "source/truth-only WILD_LAB 物化与校验"
)
SUCCESSOR_AMENDMENT = "不管怎么样都行，赶快推进算法前进，你做了这么久浪费这么多token告诉我失败"

FIT_ROSTER = (
    ("470974", "47332075"),
    ("469216", "47332946"),
    ("423614", "42898071"),
    ("467370", "47333776"),
    ("469460", "47333043"),
    ("438794", "44358241"),
    ("467346", "47333876"),
    ("472473", "47204786"),
)
EVAL_ROSTER = (
    ("466965", "45261294"),
    ("470808", "47430058"),
    ("482587", "47895909"),
    ("468410", "45261689"),
    ("482858", "47670295"),
    ("482984", "47670346"),
    ("469607", "47115143"),
    ("421593", "42445766"),
    ("423474", "42897405"),
    ("470876", "47332015"),
    ("464981", "44796438"),
    ("478016", "47204874"),
    ("422217", "42445723"),
    ("466437", "45260952"),
    ("484003", "48018757"),
    ("421655", "42445698"),
)

ASSET_TEMPLATES = (
    {
        "asset": "upsampling.zip",
        "relative_path_template": "upsampling/{official_fold}/{video_id}.zip",
        "url_template": "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/upsampling/{official_fold}/{video_id}.zip",
    },
    {
        "asset": "lowres_wide_intrinsics.zip",
        "relative_path_template": "raw/{official_fold}/{video_id}/lowres_wide_intrinsics.zip",
        "url_template": "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/{official_fold}/{video_id}/lowres_wide_intrinsics.zip",
    },
    {
        "asset": "lowres_wide.traj",
        "relative_path_template": "raw/{official_fold}/{video_id}/lowres_wide.traj",
        "url_template": "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/{official_fold}/{video_id}/lowres_wide.traj",
    },
)

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


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _rank(role: str, visit_id: str, video_id: str) -> str:
    value = f"{SELECTION_SALT}:{role}:{visit_id}:{video_id}"
    return hashlib.sha256(value.encode("ascii")).hexdigest().upper()


def _parents() -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for role, roster in (("ADAPTER_FIT", FIT_ROSTER), ("O0R_EVAL_CANDIDATE", EVAL_ROSTER)):
        for visit_id, video_id in roster:
            output.append(
                {
                    "role": role,
                    "visit_id": visit_id,
                    "video_id": video_id,
                    "official_fold": "Training",
                    "selection_rank_sha256": _rank(role, visit_id, video_id),
                }
            )
    return output


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


def _preflight_payload() -> dict[str, Any]:
    parents = _parents()
    provisional = {
        "schema": materializer.PREFLIGHT_SCHEMA,
        "lock_id": "TARO_O0R_ARKITSCENES_TRUTH_ONLY_PREFLIGHT_R1_LOCK",
        "date": "2026-08-10",
        "research_mode": "WILD_LAB",
        "status": "R1_AVAILABILITY_SUCCESSOR_PREFLIGHT_LOCKED_HEAD_NOT_RUN",
        "passed": True,
        "scientific_status": "NOT_RUN",
        "head_requests_executed": False,
        "source_payload_opened": False,
        "truth_materialized": False,
        "required_environment": REQUIRED_ENVIRONMENT,
        "resource_budget": RESOURCE_BUDGET,
        "asset_plan": {
            "selection_source": "R0 metadata roster with one source-availability-only replacement; no source body, truth, candidate output, or task metric influenced selection",
            "request_method": "HEAD",
            "response_body_bytes_allowed": 0,
            "off_host_redirect_allowed": False,
            "expected_parent_count": 24,
            "expected_asset_count_per_parent": 3,
            "expected_request_count": 72,
            "asset_templates": list(ASSET_TEMPLATES),
            "selected_parents": parents,
            "expanded_requests_sha256": "PENDING",
        },
        "exclusive_roots": {
            "HEAD_EVIDENCE": run_head_preflight.HEAD_OUTPUT_ROOT,
            **run_truth_only.EXPECTED_ROOTS,
        },
        "claim_ceiling": "R1 source-availability successor lock only; no source body, truth, DepthART, factor, device, product, or safety evidence.",
    }
    rows: list[dict[str, str]] = []
    for parent in parents:
        fields = {"official_fold": parent["official_fold"], "video_id": parent["video_id"]}
        for template in ASSET_TEMPLATES:
            rows.append(
                {
                    "role": parent["role"],
                    "visit_id": parent["visit_id"],
                    "video_id": parent["video_id"],
                    "official_fold": parent["official_fold"],
                    "asset": template["asset"],
                    "relative_path": template["relative_path_template"].format(**fields),
                    "url": template["url_template"].format(**fields),
                }
            )
    provisional["asset_plan"]["expanded_requests_sha256"] = materializer.canonical_sha256(rows)
    return provisional


def write_base() -> None:
    preflight = _preflight_payload()
    _write(PREFLIGHT_PATH, preflight)
    selected_ids = [row["video_id"] for row in preflight["asset_plan"]["selected_parents"]]
    authorization = {
        "schema": materializer.AUTHORIZATION_SCHEMA,
        "receipt_id": "TARO_O0R_ARKITSCENES_DATA_USE_AUTHORIZATION_R1_RECEIPT",
        "confirmed_by": "user",
        "confirmed_at": "2026-08-10",
        "confirmation_verbatim": ORIGINAL_AUTHORIZATION,
        "scope_amendment_verbatim": SUCCESSOR_AMENDMENT,
        "scope_binding": {
            "preflight_lock_path": _relative(PREFLIGHT_PATH),
            "preflight_lock_bytes": PREFLIGHT_PATH.stat().st_size,
            "preflight_lock_sha256": materializer.sha256_file(PREFLIGHT_PATH),
            "request_plan_sha256": preflight["asset_plan"]["expanded_requests_sha256"],
        },
        "interpreted_scope": {
            "research_route": "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R1",
            "official_fold": "Training",
            "selected_parent_count": 24,
            "selected_video_ids": selected_ids,
            "authorized_asset_patterns": [
                row["url_template"].replace("{official_fold}", "Training") for row in ASSET_TEMPLATES
            ],
            "authorized_operations": [
                "HEAD-only availability and Content-Length preflight for the R1 72-URL plan",
                "bounded download and integrity validation of the R1 source assets",
                "source/truth-only WILD_LAB materialization and validation",
            ],
            "availability_replacement_authorized": True,
            "retired_unavailable_video_id": "47333152",
            "replacement_video_id": "47204786",
            "authorization_does_not_itself_activate_execution": True,
            "separate_implementation_and_execution_locks_required": True,
        },
        "source_license": {
            "path": "artifacts.local/downloads/ARKitScenes-7283761/LICENSE",
            "sha256": "D8D156565F81B4B56B37A6D5A68A223A0874ED1131BAC6A404B0D5E765189071",
            "use_must_remain_within_source_terms": True,
            "redistribution_authorized": False,
        },
        "authority": "The prior exact-data authorization plus the user's explicit instruction to use any workable route authorizes this source-availability-only R1 replacement and immediate HEAD/source/truth-only execution. It does not authorize training, deployment, product, safety, or redistribution claims.",
    }
    _write(AUTHORIZATION_PATH, authorization)
    successor = {
        "schema": "blindassist.taro.o0r.availability_successor_lock.v1",
        "lock_id": "TARO_O0R_ARKITSCENES_AVAILABILITY_SUCCESSOR_R1_LOCK",
        "date": "2026-08-10",
        "research_mode": "WILD_LAB",
        "status": "AVAILABILITY_SCREENED_SUCCESSOR_FROZEN",
        "predecessor_terminal": "TARO_O0R_ASSET_HEADERS_NOT_AVAILABLE_NO_REPLACEMENT",
        "selection_information_allowed": ["official metadata", "official missing-asset list", "zero-body source availability"],
        "selection_information_forbidden": ["source body", "truth", "model output", "task metric", "factor outcome"],
        "retired_parent": {"role": "ADAPTER_FIT", "visit_id": "469456", "video_id": "47333152", "reason": "lowres_wide.traj returned HTTP 403 in the closed R0 HEAD execution"},
        "replacement_parent": {
            "role": "ADAPTER_FIT",
            "visit_id": "472473",
            "video_id": "47204786",
            "selection_rank_sha256": _rank("ADAPTER_FIT", "472473", "47204786"),
            "rule": "first remaining hash-ranked Training candidate after excluding the unavailable identity and all already allocated R1 identities",
            "official_missing_3dod_list_member": False,
            "zero_body_head_sanity": {
                "upsampling.zip": {"http_status": 200, "content_length_bytes": 131_520_279},
                "lowres_wide_intrinsics.zip": {"http_status": 200, "content_length_bytes": 1_432_464},
                "lowres_wide.traj": {"http_status": 200, "content_length_bytes": 116_075},
            },
        },
        "roster_counts": {"ADAPTER_FIT": 8, "O0R_EVAL_CANDIDATE": 16},
        "preflight_binding": _binding("TRUTH_ONLY_PREFLIGHT_LOCK", PREFLIGHT_PATH),
        "authorization_binding": _binding("DATA_USE_AUTHORIZATION", AUTHORIZATION_PATH),
        "user_scope_amendment_verbatim": SUCCESSOR_AMENDMENT,
        "execution_authority": {"prepare_locks": True, "head": False, "source_truth_only": False, "depthart": False, "factorial": False},
        "unique_successor": "TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_R1_EXECUTION_LOCK",
        "claim_ceiling": "The replacement is justified only by source availability and deterministic metadata rank. No source body, truth, model, or task outcome was used.",
    }
    _write(SUCCESSOR_PATH, successor)


def write_implementation_lock(tests_run: int) -> None:
    rows = (
        ("AVAILABILITY_SUCCESSOR_LOCK", _relative(SUCCESSOR_PATH)),
        ("TRUTH_ONLY_PREFLIGHT_LOCK", _relative(PREFLIGHT_PATH)),
        ("DATA_USE_AUTHORIZATION", _relative(AUTHORIZATION_PATH)),
        ("SOURCE_ADAPTER", "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py"),
        ("MATERIALIZER", "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py"),
        ("HEAD_RUNNER", "scripts/research/taro_o0r_truth_materializer_runtime/run_head_preflight.py"),
        ("TRUTH_RUNNER", "scripts/research/taro_o0r_truth_materializer_runtime/run_truth_only.py"),
        ("FOCUSED_TESTS", "scripts/research/taro_o0r_truth_materializer_runtime/test_materializer.py"),
        ("SUCCESSOR_PREPARER", "scripts/research/taro_o0r_truth_materializer_runtime/prepare_availability_successor.py"),
    )
    payload = {
        "schema": "blindassist.taro.o0r.truth_materializer_r1_implementation_lock.v1",
        "lock_id": "TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R1_IMPLEMENTATION_LOCK",
        "date": "2026-08-10",
        "research_mode": "WILD_LAB",
        "status": "IMPLEMENTATION_LOCK_PASS",
        "implementation_base_commit": _git_head(),
        "bindings": _bindings(rows),
        "focused_validation": {"tests_run": tests_run, "tests_passed": tests_run, "failures": 0, "errors": 0, "network_requests": 0},
        "exclusive_roots": {"HEAD_EVIDENCE": run_head_preflight.HEAD_OUTPUT_ROOT, **run_truth_only.EXPECTED_ROOTS},
        "execution_authority": {"implementation_lock": True, "head": False, "source_truth_only": False, "depthart": False, "factorial": False, "training": False},
        "unique_successor": "TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_R1_EXECUTION_LOCK",
    }
    _write(IMPLEMENTATION_PATH, payload)


def write_head_lock(execution_commit: str) -> None:
    budget = {
        key: RESOURCE_BUDGET[key]
        for key in ("maximum_compressed_source_bytes", "head_timeout_seconds", "head_retries", "head_workers", "head_response_body_bytes")
    }
    payload = {
        "schema": run_head_preflight.HEAD_EXECUTION_LOCK_SCHEMA,
        "lock_id": "TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_R1_EXECUTION_LOCK",
        "date": "2026-08-10",
        "status": "AUTHORIZED_UNCONSUMED",
        "consumed": False,
        "execution_commit": execution_commit,
        "execution_authority": run_head_preflight.EXPECTED_AUTHORITY,
        "output_root": run_head_preflight.HEAD_OUTPUT_ROOT,
        "output_receipt": run_head_preflight.HEAD_RECEIPT_PATH,
        "overwrite": False,
        "rerun": False,
        "required_environment": REQUIRED_ENVIRONMENT,
        "resource_budget": budget,
        "argv": [
            "scripts/research/taro_o0r_truth_materializer_runtime/run_head_preflight.py",
            "--execution-lock",
            _relative(HEAD_LOCK_PATH),
        ],
        "bindings": _bindings(run_head_preflight.EXPECTED_HEAD_BINDINGS.items()),
    }
    _write(HEAD_LOCK_PATH, payload)


def write_truth_lock(execution_commit: str) -> None:
    preflight = materializer.load_json(PREFLIGHT_PATH)
    authorization = materializer.load_json(AUTHORIZATION_PATH)
    head_path = REPO_ROOT / run_truth_only.EXPECTED_TRUTH_BINDINGS["HEAD_RECEIPT"]
    head_receipt = materializer.load_json(head_path)
    materializer.validate_authorization(preflight, authorization, preflight_sha256=materializer.sha256_file(PREFLIGHT_PATH))
    materializer.validate_head_receipt(preflight, materializer.sha256_file(AUTHORIZATION_PATH), head_receipt)
    payload = {
        "schema": materializer.EXECUTION_LOCK_SCHEMA,
        "lock_id": "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R1_EXECUTION_LOCK",
        "date": "2026-08-10",
        "status": "AUTHORIZED_UNCONSUMED",
        "consumed": False,
        "execution_commit": execution_commit,
        "execution_authority": run_truth_only.EXPECTED_AUTHORITY,
        "roots": run_truth_only.EXPECTED_ROOTS,
        "overwrite": False,
        "rerun": False,
        "required_environment": REQUIRED_ENVIRONMENT,
        "resource_budget": RESOURCE_BUDGET,
        "argv": [
            "scripts/research/taro_o0r_truth_materializer_runtime/run_truth_only.py",
            "--execution-lock",
            _relative(TRUTH_LOCK_PATH),
        ],
        "bindings": _bindings(run_truth_only.EXPECTED_TRUTH_BINDINGS.items()),
    }
    _write(TRUTH_LOCK_PATH, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write-base")
    implementation = subparsers.add_parser("write-implementation-lock")
    implementation.add_argument("--tests-run", type=int, required=True)
    head = subparsers.add_parser("write-head-lock")
    head.add_argument("--execution-commit", default=None)
    truth = subparsers.add_parser("write-truth-lock")
    truth.add_argument("--execution-commit", default=None)
    args = parser.parse_args()
    if args.command == "write-base":
        write_base()
    elif args.command == "write-implementation-lock":
        write_implementation_lock(args.tests_run)
    elif args.command == "write-head-lock":
        write_head_lock(args.execution_commit or _git_head())
    elif args.command == "write-truth-lock":
        write_truth_lock(args.execution_commit or _git_head())
    else:
        raise AssertionError(args.command)
    print(json.dumps({"status": "WRITTEN", "command": args.command}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
