#!/usr/bin/env python3
"""Download the exact R10 32-parent pool after an admitted zero-body HEAD."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r10_clear_runtime import fresh_pool
from scripts.research.taro_o1r_r10_clear_runtime import run_pool_head


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r10_fresh_pool_download_execution_lock.v1"
LOCK_ID = "TARO_O1R_R10_FRESH_POOL_SOURCE_DOWNLOAD_ONE_SHOT_EXECUTION_LOCK"
SOURCE_ROOT = "artifacts.local/datasets/taro/o1r-r10-fresh-pool-source-r0"
EVIDENCE_ROOT = "artifacts.local/evidence/taro/o1r-r10-fresh-pool-source-r0"
PASS_TERMINAL = "TARO_O1R_R10_FRESH_POOL_SOURCE_DOWNLOAD_INTEGRITY_PASS"
INVALID_TERMINAL = "TARO_O1R_R10_FRESH_POOL_SOURCE_DOWNLOAD_EXECUTION_INVALID"
ASSET_COUNT = 96
AUTHORITY_SCOPE = "Exact frozen R10 32-parent Training pool and 96-URL plan: zero-body HEAD, bounded source download, source-only Phase A and selector ranking, then FARO only for the sealed top eight; no training, deployment, product, or safety authority."
EXPECTED_BINDINGS = {
    "R10_PROTOCOL": "docs/research/taro/TARO_O1R_R10_FRESH_PARENT_SOURCE_ONLY_CLEAR_ENRICHED_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json",
    "R10_POOL_PLANNER": "scripts/research/taro_o1r_r10_clear_runtime/fresh_pool.py",
    "R10_HEAD_RECEIPT": f"{run_pool_head.OUTPUT_ROOT}/head-receipt.json",
    "R10_HEAD_RESULT": f"{run_pool_head.OUTPUT_ROOT}/result.json",
    "R10_HEAD_MANIFEST": f"{run_pool_head.OUTPUT_ROOT}/manifest.json",
    "DOWNLOAD_TRANSPORT": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "R10_DOWNLOAD_RUNNER": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_download.py",
    "R10_DOWNLOAD_TEST": "scripts/research/taro_o1r_r10_clear_runtime/test_run_pool_download.py",
}
EXPECTED_AUTHORITY = {
    "head_reuse": True,
    "source_download": True,
    "source_integrity_validation": True,
    "archive_decode": False,
    "source_frame_decode": False,
    "model_execution": False,
    "faro_read": False,
    "truth_scoring": False,
    "training": False,
    "device": False,
    "product": False,
    "safety": False,
}


class PoolDownloadError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PoolDownloadError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(not path.exists(), "R10_DOWNLOAD_OUTPUT_COLLISION", f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter.canonical_json_bytes(dict(value)) + b"\n"
    partial = path.with_name(path.name + ".partial")
    require(not partial.exists(), "R10_DOWNLOAD_PARTIAL_COLLISION", f"partial exists: {partial}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": len(payload), "sha256": materializer.sha256_bytes(payload)}


def expanded_download_plan(plan: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for request in plan["request_plan"]["requests"]:
        video = request["video_id"]
        relative = {
            "upsampling.zip": f"upsampling/Training/{video}.zip",
            "lowres_wide_intrinsics.zip": f"raw/Training/{video}/lowres_wide_intrinsics.zip",
            "lowres_wide.traj": f"raw/Training/{video}/lowres_wide.traj",
        }.get(request["asset"])
        require(relative is not None, "R10_DOWNLOAD_ASSET_INVALID", f"unexpected asset: {request['asset']}")
        rows.append({**dict(request), "relative_path": relative})
    require(len(rows) == ASSET_COUNT and len({row["relative_path"] for row in rows}) == ASSET_COUNT, "R10_DOWNLOAD_PLAN_COUNT", "download plan count/path drift")
    return rows


def _verify_head_manifest() -> None:
    manifest_path = _repo_path(EXPECTED_BINDINGS["R10_HEAD_MANIFEST"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_head_manifest.v1" and manifest.get("one_shot_consumed") is True, "R10_DOWNLOAD_HEAD_MANIFEST", "R10 HEAD manifest identity drift")
    files = manifest.get("files")
    require(isinstance(files, dict) and set(files) == {"start-receipt.json", "head-receipt.json", "result.json"}, "R10_DOWNLOAD_HEAD_MANIFEST", "R10 HEAD manifest file set drift")
    root = _repo_path(run_pool_head.OUTPUT_ROOT)
    for relative, receipt in files.items():
        target = materializer.safe_join(root, relative)
        require(target.is_file() and target.stat().st_size == receipt.get("bytes") and materializer.sha256_file(target) == receipt.get("sha256"), "R10_DOWNLOAD_HEAD_FILE", f"R10 HEAD artifact drift: {relative}")


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R10_DOWNLOAD_LOCK_IDENTITY", "download lock identity/authority drift")
    authority = lock.get("user_authority")
    require(isinstance(authority, dict) and authority.get("confirmed_by") == "user" and authority.get("scope") == AUTHORITY_SCOPE, "R10_DOWNLOAD_USER_AUTHORITY", "download user authority drift")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv and lock.get("source_root") == SOURCE_ROOT and lock.get("evidence_root") == EVIDENCE_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R10_DOWNLOAD_LOCK_POLICY", "download argv/root policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R10_DOWNLOAD_BINDINGS", "download binding count drift")
    verified = {}
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in verified and EXPECTED_BINDINGS.get(role) == relative, "R10_DOWNLOAD_BINDING_ROW", "download binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R10_DOWNLOAD_BINDING_HASH", f"download binding drift: {relative}")
        verified[role] = row
    plan = fresh_pool.build_pool(REPO_ROOT)
    head = run_pool_head.validate_head_receipt(plan, json.loads(_repo_path(EXPECTED_BINDINGS["R10_HEAD_RECEIPT"]).read_text(encoding="utf-8")), maximum_attempts=2)
    result = json.loads(_repo_path(EXPECTED_BINDINGS["R10_HEAD_RESULT"]).read_text(encoding="utf-8"))
    require(head["passed"] is True and head["terminal"] == run_pool_head.PASS_TERMINAL and head["asset_count"] == ASSET_COUNT, "R10_DOWNLOAD_HEAD_NOT_ADMITTED", "HEAD receipt not admitted")
    require(result.get("execution_valid") is True and result.get("passed") is True and result.get("terminal") == run_pool_head.PASS_TERMINAL and result.get("total_content_length_bytes") == head["total_content_length_bytes"], "R10_DOWNLOAD_HEAD_RESULT", "HEAD result not admitted")
    _verify_head_manifest()
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "R10_DOWNLOAD_AUTHORITY_DRIFT", "download authority drift")
    require(lock.get("resource_budget") == {"maximum_source_bytes": head["total_content_length_bytes"], "download_timeout_seconds_per_asset": 300, "download_wall_seconds": 10800, "maximum_evidence_bytes": 67108864}, "R10_DOWNLOAD_BUDGET_DRIFT", "download resource budget drift")
    require(lock.get("request_plan_sha256") == plan["request_plan"]["expanded_requests_sha256"] == head["request_plan_sha256"], "R10_DOWNLOAD_PLAN_DRIFT", "download request-plan drift")
    require(not _repo_path(SOURCE_ROOT).exists() and not _repo_path(EVIDENCE_ROOT).exists(), "R10_DOWNLOAD_ROOT_COLLISION", "download source/evidence root exists")
    lock["_lock_path"], lock["_verified_bindings"], lock["_plan"], lock["_head"] = lock_path, verified, plan, head
    return lock


def execute(lock_path: Path, *, download_fn: Callable[..., dict[str, Any]] = materializer.download_bound_asset) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    source_root, evidence_root = _repo_path(SOURCE_ROOT), _repo_path(EVIDENCE_ROOT)
    evidence_root.mkdir(parents=True, exist_ok=False)
    source_root.mkdir(parents=True, exist_ok=False)
    files = {}
    started = time.monotonic()
    try:
        files["start-receipt.json"] = _write_exclusive(evidence_root / "start-receipt.json", {"schema": "blindassist.taro.o1r.r10_fresh_pool_download_start.v1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "head_receipt_sha256": lock["_verified_bindings"]["R10_HEAD_RECEIPT"]["sha256"], "expected_source_bytes": lock["_head"]["total_content_length_bytes"], "one_shot_consumed_on_root_creation": True})
        rows = expanded_download_plan(lock["_plan"])
        head_lookup = {row["url"]: row for row in lock["_head"]["assets"]}
        receipts = []
        for index, row in enumerate(rows, 1):
            require(time.monotonic() - started <= lock["resource_budget"]["download_wall_seconds"], "R10_DOWNLOAD_TIMEOUT", "download wall budget exceeded")
            receipt = download_fn(row, head_lookup[row["url"]], source_root=source_root, timeout_seconds=float(lock["resource_budget"]["download_timeout_seconds_per_asset"]))
            receipt["visit_id"], receipt["video_id"] = row["visit_id"], row["video_id"]
            receipts.append(receipt)
            print(json.dumps({"downloaded": index, "asset_count": ASSET_COUNT, "asset": row["asset"], "video_id": row["video_id"], "bytes": receipt["bytes"]}, sort_keys=True), flush=True)
        total = sum(row["bytes"] for row in receipts)
        require(total == lock["_head"]["total_content_length_bytes"], "R10_DOWNLOAD_TOTAL_DRIFT", "download total differs from HEAD")
        for row in receipts:
            target = materializer.safe_join(source_root, row["relative_path"])
            require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R10_DOWNLOAD_FILE_DRIFT", f"downloaded file drift: {row['relative_path']}")
        files["download-receipts.json"] = _write_exclusive(evidence_root / "download-receipts.json", {"schema": "blindassist.taro.o1r.r10_fresh_pool_download_receipts.v1", "receipts": receipts})
        result = {"schema": "blindassist.taro.o1r.r10_fresh_pool_download_result.v1", "execution_valid": True, "terminal": PASS_TERMINAL, "passed": True, "asset_count": ASSET_COUNT, "source_bytes": total, "network_get_requests": ASSET_COUNT, "archive_decode": False, "source_frame_decode": False, "model_execution": False, "faro_read": False, "truth_scoring": False, "training": False, "one_shot_consumed": True}
        files["result.json"] = _write_exclusive(evidence_root / "result.json", result)
        _write_exclusive(evidence_root / "manifest.json", {"schema": "blindassist.taro.o1r.r10_fresh_pool_download_manifest.v1", "files": files, "one_shot_consumed": True})
        return result
    except Exception as error:
        try:
            _write_exclusive(evidence_root / "failure.json", {"schema": "blindassist.taro.o1r.r10_fresh_pool_download_failure.v1", "execution_valid": False, "terminal": INVALID_TERMINAL, "failure_code": getattr(error, "code", type(error).__name__), "message": str(error), "one_shot_consumed": True})
        except Exception:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        print(json.dumps({"execution_valid": False, "terminal": INVALID_TERMINAL, "error_code": getattr(error, "code", type(error).__name__), "message": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
