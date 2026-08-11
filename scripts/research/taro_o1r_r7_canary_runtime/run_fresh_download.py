#!/usr/bin/env python3
"""Download and seal the exact HEAD-bound TARO R7 fresh assets once."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import fresh_confirmation_cohort as cohort
from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_head as head_runtime
from scripts.research.taro_o1r_r7_canary_runtime import validate_fresh_data_lock


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r7_fresh_confirmation_download_execution_lock.v1"
LOCK_ID = "TARO_O1R_R7_FRESH_CONFIRMATION_SOURCE_DOWNLOAD_ONE_SHOT_EXECUTION_LOCK"
SOURCE_ROOT = "artifacts.local/datasets/taro/o1r-r7-fresh-confirmation-source-r0"
EVIDENCE_ROOT = "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-source-r0"
PASS_TERMINAL = "TARO_O1R_R7_FRESH_SOURCE_DOWNLOAD_INTEGRITY_PASS"
INVALID_TERMINAL = "TARO_O1R_R7_FRESH_SOURCE_DOWNLOAD_EXECUTION_INVALID_ONE_SHOT_CONSUMED"
EXPECTED_BINDING_PATHS = {
    "R7_DATA_LOCK": "docs/research/taro/TARO_O1R_R7_FRESH_CONFIRMATION_COHORT_AND_DATA_USE_LOCK_2026-08-12.json",
    "R7_HEAD_RECEIPT": "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-head-r0/head-receipt.json",
    "R7_HEAD_RUNNER": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_head.py",
    "R7_DATA_LOCK_VALIDATOR": "scripts/research/taro_o1r_r7_canary_runtime/validate_fresh_data_lock.py",
    "DOWNLOAD_TRANSPORT": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "R7_DOWNLOAD_RUNNER": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_download.py",
}
EXPECTED_AUTHORITY = {
    "head_reuse": True,
    "source_download": True,
    "source_integrity_validation": True,
    "archive_decode": False,
    "source_frame_decode": False,
    "model_execution": False,
    "truth_scoring": False,
    "training": False,
    "device": False,
    "product": False,
    "safety": False,
}


class FreshDownloadError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise FreshDownloadError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(not path.exists(), "R7_DOWNLOAD_OUTPUT_COLLISION", f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter.canonical_json_bytes(dict(value)) + b"\n"
    partial = path.with_name(path.name + ".partial")
    require(not partial.exists(), "R7_DOWNLOAD_PARTIAL_COLLISION", f"partial already exists: {partial}")
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
        require(relative is not None, "R7_DOWNLOAD_ASSET_INVALID", f"unexpected asset: {request['asset']}")
        rows.append({**dict(request), "relative_path": relative})
    require(len(rows) == 24 and len({row["relative_path"] for row in rows}) == 24, "R7_DOWNLOAD_PLAN_COUNT", "R7 download plan count/path drift")
    return rows


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID, "R7_DOWNLOAD_LOCK_IDENTITY", "R7 download lock identity drift")
    require(lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R7_DOWNLOAD_NOT_AUTHORIZED", "R7 download lock is not authorized/unconsumed")
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "R7_DOWNLOAD_AUTHORITY_DRIFT", "R7 download authority drift")
    require(lock.get("source_root") == SOURCE_ROOT and lock.get("evidence_root") == EVIDENCE_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R7_DOWNLOAD_ROOT_POLICY", "R7 download root policy drift")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv, "R7_DOWNLOAD_ARGV_DRIFT", "R7 download argv drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDING_PATHS), "R7_DOWNLOAD_BINDINGS", "R7 download binding count drift")
    verified = {}
    for row in bindings:
        require(isinstance(row, dict) and set(row) == {"role", "path", "bytes", "sha256"}, "R7_DOWNLOAD_BINDING_ROW", "R7 download binding row drift")
        role, relative = row["role"], row["path"]
        require(role not in verified and EXPECTED_BINDING_PATHS.get(role) == relative, "R7_DOWNLOAD_BINDING_PATH", "R7 download binding role/path drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R7_DOWNLOAD_BINDING_HASH", f"R7 download binding hash drift: {relative}")
        verified[role] = row
    require(set(verified) == set(EXPECTED_BINDING_PATHS), "R7_DOWNLOAD_BINDING_SET", "R7 download binding set drift")
    require(validate_fresh_data_lock.validate(_repo_path(EXPECTED_BINDING_PATHS["R7_DATA_LOCK"]))["passed"], "R7_DOWNLOAD_DATA_LOCK_INVALID", "R7 data lock validation failed")
    plan = cohort.build_plan(REPO_ROOT)
    head = json.loads(_repo_path(EXPECTED_BINDING_PATHS["R7_HEAD_RECEIPT"]).read_text(encoding="utf-8"))
    head = head_runtime.validate_head_receipt(plan, head, maximum_attempts=2)
    require(head["passed"] is True and head["terminal"] == head_runtime.PASS_TERMINAL, "R7_DOWNLOAD_HEAD_NOT_ADMITTED", "R7 HEAD receipt is not an availability PASS")
    budget = lock.get("resource_budget", {})
    require(budget == {"maximum_source_bytes": head["total_content_length_bytes"], "download_timeout_seconds_per_asset": 300, "download_wall_seconds": 3600, "maximum_evidence_bytes": 10485760}, "R7_DOWNLOAD_BUDGET_DRIFT", "R7 download budget drift")
    require(lock.get("request_plan_sha256") == plan["request_plan"]["expanded_requests_sha256"] == head["request_plan_sha256"], "R7_DOWNLOAD_PLAN_DRIFT", "R7 download plan drift")
    lock["_lock_path"], lock["_verified_bindings"], lock["_plan"], lock["_head"] = lock_path, verified, plan, head
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    source_root, evidence_root = _repo_path(SOURCE_ROOT), _repo_path(EVIDENCE_ROOT)
    require(not source_root.exists() and not evidence_root.exists(), "R7_DOWNLOAD_ROOT_COLLISION", "R7 download root already exists")
    evidence_root.mkdir(parents=True, exist_ok=False)
    source_root.mkdir(parents=True, exist_ok=False)
    files = {}
    started = time.monotonic()
    try:
        start = {"schema": "blindassist.taro.o1r.r7_fresh_download_start.v1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "head_receipt_sha256": lock["_verified_bindings"]["R7_HEAD_RECEIPT"]["sha256"], "expected_source_bytes": lock["_head"]["total_content_length_bytes"], "one_shot_consumed_on_root_creation": True}
        files["start-receipt.json"] = _write_exclusive(evidence_root / "start-receipt.json", start)
        rows = expanded_download_plan(lock["_plan"])
        head_lookup = {row["url"]: row for row in lock["_head"]["assets"]}
        receipts = []
        for index, row in enumerate(rows, 1):
            require(time.monotonic() - started <= lock["resource_budget"]["download_wall_seconds"], "R7_DOWNLOAD_TIMEOUT", "R7 download wall budget exceeded")
            receipt = materializer.download_bound_asset(row, head_lookup[row["url"]], source_root=source_root, timeout_seconds=float(lock["resource_budget"]["download_timeout_seconds_per_asset"]))
            receipt["visit_id"], receipt["video_id"] = row["visit_id"], row["video_id"]
            receipts.append(receipt)
            print(json.dumps({"downloaded": index, "asset_count": len(rows), "asset": row["asset"], "video_id": row["video_id"], "bytes": receipt["bytes"]}, sort_keys=True), flush=True)
        total = sum(row["bytes"] for row in receipts)
        require(total == lock["_head"]["total_content_length_bytes"], "R7_DOWNLOAD_TOTAL_DRIFT", "R7 download total differs from HEAD")
        for row in receipts:
            target = materializer.safe_join(source_root, row["relative_path"])
            require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R7_DOWNLOAD_FILE_DRIFT", f"R7 downloaded file differs: {row['relative_path']}")
        files["download-receipts.json"] = _write_exclusive(evidence_root / "download-receipts.json", {"schema": "blindassist.taro.o1r.r7_fresh_download_receipts.v1", "receipts": receipts})
        result = {"schema": "blindassist.taro.o1r.r7_fresh_download_result.v1", "execution_valid": True, "terminal": PASS_TERMINAL, "passed": True, "asset_count": len(receipts), "source_bytes": total, "network_get_requests": len(receipts), "archive_decode": False, "source_frame_decode": False, "model_execution": False, "truth_scoring": False, "training": False, "one_shot_consumed": True}
        files["result.json"] = _write_exclusive(evidence_root / "result.json", result)
        _write_exclusive(evidence_root / "manifest.json", {"schema": "blindassist.taro.o1r.r7_fresh_download_manifest.v1", "files": files, "one_shot_consumed": True})
        return result
    except Exception as error:
        failure = {"schema": "blindassist.taro.o1r.r7_fresh_download_failure.v1", "execution_valid": False, "terminal": INVALID_TERMINAL, "failure_code": getattr(error, "code", type(error).__name__), "message": str(error), "one_shot_consumed": True}
        try:
            _write_exclusive(evidence_root / "failure.json", failure)
        except Exception:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        print(json.dumps({"execution_valid": False, "error_code": getattr(error, "code", type(error).__name__), "message": str(error), "one_shot_consumed": _repo_path(EVIDENCE_ROOT).exists()}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
