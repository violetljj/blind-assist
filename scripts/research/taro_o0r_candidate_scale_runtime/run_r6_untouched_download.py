#!/usr/bin/env python3
"""Download and seal the exact HEAD-bound TARO R6 source assets once."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import r6_untouched_cohort as cohort
from scripts.research.taro_o0r_candidate_scale_runtime import run_r6_untouched_head as head_runtime
from scripts.research.taro_o0r_candidate_scale_runtime import validate_r6_untouched_data_lock as data_lock_validator
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o0r.r6_untouched_download_execution_lock.v1"
LOCK_ID = "TARO_O0R_R6_UNTOUCHED_SOURCE_DOWNLOAD_ONE_SHOT_EXECUTION_LOCK"
SOURCE_ROOT = "artifacts.local/datasets/taro/o0r-r6-untouched-source-r0"
EVIDENCE_ROOT = "artifacts.local/evidence/taro/o0r-r6-untouched-source-r0"
EXPECTED_BINDING_PATHS = {
    "R6_DATA_LOCK": "docs/research/taro/TARO_O0R_R6_UNTOUCHED_COHORT_AND_DATA_USE_LOCK_2026-08-11.json",
    "R6_HEAD_RECEIPT": "artifacts.local/evidence/taro/o0r-r6-untouched-head-r0/head-receipt.json",
    "R6_HEAD_RUNNER": "scripts/research/taro_o0r_candidate_scale_runtime/run_r6_untouched_head.py",
    "R6_DATA_LOCK_VALIDATOR": "scripts/research/taro_o0r_candidate_scale_runtime/validate_r6_untouched_data_lock.py",
    "DOWNLOAD_TRANSPORT": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "R6_DOWNLOAD_RUNNER": "scripts/research/taro_o0r_candidate_scale_runtime/run_r6_untouched_download.py",
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


class R6DownloadError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise R6DownloadError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(not path.exists(), "R6_DOWNLOAD_OUTPUT_COLLISION", f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter.canonical_json_bytes(dict(value)) + b"\n"
    partial = path.with_name(path.name + ".partial")
    require(not partial.exists(), "R6_DOWNLOAD_PARTIAL_COLLISION", f"partial already exists: {partial}")
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
        if request["asset"] == "upsampling.zip":
            relative = f"upsampling/Training/{video}.zip"
        elif request["asset"] == "lowres_wide_intrinsics.zip":
            relative = f"raw/Training/{video}/lowres_wide_intrinsics.zip"
        elif request["asset"] == "lowres_wide.traj":
            relative = f"raw/Training/{video}/lowres_wide.traj"
        else:
            raise R6DownloadError("R6_DOWNLOAD_ASSET_INVALID", f"unexpected asset: {request['asset']}")
        rows.append({**dict(request), "relative_path": relative})
    require(len(rows) == 24 and len({row["relative_path"] for row in rows}) == 24, "R6_DOWNLOAD_PLAN_COUNT", "download plan count/paths drift")
    return rows


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID, "R6_DOWNLOAD_LOCK_IDENTITY", "download lock identity drift")
    require(lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R6_DOWNLOAD_NOT_AUTHORIZED", "download lock not authorized/unconsumed")
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "R6_DOWNLOAD_AUTHORITY_DRIFT", "download authority drift")
    require(lock.get("source_root") == SOURCE_ROOT and lock.get("evidence_root") == EVIDENCE_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R6_DOWNLOAD_ROOT_POLICY", "download root policy drift")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv, "R6_DOWNLOAD_ARGV_DRIFT", "download argv drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDING_PATHS), "R6_DOWNLOAD_BINDINGS", "download binding count drift")
    verified = {}
    for row in bindings:
        require(isinstance(row, dict) and set(row) == {"role", "path", "bytes", "sha256"}, "R6_DOWNLOAD_BINDING_ROW", "download binding row drift")
        role = row["role"]
        relative = row["path"]
        require(role not in verified and EXPECTED_BINDING_PATHS.get(role) == relative, "R6_DOWNLOAD_BINDING_PATH", "download binding role/path drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R6_DOWNLOAD_BINDING_HASH", f"download binding hash drift: {relative}")
        verified[role] = row
    require(set(verified) == set(EXPECTED_BINDING_PATHS), "R6_DOWNLOAD_BINDING_SET", "download binding set drift")
    require(data_lock_validator.validate_file(_repo_path(EXPECTED_BINDING_PATHS["R6_DATA_LOCK"]), verify_files=True) == [], "R6_DOWNLOAD_DATA_LOCK_INVALID", "R6 data lock validation failed")
    plan = cohort.build_plan(REPO_ROOT)
    head = json.loads(_repo_path(EXPECTED_BINDING_PATHS["R6_HEAD_RECEIPT"]).read_text(encoding="utf-8"))
    head = head_runtime.validate_head_receipt(plan, head, maximum_attempts=2)
    require(head["passed"] is True and head["terminal"] == "TARO_O0R_R6_UNTOUCHED_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED", "R6_DOWNLOAD_HEAD_NOT_ADMITTED", "HEAD receipt is not an availability PASS")
    budget = lock.get("resource_budget", {})
    require(budget == {"maximum_source_bytes": head["total_content_length_bytes"], "download_timeout_seconds_per_asset": 300, "download_wall_seconds": 3600, "maximum_evidence_bytes": 10485760}, "R6_DOWNLOAD_BUDGET_DRIFT", "download budget drift")
    require(lock.get("request_plan_sha256") == plan["request_plan"]["expanded_requests_sha256"] == head["request_plan_sha256"], "R6_DOWNLOAD_PLAN_DRIFT", "download request plan drift")
    lock["_lock_path"] = lock_path
    lock["_verified_bindings"] = verified
    lock["_plan"] = plan
    lock["_head"] = head
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    source_root = _repo_path(SOURCE_ROOT)
    evidence_root = _repo_path(EVIDENCE_ROOT)
    require(not source_root.exists() and not evidence_root.exists(), "R6_DOWNLOAD_ROOT_COLLISION", "download root already exists")
    evidence_root.mkdir(parents=True, exist_ok=False)
    source_root.mkdir(parents=True, exist_ok=False)
    files = {}
    started = time.monotonic()
    try:
        start = {
            "schema": "blindassist.taro.o0r.r6_untouched_download_start.v1",
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]),
            "head_receipt_sha256": lock["_verified_bindings"]["R6_HEAD_RECEIPT"]["sha256"],
            "expected_source_bytes": lock["_head"]["total_content_length_bytes"],
            "one_shot_consumed_on_root_creation": True,
        }
        files["start-receipt.json"] = _write_exclusive(evidence_root / "start-receipt.json", start)
        download_rows = expanded_download_plan(lock["_plan"])
        head_lookup = {row["url"]: row for row in lock["_head"]["assets"]}
        receipts = []
        for index, row in enumerate(download_rows, start=1):
            require(time.monotonic() - started <= lock["resource_budget"]["download_wall_seconds"], "R6_DOWNLOAD_TIMEOUT", "download wall budget exceeded")
            receipt = materializer.download_bound_asset(
                row,
                head_lookup[row["url"]],
                source_root=source_root,
                timeout_seconds=float(lock["resource_budget"]["download_timeout_seconds_per_asset"]),
            )
            receipt["visit_id"] = row["visit_id"]
            receipt["video_id"] = row["video_id"]
            receipts.append(receipt)
            print(json.dumps({"downloaded": index, "asset_count": len(download_rows), "asset": row["asset"], "video_id": row["video_id"], "bytes": receipt["bytes"]}, sort_keys=True), flush=True)
        total = sum(row["bytes"] for row in receipts)
        require(total == lock["_head"]["total_content_length_bytes"], "R6_DOWNLOAD_TOTAL_DRIFT", "download total differs from HEAD")
        for row in receipts:
            target = materializer.safe_join(source_root, row["relative_path"])
            require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R6_DOWNLOAD_FILE_DRIFT", f"downloaded file differs: {row['relative_path']}")
        files["download-receipts.json"] = _write_exclusive(evidence_root / "download-receipts.json", {"schema": "blindassist.taro.o0r.r6_untouched_download_receipts.v1", "receipts": receipts})
        result = {
            "schema": "blindassist.taro.o0r.r6_untouched_download_result.v1",
            "execution_valid": True,
            "terminal": "TARO_O0R_R6_UNTOUCHED_SOURCE_DOWNLOAD_INTEGRITY_PASS",
            "passed": True,
            "asset_count": len(receipts),
            "source_bytes": total,
            "network_get_requests": len(receipts),
            "archive_decode": False,
            "source_frame_decode": False,
            "model_execution": False,
            "truth_scoring": False,
            "training": False,
            "one_shot_consumed": True,
        }
        files["result.json"] = _write_exclusive(evidence_root / "result.json", result)
        manifest = {"schema": "blindassist.taro.o0r.r6_untouched_download_manifest.v1", "files": files, "one_shot_consumed": True}
        _write_exclusive(evidence_root / "manifest.json", manifest)
        return result
    except Exception as error:
        failure = {"schema": "blindassist.taro.o0r.r6_untouched_download_failure.v1", "execution_valid": False, "terminal": "TARO_O0R_R6_UNTOUCHED_SOURCE_DOWNLOAD_EXECUTION_INVALID_ONE_SHOT_CONSUMED", "failure_code": getattr(error, "code", type(error).__name__), "message": str(error), "one_shot_consumed": True}
        try:
            _write_exclusive(evidence_root / "failure.json", failure)
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
        print(json.dumps({"execution_valid": False, "error_code": getattr(error, "code", type(error).__name__), "message": str(error), "one_shot_consumed": _repo_path(EVIDENCE_ROOT).exists()}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
