#!/usr/bin/env python3
"""Download the exact TARO R11 source pool after the admitted HEAD one-shot."""

from __future__ import annotations

import argparse
import binascii
import datetime as dt
import json
import os
import re
import subprocess
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r11_abstention_runtime import fresh_pool
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_head


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_download_execution_lock_attempt_02.v1"
LOCK_ID = "TARO_O1R_R11_FRESH_48_PARENT_BOUNDED_SOURCE_DOWNLOAD_ONE_SHOT_EXECUTION_LOCK_ATTEMPT_02"
LOCK_RELATIVE = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_BOUNDED_SOURCE_DOWNLOAD_ONE_SHOT_EXECUTION_LOCK_ATTEMPT_02_2026-08-12.json"
EXPECTED_ARGV = [
    "-m",
    "scripts.research.taro_o1r_r11_abstention_runtime.run_pool_download",
    "--execution-lock",
    LOCK_RELATIVE,
]
SOURCE_ROOT = "artifacts.local/datasets/taro/o1r-r11-fresh-pool-source-r0"
EVIDENCE_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-source-r0"
PASS_TERMINAL = "TARO_O1R_R11_FRESH_POOL_SOURCE_DOWNLOAD_INTEGRITY_PASS"
INVALID_TERMINAL = "TARO_O1R_R11_FRESH_POOL_SOURCE_DOWNLOAD_EXECUTION_INVALID"
ASSET_COUNT = run_pool_head.ASSET_COUNT
HEAD_TOTAL_BYTES = 2960390828
HEAD_FORMAL_RESULT = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_ZERO_BODY_HEAD_RESULT_2026-08-12.json"
EXPECTED_BINDINGS = {
    "R11_PROTOCOL": run_pool_head.PROTOCOL_RELATIVE,
    "R11_DATA_USE_AUTHORIZATION": run_pool_head.AUTHORIZATION_RELATIVE,
    "R11_POOL_PLANNER": "scripts/research/taro_o1r_r11_abstention_runtime/fresh_pool.py",
    "R11_HEAD_RUNNER": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_head.py",
    "R11_HEAD_RECEIPT": f"{run_pool_head.OUTPUT_ROOT}/head-receipt.json",
    "R11_HEAD_RESULT": f"{run_pool_head.OUTPUT_ROOT}/result.json",
    "R11_HEAD_MANIFEST": f"{run_pool_head.OUTPUT_ROOT}/manifest.json",
    "R11_HEAD_FORMAL_RESULT": HEAD_FORMAL_RESULT,
    "DOWNLOAD_TRANSPORT": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "R11_DOWNLOAD_RUNNER": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_download.py",
    "R11_DOWNLOAD_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_run_pool_download.py",
}
ARTIFACT_BINDING_ROLES = {"R11_HEAD_RECEIPT", "R11_HEAD_RESULT", "R11_HEAD_MANIFEST"}
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
    "deployment": False,
    "product": False,
    "safety": False,
}
EXPECTED_BUDGET = {
    "maximum_source_bytes": HEAD_TOTAL_BYTES,
    "download_timeout_seconds_per_asset": 300,
    "download_wall_seconds": 14400,
    "maximum_attempts_per_asset": 3,
    "maximum_get_attempts": ASSET_COUNT * 3,
    "maximum_evidence_bytes": 67108864,
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "R11_DOWNLOAD_JSON_OBJECT", f"JSON object required: {path}")
    return value


def _validate_content_seal(value: Mapping[str, Any], code: str) -> dict[str, Any]:
    record = json.loads(json.dumps(dict(value)))
    claimed = record.pop("content_sha256", None)
    require(isinstance(claimed, str) and claimed == adapter.canonical_sha256(record), code, "content seal drift")
    record["content_sha256"] = claimed
    return record


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(not path.exists(), "R11_DOWNLOAD_OUTPUT_COLLISION", f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_line(value)
    partial = path.with_name(path.name + ".partial")
    require(not partial.exists(), "R11_DOWNLOAD_PARTIAL_COLLISION", f"partial exists: {partial}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": len(payload), "sha256": materializer.sha256_bytes(payload)}


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    return adapter.canonical_json_bytes(dict(value)) + b"\n"


def _projected_evidence_bytes(files: Mapping[str, Mapping[str, Any]], final_record: Mapping[str, Any]) -> int:
    return sum(int(row["bytes"]) for row in files.values()) + len(_canonical_json_line(final_record))


def _sealed_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    require("content_sha256" not in record, "R11_DOWNLOAD_INTERNAL", "record is already sealed")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _write_failure_terminal(evidence_root: Path, error: BaseException, files: dict[str, dict[str, Any]]) -> None:
    failure = _sealed_record(
        {
            "schema": "blindassist.taro.o1r.r11_fresh_pool_download_failure.v1",
            "execution_valid": False,
            "terminal": INVALID_TERMINAL,
            "failure_code": str(getattr(error, "code", type(error).__name__)),
            "message": str(error),
            "one_shot_consumed": True,
        }
    )
    files["failure.json"] = _write_exclusive(evidence_root / "failure.json", failure)
    failure_manifest = _sealed_record(
        {
            "schema": "blindassist.taro.o1r.r11_fresh_pool_download_failure_manifest.v1",
            "terminal": INVALID_TERMINAL,
            "files": files,
            "one_shot_consumed": True,
        }
    )
    _write_exclusive(evidence_root / "manifest.json", failure_manifest)


def _reserve_execution_roots(
    source_root: Path,
    evidence_root: Path,
    *,
    execution_lock_sha256: str,
    head_receipt_sha256: str,
    files: dict[str, dict[str, Any]],
) -> None:
    evidence_root.mkdir(parents=True, exist_ok=False)
    try:
        files["start-receipt.json"] = _write_exclusive(
            evidence_root / "start-receipt.json",
            _sealed_record(
                {
                    "schema": "blindassist.taro.o1r.r11_fresh_pool_download_start.v1",
                    "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "execution_lock_sha256": execution_lock_sha256,
                    "head_receipt_sha256": head_receipt_sha256,
                    "expected_source_bytes": HEAD_TOTAL_BYTES,
                    "one_shot_consumed_on_root_creation": True,
                }
            ),
        )
        source_root.mkdir(parents=True, exist_ok=False)
    except Exception as error:
        _write_failure_terminal(evidence_root, error, files)
        raise


def expanded_download_plan(plan: Mapping[str, Any]) -> list[dict[str, str]]:
    frozen = fresh_pool.validate_pool(plan, repo_root=REPO_ROOT, recompute=False)
    rows: list[dict[str, str]] = []
    for request in frozen["request_plan"]["requests"]:
        video = request["video_id"]
        relative = {
            "upsampling.zip": f"upsampling/Training/{video}.zip",
            "lowres_wide_intrinsics.zip": f"raw/Training/{video}/lowres_wide_intrinsics.zip",
            "lowres_wide.traj": f"raw/Training/{video}/lowres_wide.traj",
        }.get(request["asset"])
        require(relative is not None, "R11_DOWNLOAD_ASSET_INVALID", f"unexpected asset: {request['asset']}")
        rows.append({**dict(request), "relative_path": relative})
    require(len(rows) == ASSET_COUNT and len({row["relative_path"] for row in rows}) == ASSET_COUNT, "R11_DOWNLOAD_PLAN_COUNT", "download plan count/path drift")
    return rows


def _verify_head_manifest() -> dict[str, Any]:
    manifest_path = _repo_path(EXPECTED_BINDINGS["R11_HEAD_MANIFEST"])
    manifest = _read_json(manifest_path)
    require(manifest.get("schema") == "blindassist.taro.o1r.r11_fresh_pool_head_manifest.v1" and manifest.get("one_shot_consumed") is True, "R11_DOWNLOAD_HEAD_MANIFEST", "R11 HEAD manifest identity drift")
    files = manifest.get("files")
    require(isinstance(files, dict) and set(files) == {"start-receipt.json", "head-receipt.json", "result.json"}, "R11_DOWNLOAD_HEAD_MANIFEST", "R11 HEAD manifest file set drift")
    root = _repo_path(run_pool_head.OUTPUT_ROOT)
    for relative, row in files.items():
        target = materializer.safe_join(root, relative)
        require(target.is_file() and target.stat().st_size == row.get("bytes") and materializer.sha256_file(target) == row.get("sha256"), "R11_DOWNLOAD_HEAD_FILE", f"R11 HEAD artifact drift: {relative}")
    return manifest


def validate_head_admission(plan: Mapping[str, Any]) -> dict[str, Any]:
    run_pool_head.validate_execution_lock(_repo_path(run_pool_head.LOCK_RELATIVE), require_output_absent=False)
    head = run_pool_head.validate_head_receipt(
        plan,
        _read_json(_repo_path(EXPECTED_BINDINGS["R11_HEAD_RECEIPT"])),
        maximum_attempts=run_pool_head.EXPECTED_BUDGET["maximum_attempts_per_request"],
    )
    result = _read_json(_repo_path(EXPECTED_BINDINGS["R11_HEAD_RESULT"]))
    require(
        head["passed"] is True
        and head["terminal"] == run_pool_head.PASS_TERMINAL
        and head["asset_count"] == ASSET_COUNT
        and head["available_asset_count"] == ASSET_COUNT
        and head["request_attempt_count"] == ASSET_COUNT
        and head["total_content_length_bytes"] == HEAD_TOTAL_BYTES,
        "R11_DOWNLOAD_HEAD_NOT_ADMITTED",
        "R11 HEAD receipt not admitted",
    )
    require(
        result.get("execution_valid") is True
        and result.get("passed") is True
        and result.get("terminal") == run_pool_head.PASS_TERMINAL
        and result.get("asset_count") == ASSET_COUNT
        and result.get("available_asset_count") == ASSET_COUNT
        and result.get("total_content_length_bytes") == HEAD_TOTAL_BYTES
        and result.get("response_body_bytes_read") == 0,
        "R11_DOWNLOAD_HEAD_RESULT",
        "R11 HEAD result not admitted",
    )
    _verify_head_manifest()
    formal = _validate_content_seal(_read_json(_repo_path(HEAD_FORMAL_RESULT)), "R11_DOWNLOAD_HEAD_FORMAL_RESULT")
    require(formal.get("passed") is True and formal.get("one_shot_consumed") is True and formal.get("content_length", {}).get("total_bytes") == HEAD_TOTAL_BYTES and formal.get("phase_firewall", {}).get("response_body_bytes_read") == 0, "R11_DOWNLOAD_HEAD_FORMAL_RESULT", "R11 HEAD formal result drift")
    return head


def _git_bytes(commit: str, relative: str) -> bytes:
    completed = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(completed.returncode == 0, "R11_DOWNLOAD_IMPLEMENTATION_COMMIT", f"binding absent from implementation commit: {relative}")
    return completed.stdout


def _validate_implementation_ancestor(commit: str) -> None:
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "R11_DOWNLOAD_IMPLEMENTATION_COMMIT", "implementation commit must be lowercase full SHA")
    completed = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO_ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(completed.returncode == 0, "R11_DOWNLOAD_IMPLEMENTATION_COMMIT", "implementation commit is not an ancestor of HEAD")


def validate_execution_lock(path: Path, *, require_roots_absent: bool = True) -> dict[str, Any]:
    lock_path = path.resolve()
    require(lock_path == _repo_path(LOCK_RELATIVE), "R11_DOWNLOAD_LOCK_PATH", "R11 download lock path drift")
    lock = _validate_content_seal(_read_json(lock_path), "R11_DOWNLOAD_LOCK_HASH")
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R11_DOWNLOAD_LOCK_IDENTITY", "R11 download lock identity drift")
    require(lock.get("argv") == EXPECTED_ARGV and lock.get("source_root") == SOURCE_ROOT and lock.get("evidence_root") == EVIDENCE_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R11_DOWNLOAD_LOCK_POLICY", "R11 download lock policy drift")
    implementation_commit = lock.get("implementation_commit")
    _validate_implementation_ancestor(implementation_commit)
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R11_DOWNLOAD_BINDINGS", "R11 download binding count drift")
    verified: dict[str, dict[str, Any]] = {}
    for row in bindings:
        role = row.get("role")
        relative = row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in verified and EXPECTED_BINDINGS.get(role) == relative, "R11_DOWNLOAD_BINDING_ROW", "R11 download binding row drift")
        target = _repo_path(relative)
        require(target.is_file(), "R11_DOWNLOAD_BINDING_HASH", f"R11 download binding absent: {relative}")
        payload = target.read_bytes()
        require(len(payload) == row["bytes"] and materializer.sha256_bytes(payload) == row["sha256"], "R11_DOWNLOAD_BINDING_HASH", f"R11 download binding drift: {relative}")
        if role not in ARTIFACT_BINDING_ROLES:
            require(payload == _git_bytes(implementation_commit, relative), "R11_DOWNLOAD_BINDING_HASH", f"R11 download implementation-commit drift: {relative}")
        verified[role] = dict(row)
    plan = fresh_pool.build_pool(REPO_ROOT)
    head = validate_head_admission(plan)
    run_pool_head.validate_authorization_receipt(_read_json(_repo_path(run_pool_head.AUTHORIZATION_RELATIVE)))
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY and lock.get("resource_budget") == EXPECTED_BUDGET, "R11_DOWNLOAD_AUTHORITY_BUDGET", "R11 download authority/budget drift")
    require(lock.get("request_plan_sha256") == plan["request_plan"]["expanded_requests_sha256"] == head["request_plan_sha256"], "R11_DOWNLOAD_PLAN_DRIFT", "R11 download request plan drift")
    user = lock.get("user_authority")
    require(isinstance(user, dict) and user.get("confirmed_by") == "user" and user.get("confirmed_at") == "2026-08-12" and user.get("confirmation_verbatim") == "授权" and user.get("scope") == run_pool_head.EXPECTED_USER_SCOPE, "R11_DOWNLOAD_USER_AUTHORITY", "R11 download user authority drift")
    require(lock.get("authorization_receipt_content_sha256") == "CF7814D52532FAB6A5EE8A4CA8EA29E9A7EF1017E075CF8FE597EEBE0834FF5F", "R11_DOWNLOAD_USER_AUTHORITY", "R11 download authorization receipt drift")
    if require_roots_absent:
        require(not _repo_path(SOURCE_ROOT).exists() and not _repo_path(EVIDENCE_ROOT).exists(), "R11_DOWNLOAD_ROOT_COLLISION", "R11 download source/evidence root exists")
    lock["_lock_path"] = lock_path
    lock["_verified_bindings"] = verified
    lock["_plan"] = plan
    lock["_head"] = head
    return lock


def _crc32_file(path: Path) -> str:
    value = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            value = binascii.crc32(block, value)
    return f"{value & 0xFFFFFFFF:08X}"


def _is_transient_transport_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 425, 429, 500, 502, 503, 504}
    return isinstance(error, (TimeoutError, ConnectionError, urllib.error.URLError))


def download_with_transient_retries(
    row: Mapping[str, str],
    head: Mapping[str, Any],
    *,
    source_root: Path,
    timeout_seconds: float,
    maximum_attempts: int,
    download_fn: Callable[..., dict[str, Any]],
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    require(1 <= maximum_attempts <= 3, "R11_DOWNLOAD_RETRY_BUDGET", "R11 download retry budget drift")
    require(timeout_seconds > 0.0, "R11_DOWNLOAD_ASSET_TIMEOUT", "R11 download asset wall budget is exhausted")
    deadline = monotonic_fn() + timeout_seconds
    prior_errors: list[str] = []
    for attempt in range(1, maximum_attempts + 1):
        try:
            remaining_seconds = deadline - monotonic_fn()
            require(remaining_seconds > 0.0, "R11_DOWNLOAD_ASSET_TIMEOUT", "R11 download asset wall budget is exhausted")
            receipt = dict(download_fn(row, head, source_root=source_root, timeout_seconds=remaining_seconds))
            require(monotonic_fn() <= deadline, "R11_DOWNLOAD_ASSET_TIMEOUT", "R11 download asset wall budget was exceeded")
            receipt["attempt_count"] = attempt
            receipt["prior_transport_errors"] = prior_errors
            return receipt
        except Exception as error:
            if not _is_transient_transport_error(error) or attempt == maximum_attempts:
                raise
            prior_errors.append(str(getattr(error, "code", type(error).__name__)))
    raise PoolDownloadError("R11_DOWNLOAD_INTERNAL", "R11 download retry loop empty")


def validate_download_receipt(row: Mapping[str, str], head: Mapping[str, Any], value: Mapping[str, Any], *, source_root: Path) -> dict[str, Any]:
    receipt = _validate_content_seal(value, "R11_DOWNLOAD_RECEIPT_HASH")
    require(receipt.get("schema") == "blindassist.taro.o1r.r11_source_asset_download_receipt.v1", "R11_DOWNLOAD_RECEIPT_SCHEMA", "R11 download receipt schema drift")
    require(all(receipt.get(field) == row[field] for field in ("visit_id", "video_id", "asset", "url", "relative_path")), "R11_DOWNLOAD_RECEIPT_IDENTITY", "R11 download receipt identity drift")
    require(
        receipt.get("bytes") == head.get("content_length_bytes") == receipt.get("head_content_length_bytes")
        and receipt.get("head_etag") == head.get("etag")
        and receipt.get("head_last_modified") == head.get("last_modified")
        and receipt.get("redirect_chain") == []
        and isinstance(receipt.get("attempt_count"), int)
        and not isinstance(receipt.get("attempt_count"), bool)
        and 1 <= receipt["attempt_count"] <= EXPECTED_BUDGET["maximum_attempts_per_asset"]
        and isinstance(receipt.get("prior_transport_errors"), list)
        and all(isinstance(item, str) for item in receipt["prior_transport_errors"])
        and len(receipt["prior_transport_errors"]) == receipt["attempt_count"] - 1,
        "R11_DOWNLOAD_RECEIPT_HEAD_BINDING",
        "R11 download receipt/HEAD binding drift",
    )
    require(
        isinstance(receipt.get("sha256"), str)
        and re.fullmatch(r"[0-9A-F]{64}", receipt["sha256"]) is not None
        and isinstance(receipt.get("crc32"), str)
        and re.fullmatch(r"[0-9A-F]{8}", receipt["crc32"]) is not None,
        "R11_DOWNLOAD_RECEIPT_DIGEST",
        "R11 download receipt digest format drift",
    )
    target = materializer.safe_join(source_root, receipt["relative_path"])
    require(target.is_file() and target.stat().st_size == receipt["bytes"] and materializer.sha256_file(target) == receipt.get("sha256") and _crc32_file(target) == receipt.get("crc32"), "R11_DOWNLOAD_FILE_DRIFT", f"R11 downloaded file drift: {receipt['relative_path']}")
    return receipt


def execute(lock_path: Path, *, download_fn: Callable[..., dict[str, Any]] = materializer.download_bound_asset) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    source_root = _repo_path(SOURCE_ROOT)
    evidence_root = _repo_path(EVIDENCE_ROOT)
    files: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    global_deadline = started + float(lock["resource_budget"]["download_wall_seconds"])
    _reserve_execution_roots(
        source_root,
        evidence_root,
        execution_lock_sha256=materializer.sha256_file(lock["_lock_path"]),
        head_receipt_sha256=lock["_verified_bindings"]["R11_HEAD_RECEIPT"]["sha256"],
        files=files,
    )
    try:
        rows = expanded_download_plan(lock["_plan"])
        head_lookup = {row["url"]: row for row in lock["_head"]["assets"]}
        receipts: list[dict[str, Any]] = []
        get_attempts = 0
        for index, row in enumerate(rows, start=1):
            remaining_global_seconds = global_deadline - time.monotonic()
            require(remaining_global_seconds > 0.0, "R11_DOWNLOAD_TIMEOUT", "R11 download wall budget exceeded")
            raw = download_with_transient_retries(
                row,
                head_lookup[row["url"]],
                source_root=source_root,
                timeout_seconds=min(float(lock["resource_budget"]["download_timeout_seconds_per_asset"]), remaining_global_seconds),
                maximum_attempts=int(lock["resource_budget"]["maximum_attempts_per_asset"]),
                download_fn=download_fn,
            )
            require(time.monotonic() <= global_deadline, "R11_DOWNLOAD_TIMEOUT", "R11 download wall budget exceeded")
            receipt = {"schema": "blindassist.taro.o1r.r11_source_asset_download_receipt.v1", **dict(raw), "visit_id": row["visit_id"], "video_id": row["video_id"]}
            receipt["content_sha256"] = adapter.canonical_sha256(receipt)
            receipt = validate_download_receipt(row, head_lookup[row["url"]], receipt, source_root=source_root)
            receipts.append(receipt)
            get_attempts += receipt["attempt_count"]
            relative = f"receipts/{index:03d}.json"
            files[relative] = _write_exclusive(evidence_root / relative, receipt)
            print(json.dumps({"downloaded": index, "asset_count": ASSET_COUNT, "asset": row["asset"], "video_id": row["video_id"], "bytes": receipt["bytes"]}, sort_keys=True), flush=True)
        total = sum(row["bytes"] for row in receipts)
        require(time.monotonic() <= global_deadline, "R11_DOWNLOAD_TIMEOUT", "R11 download wall budget exceeded before success sealing")
        require(total == HEAD_TOTAL_BYTES, "R11_DOWNLOAD_TOTAL_DRIFT", "R11 download total differs from HEAD")
        require(get_attempts <= lock["resource_budget"]["maximum_get_attempts"], "R11_DOWNLOAD_RETRY_BUDGET", "R11 download GET-attempt budget exceeded")
        files["download-receipts.json"] = _write_exclusive(evidence_root / "download-receipts.json", {"schema": "blindassist.taro.o1r.r11_fresh_pool_download_receipts.v1", "receipts": receipts})
        result = {
            "schema": "blindassist.taro.o1r.r11_fresh_pool_download_result.v1",
            "execution_valid": True,
            "terminal": PASS_TERMINAL,
            "passed": True,
            "asset_count": ASSET_COUNT,
            "source_bytes": total,
            "network_get_requests": get_attempts,
            "recovered_asset_count": sum(row["attempt_count"] > 1 for row in receipts),
            "archive_decode": False,
            "source_frame_decode": False,
            "model_execution": False,
            "faro_read": False,
            "truth_scoring": False,
            "training": False,
            "one_shot_consumed": True,
        }
        files["result.json"] = _write_exclusive(evidence_root / "result.json", result)
        success_manifest = _sealed_record({"schema": "blindassist.taro.o1r.r11_fresh_pool_download_manifest.v1", "files": files, "one_shot_consumed": True})
        require(_projected_evidence_bytes(files, success_manifest) <= lock["resource_budget"]["maximum_evidence_bytes"], "R11_DOWNLOAD_EVIDENCE_BUDGET", "R11 download evidence budget exceeded")
        require(time.monotonic() <= global_deadline, "R11_DOWNLOAD_TIMEOUT", "R11 download wall budget exceeded before manifest sealing")
        _write_exclusive(evidence_root / "manifest.json", success_manifest)
        return result
    except Exception as error:
        try:
            _write_failure_terminal(evidence_root, error, files)
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
