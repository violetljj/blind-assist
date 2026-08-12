#!/usr/bin/env python3
"""HEAD-preflight the exact TARO R10 32-parent fresh pool."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r10_clear_runtime import fresh_pool


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r10_fresh_pool_head_execution_lock.v1"
LOCK_ID = "TARO_O1R_R10_FRESH_POOL_ZERO_BODY_HEAD_ONE_SHOT_EXECUTION_LOCK"
LOCK_RELATIVE = "docs/research/taro/TARO_O1R_R10_FRESH_POOL_ZERO_BODY_HEAD_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r10-fresh-pool-head-r0"
ASSET_COUNT = 96
PASS_TERMINAL = "TARO_O1R_R10_FRESH_POOL_HEADERS_AVAILABLE_MEDIA_UNOPENED"
UNAVAILABLE_TERMINAL = "TARO_O1R_R10_FRESH_POOL_HEADERS_UNAVAILABLE_NO_REPLACEMENT"
INVALID_TERMINAL = "TARO_O1R_R10_FRESH_POOL_HEAD_EXECUTION_INVALID"
EXPECTED_BINDINGS = {
    "R10_PROTOCOL": "docs/research/taro/TARO_O1R_R10_FRESH_PARENT_SOURCE_ONLY_CLEAR_ENRICHED_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json",
    "R10_POOL_PLANNER": "scripts/research/taro_o1r_r10_clear_runtime/fresh_pool.py",
    "R10_POOL_TEST": "scripts/research/taro_o1r_r10_clear_runtime/test_fresh_pool.py",
    "HEAD_TRANSPORT": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "R10_HEAD_RUNNER": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_head.py",
    "R10_HEAD_TEST": "scripts/research/taro_o1r_r10_clear_runtime/test_run_pool_head.py",
}
EXPECTED_AUTHORITY = {"head_requests": True, "request_count": ASSET_COUNT, "response_body_bytes": 0, "source_download": False, "source_decode": False, "model_execution": False, "faro_read": False, "truth_scoring": False, "training": False}
EXPECTED_BUDGET = {"maximum_compressed_source_bytes": 8589934592, "head_timeout_seconds": 30, "head_retries": 2, "maximum_request_attempts": 192, "head_response_body_bytes": 0}


class PoolHeadError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PoolHeadError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(not path.exists(), "R10_HEAD_OUTPUT_COLLISION", f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter.canonical_json_bytes(dict(value)) + b"\n"
    partial = path.with_name(path.name + ".partial")
    require(not partial.exists(), "R10_HEAD_PARTIAL_COLLISION", f"partial exists: {partial}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": len(payload), "sha256": materializer.sha256_bytes(payload)}


def build_head_receipt(plan: Mapping[str, Any], *, lock_sha256: str, protocol_sha256: str, head_fn: Callable[[Mapping[str, str], float], Mapping[str, Any]], timeout_seconds: float, maximum_attempts: int, maximum_bytes: int) -> dict[str, Any]:
    requests = plan["request_plan"]["requests"]
    require(len(requests) == ASSET_COUNT, "R10_HEAD_PLAN_COUNT", "R10 HEAD plan count drift")
    assets = []
    total_attempts = 0
    for request in requests:
        attempts = []
        observed = None
        for attempt in range(1, maximum_attempts + 1):
            total_attempts += 1
            try:
                response = dict(head_fn(request, timeout_seconds))
                observed = {"http_status": response.get("http_status"), "content_length_bytes": response.get("content_length_bytes"), "etag": response.get("etag"), "last_modified": response.get("last_modified"), "redirect_chain": list(response.get("redirect_chain") or []), "transport_errors": list(response.get("transport_errors") or [])}
            except Exception as error:
                observed = {"http_status": None, "content_length_bytes": None, "etag": None, "last_modified": None, "redirect_chain": [], "transport_errors": [str(getattr(error, "code", type(error).__name__))]}
            attempts.append({"attempt": attempt, **observed})
            if observed["http_status"] == 200 and isinstance(observed["content_length_bytes"], int) and observed["content_length_bytes"] > 0 and observed["redirect_chain"] == [] and observed["transport_errors"] == []:
                break
        require(observed is not None, "R10_HEAD_INTERNAL", "R10 HEAD attempt loop empty")
        assets.append({**dict(request), **observed, "attempt_count": len(attempts), "attempts": attempts})
    good = lambda row: row["http_status"] == 200 and isinstance(row["content_length_bytes"], int) and row["content_length_bytes"] > 0 and row["redirect_chain"] == [] and row["transport_errors"] == []
    available = sum(good(row) for row in assets)
    total = sum(row["content_length_bytes"] for row in assets if good(row))
    passed = available == ASSET_COUNT and total <= maximum_bytes
    receipt = {"schema": "blindassist.taro.o1r.r10_fresh_pool_head_receipt.v1", "execution_lock_sha256": lock_sha256, "protocol_sha256": protocol_sha256, "request_plan_sha256": plan["request_plan"]["expanded_requests_sha256"], "request_method": "HEAD", "response_body_bytes_read": 0, "media_body_bytes_read": False, "asset_count": ASSET_COUNT, "available_asset_count": available, "request_attempt_count": total_attempts, "total_content_length_bytes": total, "maximum_compressed_source_bytes": maximum_bytes, "assets": assets, "terminal": PASS_TERMINAL if passed else UNAVAILABLE_TERMINAL, "passed": passed, "replacement_allowed": False, "source_download": False, "source_decode": False, "model_execution": False, "faro_read": False, "truth_scoring": False, "training": False}
    receipt["content_sha256"] = adapter.canonical_sha256(receipt)
    return validate_head_receipt(plan, receipt, maximum_attempts=maximum_attempts)


def validate_head_receipt(plan: Mapping[str, Any], value: Mapping[str, Any], *, maximum_attempts: int) -> dict[str, Any]:
    receipt = json.loads(json.dumps(dict(value)))
    content = receipt.pop("content_sha256", None)
    require(isinstance(content, str) and adapter.canonical_sha256(receipt) == content, "R10_HEAD_RECEIPT_HASH", "R10 HEAD receipt hash drift")
    receipt["content_sha256"] = content
    require(receipt.get("request_method") == "HEAD" and receipt.get("response_body_bytes_read") == 0 and receipt.get("media_body_bytes_read") is False, "R10_HEAD_BODY_READ", "R10 HEAD body-read drift")
    requests, assets = plan["request_plan"]["requests"], receipt.get("assets")
    require(isinstance(assets, list) and len(assets) == len(requests) == receipt.get("asset_count") == ASSET_COUNT, "R10_HEAD_RECEIPT_COUNT", "R10 HEAD receipt count drift")
    available = total = 0
    for expected, observed in zip(requests, assets, strict=True):
        require(all(observed.get(field) == expected[field] for field in ("visit_id", "video_id", "asset", "url")), "R10_HEAD_RECEIPT_IDENTITY", "R10 HEAD identity drift")
        require(isinstance(observed.get("attempt_count"), int) and 1 <= observed["attempt_count"] <= maximum_attempts and len(observed.get("attempts", [])) == observed["attempt_count"], "R10_HEAD_RECEIPT_ATTEMPTS", "R10 HEAD attempts drift")
        if observed.get("http_status") == 200 and isinstance(observed.get("content_length_bytes"), int) and observed["content_length_bytes"] > 0 and observed.get("redirect_chain") == [] and observed.get("transport_errors") == []:
            available += 1
            total += observed["content_length_bytes"]
    require(receipt.get("available_asset_count") == available and receipt.get("total_content_length_bytes") == total, "R10_HEAD_RECEIPT_SUMMARY", "R10 HEAD summary drift")
    passed = available == ASSET_COUNT and total <= receipt.get("maximum_compressed_source_bytes", -1)
    require(receipt.get("passed") is passed and receipt.get("terminal") == (PASS_TERMINAL if passed else UNAVAILABLE_TERMINAL), "R10_HEAD_RECEIPT_TERMINAL", "R10 HEAD terminal drift")
    return receipt


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    require(lock_path == _repo_path(LOCK_RELATIVE), "R10_HEAD_LOCK_PATH", "R10 HEAD lock path drift")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") in {"PENDING_USER_AUTHORITY", "AUTHORIZED_UNCONSUMED"} and lock.get("consumed") is False, "R10_HEAD_LOCK_IDENTITY", "R10 HEAD lock identity drift")
    expected_argv = ["scripts/research/taro_o1r_r10_clear_runtime/run_pool_head.py", "--execution-lock", LOCK_RELATIVE]
    require(lock.get("argv") == expected_argv and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R10_HEAD_LOCK_POLICY", "R10 HEAD lock policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R10_HEAD_BINDINGS", "R10 HEAD binding count drift")
    seen = set()
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in seen and EXPECTED_BINDINGS.get(role) == relative, "R10_HEAD_BINDING_ROW", "R10 HEAD binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R10_HEAD_BINDING_HASH", f"R10 HEAD binding drift: {relative}")
        seen.add(role)
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY and lock.get("resource_budget") == EXPECTED_BUDGET, "R10_HEAD_AUTHORITY_BUDGET", "R10 HEAD authority/budget drift")
    plan = fresh_pool.build_pool(REPO_ROOT)
    require(lock.get("request_plan_sha256") == plan["request_plan"]["expanded_requests_sha256"] and not _repo_path(OUTPUT_ROOT).exists(), "R10_HEAD_PLAN_ROOT", "R10 HEAD plan drift or root exists")
    if lock["status"] == "PENDING_USER_AUTHORITY":
        require(lock.get("user_authority") is None, "R10_HEAD_USER_AUTHORITY", "pending R10 HEAD lock unexpectedly has authority")
        raise PoolHeadError("R10_HEAD_USER_AUTHORITY_REQUIRED", "exact 32-parent/96-URL R10 zero-body HEAD requires explicit user authority")
    authority = lock.get("user_authority")
    require(isinstance(authority, dict) and authority.get("confirmed_by") == "user" and authority.get("scope") == "Exact frozen R10 32-parent Training pool and 96-URL plan: zero-body HEAD, bounded source download, source-only Phase A and selector ranking, then FARO only for the sealed top eight; no training, deployment, product, or safety authority.", "R10_HEAD_USER_AUTHORITY", "authorized R10 HEAD lock user scope drift")
    lock["_lock_path"], lock["_plan"] = lock_path, plan
    return lock


def execute(lock_path: Path, *, head_fn: Callable[[Mapping[str, str], float], Mapping[str, Any]] = materializer.production_head) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    output = _repo_path(OUTPUT_ROOT)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    try:
        files["start-receipt.json"] = _write_exclusive(output / "start-receipt.json", {"schema": "blindassist.taro.o1r.r10_fresh_pool_head_start.v1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "request_plan_sha256": lock["request_plan_sha256"], "one_shot_consumed_on_root_creation": True, "response_body_bytes_allowed": 0})
        budget = lock["resource_budget"]
        receipt = build_head_receipt(lock["_plan"], lock_sha256=materializer.sha256_file(lock["_lock_path"]), protocol_sha256=next(row["sha256"] for row in lock["bindings"] if row["role"] == "R10_PROTOCOL"), head_fn=head_fn, timeout_seconds=float(budget["head_timeout_seconds"]), maximum_attempts=int(budget["head_retries"]), maximum_bytes=int(budget["maximum_compressed_source_bytes"]))
        files["head-receipt.json"] = _write_exclusive(output / "head-receipt.json", receipt)
        result = {"schema": "blindassist.taro.o1r.r10_fresh_pool_head_result.v1", "execution_valid": True, "terminal": receipt["terminal"], "passed": receipt["passed"], "available_asset_count": receipt["available_asset_count"], "asset_count": ASSET_COUNT, "total_content_length_bytes": receipt["total_content_length_bytes"], "response_body_bytes_read": 0, "one_shot_consumed": True, "replacement_allowed": False}
        files["result.json"] = _write_exclusive(output / "result.json", result)
        _write_exclusive(output / "manifest.json", {"schema": "blindassist.taro.o1r.r10_fresh_pool_head_manifest.v1", "files": files, "one_shot_consumed": True})
        return result
    except Exception as error:
        try:
            _write_exclusive(output / "failure.json", {"schema": "blindassist.taro.o1r.r10_fresh_pool_head_failure.v1", "execution_valid": False, "terminal": INVALID_TERMINAL, "failure_code": getattr(error, "code", type(error).__name__), "message": str(error), "one_shot_consumed": True, "response_body_bytes_read": 0})
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
