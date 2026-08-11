#!/usr/bin/env python3
"""Run the one-shot HEAD-only preflight for the exact TARO R7 fresh cohort."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import fresh_confirmation_cohort as cohort
from scripts.research.taro_o1r_r7_canary_runtime import validate_fresh_data_lock


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r7_fresh_confirmation_head_execution_lock.v1"
LOCK_ID = "TARO_O1R_R7_FRESH_CONFIRMATION_CONTENT_LENGTH_HEAD_ONE_SHOT_EXECUTION_LOCK"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-head-r0"
RECEIPT_PATH = f"{OUTPUT_ROOT}/head-receipt.json"
PASS_TERMINAL = "TARO_O1R_R7_FRESH_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED"
UNAVAILABLE_TERMINAL = "TARO_O1R_R7_FRESH_ASSET_HEADERS_NOT_AVAILABLE_NO_REPLACEMENT"
INVALID_TERMINAL = "TARO_O1R_R7_FRESH_HEAD_EXECUTION_INVALID_ONE_SHOT_CONSUMED"
EXPECTED_BINDING_PATHS = {
    "R7_DATA_LOCK": "docs/research/taro/TARO_O1R_R7_FRESH_CONFIRMATION_COHORT_AND_DATA_USE_LOCK_2026-08-12.json",
    "R7_COHORT_PLANNER": "scripts/research/taro_o1r_r7_canary_runtime/fresh_confirmation_cohort.py",
    "R7_DATA_LOCK_VALIDATOR": "scripts/research/taro_o1r_r7_canary_runtime/validate_fresh_data_lock.py",
    "HEAD_TRANSPORT": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "R7_HEAD_RUNNER": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_head.py",
}
EXPECTED_AUTHORITY = {
    "head_only": True,
    "response_body_bytes_allowed": 0,
    "source_download": False,
    "source_decode": False,
    "model_execution": False,
    "truth_scoring": False,
    "training": False,
    "device": False,
    "product": False,
    "safety": False,
}


class FreshHeadError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise FreshHeadError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(not path.exists(), "R7_HEAD_OUTPUT_COLLISION", f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter.canonical_json_bytes(dict(value)) + b"\n"
    partial = path.with_name(path.name + ".partial")
    require(not partial.exists(), "R7_HEAD_PARTIAL_COLLISION", f"partial already exists: {partial}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": len(payload), "sha256": materializer.sha256_bytes(payload)}


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID, "R7_HEAD_LOCK_IDENTITY", "R7 HEAD lock identity drift")
    require(lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R7_HEAD_NOT_AUTHORIZED", "R7 HEAD lock is not authorized and unconsumed")
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "R7_HEAD_AUTHORITY_DRIFT", "R7 HEAD authority drift")
    require(lock.get("output_root") == OUTPUT_ROOT and lock.get("output_receipt") == RECEIPT_PATH and lock.get("overwrite") is False and lock.get("rerun") is False, "R7_HEAD_OUTPUT_POLICY_DRIFT", "R7 HEAD output policy drift")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv, "R7_HEAD_ARGV_DRIFT", "R7 HEAD argv drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDING_PATHS), "R7_HEAD_BINDINGS_DRIFT", "R7 HEAD binding count drift")
    verified = {}
    for row in bindings:
        require(isinstance(row, dict) and set(row) == {"role", "path", "bytes", "sha256"}, "R7_HEAD_BINDING_ROW", "R7 HEAD binding row drift")
        role, relative = row["role"], row["path"]
        require(role not in verified and EXPECTED_BINDING_PATHS.get(role) == relative, "R7_HEAD_BINDING_PATH", "R7 HEAD binding role/path drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R7_HEAD_BINDING_HASH", f"R7 HEAD binding hash drift: {relative}")
        verified[role] = row
    require(set(verified) == set(EXPECTED_BINDING_PATHS), "R7_HEAD_BINDING_SET", "R7 HEAD binding set drift")
    plan = cohort.build_plan(REPO_ROOT)
    require(validate_fresh_data_lock.validate(_repo_path(EXPECTED_BINDING_PATHS["R7_DATA_LOCK"]))["passed"], "R7_HEAD_DATA_LOCK_INVALID", "R7 fresh data-use lock validation failed")
    budget = lock.get("resource_budget", {})
    require(budget == {"maximum_compressed_source_bytes": 2147483648, "head_timeout_seconds": 30, "head_retries": 2, "maximum_request_attempts": 48, "head_response_body_bytes": 0}, "R7_HEAD_BUDGET_DRIFT", "R7 HEAD budget drift")
    require(plan["request_plan"]["expanded_requests_sha256"] == lock.get("request_plan_sha256"), "R7_HEAD_REQUEST_PLAN_DRIFT", "R7 HEAD request plan drift")
    lock["_lock_path"] = lock_path
    lock["_verified_bindings"] = verified
    lock["_plan"] = plan
    return lock


def build_head_receipt(
    plan: Mapping[str, Any],
    *,
    execution_lock_sha256: str,
    data_lock_sha256: str,
    head_fn: Callable[[Mapping[str, str], float], Mapping[str, Any]],
    timeout_seconds: float,
    maximum_attempts: int,
    maximum_compressed_source_bytes: int,
) -> dict[str, Any]:
    requests = plan["request_plan"]["requests"]
    assets = []
    total_attempts = 0
    for request in requests:
        attempts = []
        observed: dict[str, Any] | None = None
        for attempt_index in range(maximum_attempts):
            total_attempts += 1
            try:
                response = dict(head_fn(request, timeout_seconds))
                observed = {
                    "http_status": response.get("http_status"),
                    "content_length_bytes": response.get("content_length_bytes"),
                    "etag": response.get("etag"),
                    "last_modified": response.get("last_modified"),
                    "redirect_chain": list(response.get("redirect_chain") or []),
                    "transport_errors": list(response.get("transport_errors") or []),
                }
            except Exception as error:
                observed = {"http_status": None, "content_length_bytes": None, "etag": None, "last_modified": None, "redirect_chain": [], "transport_errors": [str(getattr(error, "code", type(error).__name__))]}
            attempts.append({"attempt": attempt_index + 1, **observed})
            if observed["http_status"] == 200 and isinstance(observed["content_length_bytes"], int) and observed["content_length_bytes"] > 0 and observed["redirect_chain"] == [] and observed["transport_errors"] == []:
                break
        require(observed is not None, "R7_HEAD_ATTEMPT_INTERNAL", "R7 HEAD attempt loop returned no observation")
        assets.append({**dict(request), **observed, "attempt_count": len(attempts), "attempts": attempts})
    available = sum(row["http_status"] == 200 and isinstance(row["content_length_bytes"], int) and row["content_length_bytes"] > 0 and row["redirect_chain"] == [] and row["transport_errors"] == [] for row in assets)
    total = sum(row["content_length_bytes"] for row in assets if row["http_status"] == 200 and isinstance(row["content_length_bytes"], int) and row["content_length_bytes"] > 0 and row["redirect_chain"] == [] and row["transport_errors"] == [])
    passed = available == len(requests) and total <= maximum_compressed_source_bytes
    receipt = {
        "schema": "blindassist.taro.o1r.r7_fresh_confirmation_head_receipt.v1",
        "execution_lock_sha256": execution_lock_sha256,
        "data_lock_sha256": data_lock_sha256,
        "request_plan_sha256": plan["request_plan"]["expanded_requests_sha256"],
        "request_method": "HEAD",
        "response_body_bytes_read": 0,
        "media_body_bytes_read": False,
        "asset_count": len(requests),
        "available_asset_count": available,
        "request_attempt_count": total_attempts,
        "total_content_length_bytes": total,
        "maximum_compressed_source_bytes": maximum_compressed_source_bytes,
        "assets": assets,
        "terminal": PASS_TERMINAL if passed else UNAVAILABLE_TERMINAL,
        "passed": passed,
        "replacement_allowed": False,
        "source_download": False,
        "source_decode": False,
        "model_execution": False,
        "truth_scoring": False,
        "training": False,
    }
    receipt["content_sha256"] = adapter.canonical_sha256(receipt)
    return validate_head_receipt(plan, receipt, maximum_attempts=maximum_attempts)


def validate_head_receipt(plan: Mapping[str, Any], value: Mapping[str, Any], *, maximum_attempts: int) -> dict[str, Any]:
    receipt = json.loads(json.dumps(dict(value)))
    content = receipt.pop("content_sha256", None)
    require(isinstance(content, str) and adapter.canonical_sha256(receipt) == content, "R7_HEAD_RECEIPT_HASH", "R7 HEAD receipt content hash drift")
    receipt["content_sha256"] = content
    require(receipt.get("request_method") == "HEAD" and receipt.get("response_body_bytes_read") == 0 and receipt.get("media_body_bytes_read") is False, "R7_HEAD_BODY_READ", "R7 HEAD receipt body-read drift")
    requests, assets = plan["request_plan"]["requests"], receipt.get("assets")
    require(isinstance(assets, list) and len(assets) == len(requests) == receipt.get("asset_count") == 24, "R7_HEAD_RECEIPT_COUNT", "R7 HEAD receipt asset count drift")
    available = total = 0
    for expected, observed in zip(requests, assets, strict=True):
        require(all(observed.get(field) == expected[field] for field in ("visit_id", "video_id", "asset", "url")), "R7_HEAD_RECEIPT_IDENTITY", "R7 HEAD receipt identity drift")
        require(isinstance(observed.get("attempt_count"), int) and 1 <= observed["attempt_count"] <= maximum_attempts and len(observed.get("attempts", [])) == observed["attempt_count"], "R7_HEAD_RECEIPT_ATTEMPTS", "R7 HEAD receipt attempts drift")
        if observed.get("http_status") == 200 and isinstance(observed.get("content_length_bytes"), int) and observed["content_length_bytes"] > 0 and observed.get("redirect_chain") == [] and observed.get("transport_errors") == []:
            available += 1
            total += observed["content_length_bytes"]
    require(receipt.get("available_asset_count") == available and receipt.get("total_content_length_bytes") == total, "R7_HEAD_RECEIPT_SUMMARY", "R7 HEAD receipt summary drift")
    expected_pass = available == 24 and total <= receipt.get("maximum_compressed_source_bytes", -1)
    require(receipt.get("passed") is expected_pass and receipt.get("terminal") == (PASS_TERMINAL if expected_pass else UNAVAILABLE_TERMINAL), "R7_HEAD_RECEIPT_TERMINAL", "R7 HEAD receipt terminal drift")
    return receipt


def execute(lock_path: Path, *, head_fn: Callable[[Mapping[str, str], float], Mapping[str, Any]] = materializer.production_head) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    output = _repo_path(OUTPUT_ROOT)
    require(not output.exists(), "R7_HEAD_ROOT_COLLISION", "R7 HEAD output root already exists")
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    try:
        start = {"schema": "blindassist.taro.o1r.r7_fresh_head_start.v1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "request_plan_sha256": lock["request_plan_sha256"], "one_shot_consumed_on_root_creation": True, "response_body_bytes_allowed": 0}
        files["start-receipt.json"] = _write_exclusive(output / "start-receipt.json", start)
        budget = lock["resource_budget"]
        receipt = build_head_receipt(lock["_plan"], execution_lock_sha256=materializer.sha256_file(lock["_lock_path"]), data_lock_sha256=lock["_verified_bindings"]["R7_DATA_LOCK"]["sha256"], head_fn=head_fn, timeout_seconds=float(budget["head_timeout_seconds"]), maximum_attempts=int(budget["head_retries"]), maximum_compressed_source_bytes=int(budget["maximum_compressed_source_bytes"]))
        files["head-receipt.json"] = _write_exclusive(output / "head-receipt.json", receipt)
        result = {"schema": "blindassist.taro.o1r.r7_fresh_head_result.v1", "execution_valid": True, "terminal": receipt["terminal"], "passed": receipt["passed"], "available_asset_count": receipt["available_asset_count"], "asset_count": receipt["asset_count"], "total_content_length_bytes": receipt["total_content_length_bytes"], "response_body_bytes_read": 0, "one_shot_consumed": True, "replacement_allowed": False}
        files["result.json"] = _write_exclusive(output / "result.json", result)
        _write_exclusive(output / "manifest.json", {"schema": "blindassist.taro.o1r.r7_fresh_head_manifest.v1", "files": files, "one_shot_consumed": True})
        return result
    except Exception as error:
        failure = {"schema": "blindassist.taro.o1r.r7_fresh_head_failure.v1", "execution_valid": False, "terminal": INVALID_TERMINAL, "failure_code": getattr(error, "code", type(error).__name__), "message": str(error), "one_shot_consumed": True, "response_body_bytes_read": 0}
        try:
            _write_exclusive(output / "failure.json", failure)
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
        print(json.dumps({"execution_valid": False, "error_code": getattr(error, "code", type(error).__name__), "message": str(error), "one_shot_consumed": _repo_path(OUTPUT_ROOT).exists()}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
