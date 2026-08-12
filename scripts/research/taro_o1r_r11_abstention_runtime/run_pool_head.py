#!/usr/bin/env python3
"""Run the exact TARO R11 48-parent zero-body HEAD preflight once."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r11_abstention_runtime import fresh_pool
from scripts.research.taro_o1r_r11_abstention_runtime import validate_protocol_lock


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELATIVE = validate_protocol_lock.PROTOCOL_RELATIVE
AUTHORIZATION_RELATIVE = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-12.json"
LOCK_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_head_execution_lock.v1"
LOCK_ID = "TARO_O1R_R11_FRESH_48_PARENT_ZERO_BODY_HEAD_ONE_SHOT_EXECUTION_LOCK"
LOCK_RELATIVE = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_ZERO_BODY_HEAD_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-head-r0"
ASSET_COUNT = fresh_pool.PARENT_COUNT * len(fresh_pool.ASSET_TEMPLATES)
PASS_TERMINAL = "TARO_O1R_R11_FRESH_POOL_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED"
UNAVAILABLE_TERMINAL = "TARO_O1R_R11_FRESH_POOL_ASSET_HEADERS_NOT_AVAILABLE_NO_REPLACEMENT"
INVALID_TERMINAL = "TARO_O1R_R11_FRESH_POOL_HEAD_EXECUTION_INVALID"
AUTH_SCHEMA = "blindassist.taro.o1r.r11_data_use_authorization_receipt.v1"
AUTH_ID = "TARO_O1R_R11_FRESH_48_PARENT_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-12"
EXPECTED_SCOPE = {
    "official_fold": "Training",
    "pool_parent_count": 48,
    "asset_types_per_parent": ["upsampling.zip", "lowres_wide_intrinsics.zip", "lowres_wide.traj"],
    "request_count": 144,
    "pool_content_sha256": "9F1EE94980C9B2EB0C8D7A6503A25E11587760247C5A30F656DB28E60A27FFAF",
    "request_plan_sha256": "FE3578E4F8403F9F57DA767B21DC5EFBCAF6BBF6514DF776A7B3124B966BD521",
    "selected_parent_count_after_source_only_ranking": 24,
}
EXPECTED_OPERATIONS = [
    "zero-body HEAD availability and Content-Length preflight for the exact 144 bound URLs",
    "bounded download and integrity validation of the exact authorized source assets",
    "read-only container inventory and exact frame planning",
    "source-only Phase A for all 48 parents including sealed DepthART candidate and R7/R11/R9 source records with FARO reads equal to zero",
    "source-only top-24 parent ranking and immutable selection seal",
    "FARO Phase B only for the sealed selected top 24 parents",
    "fixed frame- and parent-aware R11 confirmation reduction",
]
EXPECTED_CONSTRAINTS = {
    "authorization_does_not_itself_activate_execution": True,
    "separate_hash_bound_one_shot_lock_required_per_stage": True,
    "no_parent_replacement": True,
    "phase_a_faro_reads": 0,
    "unselected_faro_reads": 0,
    "unknown_is_negative": False,
    "source_terms_apply": True,
    "redistribution_authorized": False,
}
EXPECTED_NOT_AUTHORIZED = [
    "training",
    "device execution",
    "deployment",
    "default App change",
    "product or production use",
    "safety claim",
    "source redistribution",
]
EXPECTED_BINDINGS = {
    "R11_PROTOCOL": PROTOCOL_RELATIVE,
    "R11_DATA_USE_AUTHORIZATION": AUTHORIZATION_RELATIVE,
    "R11_POOL_PLANNER": "scripts/research/taro_o1r_r11_abstention_runtime/fresh_pool.py",
    "R11_PROTOCOL_VALIDATOR": "scripts/research/taro_o1r_r11_abstention_runtime/validate_protocol_lock.py",
    "HEAD_TRANSPORT": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "R11_HEAD_RUNNER": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_head.py",
    "R11_HEAD_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_run_pool_head.py",
}
EXPECTED_AUTHORITY = {
    "head_requests": True,
    "request_count": ASSET_COUNT,
    "response_body_bytes": 0,
    "source_download": False,
    "source_decode": False,
    "model_execution": False,
    "faro_read": False,
    "truth_scoring": False,
    "training": False,
}
EXPECTED_BUDGET = {
    "maximum_compressed_source_bytes": 12884901888,
    "head_timeout_seconds": 30,
    "maximum_attempts_per_request": 2,
    "maximum_request_attempts": ASSET_COUNT * 2,
    "head_response_body_bytes": 0,
}
EXPECTED_USER_SCOPE = (
    "Exact frozen R11 48-parent Training pool and 144-URL plan: zero-body HEAD, bounded source download and "
    "integrity validation, all-48 source-only Phase A, source-only top-24 selection, then FARO only for the "
    "sealed top 24; no training, device, deployment, product, safety, or redistribution authority."
)
EXPECTED_AUTHORIZATION_REQUEST = (
    "授权 TARO O1R R11 使用协议锁定的 48 个 ARKitScenes Training parent（pool SHA "
    "`9F1EE94980C9B2EB0C8D7A6503A25E11587760247C5A30F656DB28E60A27FFAF`）及每个 parent 的 "
    "`upsampling.zip`、`lowres_wide_intrinsics.zip`、`lowres_wide.traj`（144 URL，request-plan SHA "
    "`FE3578E4F8403F9F57DA767B21DC5EFBCAF6BBF6514DF776A7B3124B966BD521`），依冻结顺序执行 "
    "zero-body HEAD、受限下载与完整性检查、全部 48-parent source-only Phase A、top-24 选择，以及仅 "
    "selected top-24 的 FARO Phase B。不得替换 parent；不授权训练、设备、部署、产品、安全或再分发。"
)
EXPECTED_AUTHORITY_TEXT = (
    "Use of the exact locked R11 48-parent ARKitScenes Training pool and three bound asset types for the frozen "
    "source-first WILD_LAB sequence through selected-only FARO confirmation, subject to separate hash-bound "
    "one-shot locks and source terms. This receipt does not itself activate a runner or grant training, device, "
    "deployment, product, safety, or redistribution authority."
)


class PoolHeadError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PoolHeadError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "R11_HEAD_JSON_OBJECT", f"JSON object required: {path}")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _validate_content_seal(value: Mapping[str, Any], code: str) -> dict[str, Any]:
    record = json.loads(json.dumps(dict(value)))
    claimed = record.pop("content_sha256", None)
    require(isinstance(claimed, str) and claimed == adapter.canonical_sha256(record), code, "content seal drift")
    record["content_sha256"] = claimed
    return record


def validate_authorization_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _validate_content_seal(value, "R11_DATA_AUTHORIZATION_HASH")
    require(
        receipt.get("schema") == AUTH_SCHEMA
        and receipt.get("receipt_id") == AUTH_ID
        and receipt.get("confirmed_by") == "user"
        and receipt.get("confirmed_at") == "2026-08-12"
        and receipt.get("confirmation_verbatim") == "授权"
        and receipt.get("authorization_request_verbatim") == EXPECTED_AUTHORIZATION_REQUEST,
        "R11_DATA_AUTHORIZATION_IDENTITY",
        "R11 data authorization identity drift",
    )
    scope = receipt.get("scope_binding")
    require(isinstance(scope, dict), "R11_DATA_AUTHORIZATION_SCOPE", "authorization scope missing")
    protocol_path = _repo_path(PROTOCOL_RELATIVE)
    planner_path = _repo_path(EXPECTED_BINDINGS["R11_POOL_PLANNER"])
    expected_bound = {
        "protocol_path": PROTOCOL_RELATIVE,
        "protocol_bytes": protocol_path.stat().st_size,
        "protocol_sha256": materializer.sha256_file(protocol_path),
        "pool_planner_path": EXPECTED_BINDINGS["R11_POOL_PLANNER"],
        "pool_planner_bytes": planner_path.stat().st_size,
        "pool_planner_sha256": materializer.sha256_file(planner_path),
        **EXPECTED_SCOPE,
    }
    require(scope == expected_bound, "R11_DATA_AUTHORIZATION_SCOPE", "authorization scope drift")
    require(receipt.get("authorized_operations") == EXPECTED_OPERATIONS, "R11_DATA_AUTHORIZATION_OPERATIONS", "authorized operations drift")
    require(receipt.get("execution_constraints") == EXPECTED_CONSTRAINTS, "R11_DATA_AUTHORIZATION_CONSTRAINTS", "authorization constraints drift")
    require(receipt.get("not_authorized") == EXPECTED_NOT_AUTHORIZED, "R11_DATA_AUTHORIZATION_CEILING", "authorization ceiling drift")
    require(receipt.get("authority") == EXPECTED_AUTHORITY_TEXT, "R11_DATA_AUTHORIZATION_CEILING", "authorization authority text drift")
    return receipt


def _normalize_response(response: Mapping[str, Any]) -> dict[str, Any]:
    status = response.get("http_status")
    length = response.get("content_length_bytes")
    return {
        "http_status": status if isinstance(status, int) and not isinstance(status, bool) else None,
        "content_length_bytes": length if isinstance(length, int) and not isinstance(length, bool) else None,
        "etag": response.get("etag") if response.get("etag") is None or isinstance(response.get("etag"), str) else None,
        "last_modified": response.get("last_modified") if response.get("last_modified") is None or isinstance(response.get("last_modified"), str) else None,
        "redirect_chain": list(response.get("redirect_chain") or []),
        "transport_errors": [str(item) for item in (response.get("transport_errors") or [])],
    }


def _is_available(row: Mapping[str, Any]) -> bool:
    return (
        row.get("http_status") == 200
        and isinstance(row.get("content_length_bytes"), int)
        and not isinstance(row.get("content_length_bytes"), bool)
        and row["content_length_bytes"] > 0
        and row.get("redirect_chain") == []
        and row.get("transport_errors") == []
    )


def build_head_receipt(
    plan: Mapping[str, Any],
    *,
    lock_sha256: str,
    protocol_sha256: str,
    authorization_sha256: str,
    head_fn: Callable[[Mapping[str, str], float], Mapping[str, Any]],
    timeout_seconds: float,
    maximum_attempts: int,
    maximum_bytes: int,
) -> dict[str, Any]:
    frozen = fresh_pool.validate_pool(plan, repo_root=REPO_ROOT, recompute=False)
    requests = frozen["request_plan"]["requests"]
    require(len(requests) == ASSET_COUNT, "R11_HEAD_PLAN_COUNT", "R11 HEAD plan count drift")
    assets: list[dict[str, Any]] = []
    total_attempts = 0
    for request in requests:
        attempts: list[dict[str, Any]] = []
        observed: dict[str, Any] | None = None
        for attempt_index in range(1, maximum_attempts + 1):
            total_attempts += 1
            try:
                observed = _normalize_response(dict(head_fn(request, timeout_seconds)))
            except Exception as error:
                observed = {
                    "http_status": None,
                    "content_length_bytes": None,
                    "etag": None,
                    "last_modified": None,
                    "redirect_chain": [],
                    "transport_errors": [str(getattr(error, "code", type(error).__name__))],
                }
            attempts.append({"attempt": attempt_index, **observed})
            if _is_available(observed):
                break
        require(observed is not None, "R11_HEAD_INTERNAL", "R11 HEAD attempt loop empty")
        assets.append({**dict(request), **observed, "attempt_count": len(attempts), "attempts": attempts})
    available = sum(_is_available(row) for row in assets)
    total = sum(row["content_length_bytes"] for row in assets if _is_available(row))
    passed = available == ASSET_COUNT and total <= maximum_bytes
    receipt = {
        "schema": "blindassist.taro.o1r.r11_fresh_pool_head_receipt.v1",
        "execution_lock_sha256": lock_sha256,
        "protocol_sha256": protocol_sha256,
        "authorization_receipt_sha256": authorization_sha256,
        "request_plan_sha256": frozen["request_plan"]["expanded_requests_sha256"],
        "request_method": "HEAD",
        "response_body_bytes_read": 0,
        "media_body_bytes_read": False,
        "asset_count": ASSET_COUNT,
        "available_asset_count": available,
        "request_attempt_count": total_attempts,
        "total_content_length_bytes": total,
        "maximum_compressed_source_bytes": maximum_bytes,
        "assets": assets,
        "terminal": PASS_TERMINAL if passed else UNAVAILABLE_TERMINAL,
        "passed": passed,
        "replacement_allowed": False,
        "source_download": False,
        "source_decode": False,
        "model_execution": False,
        "faro_read": False,
        "truth_scoring": False,
        "training": False,
    }
    receipt["content_sha256"] = adapter.canonical_sha256(receipt)
    return validate_head_receipt(frozen, receipt, maximum_attempts=maximum_attempts)


def validate_head_receipt(plan: Mapping[str, Any], value: Mapping[str, Any], *, maximum_attempts: int) -> dict[str, Any]:
    receipt = _validate_content_seal(value, "R11_HEAD_RECEIPT_HASH")
    require(
        receipt.get("schema") == "blindassist.taro.o1r.r11_fresh_pool_head_receipt.v1"
        and receipt.get("request_method") == "HEAD"
        and receipt.get("response_body_bytes_read") == 0
        and receipt.get("media_body_bytes_read") is False,
        "R11_HEAD_BODY_READ",
        "R11 HEAD body/read schema drift",
    )
    require(
        all(isinstance(receipt.get(field), str) and re.fullmatch(r"[0-9A-F]{64}", receipt[field]) is not None for field in ("execution_lock_sha256", "protocol_sha256", "authorization_receipt_sha256"))
        and receipt.get("request_plan_sha256") == EXPECTED_SCOPE["request_plan_sha256"]
        and receipt.get("maximum_compressed_source_bytes") == EXPECTED_BUDGET["maximum_compressed_source_bytes"],
        "R11_HEAD_RECEIPT_BINDING",
        "R11 HEAD receipt binding or byte ceiling drift",
    )
    requests = plan["request_plan"]["requests"]
    assets = receipt.get("assets")
    require(isinstance(assets, list) and len(assets) == len(requests) == receipt.get("asset_count") == ASSET_COUNT, "R11_HEAD_RECEIPT_COUNT", "R11 HEAD receipt count drift")
    available = 0
    total = 0
    total_attempts = 0
    final_fields = ("http_status", "content_length_bytes", "etag", "last_modified", "redirect_chain", "transport_errors")
    for expected, observed in zip(requests, assets, strict=True):
        require(all(observed.get(field) == expected[field] for field in ("visit_id", "video_id", "asset", "url")), "R11_HEAD_RECEIPT_IDENTITY", "R11 HEAD identity drift")
        attempts = observed.get("attempts")
        count = observed.get("attempt_count")
        require(isinstance(count, int) and not isinstance(count, bool) and 1 <= count <= maximum_attempts and isinstance(attempts, list) and len(attempts) == count, "R11_HEAD_RECEIPT_ATTEMPTS", "R11 HEAD attempts drift")
        for index, attempt in enumerate(attempts, start=1):
            require(isinstance(attempt, dict) and attempt.get("attempt") == index, "R11_HEAD_RECEIPT_ATTEMPT_INDEX", "R11 HEAD attempt index drift")
            require(
                isinstance(attempt.get("redirect_chain"), list)
                and all(isinstance(item, str) for item in attempt["redirect_chain"])
                and isinstance(attempt.get("transport_errors"), list)
                and all(isinstance(item, str) for item in attempt["transport_errors"]),
                "R11_HEAD_RECEIPT_ATTEMPT_SHAPE",
                "R11 HEAD attempt shape drift",
            )
            require(not (_is_available(attempt) and index != count), "R11_HEAD_RECEIPT_RETRY_AFTER_SUCCESS", "R11 HEAD retried after success")
        require(all(observed.get(field) == attempts[-1].get(field) for field in final_fields), "R11_HEAD_RECEIPT_FINAL_ATTEMPT", "R11 HEAD final fields differ from last attempt")
        total_attempts += count
        if _is_available(observed):
            available += 1
            total += observed["content_length_bytes"]
    require(receipt.get("request_attempt_count") == total_attempts, "R11_HEAD_RECEIPT_ATTEMPT_SUMMARY", "R11 HEAD attempt total drift")
    require(receipt.get("available_asset_count") == available and receipt.get("total_content_length_bytes") == total, "R11_HEAD_RECEIPT_SUMMARY", "R11 HEAD summary drift")
    passed = available == ASSET_COUNT and total <= receipt.get("maximum_compressed_source_bytes", -1)
    require(receipt.get("passed") is passed and receipt.get("terminal") == (PASS_TERMINAL if passed else UNAVAILABLE_TERMINAL), "R11_HEAD_RECEIPT_TERMINAL", "R11 HEAD terminal drift")
    require(
        receipt.get("replacement_allowed") is False
        and receipt.get("source_download") is False
        and receipt.get("source_decode") is False
        and receipt.get("model_execution") is False
        and receipt.get("faro_read") is False
        and receipt.get("truth_scoring") is False
        and receipt.get("training") is False,
        "R11_HEAD_AUTHORITY_ESCALATION",
        "R11 HEAD receipt authority escalated",
    )
    return receipt


def _git_bytes(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(completed.returncode == 0, "R11_HEAD_IMPLEMENTATION_COMMIT", f"binding absent from implementation commit: {relative}")
    return completed.stdout


def _validate_implementation_ancestor(commit: str) -> None:
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "R11_HEAD_IMPLEMENTATION_COMMIT", "implementation commit must be lowercase full SHA")
    completed = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO_ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(completed.returncode == 0, "R11_HEAD_IMPLEMENTATION_COMMIT", "implementation commit is not an ancestor of HEAD")


def validate_execution_lock(path: Path, *, require_output_absent: bool = True) -> dict[str, Any]:
    lock_path = path.resolve()
    require(lock_path == _repo_path(LOCK_RELATIVE), "R11_HEAD_LOCK_PATH", "R11 HEAD lock path drift")
    lock = _validate_content_seal(_read_json(lock_path), "R11_HEAD_LOCK_HASH")
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R11_HEAD_LOCK_IDENTITY", "R11 HEAD lock identity drift")
    expected_argv = ["scripts/research/taro_o1r_r11_abstention_runtime/run_pool_head.py", "--execution-lock", LOCK_RELATIVE]
    require(lock.get("argv") == expected_argv and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R11_HEAD_LOCK_POLICY", "R11 HEAD lock policy drift")
    implementation_commit = lock.get("implementation_commit")
    _validate_implementation_ancestor(implementation_commit)
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R11_HEAD_BINDINGS", "R11 HEAD binding count drift")
    seen: set[str] = set()
    for row in bindings:
        role = row.get("role")
        relative = row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in seen and EXPECTED_BINDINGS.get(role) == relative, "R11_HEAD_BINDING_ROW", "R11 HEAD binding row drift")
        target = _repo_path(relative)
        require(target.is_file(), "R11_HEAD_BINDING_HASH", f"R11 HEAD binding absent: {relative}")
        payload = target.read_bytes()
        committed = _git_bytes(implementation_commit, relative)
        require(payload == committed and len(payload) == row["bytes"] and _sha256_bytes(payload) == row["sha256"], "R11_HEAD_BINDING_HASH", f"R11 HEAD binding drift: {relative}")
        seen.add(role)
    validate_protocol_lock.validate_protocol(_read_json(_repo_path(PROTOCOL_RELATIVE)), repo_root=REPO_ROOT, recompute_pool=True)
    authorization = validate_authorization_receipt(_read_json(_repo_path(AUTHORIZATION_RELATIVE)))
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY and lock.get("resource_budget") == EXPECTED_BUDGET, "R11_HEAD_AUTHORITY_BUDGET", "R11 HEAD authority/budget drift")
    plan = fresh_pool.build_pool(REPO_ROOT)
    require(
        lock.get("request_plan_sha256") == plan["request_plan"]["expanded_requests_sha256"]
        and (not require_output_absent or not _repo_path(OUTPUT_ROOT).exists()),
        "R11_HEAD_PLAN_ROOT",
        "R11 HEAD plan drift or root exists",
    )
    user = lock.get("user_authority")
    require(
        isinstance(user, dict)
        and user.get("confirmed_by") == "user"
        and user.get("confirmed_at") == "2026-08-12"
        and user.get("confirmation_verbatim") == "授权"
        and user.get("scope") == EXPECTED_USER_SCOPE,
        "R11_HEAD_USER_AUTHORITY",
        "R11 HEAD user authority drift",
    )
    require(lock.get("authorization_receipt_content_sha256") == authorization["content_sha256"], "R11_HEAD_USER_AUTHORITY", "authorization receipt seal drift")
    lock["_lock_path"] = lock_path
    lock["_plan"] = plan
    return lock


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(not path.exists(), "R11_HEAD_OUTPUT_COLLISION", f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter.canonical_json_bytes(dict(value)) + b"\n"
    partial = path.with_name(path.name + ".partial")
    require(not partial.exists(), "R11_HEAD_PARTIAL_COLLISION", f"partial exists: {partial}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": len(payload), "sha256": materializer.sha256_bytes(payload)}


def execute(lock_path: Path, *, head_fn: Callable[[Mapping[str, str], float], Mapping[str, Any]] = materializer.production_head) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    output = _repo_path(OUTPUT_ROOT)
    output.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, Any]] = {}
    try:
        files["start-receipt.json"] = _write_exclusive(
            output / "start-receipt.json",
            {
                "schema": "blindassist.taro.o1r.r11_fresh_pool_head_start.v1",
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]),
                "request_plan_sha256": lock["request_plan_sha256"],
                "one_shot_consumed_on_root_creation": True,
                "response_body_bytes_allowed": 0,
            },
        )
        budget = lock["resource_budget"]
        receipt = build_head_receipt(
            lock["_plan"],
            lock_sha256=materializer.sha256_file(lock["_lock_path"]),
            protocol_sha256=materializer.sha256_file(_repo_path(PROTOCOL_RELATIVE)),
            authorization_sha256=materializer.sha256_file(_repo_path(AUTHORIZATION_RELATIVE)),
            head_fn=head_fn,
            timeout_seconds=float(budget["head_timeout_seconds"]),
            maximum_attempts=int(budget["maximum_attempts_per_request"]),
            maximum_bytes=int(budget["maximum_compressed_source_bytes"]),
        )
        files["head-receipt.json"] = _write_exclusive(output / "head-receipt.json", receipt)
        result = {
            "schema": "blindassist.taro.o1r.r11_fresh_pool_head_result.v1",
            "execution_valid": True,
            "terminal": receipt["terminal"],
            "passed": receipt["passed"],
            "available_asset_count": receipt["available_asset_count"],
            "asset_count": ASSET_COUNT,
            "request_attempt_count": receipt["request_attempt_count"],
            "total_content_length_bytes": receipt["total_content_length_bytes"],
            "response_body_bytes_read": 0,
            "one_shot_consumed": True,
            "replacement_allowed": False,
        }
        files["result.json"] = _write_exclusive(output / "result.json", result)
        _write_exclusive(output / "manifest.json", {"schema": "blindassist.taro.o1r.r11_fresh_pool_head_manifest.v1", "files": files, "one_shot_consumed": True})
        return result
    except Exception as error:
        try:
            _write_exclusive(
                output / "failure.json",
                {
                    "schema": "blindassist.taro.o1r.r11_fresh_pool_head_failure.v1",
                    "execution_valid": False,
                    "terminal": INVALID_TERMINAL,
                    "failure_code": getattr(error, "code", type(error).__name__),
                    "message": str(error),
                    "one_shot_consumed": True,
                    "response_body_bytes_read": 0,
                },
            )
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
