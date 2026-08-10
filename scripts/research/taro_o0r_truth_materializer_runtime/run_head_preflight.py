#!/usr/bin/env python3
"""Run a future hash-bound HEAD-only TARO O0R asset availability preflight."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.taro_o0r_truth_materializer_runtime.materializer import (
    MaterializerError,
    build_head_receipt,
    canonical_json_bytes,
    load_json,
    production_head,
    require,
    safe_join,
    sha256_bytes,
    sha256_file,
    validate_authorization,
    validate_head_receipt,
)


HEAD_EXECUTION_LOCK_SCHEMA = "blindassist.taro.o0r.content_length_head_execution_lock.v1"
HEAD_OUTPUT_ROOT = "artifacts.local/evidence/taro/o0r-arkitscenes-head-r1"
HEAD_RECEIPT_PATH = f"{HEAD_OUTPUT_ROOT}/head-receipt.json"
EXPECTED_HEAD_BINDINGS = {
    "AVAILABILITY_SUCCESSOR_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_AVAILABILITY_SUCCESSOR_R1_LOCK_2026-08-10.json",
    "TRUTH_ONLY_PREFLIGHT_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_ONLY_PREFLIGHT_R1_LOCK_2026-08-10.json",
    "DATA_USE_AUTHORIZATION": "docs/research/taro/TARO_O0R_ARKITSCENES_DATA_USE_AUTHORIZATION_R1_RECEIPT_2026-08-10.json",
    "MATERIALIZER_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R1_IMPLEMENTATION_LOCK_2026-08-10.json",
    "MATERIALIZER": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "HEAD_RUNNER": "scripts/research/taro_o0r_truth_materializer_runtime/run_head_preflight.py",
}
EXPECTED_ABSENT_ROOTS = [
    "artifacts.local/datasets/taro/o0r-arkitscenes-source-adapter-r1",
    "artifacts.local/work/taro/o0r-arkitscenes-source-adapter-r1",
    "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r1",
    "artifacts.local/evidence/taro/o0r-arkitscenes-factor-headroom-r1",
]
EXPECTED_AUTHORITY = {
    "head_only": True,
    "response_body_bytes_allowed": 0,
    "source_download": False,
    "source_payload_open": False,
    "truth_materialization": False,
    "depthart_inference": False,
    "factorial_execution": False,
    "training": False,
    "device": False,
    "product": False,
    "safety": False,
}


def _repo_path(relative: str) -> Path:
    return safe_join(REPO_ROOT, relative)


def _write_exclusive(path: Path, payload: bytes) -> dict[str, Any]:
    require(not path.exists(), "HEAD_OUTPUT_OVERWRITE_FORBIDDEN", "HEAD output path already exists", path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    require(not partial.exists(), "HEAD_OUTPUT_PARTIAL_COLLISION", "HEAD output partial already exists", path=str(partial))
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": len(payload), "sha256": sha256_bytes(payload)}


def _error_code(error: Exception) -> str:
    return error.code if isinstance(error, MaterializerError) else type(error).__name__


def validate_head_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = load_json(lock_path)
    require(lock.get("schema") == HEAD_EXECUTION_LOCK_SCHEMA, "HEAD_EXECUTION_LOCK_SCHEMA_DRIFT", "HEAD execution lock schema drift")
    require(lock.get("lock_id") == "TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_R1_EXECUTION_LOCK", "HEAD_EXECUTION_LOCK_IDENTITY_DRIFT", "HEAD execution lock id drift")
    require(lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "HEAD_EXECUTION_NOT_AUTHORIZED", "HEAD execution lock is not authorized/unconsumed")
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "HEAD_EXECUTION_AUTHORITY_DRIFT", "HEAD execution authority drift")
    require(lock.get("output_root") == HEAD_OUTPUT_ROOT and lock.get("output_receipt") == HEAD_RECEIPT_PATH, "HEAD_OUTPUT_PATH_DRIFT", "HEAD output path drift")
    require(lock.get("overwrite") is False and lock.get("rerun") is False, "HEAD_ONE_SHOT_POLICY_DRIFT", "HEAD output namespace must forbid overwrite/rerun")
    required_environment = lock.get("required_environment")
    require(isinstance(required_environment, dict), "HEAD_ENVIRONMENT_MISSING", "HEAD required environment missing")
    for key, expected in required_environment.items():
        require(os.environ.get(key) == str(expected), "HEAD_ENVIRONMENT_DRIFT", "HEAD environment mismatch", key=key)
    actual_argv = [
        Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(),
        "--execution-lock",
        lock_path.relative_to(REPO_ROOT).as_posix(),
    ]
    require(lock.get("argv") == actual_argv, "HEAD_ARGV_DRIFT", "HEAD argv drift", actual=actual_argv)
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_HEAD_BINDINGS), "HEAD_BINDINGS_MISSING", "HEAD binding cardinality drift")
    verified: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        require(isinstance(binding, dict) and set(binding) == {"role", "path", "bytes", "sha256"}, "HEAD_BINDING_FIELDS", "HEAD binding fields drift")
        role, relative = str(binding["role"]), str(binding["path"])
        require(role not in verified and EXPECTED_HEAD_BINDINGS.get(role) == relative, "HEAD_BINDING_PATH", "HEAD binding role/path drift", role=role)
        bound = _repo_path(relative)
        require(bound.is_file() and bound.stat().st_size == binding["bytes"] and sha256_file(bound) == binding["sha256"], "BOUND_HASH_DRIFT", "HEAD binding drift", path=relative)
        verified[role] = dict(binding)
    require(set(verified) == set(EXPECTED_HEAD_BINDINGS), "HEAD_BINDING_ROLE_SET", "HEAD binding role set drift")
    lock["_verified_bindings"] = verified
    return lock


def execute_head_preflight(
    lock_path: Path,
    *,
    head_fn: Callable[[Mapping[str, str], float], Mapping[str, Any]] = production_head,
) -> dict[str, Any]:
    lock = validate_head_execution_lock(lock_path)
    preflight_path = _repo_path(EXPECTED_HEAD_BINDINGS["TRUTH_ONLY_PREFLIGHT_LOCK"])
    authorization_path = _repo_path(EXPECTED_HEAD_BINDINGS["DATA_USE_AUTHORIZATION"])
    preflight = load_json(preflight_path)
    authorization = load_json(authorization_path)
    require(lock.get("required_environment") == preflight.get("required_environment"), "HEAD_ENVIRONMENT_DRIFT", "HEAD environment differs from preflight lock")
    validate_authorization(preflight, authorization, preflight_sha256=sha256_file(preflight_path))
    expected_budget = {
        key: preflight["resource_budget"][key]
        for key in ("maximum_compressed_source_bytes", "head_timeout_seconds", "head_retries", "head_workers", "head_response_body_bytes")
    }
    require(lock.get("resource_budget") == expected_budget, "HEAD_BUDGET_DRIFT", "HEAD execution budget drift")
    output_root = _repo_path(HEAD_OUTPUT_ROOT)
    require(not output_root.exists(), "HEAD_OUTPUT_ROOT_COLLISION", "HEAD output root already exists")
    require(all(not _repo_path(relative).exists() for relative in EXPECTED_ABSENT_ROOTS), "TRUTH_ROOT_PRECONDITION_DRIFT", "source/work/truth/factor root must remain absent during HEAD")
    output_root.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, Any]] = {}
    try:
        start = {
            "schema": "blindassist.taro.o0r.content_length_head_start_receipt.v1",
            "execution_lock_sha256": sha256_file(lock_path),
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "argv": lock["argv"],
            "verified_binding_sha256s": {
                role: row["sha256"] for role, row in sorted(lock["_verified_bindings"].items())
            },
            "head_root_created_consumes_one_shot": True,
            "response_body_bytes_allowed": 0,
        }
        files["start-receipt.json"] = _write_exclusive(
            output_root / "start-receipt.json",
            canonical_json_bytes(start) + b"\n",
        )
        receipt = build_head_receipt(
            preflight,
            preflight_sha256=sha256_file(preflight_path),
            authorization_sha256=sha256_file(authorization_path),
            head_fn=head_fn,
        )
        receipt_bytes = canonical_json_bytes(receipt) + b"\n"
        receipt_file = _write_exclusive(_repo_path(HEAD_RECEIPT_PATH), receipt_bytes)
        files["head-receipt.json"] = receipt_file
        passed = receipt["terminal"] == "TARO_O0R_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED"
        if passed:
            validate_head_receipt(preflight, sha256_file(authorization_path), receipt)
        require(all(not _repo_path(relative).exists() for relative in EXPECTED_ABSENT_ROOTS), "HEAD_SCOPE_VIOLATION", "HEAD execution created a forbidden source/work/truth/factor root")
        result = {
            "schema": "blindassist.taro.o0r.content_length_head_result.v1",
            "terminal": receipt["terminal"],
            "passed": passed,
            "head_receipt": receipt_file,
            "request_count": receipt["asset_count"],
            "available_count": receipt["available_asset_count"],
            "response_body_bytes_read": receipt["response_body_bytes_read"],
            "head_one_shot_consumed": True,
            "source_payload_opened": False,
            "truth_one_shot_consumed": False,
        }
        result_file = _write_exclusive(output_root / "result.json", canonical_json_bytes(result) + b"\n")
        files["result.json"] = result_file
        manifest = {
            "schema": "blindassist.taro.o0r.content_length_head_manifest.v1",
            "files": files,
            "head_root_consumed": True,
        }
        _write_exclusive(output_root / "manifest.json", canonical_json_bytes(manifest) + b"\n")
        return result
    except Exception as error:
        failure = {
            "schema": "blindassist.taro.o0r.content_length_head_failure.v1",
            "terminal": "TARO_O0R_HEAD_EXECUTION_FAILED_ONE_SHOT_CONSUMED",
            "passed": False,
            "failure_code": _error_code(error),
            "message": str(error),
            "head_one_shot_consumed": True,
            "response_body_bytes_allowed": 0,
            "source_payload_opened": False,
            "truth_one_shot_consumed": False,
        }
        try:
            files["failure.json"] = _write_exclusive(
                output_root / "failure.json",
                canonical_json_bytes(failure) + b"\n",
            )
            _write_exclusive(
                output_root / "manifest.json",
                canonical_json_bytes(
                    {
                        "schema": "blindassist.taro.o0r.content_length_head_manifest.v1",
                        "files": files,
                        "head_root_consumed": True,
                    }
                )
                + b"\n",
            )
        except Exception:
            pass
        raise MaterializerError(
            "HEAD_EXECUTION_ONE_SHOT_CONSUMED",
            "HEAD execution failed after its exclusive root was created",
            failure_code=_error_code(error),
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute_head_preflight(args.execution_lock.resolve())
    except Exception as error:
        code = error.code if isinstance(error, MaterializerError) else type(error).__name__
        consumed = _repo_path(HEAD_OUTPUT_ROOT).exists()
        status = "HEAD_ONE_SHOT_CONSUMED" if consumed else "HEAD_NOT_STARTED"
        print(json.dumps({"status": status, "error_code": code, "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"]}, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
