#!/usr/bin/env python3
"""Validate the synthetic-only TARO O0R truth-materializer implementation lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.taro_o0r_truth_materializer_runtime.materializer import (
    load_json,
    sha256_file,
    validate_authorization,
)


LOCK_SCHEMA = "blindassist.taro.o0r.truth_materializer_implementation_lock.v1"
LOCK_ID = "TARO_O0R_ARKITSCENES_TRUTH_ONLY_MATERIALIZER_IMPLEMENTATION_LOCK"
EXPECTED_BINDINGS = {
    "SOURCE_AND_ADAPTER_CONTRACT": "docs/research/taro/TARO_O0R_ARKITSCENES_SOURCE_AND_ADAPTER_CONTRACT_LOCK_2026-08-10.json",
    "SOURCE_ADAPTER_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_SOURCE_ADAPTER_IMPLEMENTATION_LOCK_2026-08-10.json",
    "SOURCE_ADAPTER_RUNTIME": "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py",
    "TRUTH_ONLY_PREFLIGHT_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK_2026-08-10.json",
    "DATA_USE_AUTHORIZATION": "docs/research/taro/TARO_O0R_ARKITSCENES_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-10.json",
    "MATERIALIZER_AMENDMENT_JSON": "docs/research/taro/TARO_O0R_ARKITSCENES_MATERIALIZER_INPUT_AND_PERSISTENCE_AMENDMENT_LOCK_2026-08-10.json",
    "MATERIALIZER_AMENDMENT_MD": "docs/research/taro/TARO_O0R_ARKITSCENES_MATERIALIZER_INPUT_AND_PERSISTENCE_AMENDMENT_LOCK_2026-08-10.md",
    "MODULE_README": "scripts/research/taro_o0r_truth_materializer_runtime/README.md",
    "MATERIALIZER_RUNTIME": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "HEAD_RUNNER": "scripts/research/taro_o0r_truth_materializer_runtime/run_head_preflight.py",
    "TRUTH_RUNNER": "scripts/research/taro_o0r_truth_materializer_runtime/run_truth_only.py",
    "MATERIALIZER_TESTS": "scripts/research/taro_o0r_truth_materializer_runtime/test_materializer.py",
    "IMPLEMENTATION_VALIDATOR": "scripts/research/taro_o0r_truth_materializer_runtime/validate_implementation_lock.py",
    "IMPLEMENTATION_VALIDATOR_TESTS": "scripts/research/taro_o0r_truth_materializer_runtime/test_validate_implementation_lock.py",
}
EXPECTED_ROOTS = {
    "HEAD_EVIDENCE": "artifacts.local/evidence/taro/o0r-arkitscenes-head-r0",
    "SOURCE": "artifacts.local/datasets/taro/o0r-arkitscenes-source-adapter-r0",
    "WORK": "artifacts.local/work/taro/o0r-arkitscenes-source-adapter-r0",
    "TRUTH_EVIDENCE": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r0",
    "O0R_EVIDENCE_SEALED": "artifacts.local/evidence/taro/o0r-arkitscenes-factor-headroom-r0",
}
EXPECTED_AUTHORITY = {
    "materializer_implementation_lock": True,
    "synthetic_tests": True,
    "head_or_network": False,
    "source_payload_open": False,
    "truth_materialization": False,
    "depthart_inference": False,
    "factorial_execution": False,
    "training": False,
    "device": False,
    "product": False,
    "safety": False,
}


def validate_lock(lock: dict[str, Any], repo_root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(lock.get("schema") == LOCK_SCHEMA, "schema drift")
    check(lock.get("lock_id") == LOCK_ID, "lock id drift")
    check(lock.get("status") == "IMPLEMENTATION_LOCK_PASS", "status must be IMPLEMENTATION_LOCK_PASS")
    check(lock.get("research_mode") == "WILD_LAB", "research mode drift")
    check(lock.get("scientific_status") == "NOT_RUN", "scientific status must remain NOT_RUN")
    check(lock.get("execution_authority") == EXPECTED_AUTHORITY, "execution authority drift")
    check(lock.get("head_requests_executed") == 0, "HEAD requests must remain zero")
    check(lock.get("source_payloads_opened") == 0, "source payloads must remain unopened")
    check(lock.get("truth_frames_materialized") == 0, "truth frames must remain zero")
    check(lock.get("model_outputs_produced") == 0, "model outputs must remain zero")

    bindings = lock.get("bindings")
    check(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "binding cardinality drift")
    seen: set[str] = set()
    if isinstance(bindings, list):
        for binding in bindings:
            if not isinstance(binding, dict) or set(binding) != {"role", "path", "bytes", "sha256"}:
                errors.append("binding fields drift")
                continue
            role, relative = str(binding["role"]), str(binding["path"])
            check(role not in seen and EXPECTED_BINDINGS.get(role) == relative, f"binding role/path drift: {role}")
            seen.add(role)
            path = repo_root / relative
            check(path.is_file(), f"bound file missing: {relative}")
            if path.is_file():
                check(path.stat().st_size == binding["bytes"] and sha256_file(path) == binding["sha256"], f"bound file hash/bytes drift: {relative}")
    check(seen == set(EXPECTED_BINDINGS), "binding role set drift")

    roots = lock.get("exclusive_roots")
    check(isinstance(roots, list) and len(roots) == len(EXPECTED_ROOTS), "exclusive root cardinality drift")
    root_map: dict[str, dict[str, Any]] = {}
    if isinstance(roots, list):
        for row in roots:
            if isinstance(row, dict) and isinstance(row.get("role"), str):
                root_map[row["role"]] = row
    check(set(root_map) == set(EXPECTED_ROOTS), "exclusive root role set drift")
    for role, relative in EXPECTED_ROOTS.items():
        row = root_map.get(role, {})
        check(row == {"role": role, "path": relative, "exists": False}, f"exclusive root receipt drift: {role}")
        check(not (repo_root / relative).exists(), f"exclusive root unexpectedly exists: {relative}")

    tests = lock.get("synthetic_validation")
    check(isinstance(tests, dict) and tests.get("passed") is True, "synthetic validation must pass")
    if isinstance(tests, dict):
        check(tests.get("tests_run") == 25 and tests.get("tests_passed") == 25 and tests.get("failures") == 0 and tests.get("errors") == 0, "focused test counts drift")
        check(tests.get("network_requests") == 0 and tests.get("source_payloads_opened") == 0, "synthetic validation overclaims source/network")

    interfaces = lock.get("implemented_interfaces")
    required_interfaces = {
        "exact_72_url_head_receipt",
        "bounded_get_and_container_integrity",
        "all_exact_frame_denominators",
        "fit_before_eval_decode_firewall",
        "per_query_bound_uncertainty_lookup",
        "original_member_provenance_envelope",
        "content_addressed_ndarray_reload_gate",
        "atomic_truth_one_shot_writer",
        "complete_future_truth_runner",
        "trusted_artifacts_local_junction_containment",
    }
    check(isinstance(interfaces, list) and set(interfaces) == required_interfaces, "implemented interface set drift")
    successor = lock.get("unique_successor")
    check(
        isinstance(successor, dict)
        and successor.get("id") == "TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_EXECUTION_LOCK_ATTEMPT_02"
        and successor.get("execution_authority") is False
        and successor.get("head_requests_allowed_now") is False,
        "unique successor drift",
    )

    try:
        preflight_path = repo_root / EXPECTED_BINDINGS["TRUTH_ONLY_PREFLIGHT_LOCK"]
        authorization_path = repo_root / EXPECTED_BINDINGS["DATA_USE_AUTHORIZATION"]
        validate_authorization(
            load_json(preflight_path),
            load_json(authorization_path),
            preflight_sha256=sha256_file(preflight_path),
        )
    except Exception as error:
        errors.append(f"authorization/preflight binding invalid: {type(error).__name__}: {error}")

    materializer_path = repo_root / EXPECTED_BINDINGS["MATERIALIZER_RUNTIME"]
    if materializer_path.is_file():
        source = materializer_path.read_text(encoding="utf-8")
        for token in (
            "def derive_query_uncertainty_lookup(",
            "def validate_bound_source_frame_envelope(",
            "def package_content_addressed_artifact(",
            "def hydrate_content_addressed_artifact(",
            "def write_content_addressed_artifact(",
        ):
            check(token in source, f"materializer implementation token missing: {token}")
        check("def representative_uncertainty_observation(" not in source, "frame-level shared uncertainty lookup remains present")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock = load_json(args.lock.resolve())
    except Exception as error:
        print(json.dumps({"status": "INVALID", "errors": [f"{type(error).__name__}: {error}"]}, ensure_ascii=False, indent=2))
        return 2
    errors = validate_lock(lock)
    print(json.dumps({"status": "VALID" if not errors else "INVALID", "error_count": len(errors), "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
