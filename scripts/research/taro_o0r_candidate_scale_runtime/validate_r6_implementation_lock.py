#!/usr/bin/env python3
"""Validate the frozen TARO R6 factor-compositor implementation lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import r6_factor_split as r6


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_R6_FACTOR_SPLIT_IMPLEMENTATION_LOCK_2026-08-11.json"
SCHEMA = "blindassist.taro.o0r.r6_factor_split_implementation_lock.v1"
LOCK_ID = "TARO_O0R_R6_FACTOR_SPLIT_IMPLEMENTATION_LOCK"
SUCCESSOR = "TARO_O0R_R6_UNTOUCHED_COHORT_AND_DATA_USE_LOCK"
EXPECTED_BINDINGS = {
    "R6_FACTOR_COMPOSITOR": ("scripts/research/taro_o0r_candidate_scale_runtime/r6_factor_split.py", 26354, "9B96ED4C34B00EEB59D0DFEB55FC1AC400629F97AC3777287E07F8C87A570236"),
    "R6_FORMATION_REPLAY_RUNNER": ("scripts/research/taro_o0r_candidate_scale_runtime/run_r6_factor_split_implementation_replay.py", 6301, "DAB64D67BE42433B106154E9D72FB3CEC07892464AE9A78279CE31D913FE3CD3"),
    "R6_FACTOR_COMPOSITOR_TEST": ("scripts/research/taro_o0r_candidate_scale_runtime/test_r6_factor_split.py", 5718, "A111192322429E322F1124408859C088E7F0FF722148CB45E4368E8A11F8031B"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _require(errors: list[str], condition: bool, code: str) -> None:
    if not condition:
        errors.append(code)


def _bindings(value: Any, errors: list[str]) -> dict[str, tuple[Any, Any, Any]]:
    result = {}
    if not isinstance(value, list):
        errors.append("R6_IMPL_BINDINGS_NOT_LIST")
        return result
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"role", "path", "bytes", "sha256"} or row.get("role") in result:
            errors.append("R6_IMPL_BINDING_ROW_INVALID")
            continue
        result[row["role"]] = (row["path"], row["bytes"], row["sha256"])
    return result


def validate_payload(payload: Mapping[str, Any], *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    expected_keys = {
        "schema", "lock_id", "date", "research_mode", "status", "protocol_binding",
        "implementation_scope", "frozen_algorithm", "implementation_bindings",
        "formation_replay_receipt", "test_receipt", "execution_authority",
        "unique_successor", "claim_ceiling",
    }
    _require(errors, set(payload) == expected_keys, "R6_IMPL_TOP_LEVEL_KEY_SET_DRIFT")
    _require(errors, payload.get("schema") == SCHEMA and payload.get("lock_id") == LOCK_ID, "R6_IMPL_IDENTITY_DRIFT")
    _require(errors, payload.get("status") == "FACTOR_COMPOSITOR_IMPLEMENTATION_FROZEN_UNTOUCHED_EXECUTION_FALSE", "R6_IMPL_STATUS_DRIFT")
    _require(errors, payload.get("unique_successor") == SUCCESSOR, "R6_IMPL_SUCCESSOR_DRIFT")

    protocol = payload.get("protocol_binding", {})
    _require(errors, protocol == {"path": "docs/research/taro/TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_PARENT_CONFIRMATION_PROTOCOL_LOCK_2026-08-11.json", "bytes": 4570, "sha256": r6.PROTOCOL_LOCK_SHA256}, "R6_IMPL_PROTOCOL_BINDING_DRIFT")
    bindings = _bindings(payload.get("implementation_bindings"), errors)
    _require(errors, bindings == EXPECTED_BINDINGS, "R6_IMPL_CODE_BINDING_SET_DRIFT")

    algorithm = payload.get("frozen_algorithm", {})
    _require(errors, algorithm.get("policy_id") == r6.POLICY_ID, "R6_IMPL_POLICY_DRIFT")
    _require(errors, algorithm.get("support_owner") == algorithm.get("boundary_owner") == "PHASE_A_SELECTED_SUPPORT_BOUNDARY_COMPONENT", "R6_IMPL_SUPPORT_BOUNDARY_OWNER_DRIFT")
    _require(errors, algorithm.get("query_clearance_owner") == "ALWAYS_R1_BASELINE", "R6_IMPL_QUERY_OWNER_DRIFT")
    _require(errors, algorithm.get("owner_selection_fields_read") == {"SUPPORT_BOUNDARY": ["source_support_available"], "QUERY_CLEARANCE": []}, "R6_IMPL_SELECTION_FIELD_DRIFT")
    _require(errors, algorithm.get("outcome_dependent_reselection") is False and algorithm.get("learned_parameters") == algorithm.get("thresholds") == algorithm.get("training_steps") == 0, "R6_IMPL_RESELECTION_OR_FITTING_DRIFT")

    replay = payload.get("formation_replay_receipt", {})
    _require(errors, replay.get("terminal") == "TARO_O0R_R6_FACTOR_SPLIT_IMPLEMENTATION_REPLAY_PASS", "R6_IMPL_REPLAY_TERMINAL_DRIFT")
    _require(errors, replay.get("parent_count") == 8 and replay.get("physical_frame_count") == 211 and replay.get("factor_component_count") == replay.get("composite_query_count") == 1899, "R6_IMPL_REPLAY_COUNT_DRIFT")
    _require(errors, replay.get("all_gate_landscape_would_pass") is True and replay.get("confirmation_eligible") is False and replay.get("promotion_allowed") is False, "R6_IMPL_REPLAY_CLAIM_DRIFT")
    _require(errors, replay.get("training_steps") == replay.get("model_inference_calls") == replay.get("network_requests") == 0, "R6_IMPL_REPLAY_SIDE_EFFECT_DRIFT")

    tests = payload.get("test_receipt", {})
    _require(
        errors,
        tests.get("focused_test_count") == 15
        and tests.get("focused_test_failures") == 0
        and tests.get("lock_validator_mutation_test_count") == 4
        and tests.get("lock_validator_mutation_test_failures") == 0,
        "R6_IMPL_TEST_RECEIPT_DRIFT",
    )
    authority = payload.get("execution_authority", {})
    _require(errors, authority.get("factor_compositor_implementation_complete") is True and authority.get("formation_replay_complete") is True, "R6_IMPL_COMPLETION_DRIFT")
    for field in ("untouched_cohort_frozen", "untouched_data_use_authority", "untouched_model_execution", "untouched_truth_scoring", "training", "network", "device", "product", "safety"):
        _require(errors, authority.get(field) is False, f"R6_IMPL_AUTHORITY_DRIFT:{field}")

    if verify_files:
        for relative, size, expected_hash in [tuple(protocol.values()), *EXPECTED_BINDINGS.values()]:
            path = REPO_ROOT / relative
            _require(errors, path.is_file(), f"R6_IMPL_FILE_MISSING:{relative}")
            if path.is_file():
                _require(errors, path.stat().st_size == size, f"R6_IMPL_FILE_BYTES_DRIFT:{relative}")
                _require(errors, _sha256(path) == expected_hash, f"R6_IMPL_FILE_HASH_DRIFT:{relative}")
        root = REPO_ROOT / str(replay.get("output_root", ""))
        result_path = root / "result.json"
        manifest_path = root / "manifest.json"
        _require(errors, result_path.is_file() and manifest_path.is_file(), "R6_IMPL_REPLAY_FILES_MISSING")
        if result_path.is_file():
            _require(errors, result_path.stat().st_size == replay.get("result_file_bytes") and _sha256(result_path) == replay.get("result_file_sha256"), "R6_IMPL_REPLAY_RESULT_DRIFT")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            _require(errors, result.get("content_sha256") == replay.get("result_content_sha256") and result.get("implementation_module_sha256") == EXPECTED_BINDINGS["R6_FACTOR_COMPOSITOR"][2], "R6_IMPL_REPLAY_CONTENT_DRIFT")
        if manifest_path.is_file():
            _require(errors, manifest_path.stat().st_size == replay.get("manifest_file_bytes") and _sha256(manifest_path) == replay.get("manifest_file_sha256"), "R6_IMPL_REPLAY_MANIFEST_DRIFT")
    return errors


def validate_file(path: Path = DEFAULT_LOCK_PATH, *, verify_files: bool = True) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"R6_IMPL_LOCK_READ_FAILED:{error}"]
    return validate_payload(payload, verify_files=verify_files) if isinstance(payload, Mapping) else ["R6_IMPL_LOCK_NOT_OBJECT"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--skip-file-verification", action="store_true")
    args = parser.parse_args(argv)
    errors = validate_file(args.lock, verify_files=not args.skip_file_verification)
    print(json.dumps({"schema": "blindassist.taro.o0r.r6_implementation_lock_validation_result.v1", "passed": not errors, "error_count": len(errors), "errors": errors, "terminal": "TARO_O0R_R6_IMPLEMENTATION_LOCK_VALID" if not errors else "TARO_O0R_R6_IMPLEMENTATION_LOCK_INVALID"}, sort_keys=True, separators=(",", ":")))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
