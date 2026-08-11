#!/usr/bin/env python3
"""Validate the frozen TARO R6 prospective factor-runtime implementation lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as runtime


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / "docs/research/taro/TARO_O0R_R6_PROSPECTIVE_FACTOR_RUNTIME_IMPLEMENTATION_LOCK_2026-08-11.json"
SCHEMA = "blindassist.taro.o0r.r6_prospective_factor_runtime_implementation_lock.v1"
LOCK_ID = "TARO_O0R_R6_PROSPECTIVE_FACTOR_RUNTIME_IMPLEMENTATION_LOCK"
SUCCESSOR = "TARO_O0R_R6_PROSPECTIVE_FACTOR_RUNTIME_FORMATION_REPLAY_LOCK"
PROTOCOL_BINDING = ("docs/research/taro/TARO_O0R_R6_FACTOR_POLICY_ADOPTION_AND_PROSPECTIVE_RUNTIME_PROTOCOL_LOCK_2026-08-11.json", 7452, "B0A33DAB1532C3E3737BD213BDBB3933ACCF4D1775E9FCE8AB3F211625400296")
REPAIR_BINDING = ("docs/research/taro/TARO_O0R_R6_PROSPECTIVE_RUNTIME_QUERY_FRAME_PRE_IMPLEMENTATION_REPAIR_2026-08-11.json", 2028, "08B56A3C7BD673958DCCE9CAB07608C4251A09514B93659790029A0DF48DDAE2")
IMPLEMENTATION_BINDINGS = {
    "PROSPECTIVE_FACTOR_RUNTIME": ("scripts/research/taro_o0r_candidate_scale_runtime/prospective_factor_runtime.py", 37207, "FCDAEE3D5D343E70D1BAE87CFADF6D4B455D269BF280934F24D8A90AD5AA554D"),
    "PROSPECTIVE_FACTOR_RUNTIME_TEST": ("scripts/research/taro_o0r_candidate_scale_runtime/test_prospective_factor_runtime.py", 6167, "D3FCE7661415AD6C7A0A822D963AB26FD313B85F4B9FE2FC86E804E42E60B99B"),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _require(errors: list[str], condition: bool, code: str) -> None:
    if not condition:
        errors.append(code)


def _tuple(value: Any) -> tuple[Any, Any, Any]:
    return (value.get("path"), value.get("bytes"), value.get("sha256")) if isinstance(value, Mapping) else (None, None, None)


def _binding_map(value: Any, errors: list[str]) -> dict[str, tuple[Any, Any, Any]]:
    output = {}
    if not isinstance(value, list):
        errors.append("R6_RUNTIME_IMPL_BINDINGS_NOT_LIST")
        return output
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"role", "path", "bytes", "sha256"} or row.get("role") in output:
            errors.append("R6_RUNTIME_IMPL_BINDING_ROW_INVALID")
            continue
        output[str(row["role"])] = (row["path"], row["bytes"], row["sha256"])
    return output


def validate_payload(payload: Mapping[str, Any], *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema", "lock_id", "date", "research_mode", "status", "protocol_binding", "query_frame_repair_binding",
        "implementation_bindings", "frozen_implementation", "synthetic_test_receipt", "known_boundaries",
        "execution_authority", "unique_successor", "claim_ceiling",
    }
    _require(errors, set(payload) == expected, "R6_RUNTIME_IMPL_TOP_LEVEL_KEY_SET_DRIFT")
    _require(errors, payload.get("schema") == SCHEMA and payload.get("lock_id") == LOCK_ID, "R6_RUNTIME_IMPL_IDENTITY_DRIFT")
    _require(errors, payload.get("status") == "PROSPECTIVE_FACTOR_RUNTIME_IMPLEMENTATION_FROZEN_REAL_EXECUTION_FALSE", "R6_RUNTIME_IMPL_STATUS_DRIFT")
    _require(errors, payload.get("unique_successor") == SUCCESSOR, "R6_RUNTIME_IMPL_SUCCESSOR_DRIFT")
    _require(errors, _tuple(payload.get("protocol_binding")) == PROTOCOL_BINDING, "R6_RUNTIME_IMPL_PROTOCOL_BINDING_DRIFT")
    _require(errors, _tuple(payload.get("query_frame_repair_binding")) == REPAIR_BINDING, "R6_RUNTIME_IMPL_REPAIR_BINDING_DRIFT")
    _require(errors, _binding_map(payload.get("implementation_bindings"), errors) == IMPLEMENTATION_BINDINGS, "R6_RUNTIME_IMPL_CODE_BINDING_DRIFT")

    implementation = payload.get("frozen_implementation", {})
    _require(errors, implementation.get("runtime_id") == runtime.RUNTIME_ID and implementation.get("policy_id") == runtime.POLICY_ID, "R6_RUNTIME_IMPL_ALGORITHM_DRIFT")
    _require(errors, implementation.get("public_builder") == "build_prospective_factor_bundle" and implementation.get("public_validator") == "validate_prospective_factor_bundle" and implementation.get("public_builder_has_faro_truth_task_metric_or_outcome_argument") is False, "R6_RUNTIME_IMPL_PUBLIC_API_DRIFT")
    _require(errors, implementation.get("support_boundary_owner") == "PHASE_A_SELECTED_SUPPORT_BOUNDARY_COMPONENT" and implementation.get("query_clearance_owner") == "R1_BASELINE", "R6_RUNTIME_IMPL_FACTOR_OWNER_DRIFT")
    _require(errors, implementation.get("query_frame_owner_rule") == "DIRECT_APPLE_SUPPORT_ELSE_R1_BASELINE_ELSE_UNAVAILABLE" and implementation.get("unavailable_query_frame_rule") == "RETAIN_NINE_UNKNOWN_SLOTS", "R6_RUNTIME_IMPL_QUERY_FRAME_DRIFT")
    _require(errors, implementation.get("highres_shape_hw") == [1440, 1920] and implementation.get("query_slot_count") == 9 and implementation.get("source_surface_pixel_ids_sha256_required") is True, "R6_RUNTIME_IMPL_SURFACE_DRIFT")
    for field in ("uncertainty_attached", "deterministic_reducer_executed", "final_state_authorized"):
        _require(errors, implementation.get(field) is False, f"R6_RUNTIME_IMPL_AUTHORITY_DRIFT:{field}")
    _require(errors, implementation.get("r6_untouched_parent_ids_rejected") is True and implementation.get("summary_inputs_canonicalized_before_aggregation") is True, "R6_RUNTIME_IMPL_GOVERNANCE_DRIFT")

    tests = payload.get("synthetic_test_receipt", {})
    _require(errors, tests.get("focused_test_count") == 8 and tests.get("focused_test_failures") == 0 and tests.get("synthetic_parent_count") == 1 and tests.get("real_parent_count") == 0, "R6_RUNTIME_IMPL_TEST_COUNT_DRIFT")
    _require(errors, tests.get("query_slot_count") == 9 and tests.get("support_evaluable_query_count") == 9 and tests.get("boundary_evaluable_query_count") == 9 and tests.get("query_clearance_evaluable_query_count") == 6 and tests.get("query_clearance_unknown_query_count") == 3, "R6_RUNTIME_IMPL_SYNTHETIC_RESULT_DRIFT")
    _require(errors, tests.get("source_scale_metric") == 1.27110935399 and tests.get("baseline_support_height_m") == 0.96000000529 and tests.get("direct_support_height_m") == 1.200056157011, "R6_RUNTIME_IMPL_SYNTHETIC_METRIC_DRIFT")
    for field in ("deterministic_replay_equal", "candidate_mutation_rejected", "wrong_factor_depth_rejected", "forbidden_parent_rejected"):
        _require(errors, tests.get(field) is True, f"R6_RUNTIME_IMPL_TEST_EVIDENCE_DRIFT:{field}")
    _require(errors, tests.get("all_support_unavailable_fixture_retained_unknown_slots") == 9, "R6_RUNTIME_IMPL_UNKNOWN_RETENTION_DRIFT")

    boundaries = payload.get("known_boundaries", {})
    _require(errors, all(boundaries.get(field) is True for field in ("upstream_source_receipt_and_candidate_hashes_must_be_verified_by_future_runner", "source_surface_accuracy_not_yet_scored", "source_surface_specific_uncertainty_not_yet_frozen", "final_query_state_not_available", "formation_replay_not_run")), "R6_RUNTIME_IMPL_BOUNDARY_DRIFT")
    authority = payload.get("execution_authority", {})
    _require(errors, authority.get("implementation_complete") is True and authority.get("synthetic_tests_complete") is True and authority.get("r6_untouched_evidence_read_for_implementation") is False, "R6_RUNTIME_IMPL_COMPLETION_DRIFT")
    for field in ("formation_replay", "new_source_download", "model_inference", "truth_scoring", "training", "network", "device", "product", "safety"):
        _require(errors, authority.get(field) is False, f"R6_RUNTIME_IMPL_EXECUTION_AUTHORITY_DRIFT:{field}")

    repair = payload.get("query_frame_repair_binding", {})
    if verify_files:
        for relative, size, digest in (PROTOCOL_BINDING, REPAIR_BINDING, *IMPLEMENTATION_BINDINGS.values()):
            path = REPO_ROOT / relative
            _require(errors, path.is_file(), f"R6_RUNTIME_IMPL_FILE_MISSING:{relative}")
            if path.is_file():
                _require(errors, path.stat().st_size == size, f"R6_RUNTIME_IMPL_FILE_BYTES_DRIFT:{relative}")
                _require(errors, _sha(path) == digest, f"R6_RUNTIME_IMPL_FILE_HASH_DRIFT:{relative}")
        repair_path = REPO_ROOT / str(repair.get("path", ""))
        if repair_path.is_file():
            repair_payload = json.loads(repair_path.read_text(encoding="utf-8"))
            _require(errors, repair_payload.get("effective_query_frame_rule") == {"when_phase_a_direct_support_available": "DIRECT_APPLE_SUPPORT", "otherwise_when_source_baseline_support_available": "R1_BASELINE", "otherwise": "UNAVAILABLE_RETAIN_NINE_UNKNOWN_SLOTS", "unavailable_reason_code": "SOURCE_QUERY_FRAME_UNAVAILABLE", "query_grid_order": "LATERAL_MAJOR_THEN_YAW_ASCENDING", "query_slot_count": 9, "outcome_dependent_reselection": False}, "R6_RUNTIME_IMPL_REPAIR_CONTENT_DRIFT")
            _require(errors, repair_payload.get("repair_timing_receipt", {}).get("real_frame_runtime_execution_count") == 0 and repair_payload.get("repair_timing_receipt", {}).get("implementation_lock_exists") is False, "R6_RUNTIME_IMPL_REPAIR_TIMING_DRIFT")
    return errors


def validate_file(path: Path = DEFAULT_LOCK, *, verify_files: bool = True) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"R6_RUNTIME_IMPL_READ_FAILED:{error}"]
    return validate_payload(payload, verify_files=verify_files) if isinstance(payload, Mapping) else ["R6_RUNTIME_IMPL_NOT_OBJECT"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--skip-file-verification", action="store_true")
    args = parser.parse_args(argv)
    errors = validate_file(args.lock, verify_files=not args.skip_file_verification)
    result = {"schema": "blindassist.taro.o0r.r6_prospective_factor_runtime_implementation_lock_validation.v1", "passed": not errors, "error_count": len(errors), "errors": errors, "terminal": "TARO_O0R_R6_PROSPECTIVE_FACTOR_RUNTIME_IMPLEMENTATION_LOCK_VALID" if not errors else "TARO_O0R_R6_PROSPECTIVE_FACTOR_RUNTIME_IMPLEMENTATION_LOCK_INVALID"}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
