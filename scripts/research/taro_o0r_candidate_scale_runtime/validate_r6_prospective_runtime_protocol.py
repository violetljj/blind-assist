#!/usr/bin/env python3
"""Validate the TARO R6 prospective factor-runtime protocol lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / "docs/research/taro/TARO_O0R_R6_FACTOR_POLICY_ADOPTION_AND_PROSPECTIVE_RUNTIME_PROTOCOL_LOCK_2026-08-11.json"
SCHEMA = "blindassist.taro.o0r.r6_factor_policy_prospective_runtime_protocol_lock.v1"
LOCK_ID = "TARO_O0R_R6_FACTOR_POLICY_ADOPTION_AND_PROSPECTIVE_RUNTIME_PROTOCOL_LOCK"
SUCCESSOR = "TARO_O0R_R6_PROSPECTIVE_FACTOR_RUNTIME_IMPLEMENTATION_LOCK"
R6_UNTOUCHED_PARENTS = ["423306", "435329", "466652", "467175", "467312", "469650", "469830", "470439"]
EXPECTED_BINDINGS = {
    "R6_FACTOR_SPLIT_PROTOCOL": ("docs/research/taro/TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_PARENT_CONFIRMATION_PROTOCOL_LOCK_2026-08-11.json", 4570, "5F2802F2585861F4D2D1EB002D1AFA7050278CBD33732F665DB6AF9CA32A101C"),
    "R6_FACTOR_SPLIT_IMPLEMENTATION_LOCK": ("docs/research/taro/TARO_O0R_R6_FACTOR_SPLIT_IMPLEMENTATION_LOCK_2026-08-11.json", 5026, "34D1C30193183F8406D5A4CA5EF7598E7EE933B4D62008500A70523A5EE3C90B"),
    "R6_UNTOUCHED_RESULT_DOCUMENT": ("docs/research/taro/TARO_O0R_R6_UNTOUCHED_CONFIRMATION_RESULT_2026-08-11.md", 4201, "3A1A925FCAB171BF4D7C1209BE4FAAFAD146912A70907AB80DDDD0795219B588"),
    "R6_UNTOUCHED_RESULT": ("artifacts.local/evidence/taro/o0r-r6-untouched-confirmation-r0/result.json", 1214, "F0D946EAF77D21E940E5B5E4AA678C3ED5703A34526095D3E0687FA2D07C24AB"),
    "R6_UNTOUCHED_SUMMARY": ("artifacts.local/evidence/taro/o0r-r6-untouched-confirmation-r0/summary.json", 2917, "EDBE4FD26C59DA455D5C50876459F1E9EDD54D6372F47913534E6A4243F04E38"),
    "R6_UNTOUCHED_MANIFEST": ("artifacts.local/evidence/taro/o0r-r6-untouched-confirmation-r0/manifest.json", 145896, "B41A6BE7AA992F2DE37C4498139CFE20E22CD2F82C7B3109395417960A902B4E"),
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


def _binding_map(value: Any, errors: list[str]) -> dict[str, tuple[Any, Any, Any]]:
    rows: dict[str, tuple[Any, Any, Any]] = {}
    if not isinstance(value, list):
        errors.append("R6_RUNTIME_PROTOCOL_BINDINGS_NOT_LIST")
        return rows
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"role", "path", "bytes", "sha256"} or row.get("role") in rows:
            errors.append("R6_RUNTIME_PROTOCOL_BINDING_ROW_INVALID")
            continue
        rows[str(row["role"])] = (row["path"], row["bytes"], row["sha256"])
    return rows


def validate_payload(payload: Mapping[str, Any], *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema", "lock_id", "date", "research_mode", "status", "predecessor_bindings",
        "adopted_policy", "interface_seam_closed_by_this_protocol", "source_defined_geometry",
        "factor_runtime_rules", "phase_firewall", "data_roles", "implementation_acceptance",
        "uncertainty_and_reducer_boundary", "execution_authority", "unique_successor", "claim_ceiling",
    }
    _require(errors, set(payload) == expected, "R6_RUNTIME_PROTOCOL_TOP_LEVEL_KEY_SET_DRIFT")
    _require(errors, payload.get("schema") == SCHEMA and payload.get("lock_id") == LOCK_ID, "R6_RUNTIME_PROTOCOL_IDENTITY_DRIFT")
    _require(errors, payload.get("status") == "PROTOCOL_FROZEN_IMPLEMENTATION_ALLOWED_EXECUTION_NOT_AUTHORIZED", "R6_RUNTIME_PROTOCOL_STATUS_DRIFT")
    _require(errors, payload.get("unique_successor") == SUCCESSOR, "R6_RUNTIME_PROTOCOL_SUCCESSOR_DRIFT")
    bindings = _binding_map(payload.get("predecessor_bindings"), errors)
    _require(errors, bindings == EXPECTED_BINDINGS, "R6_RUNTIME_PROTOCOL_BINDING_SET_DRIFT")

    policy = payload.get("adopted_policy", {})
    _require(errors, policy.get("policy_id") == "R5_SELECTED_SUPPORT_BOUNDARY_PLUS_ALWAYS_R1_QUERY_CLEARANCE_V1", "R6_RUNTIME_PROTOCOL_POLICY_DRIFT")
    _require(errors, policy.get("support_owner") == policy.get("boundary_owner") == "PHASE_A_SELECTED_SUPPORT_BOUNDARY_COMPONENT", "R6_RUNTIME_PROTOCOL_FACTOR_OWNER_DRIFT")
    _require(errors, policy.get("query_clearance_owner") == "R1_BASELINE", "R6_RUNTIME_PROTOCOL_QUERY_OWNER_DRIFT")
    _require(errors, policy.get("support_boundary_selection_fields") == ["source_support_available"] and policy.get("query_clearance_selection_fields") == [], "R6_RUNTIME_PROTOCOL_SELECTION_FIELD_DRIFT")
    _require(errors, policy.get("outcome_dependent_reselection") is False and policy.get("learned_parameters") == policy.get("new_decision_thresholds") == 0, "R6_RUNTIME_PROTOCOL_TUNING_DRIFT")

    seam = payload.get("interface_seam_closed_by_this_protocol", {})
    _require(errors, seam.get("old_comparison_surface") == "FARO_TRUTH_COMMON_SUPPORT" and seam.get("old_surface_is_runtime_eligible") is False, "R6_RUNTIME_PROTOCOL_OLD_SURFACE_DRIFT")
    _require(errors, seam.get("new_runtime_surface") == "FACTOR_OWNER_DEPTH_SOURCE_DEFINED_LOCAL_SURFACE_V1", "R6_RUNTIME_PROTOCOL_NEW_SURFACE_DRIFT")
    for field in ("truth_defined_pixel_ids_allowed", "truth_defined_local_valid_fraction_allowed", "faro_or_task_metric_arguments_allowed_in_public_runtime_api"):
        _require(errors, seam.get(field) is False, f"R6_RUNTIME_PROTOCOL_TRUTH_SURFACE_DRIFT:{field}")

    geometry = payload.get("source_defined_geometry", {})
    _require(errors, geometry.get("highres_shape_hw") == [1440, 1920] and geometry.get("metric_depth_range_m") == [0.25, 6.0], "R6_RUNTIME_PROTOCOL_RASTER_DRIFT")
    _require(errors, geometry.get("boundary_and_query_unprojection_stride") == 1 and geometry.get("support_fit_unprojection_stride") == 4, "R6_RUNTIME_PROTOCOL_STRIDE_DRIFT")
    _require(errors, geometry.get("support_minimum_points") == 256 and geometry.get("support_minimum_fraction") == 0.02 and geometry.get("support_residual_tolerance_m") == 0.08 and geometry.get("support_maximum_slope_degrees") == 20.0, "R6_RUNTIME_PROTOCOL_SUPPORT_GATE_DRIFT")
    _require(errors, geometry.get("minimum_forward_m") == 0.2 and geometry.get("horizon_m") == 2.0 and geometry.get("capsule_radius_m") == 0.3, "R6_RUNTIME_PROTOCOL_QUERY_GEOMETRY_DRIFT")
    _require(errors, geometry.get("frame_geometry_built_once_then_shared_by_nine_queries") is True and geometry.get("query_slots_per_frame") == 9, "R6_RUNTIME_PROTOCOL_QUERY_COUNT_DRIFT")

    rules = payload.get("factor_runtime_rules", {})
    _require(errors, rules.get("baseline_depth") == "SEALED_RAW_DEPTHART_HIGHRES_METRES" and rules.get("selected_direct_depth") == "SEALED_SOURCE_SCALE_ANCHORED_DEPTHART_HIGHRES_METRES", "R6_RUNTIME_PROTOCOL_DEPTH_OWNER_DRIFT")
    _require(errors, rules.get("direct_support") == "EXACT_PHASE_A_APPLE_SUPPORT_PLANE" and rules.get("query_clearance") == "RAW_CANDIDATE_SOURCE_LOCAL_SURFACE_WITH_BASELINE_SUPPORT", "R6_RUNTIME_PROTOCOL_EXTRACTION_RULE_DRIFT")
    for field in ("factor_depth_sha256_required", "source_surface_pixel_ids_sha256_required", "unknown_retained_on_any_failed_validity_gate"):
        _require(errors, rules.get(field) is True, f"R6_RUNTIME_PROTOCOL_RUNTIME_RULE_DRIFT:{field}")
    _require(errors, rules.get("runtime_output_may_contain_truth_error_fields") is False and rules.get("runtime_output_may_emit_final_three_state") is False, "R6_RUNTIME_PROTOCOL_OUTPUT_AUTHORITY_DRIFT")

    firewall = payload.get("phase_firewall", {})
    forbidden = ["FARO", "QUERY_TRUTH", "TASK_METRIC", "PRIOR_OUTCOME"]
    _require(errors, firewall.get("phase_a_forbidden_payload_roles") == forbidden and firewall.get("all_source_factor_frames_sealed_before_scoring") is True and firewall.get("branch_reselection_after_scoring") is False, "R6_RUNTIME_PROTOCOL_FIREWALL_DRIFT")
    roles = payload.get("data_roles", {})
    _require(errors, roles.get("implementation") == "SYNTHETIC_ONLY" and roles.get("r6_untouched_parent_ids") == R6_UNTOUCHED_PARENTS, "R6_RUNTIME_PROTOCOL_DATA_ROLE_DRIFT")
    _require(errors, roles.get("r6_untouched_outcomes") == "FORBIDDEN_FOR_IMPLEMENTATION_FORMATION_TUNING_OR_THRESHOLD_SELECTION" and roles.get("minimum_future_confirmation_parent_count") == 8, "R6_RUNTIME_PROTOCOL_UNTOUCHED_REUSE_DRIFT")

    acceptance = payload.get("implementation_acceptance", {})
    _require(errors, acceptance.get("required_module") == "scripts/research/taro_o0r_candidate_scale_runtime/prospective_factor_runtime.py", "R6_RUNTIME_PROTOCOL_MODULE_DRIFT")
    _require(errors, isinstance(acceptance.get("required_tests"), list) and len(acceptance["required_tests"]) == 8 and "PUBLIC_API_HAS_NO_TRUTH_OR_FARO_ARGUMENT" in acceptance["required_tests"] and "R6_UNTOUCHED_PARENT_ROLE_REJECTED" in acceptance["required_tests"], "R6_RUNTIME_PROTOCOL_TEST_SURFACE_DRIFT")
    _require(errors, acceptance.get("formation_replay_promotion_allowed") is False and acceptance.get("summary_inputs_canonicalized_before_aggregation") is True, "R6_RUNTIME_PROTOCOL_REPLAY_DRIFT")

    boundary = payload.get("uncertainty_and_reducer_boundary", {})
    _require(errors, boundary.get("uncertainty_model_attached_by_this_protocol") is False and boundary.get("deterministic_reducer_executed_by_this_protocol") is False and boundary.get("final_clear_blocked_until_source_surface_specific_uncertainty_is_frozen") is True and boundary.get("unknown_is_not_negative") is True, "R6_RUNTIME_PROTOCOL_REDUCER_BOUNDARY_DRIFT")
    authority = payload.get("execution_authority", {})
    _require(errors, authority.get("implementation") is True and authority.get("synthetic_tests") is True, "R6_RUNTIME_PROTOCOL_IMPLEMENTATION_AUTHORITY_DRIFT")
    for field in ("formation_replay", "new_source_download", "model_inference", "truth_scoring", "training", "network", "device", "product", "safety"):
        _require(errors, authority.get(field) is False, f"R6_RUNTIME_PROTOCOL_EXECUTION_AUTHORITY_DRIFT:{field}")

    if verify_files:
        for relative, size, digest in EXPECTED_BINDINGS.values():
            path = REPO_ROOT / relative
            _require(errors, path.is_file(), f"R6_RUNTIME_PROTOCOL_FILE_MISSING:{relative}")
            if path.is_file():
                _require(errors, path.stat().st_size == size, f"R6_RUNTIME_PROTOCOL_FILE_BYTES_DRIFT:{relative}")
                _require(errors, _sha(path) == digest, f"R6_RUNTIME_PROTOCOL_FILE_HASH_DRIFT:{relative}")
    return errors


def validate_file(path: Path = DEFAULT_LOCK, *, verify_files: bool = True) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"R6_RUNTIME_PROTOCOL_READ_FAILED:{error}"]
    return validate_payload(payload, verify_files=verify_files) if isinstance(payload, Mapping) else ["R6_RUNTIME_PROTOCOL_NOT_OBJECT"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--skip-file-verification", action="store_true")
    args = parser.parse_args(argv)
    errors = validate_file(args.lock, verify_files=not args.skip_file_verification)
    result = {
        "schema": "blindassist.taro.o0r.r6_factor_policy_prospective_runtime_protocol_validation.v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "terminal": "TARO_O0R_R6_PROSPECTIVE_RUNTIME_PROTOCOL_VALID" if not errors else "TARO_O0R_R6_PROSPECTIVE_RUNTIME_PROTOCOL_INVALID",
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
