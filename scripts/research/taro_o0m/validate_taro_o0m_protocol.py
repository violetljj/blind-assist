#!/usr/bin/env python3
"""Static validator for the non-execution TARO O0M protocol lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.taro.validate_taro_p0_protocol import (
    canonical_sha256,
    evaluate_identifiability_case,
    load_json,
    sha256_file,
)


PROTOCOL_ID = "TARO_O0M_SYNTHETIC_IDENTIFIABILITY_AND_FACTORIAL_MECHANICS_PROTOCOL_LOCK"
SUCCESSOR_ID = "TARO_O0M_IMPLEMENTATION_LOCK"
SUITE_ID = "TARO_O0M_EXECUTION_FAMILY_R0"
FACTORS = ["SCALE", "SUPPORT", "BOUNDARY"]
ARMS = [
    "NONE",
    "SCALE",
    "SUPPORT",
    "BOUNDARY",
    "SCALE_SUPPORT",
    "SCALE_BOUNDARY",
    "SUPPORT_BOUNDARY",
    "SCALE_SUPPORT_BOUNDARY",
]
MODES = ["VALUE_ONLY_COMMON_SUPPORT", "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY"]
SCENE_IDS = {
    "o0m_exec_scale_isolated",
    "o0m_exec_support_isolated",
    "o0m_exec_boundary_isolated",
    "o0m_exec_compound",
    "o0m_exec_boundary_validity",
}
SOURCE_CASE_IDS = {
    "full_state_underdetermined_query_identifiable_clear",
    "fully_observable_occupied",
    "static_scale_sensitive_unknown",
    "pure_rotation_scale_sensitive_unknown",
    "small_baseline_scale_sensitive_unknown",
    "missing_anchor_unknown",
    "anchor_shuffle_unknown",
    "wrong_k_unknown",
    "clock_offset_unknown",
    "nonsmooth_contact_switch_unknown",
}
IDENTIFIABILITY_ID_MAP = {
    "o0m_exec_full_state_underdetermined_query_identifiable_clear": "full_state_underdetermined_query_identifiable_clear",
    "o0m_exec_fully_observable_occupied": "fully_observable_occupied",
    "o0m_exec_static_scale_sensitive_unknown": "static_scale_sensitive_unknown",
    "o0m_exec_pure_rotation_scale_sensitive_unknown": "pure_rotation_scale_sensitive_unknown",
    "o0m_exec_small_baseline_scale_sensitive_unknown": "small_baseline_scale_sensitive_unknown",
    "o0m_exec_missing_anchor_unknown": "missing_anchor_unknown",
    "o0m_exec_anchor_shuffle_unknown": "anchor_shuffle_unknown",
    "o0m_exec_wrong_k_unknown": "wrong_k_unknown",
    "o0m_exec_clock_offset_unknown": "clock_offset_unknown",
    "o0m_exec_nonsmooth_contact_switch_unknown": "nonsmooth_contact_switch_unknown",
}
SCENE_CONTRACTS = {
    "o0m_exec_scale_isolated": (0.12, 0.04, {"SCALE": -0.08, "SUPPORT": 0.0, "BOUNDARY": 0.0}, {"SCALE": True, "SUPPORT": True, "BOUNDARY": True}),
    "o0m_exec_support_isolated": (0.08, 0.01, {"SCALE": 0.0, "SUPPORT": -0.07, "BOUNDARY": 0.0}, {"SCALE": True, "SUPPORT": True, "BOUNDARY": True}),
    "o0m_exec_boundary_isolated": (-0.04, 0.04, {"SCALE": 0.0, "SUPPORT": 0.0, "BOUNDARY": 0.08}, {"SCALE": True, "SUPPORT": True, "BOUNDARY": True}),
    "o0m_exec_compound": (0.10, 0.025, {"SCALE": -0.025, "SUPPORT": -0.025, "BOUNDARY": -0.025}, {"SCALE": True, "SUPPORT": True, "BOUNDARY": True}),
    "o0m_exec_boundary_validity": (0.12, 0.12, {"SCALE": 0.0, "SUPPORT": 0.0, "BOUNDARY": 0.0}, {"SCALE": True, "SUPPORT": True, "BOUNDARY": False}),
}
SCENE_INPUT_SHA256 = {
    "o0m_exec_scale_isolated": "480E5B46BD6034D19E64EA884AF355B2B01163021640709F8BE68D8BC3110886",
    "o0m_exec_support_isolated": "D9E905F5D792742B15D140FB29995D9FF78BDD728FE9370446FA2E553794D459",
    "o0m_exec_boundary_isolated": "95B6BCA45485876BF6F5AC724979BEBF8997A621BF1CC0B0298E226243C4D76D",
    "o0m_exec_compound": "AD467BEE80353D75969F48B6E2380FADAA6B8472A8B7F1ECADF995C5E7E2FEA4",
    "o0m_exec_boundary_validity": "556C7786CDA163CC61E81E5D8D51E86478DCFA347AD1AFB8C4A4111684D7FDE8",
}
FIXTURE_TOP_LEVEL_KEYS = {
    "schema", "suite_id", "seed", "rng_used", "outcome_access", "claim_ceiling",
    "numeric_contract", "identifiability_rule", "identifiability_cases", "factorial_scenes",
    "action_filter_cases",
}
IDENTIFIABILITY_CASE_KEYS = {
    "id", "motion", "measurement_jacobian_whitened", "query_jacobian_branches_m",
    "nominal_clearance_m", "receipt", "active_contact_clearances_m", "expected", "source_p0_case_id",
}
IDENTIFIABILITY_EXTRA_KEYS = {
    "o0m_exec_nonsmooth_contact_switch_unknown": {"branch_nominal_clearances_m"},
}
FACTORIAL_SCENE_KEYS = {
    "id", "family_id", "query_id", "truth_clearance_m", "observed_base_mean_m", "current_factor_error_m",
    "current_factor_valid", "oracle_factor_valid", "factor_provenance", "oracle_provenance",
    "factor_identity_sha256", "anchor_identity", "max_source_timestamp_ns", "expected_records",
}
AUTHORITY_KEYS = {
    "protocol_lock", "static_validation", "o0m_implementation", "o0m_execution", "real_data_read",
    "factor_injection", "gauge_solver", "training", "checkpoint", "active_prompt", "android_device",
    "calibration", "confirmation", "deployment", "default_app",
}
FREEZE_SHA256 = "A66A644474FE485FA7135961DC0DB2505ADA0FF426489725DA5C04057634F9FE"
CLAIM_CEILING_SHA256 = "DCF6AA2DE29E17666017668AA52C57E3E53F28038368627EB93F49F1C516031F"
GATES_SHA256 = "534803CF575394A868CCEB8E96D7B16A2BDC06488C85FB93E210F55B2F264425"
PROHIBITED_SHA256 = "6D645C438282CA475D15D8FAA9BAEC64E5387583CE210B3DDB8442157E5C5CD4"
RESOURCE_BUDGET_SHA256 = "1AD2E8A6C3D95B73E08844847535247EB7A91616563E9365E76C0A558866AF7B"
O0R_ADMISSION_SHA256 = "05EDB2A11A176987C390C1A2A0FD9FA06D24BC78991B90753FE2355A779DD130"
STATUS_SHA256 = "6F1CB6F227A5E5BF151FC19A3BB595FAE66ADBB3000D16EC831B291130D6C3DA"
SCOPE_SHA256 = "92F690E6F570B6A9DEA80474FBEF5792B86103D08C69E8FCE266A946169C833F"
IDENTIFIABILITY_RULE_SHA256 = "817637D554D89A344D953A78C605B7DCFB264A4C1335328EABBF626BB7421DFB"
ACTION_FILTER_SHA256 = "8C875C5F0A945E69B6EA006680C34756DE5CD2E1036CE93E913AAE9E8A477234"
FIXTURE_CLAIM_SHA256 = "1228C5DB3CE8E8EE9625720FA746F68556A731CF84D956458625AA5107587436"
SUCCESSOR_POLICY_SHA256 = "1434350C6F2C57D9F71534670488309C2FF597DA01AF9478CCF07559F55FFB3A"
UNIQUE_SUCCESSOR_SHA256 = "C9B176A69AC49C9C8452A3ABA2FC9E2B77D541A5D2A8775E524C1F26E24BF28E"
PROTOCOL_SEMANTIC_CORE_SHA256 = "7B19350D115F9B2B7363BF84051D21B6F69CF81B644EFCE57BEB6A0BE7971A6D"
FIXTURE_CANONICAL_SHA256 = "D4297BE8F984A4989FE9774BBD7312AFD107F2A8BE619E48921612612CE5534F"
CLAIMS_ALLOWED_SHA256 = "9805FE425BC0F0112FC4498AA171308CD519E119763C9470D02378F3F969AD00"
CLAIMS_FORBIDDEN_SHA256 = "35B6DE285184569FF089DE592BDDC1899C4A825878FF448F25CFD66D32BED082"
RESULT_MODEL_SHA256 = "5F4B50167B4CA8CA59E2C6E5D641AF3D2DFC2292618E566243284819C4545AE8"
EXPECTED_BINDINGS = {
    "GOVERNANCE_POLICY": "configs/research_governance_v4.json",
    "P0_PROTOCOL": "docs/research/taro/TARO_P0_TASK_QUERY_IDENTIFIABILITY_AND_FACTOR_ORACLE_CANARY_PROTOCOL_LOCK_2026-08-10.json",
    "P0_RESULT": "docs/research/taro/TARO_P0_PROTOCOL_LOCK_RESULT_2026-08-10.json",
    "P0_NUMERIC_EVALUATOR": "scripts/research/taro/validate_taro_p0_protocol.py",
    "O0M_EXECUTION_FIXTURE": "docs/research/taro/TARO_O0M_EXECUTION_FIXTURE_SPEC_2026-08-10.json",
    "O0M_STATIC_VALIDATOR": "scripts/research/taro_o0m/validate_taro_o0m_protocol.py",
    "O0M_STATIC_TESTS": "scripts/research/taro_o0m/test_validate_taro_o0m_protocol.py",
}
STATIC_MODULE_ALLOWLIST = {
    "scripts/research/taro_o0m/README.md",
    "scripts/research/taro_o0m/validate_taro_o0m_protocol.py",
    "scripts/research/taro_o0m/test_validate_taro_o0m_protocol.py",
}
EXCLUSIVE_ARTIFACT_ROOT = "artifacts.local/evidence/taro/o0m-analytic-mechanics-r0"
PROHIBITED_RUNTIME_PATHS = (
    "scripts/research/taro_o0m_runtime/o0m_mechanics.py",
    "scripts/research/taro_o0m_runtime/run_o0m_canary.py",
    "scripts/research/taro_o0m_runtime/test_o0m_mechanics.py",
)


def _error(errors: list[str], condition: bool, code: str) -> None:
    if condition:
        errors.append(code)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _round(value: float) -> float:
    result = round(float(value), 12)
    return 0.0 if result == -0.0 else result


def _arm_factors(arm: str) -> set[str]:
    return set() if arm == "NONE" else set(arm.split("_"))


def expected_records(scene: dict[str, Any], numeric: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observed_base = float(scene["observed_base_mean_m"])
    current_error = scene["current_factor_error_m"]
    current_valid = scene["current_factor_valid"]
    oracle_valid = scene["oracle_factor_valid"]
    records: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        for arm in ARMS:
            patched = _arm_factors(arm)
            values = {factor: (0.0 if factor in patched else float(current_error[factor])) for factor in FACTORS}
            validity = {factor: bool(current_valid[factor]) for factor in FACTORS}
            sigma = {factor: float(numeric["sigma_factor_baseline_m"]) for factor in FACTORS}
            provenance = dict(scene["factor_provenance"])
            if mode == "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY":
                for factor in patched:
                    validity[factor] = bool(oracle_valid[factor])
                    sigma[factor] = float(numeric["sigma_factor_oracle_m"])
                    provenance[factor] = scene["oracle_provenance"][factor]
            payload = {
                "scene_id": scene["id"],
                "query_id": scene["query_id"],
                "observed_base_mean_m": _round(observed_base),
                "arm": arm,
                "mode": mode,
                "factor_error_m": {factor: _round(values[factor]) for factor in FACTORS},
                "factor_valid": validity,
                "factor_sigma_m": {factor: _round(sigma[factor]) for factor in FACTORS},
            }
            common_eligible = all(bool(current_valid[factor]) and bool(oracle_valid[factor]) for factor in FACTORS)
            support = {
                "scene_id": scene["id"],
                "ordered_query_ids": [scene["query_id"]],
                "intersection_mask": [common_eligible] if mode == MODES[0] else [all(validity.values())],
                "eligible": common_eligible if mode == MODES[0] else all(validity.values()),
                "policy": "COMMON_SUPPORT_INTERSECTION" if mode == MODES[0] else "ARM_NATIVE_FULL_BLOCK",
            }
            guard = {
                "scene_id": scene["id"],
                "arm": arm,
                "mode": mode,
                "factor_valid": validity,
                "factor_sigma_m": {factor: _round(sigma[factor]) for factor in FACTORS},
                "factor_provenance": provenance,
                "factor_identity_sha256": scene["factor_identity_sha256"],
                "anchor_identity": scene["anchor_identity"],
                "max_source_timestamp_ns": scene["max_source_timestamp_ns"],
                "observed_base_mean_m": _round(observed_base),
            }
            if not all(validity.values()):
                output = {
                    "terminal": "ABSTAIN_FACTOR_INVALID",
                    "query_state": "UNKNOWN",
                    "mean_m": None,
                    "halfwidth_m": None,
                    "lower_m": None,
                    "upper_m": None,
                }
            else:
                mean = observed_base - sum(float(current_error[factor]) for factor in patched)
                halfwidth = float(numeric["interval_multiplier"]) * math.sqrt(
                    float(numeric["sigma_measurement_m"]) ** 2 + sum(value * value for value in sigma.values())
                )
                lower, upper = mean - halfwidth, mean + halfwidth
                state = (
                    "CLEAR_OBSERVED"
                    if lower > float(numeric["clear_margin_m"])
                    else "OCCUPIED_OBSERVED"
                    if upper <= float(numeric["occupied_margin_m"])
                    else "UNKNOWN"
                )
                output = {
                    "terminal": "EVALUATED",
                    "query_state": state,
                    "mean_m": _round(mean),
                    "halfwidth_m": _round(halfwidth),
                    "lower_m": _round(lower),
                    "upper_m": _round(upper),
                }
            records[f"{arm}|{mode}"] = {
                "payload_sha256": canonical_sha256(payload),
                "output_sha256": canonical_sha256(output),
                "common_support_sha256": canonical_sha256(support),
                "intervention_guard_sha256": canonical_sha256(guard),
                "output": output,
            }
    return records


def validate_fixture(fixture: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _error(errors, set(fixture) != FIXTURE_TOP_LEVEL_KEYS, "FIXTURE_TOP_LEVEL_KEYS")
    _error(errors, canonical_sha256(fixture) != FIXTURE_CANONICAL_SHA256, "FIXTURE_CANONICAL_DRIFT")
    _error(errors, fixture.get("schema") != "blindassist.taro.o0m.execution_fixture_spec.v1", "FIXTURE_SCHEMA")
    _error(errors, fixture.get("suite_id") != SUITE_ID, "SUITE_ID")
    _error(errors, fixture.get("seed") != 1729 or fixture.get("rng_used") is not False, "RNG_CONTRACT")
    _error(errors, fixture.get("outcome_access") != "NONE_FROZEN_ANALYTIC_TRUTH", "FIXTURE_OUTCOME_ACCESS")
    _error(errors, canonical_sha256(fixture.get("claim_ceiling")) != FIXTURE_CLAIM_SHA256, "FIXTURE_CLAIM_CEILING")
    _error(errors, canonical_sha256(fixture.get("identifiability_rule")) != IDENTIFIABILITY_RULE_SHA256, "IDENTIFIABILITY_RULE_DRIFT")
    numeric = fixture.get("numeric_contract", {})
    expected_numeric = {
        "factor_order": FACTORS,
        "arms": ARMS,
        "oracle_modes": MODES,
        "sigma_measurement_m": 0.01,
        "sigma_factor_baseline_m": 0.014,
        "sigma_factor_oracle_m": 0.004,
        "interval_multiplier": 1.0,
        "interval_semantics": "DETERMINISTIC_BUDGET_HALFWIDTH_NOT_GAUSSIAN_1SIGMA_OR_95_PERCENT_COVERAGE",
        "numeric_atol_m": 1e-10,
        "hash_quantization_decimal_places": 12,
        "clear_margin_m": 0.05,
        "occupied_margin_m": 0.0,
    }
    _error(errors, numeric != expected_numeric, "NUMERIC_CONTRACT")

    identifiability = fixture.get("identifiability_cases")
    _error(errors, not isinstance(identifiability, list) or len(identifiability) != 10, "IDENTIFIABILITY_CASE_COUNT")
    if isinstance(identifiability, list):
        source_ids: set[str] = set()
        execution_ids: set[str] = set()
        for case in identifiability:
            case_id = str(case.get("id")) if isinstance(case, dict) else "INVALID"
            if not isinstance(case, dict):
                errors.append("IDENTIFIABILITY_CASE_OBJECT")
                continue
            expected_case_keys = IDENTIFIABILITY_CASE_KEYS | IDENTIFIABILITY_EXTRA_KEYS.get(case_id, set())
            _error(errors, set(case) != expected_case_keys, f"IDENTIFIABILITY_CASE_KEYS:{case_id}")
            _error(errors, case_id in execution_ids, f"DUPLICATE_EXECUTION_CASE:{case_id}")
            execution_ids.add(case_id)
            _error(errors, not case_id.startswith("o0m_exec_") or case_id == case.get("source_p0_case_id"), f"EXECUTION_FAMILY_ID:{case_id}")
            source_id = str(case.get("source_p0_case_id"))
            _error(errors, IDENTIFIABILITY_ID_MAP.get(case_id) != source_id, f"EXECUTION_SOURCE_MAPPING:{case_id}")
            _error(errors, source_id in source_ids, f"DUPLICATE_SOURCE_CASE:{source_id}")
            source_ids.add(source_id)
            try:
                calculated = evaluate_identifiability_case(case, fixture["identifiability_rule"])
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(f"IDENTIFIABILITY_CALCULATION:{case_id}")
                continue
            _error(errors, case.get("expected") != calculated, f"IDENTIFIABILITY_TRUTH:{case_id}")
        _error(errors, source_ids != SOURCE_CASE_IDS, "SOURCE_CASE_SET")
        _error(errors, execution_ids != set(IDENTIFIABILITY_ID_MAP), "EXECUTION_CASE_SET")

    scenes = fixture.get("factorial_scenes")
    scene_ids = {str(scene.get("id")) for scene in scenes if isinstance(scene, dict)} if isinstance(scenes, list) else set()
    _error(errors, not isinstance(scenes, list) or len(scenes) != 5, "FACTORIAL_SCENE_COUNT")
    _error(errors, scene_ids != SCENE_IDS, "FACTORIAL_SCENE_SET")
    if isinstance(scenes, list):
        all_states: set[str] = set()
        for scene in scenes:
            if not isinstance(scene, dict):
                errors.append("FACTORIAL_SCENE_OBJECT")
                continue
            scene_id = str(scene.get("id"))
            _error(errors, set(scene) != FACTORIAL_SCENE_KEYS, f"FACTORIAL_SCENE_KEYS:{scene_id}")
            _error(errors, scene.get("family_id") != SUITE_ID, f"SCENE_FAMILY:{scene_id}")
            scene_input = {key: value for key, value in scene.items() if key != "expected_records"}
            _error(errors, canonical_sha256(scene_input) != SCENE_INPUT_SHA256.get(scene_id), f"SCENE_INPUT_DRIFT:{scene_id}")
            contract = SCENE_CONTRACTS.get(scene_id)
            if contract is not None:
                _error(
                    errors,
                    scene.get("truth_clearance_m") != contract[0]
                    or scene.get("observed_base_mean_m") != contract[1]
                    or scene.get("current_factor_error_m") != contract[2]
                    or scene.get("current_factor_valid") != contract[3],
                    f"SCENE_CONTRACT:{scene_id}",
                )
            for field in ("current_factor_error_m", "current_factor_valid", "oracle_factor_valid", "factor_provenance", "oracle_provenance"):
                value = scene.get(field)
                _error(errors, not isinstance(value, dict) or list(value) != FACTORS, f"FACTOR_BLOCK:{scene_id}:{field}")
            _error(errors, not _finite(scene.get("truth_clearance_m")), f"TRUTH:{scene_id}")
            _error(errors, not _finite(scene.get("observed_base_mean_m")), f"OBSERVED_BASE:{scene_id}")
            try:
                oracle_mean = float(scene["observed_base_mean_m"]) - sum(float(scene["current_factor_error_m"][factor]) for factor in FACTORS)
                _error(errors, abs(oracle_mean - float(scene["truth_clearance_m"])) > float(numeric["numeric_atol_m"]), f"ORACLE_TRUTH_CLOSURE:{scene_id}")
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(f"ORACLE_TRUTH_CLOSURE:{scene_id}")
            try:
                calculated_records = expected_records(scene, numeric)
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(f"RECORD_CALCULATION:{scene_id}")
                continue
            _error(errors, scene.get("expected_records") != calculated_records, f"RECORD_TRUTH:{scene_id}")
            _error(errors, len(calculated_records) != 80 // 5, f"RECORD_COUNT:{scene_id}")
            for record in calculated_records.values():
                all_states.add(str(record["output"]["query_state"]))
            value_hashes = {
                record["common_support_sha256"]
                for key, record in calculated_records.items()
                if key.endswith("|VALUE_ONLY_COMMON_SUPPORT")
            }
            _error(errors, len(value_hashes) != 1, f"COMMON_SUPPORT_DRIFT:{scene_id}")
        _error(errors, not {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}.issubset(all_states), "STATE_COLLAPSE")

    actions = fixture.get("action_filter_cases")
    _error(errors, not isinstance(actions, list) or len(actions) != 2, "ACTION_CASE_COUNT")
    _error(errors, canonical_sha256(actions) != ACTION_FILTER_SHA256, "ACTION_FILTER_DRIFT")
    if isinstance(actions, list):
        for action in actions:
            if isinstance(action, dict) and action.get("requires_body_motion") is True:
                _error(errors, action.get("expected_allowed") is not False or action.get("expected_reason") != "BODY_MOTION_FORBIDDEN", "BODY_MOTION_FILTER")
    return sorted(set(errors))


def validate_binding_contract(protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    bindings = protocol.get("bindings")
    if not isinstance(bindings, list):
        return ["BINDINGS"]
    roles: list[str] = []
    paths: list[str] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            errors.append("BINDING_FIELDS")
            continue
        _error(errors, set(binding) != {"role", "path", "sha256"}, "BINDING_FIELDS")
        role = binding.get("role")
        path = binding.get("path")
        if not isinstance(role, str) or not isinstance(path, str) or not isinstance(binding.get("sha256"), str):
            errors.append("BINDING_FIELDS")
            continue
        roles.append(role)
        paths.append(path)
        _error(errors, EXPECTED_BINDINGS.get(role) != path, f"BINDING_ROLE_PATH:{role}")
    _error(errors, len(roles) != len(set(roles)), "BINDING_ROLE_DUPLICATE")
    _error(errors, len(paths) != len(set(paths)), "BINDING_PATH_DUPLICATE")
    _error(errors, set(roles) != set(EXPECTED_BINDINGS), "BINDING_ROLE_SET")
    return sorted(set(errors))


def validate_protocol(protocol: dict[str, Any], fixture: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    semantic_core = {key: value for key, value in protocol.items() if key != "bindings"}
    _error(errors, canonical_sha256(semantic_core) != PROTOCOL_SEMANTIC_CORE_SHA256, "PROTOCOL_SEMANTIC_CORE_DRIFT")
    _error(errors, canonical_sha256(protocol.get("claims_allowed")) != CLAIMS_ALLOWED_SHA256, "CLAIMS_ALLOWED_DRIFT")
    _error(errors, canonical_sha256(protocol.get("claims_forbidden")) != CLAIMS_FORBIDDEN_SHA256, "CLAIMS_FORBIDDEN_DRIFT")
    _error(errors, canonical_sha256(protocol.get("result_model")) != RESULT_MODEL_SHA256, "RESULT_MODEL_DRIFT")
    errors.extend(validate_binding_contract(protocol))
    _error(errors, protocol.get("schema_version") != "blindassist.research_protocol.v1", "PROTOCOL_SCHEMA")
    _error(errors, protocol.get("protocol_id") != PROTOCOL_ID, "PROTOCOL_ID")
    _error(errors, protocol.get("profile") != "CANARY_LITE" or protocol.get("stage") != "DISCOVERY", "PROFILE_STAGE")
    _error(errors, protocol.get("scientific_status") != "NOT_RUN" or protocol.get("outcome_access_started") is not False, "OUTCOME_FIREWALL")
    _error(errors, protocol.get("fixture_suite_id") != fixture.get("suite_id"), "FIXTURE_BINDING")
    authority = protocol.get("execution_authority", {})
    allowed = {"protocol_lock", "static_validation"}
    _error(errors, not isinstance(authority, dict) or set(authority) != AUTHORITY_KEYS or any(value is not (key in allowed) for key, value in authority.items()), "AUTHORITY_EXCEEDED")
    successor = protocol.get("unique_successor", {})
    _error(errors, successor.get("id") != SUCCESSOR_ID or successor.get("execution_authority") is not False, "SUCCESSOR")
    _error(errors, canonical_sha256(successor) != UNIQUE_SUCCESSOR_SHA256, "UNIQUE_SUCCESSOR_DRIFT")
    _error(errors, canonical_sha256(protocol.get("successor_policy")) != SUCCESSOR_POLICY_SHA256, "SUCCESSOR_POLICY_DRIFT")
    _error(errors, protocol.get("resource_budget", {}).get("network") is not False, "NETWORK_BUDGET")
    _error(errors, protocol.get("resource_budget", {}).get("real_data") is not False, "REAL_DATA_BUDGET")
    _error(errors, protocol.get("resource_budget", {}).get("gpu") is not False, "GPU_BUDGET")
    _error(errors, canonical_sha256(protocol.get("resource_budget")) != RESOURCE_BUDGET_SHA256, "RESOURCE_BUDGET_DRIFT")
    _error(errors, canonical_sha256(protocol.get("freeze")) != FREEZE_SHA256, "FREEZE_DRIFT")
    _error(errors, canonical_sha256(protocol.get("claim_ceiling")) != CLAIM_CEILING_SHA256, "CLAIM_CEILING_DRIFT")
    _error(errors, canonical_sha256(protocol.get("o0m_gates")) != GATES_SHA256, "GATE_CONTRACT_DRIFT")
    _error(errors, canonical_sha256(protocol.get("prohibited_artifacts_in_this_lock")) != PROHIBITED_SHA256, "PROHIBITED_SCOPE_DRIFT")
    _error(errors, canonical_sha256(protocol.get("o0r_admission")) != O0R_ADMISSION_SHA256, "O0R_ADMISSION_DRIFT")
    _error(errors, canonical_sha256(protocol.get("status")) != STATUS_SHA256, "STATUS_DRIFT")
    _error(errors, canonical_sha256(protocol.get("o0m_protocol_scope")) != SCOPE_SHA256, "SCOPE_DRIFT")
    return sorted(set(errors))


def validate_bindings(protocol: dict[str, Any], repo_root: Path) -> list[str]:
    errors = validate_binding_contract(protocol)
    bindings = protocol.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        return ["BINDINGS"]
    for binding in bindings:
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str) or not isinstance(binding.get("sha256"), str):
            errors.append("BINDING_FIELDS")
            continue
        path = repo_root / binding["path"]
        if not path.is_file():
            errors.append(f"BINDING_ABSENT:{binding['path']}")
        elif sha256_file(path) != binding["sha256"].upper():
            errors.append(f"BINDING_SHA:{binding['path']}")
    return errors


def validate_runtime_absence(repo_root: Path) -> list[str]:
    errors = [f"PROHIBITED_RUNTIME_PRESENT:{path}" for path in PROHIBITED_RUNTIME_PATHS if (repo_root / path).exists()]
    runtime_root = repo_root / "scripts/research/taro_o0m_runtime"
    if runtime_root.is_dir():
        for path in runtime_root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                relative = path.relative_to(repo_root).as_posix()
                if f"PROHIBITED_RUNTIME_PRESENT:{relative}" not in errors:
                    errors.append(f"PROHIBITED_RUNTIME_PRESENT:{relative}")
    static_root = repo_root / "scripts/research/taro_o0m"
    if static_root.is_dir():
        for path in static_root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(repo_root).as_posix()
            if relative not in STATIC_MODULE_ALLOWLIST:
                errors.append(f"PROHIBITED_STATIC_MODULE_PRESENT:{relative}")
    artifact_root = repo_root / EXCLUSIVE_ARTIFACT_ROOT
    if artifact_root.exists():
        errors.append(f"PROHIBITED_ARTIFACT_ROOT_PRESENT:{EXCLUSIVE_ARTIFACT_ROOT}")
    return errors


def validate_static_contract(protocol: dict[str, Any], fixture: dict[str, Any]) -> list[str]:
    return sorted(set(validate_fixture(fixture) + validate_protocol(protocol, fixture)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    protocol = load_json(args.protocol.resolve())
    fixture = load_json(args.fixtures.resolve())
    errors = validate_static_contract(protocol, fixture)
    errors.extend(validate_bindings(protocol, repo_root))
    errors.extend(validate_runtime_absence(repo_root))
    result = {
        "schema": "blindassist.taro.o0m.protocol_static_validation.v1",
        "status": "VALID" if not errors else "INVALID",
        "errors": sorted(set(errors)),
        "scientific_status": "NOT_RUN",
        "implementation_authority": False,
        "execution_authority": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
