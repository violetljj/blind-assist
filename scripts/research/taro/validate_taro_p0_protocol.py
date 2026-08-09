#!/usr/bin/env python3
"""Static validator for the non-execution TARO P0 protocol lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


PROTOCOL_ID = "TARO_P0_TASK_QUERY_IDENTIFIABILITY_AND_FACTOR_ORACLE_CANARY_PROTOCOL_LOCK"
SUCCESSOR_ID = "TARO_O0M_SYNTHETIC_IDENTIFIABILITY_AND_FACTORIAL_MECHANICS_PROTOCOL_LOCK"
SCHEMA_VERSION = "blindassist.taro.p0.schema_bundle.v1"
FIXTURE_SCHEMA = "blindassist.taro.p0.analytic_fixture_spec.v1"
GOVERNANCE_POLICY_DIGEST = "c088093d765ce3900848a80f8a45585bc217409c344c6e7212d87e6bda6d624e"
TARGET_SCHEMA_DEFS = {
    "TaroFrameReceipt",
    "TaroFactorPosterior",
    "TaroTaskQuery",
    "TaroObservationCandidate",
}
STATE_ORDER = ["log_scale", "support_tangent_x", "support_tangent_y", "support_offset_m"]
STATE_ONE_UNIT = [0.1, 0.05, 0.05, 0.1]
FACTORIAL_ARMS = [
    "NONE",
    "SCALE",
    "SUPPORT",
    "BOUNDARY",
    "SCALE_SUPPORT",
    "SCALE_BOUNDARY",
    "SUPPORT_BOUNDARY",
    "SCALE_SUPPORT_BOUNDARY",
]
ORACLE_MODES = ["VALUE_ONLY_COMMON_SUPPORT", "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY"]
IDENTIFIABILITY_CASES = {
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
ACTION_CASES = {"stationary_camera_micro_baseline_allowed", "step_sideways_forbidden"}
FACTOR_MECHANICS_CASES = {
    "scale_only_causal_patch",
    "support_only_causal_patch",
    "boundary_only_causal_patch",
    "three_factor_interaction_patch",
    "k_corruption_separate_negative_control",
    "missing_anchor_separate_negative_control",
}
FACTOR_ORDER = ["SCALE", "SUPPORT", "BOUNDARY"]
O0M_GATES = {
    "O0M_G01_BINDING_AND_INTEGRITY",
    "O0M_G02_ORACLE_POSITIVE_CONTROL",
    "O0M_G03_DISCRIMINATING_OPPORTUNITY",
    "O0M_G04_IDENTIFIABILITY_TRUTH",
    "O0M_G05_DEGENERATE_FAIL_CLOSED",
    "O0M_G06_INTERVENTION_PURITY",
    "O0M_G07_FACTOR_SPECIFICITY",
    "O0M_G08_COMPOUND_CLOSURE",
    "O0M_G09_MONOTONICITY_AND_DETERMINISM",
    "O0M_G10_LEAKAGE_FIREWALL",
}
O0M_GATES_SHA256 = "534803CF575394A868CCEB8E96D7B16A2BDC06488C85FB93E210F55B2F264425"
PROHIBITED_RUNTIME_PATHS = (
    "scripts/research/taro/taro_identifiability.py",
    "scripts/research/taro/taro_factorial_oracle.py",
    "scripts/research/taro/run_o0m_canary.py",
    "scripts/research/taro/materialize_o0r_data.py",
    "scripts/research/taro/train_taro.py",
)
P0_MODULE_ALLOWLIST = {
    "scripts/research/taro/README.md",
    "scripts/research/taro/validate_taro_p0_protocol.py",
    "scripts/research/taro/test_validate_taro_p0_protocol.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON value must be an object: {path}")
    return value


def _error(errors: list[str], condition: bool, code: str) -> None:
    if condition:
        errors.append(code)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return _is_number(value)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any] | None:
    if not ref.startswith("#/"):
        return None
    current: Any = root
    for component in ref[2:].split("/"):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current if isinstance(current, dict) else None


SUPPORTED_SCHEMA_KEYWORDS = {
    "$schema",
    "$id",
    "$defs",
    "$ref",
    "title",
    "description",
    "schema_version",
    "type",
    "const",
    "enum",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "minLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "items",
    "additionalProperties",
    "required",
    "properties",
}


def validate_schema_keywords(schema: Any, path: str = "schema") -> list[str]:
    """Reject any schema keyword the local subset evaluator would otherwise ignore."""

    if not isinstance(schema, dict):
        return [f"SCHEMA_NODE:{path}"]
    errors = [f"UNSUPPORTED_SCHEMA_KEYWORD:{path}.{key}" for key in schema if key not in SUPPORTED_SCHEMA_KEYWORDS]
    defs = schema.get("$defs", {})
    if isinstance(defs, dict):
        for name, child in defs.items():
            errors.extend(validate_schema_keywords(child, f"{path}.$defs.{name}"))
    elif "$defs" in schema:
        errors.append(f"SCHEMA_DEFS:{path}")
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for name, child in properties.items():
            errors.extend(validate_schema_keywords(child, f"{path}.properties.{name}"))
    elif "properties" in schema:
        errors.append(f"SCHEMA_PROPERTIES:{path}")
    items = schema.get("items")
    if items is not None:
        errors.extend(validate_schema_keywords(items, f"{path}.items"))
    return errors


def validate_schema_instance(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str,
) -> list[str]:
    """Validate the JSON-Schema subset used by the frozen P0 bundle."""

    errors: list[str] = []
    if "$ref" in schema:
        target = _resolve_ref(root, str(schema["$ref"]))
        if target is None:
            return [f"SCHEMA_REF:{path}"]
        return validate_schema_instance(value, target, root, path)

    if "const" in schema and value != schema["const"]:
        errors.append(f"SCHEMA_CONST:{path}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"SCHEMA_ENUM:{path}")

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        accepted = [expected_type]
    elif isinstance(expected_type, list):
        accepted = [str(item) for item in expected_type]
    else:
        accepted = []
    if accepted and not any(_type_matches(value, item) for item in accepted):
        return errors + [f"SCHEMA_TYPE:{path}"]

    if isinstance(value, str):
        if isinstance(schema.get("minLength"), int) and len(value) < int(schema["minLength"]):
            errors.append(f"SCHEMA_MIN_LENGTH:{path}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
            errors.append(f"SCHEMA_PATTERN:{path}")

    if _is_number(value):
        numeric = float(value)
        if "minimum" in schema and numeric < float(schema["minimum"]):
            errors.append(f"SCHEMA_MINIMUM:{path}")
        if "maximum" in schema and numeric > float(schema["maximum"]):
            errors.append(f"SCHEMA_MAXIMUM:{path}")
        if "exclusiveMinimum" in schema and numeric <= float(schema["exclusiveMinimum"]):
            errors.append(f"SCHEMA_EXCLUSIVE_MINIMUM:{path}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(f"SCHEMA_MIN_ITEMS:{path}")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(f"SCHEMA_MAX_ITEMS:{path}")
        if schema.get("uniqueItems") is True:
            canonical = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value]
            if len(canonical) != len(set(canonical)):
                errors.append(f"SCHEMA_UNIQUE_ITEMS:{path}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(validate_schema_instance(item, item_schema, root, f"{path}[{index}]"))

    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(f"SCHEMA_REQUIRED:{path}.{key}")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            if schema.get("additionalProperties") is False:
                for key in value:
                    if key not in properties:
                        errors.append(f"SCHEMA_ADDITIONAL_PROPERTY:{path}.{key}")
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    errors.extend(validate_schema_instance(value[key], child_schema, root, f"{path}.{key}"))
    return errors


def _ids(items: Any) -> set[str]:
    if not isinstance(items, list):
        return set()
    return {str(item.get("id")) for item in items if isinstance(item, dict)}


def _matrix4(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(row, list) and len(row) == 4 and all(_is_number(item) for item in row) for row in value)
    )


def _is_symmetric_psd4(value: Any, tolerance: float = 1e-12) -> bool:
    if not isinstance(value, list) or len(value) != 16 or any(not _is_number(item) for item in value):
        return False
    matrix = [[float(value[4 * row + column]) for column in range(4)] for row in range(4)]
    if any(abs(matrix[i][j] - matrix[j][i]) > tolerance for i in range(4) for j in range(4)):
        return False
    lower = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i + 1):
            residual = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if residual < -tolerance:
                    return False
                lower[i][j] = math.sqrt(max(0.0, residual))
            elif lower[j][j] > tolerance:
                lower[i][j] = residual / lower[j][j]
            elif abs(residual) > tolerance:
                return False
    return True


def _round_metric(value: float) -> float:
    rounded = round(float(value), 9)
    return 0.0 if rounded == -0.0 else rounded


def _right_singular_system(matrix: list[list[float]]) -> list[tuple[float, list[float]]]:
    """Pure-Python Jacobi eigensolver for A^T A; sufficient for the frozen 4-D canary."""

    gram = [
        [sum(float(matrix[k][i]) * float(matrix[k][j]) for k in range(4)) for j in range(4)]
        for i in range(4)
    ]
    vectors = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for _ in range(128):
        p, q = max(((i, j) for i in range(4) for j in range(i + 1, 4)), key=lambda ij: abs(gram[ij[0]][ij[1]]))
        if abs(gram[p][q]) <= 1e-14:
            break
        angle = 0.5 * math.atan2(2.0 * gram[p][q], gram[q][q] - gram[p][p])
        cosine, sine = math.cos(angle), math.sin(angle)
        for k in range(4):
            if k in (p, q):
                continue
            gkp, gkq = gram[k][p], gram[k][q]
            gram[k][p] = gram[p][k] = cosine * gkp - sine * gkq
            gram[k][q] = gram[q][k] = sine * gkp + cosine * gkq
        app, aqq, apq = gram[p][p], gram[q][q], gram[p][q]
        gram[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        gram[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        gram[p][q] = gram[q][p] = 0.0
        for k in range(4):
            vkp, vkq = vectors[k][p], vectors[k][q]
            vectors[k][p] = cosine * vkp - sine * vkq
            vectors[k][q] = sine * vkp + cosine * vkq
    system = [
        (math.sqrt(max(0.0, gram[i][i])), [vectors[row][i] for row in range(4)])
        for i in range(4)
    ]
    return sorted(system, key=lambda item: item[0], reverse=True)


def _receipt_failure(receipt: dict[str, Any]) -> tuple[str, str] | None:
    checks = [
        ("anchor_valid", "MISSING_METRIC_ANCHOR"),
        ("anchor_identity_matches_parent", "ANCHOR_SHUFFLED"),
        ("k_valid", "K_INVALID"),
        ("transform_valid", "TRANSFORM_INVALID"),
        ("clock_valid", "CLOCK_INVALID"),
        ("factor_valid", "FACTOR_INVALID"),
    ]
    for field, reason in checks:
        if receipt.get(field) is not True:
            return "ABSTAIN", reason
    return None


def evaluate_identifiability_case(case: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    matrix = case.get("measurement_jacobian_whitened")
    branches = case.get("query_jacobian_branches_m")
    if not _matrix4(matrix) or not isinstance(branches, list) or not branches:
        raise ValueError("invalid analytic identifiability matrix")
    system = _right_singular_system(matrix)
    sigma_max = system[0][0] if system else 0.0
    threshold = max(float(rule["strong_singular_value_min"]), float(rule["relative_singular_value_min"]) * sigma_max)
    weak_vectors = [vector for singular, vector in system if singular < threshold]
    null_vectors = [vector for singular, vector in system if singular <= 1e-10]
    strong_system = [(singular, vector) for singular, vector in system if singular >= threshold]

    def projected_radius(branch: list[float], vectors: list[list[float]]) -> float:
        return float(rule["weak_trust_region_l2"]) * math.sqrt(
            sum(sum(float(branch[i]) * vector[i] for i in range(4)) ** 2 for vector in vectors)
        )

    ambiguity = max(projected_radius(branch, weak_vectors) for branch in branches)
    null_ambiguity = max(projected_radius(branch, null_vectors) for branch in branches)
    measurement_halfwidth = max(
        float(rule["measurement_noise_budget_l2_95"])
        * math.sqrt(
            sum(
                (sum(float(branch[i]) * vector[i] for i in range(4)) / singular) ** 2
                for singular, vector in strong_system
            )
        )
        for branch in branches
    )
    strong_rank = sum(1 for singular, _ in system if singular >= threshold)
    update_decision = "FREEZE" if strong_rank == 0 else "UPDATE_STRONG_SUBSPACE"
    result: dict[str, Any] = {
        "query_identifiable": True,
        "query_state": "UNKNOWN",
        "update_decision": update_decision,
        "reason_code": "NONE",
        "strong_rank": strong_rank,
        "task_ambiguity_radius_m": _round_metric(ambiguity),
        "measurement_halfwidth_m_95": _round_metric(measurement_halfwidth),
        "limiting_null_invariant": null_ambiguity <= 1e-10,
        "interval_lower_m": None,
        "interval_upper_m": None,
    }
    receipt = case.get("receipt", {})
    failure = _receipt_failure(receipt if isinstance(receipt, dict) else {})
    if failure is not None:
        result.update(query_identifiable=False, query_state="UNKNOWN", update_decision=failure[0], reason_code=failure[1])
        return result

    active = case.get("active_contact_clearances_m", [])
    branch_nominal = case.get("branch_nominal_clearances_m", [])
    competition = (
        isinstance(active, list)
        and len(active) >= 2
        and all(_is_number(value) for value in active)
        and abs(sorted(float(value) for value in active)[1] - sorted(float(value) for value in active)[0])
        <= float(rule["contact_competition_margin_m"])
    )
    branch_conflict = (
        isinstance(branch_nominal, list)
        and len(branch_nominal) >= 2
        and all(_is_number(value) for value in branch_nominal)
        and max(float(value) for value in branch_nominal) - min(float(value) for value in branch_nominal)
        > float(rule["contact_branch_spread_max_m"])
    )
    if competition and branch_conflict:
        result.update(query_identifiable=False, query_state="UNKNOWN", reason_code="NONSMOOTH_CONTACT_SWITCH")
        return result
    if ambiguity > float(rule["task_ambiguity_radius_max_m"]) + 1e-12:
        reason = {
            "pure_rotation": "PURE_ROTATION_SCALE_WEAK",
            "small_baseline": "BASELINE_TOO_SMALL",
        }.get(str(case.get("motion")), "WEAK_QUERY_DIRECTION")
        result.update(query_identifiable=False, query_state="UNKNOWN", reason_code=reason)
        nominal = float(case["nominal_clearance_m"])
        total_halfwidth = ambiguity + measurement_halfwidth
        result["interval_lower_m"] = _round_metric(nominal - total_halfwidth)
        result["interval_upper_m"] = _round_metric(nominal + total_halfwidth)
        return result
    nominal = float(case["nominal_clearance_m"])
    total_halfwidth = ambiguity + measurement_halfwidth
    lower, upper = nominal - total_halfwidth, nominal + total_halfwidth
    result["interval_lower_m"] = _round_metric(lower)
    result["interval_upper_m"] = _round_metric(upper)
    if lower > 0.05:
        result["query_state"] = "CLEAR_OBSERVED"
    elif upper <= 0.0:
        result["query_state"] = "OCCUPIED_OBSERVED"
    else:
        result["query_state"] = "UNKNOWN"
    return result


def _arm_factors(arm: str) -> set[str]:
    return set() if arm == "NONE" else set(arm.split("_"))


def expected_oracle_receipts(case: dict[str, Any], contract: dict[str, Any]) -> dict[str, str]:
    truth = float(case["truth_clearance_m"])
    base_sigma = float(contract["base_query_sigma_m"])
    coverage_z = float(contract["coverage_z"])
    current_values = case["current_factor_error_m"]
    current_valid = case["current_factor_valid"]
    oracle_valid = case["oracle_factor_valid"]
    current_sigma = case["current_factor_sigma_m"]
    oracle_sigma = case["oracle_factor_sigma_m"]
    receipts: dict[str, str] = {}
    for mode in ORACLE_MODES:
        for arm in FACTORIAL_ARMS:
            patched = _arm_factors(arm)
            values = {factor: (0.0 if factor in patched else float(current_values[factor])) for factor in FACTOR_ORDER}
            valid = {factor: bool(current_valid[factor]) for factor in FACTOR_ORDER}
            sigmas = {factor: float(current_sigma[factor]) for factor in FACTOR_ORDER}
            if mode == "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY":
                for factor in patched:
                    valid[factor] = bool(oracle_valid[factor])
                    sigmas[factor] = float(oracle_sigma[factor])
            payload = {
                "case_id": case["id"],
                "arm": arm,
                "mode": mode,
                "factor_value_m": {factor: _round_metric(values[factor]) for factor in FACTOR_ORDER},
                "factor_valid": valid,
                "factor_sigma_m": {factor: _round_metric(sigmas[factor]) for factor in FACTOR_ORDER},
                "receipt_k_valid": bool(case["receipt_k_valid"]),
                "metric_anchor_valid": bool(case["metric_anchor_valid"]),
            }
            common_eligible = (
                bool(case["receipt_k_valid"])
                and bool(case["metric_anchor_valid"])
                and all(bool(current_valid[factor]) and bool(oracle_valid[factor]) for factor in FACTOR_ORDER)
            )
            support = {
                "case_id": case["id"],
                "unit_id": f"{case['id']}:analytic-query",
                "truth_id": f"{case['id']}:frozen-truth",
                "eligible": common_eligible if mode == "VALUE_ONLY_COMMON_SUPPORT" else (
                    bool(case["receipt_k_valid"])
                    and bool(case["metric_anchor_valid"])
                    and all(valid.values())
                ),
                "policy": "COMMON_SUPPORT_INTERSECTION" if mode == "VALUE_ONLY_COMMON_SUPPORT" else "ARM_NATIVE_FULL_BLOCK",
            }
            if not case["receipt_k_valid"]:
                output = {"terminal": "ABSTAIN_K_INVALID", "query_state": "UNKNOWN"}
            elif not case["metric_anchor_valid"]:
                output = {"terminal": "ABSTAIN_MISSING_METRIC_ANCHOR", "query_state": "UNKNOWN"}
            elif not all(valid.values()):
                output = {"terminal": "ABSTAIN_FACTOR_INVALID", "query_state": "UNKNOWN"}
            else:
                predicted = truth + sum(values.values())
                sigma = math.sqrt(base_sigma * base_sigma + sum(value * value for value in sigmas.values()))
                lower, upper = predicted - coverage_z * sigma, predicted + coverage_z * sigma
                state = "CLEAR_OBSERVED" if lower > 0.05 else "OCCUPIED_OBSERVED" if upper <= 0.0 else "UNKNOWN"
                output = {
                    "terminal": "EVALUATED",
                    "predicted_clearance_m": _round_metric(predicted),
                    "absolute_error_m": _round_metric(abs(predicted - truth)),
                    "interval_lower_m": _round_metric(lower),
                    "interval_upper_m": _round_metric(upper),
                    "query_state": state,
                }
            receipts[f"{arm}|{mode}"] = ".".join(
                [canonical_sha256(payload), canonical_sha256(output), canonical_sha256(support)]
            )
    return receipts


def validate_fixtures(fixtures: dict[str, Any], schema_bundle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _error(errors, fixtures.get("schema") != FIXTURE_SCHEMA, "FIXTURE_SCHEMA")
    normalized = fixtures.get("normalized_state", {})
    _error(errors, normalized.get("order") != STATE_ORDER, "STATE_ORDER")
    _error(errors, normalized.get("one_unit") != STATE_ONE_UNIT, "STATE_NORMALIZATION")

    rule = fixtures.get("identifiability_rule", {})
    measurement = str(rule.get("measurement_matrix", "")).lower()
    _error(errors, "whitened" not in measurement or "measurement" not in measurement, "MEASUREMENT_ONLY_INFORMATION")
    _error(errors, "exclude priors" not in measurement or "regularizers" not in measurement, "PRIOR_REGULARIZER_FIREWALL")
    _error(errors, rule.get("strong_singular_value_min") != 1.0, "STRONG_SINGULAR_MIN")
    _error(errors, rule.get("relative_singular_value_min") != 0.001, "RELATIVE_SINGULAR_MIN")
    _error(errors, rule.get("weak_trust_region_l2") != 1.0, "WEAK_TRUST_REGION")
    _error(errors, rule.get("task_ambiguity_radius_max_m") != 0.02, "TASK_AMBIGUITY_RADIUS")
    _error(errors, rule.get("measurement_noise_budget_l2_95") != 1.0, "MEASUREMENT_NOISE_BUDGET")
    _error(errors, "strong-subspace" not in str(rule.get("measurement_interval_rule", "")) or "never add or remove measurement rank" not in str(rule.get("measurement_interval_rule", "")), "MEASUREMENT_INTERVAL_SEPARATION")
    _error(errors, rule.get("contact_competition_margin_m") != 0.01, "CONTACT_COMPETITION_MARGIN")
    _error(errors, rule.get("contact_branch_spread_max_m") != 0.02, "CONTACT_BRANCH_SPREAD")

    factorial = fixtures.get("factorial_contract", {})
    _error(errors, factorial.get("factors") != ["SCALE", "SUPPORT", "BOUNDARY"], "FACTOR_SET")
    _error(errors, factorial.get("arms") != FACTORIAL_ARMS, "FACTORIAL_ARMS")
    _error(errors, factorial.get("oracle_modes") != ORACLE_MODES, "ORACLE_MODES")
    _error(errors, "K" in factorial.get("arms", []), "K_MIXED_IN_FACTORIAL")
    _error(errors, factorial.get("primary_real_comparison") != "SCALE_SUPPORT_BOUNDARY versus NONE under VALUE_ONLY_COMMON_SUPPORT", "PRIMARY_REAL_COMPARISON")

    errors.extend(validate_schema_keywords(schema_bundle))
    defs = schema_bundle.get("$defs", {})
    _error(errors, schema_bundle.get("schema_version") != SCHEMA_VERSION, "SCHEMA_BUNDLE_VERSION")
    _error(errors, not TARGET_SCHEMA_DEFS.issubset(set(defs)), "TARGET_SCHEMA_DEFS")
    examples = fixtures.get("schema_examples", {})
    _error(errors, set(examples) != TARGET_SCHEMA_DEFS, "SCHEMA_EXAMPLE_SET")
    if isinstance(defs, dict) and isinstance(examples, dict):
        for name in sorted(TARGET_SCHEMA_DEFS):
            target_schema = defs.get(name)
            if isinstance(target_schema, dict) and name in examples:
                errors.extend(validate_schema_instance(examples[name], target_schema, schema_bundle, name))

    frame = examples.get("TaroFrameReceipt", {}) if isinstance(examples, dict) else {}
    if isinstance(frame, dict):
        _error(errors, not _is_number(frame.get("max_source_timestamp_ns")) or not _is_number(frame.get("sensor_timestamp_ns")) or frame.get("max_source_timestamp_ns") > frame.get("sensor_timestamp_ns"), "TIMESTAMP_CEILING")
        _error(errors, frame.get("action_budget", {}).get("body_motion_allowed") is not False, "BODY_MOTION_BUDGET")
        _error(errors, frame.get("governance", {}).get("selection_influence") is not False, "P0_SELECTION_INFLUENCE")
        tracks = frame.get("sparse_tracks", [])
        if isinstance(tracks, list):
            evidence_ids = [item.get("evidence_id") for item in tracks if isinstance(item, dict)]
            track_ids = [item.get("track_id") for item in tracks if isinstance(item, dict)]
            _error(errors, len(evidence_ids) != len(set(evidence_ids)), "DUPLICATE_EVIDENCE_ID")
            _error(errors, len(track_ids) != len(set(track_ids)), "DUPLICATE_TRACK_ID")
        anchor = frame.get("metric_anchor", {})
        if isinstance(anchor, dict):
            source_range = anchor.get("source_timestamp_range_ns")
            _error(
                errors,
                not isinstance(source_range, list)
                or len(source_range) != 2
                or not all(_is_number(value) for value in source_range)
                or source_range[0] > source_range[1]
                or source_range[1] > frame.get("max_source_timestamp_ns", -1),
                "ANCHOR_TIMESTAMP_FIREWALL",
            )
            _error(
                errors,
                any(anchor.get(field) is not False for field in ("shared_with_factor_predictor", "shared_with_oracle_truth", "shared_with_outcome")),
                "ANCHOR_INDEPENDENCE_FIREWALL",
            )
            track_groups = {item.get("source_group_id") for item in tracks if isinstance(item, dict)} if isinstance(tracks, list) else set()
            _error(errors, anchor.get("source_independence_group") in track_groups, "ANCHOR_SOURCE_GROUP_OVERLAP")

    task_query = examples.get("TaroTaskQuery", {}) if isinstance(examples, dict) else {}
    if isinstance(task_query, dict):
        _error(
            errors,
            task_query.get("frame_id") != frame.get("frame_id")
            or task_query.get("query_id") != frame.get("query_id")
            or task_query.get("body_profile_id") != frame.get("body_profile_id")
            or task_query.get("path_id") != frame.get("path_id"),
            "TASK_QUERY_FRAME_BINDING",
        )

    posterior = examples.get("TaroFactorPosterior", {}) if isinstance(examples, dict) else {}
    if isinstance(posterior, dict):
        identifiable = posterior.get("query_identifiable")
        state = posterior.get("query_state")
        interval = posterior.get("query_interval_m", {})
        _error(errors, identifiable is False and state != "UNKNOWN", "POSTERIOR_CONTRADICTORY_IDENTIFIABILITY")
        _error(errors, state in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED"} and identifiable is not True, "POSTERIOR_OBSERVED_WITHOUT_IDENTIFIABILITY")
        if isinstance(interval, dict):
            _error(errors, not _is_number(interval.get("lower")) or not _is_number(interval.get("upper")) or interval.get("lower") > interval.get("upper"), "POSTERIOR_INTERVAL_ORDER")
        _error(errors, not _is_symmetric_psd4(posterior.get("state_covariance")), "POSTERIOR_COVARIANCE_NOT_SYMMETRIC_PSD")
        _error(
            errors,
            posterior.get("frame_id") != frame.get("frame_id")
            or posterior.get("query_id") != task_query.get("query_id"),
            "POSTERIOR_QUERY_BINDING",
        )
        _error(
            errors,
            not _is_number(posterior.get("max_source_timestamp_ns"))
            or posterior.get("max_source_timestamp_ns") != frame.get("max_source_timestamp_ns"),
            "POSTERIOR_TIMESTAMP_FIREWALL",
        )
        posterior_provenance = posterior.get("input_provenance")
        anchor_identity = frame.get("metric_anchor", {}).get("identity")
        _error(
            errors,
            not isinstance(posterior_provenance, list)
            or frame.get("frame_id") not in posterior_provenance
            or anchor_identity not in posterior_provenance
            or len(posterior_provenance) != len(set(posterior_provenance)),
            "POSTERIOR_PROVENANCE",
        )
        factor_reference = posterior.get("factor_reference", {})
        factor_identity = frame.get("factor_identity", {})
        _error(
            errors,
            not isinstance(factor_reference, dict)
            or factor_reference.get("schema_id") != factor_identity.get("schema_id")
            or factor_reference.get("content_sha256") != factor_identity.get("content_sha256")
            or factor_reference.get("uncertainty_present") is not True,
            "POSTERIOR_FACTOR_BINDING",
        )

    candidate = examples.get("TaroObservationCandidate", {}) if isinstance(examples, dict) else {}
    if isinstance(candidate, dict):
        _error(errors, candidate.get("frame_id") != frame.get("frame_id") or candidate.get("frame_id") != posterior.get("frame_id") or candidate.get("query_id") != task_query.get("query_id") or candidate.get("query_id") != posterior.get("query_id"), "CANDIDATE_QUERY_BINDING")
        _error(errors, not _is_number(candidate.get("max_source_timestamp_ns")) or candidate.get("max_source_timestamp_ns") != frame.get("max_source_timestamp_ns") or candidate.get("max_source_timestamp_ns") != posterior.get("max_source_timestamp_ns"), "CANDIDATE_TIMESTAMP_FIREWALL")
        provenance = candidate.get("input_provenance")
        _error(errors, not isinstance(provenance, list) or provenance != posterior.get("input_provenance") or frame.get("frame_id") not in provenance or len(provenance) != len(set(provenance)), "CANDIDATE_PROVENANCE")
        translation = candidate.get("intended_camera_delta", {}).get("translation_m")
        predicted_baseline = candidate.get("predicted_baseline_m")
        _error(
            errors,
            not isinstance(translation, list)
            or len(translation) != 3
            or any(not _is_number(value) for value in translation)
            or not _is_number(predicted_baseline)
            or abs(math.sqrt(sum(float(value) ** 2 for value in translation)) - float(predicted_baseline)) > 1e-12,
            "CANDIDATE_PREDICTED_BASELINE",
        )
        _error(errors, candidate.get("requires_body_motion") is True and (candidate.get("allowed") is not False or candidate.get("filter_reason") != "BODY_MOTION_FORBIDDEN"), "CANDIDATE_BODY_MOTION_FILTER")
        _error(errors, candidate.get("allowed") is True and candidate.get("filter_reason") != "NONE", "CANDIDATE_ALLOWED_REASON")
        _error(errors, candidate.get("allowed") is False and candidate.get("filter_reason") == "NONE", "CANDIDATE_DISALLOWED_REASON")
        if candidate.get("realized") is True:
            _error(errors, candidate.get("allowed") is not True or not isinstance(candidate.get("actual_receipt_frame_id"), str) or not candidate.get("actual_receipt_frame_id") or not _is_number(candidate.get("realized_baseline_m")), "CANDIDATE_REALIZED_RECEIPT")
        else:
            _error(errors, candidate.get("actual_receipt_frame_id") is not None or candidate.get("realized_baseline_m") is not None, "CANDIDATE_UNREALIZED_NULLS")

    cases = fixtures.get("identifiability_cases")
    _error(errors, _ids(cases) != IDENTIFIABILITY_CASES, "IDENTIFIABILITY_CASE_SET")
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                errors.append("IDENTIFIABILITY_CASE_OBJECT")
                continue
            case_id = str(case.get("id"))
            _error(errors, not _matrix4(case.get("measurement_jacobian_whitened")), f"MEASUREMENT_MATRIX:{case_id}")
            branches = case.get("query_jacobian_branches_m")
            _error(errors, not isinstance(branches, list) or not branches or any(not isinstance(row, list) or len(row) != 4 or not all(_is_number(item) for item in row) for row in branches), f"QUERY_JACOBIAN:{case_id}")
            expected = case.get("expected", {})
            _error(errors, expected.get("query_state") not in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}, f"QUERY_STATE:{case_id}")
            try:
                calculated = evaluate_identifiability_case(case, rule)
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(f"IDENTIFIABILITY_CALCULATION:{case_id}")
                calculated = None
            if calculated is not None:
                _error(errors, expected != calculated, f"IDENTIFIABILITY_TRUTH:{case_id}")
            receipt = case.get("receipt", {})
            if case_id in {"missing_anchor_unknown", "anchor_shuffle_unknown", "wrong_k_unknown", "clock_offset_unknown"}:
                _error(errors, expected.get("query_identifiable") is not False or expected.get("query_state") != "UNKNOWN", f"DEGENERATE_NOT_UNKNOWN:{case_id}")
            if case_id == "missing_anchor_unknown":
                _error(errors, receipt.get("anchor_valid") is not False or expected.get("reason_code") != "MISSING_METRIC_ANCHOR", "MISSING_ANCHOR_EXPECTATION")
            if case_id == "full_state_underdetermined_query_identifiable_clear":
                _error(errors, expected.get("query_identifiable") is not True or expected.get("query_state") != "CLEAR_OBSERVED", "QUERY_INVARIANT_POSITIVE_CONTROL")
            if case_id == "nonsmooth_contact_switch_unknown":
                _error(errors, expected.get("reason_code") != "NONSMOOTH_CONTACT_SWITCH", "NONSMOOTH_EXPECTATION")

    action_cases = fixtures.get("action_filter_cases")
    _error(errors, _ids(action_cases) != ACTION_CASES, "ACTION_CASE_SET")
    if isinstance(action_cases, list):
        body_motion = next((item for item in action_cases if isinstance(item, dict) and item.get("id") == "step_sideways_forbidden"), {})
        _error(errors, body_motion.get("requires_body_motion") is not True or body_motion.get("expected_allowed") is not False, "BODY_MOTION_FILTER")

    mechanics = fixtures.get("factor_oracle_mechanics_cases")
    _error(errors, _ids(mechanics) != FACTOR_MECHANICS_CASES, "FACTOR_MECHANICS_CASE_SET")
    oracle_contract = fixtures.get("oracle_numeric_contract", {})
    _error(errors, oracle_contract.get("factor_order") != FACTOR_ORDER, "ORACLE_FACTOR_ORDER")
    _error(errors, oracle_contract.get("base_query_sigma_m") != 0.005, "ORACLE_BASE_SIGMA")
    _error(errors, oracle_contract.get("coverage_z") != 1.96, "ORACLE_COVERAGE_Z")
    _error(errors, oracle_contract.get("numeric_tolerance_m") != 1e-09, "ORACLE_NUMERIC_TOLERANCE")
    if isinstance(mechanics, list):
        for case in mechanics:
            if not isinstance(case, dict):
                errors.append("FACTOR_MECHANICS_CASE_OBJECT")
                continue
            case_id = str(case.get("id"))
            for field in ("current_factor_error_m", "current_factor_sigma_m", "oracle_factor_sigma_m"):
                values = case.get(field)
                _error(errors, not isinstance(values, dict) or list(values) != FACTOR_ORDER or any(not _is_number(value) or ("sigma" in field and value < 0) for value in values.values()), f"FACTOR_NUMERIC_BLOCK:{case_id}:{field}")
            for field in ("current_factor_valid", "oracle_factor_valid"):
                values = case.get(field)
                _error(errors, not isinstance(values, dict) or list(values) != FACTOR_ORDER or any(not isinstance(value, bool) for value in values.values()), f"FACTOR_VALIDITY_BLOCK:{case_id}:{field}")
            _error(errors, not _is_number(case.get("truth_clearance_m")), f"FACTOR_TRUTH:{case_id}")
            _error(errors, not isinstance(case.get("receipt_k_valid"), bool) or not isinstance(case.get("metric_anchor_valid"), bool), f"FACTOR_RECEIPT:{case_id}")
            if any(code.startswith((f"FACTOR_NUMERIC_BLOCK:{case_id}", f"FACTOR_VALIDITY_BLOCK:{case_id}", f"FACTOR_TRUTH:{case_id}", f"FACTOR_RECEIPT:{case_id}")) for code in errors):
                continue
            try:
                calculated_receipts = expected_oracle_receipts(case, oracle_contract)
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(f"FACTOR_RECEIPT_CALCULATION:{case_id}")
                continue
            _error(errors, case.get("arm_receipts") != calculated_receipts, f"FACTOR_ARM_RECEIPTS:{case_id}")
            causal = case.get("causal_factors", [])
            _error(errors, not isinstance(causal, list) or any(factor not in FACTOR_ORDER for factor in causal) or len(causal) != len(set(causal)), f"CAUSAL_FACTOR_SET:{case_id}")
            derived_causal = [
                factor
                for factor in FACTOR_ORDER
                if abs(float(case["current_factor_error_m"][factor])) > float(oracle_contract["numeric_tolerance_m"])
            ]
            _error(errors, causal != derived_causal, f"CAUSAL_FACTOR_TRUTH:{case_id}")
            if case.get("receipt_k_valid") and case.get("metric_anchor_valid"):
                errors_by_arm = {
                    arm: abs(sum(float(case["current_factor_error_m"][factor]) for factor in FACTOR_ORDER if factor not in _arm_factors(arm)))
                    for arm in FACTORIAL_ARMS
                }
                minimum = min(errors_by_arm.values())
                minimizing = [arm for arm in FACTORIAL_ARMS if abs(errors_by_arm[arm] - minimum) <= float(oracle_contract["numeric_tolerance_m"])]
                _error(errors, case.get("expected_primary_minimizing_arms") != minimizing, f"PRIMARY_MINIMIZING_ARMS:{case_id}")
    return errors


def validate_protocol(protocol: dict[str, Any], fixtures: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _error(errors, protocol.get("schema_version") != "blindassist.research_protocol.v1", "PROTOCOL_SCHEMA")
    _error(errors, protocol.get("protocol_id") != PROTOCOL_ID, "PROTOCOL_ID")
    _error(errors, protocol.get("profile") != "CANARY_LITE", "PROFILE")
    _error(errors, protocol.get("governance_policy_sha256") != GOVERNANCE_POLICY_DIGEST, "GOVERNANCE_POLICY_DIGEST")
    _error(errors, protocol.get("stage") != "DISCOVERY", "STAGE")
    _error(errors, protocol.get("scientific_status") != "NOT_RUN", "SCIENTIFIC_STATUS")
    _error(errors, protocol.get("outcome_access_started") is not False, "OUTCOME_ACCESS")

    state = protocol.get("state_contract", {})
    _error(errors, state.get("open_state_order") != STATE_ORDER, "PROTOCOL_STATE_ORDER")
    _error(errors, state.get("normalization_one_unit") != STATE_ONE_UNIT, "PROTOCOL_STATE_NORMALIZATION")
    _error(errors, state.get("fixed_controls") != ["delta_k", "delta_pose", "delta_time"], "FIXED_CONTROL_SET")
    _error(errors, state.get("boundary_role") != "O0_FACTORIAL_TREATMENT_NOT_GAUGEFIX_STATE", "BOUNDARY_STATE_ROLE")

    identifiability = protocol.get("identifiability_contract", {})
    _error(errors, identifiability.get("information_source") != "MEASUREMENT_ONLY_WHITENED_RESIDUAL_JACOBIAN", "IDENTIFIABILITY_INFORMATION_SOURCE")
    _error(errors, identifiability.get("prior_damping_regularizer_may_add_rank") is not False, "SYNTHETIC_RANK_FORBIDDEN")
    _error(errors, identifiability.get("finite_task_ambiguity_primary") is not True, "FINITE_TASK_AMBIGUITY")

    factorial = protocol.get("factorial_contract", {})
    _error(errors, factorial.get("arms") != FACTORIAL_ARMS, "PROTOCOL_FACTORIAL_ARMS")
    _error(errors, factorial.get("oracle_modes") != ORACLE_MODES, "PROTOCOL_ORACLE_MODES")
    _error(errors, factorial.get("receipt_k_corruption_is_separate_control") is not True, "PROTOCOL_K_CONTROL")
    _error(errors, factorial.get("factor_specific_arms_are_diagnostic") is not True, "FACTOR_ARM_SELECTION_FIREWALL")

    gates = protocol.get("o0m_gates")
    gate_ids = _ids(gates)
    _error(errors, gate_ids != O0M_GATES, "O0M_GATE_SET")
    _error(errors, not isinstance(gates, list) or canonical_sha256(gates) != O0M_GATES_SHA256, "O0M_GATE_CONTRACT_DRIFT")
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, dict):
                errors.append("O0M_GATE_OBJECT")
                continue
            gate_id = str(gate.get("id"))
            _error(errors, set(gate) != {"id", "condition", "pass", "failure_terminal"}, f"O0M_GATE_FIELDS:{gate_id}")
            for field in ("condition", "pass", "failure_terminal"):
                _error(errors, not isinstance(gate.get(field), str) or not gate[field].strip(), f"O0M_GATE_EMPTY:{gate_id}:{field}")

    o0r = protocol.get("o0r_admission", {})
    _error(errors, o0r.get("current_terminal") != "TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE", "O0R_CURRENT_TERMINAL")
    _error(errors, o0r.get("execution_authority") is not False, "O0R_AUTHORITY")
    _error(errors, o0r.get("complete_factor_truth_required") is not True, "O0R_COMPLETE_FACTOR_TRUTH")
    _error(errors, o0r.get("fresh_paired_outcome_required") is not True, "O0R_FRESH_PAIRED")

    authority = protocol.get("execution_authority", {})
    allowed_true = {"protocol_lock", "static_validation"}
    if not isinstance(authority, dict):
        errors.append("EXECUTION_AUTHORITY_OBJECT")
    else:
        _error(errors, any(value is not True for key, value in authority.items() if key in allowed_true), "STATIC_AUTHORITY_MISSING")
        _error(errors, any(value is not False for key, value in authority.items() if key not in allowed_true), "EXECUTION_AUTHORITY_EXCEEDED")
        _error(errors, set(authority) != {
            "protocol_lock", "static_validation", "o0m_implementation", "o0m_execution", "real_data_read",
            "factor_injection", "gauge_solver", "training", "checkpoint", "active_prompt", "android_device",
            "calibration", "confirmation", "deployment", "default_app"
        }, "EXECUTION_AUTHORITY_KEYS")

    successor = protocol.get("unique_successor", {})
    _error(errors, successor.get("id") != SUCCESSOR_ID, "SUCCESSOR_ID")
    _error(errors, successor.get("execution_authority") is not False, "SUCCESSOR_AUTHORITY")
    _error(errors, protocol.get("fixture_suite_id") != fixtures.get("suite_id"), "FIXTURE_SUITE_BINDING")
    return errors


def validate_bindings(protocol: dict[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    bindings = protocol.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        return ["BINDINGS"]
    for item in bindings:
        if not isinstance(item, dict):
            errors.append("BINDING_OBJECT")
            continue
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append("BINDING_FIELDS")
            continue
        path = repo_root / relative
        if not path.is_file():
            errors.append(f"BINDING_ABSENT:{relative}")
        elif sha256_file(path) != expected.upper():
            errors.append(f"BINDING_SHA:{relative}")
    return errors


def validate_runtime_absence(repo_root: Path) -> list[str]:
    errors = [f"PROHIBITED_RUNTIME_PRESENT:{relative}" for relative in PROHIBITED_RUNTIME_PATHS if (repo_root / relative).exists()]
    module_root = repo_root / "scripts/research/taro"
    if module_root.is_dir():
        for path in module_root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(repo_root).as_posix()
            if relative not in P0_MODULE_ALLOWLIST:
                errors.append(f"UNAUTHORIZED_P0_MODULE_FILE:{relative}")
    return errors


def validate_static_contract(
    protocol: dict[str, Any],
    schema_bundle: dict[str, Any],
    fixtures: dict[str, Any],
) -> list[str]:
    errors = validate_fixtures(fixtures, schema_bundle)
    errors.extend(validate_protocol(protocol, fixtures))
    return sorted(set(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schema-bundle", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    protocol = load_json(args.protocol.resolve())
    schema_bundle = load_json(args.schema_bundle.resolve())
    fixtures = load_json(args.fixtures.resolve())
    errors = validate_static_contract(protocol, schema_bundle, fixtures)
    errors.extend(validate_bindings(protocol, repo_root))
    errors.extend(validate_runtime_absence(repo_root))
    result = {
        "schema": "blindassist.taro.p0.static_validation.v1",
        "status": "VALID" if not errors else "INVALID",
        "errors": sorted(set(errors)),
        "scientific_status": "NOT_RUN",
        "o0m_execution_authority": False,
        "o0r_current_terminal": "TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
