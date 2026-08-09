#!/usr/bin/env python3
"""Independent analytic mechanics for the TARO O0M synthetic canary.

This module consumes pre-whitened analytic matrices and explicit factor blocks.
It never imports the P0/O0M static evaluators and never consumes verifier truth
or frozen expected outputs as solver inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


FACTORS = ("SCALE", "SUPPORT", "BOUNDARY")
ARMS = (
    "NONE",
    "SCALE",
    "SUPPORT",
    "BOUNDARY",
    "SCALE_SUPPORT",
    "SCALE_BOUNDARY",
    "SUPPORT_BOUNDARY",
    "SCALE_SUPPORT_BOUNDARY",
)
MODES = ("VALUE_ONLY_COMMON_SUPPORT", "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY")

IDENTIFIABILITY_KEYS = {
    "id",
    "motion",
    "measurement_jacobian_whitened",
    "query_jacobian_branches_m",
    "nominal_clearance_m",
    "receipt",
    "active_contact_clearances_m",
    "branch_nominal_clearances_m",
}
IDENTIFIABILITY_REQUIRED_KEYS = IDENTIFIABILITY_KEYS - {"branch_nominal_clearances_m"}
SCENE_KEYS = {
    "id",
    "family_id",
    "query_id",
    "observed_base_mean_m",
    "current_factor_error_m",
    "current_factor_valid",
    "oracle_factor_valid",
    "factor_provenance",
    "oracle_provenance",
    "factor_identity_sha256",
    "anchor_identity",
    "max_source_timestamp_ns",
}
NUMERIC_KEYS = {
    "factor_order",
    "arms",
    "oracle_modes",
    "sigma_measurement_m",
    "sigma_factor_baseline_m",
    "sigma_factor_oracle_m",
    "interval_multiplier",
    "interval_semantics",
    "numeric_atol_m",
    "hash_quantization_decimal_places",
    "clear_margin_m",
    "occupied_margin_m",
}
ACTION_KEYS = {"id", "requires_body_motion"}
RECEIPT_REASONS = (
    ("anchor_valid", "MISSING_METRIC_ANCHOR"),
    ("anchor_identity_matches_parent", "ANCHOR_SHUFFLED"),
    ("k_valid", "K_INVALID"),
    ("transform_valid", "TRANSFORM_INVALID"),
    ("clock_valid", "CLOCK_INVALID"),
    ("factor_valid", "FACTOR_INVALID"),
)


class MechanicsError(RuntimeError):
    """Fail-closed input or numeric contract error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise MechanicsError(code, message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _round(value: float, places: int) -> float:
    result = round(float(value), places)
    return 0.0 if result == -0.0 else result


def _exact_object(value: Any, allowed: set[str], required: set[str], code: str) -> dict[str, Any]:
    require(isinstance(value, dict), code, "input must be an object")
    keys = set(value)
    require(keys <= allowed and required <= keys, code, f"input keys violate whitelist: {sorted(keys)}")
    return value


def _factor_block(value: Any, code: str) -> dict[str, Any]:
    require(isinstance(value, dict) and tuple(value) == FACTORS, code, "factor block order/keys mismatch")
    return value


def _projector(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((4, 4), dtype=np.float64)
    return vectors.T @ vectors


def _matrix_payload(matrix: np.ndarray, places: int = 12) -> list[list[float]]:
    return [[_round(item, places) for item in row] for row in matrix.tolist()]


def solve_identifiability(case_input: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    """Solve one pre-whitened 4-D analytic identifiability case."""

    case = _exact_object(case_input, IDENTIFIABILITY_KEYS, IDENTIFIABILITY_REQUIRED_KEYS, "IDENT_INPUT_KEYS")
    require(isinstance(rule, dict), "IDENT_RULE", "identifiability rule must be an object")
    require("prior" not in rule and "damping" not in rule and "regularizer" not in rule, "PRIOR_INFORMATION_FORBIDDEN", "prior-like information is forbidden")
    try:
        matrix = np.asarray(case["measurement_jacobian_whitened"], dtype=np.float64)
        branches = np.asarray(case["query_jacobian_branches_m"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MechanicsError("IDENT_MATRIX", "matrix conversion failed") from exc
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), "IDENT_MATRIX", "measurement matrix must be finite 4x4")
    require(branches.ndim == 2 and branches.shape[1] == 4 and len(branches) > 0 and np.isfinite(branches).all(), "QUERY_JACOBIAN", "query branches must be finite Nx4")
    try:
        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=True)
    except np.linalg.LinAlgError as exc:
        raise MechanicsError("SVD_FAILED", "SVD did not converge") from exc

    sigma_max = float(singular_values[0]) if len(singular_values) else 0.0
    threshold = max(float(rule["strong_singular_value_min"]), float(rule["relative_singular_value_min"]) * sigma_max)
    strong_mask = singular_values >= threshold
    weak_mask = singular_values < threshold
    null_mask = singular_values <= 1e-10
    strong_vectors = vt[strong_mask]
    weak_vectors = vt[weak_mask]
    null_vectors = vt[null_mask]

    def projected_radius(branch: np.ndarray, vectors: np.ndarray, scale: float) -> float:
        if vectors.size == 0:
            return 0.0
        return scale * float(np.linalg.norm(vectors @ branch, ord=2))

    weak_radius = float(rule["weak_trust_region_l2"])
    ambiguity = max(projected_radius(branch, weak_vectors, weak_radius) for branch in branches)
    null_ambiguity = max(projected_radius(branch, null_vectors, weak_radius) for branch in branches)
    if strong_vectors.size == 0:
        measurement_halfwidth = 0.0
    else:
        strong_singular = singular_values[strong_mask]
        noise_budget = float(rule["measurement_noise_budget_l2_95"])
        measurement_halfwidth = max(
            noise_budget * float(np.linalg.norm((strong_vectors @ branch) / strong_singular, ord=2))
            for branch in branches
        )

    strong_rank = int(np.count_nonzero(strong_mask))
    update_decision = "FREEZE" if strong_rank == 0 else "UPDATE_STRONG_SUBSPACE"
    actual: dict[str, Any] = {
        "query_identifiable": True,
        "query_state": "UNKNOWN",
        "update_decision": update_decision,
        "reason_code": "NONE",
        "strong_rank": strong_rank,
        "task_ambiguity_radius_m": _round(ambiguity, 9),
        "measurement_halfwidth_m_95": _round(measurement_halfwidth, 9),
        "limiting_null_invariant": null_ambiguity <= 1e-10,
        "interval_lower_m": None,
        "interval_upper_m": None,
    }
    diagnostics = {
        "singular_values": [_round(item, 12) for item in singular_values.tolist()],
        "threshold": _round(threshold, 12),
        "strong_projector": _matrix_payload(_projector(strong_vectors)),
        "weak_projector": _matrix_payload(_projector(weak_vectors)),
    }
    diagnostics["strong_projector_sha256"] = canonical_sha256(diagnostics["strong_projector"])
    diagnostics["weak_projector_sha256"] = canonical_sha256(diagnostics["weak_projector"])

    receipt = case.get("receipt")
    require(isinstance(receipt, dict), "RECEIPT", "receipt must be an object")
    for field, reason in RECEIPT_REASONS:
        if receipt.get(field) is not True:
            actual.update(query_identifiable=False, query_state="UNKNOWN", update_decision="ABSTAIN", reason_code=reason)
            return {"actual": actual, "diagnostics": diagnostics}

    active = case.get("active_contact_clearances_m")
    branch_nominal = case.get("branch_nominal_clearances_m", [])
    competition = (
        isinstance(active, list)
        and len(active) >= 2
        and all(_finite(value) for value in active)
        and abs(sorted(float(value) for value in active)[1] - sorted(float(value) for value in active)[0])
        <= float(rule["contact_competition_margin_m"])
    )
    branch_conflict = (
        isinstance(branch_nominal, list)
        and len(branch_nominal) >= 2
        and all(_finite(value) for value in branch_nominal)
        and max(float(value) for value in branch_nominal) - min(float(value) for value in branch_nominal)
        > float(rule["contact_branch_spread_max_m"])
    )
    if competition and branch_conflict:
        actual.update(query_identifiable=False, query_state="UNKNOWN", reason_code="NONSMOOTH_CONTACT_SWITCH")
        return {"actual": actual, "diagnostics": diagnostics}

    nominal = float(case["nominal_clearance_m"])
    total_halfwidth = ambiguity + measurement_halfwidth
    actual["interval_lower_m"] = _round(nominal - total_halfwidth, 9)
    actual["interval_upper_m"] = _round(nominal + total_halfwidth, 9)
    if ambiguity > float(rule["task_ambiguity_radius_max_m"]) + 1e-12:
        reason = {
            "pure_rotation": "PURE_ROTATION_SCALE_WEAK",
            "small_baseline": "BASELINE_TOO_SMALL",
        }.get(str(case.get("motion")), "WEAK_QUERY_DIRECTION")
        actual.update(query_identifiable=False, query_state="UNKNOWN", reason_code=reason)
        return {"actual": actual, "diagnostics": diagnostics}
    if nominal - total_halfwidth > 0.05:
        actual["query_state"] = "CLEAR_OBSERVED"
    elif nominal + total_halfwidth <= 0.0:
        actual["query_state"] = "OCCUPIED_OBSERVED"
    return {"actual": actual, "diagnostics": diagnostics}


def _arm_factors(arm: str) -> set[str]:
    require(arm in ARMS, "FACTORIAL_ARM", f"unknown arm: {arm}")
    return set() if arm == "NONE" else set(arm.split("_"))


def apply_factorial_arm(scene_input: dict[str, Any], numeric: dict[str, Any], arm: str, mode: str) -> dict[str, Any]:
    """Apply one factor intervention without reading verifier truth."""

    scene = _exact_object(scene_input, SCENE_KEYS, SCENE_KEYS, "SCENE_INPUT_KEYS")
    _exact_object(numeric, NUMERIC_KEYS, NUMERIC_KEYS, "NUMERIC_INPUT_KEYS")
    require(tuple(numeric["factor_order"]) == FACTORS and tuple(numeric["arms"]) == ARMS and tuple(numeric["oracle_modes"]) == MODES, "NUMERIC_ENUMS", "numeric enum contract mismatch")
    require(mode in MODES, "ORACLE_MODE", f"unknown oracle mode: {mode}")
    patched = _arm_factors(arm)
    current_error = _factor_block(scene["current_factor_error_m"], "CURRENT_FACTOR_ERROR")
    current_valid = _factor_block(scene["current_factor_valid"], "CURRENT_FACTOR_VALID")
    oracle_valid = _factor_block(scene["oracle_factor_valid"], "ORACLE_FACTOR_VALID")
    current_provenance = _factor_block(scene["factor_provenance"], "CURRENT_PROVENANCE")
    oracle_provenance = _factor_block(scene["oracle_provenance"], "ORACLE_PROVENANCE")
    observed_base = float(scene["observed_base_mean_m"])
    places = int(numeric["hash_quantization_decimal_places"])

    values = {factor: (0.0 if factor in patched else float(current_error[factor])) for factor in FACTORS}
    validity = {factor: bool(current_valid[factor]) for factor in FACTORS}
    sigma = {factor: float(numeric["sigma_factor_baseline_m"]) for factor in FACTORS}
    provenance = dict(current_provenance)
    if mode == "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY":
        for factor in patched:
            validity[factor] = bool(oracle_valid[factor])
            sigma[factor] = float(numeric["sigma_factor_oracle_m"])
            provenance[factor] = oracle_provenance[factor]

    payload = {
        "scene_id": scene["id"],
        "query_id": scene["query_id"],
        "observed_base_mean_m": _round(observed_base, places),
        "arm": arm,
        "mode": mode,
        "factor_error_m": {factor: _round(values[factor], places) for factor in FACTORS},
        "factor_valid": validity,
        "factor_sigma_m": {factor: _round(sigma[factor], places) for factor in FACTORS},
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
        "factor_sigma_m": {factor: _round(sigma[factor], places) for factor in FACTORS},
        "factor_provenance": provenance,
        "factor_identity_sha256": scene["factor_identity_sha256"],
        "anchor_identity": scene["anchor_identity"],
        "max_source_timestamp_ns": scene["max_source_timestamp_ns"],
        "observed_base_mean_m": _round(observed_base, places),
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
            "mean_m": _round(mean, places),
            "halfwidth_m": _round(halfwidth, places),
            "lower_m": _round(lower, places),
            "upper_m": _round(upper, places),
        }
    record = {
        "payload_sha256": canonical_sha256(payload),
        "output_sha256": canonical_sha256(output),
        "common_support_sha256": canonical_sha256(support),
        "intervention_guard_sha256": canonical_sha256(guard),
        "output": output,
    }
    runtime_input = {"scene": scene, "numeric": numeric, "arm": arm, "mode": mode}
    return {
        "record": record,
        "runtime_input_sha256": canonical_sha256(runtime_input),
        "declared_factors": sorted(patched),
    }


def evaluate_action_filter(action_input: dict[str, Any]) -> dict[str, Any]:
    action = _exact_object(action_input, ACTION_KEYS, ACTION_KEYS, "ACTION_INPUT_KEYS")
    if action["requires_body_motion"] is True:
        return {"allowed": False, "reason": "BODY_MOTION_FORBIDDEN"}
    require(action["requires_body_motion"] is False, "ACTION_BODY_MOTION", "requires_body_motion must be boolean")
    return {"allowed": True, "reason": "NONE"}


def compute_suite(runtime_input: dict[str, Any]) -> dict[str, Any]:
    required = {"identifiability_rule", "identifiability_cases", "numeric_contract", "factorial_scenes", "action_filter_cases"}
    suite = _exact_object(runtime_input, required, required, "SUITE_INPUT_KEYS")
    identifiability = [solve_identifiability(case, suite["identifiability_rule"]) for case in suite["identifiability_cases"]]
    factorial = [
        apply_factorial_arm(scene, suite["numeric_contract"], arm, mode)
        for scene in suite["factorial_scenes"]
        for mode in MODES
        for arm in ARMS
    ]
    actions = [evaluate_action_filter(action) for action in suite["action_filter_cases"]]
    return {"identifiability": identifiability, "factorial": factorial, "actions": actions}
