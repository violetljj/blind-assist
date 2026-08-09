#!/usr/bin/env python3
"""Disjoint unit tests for the TARO O0M runtime implementation."""

from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path

import numpy as np

from scripts.research.taro_o0m_runtime.o0m_mechanics import (
    MechanicsError,
    apply_factorial_arm,
    canonical_json_bytes,
    canonical_sha256,
    evaluate_action_filter,
    solve_identifiability,
)


def rule() -> dict:
    return {
        "strong_singular_value_min": 1.0,
        "relative_singular_value_min": 0.001,
        "weak_trust_region_l2": 1.0,
        "task_ambiguity_radius_max_m": 0.02,
        "measurement_noise_budget_l2_95": 1.0,
        "contact_competition_margin_m": 0.01,
        "contact_branch_spread_max_m": 0.02,
    }


def receipt(**patch: bool) -> dict:
    value = {
        "anchor_valid": True,
        "anchor_identity_matches_parent": True,
        "k_valid": True,
        "transform_valid": True,
        "clock_valid": True,
        "factor_valid": True,
    }
    value.update(patch)
    return value


def ident_case(matrix: list[list[float]], query: list[float], *, motion: str = "mixed_6dof", nominal: float = 0.12) -> dict:
    return {
        "id": "impl_unit_ident",
        "motion": motion,
        "measurement_jacobian_whitened": matrix,
        "query_jacobian_branches_m": [query],
        "nominal_clearance_m": nominal,
        "receipt": receipt(),
        "active_contact_clearances_m": [nominal],
    }


def numeric() -> dict:
    return {
        "factor_order": ["SCALE", "SUPPORT", "BOUNDARY"],
        "arms": ["NONE", "SCALE", "SUPPORT", "BOUNDARY", "SCALE_SUPPORT", "SCALE_BOUNDARY", "SUPPORT_BOUNDARY", "SCALE_SUPPORT_BOUNDARY"],
        "oracle_modes": ["VALUE_ONLY_COMMON_SUPPORT", "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY"],
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


def scene() -> dict:
    return {
        "id": "impl_unit_scale",
        "family_id": "IMPL_UNIT_DISJOINT",
        "query_id": "impl_unit_scale:query",
        "observed_base_mean_m": 0.04,
        "current_factor_error_m": {"SCALE": -0.08, "SUPPORT": 0.0, "BOUNDARY": 0.0},
        "current_factor_valid": {"SCALE": True, "SUPPORT": True, "BOUNDARY": True},
        "oracle_factor_valid": {"SCALE": True, "SUPPORT": True, "BOUNDARY": True},
        "factor_provenance": {"SCALE": "unit:scale:current", "SUPPORT": "unit:support:current", "BOUNDARY": "unit:boundary:current"},
        "oracle_provenance": {"SCALE": "unit:scale:oracle", "SUPPORT": "unit:support:oracle", "BOUNDARY": "unit:boundary:oracle"},
        "factor_identity_sha256": "A" * 64,
        "anchor_identity": "impl-unit-anchor",
        "max_source_timestamp_ns": 123,
    }


class O0MMechanicsTests(unittest.TestCase):
    def test_canonical_bytes_are_stable(self) -> None:
        left = {"b": [2, 1], "a": -0.0}
        right = {"a": -0.0, "b": [2, 1]}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(canonical_sha256(left), canonical_sha256(right))

    def test_full_rank_clear(self) -> None:
        case = ident_case(np.diag([4.0, 4.0, 4.0, 4.0]).tolist(), [0.1, 0.0, 0.0, 0.0])
        actual = solve_identifiability(case, rule())["actual"]
        self.assertEqual(actual["strong_rank"], 4)
        self.assertTrue(actual["query_identifiable"])
        self.assertEqual(actual["query_state"], "CLEAR_OBSERVED")

    def test_rank_deficient_query_invariant(self) -> None:
        case = ident_case(np.diag([4.0, 0.0, 0.0, 4.0]).tolist(), [0.1, 0.0, 0.0, 0.02])
        actual = solve_identifiability(case, rule())["actual"]
        self.assertEqual(actual["strong_rank"], 2)
        self.assertEqual(actual["task_ambiguity_radius_m"], 0.0)
        self.assertTrue(actual["query_identifiable"])

    def test_weak_query_direction_is_unknown(self) -> None:
        case = ident_case(np.zeros((4, 4)).tolist(), [0.3, 0.0, 0.0, 0.0], motion="static")
        actual = solve_identifiability(case, rule())["actual"]
        self.assertFalse(actual["query_identifiable"])
        self.assertEqual(actual["reason_code"], "WEAK_QUERY_DIRECTION")
        self.assertEqual(actual["query_state"], "UNKNOWN")

    def test_invalid_receipt_abstains(self) -> None:
        case = ident_case(np.eye(4).tolist(), [0.1, 0.0, 0.0, 0.0])
        case["receipt"] = receipt(k_valid=False)
        actual = solve_identifiability(case, rule())["actual"]
        self.assertEqual((actual["update_decision"], actual["reason_code"]), ("ABSTAIN", "K_INVALID"))

    def test_nonsmooth_contact_switch_is_unknown(self) -> None:
        case = ident_case(np.eye(4).tolist(), [0.1, 0.0, 0.0, 0.0])
        case["active_contact_clearances_m"] = [0.04, 0.045]
        case["branch_nominal_clearances_m"] = [0.03, 0.06]
        actual = solve_identifiability(case, rule())["actual"]
        self.assertEqual(actual["reason_code"], "NONSMOOTH_CONTACT_SWITCH")
        self.assertEqual(actual["query_state"], "UNKNOWN")

    def test_non_axis_reparameterization_preserves_verdict_and_projector(self) -> None:
        base_case = ident_case(np.diag([4.0, 0.0, 0.0, 4.0]).tolist(), [0.1, 0.0, 0.0, 0.02])
        base = solve_identifiability(base_case, rule())
        q = np.asarray([[0.8, -0.6, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 0.8, -0.6], [0.0, 0.0, 0.6, 0.8]])
        changed_case = copy.deepcopy(base_case)
        changed_case["measurement_jacobian_whitened"] = (np.asarray(base_case["measurement_jacobian_whitened"]) @ q.T).tolist()
        changed_case["query_jacobian_branches_m"] = (np.asarray(base_case["query_jacobian_branches_m"]) @ q.T).tolist()
        changed = solve_identifiability(changed_case, rule())
        self.assertEqual(base["actual"], changed["actual"])
        mapped = q.T @ np.asarray(changed["diagnostics"]["strong_projector"]) @ q
        self.assertTrue(np.allclose(mapped, np.asarray(base["diagnostics"]["strong_projector"]), atol=1e-10, rtol=0.0))

    def test_factorial_value_only_repairs_only_declared_value(self) -> None:
        base = apply_factorial_arm(scene(), numeric(), "NONE", "VALUE_ONLY_COMMON_SUPPORT")
        patched = apply_factorial_arm(scene(), numeric(), "SCALE", "VALUE_ONLY_COMMON_SUPPORT")
        self.assertEqual(base["record"]["output"]["mean_m"], 0.04)
        self.assertEqual(patched["record"]["output"]["mean_m"], 0.12)
        self.assertEqual(base["record"]["output"]["halfwidth_m"], patched["record"]["output"]["halfwidth_m"])
        self.assertEqual(base["record"]["common_support_sha256"], patched["record"]["common_support_sha256"])

    def test_full_block_can_shrink_uncertainty(self) -> None:
        value = apply_factorial_arm(scene(), numeric(), "SCALE", "VALUE_ONLY_COMMON_SUPPORT")["record"]["output"]
        full = apply_factorial_arm(scene(), numeric(), "SCALE", "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY")["record"]["output"]
        self.assertLess(full["halfwidth_m"], value["halfwidth_m"])

    def test_truth_and_future_fields_are_not_solver_inputs(self) -> None:
        contaminated = scene()
        contaminated["truth_clearance_m"] = 99.0
        with self.assertRaises(MechanicsError):
            apply_factorial_arm(contaminated, numeric(), "SCALE", "VALUE_ONLY_COMMON_SUPPORT")
        case = ident_case(np.eye(4).tolist(), [0.1, 0.0, 0.0, 0.0])
        case["expected"] = {"leak": True}
        with self.assertRaises(MechanicsError):
            solve_identifiability(case, rule())

    def test_uncertainty_increase_cannot_create_confidence(self) -> None:
        base = apply_factorial_arm(scene(), numeric(), "NONE", "VALUE_ONLY_COMMON_SUPPORT")["record"]["output"]
        wider_numeric = numeric()
        wider_numeric["sigma_measurement_m"] *= 3.0
        wider_numeric["sigma_factor_baseline_m"] *= 3.0
        wider = apply_factorial_arm(scene(), wider_numeric, "NONE", "VALUE_ONLY_COMMON_SUPPORT")["record"]["output"]
        self.assertGreater(wider["halfwidth_m"], base["halfwidth_m"])
        self.assertEqual(wider["query_state"], "UNKNOWN")

    def test_action_filter_rejects_body_motion(self) -> None:
        self.assertEqual(evaluate_action_filter({"id": "impl_unit_camera", "requires_body_motion": False}), {"allowed": True, "reason": "NONE"})
        self.assertEqual(evaluate_action_filter({"id": "impl_unit_step", "requires_body_motion": True}), {"allowed": False, "reason": "BODY_MOTION_FORBIDDEN"})

    def test_mechanics_source_does_not_import_static_oracles(self) -> None:
        source = (Path(__file__).with_name("o0m_mechanics.py")).read_text(encoding="utf-8")
        self.assertNotIn("validate_taro_p0_protocol", source)
        self.assertNotIn("validate_taro_o0m_protocol", source)
        self.assertNotIn("expected_records", source)


if __name__ == "__main__":
    unittest.main()
