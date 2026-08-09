#!/usr/bin/env python3
"""Focused tests for the zero-parameter F1 FactorTensor adapter."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

from scripts.research.assistive_geometry.factor_tensor_adapter import (
    AdapterError,
    adapt_factor_tensor,
    canonical_json_bytes,
    canonical_sha256,
)
from scripts.research.assistive_geometry.geometry_r2_reducer import reduce_frame, state_map


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_FIXTURE_2026-08-10.json"
F0_FIXTURE = Path(__file__).resolve().parent / "fixtures/geometry_r2_f0_cases.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        merged = copy.deepcopy(base)
        for key, value in patch.items():
            merged[key] = deep_merge(merged[key], value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(patch)


def suite() -> dict[str, Any]:
    return load_json(FIXTURE)


def reducer_profile() -> dict[str, Any]:
    return load_json(F0_FIXTURE)["reducer_profile"]


def case_input(case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value = suite()
    case = next(item for item in value["cases"] if item["id"] == case_id)
    return deep_merge(value["base_input"], case.get("patch", {})), case


def states_for(case_id: str) -> dict[str, tuple[str, ...]]:
    adapter_input, _ = case_input(case_id)
    return state_map(reduce_frame(adapt_factor_tensor(adapter_input), reducer_profile()))


class FactorTensorAdapterTests(unittest.TestCase):
    def test_nominal_maps_to_complete_valid_frame(self) -> None:
        adapter_input, _ = case_input("nominal_landscape_single_component")
        frame = adapt_factor_tensor(adapter_input)
        self.assertTrue(frame["input_geometry"]["k_valid"])
        self.assertTrue(frame["depth_scale"]["valid"])
        self.assertTrue(frame["support"]["valid"])
        self.assertTrue(frame["boundary"]["valid"])
        self.assertEqual(len(frame["boundary"]["obstacles"]), 1)
        self.assertTrue(frame["boundary"]["obstacles"][0]["depth_valid"])

    def test_receipt_identity_mismatch_is_complete_invalid_frame(self) -> None:
        adapter_input, _ = case_input("geometry_receipt_identity_mismatch")
        frame = adapt_factor_tensor(adapter_input)
        self.assertFalse(frame["input_geometry"]["k_valid"])
        self.assertFalse(frame["input_geometry"]["transform_valid"])
        self.assertTrue(all(state == "UNKNOWN" for states in state_map(reduce_frame(frame, reducer_profile())).values() for state in states))

    def test_final_task_shortcut_is_rejected_recursively(self) -> None:
        adapter_input, _ = case_input("nominal_landscape_single_component")
        adapter_input["prediction"]["factor_identity"]["nested"] = {"final_state": "CLEAR_OBSERVED"}
        with self.assertRaises(AdapterError) as caught:
            adapt_factor_tensor(adapter_input)
        self.assertEqual(caught.exception.code, "FORBIDDEN_FINAL_TASK_FIELD")

    def test_local_missing_depth_stays_local_and_never_occupied(self) -> None:
        adapter_input, _ = case_input("local_component_depth_missing")
        frame = adapt_factor_tensor(adapter_input)
        self.assertFalse(frame["boundary"]["obstacles"][0]["depth_valid"])
        states = state_map(reduce_frame(frame, reducer_profile()))
        self.assertNotIn("OCCUPIED_OBSERVED", states["center"])

    def test_support_invalid_reduces_to_all_unknown(self) -> None:
        adapter_input, _ = case_input("support_invalid_fail_closed")
        frame = adapt_factor_tensor(adapter_input)
        self.assertFalse(frame["support"]["valid"])
        self.assertTrue(all(state == "UNKNOWN" for states in state_map(reduce_frame(frame, reducer_profile())).values() for state in states))

    def test_components_split_order_and_bridge_merge(self) -> None:
        split_input, _ = case_input("two_components_canonical_left_then_right")
        split = adapt_factor_tensor(split_input)["boundary"]["obstacles"]
        self.assertEqual(len(split), 2)
        self.assertLess(split[0]["lateral_center_m"], split[1]["lateral_center_m"])
        merged_input, _ = case_input("bridge_pixel_merges_components")
        self.assertEqual(len(adapt_factor_tensor(merged_input)["boundary"]["obstacles"]), 1)

    def test_increased_depth_uncertainty_does_not_strengthen_states(self) -> None:
        nominal_input, _ = case_input("nominal_landscape_single_component")
        high_input, _ = case_input("high_depth_uncertainty_monotone")
        nominal_frame = adapt_factor_tensor(nominal_input)
        high_frame = adapt_factor_tensor(high_input)
        self.assertGreaterEqual(high_frame["depth_scale"]["scale_sigma_m"], nominal_frame["depth_scale"]["scale_sigma_m"])
        nominal_states = state_map(reduce_frame(nominal_frame, reducer_profile()))
        high_states = state_map(reduce_frame(high_frame, reducer_profile()))
        for band in ("left", "center", "right"):
            for before, after in zip(nominal_states[band], high_states[band]):
                self.assertIn(after, {before, "UNKNOWN"})

    def test_display_upright_orientation_fixture_has_task_state_parity(self) -> None:
        self.assertEqual(
            states_for("nominal_landscape_single_component"),
            states_for("portrait_equivalent_single_component"),
        )

    def test_every_frozen_case_matches_adapter_expectations(self) -> None:
        value = suite()
        profile = reducer_profile()
        for case in value["cases"]:
            with self.subTest(case=case["id"]):
                adapter_input = deep_merge(value["base_input"], case.get("patch", {}))
                frame = adapt_factor_tensor(adapter_input)
                expected = case["expected"]
                valid = frame["input_geometry"]["k_valid"] and frame["input_geometry"]["transform_valid"]
                actual_terminal = "ADAPTER_FRAME_VALID" if valid else "ADAPTER_INPUT_INVALID"
                self.assertEqual(actual_terminal, expected["terminal"])
                if "obstacle_count" in expected:
                    self.assertEqual(len(frame["boundary"]["obstacles"]), expected["obstacle_count"])
                if "obstacle_depth_valid" in expected:
                    self.assertEqual([item["depth_valid"] for item in frame["boundary"]["obstacles"]], expected["obstacle_depth_valid"])
                if "support_valid" in expected:
                    self.assertEqual(frame["support"]["valid"], expected["support_valid"])
                states = state_map(reduce_frame(frame, profile))
                if expected.get("all_unknown") is True:
                    self.assertTrue(all(state == "UNKNOWN" for band in states.values() for state in band))
                if "forbidden_state" in expected:
                    self.assertTrue(all(expected["forbidden_state"] not in band for band in states.values()))

    def test_replay_is_byte_canonical(self) -> None:
        adapter_input, _ = case_input("nominal_landscape_single_component")
        first = adapt_factor_tensor(adapter_input)
        second = adapt_factor_tensor(copy.deepcopy(adapter_input))
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(canonical_sha256(first), canonical_sha256(second))


if __name__ == "__main__":
    unittest.main()
