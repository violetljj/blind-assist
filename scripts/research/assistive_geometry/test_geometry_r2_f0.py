#!/usr/bin/env python3
"""Focused tests for the Assistive Geometry R2 F0 reducer and fixtures."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.geometry_r2_reducer import (
    ReducerError,
    band_for_lateral,
    canonical_sha256,
    load_profile,
    reduce_frame,
    state_map,
)
from scripts.research.assistive_geometry.run_geometry_r2_f0_canary import deep_merge


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "geometry_r2_f0_cases.json"


def suite() -> dict:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def case_frame(case_id: str) -> tuple[dict, dict, dict]:
    value = suite()
    case = next(item for item in value["cases"] if item["id"] == case_id)
    frame = deep_merge(value["base_factor_frame"], case.get("patch", {}))
    frame["frame_id"] = case_id
    return frame, value["reducer_profile"], case


class GeometryR2F0Tests(unittest.TestCase):
    def test_perfect_empty_corridor_is_clear(self) -> None:
        frame, profile, _ = case_frame("perfect_metric_depth_flat_empty")
        self.assertEqual(
            state_map(reduce_frame(frame, profile)),
            {"left": ("CLEAR_OBSERVED",) * 3, "center": ("CLEAR_OBSERVED",) * 3, "right": ("CLEAR_OBSERVED",) * 3},
        )

    def test_positive_obstacle_requires_positive_evidence(self) -> None:
        frame, profile, _ = case_frame("non_ground_center_obstacle_depth_discontinuity_landscape")
        states = state_map(reduce_frame(frame, profile))
        self.assertEqual(states["center"], ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "OCCUPIED_OBSERVED"))
        false_frame, _, _ = case_frame("false_texture_boundary_ignored")
        self.assertNotIn("OCCUPIED_OBSERVED", state_map(reduce_frame(false_frame, profile))["center"])

    def test_local_missing_depth_is_unknown_not_blocked(self) -> None:
        frame, profile, _ = case_frame("local_center_depth_missing")
        states = state_map(reduce_frame(frame, profile))
        self.assertEqual(states["center"], ("UNKNOWN", "UNKNOWN", "UNKNOWN"))
        self.assertEqual(states["left"], ("CLEAR_OBSERVED",) * 3)
        self.assertEqual(states["right"], ("CLEAR_OBSERVED",) * 3)

    def test_uncertainty_degrades_definite_states_only_to_unknown(self) -> None:
        low, profile, _ = case_frame("non_ground_center_obstacle_depth_discontinuity_landscape")
        medium, _, _ = case_frame("depth_noise_center_obstacle_medium")
        high, _, _ = case_frame("depth_noise_center_obstacle_high")
        maps = [state_map(reduce_frame(frame, profile)) for frame in (low, medium, high)]
        for before, after in zip(maps, maps[1:]):
            for band in ("left", "center", "right"):
                for prior_state, next_state in zip(before[band], after[band]):
                    self.assertIn(next_state, {prior_state, "UNKNOWN"})

    def test_scale_bias_without_proof_cannot_invent_occupancy(self) -> None:
        frame, profile, _ = case_frame("scale_bias_insufficient_to_prove_blocked")
        self.assertEqual(
            state_map(reduce_frame(frame, profile))["center"],
            ("CLEAR_OBSERVED", "UNKNOWN", "UNKNOWN"),
        )

    def test_final_task_shortcut_is_rejected(self) -> None:
        frame, profile, case = case_frame("forbidden_learned_final_task_shortcut")
        with self.assertRaises(ReducerError) as caught:
            reduce_frame(frame, profile)
        self.assertEqual(caught.exception.code, case["expected_error_code"])

    def test_band_boundary_ownership_is_unique(self) -> None:
        profile = load_profile(suite()["reducer_profile"])
        self.assertEqual(band_for_lateral(profile, -0.25), "center")
        self.assertEqual(band_for_lateral(profile, 0.25), "right")
        self.assertEqual(band_for_lateral(profile, 0.75), "right")
        self.assertIsNone(band_for_lateral(profile, 0.751))

    def test_replay_is_byte_canonical_deterministic(self) -> None:
        frame, profile, _ = case_frame("side_obstacles_center_corridor_open")
        first = reduce_frame(frame, profile)
        second = reduce_frame(copy.deepcopy(frame), copy.deepcopy(profile))
        self.assertEqual(canonical_sha256(first), canonical_sha256(second))

    def test_all_frozen_fixture_oracles_match_exactly(self) -> None:
        value = suite()
        for case in value["cases"]:
            with self.subTest(case=case["id"]):
                frame = deep_merge(value["base_factor_frame"], case.get("patch", {}))
                frame["frame_id"] = case["id"]
                if "expected_error_code" in case:
                    with self.assertRaises(ReducerError) as caught:
                        reduce_frame(frame, value["reducer_profile"])
                    self.assertEqual(caught.exception.code, case["expected_error_code"])
                else:
                    actual = {key: list(states) for key, states in state_map(reduce_frame(frame, value["reducer_profile"])).items()}
                    self.assertEqual(actual, case["expected_states"])


if __name__ == "__main__":
    unittest.main()
