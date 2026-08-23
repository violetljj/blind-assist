from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import oracle_competing_identity_probe as sut


class OracleCompetingIdentityProbeTest(unittest.TestCase):
    def test_provider_prompt_precedes_variadic_image_flags(self) -> None:
        command = sut._provider_command(Path("codex.exe"), Path("pair"))
        self.assertLess(command.index(sut.PROMPT), command.index("--image"))

    def test_square_crop_bounds_are_square_and_clipped(self) -> None:
        crop = sut._square_crop_bounds([0.0, 0.1, 0.2, 0.5])
        self.assertAlmostEqual(crop[2] - crop[0], crop[3] - crop[1])
        self.assertGreaterEqual(min(crop), 0.0)
        self.assertLessEqual(max(crop), 1.0)

    def test_competitor_reuses_historical_wrong_instance(self) -> None:
        distractors = [
            {"native_object_id": 2, "bbox_area_fraction": 0.2},
            {"native_object_id": 3, "bbox_area_fraction": 0.5},
        ]
        row = {"identity_outcome": "SAME_CLASS_DISTRACTOR", "assigned_instance": {"native_object_id": 2}}
        self.assertEqual(2, sut._select_competitor(row, distractors)["native_object_id"])

    def test_control_competitor_uses_largest_area_then_object_id(self) -> None:
        distractors = [
            {"native_object_id": 3, "bbox_area_fraction": 0.5},
            {"native_object_id": 2, "bbox_area_fraction": 0.5},
            {"native_object_id": 1, "bbox_area_fraction": 0.2},
        ]
        row = {"identity_outcome": "SAME_INSTANCE"}
        self.assertEqual(2, sut._select_competitor(row, distractors)["native_object_id"])

    def test_target_position_alternates(self) -> None:
        self.assertEqual(["A", "B", "A", "B"], [sut._target_position(index) for index in range(4)])

    def test_counterbalance_swaps_all_slot_bound_private_fields(self) -> None:
        source = {
            "pair_id": "pair-001",
            "target_position": "A",
            "candidate_a_physical_instance_id": "target",
            "candidate_b_physical_instance_id": "distractor",
            "candidate_a_native_object_id": 40,
            "candidate_b_native_object_id": 49,
            "crop_bounds_xyxy_normalized": {"A": [0, 0, 1, 1], "B": [2, 2, 3, 3]},
        }
        swapped = sut._swapped_private_pair(source, "swap-001", "pair-001")
        self.assertEqual("B", swapped["target_position"])
        self.assertEqual("distractor", swapped["candidate_a_physical_instance_id"])
        self.assertEqual("target", swapped["candidate_b_physical_instance_id"])
        self.assertEqual(49, swapped["candidate_a_native_object_id"])
        self.assertEqual(40, swapped["candidate_b_native_object_id"])
        self.assertEqual([2, 2, 3, 3], swapped["crop_bounds_xyxy_normalized"]["A"])
        self.assertEqual([0, 0, 1, 1], swapped["crop_bounds_xyxy_normalized"]["B"])

    def test_summary_separates_rescue_and_collateral(self) -> None:
        rows = [
            {"stratum": "HISTORICAL_WRONG", "evaluation": "TARGET_SELECTED", "target_position": "A", "provider_decision": "A"},
            {"stratum": "HISTORICAL_WRONG", "evaluation": "CONTESTED", "target_position": "B", "provider_decision": "CONTESTED"},
            {"stratum": "BASELINE_CORRECT_CONTROL", "evaluation": "TARGET_SELECTED", "target_position": "A", "provider_decision": "A"},
            {"stratum": "BASELINE_CORRECT_CONTROL", "evaluation": "DISTRACTOR_SELECTED", "target_position": "B", "provider_decision": "A"},
        ]
        summary = sut.summarize(
            rows,
            {
                "baseline_correct_total_before_oracle_filter": 16,
                "baseline_correct_excluded_no_same_class_candidate_count": 3,
            },
        )
        self.assertEqual(1, summary["historical_wrong"]["target_selected_count"])
        self.assertEqual(1, summary["historical_wrong"]["contested_count"])
        self.assertEqual(1, summary["baseline_correct_control"]["distractor_selected_count"])


if __name__ == "__main__":
    unittest.main()
