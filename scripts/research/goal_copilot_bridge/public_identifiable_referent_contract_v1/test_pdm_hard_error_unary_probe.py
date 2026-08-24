from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    pdm_hard_error_unary_probe as sut,
)


class PdmHardErrorUnaryProbeTest(unittest.TestCase):
    def test_visible_bbox_enforces_visibility_and_area(self) -> None:
        good = {"bbox_visib": [10, 20, 200, 100], "visib_fract": 0.8}
        self.assertIsNotNone(sut._visible_bbox(good))
        self.assertIsNone(sut._visible_bbox({**good, "visib_fract": 0.5}))
        self.assertIsNone(sut._visible_bbox({"bbox_visib": [0, 0, 1, 1], "visib_fract": 1.0}))

    def test_distractor_selection_is_candidate_independent_and_deterministic(self) -> None:
        candidates = [
            {"native_object_id": 1, "visible_area_fraction": 0.2, "visible_fraction": 1.0, "gt_index": 0},
            {"native_object_id": 2, "visible_area_fraction": 0.1, "visible_fraction": 1.0, "gt_index": 1},
            {"native_object_id": 3, "visible_area_fraction": 0.3, "visible_fraction": 0.8, "gt_index": 2},
        ]
        self.assertEqual(3, sut._choose_distractor(candidates, 1)["native_object_id"])
        self.assertEqual(3, sut._choose_distractor(list(reversed(candidates)), 1)["native_object_id"])

    def test_spread_indices_are_fixed_and_cover_endpoints(self) -> None:
        self.assertEqual([0, 4, 8], sut._spread_indices(9, 3))
        self.assertEqual([0, 1, 2], sut._spread_indices(3, 3))

    def test_target_slot_is_predeclared_and_balanced(self) -> None:
        self.assertEqual(["A", "B", "A", "B"], [sut._target_slot(i) for i in range(4)])

    def test_public_firewall_rejects_truth_and_baseline_stratum(self) -> None:
        sut._assert_public_blind(
            {"protocol_id": "FRESH_SOURCE_DISJOINT_HARD_ERROR_UNARY_VERIFIER_V0", "pair_id": "x"}
        )
        with self.assertRaises(sut.PdmHardErrorProbeError):
            sut._assert_public_blind({"target_slot": "A"})
        with self.assertRaises(sut.PdmHardErrorProbeError):
            sut._assert_public_blind({"stratum": "HARD"})

    def test_challenger_selection_takes_all_errors_and_fixed_low_margin_tail(self) -> None:
        rows = []
        for index in range(20):
            rows.append(
                {
                    "pair_id": f"p{index:02d}",
                    "evaluation": "DISTRACTOR_OUTRANKS" if index in (0, 1) else "TARGET_OUTRANKS",
                    "target_margin": -0.1 if index in (0, 1) else index / 100.0,
                }
            )
        hard, controls = sut._select_challenger_rows(rows)
        self.assertEqual({"p00", "p01", "p02", "p03"}, set(hard))
        self.assertEqual(["p04", "p05", "p06", "p07"], controls)

    def test_evaluation_treats_tie_as_non_target(self) -> None:
        self.assertEqual("TARGET_OUTRANKS", sut._evaluation(0.2, 0.1))
        self.assertEqual("DISTRACTOR_OUTRANKS", sut._evaluation(0.1, 0.2))
        self.assertEqual("TIE", sut._evaluation(0.1, 0.1))


if __name__ == "__main__":
    unittest.main()
