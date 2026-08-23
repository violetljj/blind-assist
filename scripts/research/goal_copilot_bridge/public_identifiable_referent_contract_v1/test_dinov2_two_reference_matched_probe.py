from __future__ import annotations

import math
import unittest

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    dinov2_two_reference_matched_probe as sut,
)


def _polygon(object_id: int, x0: float, y0: float, x1: float, y1: float) -> dict[str, object]:
    return {
        "object": object_id,
        "x": [x0, x1, x1, x0],
        "y": [y0, y0, y1, y1],
    }


class Dinov2TwoReferenceMatchedProbeTest(unittest.TestCase):
    def test_reference_aggregation_is_order_free_and_reports_ties(self) -> None:
        score, winners = sut._aggregate_reference_scores({"R1": 0.2, "R2": 0.7})
        swapped_score, swapped_winners = sut._aggregate_reference_scores({"R2": 0.7, "R1": 0.2})
        self.assertEqual((0.7, ["R2"]), (score, winners))
        self.assertEqual((score, winners), (swapped_score, swapped_winners))
        self.assertEqual((0.5, ["R1", "R2"]), sut._aggregate_reference_scores({"R1": 0.5, "R2": 0.5}))

    def test_reference_aggregation_fails_closed_on_non_finite_or_missing_score(self) -> None:
        with self.assertRaises(sut.TwoReferenceProbeError):
            sut._aggregate_reference_scores({"R1": 0.1, "R2": math.nan})
        with self.assertRaises(sut.TwoReferenceProbeError):
            sut._aggregate_reference_scores({"R1": 0.1})

    def test_parent_burn_set_includes_target_frames_and_all_listed_distractors(self) -> None:
        filenames, object_ids = sut._used_parent_units(
            {
                "native_object_id": 7,
                "reference": {"source_filename": "reference.jpg"},
                "later_observations": [
                    {"source_filename": "later-1.jpg", "same_class_distractor_object_ids": [2, 3]},
                    {"source_filename": "later-2.jpg", "same_class_distractor_object_ids": [3, 4]},
                ],
            }
        )
        self.assertEqual({"reference.jpg", "later-1.jpg", "later-2.jpg"}, filenames)
        self.assertEqual({2, 3, 4, 7}, object_ids)

    def test_distractor_selection_excludes_burned_ids_and_orders_by_area(self) -> None:
        annotation = {
            "objects": [
                {"name": "chair"},
                {"name": "chair"},
                {"name": "chair"},
                {"name": "table"},
            ]
        }
        frame = {
            "polygon": [
                _polygon(0, 0, 0, 200, 200),
                _polygon(1, 0, 0, 150, 150),
                _polygon(2, 0, 0, 250, 250),
                _polygon(3, 0, 0, 300, 300),
            ]
        }
        found = sut._eligible_distractors(frame, annotation, 0, "chair", {2})
        self.assertEqual([1], [item["native_object_id"] for item in found])

    def test_slot_assignment_is_predeclared_and_balanced(self) -> None:
        self.assertEqual(["A", "B", "A", "B"], [sut._target_slot(index) for index in range(4)])

    def test_private_mapping_is_rejected_from_score_config(self) -> None:
        sut._assert_score_config_blind({"references": ["R1", "R2"], "candidate_slots": ["A", "B"]})
        with self.assertRaises(sut.TwoReferenceProbeError):
            sut._assert_score_config_blind({"target_slot": "A"})

    def test_transition_treats_tie_as_non_target(self) -> None:
        self.assertEqual(
            "SINGLE_NON_TARGET_TO_TWO_TARGET",
            sut._transition("TIE", "TARGET_OUTRANKS"),
        )
        self.assertEqual(
            "SINGLE_TARGET_TO_TWO_NON_TARGET",
            sut._transition("TARGET_OUTRANKS", "DISTRACTOR_OUTRANKS"),
        )


if __name__ == "__main__":
    unittest.main()
