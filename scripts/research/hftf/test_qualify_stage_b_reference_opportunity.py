from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qualify_stage_b_reference_opportunity as qualifier


class StageBReferenceOpportunityQualificationTest(unittest.TestCase):
    def protocol(self) -> dict:
        return {
            "obstacle_opportunity_qualification": {
                "minimum_known_coverage_each_height": 0.1,
                "minimum_positive_known_cells_each_height": 5,
                "minimum_negative_known_cells_each_height": 20,
            },
            "ground_opportunity_qualification": {
                "minimum_ground_known_coverage": 0.1,
                "minimum_reference_risk_cells": 5,
                "minimum_distinct_frames_with_reference_risk": 3,
                "minimum_distinct_directions_with_reference_risk": 2,
            },
        }

    def test_all_reference_opportunity_gates_are_required(self) -> None:
        checks = qualifier._reference_decision(
            {"foot": 0.2, "body": 0.2, "head": 0.2},
            {"foot": 5, "body": 6, "head": 7},
            {"foot": 20, "body": 21, "head": 22},
            {
                "1": {"positive": 1, "negative": 1},
                "2": {"positive": 1, "negative": 1},
            },
            0.2,
            5,
            3,
            2,
            self.protocol(),
        )
        self.assertTrue(all(checks.values()), checks)

    def test_ground_persistence_cannot_be_replaced_by_one_frame(
        self,
    ) -> None:
        checks = qualifier._reference_decision(
            {"foot": 0.2, "body": 0.2, "head": 0.2},
            {"foot": 5, "body": 5, "head": 5},
            {"foot": 20, "body": 20, "head": 20},
            {"2": {"positive": 5, "negative": 20}},
            0.2,
            5,
            1,
            2,
            self.protocol(),
        )
        self.assertFalse(
            checks["ground_reference_risk_frames"]
        )

    def test_qualifier_has_no_angular_baseline_helper(self) -> None:
        source = inspect.getsource(qualifier)
        banned_helper = "_" + "bin_obstacle_support"
        self.assertNotIn(banned_helper, source)


if __name__ == "__main__":
    unittest.main()
