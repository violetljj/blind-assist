from __future__ import annotations

import unittest

from aggregate_r4_split_source_result import _terminal


class R4SplitSourceResultTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ordered = [
            "R4_OBSTACLE_OPPORTUNITY_COHORT_NOT_EVALUABLE",
            "R4_OBSTACLE_ENVELOPE_GAIN_NOT_SUPPORTED_STOP",
            "R4_ANALYTIC_TERRAIN_MECHANICS_NOT_SUPPORTED_STOP",
            "R4_STAGE_B_SPLIT_SOURCE_TEACHER_MECHANICS_SUPPORTED",
        ]

    def test_both_components_supported_reaches_joint_success(self) -> None:
        self.assertEqual(
            self.ordered[3],
            _terminal(
                "R4_OBSTACLE_ENVELOPE_GAIN_SUPPORTED",
                "R4_ANALYTIC_TERRAIN_MECHANICS_SUPPORTED",
                self.ordered,
            ),
        )

    def test_obstacle_failure_precedes_terrain(self) -> None:
        self.assertEqual(
            self.ordered[1],
            _terminal(
                "R4_OBSTACLE_ENVELOPE_GAIN_NOT_SUPPORTED_STOP",
                "R4_ANALYTIC_TERRAIN_MECHANICS_SUPPORTED",
                self.ordered,
            ),
        )

    def test_terrain_failure_follows_obstacle_success(self) -> None:
        self.assertEqual(
            self.ordered[2],
            _terminal(
                "R4_OBSTACLE_ENVELOPE_GAIN_SUPPORTED",
                "R4_ANALYTIC_TERRAIN_MECHANICS_NOT_SUPPORTED_STOP",
                self.ordered,
            ),
        )


if __name__ == "__main__":
    unittest.main()
