from __future__ import annotations

import unittest

import numpy as np

from scripts.research.hftf.deployment.depthart.analyze_depthart_d3r5_parent_relative_veto import (
    average_rank_scaled,
    select_zero_false_block_threshold,
)


class ParentRelativeVetoTest(unittest.TestCase):
    def test_average_rank_is_affine_invariant_and_tie_stable(self) -> None:
        original = average_rank_scaled(np.asarray([3.0, 1.0, 1.0, 7.0]))
        affine = average_rank_scaled(np.asarray([16.0, 12.0, 12.0, 24.0]))
        np.testing.assert_allclose(original, affine)
        self.assertEqual(original[1], original[2])
        self.assertLess(original[0], original[3])

    def test_threshold_requires_zero_direct_false_positive(self) -> None:
        rows = [
            {
                "threshold": 0.9,
                "actions": {"false_positive_actions": 1},
                "false_clear_all_known_improvement": 0.2,
                "known_coverage_decrease": 0.0,
            },
            {
                "threshold": 0.99,
                "actions": {"false_positive_actions": 0},
                "false_clear_all_known_improvement": 0.03,
                "known_coverage_decrease": 0.0,
            },
            {
                "threshold": 0.999,
                "actions": {"false_positive_actions": 0},
                "false_clear_all_known_improvement": 0.02,
                "known_coverage_decrease": 0.0,
            },
        ]
        self.assertEqual(select_zero_false_block_threshold(rows)["threshold"], 0.99)


if __name__ == "__main__":
    unittest.main()
