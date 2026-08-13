from __future__ import annotations

import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import balanced_pose_source_frontdoor as frontdoor


class BalancedPoseSourceFrontdoorTest(unittest.TestCase):
    def test_tartanground_axis_mapping_matches_standard_pinhole(self) -> None:
        transform = frontdoor.tartanground_pose_to_standard_camera(
            np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )
        np.testing.assert_allclose(transform[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(transform[:3, :3] @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(transform[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(transform[:3, :3] @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])

    def test_gate_pass_requires_all_three_denominators(self) -> None:
        parents = {
            f"p{index}": {
                "static_unknown_occupied_opportunity": 1,
                "truth_clear": 1,
            }
            for index in range(4)
        }
        checks, terminal = frontdoor.decide_frontdoor(48, parents)
        self.assertTrue(all(checks.values()))
        self.assertEqual(
            terminal,
            "TARO_TASK_OBSERVABILITY_BALANCED_POSE_SOURCE_FRONTDOOR_PASS",
        )

    def test_unknown_never_satisfies_clear_denominator(self) -> None:
        parents = {
            f"p{index}": {
                "static_unknown_occupied_opportunity": 1,
                "truth_clear": 0,
                "truth_unknown": 100,
            }
            for index in range(4)
        }
        checks, terminal = frontdoor.decide_frontdoor(48, parents)
        self.assertFalse(checks["minimum_clear_denominator_parents"])
        self.assertEqual(terminal, "NOT_EVALUABLE_DATA_OBSERVABILITY_DENOMINATOR")


if __name__ == "__main__":
    unittest.main()
