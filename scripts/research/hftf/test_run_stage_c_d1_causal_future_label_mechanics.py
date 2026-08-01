from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d1_causal_future_label_mechanics import (
    _causal_origin,
    _formal_anchors,
    _odometry_mapping,
    _structural_canaries,
)


REBIN = {
    "target_direction_degrees": [-30, -15, 0, 15, 30],
    "target_distance_m": [1.4, 1.8, 2.2, 2.6, 3.0],
    "maximum_direction_error_degrees": 7.5,
    "maximum_distance_error_m": 0.2,
}

D0_PROFILE = {
    "minimum_known_sections_per_direction": 4,
    "rise_risk_if_adjacent_delta_m_strictly_greater_than": 0.18,
    "drop_risk_if_adjacent_delta_m_strictly_less_than": -0.15,
}


class StageCD1CausalFutureLabelMechanicsTest(unittest.TestCase):
    def test_formal_anchors_preserve_history_and_future(self) -> None:
        self.assertEqual([5], _formal_anchors(10))
        self.assertEqual([5, 10], _formal_anchors(15))

    def test_causal_origin_uses_only_history(self) -> None:
        origin, speed = _causal_origin(
            np.array([1.0, 0.0]),
            np.array([0.6, 0.0]),
            0.4,
            0.8,
        )
        np.testing.assert_allclose(origin, [1.8, 0.0])
        self.assertAlmostEqual(1.0, speed)

    def test_odometry_mapping_recovers_forward_motion(self) -> None:
        x = np.arange(0.0, 4.0, 0.2)
        y = np.zeros_like(x)
        yaw = np.zeros_like(x)
        result = _odometry_mapping(x, y, yaw, 2, 0.05)
        self.assertEqual(18, result["moving_interval_count"])
        self.assertAlmostEqual(
            1.0, result["motion_yaw_circular_resultant"]
        )
        self.assertAlmostEqual(
            0.0, result["median_abs_motion_yaw_error_degrees"]
        )

    def test_all_structural_canaries_pass(self) -> None:
        result = _structural_canaries(REBIN, D0_PROFILE)
        self.assertEqual(7, len(result))
        self.assertTrue(all(result.values()), result)


if __name__ == "__main__":
    unittest.main()
