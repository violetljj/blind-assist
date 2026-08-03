#!/usr/bin/env python3

import unittest

from evaluate_metric3d_probabilistic_occupancy_a0 import (
    empirical_occupied_probability,
    expected_calibration_error,
)


class Metric3dProbabilisticOccupancyA0Test(unittest.TestCase):
    def test_empirical_probability_is_monotonic_in_horizon(self) -> None:
        residuals = [-0.5, 0.0, 0.5]
        near = empirical_occupied_probability(2.0, 1.0, residuals)
        far = empirical_occupied_probability(2.0, 2.0, residuals)
        self.assertLess(near, far)

    def test_empirical_probability_uses_half_count_smoothing(self) -> None:
        self.assertEqual(
            empirical_occupied_probability(10.0, 1.0, [0.0, 1.0, 2.0]),
            0.125,
        )

    def test_ece_is_zero_for_matching_bins(self) -> None:
        self.assertAlmostEqual(
            expected_calibration_error([0.0, 1.0], [False, True]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
