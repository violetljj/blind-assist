#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_motion_conditioned_occupancy_a0 import FEATURE_NAMES


class FrozenMotionConditionedOccupancyA0Test(unittest.TestCase):
    def test_frozen_feature_count_is_eighteen(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 18)
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))

    def test_standardized_zero_uses_intercept(self) -> None:
        mean = np.asarray([1.0, 2.0])
        scale = np.asarray([2.0, 4.0])
        weights = np.asarray([0.25, 3.0, -2.0])
        design = np.r_[1.0, (mean - mean) / scale]
        self.assertAlmostEqual(float(design @ weights), 0.25)


if __name__ == "__main__":
    unittest.main()
