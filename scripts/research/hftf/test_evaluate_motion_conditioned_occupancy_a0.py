#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_motion_conditioned_occupancy_a0 import (
    fit_logistic,
    motion_summary,
    predict_logistic,
)


class MotionConditionedOccupancyA0Test(unittest.TestCase):
    def test_logistic_separates_simple_margin(self) -> None:
        x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.asarray([1.0, 1.0, 0.0, 0.0])
        probabilities = predict_logistic(x, fit_logistic(x, y))
        self.assertGreater(probabilities[0], 0.5)
        self.assertLess(probabilities[-1], 0.5)

    def test_zero_flow_has_zero_motion(self) -> None:
        summary = motion_summary(np.zeros((2, 32, 56), dtype=np.float32))
        np.testing.assert_allclose(summary[:6], np.zeros(6), atol=1e-12)
        self.assertEqual(summary[6], 1.0)
        np.testing.assert_allclose(summary[7:], np.zeros(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
