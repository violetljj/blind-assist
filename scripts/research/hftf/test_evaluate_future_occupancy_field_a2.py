import unittest

import numpy as np

from evaluate_future_occupancy_field_a2 import motion_modes, sigmoid_margin


class FutureOccupancyFieldA2Test(unittest.TestCase):
    def test_sigmoid_margin_is_half_at_boundary(self) -> None:
        self.assertAlmostEqual(sigmoid_margin(1.5, 1.5, 0.2), 0.5)

    def test_cv_extrapolates_linear_history(self) -> None:
        times = np.asarray([-0.4, -0.3, -0.2, -0.1, 0.0])
        values = 2.0 - times
        result = motion_modes(times, values, horizon=1.5)
        self.assertAlmostEqual(result["cv"], 1.5, places=8)
        self.assertAlmostEqual(result["slope"], -1.0, places=8)
        self.assertAlmostEqual(result["cv_rmse"], 0.0, places=8)

    def test_ca_extrapolates_quadratic_history(self) -> None:
        times = np.asarray([-0.4, -0.3, -0.2, -0.1, 0.0])
        values = 2.0 + times * times
        result = motion_modes(times, values, horizon=1.5)
        self.assertAlmostEqual(result["ca"], 2.25, places=8)
        self.assertAlmostEqual(result["acceleration"], 2.0, places=8)
        self.assertAlmostEqual(result["ca_rmse"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
