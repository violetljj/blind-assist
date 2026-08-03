import unittest

import numpy as np

from evaluate_collision_risk_field_a1 import (
    binary_metrics,
    image_third_band,
    two_d_clearances,
)


class CollisionRiskFieldA1Test(unittest.TestCase):
    def test_image_thirds(self) -> None:
        self.assertEqual(image_third_band(10, 90), "left")
        self.assertEqual(image_third_band(45, 90), "center")
        self.assertEqual(image_third_band(80, 90), "right")

    def test_binary_metrics_perfect(self) -> None:
        result = binary_metrics(
            np.asarray([0.0, 1.0, 0.0, 1.0]),
            np.asarray([0.1, 0.9, 0.2, 0.8]),
            0.5,
        )
        self.assertEqual(result["mcc"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["false_positive_rate"], 0.0)

    def test_two_d_clearance_uses_metric_lateral_bands(self) -> None:
        depth = np.full((20, 30), 2.0, dtype=np.float32)
        result = two_d_clearances(depth, fx=20.0, cx=15.0)
        self.assertAlmostEqual(result["center"], 2.0)
        self.assertAlmostEqual(result["left"], 2.0)
        self.assertAlmostEqual(result["right"], 2.0)


if __name__ == "__main__":
    unittest.main()
