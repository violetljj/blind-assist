from __future__ import annotations

import unittest

import numpy as np

from evaluate_phone_development import (
    center_median,
    rotate_clockwise_with_intrinsics,
    scaled_raw_intrinsics,
)


class EvaluatePhoneDevelopmentTest(unittest.TestCase):
    def test_scales_and_rotates_intrinsics_with_image(self) -> None:
        receipt = {
            "intrinsic_calibration": [400.0, 300.0, 200.0, 150.0, 0.0],
            "active_array": "0 0 400 300",
        }
        raw = scaled_raw_intrinsics(receipt, 400, 300)
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        rotated, matrix = rotate_clockwise_with_intrinsics(image, raw)
        self.assertEqual((400, 300, 3), rotated.shape)
        np.testing.assert_allclose(
            matrix,
            np.asarray([[300.0, 0.0, 149.0], [0.0, 400.0, 200.0], [0.0, 0.0, 1.0]]),
        )

    def test_center_median_ignores_invalid_values(self) -> None:
        depth = np.ones((100, 100), dtype=np.float64)
        depth[45:55, 45:55] = 2.0
        depth[49, 49] = np.nan
        self.assertEqual(2.0, center_median(depth, fraction=0.1))


if __name__ == "__main__":
    unittest.main()
