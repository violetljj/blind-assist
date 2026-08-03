#!/usr/bin/env python3

import unittest

import numpy as np
from common import affine_depth, fit_dense_affine

CONFIG = {
    "sample_stride_px": 1,
    "minimum_pairs": 20,
    "valid_depth_range_m": [0.25, 6.0],
    "minimum_inlier_fraction": 0.5,
    "slope_bounds": [0.25, 4.0],
    "maximum_inlier_median_absolute_residual_m": 0.25,
}


class DenseAffineTest(unittest.TestCase):
    def test_recovers_affine_with_sparse_outliers(self):
        fast = np.linspace(0.5, 4.0, 100, dtype=np.float32).reshape(10, 10)
        metric = 1.4 * fast + 0.2
        metric.flat[::10] = 5.9
        fit = fit_dense_affine(fast, metric, CONFIG)
        self.assertEqual(fit["status"], "VALID")
        self.assertAlmostEqual(fit["slope"], 1.4, delta=0.02)
        self.assertAlmostEqual(fit["intercept_m"], 0.2, delta=0.03)

    def test_rejects_too_few_valid_pairs(self):
        fast = np.full((10, 10), np.nan, dtype=np.float32)
        metric = np.ones((10, 10), dtype=np.float32)
        fit = fit_dense_affine(fast, metric, CONFIG)
        self.assertEqual(fit["status"], "UNKNOWN_AFFINE_PAIRS")

    def test_affine_depth_requires_valid_fit(self):
        with self.assertRaisesRegex(ValueError, "invalid affine"):
            affine_depth(np.ones((2, 2), dtype=np.float32), {"status": "UNKNOWN"})


if __name__ == "__main__":
    unittest.main()
