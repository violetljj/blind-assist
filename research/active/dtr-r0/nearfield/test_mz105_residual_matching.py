"""Numerical and geometry checks for the diagnostic, without task labels."""
import unittest
import numpy as np
import torch
from mz105_residual_matching import zncc_volume, point_features, FOCAL_BASELINE


class MatchingTests(unittest.TestCase):
    def test_disparity_direction_and_independent_pearson(self):
        rng = np.random.default_rng(105)
        left = rng.integers(0, 256, (24, 128), dtype=np.uint8)
        right = rng.integers(0, 256, left.shape, dtype=np.uint8)
        right[:, :-20] = left[:, 20:]
        volume = zncc_volume(left, right, 'cpu')
        self.assertAlmostEqual(float(volume[20, 10, 70]), 1., places=10)
        for disparity in (0, 5, 20, 60):
            a = left[8:13, 68:73].astype(float).ravel()
            b = right[8:13, 68-disparity:73-disparity].astype(float).ravel()
            expected = np.corrcoef(a, b)[0, 1]
            self.assertAlmostEqual(float(volume[disparity, 10, 70]), expected, places=10)
        depth = np.full(left.shape, np.nan)
        depth[10, 70] = FOCAL_BASELINE/20
        features = point_features(volume, np.array([10]), np.array([70]), depth)[0]
        self.assertAlmostEqual(features[0], 1., places=10)
        self.assertGreater(features[1], 0.)
        self.assertGreater(features[2], 0.)

    def test_flat_and_incomplete_patches_unavailable(self):
        constant = np.full((12, 128), 240, dtype=np.uint8)
        flat = zncc_volume(constant, constant, 'cpu')
        self.assertFalse(bool(torch.isfinite(flat).any()))
        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 256, (12, 128), dtype=np.uint8)
        volume = zncc_volume(rgb, rgb, 'cpu')
        self.assertFalse(bool(torch.isfinite(volume[:, :2]).any()))
        self.assertFalse(bool(torch.isfinite(volume[20, :, :22]).any()))
        self.assertAlmostEqual(float(volume[0, 6, 60]), 1., places=10)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
