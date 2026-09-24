"""Geometry and observability checks for the RGB/CNH fusion candidate."""
import unittest

import numpy as np

from cnh_rgb_frustum import zone_rgb_support


class FrustumSupportTest(unittest.TestCase):
    def setUp(self):
        focal = 640 / (2 * np.tan(np.deg2rad(50)))
        self.k = np.array([[focal, 0, 319.5], [0, focal, 179.5], [0, 0, 1]])
        self.transform = np.eye(4)

    def test_local_support_and_zone_order(self):
        support = zone_rgb_support(self.k, self.transform, (640, 360), (36, 64))
        self.assertEqual(support.shape, (64, 36, 64))
        self.assertTrue(np.isfinite(support).all())
        self.assertTrue(((support >= 0) & (support <= 1)).all())
        self.assertTrue((support.sum((1, 2)) > 0).all())
        self.assertEqual(float(support[0, -1, -1]), 0.0)
        self.assertGreater(float(support[0, 8, 21]), float(support[0, 8, 43]))
        self.assertGreater(float(support[63, 27, 43]), float(support[63, 27, 21]))

    def test_baseline_translation_changes_possible_support(self):
        coincident = zone_rgb_support(self.k, self.transform, (640, 360), (36, 64))
        shifted = self.transform.copy()
        shifted[0, 3] = .02
        offset = zone_rgb_support(self.k, shifted, (640, 360), (36, 64))
        self.assertGreater(float(np.max(np.abs(coincident - offset))), .01)

    def test_depth_sampling_cannot_drop_geometric_support(self):
        shifted = self.transform.copy()
        shifted[:3, 3] = [.15, -.08, .04]
        sparse = zone_rgb_support(self.k, shifted, (640, 360), (36, 64), depth_samples=2)
        dense = zone_rgb_support(self.k, shifted, (640, 360), (36, 64), depth_samples=48)
        self.assertTrue(np.array_equal(sparse > 0, dense > 0))

    def test_rejects_nonrigid_or_invalid_calibration(self):
        bad = self.transform.copy()
        bad[0, 0] = 1.2
        with self.assertRaisesRegex(ValueError, 'rigid'):
            zone_rgb_support(self.k, bad, (640, 360), (36, 64))
        with self.assertRaisesRegex(ValueError, 'feature'):
            zone_rgb_support(self.k, self.transform, (640, 360), (361, 64))


if __name__ == '__main__':
    unittest.main()
