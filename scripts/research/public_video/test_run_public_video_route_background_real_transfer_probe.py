#!/usr/bin/env python3

import unittest

import numpy as np

import run_public_video_route_background_real_transfer_probe as probe


class RouteBackgroundProbeTest(unittest.TestCase):
    def test_combine_features_preserves_pre_registered_order(self) -> None:
        route = np.arange(26, dtype=np.float64).reshape(2, 13)
        global_values = np.arange(14, dtype=np.float64).reshape(2, 7) + 100
        result = probe.combine_features(route, global_values)
        self.assertEqual(result.shape, (2, 20))
        np.testing.assert_array_equal(result[:, :13], route)
        np.testing.assert_array_equal(result[:, 13:], global_values)

    def test_combine_features_rejects_misalignment(self) -> None:
        with self.assertRaises(ValueError):
            probe.combine_features(np.zeros((2, 13)), np.zeros((3, 7)))


if __name__ == "__main__":
    unittest.main()
