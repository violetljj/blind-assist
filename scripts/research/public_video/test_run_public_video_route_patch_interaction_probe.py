#!/usr/bin/env python3

import unittest

import numpy as np

import run_public_video_route_patch_interaction_probe as probe


class RoutePatchInteractionTest(unittest.TestCase):
    def test_route_field_peaks_near_polyline(self) -> None:
        field = probe.route_field(16, 16, [[0.5, 0.9], [0.5, 0.6], [0.5, 0.3]], 1.5)
        self.assertGreater(float(field[:, 8].mean()), float(field[:, 0].mean()))

    def test_uniform_control_zeroes_contrast(self) -> None:
        visual = np.arange(4 * 4 * 3, dtype=np.float64).reshape(4, 4, 3)
        result = probe.interaction_features(visual, np.full((4, 4), 0.5))
        self.assertEqual(result.shape, (9,))
        np.testing.assert_allclose(result[-3:], 0.0)

    def test_projection_is_exactly_repeatable(self) -> None:
        first = probe.fixed_projection(384, 32, 0); second = probe.fixed_projection(384, 32, 0)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__": unittest.main()
