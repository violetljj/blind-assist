#!/usr/bin/env python3

import unittest

import numpy as np

import run_public_video_real_marker_distance_transfer_probe as probe


class RealMarkerDistanceTest(unittest.TestCase):
    def test_union_distance_target_handles_clear_and_multiple_boxes(self) -> None:
        clear = probe.union_distance_target([], image_width=100, image_height=100, grid_width=10, grid_height=10, sigma_patches=1.5)
        self.assertTrue(np.all(clear == 0))
        target = probe.union_distance_target([{"xyxy": [10, 10, 30, 30]}, {"xyxy": [70, 70, 90, 90]}],
                                             image_width=100, image_height=100, grid_width=10, grid_height=10, sigma_patches=1.5)
        self.assertEqual(float(target.max()), 1.0)
        self.assertGreater(int((target == 1).sum()), 1)

    def test_patch_weights_equalize_near_far(self) -> None:
        targets = np.asarray([0.0, 0.0, 0.5, 1.0, 0.0, 0.8])
        sources = np.asarray(["a", "a", "a", "a", "b", "b"])
        frames = np.asarray(["a1", "a1", "a1", "a1", "b1", "b1"])
        weights = probe.patch_weights(targets, sources, frames, 0.25)
        self.assertAlmostEqual(float(weights[targets < .25].sum()), float(weights[targets >= .25].sum()))


if __name__ == "__main__": unittest.main()
