#!/usr/bin/env python3
"""Pure tests for camera-motion-compensated occupancy features."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import run_public_silver_motion_compensated_occupancy_probe as probe


def textured_scene(size: int = 320) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    rng = np.random.default_rng(20260717)
    for _ in range(500):
        x, y = rng.integers(0, size, size=2)
        color = int(rng.integers(80, 255))
        cv2.circle(image, (int(x), int(y)), 2, (color, color, color), -1)
    return image


class PublicSilverMotionCompensatedOccupancyProbeTest(unittest.TestCase):
    def test_global_translation_is_compensated(self) -> None:
        previous = textured_scene()
        transform = np.float32([[1, 0, 8], [0, 1, 5]])
        current = cv2.warpAffine(previous, transform, (320, 320))
        _vector, summary = probe.frame_pair_descriptor(previous, current)
        self.assertTrue(summary["homography_success"])
        self.assertLess(summary["compensated_lower_mean"], 0.04)

    def test_near_field_occluder_survives_compensation(self) -> None:
        previous = textured_scene()
        current = previous.copy()
        cv2.rectangle(current, (70, 150), (285, 319), (235, 235, 235), -1)
        _vector, summary = probe.frame_pair_descriptor(previous, current)
        self.assertGreater(summary["compensated_lower_fraction_ge_018"], 0.20)

    def test_episode_pool_is_temporally_sensitive(self) -> None:
        first = np.zeros(31, dtype=np.float64)
        second = np.ones(31, dtype=np.float64)
        forward = probe.episode_vector(np.stack([first, second]))
        reverse = probe.episode_vector(np.stack([second, first]))
        self.assertEqual(124, len(forward))
        self.assertFalse(np.array_equal(forward, reverse))

    def test_compact_pool_ignores_unregistered_residual_magnitude(self) -> None:
        registered = np.zeros(31, dtype=np.float64)
        registered[0] = 1.0
        registered[4] = 0.2
        registered[5] = 0.8
        registered[13] = 0.08
        registered[15] = 0.20
        registered[17] = 0.10
        failed = np.ones(31, dtype=np.float64)
        failed[0] = 0.0
        failed[4] = 0.0
        failed[5] = 0.0
        compact = probe.compact_episode_vector(np.stack([registered, failed]))
        self.assertEqual(7, len(compact))
        self.assertAlmostEqual(0.5, compact[0])
        self.assertAlmostEqual(0.08, compact[4])

    def test_independent_direction_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "secondary-corridor-causal" / "motion.json"
            with self.assertRaisesRegex(ValueError, "independent model direction"):
                probe.reject_independent_direction(path)


if __name__ == "__main__":
    unittest.main()
