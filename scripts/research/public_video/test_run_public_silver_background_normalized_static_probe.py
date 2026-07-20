#!/usr/bin/env python3
"""Pure tests for the background-normalized static feature."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import run_public_silver_background_normalized_static_probe as probe


def textured_scene(size: int = 320) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    rng = np.random.default_rng(20260719)
    for _ in range(800):
        x, y = rng.integers(0, size, size=2)
        value = int(rng.integers(40, 255))
        cv2.circle(image, (int(x), int(y)), 2, (value, value, value), -1)
    return image


class BackgroundNormalizedStaticProbeTest(unittest.TestCase):
    def test_masks_are_disjoint_and_nonempty(self) -> None:
        center, background = probe.comparison_masks(320)
        self.assertTrue(center.any())
        self.assertTrue(background.any())
        self.assertFalse(np.any(center & background))

    def test_global_translation_has_low_background_normalized_excess(self) -> None:
        previous = textured_scene()
        transform = np.float32([[1, 0, 8], [0, 1, 5]])
        current = cv2.warpAffine(previous, transform, (320, 320))
        row = probe.frame_pair_descriptor(previous, current)
        self.assertTrue(row["reliable"])
        self.assertLess(row["q90_excess"], 0.04)

    def test_central_nearfield_occluder_exceeds_background(self) -> None:
        previous = textured_scene()
        current = previous.copy()
        cv2.rectangle(current, (105, 180), (215, 319), (245, 245, 245), -1)
        row = probe.frame_pair_descriptor(previous, current)
        self.assertTrue(row["reliable"])
        self.assertGreater(row["q90_excess"], 0.10)

    def test_peripheral_change_does_not_create_central_excess(self) -> None:
        previous = textured_scene()
        current = previous.copy()
        cv2.rectangle(current, (0, 180), (55, 290), (245, 245, 245), -1)
        cv2.rectangle(current, (265, 180), (319, 290), (245, 245, 245), -1)
        row = probe.frame_pair_descriptor(previous, current)
        self.assertTrue(row["reliable"])
        self.assertLess(row["q90_excess"], 0.04)

    def test_score_uses_robust_levels_and_ignores_unreliable_rows(self) -> None:
        rows = []
        for value in (0.0, 0.1, 0.2, 0.3):
            row = {"reliable": True}
            for key in probe.FEATURE_KEYS:
                row[key] = value
            rows.append(row)
        unreliable = {"reliable": False}
        for key in probe.FEATURE_KEYS:
            unreliable[key] = 99.0
        rows.append(unreliable)
        scores = probe.score_descriptors(rows)
        self.assertEqual(4, scores["reliable_transition_count"])
        self.assertAlmostEqual(0.15, scores["median_q90_excess"])
        self.assertAlmostEqual(0.225, scores["q75_q90_excess"])


if __name__ == "__main__":
    unittest.main()
