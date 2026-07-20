#!/usr/bin/env python3
"""Pure tests for traffic-cone detection visual features."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import extract_public_video_traffic_cone_detection_features as subject


class TrafficConeDetectionFeaturesTest(unittest.TestCase):
    def test_orange_box_has_high_warm_fraction(self) -> None:
        hsv = np.zeros((20, 20, 3), dtype=np.uint8)
        hsv[:, :, 0] = 15
        hsv[:, :, 1] = 220
        hsv[:, :, 2] = 220
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        features = subject.box_visual_features(image, [0, 0, 20, 20])
        self.assertGreater(features["warm_color_fraction"], 0.95)
        self.assertGreater(features["high_saturation_fraction"], 0.95)

    def test_black_box_has_high_dark_fraction(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        features = subject.box_visual_features(image, [0, 0, 20, 20])
        self.assertGreater(features["dark_fraction"], 0.95)
        self.assertEqual(0.0, features["warm_color_fraction"])

    def test_empty_clipped_box_fails_closed(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            subject.box_visual_features(image, [30, 30, 40, 40])


if __name__ == "__main__":
    unittest.main()
