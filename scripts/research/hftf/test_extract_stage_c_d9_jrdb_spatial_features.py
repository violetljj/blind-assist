#!/usr/bin/env python3
"""Tests for JRDB front-crop spatial feature extraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_stage_c_d9_jrdb_spatial_features import front_crop_tensor


class JrdbSpatialFeatureTests(unittest.TestCase):
    def test_front_crop_uses_panorama_center(self) -> None:
        image = np.zeros((480, 3760, 3), dtype=np.uint8)
        image[:, 1253:2507] = 255
        tensor = front_crop_tensor(image)
        self.assertEqual(tuple(tensor.shape), (3, 128, 224))
        self.assertGreater(float(tensor.mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
