#!/usr/bin/env python3
"""Structural tests for the D16 TartanGround feature extractor."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import frame_tensor


class D16TartanGroundFeatureTests(unittest.TestCase):
    def test_frame_tensor_accepts_square_tartanground_rgb(self) -> None:
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        tensor = frame_tensor(image)
        self.assertEqual(tuple(tensor.shape), (3, 128, 224))
        self.assertTrue(np.isfinite(tensor.numpy()).all())


if __name__ == "__main__":
    unittest.main()
