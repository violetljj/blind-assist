#!/usr/bin/env python3
"""Tests for the trainable THOR RGB history cache."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache_stage_c_d10_thor_magni_rgb_history import rgb_uint8


class ThorRgbHistoryCacheTests(unittest.TestCase):
    def test_rgb_resize_preserves_channel_order(self) -> None:
        bgr = np.zeros((32, 48, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255
        result = rgb_uint8(bgr)
        self.assertEqual(result.shape, (128, 224, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(int(result[:, :, 0].min()), 255)
        self.assertEqual(int(result[:, :, 1].max()), 0)


if __name__ == "__main__":
    unittest.main()
