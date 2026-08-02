#!/usr/bin/env python3
"""Tests for D14 direction-preserving RAFT features."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_stage_c_d14_thor_magni_explicit_motion_features import (
    direction_preserving_grid,
)


class D14ExplicitMotionFeatureTests(unittest.TestCase):
    def test_grid_preserves_horizontal_direction(self) -> None:
        flow = np.zeros((2, 12, 18), dtype=np.float32)
        flow[0] = 1.8
        residual = flow.copy()
        features = direction_preserving_grid(flow, residual)
        self.assertEqual(features.shape, (8, 3, 6))
        np.testing.assert_allclose(features[0], 0.1)
        np.testing.assert_allclose(features[1], 0.0)
        np.testing.assert_allclose(features[3], 0.1)
        np.testing.assert_allclose(features[4], 0.0)


if __name__ == "__main__":
    unittest.main()
