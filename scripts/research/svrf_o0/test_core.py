from __future__ import annotations

import unittest

import numpy as np

from scripts.research.svrf_o0.core import (
    fuse_region_risk,
    relative_depth_approach_rate,
    robust_scale_shift_align,
    rotation_compensated_local_expansion,
)


class SvrfO0CoreTest(unittest.TestCase):
    def test_scale_shift_alignment_recovers_affine_drift(self) -> None:
        previous = np.linspace(1.0, 4.0, 100).reshape(10, 10)
        current = (previous - 0.4) / 1.7
        result = robust_scale_shift_align(previous, current, np.ones_like(previous, dtype=bool))
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.scale, 1.7, places=8)
        self.assertAlmostEqual(result.shift, 0.4, places=8)

    def test_depth_approach_rate_is_positive_when_aligned_depth_decreases(self) -> None:
        previous = np.full((8, 8), 2.0)
        current = np.full((8, 8), 1.8)
        rate = relative_depth_approach_rate(previous, current, 0.1, np.ones((8, 8), dtype=bool))
        self.assertGreater(float(np.nanmedian(rate)), 0.0)

    def test_rotation_compensated_affine_expansion_recovers_rate(self) -> None:
        grid = np.asarray([(x, y) for y in range(4) for x in range(4)], dtype=np.float64)
        center = np.median(grid, axis=0)
        current = center + 1.05 * (grid - center) + np.asarray([2.0, -1.0])
        result = rotation_compensated_local_expansion(grid, current, 0.1)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.expansion_per_second, 0.5, places=7)

    def test_fusion_requires_quality_and_agreement(self) -> None:
        high = fuse_region_risk(0.9, 0.8, 0.8, 0.9, 0.95)
        self.assertEqual(high["state"], "HIGH_RELATIVE_RISK")
        self.assertEqual(fuse_region_risk(0.9, 0.8, 0.8, 0.9, 0.4)["state"], "UNKNOWN")
