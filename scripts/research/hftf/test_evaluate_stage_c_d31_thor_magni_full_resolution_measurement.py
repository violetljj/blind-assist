#!/usr/bin/env python3
"""Tests for D31 full-resolution static box materialization."""

from __future__ import annotations

import unittest

import numpy as np

from extract_stage_c_d29_thor_magni_object_slots import (
    FEATURE_COUNT,
    MAX_SLOTS,
)
from evaluate_stage_c_d31_thor_magni_full_resolution_measurement import (
    static_slots,
)


class D31FullResolutionMeasurementTests(unittest.TestCase):
    def test_static_slots_match_frozen_normalization_and_selection(self) -> None:
        boxes = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.8, 0.5],
                [0.6, 0.4, 0.9, 0.9, 0.9],
            ],
            dtype=np.float32,
        )
        slots, mask = static_slots(boxes)
        self.assertEqual(slots.shape, (MAX_SLOTS, FEATURE_COUNT))
        self.assertEqual(mask.tolist(), [True, True] + [False] * 6)
        self.assertAlmostEqual(float(slots[0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(slots[0, 3]), 0.5, places=6)
        self.assertAlmostEqual(float(slots[0, 5]), 0.9, places=6)
        self.assertTrue(np.allclose(slots[:, 6:], 0.0))


if __name__ == "__main__":
    unittest.main()
