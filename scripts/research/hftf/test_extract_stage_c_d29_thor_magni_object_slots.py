#!/usr/bin/env python3
"""Tests for D29 object-slot feature extraction."""

from __future__ import annotations

import unittest

import numpy as np

from extract_stage_c_d29_thor_magni_object_slots import (
    FEATURE_COUNT,
    LAG_FEATURE_COUNT,
    MAX_SLOTS,
    STATIC_FEATURE_COUNT,
    selected_boxes,
    slot_features,
)


class D29ObjectSlotTests(unittest.TestCase):
    def test_selection_uses_confidence_times_sqrt_area(self) -> None:
        boxes = np.asarray(
            [
                [0.1, 0.1, 0.2, 0.2, 0.9],
                [0.2, 0.2, 0.6, 0.6, 0.5],
                [0.3, 0.3, 0.5, 0.5, 0.8],
            ],
            dtype=np.float32,
        )
        selected = selected_boxes(boxes, max_slots=2)
        self.assertTrue(np.allclose(selected[0], boxes[1]))
        self.assertTrue(np.allclose(selected[1], boxes[2]))

    def test_uniform_flow_produces_normalized_slot_features(self) -> None:
        boxes = np.asarray(
            [[0.25, 0.25, 0.75, 0.75, 0.8]],
            dtype=np.float32,
        )
        flow = np.zeros((4, 2, 20, 40), dtype=np.float32)
        for lag in range(4):
            flow[lag, 0] = float(lag + 1)
            flow[lag, 1] = float(-(lag + 1))
        slots, mask, lag_valid = slot_features(boxes, flow)
        self.assertEqual(slots.shape, (MAX_SLOTS, FEATURE_COUNT))
        self.assertEqual(mask.tolist(), [True] + [False] * 7)
        self.assertTrue(np.allclose(slots[0, :6], [0, 0.75, 0.5, 0.5, 0.5, 0.8]))
        for lag in range(4):
            offset = STATIC_FEATURE_COUNT + lag * LAG_FEATURE_COUNT
            self.assertAlmostEqual(
                float(slots[0, offset]),
                float(lag + 1) / 40.0,
                places=6,
            )
            self.assertAlmostEqual(
                float(slots[0, offset + 1]),
                float(-(lag + 1)) / 20.0,
                places=6,
            )
            self.assertAlmostEqual(float(slots[0, offset + 2]), 0.0)
            self.assertAlmostEqual(float(slots[0, offset + 3]), 0.0)
            self.assertGreater(float(lag_valid[0, lag]), 0.0)


if __name__ == "__main__":
    unittest.main()
