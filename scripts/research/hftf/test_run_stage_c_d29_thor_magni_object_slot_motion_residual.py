#!/usr/bin/env python3
"""Tests for the D29 paired object-slot student."""

from __future__ import annotations

import unittest

import torch

from extract_stage_c_d29_thor_magni_object_slots import (
    FEATURE_COUNT,
    MAX_SLOTS,
    STATIC_FEATURE_COUNT,
)
from run_stage_c_d29_thor_magni_object_slot_motion_residual import (
    ObjectSlotMotionResidual,
    flip_slot_features,
)


class D29ObjectSlotStudentTests(unittest.TestCase):
    def test_zero_initialized_residual_and_monotonic_fields(self) -> None:
        torch.manual_seed(17)
        model = ObjectSlotMotionResidual().eval()
        slots = torch.randn(3, MAX_SLOTS, FEATURE_COUNT)
        mask = torch.tensor(
            [
                [True, True, False, False, False, False, False, False],
                [True, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
            ]
        )
        current, history = model(slots, mask)
        self.assertEqual(tuple(current.shape), (3, 3, 4))
        self.assertTrue(torch.equal(current, history))
        self.assertTrue(torch.all(current <= 0))
        self.assertTrue(torch.all(current >= -1))
        self.assertTrue(torch.all(current[:, :, 1:] >= current[:, :, :-1]))

    def test_horizontal_flip_changes_only_signed_x_features(self) -> None:
        slots = torch.arange(
            MAX_SLOTS * FEATURE_COUNT,
            dtype=torch.float32,
        ).reshape(MAX_SLOTS, FEATURE_COUNT)
        flipped = flip_slot_features(slots)
        self.assertTrue(torch.equal(flipped[:, 0], -slots[:, 0]))
        self.assertTrue(
            torch.equal(
                flipped[:, 1:STATIC_FEATURE_COUNT],
                slots[:, 1:STATIC_FEATURE_COUNT],
            )
        )
        changed = {0}
        for lag in range(4):
            offset = STATIC_FEATURE_COUNT + lag * 7
            changed.update({offset, offset + 2})
        for index in range(FEATURE_COUNT):
            if index not in changed:
                self.assertTrue(
                    torch.equal(flipped[:, index], slots[:, index])
                )


if __name__ == "__main__":
    unittest.main()
