#!/usr/bin/env python3
"""Tests for the D10 trainable-tail temporal student."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d10_thor_magni_trainable_tail_canary import (
    DEFAULT_PRETRAINED,
    TrainableTailTemporalStudent,
)


class D10TrainableTailStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = TrainableTailTemporalStudent(DEFAULT_PRETRAINED).eval()

    def test_only_late_backbone_is_trainable(self) -> None:
        early = list(self.model.encoder[:9].parameters())
        late = list(self.model.encoder[9:].parameters())
        self.assertTrue(early)
        self.assertTrue(late)
        self.assertFalse(any(parameter.requires_grad for parameter in early))
        self.assertTrue(all(parameter.requires_grad for parameter in late))

    def test_repeated_current_is_invariant_to_temporal_weights(self) -> None:
        frames = torch.zeros(1, 5, 3, 128, 224)
        first = self.model(frames, arm="current")
        with torch.no_grad():
            self.model.temporal_residual_weight.fill_(2.0)
        second = self.model(frames, arm="current")
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
