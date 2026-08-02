#!/usr/bin/env python3
"""Tests for D16 TartanGround onset temporal baseline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d8_thor_magni_equal_capacity_temporal_head import (
    TemporalSpatialActionabilityHead,
)


class D16TartanGroundTemporalTests(unittest.TestCase):
    def test_four_output_current_arm_is_temporally_invariant(self) -> None:
        torch.manual_seed(17)
        model = TemporalSpatialActionabilityHead(
            576,
            4,
            7,
            output_count=4,
        ).eval()
        current = torch.randn(2, 1, 576, 4, 7)
        repeated = current.repeat(1, 5, 1, 1, 1)
        first = model(repeated)
        with torch.no_grad():
            model.temporal_residual_weight.fill_(3.0)
        second = model(repeated)
        torch.testing.assert_close(first, second)
        self.assertEqual(tuple(first.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
