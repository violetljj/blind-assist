#!/usr/bin/env python3
"""Tests for D14 explicit-motion onset canary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d14_thor_magni_explicit_motion_onset_canary import (
    CurrentMotionOnsetHead,
)


class D14ExplicitMotionCanaryTests(unittest.TestCase):
    def test_current_arm_is_invariant_to_motion_values(self) -> None:
        torch.manual_seed(17)
        model = CurrentMotionOnsetHead().eval()
        current = torch.randn(2, 576, 4, 7)
        first = model(
            current,
            torch.zeros(2, 4, 8, 3, 6),
            arm="current",
        )
        second = model(
            current,
            torch.randn(2, 4, 8, 3, 6),
            arm="current",
        )
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
