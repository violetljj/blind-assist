#!/usr/bin/env python3
"""Tests for D18 flow-aligned early temporal fusion."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torchvision.models import mobilenet_v3_small

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d18_tartanground_flow_aligned_onset_canary import (
    FlowAlignedEarlyTemporalOnsetEncoder,
    backward_warp,
)


class D18FlowAlignedOnsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.weights = Path(cls.temporary.name) / "mobilenet.pth"
        torch.save(mobilenet_v3_small(weights=None).state_dict(), cls.weights)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_current_to_history_flow_warps_history_to_current(self) -> None:
        history = torch.zeros(1, 1, 1, 5, 7)
        history[0, 0, 0, 2, 4] = 1.0
        flow = torch.zeros(1, 1, 2, 5, 7)
        flow[:, :, 0] = 1.0
        warped, valid = backward_warp(history, flow)
        self.assertEqual(float(warped[0, 0, 0, 2, 3]), 1.0)
        self.assertEqual(float(valid[0, 0, 0, 2, 3]), 1.0)
        self.assertEqual(float(valid[0, 0, 0, 2, -1]), 0.0)

    def test_repeated_current_zero_flow_is_exactly_invariant(self) -> None:
        torch.manual_seed(17)
        model = FlowAlignedEarlyTemporalOnsetEncoder(self.weights).eval()
        current = torch.randn(2, 1, 3, 128, 224)
        repeated = current.repeat(1, 5, 1, 1, 1)
        flow = torch.zeros(2, 4, 2, 64, 112)
        with torch.no_grad():
            first = model(repeated, flow)
            model.temporal_motion[0].weight.fill_(3.0)
            model.temporal_motion[3].weight.fill_(2.0)
            model.temporal_motion[6].weight.fill_(4.0)
            model.motion_output.weight.fill_(5.0)
            second = model(repeated, flow)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(first.shape), (2, 4, 6, 6))


if __name__ == "__main__":
    unittest.main()
