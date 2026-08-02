#!/usr/bin/env python3
"""Tests for the D17 early-temporal onset representation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torchvision.models import mobilenet_v3_small

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d17_tartanground_early_temporal_onset_canary import (
    EarlyTemporalOnsetEncoder,
    masked_cell_loss,
)


class D17EarlyTemporalOnsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.weights = Path(cls.temporary.name) / "mobilenet.pth"
        torch.save(mobilenet_v3_small(weights=None).state_dict(), cls.weights)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_repeated_current_is_exact_temporal_zero(self) -> None:
        torch.manual_seed(17)
        model = EarlyTemporalOnsetEncoder(self.weights).eval()
        current = torch.randn(2, 1, 3, 64, 96)
        repeated = current.repeat(1, 5, 1, 1, 1)
        with torch.no_grad():
            first = model(repeated)
            model.temporal_motion[0].weight.fill_(3.0)
            model.temporal_motion[3].weight.fill_(2.0)
            model.temporal_motion[6].weight.fill_(4.0)
            model.motion_output.weight.fill_(5.0)
            second = model(repeated)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(first.shape), (2, 4, 6, 6))

    def test_real_history_reaches_early_motion_branch(self) -> None:
        torch.manual_seed(23)
        model = EarlyTemporalOnsetEncoder(self.weights).eval()
        frames = torch.randn(1, 5, 3, 64, 96)
        repeated = frames[:, -1:].repeat(1, 5, 1, 1, 1)
        with torch.no_grad():
            model.motion_output.weight.fill_(0.1)
            history_output = model(frames)
            current_output = model(repeated)
        self.assertFalse(torch.equal(history_output, current_output))

    def test_masked_loss_ignores_ineligible_cells(self) -> None:
        logits = torch.zeros(1, 4, 6, 6)
        onset = torch.zeros_like(logits)
        eligible = torch.zeros_like(logits)
        eligible[0, 0, 2, 3] = 1.0
        first = masked_cell_loss(
            logits,
            onset,
            eligible,
            torch.ones(1),
            torch.ones(4),
        )
        logits[0, 1, 0, 0] = 100.0
        second = masked_cell_loss(
            logits,
            onset,
            eligible,
            torch.ones(1),
            torch.ones(4),
        )
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
