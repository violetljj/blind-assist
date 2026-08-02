#!/usr/bin/env python3
"""Tests for D21 ConvGRU future-state dynamics."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torchvision.models import mobilenet_v3_small

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d21_tartanground_convgru_future_state import (
    ConvGRUCell,
    ConvGRUFutureStateOnsetEncoder,
)


class D21ConvGRUFutureStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.weights = Path(cls.temporary.name) / "mobilenet.pth"
        torch.save(mobilenet_v3_small(weights=None).state_dict(), cls.weights)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_zero_input_preserves_exact_zero_hidden_state(self) -> None:
        cell = ConvGRUCell(20, 16)
        value = torch.zeros(2, 20, 5, 7)
        hidden = torch.zeros(2, 16, 5, 7)
        for _ in range(4):
            hidden = cell(value, hidden)
        self.assertEqual(int(torch.count_nonzero(hidden)), 0)

    def test_repeated_current_is_invariant_to_recurrent_weights(self) -> None:
        torch.manual_seed(17)
        model = ConvGRUFutureStateOnsetEncoder(self.weights).eval()
        current = torch.randn(1, 1, 3, 128, 224)
        repeated = current.repeat(1, 5, 1, 1, 1)
        flow = torch.zeros(1, 4, 2, 64, 112)
        with torch.no_grad():
            first = model(repeated, flow)
            model.future_state.gates.weight.fill_(3.0)
            model.future_state.candidate.weight.fill_(2.0)
            model.motion_output.weight.fill_(5.0)
            second = model(repeated, flow)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
