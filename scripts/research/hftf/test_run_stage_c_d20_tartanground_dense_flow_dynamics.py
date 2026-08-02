#!/usr/bin/env python3
"""Tests for D20 explicit dense-flow dynamics."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torchvision.models import mobilenet_v3_small

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d20_tartanground_dense_flow_dynamics import (
    DenseFlowDynamicsOnsetEncoder,
    dense_dynamics_tensor,
)


class D20DenseFlowDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.weights = Path(cls.temporary.name) / "mobilenet.pth"
        torch.save(mobilenet_v3_small(weights=None).state_dict(), cls.weights)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_zero_flow_produces_exact_zero_dynamics(self) -> None:
        current = torch.randn(2, 16, 5, 7)
        history = current[:, None].repeat(1, 4, 1, 1, 1)
        flow = torch.zeros(2, 4, 2, 5, 7)
        valid = torch.ones(2, 4, 1, 5, 7)
        dynamics = dense_dynamics_tensor(
            history,
            current,
            flow,
            valid,
        )
        self.assertEqual(tuple(dynamics.shape), (2, 4, 20, 5, 7))
        self.assertEqual(int(torch.count_nonzero(dynamics)), 0)

    def test_flow_components_are_normalized_and_directional(self) -> None:
        current = torch.zeros(1, 16, 4, 8)
        history = current[:, None].repeat(1, 4, 1, 1, 1)
        flow = torch.zeros(1, 4, 2, 4, 8)
        flow[:, :, 0] = 4.0
        flow[:, :, 1] = -1.0
        valid = torch.ones(1, 4, 1, 4, 8)
        dynamics = dense_dynamics_tensor(
            history,
            current,
            flow,
            valid,
        )
        self.assertAlmostEqual(float(dynamics[0, 0, 16, 0, 0]), 0.5)
        self.assertAlmostEqual(float(dynamics[0, 0, 17, 0, 0]), -0.25)

    def test_repeated_current_is_invariant_to_temporal_weights(self) -> None:
        torch.manual_seed(17)
        model = DenseFlowDynamicsOnsetEncoder(self.weights).eval()
        current = torch.randn(1, 1, 3, 128, 224)
        repeated = current.repeat(1, 5, 1, 1, 1)
        flow = torch.zeros(1, 4, 2, 64, 112)
        with torch.no_grad():
            first = model(repeated, flow)
            model.temporal_motion[0].weight.fill_(3.0)
            model.temporal_motion[3].weight.fill_(2.0)
            model.temporal_motion[6].weight.fill_(4.0)
            model.motion_output.weight.fill_(5.0)
            second = model(repeated, flow)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
