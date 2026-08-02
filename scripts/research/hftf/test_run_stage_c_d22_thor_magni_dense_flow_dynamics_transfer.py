#!/usr/bin/env python3
"""Tests for D22 THOR dense-flow dynamics transfer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torchvision.models import mobilenet_v3_small

from run_stage_c_d20_tartanground_dense_flow_dynamics import (
    dense_dynamics_tensor,
)
from run_stage_c_d22_thor_magni_dense_flow_dynamics_transfer import (
    ThorDenseFlowDynamicsEncoder,
    build_gate,
    masked_loss,
)


class D22ThorDenseFlowTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.weights = Path(cls.temporary.name) / "mobilenet.pth"
        torch.save(mobilenet_v3_small(weights=None).state_dict(), cls.weights)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_zero_flow_repeated_current_has_zero_dynamics(self) -> None:
        current = torch.randn(2, 16, 64, 112)
        aligned = current.unsqueeze(1).repeat(1, 4, 1, 1, 1)
        flow = torch.zeros(2, 4, 2, 64, 112)
        valid = torch.ones(2, 4, 1, 64, 112)
        dynamics = dense_dynamics_tensor(aligned, current, flow, valid)
        self.assertEqual(int(torch.count_nonzero(dynamics)), 0)

    def test_model_forward_backward_and_gate(self) -> None:
        model = ThorDenseFlowDynamicsEncoder(self.weights)
        frames = torch.randn(1, 5, 3, 128, 224)
        flow = torch.zeros(1, 4, 2, 64, 112)
        logits = model(frames, flow)
        self.assertEqual(tuple(logits.shape), (1, 2))
        logits.sum().backward()
        aggregate = {
            "source_macro.auroc": {"mean": 0.011, "positive_count": 3},
            "source_macro.average_precision": {
                "mean": 0.006,
                "positive_count": 3,
            },
            "pooled_macro.auroc": {"mean": -0.004},
            "pooled_macro.average_precision": {"mean": 0.0},
            "by_target.proximity.source_macro.auroc": {"mean": 0.001},
            "by_target.proximity.source_macro.average_precision": {
                "mean": 0.001
            },
            "by_target.corridor.source_macro.auroc": {"mean": 0.001},
            "by_target.corridor.source_macro.average_precision": {
                "mean": 0.001
            },
        }
        self.assertTrue(build_gate(aggregate)["supported"])

    def test_loss_accepts_one_evaluable_target(self) -> None:
        logits = torch.zeros(2, 2, requires_grad=True)
        labels = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        eligible = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        sample_weights = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        loss = masked_loss(
            logits,
            labels,
            eligible,
            sample_weights,
            torch.ones(2),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
