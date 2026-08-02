#!/usr/bin/env python3
"""Tests for D19 geometry-dynamics pretraining."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torchvision.models import mobilenet_v3_small

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d19_tartanground_geometry_dynamics_pretraining import (
    GeometryDynamicsOnsetEncoder,
    decode_teacher_fields,
)


class D19GeometryDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.weights = Path(cls.temporary.name) / "mobilenet.pth"
        torch.save(mobilenet_v3_small(weights=None).state_dict(), cls.weights)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_teacher_decoder_selects_body_and_head(self) -> None:
        labels = {}
        for horizon_index, horizon in enumerate(
            ("current", "near", "far")
        ):
            risk = np.empty((3, 6, 6), dtype=object)
            risk[:] = None
            known = np.zeros((3, 6, 6), dtype=np.int64)
            for height_index in (1, 2):
                known[height_index] = 1
                risk[height_index] = (
                    horizon_index * 0.25 + height_index * 0.1
                )
            labels[horizon] = {
                "known_target": known.tolist(),
                "risk_score_target_nullable": risk.tolist(),
            }
        risk, known = decode_teacher_fields(labels)
        self.assertEqual(tuple(risk.shape), (6, 6, 6))
        self.assertEqual(tuple(known.shape), (6, 6, 6))
        self.assertAlmostEqual(float(risk[0, 0, 0]), 0.1)
        self.assertAlmostEqual(float(risk[-1, 0, 0]), 0.7)

    def test_future_head_transfers_exactly_to_onset(self) -> None:
        torch.manual_seed(17)
        model = GeometryDynamicsOnsetEncoder(self.weights)
        with torch.no_grad():
            model.dynamics_head.weight.copy_(
                torch.arange(
                    model.dynamics_head.weight.numel(),
                    dtype=torch.float32,
                ).reshape_as(model.dynamics_head.weight)
            )
            model.dynamics_head.bias.copy_(torch.arange(6).float())
            model.transfer_future_head_to_onset()
        torch.testing.assert_close(
            model.onset_head.weight,
            model.dynamics_head.weight[2:6],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            model.onset_head.bias,
            model.dynamics_head.bias[2:6],
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
