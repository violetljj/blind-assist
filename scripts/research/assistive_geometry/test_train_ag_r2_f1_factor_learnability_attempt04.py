#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ag_r2_f1_factor_learnability_attempt04 import (  # noqa: E402
    DEPTH_UNCERTAINTY_CONFIG,
    GEOMETRY_CONFIG,
    calibrated_depth_sigma,
)


class Attempt04Test(unittest.TestCase):
    def test_frozen_depth_uncertainty_formula(self) -> None:
        log_sigma = torch.log(torch.tensor([[[[0.4, 0.5]]]]))
        point = torch.log(torch.tensor([[[[2.0, 4.0]]]]))
        logs = torch.stack([point, point + 0.1, point - 0.1])
        base = point + torch.tensor([[[[0.2, -0.3]]]])
        valid = torch.tensor([[[[0.9, 0.8]]]])
        value = calibrated_depth_sigma(log_sigma, logs, point, base, valid)
        epistemic = logs.std(dim=0, correction=0)
        expected = (log_sigma.exp() + epistemic + 2.0 * (point - base).abs()) * DEPTH_UNCERTAINTY_CONFIG["scale"]
        self.assertTrue(torch.allclose(value, expected, atol=1.0e-7))

    def test_geometry_config_is_sensor_conditioned(self) -> None:
        self.assertEqual(GEOMETRY_CONFIG["height"]["scope"], "parent_vio_world_plane")
        self.assertEqual(GEOMETRY_CONFIG["support_sigma"]["source"], "coverage_complement")


if __name__ == "__main__":
    unittest.main()
