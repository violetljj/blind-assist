#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import torch

from train_ag_r2_f1_factor_learnability import (
    DEPTHART_PYRAMID_CHANNELS,
    FactorOnlyHead,
    PRIMARY_METRICS,
    compute_losses,
    eligibility,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-fit-baselines-tum13-r0/result.json"


class FactorLearnabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        result = json.loads(BASELINE_RESULT.read_text(encoding="utf-8"))
        cls.baseline = result["baseline_parameters"]
        cls.normalization = result["optimizer_normalization"]

    def test_head_emits_complete_factor_shapes(self) -> None:
        model = FactorOnlyHead(self.baseline, hidden=16).eval()
        feature = torch.zeros(1, DEPTHART_PYRAMID_CHANNELS, 6, 8)
        base = torch.full((1, 1, 6, 8), 2.0)
        with torch.no_grad():
            output = model(feature, base)
        self.assertEqual(output["predicted_log_depth"].shape, (1, 1, 6, 8))
        self.assertEqual(output["support_plane_normal_camera_xyz"].shape, (1, 3))
        self.assertEqual(output["camera_height_m"].shape, (1,))
        self.assertEqual(output["support_residual_sigma_m"].shape, (1,))
        self.assertTrue(torch.isfinite(torch.cat([value.flatten() for value in output.values()])).all())
        self.assertAlmostEqual(float(torch.linalg.vector_norm(output["support_plane_normal_camera_xyz"][0])), 1.0, places=5)
        self.assertLess(float(output["depth_gate"][0]), 1.0e-4)

    def test_thirteen_component_objective_is_finite(self) -> None:
        model = FactorOnlyHead(self.baseline, hidden=16)
        output = model(
            torch.randn(1, DEPTHART_PYRAMID_CHANNELS, 6, 8),
            torch.full((1, 1, 6, 8), 2.0),
        )
        targets = {
            "metric_depth_m": torch.full((1, 6, 8), 2.2),
            "metric_valid": torch.ones(1, 6, 8, dtype=torch.bool),
            "support": torch.rand(1, 6, 8),
            "support_valid": torch.ones(1, 6, 8, dtype=torch.bool),
            "support_residual": torch.randn(1, 6, 8) * 0.1,
            "obstacle": torch.rand(1, 6, 8),
            "boundary_distance": torch.rand(1, 6, 8) * 32.0,
            "evidence_valid": torch.ones(1, 6, 8, dtype=torch.bool),
            "support_plane_valid": torch.tensor(True),
            "support_normal": torch.tensor([0.0, 1.0, 0.0]),
            "camera_height_m": torch.tensor(1.4),
        }
        objective, losses = compute_losses(output, targets, self.normalization)
        self.assertEqual(len(losses), 13)
        self.assertTrue(torch.isfinite(objective))
        objective.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_unknown_masks_factor_values_but_trains_validity(self) -> None:
        model = FactorOnlyHead(self.baseline, hidden=16)
        output = model(
            torch.zeros(1, DEPTHART_PYRAMID_CHANNELS, 4, 4),
            torch.ones(1, 1, 4, 4),
        )
        false = torch.zeros(1, 4, 4, dtype=torch.bool)
        targets = {
            "metric_depth_m": torch.ones(1, 4, 4),
            "metric_valid": torch.ones(1, 4, 4, dtype=torch.bool),
            "support": torch.zeros(1, 4, 4),
            "support_valid": false,
            "support_residual": torch.zeros(1, 4, 4),
            "obstacle": torch.zeros(1, 4, 4),
            "boundary_distance": torch.full((1, 4, 4), 32.0),
            "evidence_valid": false,
            "support_plane_valid": torch.tensor(False),
            "support_normal": torch.zeros(3),
            "camera_height_m": torch.tensor(1.0),
        }
        _, losses = compute_losses(output, targets, self.normalization)
        for key in (
            "support_probability_brier",
            "support_plane_angular",
            "camera_height_log_huber",
            "support_residual_heteroscedastic_nll",
            "obstacle_evidence_brier",
            "boundary_probability_brier",
            "boundary_localization_heteroscedastic_nll",
        ):
            self.assertEqual(float(losses[key].detach()), 0.0)
        self.assertGreater(float(losses["support_validity_brier"].detach()), 0.0)
        self.assertGreater(float(losses["evidence_validity_brier"].detach()), 0.0)

    def test_selection_gate_requires_no_regret(self) -> None:
        baseline = {"overall_metrics": {key: 1.0 for key in PRIMARY_METRICS}}
        candidate = {"overall_metrics": {key: 1.0 for key in PRIMARY_METRICS}}
        candidate["overall_metrics"][PRIMARY_METRICS[0]] = 0.9
        self.assertTrue(eligibility(candidate, baseline)["eligible"])
        candidate["overall_metrics"][PRIMARY_METRICS[1]] = 1.1
        self.assertFalse(eligibility(candidate, baseline)["eligible"])


if __name__ == "__main__":
    unittest.main()
