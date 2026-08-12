#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from train_ag_r2_f1_factor_learnability import DEFAULT_BASELINE_RESULT, DEPTHART_PYRAMID_CHANNELS
from train_ag_r2_f1_factor_learnability_attempt02 import (
    COMPONENT_METRICS,
    FactorSplitHead,
    choose_components,
)


class Attempt02Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = json.loads(DEFAULT_BASELINE_RESULT.read_text(encoding="utf-8"))[
            "baseline_parameters"
        ]

    def test_zero_residual_is_exact_nonlearned_baseline(self) -> None:
        model = FactorSplitHead(self.baseline).eval()
        feature = torch.randn(1, DEPTHART_PYRAMID_CHANNELS, 7, 9)
        base = torch.rand(1, 1, 7, 9) + 0.5
        with torch.no_grad():
            output = model(feature, base)
        self.assertTrue(
            torch.allclose(
                output["predicted_log_depth"],
                torch.full_like(output["predicted_log_depth"], self.baseline["depth_log_scale"]),
            )
        )
        self.assertTrue(
            torch.allclose(
                output["support_probability"],
                torch.full_like(output["support_probability"], self.baseline["support_probability"]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output["obstacle_probability"],
                torch.full_like(
                    output["obstacle_probability"],
                    self.baseline["obstacle_evidence_probability"],
                ),
                atol=1e-6,
            )
        )
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 88566)
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_component_selector_can_mix_safe_steps(self) -> None:
        metric_names = sorted({metric for values in COMPONENT_METRICS.values() for metric in values})
        baseline = {"overall_metrics": {metric: 1.0 for metric in metric_names}}
        candidates = []
        for step in (0, 100, 200):
            values = {metric: 1.0 for metric in metric_names}
            if step == 100:
                values["obstacle_brier"] = 0.8
                values["camera_height_abs_log_error"] = 1.2
            if step == 200:
                values["camera_height_abs_log_error"] = 0.7
                values["obstacle_brier"] = 1.1
            candidates.append(
                {
                    "step": step,
                    "evaluation": {"overall_metrics": values},
                    "checkpoint": {"sha256": f"{step:064d}", "path": "unused"},
                }
            )
        selected, _ = choose_components(candidates, baseline)
        self.assertEqual(selected["obstacle"]["step"], 100)
        self.assertEqual(selected["camera_height"]["step"], 200)
        self.assertEqual(selected["support_probability"]["step"], 0)


if __name__ == "__main__":
    unittest.main()
