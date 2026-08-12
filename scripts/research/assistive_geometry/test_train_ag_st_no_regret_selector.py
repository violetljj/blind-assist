from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ag_st_masked_student import TIER_A_SOURCE  # noqa: E402
from train_ag_st_no_regret_selector import (  # noqa: E402
    NoRegretDepthSelector,
    SelectorObservation,
    calibrate_selector_threshold,
    compute_no_regret_selector_loss,
    split_parent_roles,
)


class NoRegretSelectorTest(unittest.TestCase):
    def test_selector_starts_near_fallback_and_preserves_shape(self) -> None:
        model = NoRegretDepthSelector(feature_channels=48, hidden=32)
        outputs = model(
            torch.randn(1, 48, 4, 5),
            torch.full((1, 1, 16, 20), 2.0),
            torch.full((1, 1, 16, 20), 2.2),
            torch.full((1, 1, 16, 20), 0.25),
            (16, 20),
        )
        self.assertEqual((1, 1, 16, 20), tuple(outputs["selector_probability"].shape))
        self.assertAlmostEqual(
            0.10,
            float(outputs["selector_probability"].mean().detach()),
            places=6,
        )

    def test_global_context_selector_accepts_mean_std_profile(self) -> None:
        model = NoRegretDepthSelector(
            feature_channels=48,
            hidden=32,
            global_context_profile="mean_std",
        )
        outputs = model(
            torch.randn(2, 48, 4, 5),
            torch.full((2, 1, 16, 20), 2.0),
            torch.full((2, 1, 16, 20), 2.2),
            torch.full((2, 1, 16, 20), 0.25),
            (16, 20),
        )
        self.assertEqual((2, 1, 16, 20), tuple(outputs["selector_probability"].shape))
        self.assertIsNotNone(model.global_context)
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 20_000)

    def test_harmful_selection_has_higher_loss_than_fallback(self) -> None:
        truth = torch.ones(1, 1, 2, 2)
        base = truth.clone()
        expert = torch.full_like(truth, 1.5)
        targets = {
            "metric_depth_m": truth,
            "metric_valid": torch.ones_like(truth, dtype=torch.bool),
            "metric_tier": torch.full_like(truth, TIER_A_SOURCE, dtype=torch.uint8),
        }

        def loss(logit: float) -> float:
            logits = torch.full_like(truth, logit)
            outputs = {
                "selector_logits": logits,
                "selector_probability": torch.sigmoid(logits),
            }
            return float(
                compute_no_regret_selector_loss(outputs, base, expert, targets)[
                    "total"
                ]
            )

        self.assertGreater(loss(3.0), loss(-3.0))

    def test_parent_split_is_deterministic_and_disjoint(self) -> None:
        first = split_parent_roles(
            [f"p{index}" for index in range(8)],
            calibration_count=2,
            domain="BONN",
        )
        second = split_parent_roles(
            reversed([f"p{index}" for index in range(8)]),
            calibration_count=2,
            domain="BONN",
        )
        self.assertEqual(first, second)
        self.assertEqual((6, 2), (len(first[0]), len(first[1])))
        self.assertFalse(set(first[0]) & set(first[1]))

    @staticmethod
    def observation(
        parent: str,
        domain: str,
        *,
        expert: np.ndarray,
        probability: np.ndarray,
    ) -> SelectorObservation:
        truth = np.ones((1, 4), dtype=np.float32)
        base = np.asarray([[1.3, 1.3, 1.0, 1.0]], dtype=np.float32)
        return SelectorObservation(
            parent_id=parent,
            domain=domain,
            truth_depth_m=truth,
            valid=np.ones_like(truth, dtype=bool),
            base_depth_m=base,
            expert_depth_m=expert.astype(np.float32),
            selector_probability=probability.astype(np.float32),
        )

    def test_calibration_selects_nontrivial_no_regret_threshold(self) -> None:
        expert = np.asarray([[1.0, 1.0, 1.4, 1.4]], dtype=np.float32)
        probability = np.asarray([[0.9, 0.9, 0.1, 0.1]], dtype=np.float32)
        observations = [
            self.observation("a", "ARKIT", expert=expert, probability=probability),
            self.observation("b", "BONN", expert=expert, probability=probability),
        ]
        calibrated = calibrate_selector_threshold(
            observations,
            candidates=(0.5, 0.95),
            minimum_coverage=0.1,
        )
        self.assertEqual("NONTRIVIAL_NO_REGRET_THRESHOLD_FROZEN", calibrated["decision"])
        self.assertEqual(0.5, calibrated["threshold"])
        self.assertLess(
            calibrated["selected_summary"]["parent_macro"][
                "selected_mae_delta_vs_base_m"
            ],
            0.0,
        )

    def test_calibration_falls_back_when_no_threshold_is_admissible(self) -> None:
        expert = np.full((1, 4), 1.5, dtype=np.float32)
        probability = np.full((1, 4), 0.9, dtype=np.float32)
        calibrated = calibrate_selector_threshold(
            [self.observation("b", "BONN", expert=expert, probability=probability)],
            candidates=(0.5, 0.95),
            minimum_coverage=0.1,
        )
        self.assertEqual(
            "NO_ADMISSIBLE_THRESHOLD_BASE_FALLBACK_FROZEN",
            calibrated["decision"],
        )
        self.assertGreater(calibrated["threshold"], 1.0)
        self.assertEqual(
            0.0,
            calibrated["selected_summary"]["parent_macro"][
                "selected_mae_delta_vs_base_m"
            ],
        )


if __name__ == "__main__":
    unittest.main()
