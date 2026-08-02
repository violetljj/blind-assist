#!/usr/bin/env python3
"""Tests for the D26 counterfactual collision field canary."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from run_stage_c_d26_thor_magni_counterfactual_collision_field import (
    DIRECTION_NAMES,
    build_gate,
    cumulative_probabilities,
    rotate,
    swap_direction_labels,
)


class D26CounterfactualCollisionFieldTests(unittest.TestCase):
    def test_rotation_and_flip_swap(self) -> None:
        np.testing.assert_allclose(
            rotate(np.asarray((1.0, 0.0)), 90.0),
            (0.0, 1.0),
            atol=1e-12,
        )
        labels = torch.tensor((1, 2, 3))
        self.assertEqual(
            swap_direction_labels(labels, True).tolist(),
            [3, 2, 1],
        )
        self.assertEqual(
            swap_direction_labels(labels, False).tolist(),
            [1, 2, 3],
        )

    def test_cumulative_probabilities_are_monotone(self) -> None:
        probability = np.tile(
            np.asarray((0.1, 0.2, 0.3, 0.1, 0.3)),
            (2, 3, 1),
        )
        cumulative = cumulative_probabilities(probability)
        self.assertEqual(cumulative.shape, (2, 3, 4))
        np.testing.assert_allclose(
            cumulative[0, 0],
            (0.1, 0.3, 0.6, 0.7),
        )

    def test_supported_gate(self) -> None:
        def summary(value: float) -> dict:
            return {
                "mean": value,
                "positive_folds": 5 if value > 0 else 0,
            }

        aggregate = {
            "source_macro_direction_horizon_macro.auroc": summary(
                0.02
            ),
            (
                "source_macro_direction_horizon_macro."
                "average_precision"
            ): summary(0.01),
            "pooled_direction_horizon_macro.auroc": summary(0.01),
            "pooled_direction_horizon_macro.average_precision": summary(
                0.01
            ),
            "safe_choice.source_macro_accuracy": summary(0.03),
        }
        for name in DIRECTION_NAMES:
            aggregate[
                f"by_direction.{name}."
                "source_macro_horizon_macro.auroc"
            ] = summary(0.01)
            aggregate[
                f"by_direction.{name}."
                "source_macro_horizon_macro.average_precision"
            ] = summary(0.01)
        self.assertTrue(build_gate(aggregate, 0)["supported"])


if __name__ == "__main__":
    unittest.main()
