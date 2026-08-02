#!/usr/bin/env python3
"""Tests for the D28 kinematic field distillation canary."""

from __future__ import annotations

import unittest

import torch

from run_stage_c_d28_thor_magni_kinematic_field_distillation import (
    DIRECTION_NAMES,
    build_gate,
)


class D28KinematicFieldDistillationTests(unittest.TestCase):
    def test_cumulative_max_is_monotone(self) -> None:
        bounded = -torch.sigmoid(
            torch.tensor([[[0.0, 2.0, -2.0, 1.0]]])
        )
        prediction = torch.cummax(bounded, dim=2).values
        self.assertTrue(torch.all(torch.diff(prediction, dim=2) >= 0))

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
            "teacher_fit.source_macro_mae_m": summary(0.10),
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
