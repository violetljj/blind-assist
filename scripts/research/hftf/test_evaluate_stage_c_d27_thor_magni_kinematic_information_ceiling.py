#!/usr/bin/env python3
"""Tests for the D27 kinematic information-ceiling oracle."""

from __future__ import annotations

import unittest

from evaluate_stage_c_d27_thor_magni_kinematic_information_ceiling import (
    DIRECTION_NAMES,
    build_gate,
)


class D27KinematicInformationCeilingTests(unittest.TestCase):
    def test_supported_gate(self) -> None:
        def summary(value: float) -> dict:
            return {
                "mean": value,
                "positive_folds": 5 if value > 0 else 0,
            }

        aggregate = {
            "source_macro_direction_horizon_macro.auroc": summary(
                0.03
            ),
            (
                "source_macro_direction_horizon_macro."
                "average_precision"
            ): summary(0.02),
            "pooled_direction_horizon_macro.auroc": summary(0.01),
            "pooled_direction_horizon_macro.average_precision": summary(
                0.01
            ),
            "safe_choice.source_macro_accuracy": summary(0.06),
        }
        for name in DIRECTION_NAMES:
            aggregate[
                f"by_direction.{name}."
                "source_macro_horizon_macro.auroc"
            ] = summary(0.02)
            aggregate[
                f"by_direction.{name}."
                "source_macro_horizon_macro.average_precision"
            ] = summary(0.02)
        self.assertTrue(build_gate(aggregate, 0)["supported"])


if __name__ == "__main__":
    unittest.main()
