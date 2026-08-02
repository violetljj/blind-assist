#!/usr/bin/env python3
"""Tests for the D25 ordinal time-to-entry canary."""

from __future__ import annotations

import unittest

import numpy as np

from run_stage_c_d25_thor_magni_time_to_entry import (
    HORIZON_NAMES,
    build_gate,
    cumulative_probabilities,
    entry_bin,
)


class D25TimeToEntryTests(unittest.TestCase):
    def test_entry_bins_are_left_closed_by_horizon(self) -> None:
        self.assertEqual(entry_bin(0.1), 0)
        self.assertEqual(entry_bin(0.5), 0)
        self.assertEqual(entry_bin(0.6), 1)
        self.assertEqual(entry_bin(1.5), 2)
        self.assertEqual(entry_bin(2.0), 3)
        self.assertEqual(entry_bin(None), 4)

    def test_cumulative_probabilities_are_monotone(self) -> None:
        probability = np.asarray([[0.1, 0.2, 0.3, 0.1, 0.3]])
        cumulative = cumulative_probabilities(probability)
        np.testing.assert_allclose(
            cumulative,
            [[0.1, 0.3, 0.6, 0.7]],
        )

    def test_supported_gate(self) -> None:
        def summary(value: float) -> dict:
            return {
                "mean": value,
                "positive_folds": 5 if value > 0 else 0,
            }

        aggregate = {
            "source_macro_horizon_macro.auroc": summary(0.02),
            "source_macro_horizon_macro.average_precision": summary(
                0.01
            ),
            "pooled_horizon_macro.auroc": summary(0.01),
            "pooled_horizon_macro.average_precision": summary(0.01),
        }
        for name in HORIZON_NAMES:
            aggregate[
                f"by_horizon.{name}.source_macro.auroc"
            ] = summary(0.01)
            aggregate[
                f"by_horizon.{name}.source_macro.average_precision"
            ] = summary(0.01)
        self.assertTrue(build_gate(aggregate, 0)["supported"])


if __name__ == "__main__":
    unittest.main()
