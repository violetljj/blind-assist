#!/usr/bin/env python3
"""Tests for the THOR-MAGNI RGB-history separability screen."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    binary_metrics,
    fit_binary,
    history_design,
)


class ThorMagniRgbHistoryScreenTests(unittest.TestCase):
    def test_history_design_has_fixed_residual_blocks(self) -> None:
        features = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
        result = history_design(features)
        self.assertEqual(result.shape, (2, 12))
        np.testing.assert_array_equal(result[:, :3], features[:, -1])
        np.testing.assert_array_equal(
            result[:, 3:6],
            features[:, -1] - features[:, 0],
        )
        np.testing.assert_array_equal(
            result[:, 6:9],
            features[:, -1] - features[:, -2],
        )

    def test_binary_metrics_refuses_single_class_claim(self) -> None:
        result = binary_metrics(
            np.ones(4, dtype=np.int64),
            np.full(4, 0.5),
        )
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["average_precision"])

    def test_binary_metrics_are_tie_aware(self) -> None:
        perfect = binary_metrics(
            np.asarray([0, 1]),
            np.asarray([0.0, 1.0]),
        )
        self.assertEqual(perfect["auroc"], 1.0)
        self.assertEqual(perfect["average_precision"], 1.0)
        tied = binary_metrics(
            np.asarray([0, 1]),
            np.asarray([0.5, 0.5]),
        )
        self.assertEqual(tied["auroc"], 0.5)
        self.assertEqual(tied["average_precision"], 0.5)

    def test_constant_train_label_returns_constant_probability(self) -> None:
        probability = fit_binary(
            np.zeros((3, 2)),
            np.ones(3, dtype=np.int64),
            np.zeros((2, 2)),
        )
        np.testing.assert_array_equal(probability, np.ones(2))


if __name__ == "__main__":
    unittest.main()
