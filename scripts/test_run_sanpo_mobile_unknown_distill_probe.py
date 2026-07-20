#!/usr/bin/env python3
"""Tests for the frozen-feature unknown-distillation feasibility math."""

from __future__ import annotations

import unittest

import numpy as np

import run_sanpo_mobile_unknown_distill_probe as subject


class MobileUnknownDistillProbeTest(unittest.TestCase):
    def test_even_sampling_and_spearman_direction(self) -> None:
        np.testing.assert_array_equal(np.array([0, 4, 9]), subject.evenly_spaced_rows(10, count=3))
        self.assertAlmostEqual(1.0, subject.spearman_correlation(np.array([1., 2., 3.]), np.array([2., 4., 6.])))
        self.assertAlmostEqual(-1.0, subject.spearman_correlation(np.array([1., 2., 3.]), np.array([6., 4., 2.])))

    def test_closed_form_ridge_reconstructs_linear_target(self) -> None:
        features = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.], [4., 5.]])
        target = 2.0 * features[:, 0] - features[:, 1]
        fitted = subject.fit_ridge_regression(features, target, ridge=1e-8)
        prediction = subject.predict_ridge(features, fitted)
        self.assertLess(float(np.max(np.abs(prediction - target))), 1e-4)


if __name__ == "__main__":
    unittest.main()
