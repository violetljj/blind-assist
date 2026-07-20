#!/usr/bin/env python3
"""Unit tests for the no-event-label corridor anomaly diagnostic."""

from __future__ import annotations

import unittest

import numpy as np

import run_sanpo_corridor_anomaly_probe as subject


class CorridorAnomalyProbeTest(unittest.TestCase):
    def test_even_indices_are_repeatable_and_bounded(self) -> None:
        mask = np.array([[False, True, False], [True, False, True]])
        np.testing.assert_array_equal(np.array([1, 5]), subject.even_indices(mask, count=2))
        self.assertEqual(0, len(subject.even_indices(np.zeros((2, 2), dtype=bool), count=3)))

    def test_pca_reconstruction_separates_off_subspace_vectors(self) -> None:
        train = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        fitted = subject.fit_pca_reconstruction(train, components=1)
        errors = subject.reconstruction_error(np.array([[1.5, 0.0], [1.5, 4.0]]), fitted)
        self.assertLess(errors[0], 1e-12)
        self.assertGreater(errors[1], 7.0)

    def test_auc_is_tie_aware_and_higher_scores_mean_outlier(self) -> None:
        self.assertEqual(1.0, subject.binary_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True])))
        self.assertEqual(0.5, subject.binary_auc(np.ones(4), np.array([False, False, True, True])))
        with self.assertRaisesRegex(ValueError, "both positives"):
            subject.binary_auc(np.array([0.0, 1.0]), np.array([True, True]))


if __name__ == "__main__":
    unittest.main()
