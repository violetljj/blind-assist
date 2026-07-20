#!/usr/bin/env python3
"""Pure tests for frozen-DINO public-video exit retrieval."""

from __future__ import annotations

import unittest

import numpy as np

import retrieve_public_video_exit_windows_with_frozen_dino as subject


class FrozenDinoExitRetrievalTest(unittest.TestCase):
    def test_robust_zscores_preserve_order_and_center(self) -> None:
        scores = subject.robust_zscores([1.0, 2.0, 3.0, 10.0])
        self.assertTrue(np.all(np.diff(scores) > 0))
        self.assertAlmostEqual(0.0, float(np.median(scores)))

    def test_prototype_direction_points_from_clear_to_present(self) -> None:
        direction, positive, negative = subject.prototype_direction(
            np.asarray([[3.0, 1.0], [5.0, 1.0]]),
            np.asarray([[1.0, 1.0], [1.0, 1.0]]),
        )
        self.assertTrue(np.allclose([1.0, 0.0], direction))
        self.assertGreater(float(positive @ direction), float(negative @ direction))

    def test_sustained_drop_ranks_true_exit_first(self) -> None:
        timestamps = [index * 1000 for index in range(10)]
        scores = [0.0, 0.0, 4.0, 4.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0]
        rows = subject.rank_exit_transitions(
            timestamps,
            scores,
            sample_interval_ms=1000,
            prior_samples=2,
            future_samples=3,
            top_k=2,
            minimum_separation_ms=2000,
        )
        self.assertEqual(4000, rows[0]["clear_timestamp_ms"])
        self.assertGreater(rows[0]["sustained_drop_z"], 1.0)

    def test_gap_breaks_transition_context(self) -> None:
        rows = subject.rank_exit_transitions(
            [0, 1000, 2000, 5000, 6000, 7000],
            [3.0, 3.0, 3.0, -2.0, -2.0, -2.0],
            sample_interval_ms=1000,
            prior_samples=2,
            future_samples=3,
            top_k=2,
            minimum_separation_ms=1000,
        )
        self.assertEqual([], rows)


if __name__ == "__main__":
    unittest.main()
