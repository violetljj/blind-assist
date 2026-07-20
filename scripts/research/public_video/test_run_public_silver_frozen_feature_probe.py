#!/usr/bin/env python3
"""Pure unit tests for the public-silver frozen-feature probe."""

from __future__ import annotations

import unittest

import numpy as np

import run_public_silver_frozen_feature_probe as probe
import sanpo_deterministic_linear_probe as ridge_probe


class PublicSilverFrozenFeatureProbeTest(unittest.TestCase):
    def test_pooling_is_deterministic_and_keeps_temporal_delta(self) -> None:
        first = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        second = first + 2
        frame = probe.pool_frame_map(first)
        episode = probe.pool_episode(np.stack([frame, probe.pool_frame_map(second)]))
        self.assertEqual(12, len(frame))
        self.assertEqual(36, len(episode))
        np.testing.assert_array_equal(episode[-12:], np.full(12, 2.0))

    def test_corridor_relative_pooling_separates_walkable_and_nonwalkable_weights(self) -> None:
        features = np.zeros((2, 4, 1), dtype=np.float64)
        features[1, 1, 0] = 2.0
        features[1, 2, 0] = 8.0
        logits = np.zeros((2, 4, 2), dtype=np.float64)
        logits[1, 1] = [8.0, 0.0]
        logits[1, 2] = [0.0, 8.0]
        pooled = probe.pool_corridor_relative_frame(features, logits)
        self.assertEqual(6, len(pooled))
        self.assertLess(pooled[0], pooled[1])
        self.assertAlmostEqual(pooled[1] - pooled[0], pooled[2])
        self.assertGreater(pooled[-1], 0.99)

    def test_residual_motion_removes_uniform_translation_and_keeps_local_expansion(self) -> None:
        uniform = np.ones((8, 8, 2), dtype=np.float64) * np.asarray([3.0, -2.0])
        self.assertAlmostEqual(0.0, probe.residual_motion_descriptor(uniform)[0])
        expanding = uniform.copy()
        expanding[4:, 2:6, 0] += np.tile(np.asarray([-2.0, -1.0, 1.0, 2.0]), (4, 1))
        descriptor = probe.residual_motion_descriptor(expanding)
        self.assertGreater(descriptor[0], 0.0)
        self.assertGreater(descriptor[5], 0.0)

    def test_leave_one_episode_out_has_no_training_self_leakage(self) -> None:
        features = np.asarray([[-3.0], [-2.0], [1.0], [2.0], [3.0]], dtype=np.float64)
        labels = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
        episode_ids = [f"e{index}" for index in range(5)]
        result = probe.leave_one_episode_out(features, labels, episode_ids, ridge=1.0, class_balanced=True)
        self.assertEqual(5, len(result["folds"]))
        self.assertEqual(episode_ids, [fold["held_out_source_id"] for fold in result["folds"]])
        self.assertEqual([[item] for item in episode_ids], [fold["held_out_episode_ids"] for fold in result["folds"]])
        self.assertGreaterEqual(result["metrics"]["balanced_accuracy"], 0.5)

    def test_source_group_holdout_keeps_two_episodes_together(self) -> None:
        features = np.asarray([[-3.0], [-2.0], [-1.5], [1.0], [2.0], [3.0]], dtype=np.float64)
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        episode_ids = ["n0", "n1", "n2", "p0", "p1", "p2"]
        source_ids = ["shared-negative", "shared-negative", "other-negative", "p0", "p1", "p2"]
        result = probe.leave_one_source_group_out(features, labels, episode_ids, source_ids, ridge=1.0)
        first = result["folds"][0]
        self.assertEqual("shared-negative", first["held_out_source_id"])
        self.assertEqual(["n0", "n1"], first["held_out_episode_ids"])
        self.assertEqual(5, len(result["folds"]))

    def test_balanced_episode_ridge_uses_small_dual_system(self) -> None:
        features = np.asarray([[-2.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        labels = np.asarray([0, 0, 1, 1, 1])
        fitted = probe.fit_episode_ridge(features, labels, ridge=1.0, class_balanced=True)
        predictions = ridge_probe.predict_labels(features, fitted["kernel"], fitted["bias"])
        np.testing.assert_array_equal(labels, predictions)

    def test_binary_metrics_reports_each_class(self) -> None:
        metrics = probe.binary_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 1, 1]))
        self.assertEqual([[1, 1], [0, 2]], metrics["confusion_matrix_rows_truth_columns_prediction"])
        self.assertEqual(0.5, metrics["candidate_no_alert_recall"])
        self.assertEqual(1.0, metrics["candidate_alert_recall"])
        self.assertEqual(0.75, metrics["balanced_accuracy"])

    def test_counterfactual_delta_alignment_detects_shared_direction(self) -> None:
        episodes = [
            {"label": 0, "counterfactual_pair_id": "a"},
            {"label": 1, "counterfactual_pair_id": "a"},
            {"label": 0, "counterfactual_pair_id": "b"},
            {"label": 1, "counterfactual_pair_id": "b"},
        ]
        aligned = probe.counterfactual_delta_alignment(np.asarray([[0.0], [2.0], [1.0], [4.0]]), episodes)
        opposed = probe.counterfactual_delta_alignment(np.asarray([[0.0], [2.0], [4.0], [1.0]]), episodes)
        self.assertTrue(aligned["passed"])
        self.assertEqual(1.0, aligned["mean_pairwise_cosine"])
        self.assertFalse(opposed["passed"])
        self.assertEqual(-1.0, opposed["mean_pairwise_cosine"])


if __name__ == "__main__":
    unittest.main()
