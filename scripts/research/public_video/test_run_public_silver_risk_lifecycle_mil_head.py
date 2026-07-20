#!/usr/bin/env python3
"""Pure tests for the weakly supervised risk/lifecycle MIL head."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import run_public_silver_risk_lifecycle_mil_head as subject


def detection(group: str, *, area: float, bottom: float, overlap: float, threat: float) -> dict[str, object]:
    return {
        "group": group,
        "confidence": 0.8,
        "area": area,
        "bottom": bottom,
        "corridor_overlap": overlap,
        "threat": threat,
    }


class PublicSilverRiskLifecycleMilHeadTest(unittest.TestCase):
    def test_frame_profile_and_causal_delta_preserve_growth(self) -> None:
        frames = [
            [detection("person", area=0.02, bottom=0.4, overlap=0.2, threat=0.1)],
            [detection("person", area=0.08, bottom=0.7, overlap=0.8, threat=0.6)],
        ]
        sequence = subject.causal_profile_sequence(frames)
        self.assertEqual((2, 64), sequence.shape)
        self.assertGreater(sequence[1, 32 + 2], 0.0)
        self.assertGreater(sequence[1, 32 + 5], 0.0)

    def test_auxiliary_frame_channel_is_aligned_and_differenced(self) -> None:
        frames = [
            [detection("person", area=0.02, bottom=0.4, overlap=0.2, threat=0.1)],
            [detection("person", area=0.08, bottom=0.7, overlap=0.8, threat=0.6)],
        ]
        auxiliary = np.asarray([[0.1, 1.0], [0.4, 1.0]])
        sequence = subject.causal_profile_sequence(frames, auxiliary)
        self.assertEqual((2, 68), sequence.shape)
        self.assertAlmostEqual(0.1, sequence[0, 32])
        self.assertAlmostEqual(0.3, sequence[1, 34 + 32])

    def test_misaligned_auxiliary_frame_channel_fails_closed(self) -> None:
        frames = [
            [detection("person", area=0.02, bottom=0.4, overlap=0.2, threat=0.1)],
            [detection("person", area=0.08, bottom=0.7, overlap=0.8, threat=0.6)],
        ]
        with self.assertRaisesRegex(ValueError, "aligned"):
            subject.causal_profile_sequence(frames, np.asarray([[0.1, 1.0]]))

    def test_object_occupancy_baseline_is_source_relative(self) -> None:
        frames = [
            [detection("person", area=0.02, bottom=0.4, overlap=0.5, threat=0.1)],
            [detection("person", area=0.08, bottom=0.7, overlap=0.75, threat=0.6)],
        ]
        feature = subject.object_occupancy_baseline_feature(frames)
        self.assertEqual((2, 1), feature.shape)
        self.assertAlmostEqual(0.0, feature[0, 0])
        self.assertAlmostEqual(0.05, feature[1, 0])

    def test_corridor_appearance_channel_has_optional_first_frame_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths: list[str] = []
            for index, color in enumerate(((20, 80, 160), (220, 220, 220))):
                path = Path(directory) / f"{index}.png"
                cv2.imwrite(str(path), np.full((48, 64, 3), color, dtype=np.uint8))
                paths.append(str(path))
            plain = subject.corridor_appearance_frame_features(paths, size=320)
            relative = subject.corridor_appearance_frame_features(
                paths,
                size=320,
                include_baseline_delta=True,
            )
            self.assertEqual((2, 16), plain.shape)
            self.assertEqual((2, 32), relative.shape)
            self.assertTrue(np.allclose(relative[0, 16:], 0.0))
            self.assertGreater(float(np.linalg.norm(relative[1, 16:])), 0.0)

    def test_smooth_max_attention_and_gradient_are_consistent(self) -> None:
        scores = np.asarray([-1.0, 0.0, 2.0])
        pooled, attention = subject.smooth_max(scores, temperature=0.4)
        self.assertGreater(pooled, 1.0)
        self.assertAlmostEqual(1.0, float(attention.sum()))
        epsilon = 1e-6
        perturbed = scores.copy()
        perturbed[2] += epsilon
        changed, _ = subject.smooth_max(perturbed, temperature=0.4)
        self.assertAlmostEqual(attention[2], (changed - pooled) / epsilon, places=5)

    def test_terminal_pooling_uses_only_current_causal_state(self) -> None:
        scores = np.asarray([3.0, 2.0, -1.0])
        pooled, attention = subject.aggregate_sequence_scores(
            scores,
            temperature=0.35,
            pooling="terminal",
        )
        self.assertAlmostEqual(-1.0, pooled)
        self.assertTrue(np.array_equal(np.asarray([0.0, 0.0, 1.0]), attention))

    def test_terminal_pooling_gradient_ignores_stale_peak(self) -> None:
        sequence = np.asarray([[10.0], [5.0], [-1.0]])
        _loss, gradient, _bias = subject.loss_and_gradients(
            [sequence, np.asarray([[0.0], [0.0], [1.0]])],
            np.asarray([0, 1]),
            np.asarray([1.0]),
            0.0,
            temperature=0.35,
            weight_decay=0.0,
            episode_pooling="terminal",
        )
        self.assertLess(gradient[0], 0.0)

    def test_optimizer_reduces_episode_loss(self) -> None:
        sequences = [
            np.asarray([[-2.0], [-1.5]]),
            np.asarray([[-1.0], [-0.8]]),
            np.asarray([[0.5], [1.0]]),
            np.asarray([[1.0], [2.0]]),
        ]
        labels = np.asarray([0, 0, 1, 1])
        weights, bias = subject.prototype_initialization(sequences, labels)
        fitted = subject.fit_head(
            sequences,
            labels,
            weights,
            bias,
            steps=80,
            learning_rate=0.03,
            weight_decay=0.01,
            temperature=0.35,
        )
        self.assertLess(fitted["loss_first_last"][-1], fitted["loss_first_last"][0])

    def test_confidence_weighting_changes_only_episode_loss_mass(self) -> None:
        sequences = [
            np.asarray([[-1.0], [-0.5]]),
            np.asarray([[-0.2], [0.1]]),
            np.asarray([[0.3], [0.8]]),
            np.asarray([[1.0], [1.5]]),
        ]
        labels = np.asarray([0, 0, 1, 1])
        weights = np.asarray([0.4])
        equal = subject.loss_and_gradients(
            sequences,
            labels,
            weights,
            0.0,
            temperature=0.35,
            weight_decay=0.0,
        )
        weighted = subject.loss_and_gradients(
            sequences,
            labels,
            weights,
            0.0,
            temperature=0.35,
            weight_decay=0.0,
            confidence_weights=np.asarray([0.9, 0.2, 0.9, 0.2]),
        )
        self.assertNotAlmostEqual(equal[0], weighted[0])
        self.assertFalse(np.allclose(equal[1], weighted[1]))

    def test_invalid_confidence_weights_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            subject.loss_and_gradients(
                [np.asarray([[-1.0], [-0.5]]), np.asarray([[0.5], [1.0]])],
                np.asarray([0, 1]),
                np.asarray([0.1]),
                0.0,
                temperature=0.35,
                weight_decay=0.0,
                confidence_weights=np.asarray([0.8, 0.0]),
            )

    def test_pairwise_ranking_gradient_increases_positive_gap(self) -> None:
        negative = np.asarray([[-1.0], [-0.5]])
        positive = np.asarray([[0.2], [0.4]])
        weights = np.asarray([0.0])
        loss, gradient, bias_gradient = subject.loss_and_gradients(
            [negative, positive],
            np.asarray([0, 1]),
            weights,
            0.0,
            temperature=0.35,
            weight_decay=0.0,
            ranking_pairs=[(negative, positive)],
            pairwise_ranking_weight=0.25,
            pairwise_margin=0.25,
        )
        self.assertGreater(loss, 0.0)
        self.assertLess(gradient[0], 0.0)
        self.assertAlmostEqual(0.0, bias_gradient)

    def test_zero_pairwise_weight_preserves_equal_loss(self) -> None:
        sequences = [np.asarray([[-1.0]]), np.asarray([[1.0]])]
        labels = np.asarray([0, 1])
        baseline = subject.loss_and_gradients(
            sequences, labels, np.asarray([0.2]), 0.0,
            temperature=0.35, weight_decay=0.01,
        )
        disabled = subject.loss_and_gradients(
            sequences, labels, np.asarray([0.2]), 0.0,
            temperature=0.35, weight_decay=0.01,
            ranking_pairs=[(sequences[0], sequences[1])],
            pairwise_ranking_weight=0.0,
            pairwise_margin=0.25,
        )
        self.assertAlmostEqual(baseline[0], disabled[0])
        self.assertTrue(np.array_equal(baseline[1], disabled[1]))
        self.assertAlmostEqual(baseline[2], disabled[2])

    def test_lifecycle_decoder_is_explicitly_curve_derived(self) -> None:
        self.assertEqual(
            ["approach", "alertable", "alertable", "post_event"],
            subject.decode_lifecycle([0.2, 0.6, 0.8, 0.3]),
        )
        self.assertEqual(["non_alert", "non_alert"], subject.decode_lifecycle([0.2, 0.4]))

    def test_augmentation_alignment_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            subject.evaluate_seed(
                [np.asarray([[-1.0], [-0.5]]), np.asarray([[0.5], [1.0]])],
                np.asarray([0, 1]),
                ["negative", "positive"],
                ["source_negative", "source_positive"],
                [None, None],
                np.ones(2),
                seed=7,
                steps=2,
                learning_rate=0.01,
                weight_decay=0.01,
                temperature=0.35,
                pairwise_ranking_weight=0.0,
                pairwise_margin=0.25,
                minimum_pair_confidence=0.65,
                augmentation_sequences=[np.asarray([[-1.0]])],
                augmentation_labels=np.asarray([0]),
            )

    def test_parent_matched_augmentation_is_excluded_from_each_fold(self) -> None:
        result = subject.evaluate_seed(
            [
                np.asarray([[-2.0], [-1.5]]),
                np.asarray([[1.0], [1.5]]),
                np.asarray([[-1.7], [-1.2]]),
                np.asarray([[0.8], [1.3]]),
            ],
            np.asarray([0, 1, 0, 1]),
            ["a-negative", "a-positive", "b-negative", "b-positive"],
            ["source-a", "source-a", "source-b", "source-b"],
            ["a-pair", "a-pair", "b-pair", "b-pair"],
            np.ones(4),
            seed=7,
            steps=5,
            learning_rate=0.01,
            weight_decay=0.01,
            temperature=0.35,
            pairwise_ranking_weight=0.1,
            pairwise_margin=0.25,
            minimum_pair_confidence=0.65,
            augmentation_sequences=[
                np.asarray([[-1.0], [-0.5]]),
                np.asarray([[0.5], [1.0]]),
                np.asarray([[-0.8], [-0.3]]),
                np.asarray([[0.7], [1.2]]),
            ],
            augmentation_labels=np.asarray([0, 1, 0, 1]),
            augmentation_source_ids=["synthetic-a", "synthetic-a", "synthetic-b", "synthetic-b"],
            augmentation_pair_ids=["synthetic-a-pair", "synthetic-a-pair", "synthetic-b-pair", "synthetic-b-pair"],
            augmentation_confidence_weights=np.ones(4),
            augmentation_parent_source_ids=["source-a", "source-a", "source-b", "source-b"],
        )
        by_source = {fold["held_out_source_id"]: fold for fold in result["folds"]}
        self.assertEqual(2, by_source["source-a"]["train_only_augmentation_episode_count"])
        self.assertEqual(2, by_source["source-a"]["parent_matched_augmentation_excluded_count"])
        self.assertEqual(["synthetic-b"], by_source["source-a"]["train_only_augmentation_source_ids"])
        self.assertEqual(["synthetic-a"], by_source["source-b"]["train_only_augmentation_source_ids"])


if __name__ == "__main__":
    unittest.main()
