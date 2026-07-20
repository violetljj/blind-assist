#!/usr/bin/env python3
"""Pure tests for prototype/bootstrap short head runs."""

from __future__ import annotations

import unittest

import numpy as np

import run_public_silver_prototype_bootstrap_short_runs as short


class PublicSilverPrototypeBootstrapShortRunsTest(unittest.TestCase):
    def test_prototype_initialization_separates_simple_classes(self) -> None:
        features = np.asarray([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        labels = np.asarray([0, 0, 1, 1])
        weights, bias = short.prototype_head(features, labels)
        predictions = np.argmax(features @ weights + bias, axis=1)
        np.testing.assert_array_equal(labels, predictions)

    def test_bootstrap_operates_on_sources_and_is_deterministic(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1])
        sources = ["n0", "n0", "n1", "p0", "p1"]
        first = short.bootstrap_source_indices(labels, sources, seed=7)
        second = short.bootstrap_source_indices(labels, sources, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual({0, 1}, set(labels[first].tolist()))
        for source in set(np.asarray(sources, dtype=object)[first]):
            selected = [index for index in first if sources[index] == source]
            original = [index for index, value in enumerate(sources) if value == source]
            self.assertEqual(0, len(selected) % len(original))

    def test_short_optimizer_is_deterministic_and_reduces_loss(self) -> None:
        features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.asarray([0, 0, 1, 1])
        weights, bias = short.prototype_head(features, labels)
        first = short.fit_short_head(features, labels, weights, bias, steps=80, learning_rate=0.03, weight_decay=0.01)
        second = short.fit_short_head(features, labels, weights, bias, steps=80, learning_rate=0.03, weight_decay=0.01)
        self.assertEqual(first["coefficient_sha256"], second["coefficient_sha256"])
        self.assertLess(first["loss_first_last"][-1], first["loss_first_last"][0])

    def test_seed_evaluation_keeps_source_pair_together(self) -> None:
        features = np.asarray([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        episodes = ["n0", "n1", "n2", "p0", "p1", "p2"]
        sources = ["shared", "n1", "n2", "shared", "p1", "p2"]
        result = short.evaluate_seed(features, labels, episodes, sources, seed=11, steps=80, learning_rate=0.03, weight_decay=0.01)
        shared = result["folds"][0]
        self.assertEqual("shared", shared["held_out_source_id"])
        self.assertEqual(["n0", "p0"], shared["held_out_episode_ids"])


if __name__ == "__main__":
    unittest.main()
