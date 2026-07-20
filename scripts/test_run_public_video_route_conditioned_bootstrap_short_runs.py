#!/usr/bin/env python3

import unittest

import numpy as np

import run_public_video_route_conditioned_bootstrap_short_runs as subject


class RouteConditionedBootstrapShortRunsTest(unittest.TestCase):
    def test_bootstrap_preserves_every_source_class_cell(self) -> None:
        indices = np.arange(8)
        sources = np.asarray(["a"] * 4 + ["b"] * 4, dtype=object)
        labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        sampled = subject.stratified_source_class_bootstrap(indices, sources, labels, np.random.default_rng(1))
        self.assertEqual(8, len(sampled))
        self.assertEqual({("a", 0), ("a", 1), ("b", 0), ("b", 1)},
                         {(sources[index], int(labels[index])) for index in sampled})

    def test_prototype_short_head_learns_separable_features(self) -> None:
        features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.asarray([0, 0, 1, 1])
        head = subject.fit_prototype_short_head(features, labels, steps=20, learning_rate=0.05, l2=0.001)
        np.testing.assert_array_equal(labels, subject.predict_short_head(features, head))

    def test_head_hash_is_deterministic(self) -> None:
        features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.asarray([0, 0, 1, 1])
        first = subject.fit_prototype_short_head(features, labels, steps=5, learning_rate=0.05, l2=0.001)
        second = subject.fit_prototype_short_head(features, labels, steps=5, learning_rate=0.05, l2=0.001)
        self.assertEqual(subject.head_sha256(first), subject.head_sha256(second))


if __name__ == "__main__":
    unittest.main()
