import unittest

import numpy as np

import run_public_video_marker_relation_pair_bootstrap_short_runs as subject


class MarkerRelationPairBootstrapShortRunsTest(unittest.TestCase):
    def test_bootstrap_keeps_complete_pair_sources(self) -> None:
        sources = np.asarray(["a", "a", "b", "c", "c"])
        indices, weights, draws = subject.bootstrap_pair_rows(sources, 9)
        self.assertEqual(3, len(draws))
        self.assertAlmostEqual(1.0, float(weights.sum()))
        self.assertEqual(len(indices), len(weights))

    def test_pair_head_preserves_simple_order(self) -> None:
        deltas = np.asarray([[1.0, 0.0], [2.0, 0.1], [1.5, -0.1]])
        sources = np.asarray(["a", "b", "c"])
        optimizer = {"steps": 80, "learning_rate": 0.03, "weight_decay": 0.01, "pair_margin": 1.0}
        model, prototype, audit = subject.fit_pair_head(deltas, sources, 5, optimizer)
        self.assertTrue(np.all(subject.projection(model, deltas) > 0.0))
        self.assertTrue(np.all(subject.projection(prototype, deltas) > 0.0))
        self.assertLess(audit["loss_first_last"][-1], audit["loss_first_last"][0])


if __name__ == "__main__":
    unittest.main()
