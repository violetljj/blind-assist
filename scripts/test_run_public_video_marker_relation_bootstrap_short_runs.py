import unittest

import numpy as np

import run_public_video_marker_relation_bootstrap_short_runs as subject


class MarkerRelationBootstrapShortRunsTest(unittest.TestCase):
    def test_prototype_orders_separable_classes(self) -> None:
        x = np.asarray([[0.0], [1.0], [3.0], [4.0]])
        active = np.asarray([False, False, True, True])
        weight, bias = subject.prototype_initialization(x, active, np.full(4, 0.25))
        self.assertGreater(subject.sigmoid(x[-1:] @ weight + bias)[0], subject.sigmoid(x[:1] @ weight + bias)[0])

    def test_bootstrap_keeps_complete_source_groups(self) -> None:
        sources = np.asarray(["a", "a", "b", "c", "c"])
        active = np.asarray([False, True, False, True, True])
        indices, weights, draws = subject.bootstrap_source_class_rows(sources, active, 7)
        self.assertEqual(0.5, float(weights[:sum(~active[indices])].sum()))
        self.assertAlmostEqual(1.0, float(weights.sum()))
        self.assertEqual({"inactive", "active"}, set(draws))
        self.assertEqual(len(indices), len(weights))

    def test_soft_target_head_reduces_loss_and_orders(self) -> None:
        x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.asarray([0.0, 0.0, 1.0, 1.0])
        model = subject.fit_soft_target_head(x, y, np.full(4, 0.25), np.asarray([0.1]), 0.0,
                                             steps=80, learning_rate=0.03, weight_decay=0.01)
        self.assertLess(model["loss_first_last"][-1], model["loss_first_last"][0])
        self.assertGreater(subject.predict(model, x)[-1], subject.predict(model, x)[0])

    def test_source_macro_metrics_do_not_weight_long_source_more(self) -> None:
        active = np.asarray([True, False, False, True, False])
        scores = np.asarray([0.9, 0.1, 0.1, 0.2, 0.1])
        sources = np.asarray(["a", "a", "a", "b", "b"])
        metrics = subject.source_macro_metrics(active, scores, sources)
        self.assertEqual(0.5, metrics["source_macro_positive_recall"])
        self.assertEqual(1.0, metrics["source_macro_negative_recall"])


if __name__ == "__main__":
    unittest.main()
