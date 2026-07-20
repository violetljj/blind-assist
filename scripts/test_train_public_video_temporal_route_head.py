import unittest

import numpy as np

import train_public_video_temporal_route_head as subject


class TemporalRouteHeadTrainingTest(unittest.TestCase):
    def test_source_balancing_equalizes_total_weight(self):
        sources = np.asarray(["a", "a", "a", "b"])
        weights = subject.source_balanced_weights(sources)
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3:].sum()))

    def test_efficient_auroc_handles_ties(self):
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([1.0, 0.5, 0.5, 0.0])
        self.assertAlmostEqual(0.875, subject.efficient_binary_auroc(labels, scores))

    def test_model_stays_under_frozen_parameter_limit(self):
        model = subject.TemporalRouteHead(43)
        count = sum(parameter.numel() for parameter in model.parameters())
        self.assertLessEqual(count, 70000)


if __name__ == "__main__":
    unittest.main()
