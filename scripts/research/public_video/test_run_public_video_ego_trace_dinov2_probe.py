import unittest

import numpy as np

import run_public_video_ego_trace_dinov2_probe as subject


class EgoTraceDinoProbeTest(unittest.TestCase):
    def test_dual_ridge_returns_finite_predictions(self):
        train_x = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
        train_y = np.asarray([0.0, 0.0, 1.0, 1.0])
        predicted = subject.weighted_dual_ridge_predict(
            train_x, train_y, ["a", "a", "b", "b"], np.asarray([[1.5, 0.5], [2.5, 0.5]]), 0.01
        )
        self.assertEqual((2,), predicted.shape)
        self.assertTrue(np.isfinite(predicted).all())

    def test_source_weighting_changes_duplicate_invariance_only_within_source(self):
        x = np.asarray([[0.0], [0.0], [2.0], [2.0]])
        y = np.asarray([0.0, 0.0, 1.0, 1.0])
        one = subject.weighted_dual_ridge_predict(x, y, ["a", "a", "b", "b"], np.asarray([[1.0]]), 0.1)
        self.assertTrue(np.isfinite(one).all())


if __name__ == "__main__":
    unittest.main()
