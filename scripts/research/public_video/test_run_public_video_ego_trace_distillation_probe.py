import unittest

import numpy as np

import run_public_video_ego_trace_distillation_probe as subject


class EgoTraceDistillationProbeTest(unittest.TestCase):
    def test_point_box_distance_is_zero_inside_and_positive_outside(self):
        detections = [{"features": {"center_x_norm": 0.5, "bottom_y_norm": 0.9,
                                     "width_norm": 0.1, "height_norm": 0.2}}]
        self.assertEqual(0.0, subject.point_box_distance((0.5, 0.8), detections))
        self.assertGreater(subject.point_box_distance((0.0, 0.0), detections), 0.0)

    def test_binary_auroc_counts_ties(self):
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([1.0, 0.5, 0.5, 0.0])
        self.assertAlmostEqual(0.875, subject.binary_auroc(labels, scores))

    def test_weighted_ridge_returns_one_prediction_per_test_row(self):
        train_x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        train_y = np.asarray([0.0, 0.0, 1.0, 1.0])
        predicted = subject.weighted_ridge_predict(train_x, train_y, ["a", "a", "b", "b"], np.asarray([[1.5], [2.5]]), 0.01)
        self.assertEqual((2,), predicted.shape)


if __name__ == "__main__":
    unittest.main()
