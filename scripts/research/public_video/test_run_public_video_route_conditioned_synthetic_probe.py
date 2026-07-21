#!/usr/bin/env python3

import unittest

import numpy as np

import run_public_video_route_conditioned_synthetic_probe as subject


class RouteConditionedSyntheticProbeTest(unittest.TestCase):
    def test_ordered_example_ids_preserve_prediction_binding(self) -> None:
        rows = [{"example_id": "e-left"}, {"example_id": "e-straight"}, {"example_id": "e-right"}]
        self.assertEqual(["e-left", "e-straight", "e-right"], subject.ordered_example_ids(rows))

    def test_ordered_example_ids_reject_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            subject.ordered_example_ids([{"example_id": "e"}, {"example_id": "e"}])

    def test_ordered_example_ids_reject_missing_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            subject.ordered_example_ids([{}])

    def test_global_features_change_route_token_but_not_visual_stats(self) -> None:
        scores = np.arange(16, dtype=np.float64).reshape(4, 4)
        left = subject.global_risk_features(scores, "LEFT")
        right = subject.global_risk_features(scores, "RIGHT")
        np.testing.assert_allclose(left[:-3], right[:-3])
        self.assertFalse(np.array_equal(left[-3:], right[-3:]))

    def test_route_conditioned_features_sample_selected_path(self) -> None:
        scores = np.zeros((5, 5), dtype=np.float64)
        scores[4, 0] = 10.0
        left = subject.route_conditioned_risk_features(scores, [[0.0, 1.0], [0.0, 0.75], [0.0, 0.5]])
        right = subject.route_conditioned_risk_features(scores, [[1.0, 1.0], [1.0, 0.75], [1.0, 0.5]])
        self.assertGreater(left.max(), right.max())

    def test_route_conditioned_features_require_three_waypoints(self) -> None:
        with self.assertRaises(ValueError):
            subject.route_conditioned_risk_features(np.zeros((3, 3)), [[0.5, 0.5]])

    def test_exact_intersection_feature_is_monotonic(self) -> None:
        values = np.asarray([[value] for value in (0.0, 1 / 3, 2 / 3, 1.0)])
        self.assertTrue(np.all(np.diff(values[:, 0]) > 0))

    def test_bbox_distance_target_peaks_inside_bbox(self) -> None:
        target = subject.bbox_distance_target([40, 40, 60, 60], image_width=100, image_height=100,
                                              grid_width=10, grid_height=10, sigma_patches=1.5)
        self.assertEqual(1.0, target[4, 4])
        self.assertGreater(target[4, 4], target[0, 0])

    def test_weighted_distance_ridge_fits_two_levels(self) -> None:
        features = np.asarray([[0.0], [0.1], [0.9], [1.0]])
        targets = np.asarray([0.0, 0.0, 1.0, 1.0])
        fitted = subject.fit_weighted_ridge_regression(features, targets, ridge=0.01)
        predictions = features @ fitted["kernel"] + fitted["bias"]
        self.assertGreater(predictions[-1], predictions[0])


if __name__ == "__main__":
    unittest.main()
