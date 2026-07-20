import unittest

import numpy as np

import run_public_video_event_route_role_linear_probe as subject


class EventRouteRoleLinearProbeTest(unittest.TestCase):
    def test_marker_vectors_are_132_dimensional(self) -> None:
        grids = np.ones((2, 43, 16, 16), dtype=np.float64)
        masks = np.zeros((2, 16, 16), dtype=bool)
        masks[:, 8, 8] = True
        self.assertEqual((2, 132), subject.marker_vectors(grids, masks).shape)

    def test_empty_mask_uses_frozen_detection_fallback(self) -> None:
        grids = np.ones((1, 43, 16, 16), dtype=np.float64)
        masks = np.zeros((1, 16, 16), dtype=bool)
        detections = {("s", 1000): [{"features": {
            "center_x_norm": .5, "bottom_y_norm": .5, "height_norm": .01,
            "width_norm": .01}}]}
        values, eligible, count = subject.marker_vectors_with_fallback(
            grids, masks, np.asarray(["s"]), np.asarray([1000]), detections, .5)
        self.assertEqual((1, 132), values.shape)
        np.testing.assert_array_equal([0], eligible)
        self.assertEqual(1, count)

    def test_marker_absent_row_is_excluded_not_fabricated(self) -> None:
        grids = np.ones((2, 43, 16, 16), dtype=np.float64)
        masks = np.zeros((2, 16, 16), dtype=bool)
        masks[0, 8, 8] = True
        values, eligible, count = subject.marker_vectors_with_fallback(
            grids, masks, np.asarray(["s", "s"]), np.asarray([0, 1000]), {}, .5)
        self.assertEqual((1, 132), values.shape)
        np.testing.assert_array_equal([0], eligible)
        self.assertEqual(0, count)

    def test_class_source_weights_balance_classes(self) -> None:
        sources = np.asarray(["a", "a", "b", "c", "c", "d"])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        weights = subject.class_source_balanced_weights(sources, labels)
        self.assertAlmostEqual(3.0, weights[labels == 0].sum())
        self.assertAlmostEqual(3.0, weights[labels == 1].sum())

    def test_event_aggregation_uses_mean_and_fixed_threshold(self) -> None:
        rows = subject.aggregate_events(
            np.asarray([.2, .4, .7]), np.asarray(["a", "a", "b"]),
            np.asarray([0, 0, 1]), np.asarray(["s1", "s1", "s2"]), .5)
        self.assertEqual(0, rows[0]["predicted_label"])
        self.assertEqual(1, rows[1]["predicted_label"])


if __name__ == "__main__":
    unittest.main()
