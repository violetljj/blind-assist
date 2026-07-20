import unittest

import numpy as np

import run_public_video_causal_waypoint_linear_probe as subject


class CausalWaypointLinearProbeTest(unittest.TestCase):
    def test_pooling_shape_is_frozen(self) -> None:
        values = np.arange(2 * 43 * 16 * 16, dtype=np.float64).reshape(2, 43, 16, 16)
        self.assertEqual((2, 731), subject.pooled_features(values).shape)

    def test_waypoint_target_uses_argmax_patch_center(self) -> None:
        fields = np.zeros((1, 3, 16, 16), dtype=np.float64)
        fields[0, 0, 2, 4] = 1
        fields[0, 1, 8, 9] = 1
        fields[0, 2, 15, 0] = 1
        result = subject.waypoint_targets(fields)
        np.testing.assert_allclose(result[0], [4.5 / 16, 2.5 / 16, 9.5 / 16, 8.5 / 16, .5 / 16, 15.5 / 16])

    def test_ridge_recovers_linear_coordinates(self) -> None:
        x = np.arange(30, dtype=np.float64).reshape(10, 3) / 30
        y = np.column_stack([x[:, 0] * .5 + .2, x[:, 1] * .25 + .1])
        model = subject.fit_ridge(x, y, np.ones(10), 1e-8)
        np.testing.assert_allclose(subject.predict(model, x), y, atol=1e-6)

    def test_event_predictions_maps_coordinates_to_obstacle(self) -> None:
        points = np.asarray([[.5, .5, .5, .5, .1, .1]])
        obstacle = np.zeros((1, 16, 16), dtype=bool)
        obstacle[0, 8, 8] = True
        rows = subject.event_predictions(points, np.asarray(["e"]), np.asarray([1]), obstacle)
        self.assertAlmostEqual(2 / 3, rows[0]["predicted_horizon_hit_fraction"])


if __name__ == "__main__":
    unittest.main()
