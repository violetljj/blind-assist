from __future__ import annotations

import math
import unittest

import numpy as np

import dtr_carla_classic_motion_baselines as classic
import dtr_carla_x24_plan_adherent_predictor as x24
import dtr_carla_x24_plan_route_core as route


def measurement(x: float, y: float = 0.0) -> x24.Measurement:
    return x24.Measurement(0, "person", 0.9, (0.1, 0.1, 0.2, 0.3), np.asarray([x, y]), 10)


class ClassicMotionBaselinesTest(unittest.TestCase):
    def test_finite_difference_is_current_measurement_only(self) -> None:
        tracker = classic.CausalFiniteDifferenceTracker()
        self.assertEqual([], tracker.update([measurement(4.0)], 0.0))
        emitted = tracker.update([measurement(3.8)], 0.1)
        self.assertEqual(1, len(emitted))
        self.assertAlmostEqual(-2.0, emitted[0]["velocity_forward_mps"])
        self.assertEqual([], tracker.update([], 0.2))

    def test_ctrv_curves_and_finds_route_entry(self) -> None:
        curved = classic.ctrv_position((0.0, 0.0), (1.0, 0.0), math.pi / 2.0, 1.0)
        self.assertGreater(curved[1], 0.0)
        segments = (route.RouteSegment(0.0, 3.0, (0.0, 0.0), (1.0, 0.0)),)
        entry = classic.first_ctrv_route_entry_s(
            target_position_xy=(2.0, 0.2),
            target_velocity_xy=(-1.0, 0.0),
            yaw_rate_rad_s=0.0,
            route_segments=segments,
        )
        self.assertIsNotNone(entry)

        receding = classic.first_ctrv_route_entry_s(
            target_position_xy=(0.2, 0.0),
            target_velocity_xy=(2.0, 0.0),
            yaw_rate_rad_s=0.0,
            route_segments=segments,
        )
        self.assertIsNone(receding)

    def test_tiny_logistic_fit_is_deterministic_and_discriminative(self) -> None:
        negative = [[4.0, 4.0, 4.0, 10.0, 0.0, 0.0, 0.0, 0.0]] * 12
        positive = [[0.2, 0.2, 0.2, 1.0, 2.0, 1.0, 1.0, 1.0]] * 12
        features = negative + positive
        labels = [False] * len(negative) + [True] * len(positive)
        first = classic.fit_tiny_logistic(features, labels)
        second = classic.fit_tiny_logistic(features, labels)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertLess(first.probability(negative[0]), 0.5)
        self.assertGreater(first.probability(positive[0]), 0.5)


if __name__ == "__main__":
    unittest.main()
