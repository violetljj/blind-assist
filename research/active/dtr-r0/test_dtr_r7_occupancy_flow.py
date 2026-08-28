from __future__ import annotations

import math
import unittest

import numpy as np

from dtr_r7_occupancy_flow_canary import (
    Component,
    FROZEN_FLOW_CONFIG,
    _causal_pose,
    _ego_to_world,
    _entry_s,
    _match_components,
    _world_to_ego_xy,
)


class DTRR7OccupancyFlowTest(unittest.TestCase):
    def test_pose_lookup_is_strictly_causal(self) -> None:
        samples = [
            {
                "timestamp_ns": 100,
                "translation": [1.0, 2.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "timestamp_ns": 200,
                "translation": [9.0, 9.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        ]
        pose = _causal_pose(samples, 150)
        self.assertEqual((pose["x_m"], pose["y_m"]), (1.0, 2.0))

    def test_ego_world_round_trip(self) -> None:
        pose = {"x_m": 3.0, "y_m": -2.0, "yaw_rad": math.pi / 3.0}
        local = np.asarray([[1.2, -0.4, 0.3], [2.0, 1.0, -0.2]])
        world = _ego_to_world(local, pose)
        restored = _world_to_ego_xy(world[:, :2], pose)
        np.testing.assert_allclose(restored, local[:, :2], atol=1e-10)

    def test_component_correspondence_rejects_static_and_keeps_motion(self) -> None:
        previous = Component(
            frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
            0.12,
            0.12,
            0.24,
            0.24,
        )
        static = Component(previous.keys, 0.12, 0.12, 0.24, 0.24)
        self.assertEqual(_match_components([previous], [static], 0.36, FROZEN_FLOW_CONFIG), [])

        moving = Component(
            frozenset({(3, 0), (3, 1), (4, 0), (4, 1)}),
            0.48,
            0.12,
            0.24,
            0.24,
        )
        matches = _match_components([previous], [moving], 0.36, FROZEN_FLOW_CONFIG)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0][2], 1.0)
        self.assertAlmostEqual(matches[0][3], 0.0)

    def test_flow_entry_uses_frozen_three_second_tube(self) -> None:
        entry = _entry_s(2.0, 0.0, -1.0, 0.0)
        self.assertIsNotNone(entry)
        self.assertAlmostEqual(float(entry), 1.35)
        self.assertIsNone(_entry_s(2.0, 0.0, 1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
