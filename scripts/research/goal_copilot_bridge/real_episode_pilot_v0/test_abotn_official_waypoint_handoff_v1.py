from __future__ import annotations

import math
import unittest

import numpy as np

from .run_abotn_official_waypoint_handoff_v1 import (
    _CanonicalViewRenderer,
    _action_prediction,
)


class _Renderer:
    def render_at_pose(self, *args, **kwargs):
        return ["left", "right", "front"]


class OfficialWaypointAdapterTest(unittest.TestCase):
    def test_repairs_official_current_view_order(self) -> None:
        renderer = _CanonicalViewRenderer(_Renderer())
        self.assertEqual(["left", "front", "right"], renderer.render_at_pose(None))

    def test_forward_is_two_metres_in_official_front_left_coordinates(self) -> None:
        waypoint, direction, stop = _action_prediction("FORWARD")
        np.testing.assert_allclose([[2.0, 0.0]], waypoint)
        np.testing.assert_allclose([[1.0, 0.0]], direction)
        self.assertFalse(stop)

    def test_turns_are_in_place_and_have_opposite_frozen_yaw(self) -> None:
        left_wp, left_dir, left_stop = _action_prediction("TURN_LEFT")
        right_wp, right_dir, right_stop = _action_prediction("TURN_RIGHT")
        np.testing.assert_allclose([[0.0, 0.0]], left_wp)
        np.testing.assert_allclose([[0.0, 0.0]], right_wp)
        self.assertAlmostEqual(math.radians(12), math.atan2(left_dir[0, 1], left_dir[0, 0]))
        self.assertAlmostEqual(math.radians(-12), math.atan2(right_dir[0, 1], right_dir[0, 0]))
        self.assertFalse(left_stop)
        self.assertFalse(right_stop)

    def test_terminal_control_uses_stop_transport_without_motion(self) -> None:
        waypoint, direction, stop = _action_prediction(None)
        np.testing.assert_allclose([[0.0, 0.0]], waypoint)
        np.testing.assert_allclose([[1.0, 0.0]], direction)
        self.assertTrue(stop)

    def test_rescan_is_a_fresh_in_place_sweep(self) -> None:
        waypoint, direction, stop = _action_prediction("RESCAN_HOLD")
        np.testing.assert_allclose([[0.0, 0.0]], waypoint)
        self.assertAlmostEqual(math.radians(12), math.atan2(direction[0, 1], direction[0, 0]))
        self.assertFalse(stop)


if __name__ == "__main__":
    unittest.main()
