#!/usr/bin/env python3

import unittest

from grail_procthor_native_m0 import (
    action_pair,
    counterfactuals,
    has_local_stability,
    interaction_pose_success,
    is_action_target,
    reachable_path_exists,
)


def pose(x: float, z: float, rotation: float) -> dict:
    return {"x": x, "y": 0.9, "z": z, "rotation": rotation, "standing": True, "horizon": 0.0}


class GrailProcThorNativeM0Tests(unittest.TestCase):
    def test_action_target_requires_stationary_nonpickup_action(self) -> None:
        self.assertTrue(is_action_target({"openable": True}))
        self.assertTrue(is_action_target({"toggleable": True}))
        self.assertFalse(is_action_target({"openable": True, "pickupable": True}))
        self.assertFalse(is_action_target({"toggleable": True, "moveable": True}))
        self.assertFalse(is_action_target({"receptacle": True}))

    def test_action_pair_uses_current_state_and_reverts(self) -> None:
        self.assertEqual(action_pair({"openable": True, "isOpen": False}), ("OpenObject", "CloseObject"))
        self.assertEqual(action_pair({"toggleable": True, "isToggled": True}), ("ToggleObjectOff", "ToggleObjectOn"))

    def test_pose_success_uses_declared_position_and_yaw_tolerances(self) -> None:
        truth = [pose(1.0, 1.0, 350.0)]
        self.assertTrue(interaction_pose_success(pose(1.3, 1.1, 5.0), truth))
        self.assertFalse(interaction_pose_success(pose(1.6, 1.0, 350.0), truth))
        self.assertFalse(interaction_pose_success(pose(1.0, 1.0, 30.1), truth))

    def test_reachable_path_requires_connected_grid_component(self) -> None:
        connected = [{"x": x, "y": 0.9, "z": 0.0} for x in (0.0, 0.25, 0.5)]
        disconnected = connected + [{"x": 2.0, "y": 0.9, "z": 0.0}]
        self.assertTrue(reachable_path_exists(connected, pose(0.5, 0.0, 0.0)))
        self.assertFalse(reachable_path_exists(disconnected, pose(0.5, 0.0, 0.0)))

    def test_local_stability_requires_neighbor_pose(self) -> None:
        candidate = pose(0.0, 0.0, 0.0)
        self.assertTrue(has_local_stability(candidate, [candidate, pose(0.25, 0.0, 0.0)]))
        self.assertFalse(has_local_stability(candidate, [candidate, pose(1.0, 0.0, 0.0)]))

    def test_counterfactuals_construct_two_rejected_families(self) -> None:
        truth = [pose(0.0, 0.0, 0.0)]
        reachable = [{"x": 0.0, "y": 0.9, "z": 0.0}, {"x": 3.0, "y": 0.9, "z": 0.0}]
        rows = counterfactuals(truth[0], truth, reachable, {"x": 0.0, "y": 0.0, "z": 0.0})
        self.assertEqual(
            [row["family"] for row in rows],
            ["BACK_FACING", "FREE_BUT_UNRELATED", "OUTSIDE_REACHABLE_NAVMESH"],
        )
        self.assertTrue(all(row["rejected"] for row in rows))


if __name__ == "__main__":
    unittest.main()
