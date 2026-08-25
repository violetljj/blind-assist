#!/usr/bin/env python3

import math
import unittest

from grail_m0 import (
    Pose,
    angle_delta,
    interaction_pose_set,
    judge_pose,
    make_scene,
    pose_matches,
    shortest_path,
)


class GrailM0Tests(unittest.TestCase):
    def test_positive_scene_has_set_valued_reachable_truth(self) -> None:
        scene = make_scene("HELD_OUT", 1)
        truth = interaction_pose_set(scene)
        self.assertGreaterEqual(len(truth), 4)
        self.assertTrue(all(judge_pose(scene, scene.target, pose).valid for pose in truth))
        self.assertTrue(all(shortest_path(scene, (pose.x, pose.y)) for pose in truth))

    def test_no_pose_scenes_fail_closed(self) -> None:
        blocked = make_scene("HELD_OUT", 3)
        isolated = make_scene("HELD_OUT", 0)
        self.assertEqual(blocked.no_pose_reason, "BLOCKED_FRONT")
        self.assertEqual(isolated.no_pose_reason, "ISOLATED_BY_WALL")
        self.assertEqual(interaction_pose_set(blocked), ())
        self.assertEqual(interaction_pose_set(isolated), ())

    def test_back_side_is_not_an_interaction_pose(self) -> None:
        scene = make_scene("HELD_OUT", 1)
        target = scene.target
        back = Pose(
            target.cx - target.front_x * 0.85,
            target.cy - target.front_y * 0.85,
            math.atan2(target.front_y, target.front_x),
        )
        judgement = judge_pose(scene, target, back)
        self.assertFalse(judgement.valid)
        self.assertIn("NOT_FUNCTIONAL_SIDE", judgement.reasons)

    def test_pose_matching_is_set_valued_and_checks_yaw(self) -> None:
        truth = (Pose(1.0, 1.0, 0.0), Pose(2.0, 2.0, math.pi / 2))
        self.assertTrue(pose_matches(Pose(2.1, 2.0, math.pi / 2 + 0.1), truth))
        self.assertFalse(pose_matches(Pose(2.1, 2.0, math.pi), truth))
        self.assertAlmostEqual(angle_delta(-math.pi + 0.1, math.pi - 0.1), 0.2)

    def test_scene_and_instance_ids_are_split_disjoint(self) -> None:
        dev = {make_scene("DEVELOPMENT", i).scene_id for i in range(12)}
        held = {make_scene("HELD_OUT", i).scene_id for i in range(36)}
        self.assertFalse(dev & held)


if __name__ == "__main__":
    unittest.main()
