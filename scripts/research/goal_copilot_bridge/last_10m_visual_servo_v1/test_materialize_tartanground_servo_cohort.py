import io
import zipfile

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanground_servo_cohort import read_poses, route_plan, tg_member


def test_tartanground_members_are_per_trajectory():
    assert tg_member(12, "image") == "image_lcam_front/000012_lcam_front.png"
    assert tg_member(12, "depth") == "depth_lcam_front/000012_lcam_front_depth.png"


def test_route_plan_uses_public_future_pose_without_truth():
    poses = np.zeros((31, 7), dtype=np.float64)
    poses[:, 6] = 1.0
    poses[30, :2] = [3.0, 1.0]
    plan = route_plan(poses, 0)
    assert plan["bearing_fraction"] > 0.5
    assert plan["derived_without_semantic_or_target_truth"] is True


def test_route_plan_keeps_episode_waypoint_stable_for_near_phase():
    poses = np.zeros((41, 7), dtype=np.float64)
    poses[:, 6] = 1.0
    poses[30, 0] = 3.0
    poses[40, 1] = 8.0
    plan = route_plan(poses, 20, waypoint_frame=30)
    assert plan["waypoint_frame_id"] == 30
    assert plan["bearing_fraction"] == 0.5


def test_read_pose_contract():
    payload = "0 0 0 0 0 0 1\n1 0 0 0 0 0 1\n"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("pose_lcam_front.txt", payload)
    with zipfile.ZipFile(stream) as archive:
        poses = read_poses(archive)
    assert poses.shape == (2, 7)
