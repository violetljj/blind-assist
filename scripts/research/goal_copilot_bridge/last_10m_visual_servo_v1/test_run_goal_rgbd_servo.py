import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.run_goal_rgbd_servo import candidate_depth, guidance_action


def test_candidate_depth_uses_interior_and_ignores_floor_strip():
    depth = np.full((100, 100), 4.0, dtype=np.float32)
    depth[75:, :] = 0.5
    assert candidate_depth(depth, [0, 0, 100, 100]) == 4.0


def test_guidance_action_stops_near_and_bears_when_far():
    assert guidance_action([0, 0, 20, 50], 100, 1.2) == "STOP"
    assert guidance_action([0, 0, 20, 50], 100, 3.0) == "TURN_LEFT"
