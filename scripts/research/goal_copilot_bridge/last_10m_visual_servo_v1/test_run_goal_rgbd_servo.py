import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.run_goal_rgbd_servo import candidate_depth, guidance_action, round_robin_guidance, should_stop


def test_candidate_depth_uses_interior_and_ignores_floor_strip():
    depth = np.full((100, 100), 4.0, dtype=np.float32)
    depth[75:, :] = 0.5
    assert candidate_depth(depth, [0, 0, 100, 100]) == 4.0


def test_guidance_action_stops_near_and_bears_when_far():
    assert guidance_action([0, 0, 20, 50], 100, 1.2) == "STOP"
    assert guidance_action([0, 0, 20, 50], 100, 3.0) == "TURN_LEFT"


def test_mask_hint_requires_bounded_interior_depth_support():
    assert should_stop(2.0, 1.0) is True
    assert should_stop(3.0, 1.0) is False


def test_guidance_round_robin_keeps_route_stop_and_other_hypotheses():
    groups = [[{"proposal_rank": 1}, {"proposal_rank": 2}], [{"proposal_rank": 3}], [{"proposal_rank": 4}]]
    assert [row["proposal_rank"] for row in round_robin_guidance(groups)] == [1, 3, 4]
