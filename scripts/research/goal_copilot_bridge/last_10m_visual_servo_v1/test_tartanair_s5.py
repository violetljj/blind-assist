from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.tartanair_s5 import select_s5_candidate


def test_s5_requires_depth_distribution_to_straddle_boundary() -> None:
    base = {"provider_rank": 1, "bbox_xyxy": [0, 0, 4, 4], "proposal_score": 0.5, "sensor_region_depth_p20_m": 1.0}
    dino = [{"bbox_xyxy": [0, 0, 4, 4], "score": 0.9}]
    assert select_s5_candidate([base | {"sensor_region_depth_m": 1.5}], dino, 4, 4) is None
    assert select_s5_candidate([base | {"sensor_region_depth_m": 3.0}], dino, 4, 4) is not None
