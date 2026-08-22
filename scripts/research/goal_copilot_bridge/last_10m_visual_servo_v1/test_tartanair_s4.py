from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.tartanair_s4 import select_s4_candidate


def test_apparent_height_proxy_rejects_short_near_panel() -> None:
    dino = [{"bbox_xyxy": [100, 100, 500, 600], "score": 0.9}]
    base = {"provider_rank": 1, "bbox_xyxy": [100, 100, 500, 600], "proposal_score": 0.5, "sensor_region_depth_m": 1.5}
    assert select_s4_candidate([base], dino, 640, 640) is not None
    short_near = base | {"bbox_xyxy": [288, 343, 546, 640], "sensor_region_depth_m": 0.72}
    assert select_s4_candidate([short_near], dino, 640, 640) is None
