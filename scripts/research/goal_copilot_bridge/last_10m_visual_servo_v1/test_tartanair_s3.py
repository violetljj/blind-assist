from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.tartanair_s3 import select_geometric_candidate


def test_geometric_selector_requires_all_frozen_gates() -> None:
    dino = [{"bbox_xyxy": [100, 100, 500, 600], "score": 0.9}]
    good = {"provider_rank": 1, "bbox_xyxy": [100, 100, 500, 600], "proposal_score": 0.5, "sensor_region_depth_m": 1.5}
    assert select_geometric_candidate([good], dino, 640, 640) is not None
    assert select_geometric_candidate([good | {"sensor_region_depth_m": 1.81}], dino, 640, 640) is None
    assert select_geometric_candidate([good | {"bbox_xyxy": [200, 350, 440, 600]}], dino, 640, 640) is None
    assert select_geometric_candidate([good], [{"bbox_xyxy": [0, 0, 120, 120], "score": 0.9}], 640, 640) is None
