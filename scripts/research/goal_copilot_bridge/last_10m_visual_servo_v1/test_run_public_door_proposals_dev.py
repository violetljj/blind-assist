import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.run_public_door_proposals_dev import semantic_components


def test_semantic_components_scale_and_rank():
    class_map = np.zeros((10, 10), dtype=np.int64)
    class_map[1:5, 1:4] = 1
    class_map[6:9, 7:9] = 1
    candidates = semantic_components(class_map, 1, 100, 200, minimum_pixels=2)
    assert candidates[0]["bbox_xyxy"] == [10.0, 20.0, 40.0, 100.0]
    assert candidates[0]["semantic_pixel_count"] == 12
