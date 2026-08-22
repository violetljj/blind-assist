import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.public_door_semantic_fusion_dev import door_fraction


def test_door_fraction_uses_candidate_region() -> None:
    class_map = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    assert door_fraction(class_map, [1, 0, 2, 2], 2, 2, 1) == 1.0
    assert door_fraction(class_map, [0, 0, 1, 2], 2, 2, 1) == 0.0
