import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.train_domain_adapted_functional_door import components_to_yolo, trajectory_split


def test_component_box_is_converted_to_yolo() -> None:
    segmentation = np.zeros((100, 200), dtype=np.uint8)
    segmentation[20:60, 50:150] = 7
    rows = components_to_yolo(segmentation, 7, 0)
    assert rows == ["0 0.50000000 0.40000000 0.50000000 0.40000000"]


def test_trajectory_split_is_frozen_at_trajectory_level() -> None:
    assert trajectory_split("P004") == "val"
    assert trajectory_split("P000") == "train"
