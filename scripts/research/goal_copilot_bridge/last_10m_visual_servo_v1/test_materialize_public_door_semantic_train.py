import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_public_door_semantic_train import binary_door_mask, split_for_trajectory


def test_binary_door_mask_keeps_only_exact_door() -> None:
    source = np.array([[4, 7], [7, 9]], dtype=np.uint8)
    assert binary_door_mask(source, 7).tolist() == [[0, 1], [1, 0]]


def test_trajectory_split_is_deterministic_and_environment_scoped() -> None:
    assert split_for_trajectory("Hospital", "P001") == split_for_trajectory("Hospital", "P001")
    assert split_for_trajectory("Hospital", "P001") in {"train", "val"}
