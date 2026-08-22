from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.train_candidate_verifier import balanced_accuracy


def test_balanced_accuracy_weights_classes_equally() -> None:
    assert balanced_accuracy([9, 1], [10, 2]) == 0.7
