from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.train_depth_structure_verifier import FEATURE_NAMES


def test_depth_structure_feature_contract_is_unique() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 21
