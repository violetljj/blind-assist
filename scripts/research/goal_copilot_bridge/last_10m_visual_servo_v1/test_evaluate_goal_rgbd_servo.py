from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.evaluate_goal_rgbd_servo import evaluate_cases


def test_target_action_requires_both_functional_match_and_action():
    prediction = {"cases": [{"case_id": "c", "image_width": 100, "image_height": 100, "candidates": [
        {"bbox_xyxy": [60, 0, 100, 100], "action": "TURN_RIGHT"},
        {"bbox_xyxy": [60, 0, 100, 100], "action": "STOP"},
    ]}]}
    private = {"cases": [{"case_id": "c", "phase": "NEAR_STOP", "legal_targets": [{"target_bbox_xyxy": [70, 10, 80, 20], "desired_action": "STOP"}]}]}
    row = evaluate_cases(prediction, private)[0]
    assert row["target_action_recall_at_1"] is False
    assert row["target_action_recall_at_3"] is True
