from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.evaluate_future_approach_proposals import bearing_action, evaluate_cases, overlap_metrics


def test_bearing_action_boundaries():
    assert bearing_action([0, 0, 20, 40], 100) == "TURN_LEFT"
    assert bearing_action([40, 0, 60, 40], 100) == "ADVANCE"
    assert bearing_action([80, 0, 100, 40], 100) == "TURN_RIGHT"


def test_positive_target_recall_does_not_label_other_candidates_false():
    prediction = {"cases": [{"case_id": "c", "image_width": 100, "candidates": [
        {"provider_rank": 1, "bbox_xyxy": [0, 0, 10, 10]},
        {"provider_rank": 2, "bbox_xyxy": [70, 0, 90, 40]},
    ]}]}
    private = {"cases": [{"case_id": "c", "legal_targets": [{"target_bbox_xyxy": [70, 0, 90, 40], "demonstrated_action": "TURN_RIGHT"}]}]}
    row = evaluate_cases(prediction, private)[0]
    assert row["recall_at_1"] is False
    assert row["recall_at_3"] is True
    assert row["first_hit_rank"] == 2
    assert row["bearing_action_agreement"] is True


def test_overlap_metrics_expose_containment_when_iou_is_small():
    metrics = overlap_metrics([0, 0, 100, 100], [10, 10, 20, 20])
    assert metrics["iou"] == 0.01
    assert metrics["target_coverage"] == 1.0
    assert metrics["center_containment"] is True
