from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.evaluate_functional_region_availability import evaluate_cases, functional_region_hit


def test_bounded_containment_accepts_small_component_inside_bounded_region():
    hit, metrics = functional_region_hit([0, 0, 40, 100], [10, 10, 20, 20], 100, 100)
    assert hit is True
    assert metrics["target_coverage"] == 1.0
    assert metrics["candidate_area_fraction"] == 0.4


def test_bounded_containment_rejects_near_full_frame_proposal():
    hit, metrics = functional_region_hit([0, 0, 99, 99], [10, 10, 20, 20], 100, 100)
    assert hit is False
    assert metrics["target_coverage"] == 1.0


def test_evaluator_reports_containment_hit_without_calling_alternatives_false():
    prediction = {"cases": [{"case_id": "c", "image_width": 100, "image_height": 100, "candidates": [
        {"provider_rank": 1, "bbox_xyxy": [0, 0, 10, 10]},
        {"provider_rank": 2, "bbox_xyxy": [60, 0, 100, 100]},
    ]}]}
    private = {"cases": [{"case_id": "c", "legal_targets": [{"target_bbox_xyxy": [70, 10, 80, 20], "demonstrated_action": "TURN_RIGHT"}]}]}
    row = evaluate_cases(prediction, private)[0]
    assert row["functional_recall_at_1"] is False
    assert row["functional_recall_at_3"] is True
    assert row["first_functional_hit_kind"] == "BOUNDED_CONTAINMENT"
    assert row["bearing_action_agreement"] is True
