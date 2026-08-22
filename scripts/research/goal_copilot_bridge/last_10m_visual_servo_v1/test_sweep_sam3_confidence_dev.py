from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.sweep_sam3_confidence_dev import sweep


def test_sweep_prefers_stricter_threshold_after_low_score_false_positive():
    prediction = {
        "cases": [
            {"case_id": "near", "image_width": 100, "candidates": [{"proposal_score": 0.3, "bbox_xyxy": [40, 0, 60, 100], "mask_height_fraction": 1.0, "ground_contact_pixel_count": 100, "ground_contact_depth_median_m": 1.0}]},
            {"case_id": "far", "image_width": 100, "candidates": [{"proposal_score": 0.1, "bbox_xyxy": [40, 0, 60, 100], "mask_height_fraction": 1.0, "ground_contact_pixel_count": 100, "ground_contact_depth_median_m": 1.0}]},
        ]
    }
    private = {"cases": [
        {"case_id": "near", "legal_targets": [{"target_bbox_xyxy": [40, 0, 60, 100], "target_depth_median_m": 1.0}]},
        {"case_id": "far", "legal_targets": [{"target_bbox_xyxy": [0, 0, 10, 100], "target_depth_median_m": 3.0}]},
    ]}
    rows = {row["confidence_threshold"]: row for row in sweep(prediction, private)}
    assert rows[0.10]["false_count"] == 1
    assert rows[0.15]["false_count"] == 0
    assert rows[0.15]["correct_count"] == 1
