from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.visual_servo import choose_action, crop_geometry, transform_box


CONFIG = {
    "no_candidate_action": "TURN_LEFT",
    "align_left": 0.45,
    "align_right": 0.55,
    "arrival_min_height": 0.55,
}


class VisualServoMechanicsTest(unittest.TestCase):
    def test_goal_directed_no_candidate_scan_is_left(self) -> None:
        self.assertEqual(("TURN_LEFT", None), choose_action([], pending_confirmation=False, config=CONFIG))

    def test_candidate_alignment_and_two_frame_completion(self) -> None:
        left = {"provider_rank": 1, "bbox_xyxy": [10, 100, 200, 500], "proposal_score": 0.9}
        right = {"provider_rank": 1, "bbox_xyxy": [440, 100, 620, 500], "proposal_score": 0.9}
        centered_far = {"provider_rank": 1, "bbox_xyxy": [250, 180, 390, 460], "proposal_score": 0.9}
        centered_near = {"provider_rank": 1, "bbox_xyxy": [180, 100, 460, 500], "proposal_score": 0.9}
        self.assertEqual("TURN_LEFT", choose_action([left], pending_confirmation=False, config=CONFIG)[0])
        self.assertEqual("TURN_RIGHT", choose_action([right], pending_confirmation=False, config=CONFIG)[0])
        self.assertEqual("FORWARD", choose_action([centered_far], pending_confirmation=False, config=CONFIG)[0])
        self.assertEqual("ARRIVAL_CONFIRM", choose_action([centered_near], pending_confirmation=False, config=CONFIG)[0])
        self.assertEqual("COMPLETE", choose_action([centered_near], pending_confirmation=True, config=CONFIG)[0])

    def test_leftmost_relation_selection_precedes_action(self) -> None:
        wrong_high_score = {"provider_rank": 1, "bbox_xyxy": [400, 100, 600, 500], "proposal_score": 0.95}
        left = {"provider_rank": 2, "bbox_xyxy": [20, 100, 220, 500], "proposal_score": 0.2}
        action, selected = choose_action([wrong_high_score, left], pending_confirmation=False, config=CONFIG)
        self.assertEqual("TURN_LEFT", action)
        self.assertEqual(2, selected["provider_rank"])

    def test_private_truth_transform_reports_visibility(self) -> None:
        geometry = crop_geometry(1000, 800, 2.0, 0.0, 0)
        rendered, visibility = transform_box([10, 200, 210, 600], geometry)
        self.assertIsNotNone(rendered)
        self.assertAlmostEqual(1.0, visibility)
        absent, absent_visibility = transform_box([800, 200, 950, 600], geometry)
        self.assertIsNone(absent)
        self.assertEqual(0.0, absent_visibility)


if __name__ == "__main__":
    unittest.main()
