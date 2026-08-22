from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.candidate_depth_fusion import _select


class CandidateDepthFusionTest(unittest.TestCase):
    def test_selects_highest_score_centered_near_candidate(self) -> None:
        candidates = [
            {"provider_rank": 1, "proposal_score": 0.9, "bbox_xyxy": [0, 0, 60, 100], "predicted_region_depth_m": 1.0},
            {"provider_rank": 2, "proposal_score": 0.8, "bbox_xyxy": [40, 0, 80, 100], "predicted_region_depth_m": 1.5},
            {"provider_rank": 3, "proposal_score": 0.7, "bbox_xyxy": [30, 0, 90, 100], "predicted_region_depth_m": 3.0},
        ]
        self.assertEqual(2, _select(candidates, 70.0, 2.0)["provider_rank"])

    def test_abstains_without_centered_near_candidate(self) -> None:
        candidates = [{"provider_rank": 1, "proposal_score": 0.9, "bbox_xyxy": [0, 0, 40, 100], "predicted_region_depth_m": 1.0}]
        self.assertIsNone(_select(candidates, 70.0, 2.0))


if __name__ == "__main__":
    unittest.main()
