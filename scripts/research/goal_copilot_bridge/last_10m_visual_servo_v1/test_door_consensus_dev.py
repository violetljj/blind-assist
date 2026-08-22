from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.door_consensus_dev import select_consensus


class DoorConsensusDevTest(unittest.TestCase):
    def test_requires_center_depth_and_cross_provider_overlap(self) -> None:
        yoloe = [
            {"provider_rank": 1, "proposal_score": 0.8, "bbox_xyxy": [20, 0, 80, 100], "predicted_region_depth_m": 1.5},
            {"provider_rank": 2, "proposal_score": 0.7, "bbox_xyxy": [30, 0, 90, 100], "predicted_region_depth_m": 3.0},
        ]
        dino = [{"score": 0.9, "bbox_xyxy": [22, 0, 78, 100]}]
        self.assertEqual(1, select_consensus(yoloe, dino, 50.0)["provider_rank"])

    def test_abstains_without_consensus(self) -> None:
        yoloe = [{"provider_rank": 1, "proposal_score": 0.8, "bbox_xyxy": [20, 0, 80, 100], "predicted_region_depth_m": 1.5}]
        dino = [{"score": 0.9, "bbox_xyxy": [100, 0, 160, 100]}]
        self.assertIsNone(select_consensus(yoloe, dino, 50.0))


if __name__ == "__main__":
    unittest.main()
