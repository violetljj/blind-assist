from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.door_consensus_dev import select_consensus


class SensorDepthFusionTest(unittest.TestCase):
    def test_sensor_depth_can_authorize_consensus_without_monocular_value(self) -> None:
        candidates = [{"provider_rank": 1, "proposal_score": 0.8, "bbox_xyxy": [20, 0, 80, 100], "monocular_region_depth_m": 3.5, "predicted_region_depth_m": 1.5}]
        dino = [{"score": 0.9, "bbox_xyxy": [21, 0, 79, 100]}]
        self.assertIsNotNone(select_consensus(candidates, dino, 50.0))


if __name__ == "__main__":
    unittest.main()
