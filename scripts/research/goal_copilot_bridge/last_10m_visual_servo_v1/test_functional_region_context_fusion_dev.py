from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.functional_region_context_fusion_dev import DOOR_PROBABILITY_MIN


class FunctionalRegionContextFusionDevTest(unittest.TestCase):
    def test_threshold_is_frozen(self) -> None:
        self.assertEqual(DOOR_PROBABILITY_MIN, 0.5)


if __name__ == "__main__":
    unittest.main()
