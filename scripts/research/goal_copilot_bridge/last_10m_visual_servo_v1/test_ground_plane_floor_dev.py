from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.ground_plane_floor_dev import FLOOR_PRECISION_MIN, FLOOR_RECALL_MIN, MIN_FIT_COUNT


class GroundPlaneFloorDevTest(unittest.TestCase):
    def test_gate_is_fail_closed(self) -> None:
        self.assertEqual(FLOOR_PRECISION_MIN, 0.80)
        self.assertEqual(FLOOR_RECALL_MIN, 0.50)
        self.assertEqual(MIN_FIT_COUNT, 40)


if __name__ == "__main__":
    unittest.main()
