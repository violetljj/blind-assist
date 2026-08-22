from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.functional_region_clip_fusion_dev import LABELS


class FunctionalRegionClipFusionDevTest(unittest.TestCase):
    def test_positive_label_is_passage_door(self) -> None:
        self.assertEqual(LABELS[0], "a full-size room door used for people to pass through")


if __name__ == "__main__":
    unittest.main()
