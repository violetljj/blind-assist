from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.sam3_functional_region_dev import CONFIDENCE_THRESHOLD, PROMPT, select_sam3_functional_candidate


class Sam3FunctionalRegionDevTest(unittest.TestCase):
    def test_provider_contract_is_frozen(self) -> None:
        self.assertEqual(PROMPT, "door")
        self.assertEqual(CONFIDENCE_THRESHOLD, 0.5)

    def test_selection_requires_functional_contact(self) -> None:
        base = {"bbox_xyxy": [10, 0, 90, 100], "mask_height_fraction": 0.8, "ground_contact_pixel_count": 100, "ground_contact_depth_median_m": 1.5, "proposal_score": 0.8}
        self.assertIsNotNone(select_sam3_functional_candidate([base], 100))
        self.assertIsNone(select_sam3_functional_candidate([base | {"ground_contact_pixel_count": 0}], 100))


if __name__ == "__main__":
    unittest.main()
