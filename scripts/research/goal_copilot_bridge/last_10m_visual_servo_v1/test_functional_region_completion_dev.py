from __future__ import annotations

import unittest

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.functional_region_completion_dev import functional_observation, observer_connected_ground, select_functional_candidate


class FunctionalRegionCompletionDevTest(unittest.TestCase):
    def test_connected_ground_keeps_observer_component(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[60:100, 30:70] = True
        mask[10:30, 10:30] = True
        connected = observer_connected_ground(mask)
        self.assertTrue(connected[90, 50])
        self.assertFalse(connected[20, 20])

    def test_functional_contact_uses_adjacent_ground_depth(self) -> None:
        door = np.zeros((100, 100), dtype=bool)
        door[20:80, 45:55] = True
        ground = np.zeros((100, 100), dtype=bool)
        ground[80:100, :] = True
        depth = np.full((100, 100), 4.0, dtype=np.float32)
        depth[ground] = 1.5
        observed = functional_observation(door, ground, depth)
        self.assertGreater(observed["ground_contact_pixel_count"], 0)
        self.assertEqual(observed["ground_contact_depth_median_m"], 1.5)

    def test_selection_requires_floor_contact(self) -> None:
        base = {"provider_rank": 1, "proposal_score": 0.2, "bbox_xyxy": [30, 10, 70, 95], "mask_height_fraction": 0.8, "ground_contact_pixel_count": 100, "ground_contact_depth_median_m": 1.5}
        dino = [{"bbox_xyxy": [31, 11, 69, 94], "score": 0.8, "label": "door"}]
        self.assertIsNotNone(select_functional_candidate([base], dino, 100))
        self.assertIsNone(select_functional_candidate([base | {"ground_contact_pixel_count": 0}], dino, 100))


if __name__ == "__main__":
    unittest.main()
