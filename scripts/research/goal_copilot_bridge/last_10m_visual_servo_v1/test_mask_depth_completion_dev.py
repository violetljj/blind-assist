from __future__ import annotations

import unittest

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.mask_depth_completion_dev import binary_mask_iou, mask_observation, select_candidate


class MaskDepthCompletionDevTest(unittest.TestCase):
    def test_mask_observation_uses_only_masked_depth(self) -> None:
        depth = np.full((100, 100), 7.0, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:90, 45:55] = 1.0
        depth[mask > 0] = 1.25
        observed = mask_observation(mask, depth, [40, 10, 60, 95], 100, 100)
        self.assertEqual(observed["mask_depth_median_m"], 1.25)
        self.assertEqual(observed["mask_height_fraction"], 0.70)
        self.assertGreater(observed["mask_center_band_pixel_count"], 0)

    def test_selection_requires_centered_near_mask_and_consensus(self) -> None:
        base = {"provider_rank": 1, "proposal_score": 0.2, "bbox_xyxy": [30, 10, 70, 95], "mask_height_fraction": 0.8, "mask_center_band_pixel_count": 50, "mask_depth_median_m": 1.5}
        selected = select_candidate([base], [{"bbox_xyxy": [31, 11, 69, 94], "score": 0.8, "label": "door"}])
        self.assertIsNotNone(selected)
        self.assertIsNone(select_candidate([base | {"mask_depth_median_m": 2.1}], [{"bbox_xyxy": [31, 11, 69, 94], "score": 0.8, "label": "door"}]))

    def test_binary_mask_iou(self) -> None:
        left = np.zeros((4, 4), dtype=bool)
        right = np.zeros((4, 4), dtype=bool)
        left[0:2, 0:2] = True
        right[1:3, 1:3] = True
        self.assertAlmostEqual(binary_mask_iou(left, right), 1 / 7)


if __name__ == "__main__":
    unittest.main()
