#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_silver_segformer_free_space_probe as probe


class SegformerFreeSpaceProbeTest(unittest.TestCase):
    def test_masks_are_nonempty_and_core_is_inside_lower(self) -> None:
        masks = probe.clearance_masks(128, 128)
        self.assertTrue(np.all(~masks["core"] | masks["lower"]))
        self.assertFalse(np.any(masks["core"] & masks["peripheral"]))

    def test_center_obstruction_increases_soft_nonwalkable_score(self) -> None:
        masks = probe.clearance_masks(64, 64)
        clear = np.zeros((2, 64, 64), dtype=np.float64)
        clear[0] = 0.9
        clear[1] = 0.1
        blocked = clear.copy()
        blocked[0, masks["core"]] = 0.05
        blocked[1, masks["core"]] = 0.95
        a = probe.frame_descriptor(clear, [0])
        b = probe.frame_descriptor(blocked, [0])
        self.assertGreater(b["core_nonwalkable_mean"], a["core_nonwalkable_mean"])
        self.assertGreater(b["core_center_excess"], a["core_center_excess"])

    def test_peripheral_obstruction_does_not_create_center_excess(self) -> None:
        masks = probe.clearance_masks(64, 64)
        values = np.zeros((2, 64, 64), dtype=np.float64)
        values[0] = 0.9
        values[1] = 0.1
        values[0, masks["peripheral"]] = 0.05
        values[1, masks["peripheral"]] = 0.95
        row = probe.frame_descriptor(values, [0])
        self.assertEqual(0.0, row["core_center_excess"])

    def test_adaptive_path_distinguishes_detour_from_full_blockage(self) -> None:
        clear = np.zeros((2, 64, 64), dtype=np.float64)
        clear[0] = 0.9
        clear[1] = 0.1
        detour = clear.copy()
        detour[0, 35:, 25:39] = 0.05
        detour[1, 35:, 25:39] = 0.95
        blocked = clear.copy()
        blocked[0, 35:, :] = 0.05
        blocked[1, 35:, :] = 0.95
        detour_row = probe.frame_descriptor(detour, [0])
        blocked_row = probe.frame_descriptor(blocked, [0])
        self.assertLess(
            detour_row["path_lower_nonwalkable_mean"],
            blocked_row["path_lower_nonwalkable_mean"],
        )
        self.assertGreater(detour_row["path_offset_maximum"], 0.05)


if __name__ == "__main__":
    unittest.main()
