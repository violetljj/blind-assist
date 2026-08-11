#!/usr/bin/env python3
"""Focused test for exact TARO R6 source inventory."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.taro_o0r_candidate_scale_runtime import run_r6_untouched_inventory as inventory


class R6UntouchedInventoryTests(unittest.TestCase):
    def test_downloaded_containers_reproduce_exact_120_frame_plan(self) -> None:
        value = inventory.build_inventory(Path(__file__).resolve().parents[3])
        self.assertEqual(8, value["parent_count"])
        self.assertEqual(120, value["exact_pose_bounded_frame_count"])
        self.assertEqual([16, 14, 8, 13, 11, 24, 5, 29], [row["frame_plan"]["exact_pose_bounded_frame_count"] for row in value["parents"]])
        self.assertFalse(value["pixel_arrays_decoded"])
        self.assertFalse(value["truth_values_interpreted"])


if __name__ == "__main__":
    unittest.main()
