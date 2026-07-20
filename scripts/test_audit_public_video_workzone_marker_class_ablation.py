#!/usr/bin/env python3
"""Pure tests for post-failure marker-class ablation."""

from __future__ import annotations

import unittest

import audit_public_video_workzone_marker_class_ablation as subject


class WorkzoneMarkerClassAblationTest(unittest.TestCase):
    def test_filter_drops_only_requested_marker(self) -> None:
        rows = [{
            "timestamp_ms": 0,
            "semantic_class_counts": {
                "barricade": 1,
                "traffic cone": 2,
                "construction site": 1,
                "sand": 1,
            },
            "semantic_group_counts": {"barrier_structure": 4, "surface_material": 1},
        }]
        filtered = subject.filtered_samples(rows, {"traffic cone"})[0]
        self.assertEqual(3, filtered["semantic_group_counts"]["barrier_structure"])
        self.assertEqual(1, filtered["semantic_group_counts"]["surface_material"])
        self.assertNotIn("barricade", filtered["semantic_class_counts"])

    def test_filter_can_remove_all_exploratory_markers(self) -> None:
        rows = [{
            "timestamp_ms": 0,
            "semantic_class_counts": {"barricade": 1, "traffic cone": 2},
            "semantic_group_counts": {"barrier_structure": 3},
        }]
        filtered = subject.filtered_samples(rows, set())[0]
        self.assertEqual({}, filtered["semantic_group_counts"])
        self.assertEqual({}, filtered["semantic_class_counts"])


if __name__ == "__main__":
    unittest.main()
