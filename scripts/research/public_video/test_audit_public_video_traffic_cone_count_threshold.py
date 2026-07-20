#!/usr/bin/env python3
"""Pure tests for the multi-cone corridor threshold diagnostic."""

from __future__ import annotations

import unittest

import audit_public_video_traffic_cone_count_threshold as subject


class TrafficConeCountThresholdTest(unittest.TestCase):
    def test_threshold_uses_only_traffic_cone_count(self) -> None:
        rows = [{
            "timestamp_ms": 0,
            "semantic_class_counts": {
                "traffic cone": 2,
                "construction site": 4,
                "barricade": 3,
            },
            "semantic_group_counts": {"barrier_structure": 9},
        }]
        kept = subject.cone_expert_samples(rows, 2)[0]
        self.assertEqual({"barrier_structure": 1}, kept["semantic_group_counts"])
        self.assertEqual({"traffic cone": 2}, kept["semantic_class_counts"])

    def test_below_threshold_is_inactive(self) -> None:
        rows = [{
            "timestamp_ms": 0,
            "semantic_class_counts": {"traffic cone": 1},
            "semantic_group_counts": {"barrier_structure": 1},
        }]
        dropped = subject.cone_expert_samples(rows, 2)[0]
        self.assertEqual({}, dropped["semantic_group_counts"])

    def test_nonpositive_threshold_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            subject.cone_expert_samples([], 0)


if __name__ == "__main__":
    unittest.main()
