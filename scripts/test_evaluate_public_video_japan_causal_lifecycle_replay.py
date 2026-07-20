#!/usr/bin/env python3
"""Pure tests for causal Japan lifecycle replay helpers."""

from __future__ import annotations

import unittest

import evaluate_public_video_japan_causal_lifecycle_replay as subject


class JapanCausalLifecycleReplayTest(unittest.TestCase):
    def test_first_supported_entry_requires_both_signals(self) -> None:
        rows = [
            {"timestamp_ms": 1, "radial_prefix_passed": False, "route_intrusion_score": 1.0},
            {"timestamp_ms": 2, "radial_prefix_passed": True, "route_intrusion_score": 0.2},
            {"timestamp_ms": 3, "radial_prefix_passed": True, "route_intrusion_score": 0.6},
        ]
        self.assertEqual(subject.first_supported_entry(rows, 0.4), 3)

    def test_first_supported_entry_returns_none_without_joint_support(self) -> None:
        rows = [{"timestamp_ms": 1, "radial_prefix_passed": True, "route_intrusion_score": 0.4}]
        self.assertIsNone(subject.first_supported_entry(rows, 0.4))


if __name__ == "__main__":
    unittest.main()
