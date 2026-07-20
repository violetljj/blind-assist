#!/usr/bin/env python3
"""Pure tests for the public-video temporal-reversal builder."""

from __future__ import annotations

import unittest
from pathlib import Path

import build_public_video_temporal_reverse_counterfactual as subject
import run_public_silver_risk_lifecycle_mil_head as mil


class TemporalReverseCounterfactualTest(unittest.TestCase):
    def test_mapping_reverses_source_time_and_assigns_synthetic_time(self) -> None:
        mapping = subject.assign_synthetic_timestamps(
            subject.reversed_timestamp_mapping([1000, 1100, 1200]),
            target_fps=10.0,
        )
        self.assertEqual([1200, 1100, 1000], [row["original_source_timestamp_ms"] for row in mapping])
        self.assertEqual([0, 100, 200], [row["synthetic_timestamp_ms"] for row in mapping])

    def test_mapping_rejects_non_monotonic_source_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            subject.reversed_timestamp_mapping([1000, 1000])

    def test_independent_direction_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "independent model direction"):
            mil.reject_independent_direction(Path("secondary-corridor-causal/reverse.mp4"))


if __name__ == "__main__":
    unittest.main()
