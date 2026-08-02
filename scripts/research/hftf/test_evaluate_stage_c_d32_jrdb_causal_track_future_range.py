#!/usr/bin/env python3
"""Tests for D32 causal-track future-range evaluation."""

from __future__ import annotations

import math
import unittest

from evaluate_stage_c_d32_jrdb_causal_track_future_range import (
    MINIMUM_DIRECTION_EVIDENCE,
    MINIMUM_DISTINCT_TRACKS,
    MINIMUM_SEQUENCES_WITH_EVIDENCE,
    MINIMUM_TOTAL_EVIDENCE,
    determine_terminal,
    source_decision,
)


def history(heights: list[float]) -> list[dict[str, object]]:
    return [
        {
            "frame_index": index,
            "timestamp_ns": index * 100_000_000,
            "height": height,
        }
        for index, height in enumerate(heights)
    ]


class D32CausalTrackFutureRangeTests(unittest.TestCase):
    def test_source_decision_preserves_frozen_monotonic_rule(self) -> None:
        confirm, confirm_slope = source_decision(
            history([10, 11, 12, 13, 14, 15, 16])
        )
        contradict, contradict_slope = source_decision(
            history([16, 15, 14, 13, 12, 11, 10])
        )
        abstain, _ = source_decision(
            history([10, 11, 12, 11, 14, 15, 16])
        )
        self.assertEqual(confirm, "CONFIRM_APPROACH")
        self.assertGreaterEqual(confirm_slope, 0.2)
        self.assertEqual(contradict, "CONTRADICT_APPROACH")
        self.assertLessEqual(contradict_slope, -0.2)
        self.assertEqual(abstain, "ABSTAIN")

    def test_non_contiguous_history_is_engineering_error(self) -> None:
        rows = history([10, 11, 12, 13, 14, 15, 16])
        rows[-1]["frame_index"] = 7
        with self.assertRaisesRegex(ValueError, "not frame-contiguous"):
            source_decision(rows)

    def test_insufficient_evidence_is_not_evaluable(self) -> None:
        terminal, evaluable, supported = determine_terminal(
            evidence_rows=MINIMUM_TOTAL_EVIDENCE - 1,
            distinct_tracks=MINIMUM_DISTINCT_TRACKS,
            sequences_with_evidence=MINIMUM_SEQUENCES_WITH_EVIDENCE,
            confirm_rows=MINIMUM_DIRECTION_EVIDENCE,
            contradict_rows=MINIMUM_DIRECTION_EVIDENCE,
            effect_gates=[True],
        )
        self.assertFalse(evaluable)
        self.assertFalse(supported)
        self.assertTrue(terminal.endswith("NOT_EVALUABLE"))

    def test_evaluable_effect_failure_is_scientific_negative(self) -> None:
        terminal, evaluable, supported = determine_terminal(
            evidence_rows=MINIMUM_TOTAL_EVIDENCE,
            distinct_tracks=MINIMUM_DISTINCT_TRACKS,
            sequences_with_evidence=MINIMUM_SEQUENCES_WITH_EVIDENCE,
            confirm_rows=MINIMUM_DIRECTION_EVIDENCE,
            contradict_rows=MINIMUM_DIRECTION_EVIDENCE,
            effect_gates=[True, False],
        )
        self.assertTrue(evaluable)
        self.assertFalse(supported)
        self.assertTrue(terminal.endswith("NOT_SUPPORTED"))

    def test_source_slope_is_finite(self) -> None:
        _, slope = source_decision(
            history([10, 11, 12, 13, 14, 15, 16])
        )
        self.assertTrue(math.isfinite(slope))


if __name__ == "__main__":
    unittest.main()
