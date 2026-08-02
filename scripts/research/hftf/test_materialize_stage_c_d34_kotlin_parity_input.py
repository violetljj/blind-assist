#!/usr/bin/env python3
"""Tests for D34 Kotlin parity input materialization."""

from __future__ import annotations

import unittest

from evaluate_stage_c_d32_jrdb_causal_track_future_range import (
    source_decision,
)


class D34KotlinParityInputTests(unittest.TestCase):
    def test_reference_decision_uses_contiguous_seven_frame_history(
        self,
    ) -> None:
        history = [
            {
                "frame_index": index,
                "timestamp_ns": index * 100_000_000,
                "height": 100.0 + 5.0 * index,
            }
            for index in range(7)
        ]
        decision, slope = source_decision(history)
        self.assertEqual(decision, "CONFIRM_APPROACH")
        self.assertGreaterEqual(slope, 0.2)


if __name__ == "__main__":
    unittest.main()
