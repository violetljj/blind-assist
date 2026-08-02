#!/usr/bin/env python3
"""Tests for D40 continuous track-projected risk evaluation."""

from __future__ import annotations

import unittest

from evaluate_stage_c_d40_thor_magni_continuous_track_projected_risk import (
    SUPPORTED_STATUS,
    build_gate,
)


class D40ContinuousTrackProjectedRiskTests(unittest.TestCase):
    def test_pareto_support_requires_meaningful_gain(self) -> None:
        pooled = {
            "baseline_positive_alerted": 30,
            "baseline_negative_alerted": 50,
            "positive_event_hit_delta": 0,
            "positive_anchor_recall_delta": 0.0,
            "negative_alert_delta": 0,
            "candidate_only_negative_windows": 0,
            "positive_event_gains": 0,
            "negative_alert_reduction": 0,
        }
        rows = [
            {
                "source_session_id": f"s{index % 19}",
                "forecast_frames": int(index < 100),
                "segment_count": 1,
                "maximum_absolute_slope": 0.2,
            }
            for index in range(530)
        ]
        folds = [
            {
                "positive_event_hit_delta": 0,
                "negative_alert_delta": 0,
                "negative_alert_reduction": 0,
            }
            for _ in range(5)
        ]
        receipt = {
            "unique_requested_frames": 3_710,
            "detection_rows": 14_364,
            "anchor_count_mismatches": 0,
            "anchor_mask_mismatches": 0,
            "maximum_anchor_slot_error": 0.0,
            "anchor_parity_tolerance": 1e-5,
        }
        evaluability, support, status = build_gate(
            pooled,
            rows,
            folds,
            receipt,
        )
        self.assertTrue(all(evaluability.values()))
        self.assertFalse(support["meaningful_strict_gain"])
        self.assertTrue(status.endswith("_NOT_SUPPORTED"))

    def test_supported_status_requires_pareto_gain_in_three_folds(self) -> None:
        pooled = {
            "baseline_positive_alerted": 30,
            "baseline_negative_alerted": 50,
            "positive_event_hit_delta": 5,
            "positive_anchor_recall_delta": 0.04,
            "negative_alert_delta": -2,
            "candidate_only_negative_windows": 2,
            "positive_event_gains": 5,
            "negative_alert_reduction": 2,
        }
        rows = [
            {
                "source_session_id": f"s{index % 19}",
                "forecast_frames": int(index < 100),
                "segment_count": 1,
                "maximum_absolute_slope": 0.2,
            }
            for index in range(530)
        ]
        folds = [
            {
                "positive_event_hit_delta": int(index < 3),
                "negative_alert_delta": 0,
                "negative_alert_reduction": 0,
            }
            for index in range(5)
        ]
        receipt = {
            "unique_requested_frames": 3_710,
            "detection_rows": 14_364,
            "anchor_count_mismatches": 0,
            "anchor_mask_mismatches": 0,
            "maximum_anchor_slot_error": 0.0,
            "anchor_parity_tolerance": 1e-5,
        }
        evaluability, support, status = build_gate(
            pooled,
            rows,
            folds,
            receipt,
        )
        self.assertTrue(all(evaluability.values()))
        self.assertTrue(all(support.values()))
        self.assertEqual(SUPPORTED_STATUS, status)


if __name__ == "__main__":
    unittest.main()
