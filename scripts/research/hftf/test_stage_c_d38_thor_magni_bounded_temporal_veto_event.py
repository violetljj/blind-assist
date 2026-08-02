#!/usr/bin/env python3
"""Tests for D38 adaptive bounded temporal veto evaluation."""

from __future__ import annotations

import unittest

from evaluate_stage_c_d38_thor_magni_bounded_temporal_veto_event import (
    SUPPORTED_STATUS,
    build_gate,
)


class D38BoundedTemporalVetoEventTests(unittest.TestCase):
    def test_latch_opportunity_is_an_evaluability_gate(self) -> None:
        pooled = {
            "baseline_positive_alerted": 30,
            "baseline_negative_alerted": 50,
            "positive_event_losses": 0,
            "positive_anchor_losses": 0,
            "positive_anchor_recall_delta": 0.0,
            "negative_alert_reduction": 12,
            "negative_alert_relative_reduction": 0.24,
            "candidate_only_triggered_windows": 0,
        }
        rows = [
            {
                "source_session_id": f"s{index % 19}",
                "admitted_contradict_frames": 1,
                "latch_only_suppressed_frames": int(index < 9),
                "raw_risk_mismatches": 0,
                "stable_risk_mismatches": 0,
                "non_scene_source_observations": 0,
            }
            for index in range(530)
        ]
        receipt = {
            "anchor_count_mismatches": 0,
            "anchor_mask_mismatches": 0,
            "maximum_anchor_slot_error": 0.0,
            "anchor_parity_tolerance": 1e-5,
        }
        folds = [
            {"negative_alert_reduction": value}
            for value in (1, 1, 1, 0, 0)
        ]
        evaluability, support, status = build_gate(
            pooled,
            rows,
            folds,
            receipt,
        )
        self.assertFalse(
            evaluability["latch_only_suppression_opportunity"]
        )
        self.assertTrue(all(support.values()))
        self.assertTrue(status.endswith("_NOT_EVALUABLE"))

    def test_supported_status_requires_all_frozen_gates(self) -> None:
        pooled = {
            "baseline_positive_alerted": 30,
            "baseline_negative_alerted": 50,
            "positive_event_losses": 0,
            "positive_anchor_losses": 0,
            "positive_anchor_recall_delta": 0.0,
            "negative_alert_reduction": 12,
            "negative_alert_relative_reduction": 0.24,
            "candidate_only_triggered_windows": 0,
        }
        rows = [
            {
                "source_session_id": f"s{index % 19}",
                "admitted_contradict_frames": 1,
                "latch_only_suppressed_frames": int(index < 20),
                "raw_risk_mismatches": 0,
                "stable_risk_mismatches": 0,
                "non_scene_source_observations": 0,
            }
            for index in range(530)
        ]
        receipt = {
            "anchor_count_mismatches": 0,
            "anchor_mask_mismatches": 0,
            "maximum_anchor_slot_error": 0.0,
            "anchor_parity_tolerance": 1e-5,
        }
        folds = [
            {"negative_alert_reduction": value}
            for value in (1, 1, 1, 0, 0)
        ]
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
