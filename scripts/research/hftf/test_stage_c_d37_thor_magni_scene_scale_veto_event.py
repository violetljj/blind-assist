#!/usr/bin/env python3
"""Tests for the frozen D37 scene-scale event evaluator."""

from __future__ import annotations

import unittest

from evaluate_stage_c_d37_thor_magni_scene_scale_veto_event import (
    build_gate,
    is_supported_status,
)


class D37SceneScaleVetoEventTests(unittest.TestCase):
    def test_gate_separates_evaluability_from_effect(self) -> None:
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
                "source_session_id": f"s{index % 5}",
                "admitted_contradict_frames": 1,
                "raw_risk_mismatches": 0,
                "stable_risk_mismatches": 0,
                "non_scene_source_observations": 0,
            }
            for index in range(10)
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
        self.assertFalse(evaluability["complete_cohort"])
        self.assertTrue(evaluability["contradict_opportunity"])
        self.assertTrue(all(support.values()))
        self.assertTrue(status.endswith("_NOT_EVALUABLE"))

    def test_opportunity_requires_five_sessions(self) -> None:
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
                "source_session_id": f"s{index % 4}",
                "admitted_contradict_frames": 1,
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
        evaluability, _, status = build_gate(
            pooled,
            rows,
            folds,
            receipt,
        )
        self.assertFalse(evaluability["complete_cohort"])
        self.assertFalse(evaluability["contradict_opportunity"])
        self.assertTrue(status.endswith("_NOT_EVALUABLE"))

    def test_not_supported_status_is_not_supported_boolean(self) -> None:
        self.assertFalse(
            is_supported_status(
                "D37_THOR_MAGNI_PRODUCTION_SCENE_SCALE_VETO_EVENT_"
                "NOT_SUPPORTED"
            )
        )
        self.assertTrue(
            is_supported_status(
                "D37_THOR_MAGNI_PRODUCTION_SCENE_SCALE_VETO_EVENT_SUPPORTED"
            )
        )


if __name__ == "__main__":
    unittest.main()
