#!/usr/bin/env python3
"""Tests for D39 symmetric confirm-release event evaluation."""

from __future__ import annotations

import unittest

from evaluate_stage_c_d39_thor_magni_confirm_release_veto_event import (
    SUPPORTED_STATUS,
    build_gate,
)


class D39ConfirmReleaseVetoEventTests(unittest.TestCase):
    def base_inputs(
        self,
        confirm_rows: int,
        release_rows: int,
    ) -> tuple[
        dict[str, object],
        list[dict[str, object]],
        list[dict[str, int]],
        dict[str, object],
    ]:
        pooled: dict[str, object] = {
            "baseline_positive_alerted": 30,
            "baseline_negative_alerted": 50,
            "positive_event_losses": 0,
            "positive_anchor_losses": 0,
            "positive_anchor_recall_delta": 0.0,
            "negative_alert_reduction": 12,
            "negative_alert_relative_reduction": 0.24,
            "candidate_only_triggered_windows": 0,
        }
        rows: list[dict[str, object]] = [
            {
                "source_session_id": f"s{index % 19}",
                "admitted_contradict_frames": 1,
                "admitted_confirm_frames": int(index < confirm_rows),
                "confirm_release_frames": int(index < release_rows),
                "latch_only_suppressed_frames": int(index < 20),
                "raw_risk_mismatches": 0,
                "stable_risk_mismatches": 0,
                "non_scene_source_observations": 0,
            }
            for index in range(530)
        ]
        folds = [
            {"negative_alert_reduction": value}
            for value in (1, 1, 1, 0, 0)
        ]
        receipt: dict[str, object] = {
            "anchor_count_mismatches": 0,
            "anchor_mask_mismatches": 0,
            "maximum_anchor_slot_error": 0.0,
            "anchor_parity_tolerance": 1e-5,
        }
        return pooled, rows, folds, receipt

    def test_confirm_release_is_required_for_evaluability(self) -> None:
        pooled, rows, folds, receipt = self.base_inputs(20, 9)
        evaluability, support, status = build_gate(
            pooled,
            rows,
            folds,
            receipt,
        )
        self.assertTrue(evaluability["confirm_opportunity"])
        self.assertFalse(evaluability["confirm_releases_live_latch"])
        self.assertTrue(all(support.values()))
        self.assertTrue(status.endswith("_NOT_EVALUABLE"))

    def test_supported_status_requires_all_frozen_gates(self) -> None:
        pooled, rows, folds, receipt = self.base_inputs(20, 20)
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
