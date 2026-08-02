#!/usr/bin/env python3
"""Tests for D36 source windows and paired event evaluation."""

from __future__ import annotations

import unittest

from evaluate_stage_c_d36_thor_magni_track_veto_event import (
    build_gate,
    positive_event_groups,
    summarize,
)
from produce_stage_c_d36_thor_magni_track_veto_input import (
    window_frames,
)


class D36TrackVetoEventTests(unittest.TestCase):
    def test_window_uses_seven_causal_frames_near_fifteen_hz(self) -> None:
        step, frames = window_frames(100, 29.97)
        self.assertEqual(2, step)
        self.assertEqual([88, 90, 92, 94, 96, 98, 100], frames)

    def test_positive_events_do_not_cross_gap(self) -> None:
        rows = [
            {"positive": True, "anchor_scene_frame": 10},
            {"positive": True, "anchor_scene_frame": 40},
            {"positive": True, "anchor_scene_frame": 100},
            {"positive": False, "anchor_scene_frame": 110},
        ]
        self.assertEqual(
            [[10, 40], [100]],
            [
                [row["anchor_scene_frame"] for row in group]
                for group in positive_event_groups(rows)
            ],
        )

    def test_summarize_preserves_paired_losses_and_reductions(self) -> None:
        rows = [
            self.row("s", 10, True, True, True),
            self.row("s", 20, True, True, False),
            self.row("s", 100, False, True, False),
            self.row("s", 130, False, True, True),
        ]
        summary = summarize(rows)
        self.assertEqual(1, summary["positive_events"])
        self.assertEqual(1, summary["positive_anchor_losses"])
        self.assertEqual(0, summary["positive_event_losses"])
        self.assertEqual(1, summary["negative_alert_reduction"])
        self.assertEqual(0.5, summary["negative_alert_relative_reduction"])

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
                "non_track_admitted_evidence": 0,
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
        # Cohort size/session count are deliberately insufficient here.
        evaluability, support, status = build_gate(
            pooled,
            rows,
            folds,
            receipt,
        )
        self.assertFalse(evaluability["complete_cohort"])
        self.assertTrue(all(support.values()))
        self.assertTrue(status.endswith("_NOT_EVALUABLE"))

    @staticmethod
    def row(
        source: str,
        frame: int,
        positive: bool,
        baseline: bool,
        candidate: bool,
    ) -> dict[str, object]:
        return {
            "source_session_id": source,
            "anchor_scene_frame": frame,
            "positive": positive,
            "baseline_any_triggered": baseline,
            "candidate_any_triggered": candidate,
            "candidate_only_triggered_window": candidate and not baseline,
            "candidate_only_triggered_frames": 0,
            "suppressed_frames": 0,
            "admitted_contradict_frames": 0,
            "admitted_confirm_frames": 0,
        }


if __name__ == "__main__":
    unittest.main()
