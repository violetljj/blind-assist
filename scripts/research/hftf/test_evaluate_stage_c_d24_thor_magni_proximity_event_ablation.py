#!/usr/bin/env python3
"""Tests for the D24 THOR-MAGNI event input ablation."""

from __future__ import annotations

import unittest

import numpy as np

from evaluate_stage_c_d24_thor_magni_proximity_event_ablation import (
    build_gate,
    operating_threshold,
    positive_event_groups,
    summarize_delta,
)


def record(frame: int, positive: bool) -> dict:
    return {
        "anchor_scene_frame": frame,
        "future_onset_target": {"proximity_onset": positive},
    }


class D24EventAblationTests(unittest.TestCase):
    def test_operating_threshold_respects_false_active_cap(self) -> None:
        negative = np.arange(20, dtype=np.float64)
        threshold = operating_threshold(negative, 0.10)
        self.assertLessEqual(float(np.mean(negative > threshold)), 0.10)
        self.assertEqual(int(np.sum(negative > threshold)), 2)

    def test_event_grouping_does_not_cross_missing_anchor(self) -> None:
        groups = positive_event_groups(
            [
                record(30, True),
                record(60, True),
                record(90, False),
                record(120, True),
            ]
        )
        self.assertEqual(groups, [[0, 1], [3]])

    def test_summary_and_supported_gate(self) -> None:
        units = []
        for fold in range(5):
            for seed in (17, 23, 41):
                units.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "history_minus_zero_dynamics": {
                            "event_auroc": 0.02,
                            "event_recall_at_false_active_cap": 0.03,
                            "anchor_recall_at_false_active_cap": 0.01,
                            "lead_time_credit_seconds": 0.02,
                        },
                    }
                )
        aggregate = {
            metric: summarize_delta(units, metric)
            for metric in (
                "event_auroc",
                "event_recall_at_false_active_cap",
                "anchor_recall_at_false_active_cap",
                "lead_time_credit_seconds",
            )
        }
        self.assertTrue(build_gate(aggregate)["supported"])


if __name__ == "__main__":
    unittest.main()
