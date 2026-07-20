#!/usr/bin/env python3
"""Pure tests for the external public-video exit challenge evaluator."""

from __future__ import annotations

import unittest

import evaluate_public_video_exit_router_external_challenge as subject


def sample(timestamp_ms: int, *groups: str) -> dict[str, object]:
    return {
        "timestamp_ms": timestamp_ms,
        "semantic_group_counts": {group: 1 for group in groups},
    }


class ExternalExitChallengeTest(unittest.TestCase):
    def test_activity_window_reports_sparse_risk_evidence(self) -> None:
        rows = [
            sample(0, "barrier_structure"),
            sample(1000),
            sample(2000),
            sample(3000, "barrier_structure"),
        ]
        report = subject.activity_window_diagnostics(
            rows, ["barrier_structure"], [0, 3000]
        )
        self.assertEqual(4, report["sample_count"])
        self.assertEqual(2, report["active_sample_count"])
        self.assertEqual(0.5, report["active_fraction"])
        self.assertEqual(2, report["longest_absent_run_samples"])

    def test_activity_window_is_inclusive_and_group_scoped(self) -> None:
        rows = [
            sample(0, "surface_material"),
            sample(1000, "barrier_structure"),
            sample(2000, "barrier_structure"),
            sample(3000, "surface_material"),
        ]
        report = subject.activity_window_diagnostics(
            rows, ["barrier_structure"], [1000, 2000]
        )
        self.assertEqual(2, report["sample_count"])
        self.assertEqual(2, report["active_sample_count"])
        self.assertEqual(2, report["longest_active_run_samples"])
        self.assertEqual(0, report["longest_absent_run_samples"])

    def test_empty_activity_window_is_explicit(self) -> None:
        report = subject.activity_window_diagnostics(
            [sample(0)], ["barrier_structure"], [1000, 2000]
        )
        self.assertEqual(0, report["sample_count"])
        self.assertIsNone(report["active_fraction"])

    def test_invalid_window_order_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            subject.activity_window_diagnostics(
                [sample(0)], ["barrier_structure"], [2000, 1000]
            )


if __name__ == "__main__":
    unittest.main()
