#!/usr/bin/env python3
"""Pure tests for event risk-profile composition."""

from __future__ import annotations

import unittest

import run_public_video_event_risk_profile_lifecycle_gate as subject


class EventRiskProfileLifecycleGateTest(unittest.TestCase):
    def test_entry_requires_radial_and_route_support(self) -> None:
        self.assertEqual(subject.event_row(sample_id="x", label=1, radial_count=1, route_delta=0.2)["predicted_event_alert"], 1)
        self.assertEqual(subject.event_row(sample_id="x", label=0, radial_count=0, route_delta=0.2)["predicted_event_alert"], 0)
        self.assertEqual(subject.event_row(sample_id="x", label=0, radial_count=1, route_delta=0.0)["predicted_event_alert"], 0)

    def test_metrics_are_perfect_for_two_events_and_five_controls(self) -> None:
        rows = [subject.event_row(sample_id=f"p{i}", label=1, radial_count=1, route_delta=0.1) for i in range(2)]
        rows += [subject.event_row(sample_id=f"n{i}", label=0, radial_count=0, route_delta=1.0) for i in range(5)]
        self.assertEqual(subject.metric_rows(rows)["balanced_accuracy"], 1.0)

    def test_passed_event_count_ignores_failed_candidates(self) -> None:
        source = {"events": [{"radial_approach_passed": True}, {"radial_approach_passed": False}, {}]}
        self.assertEqual(subject.passed_event_count(source), 1)


if __name__ == "__main__":
    unittest.main()
