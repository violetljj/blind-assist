#!/usr/bin/env python3
"""Pure tests for the external tri-state lifecycle challenge."""

from __future__ import annotations

import unittest

import evaluate_public_video_tristate_lifecycle_external_challenge as subject


def sample(timestamp_ms: int, active: bool) -> dict[str, object]:
    return {
        "timestamp_ms": timestamp_ms,
        "semantic_group_counts": {"barrier_structure": 1} if active else {},
    }


class TristateLifecycleChallengeTest(unittest.TestCase):
    def test_two_sample_dropout_stays_inside_one_event(self) -> None:
        rows = [sample(index * 1000, value) for index, value in enumerate(
            [True, True, True, False, False, True, True, False, False, False]
        )]
        report = subject.tristate_exit_intervals(
            rows, ["barrier_structure"]
        )
        self.assertEqual(1, len(report["intervals"]))
        interval = report["intervals"][0]
        self.assertEqual(6000, interval["last_active_timestamp_ms"])
        self.assertEqual(7000, interval["first_absent_timestamp_ms"])
        self.assertEqual(9000, interval["confirmed_clear_timestamp_ms"])

    def test_single_spike_cannot_open_event(self) -> None:
        rows = [sample(index * 1000, value) for index, value in enumerate(
            [False, False, True, False, False, False]
        )]
        report = subject.tristate_exit_intervals(
            rows, ["barrier_structure"]
        )
        self.assertEqual([], report["intervals"])
        self.assertIsNone(report["open_event"])
        self.assertEqual("clear", report["terminal_state"])

    def test_terminal_present_exposes_open_event(self) -> None:
        rows = [sample(index * 1000, True) for index in range(4)]
        report = subject.tristate_exit_intervals(rows, ["barrier_structure"])
        self.assertEqual("present", report["terminal_state"])
        self.assertEqual(0, report["open_event"]["event_entry_timestamp_ms"])
        self.assertEqual(3000, report["open_event"]["last_active_timestamp_ms"])

    def test_reappearance_cancels_uncertain_exit(self) -> None:
        rows = [sample(index * 1000, value) for index, value in enumerate(
            [True, True, False, True, False, False, False]
        )]
        report = subject.tristate_exit_intervals(
            rows, ["barrier_structure"]
        )
        self.assertEqual(1, len(report["intervals"]))
        self.assertEqual(3000, report["intervals"][0]["last_active_timestamp_ms"])

    def test_reference_inside_only_interval_passes(self) -> None:
        score = subject.score_intervals([{
            "last_active_timestamp_ms": 177000,
            "first_absent_timestamp_ms": 178000,
            "confirmed_clear_timestamp_ms": 180000,
        }], {
            "present_timestamp_ms": 178000,
            "absent_timestamp_ms": 179000,
        })
        self.assertTrue(score["passed"])


if __name__ == "__main__":
    unittest.main()
