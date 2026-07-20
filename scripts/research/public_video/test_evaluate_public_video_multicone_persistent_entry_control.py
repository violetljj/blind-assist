#!/usr/bin/env python3
"""Pure tests for persistent-entry multi-cone control scoring."""

from __future__ import annotations

import unittest

import evaluate_public_video_multicone_persistent_entry_control as subject


class MulticonePersistentEntryControlTest(unittest.TestCase):
    def test_open_present_event_inside_window_passes(self) -> None:
        result = {
            "intervals": [],
            "terminal_state": "present",
            "open_event": {"event_entry_timestamp_ms": 21000},
        }
        self.assertTrue(
            subject.score_persistent_entry(result, [20000, 22000])["passed"]
        )

    def test_missing_open_event_fails(self) -> None:
        result = {"intervals": [], "terminal_state": "clear", "open_event": None}
        self.assertFalse(
            subject.score_persistent_entry(result, [20000, 22000])["passed"]
        )

    def test_open_uncertain_event_remains_persistent(self) -> None:
        result = {
            "intervals": [],
            "terminal_state": "uncertain",
            "open_event": {"event_entry_timestamp_ms": 21000},
        }
        self.assertTrue(
            subject.score_persistent_entry(result, [20000, 22000])["passed"]
        )

    def test_premature_completed_exit_fails(self) -> None:
        result = {
            "intervals": [{"confirmed_clear_timestamp_ms": 30000}],
            "terminal_state": "clear",
            "open_event": None,
        }
        self.assertFalse(
            subject.score_persistent_entry(result, [20000, 22000])["passed"]
        )


if __name__ == "__main__":
    unittest.main()
