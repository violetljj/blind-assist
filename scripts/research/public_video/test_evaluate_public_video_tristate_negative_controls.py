#!/usr/bin/env python3
"""Pure tests for tri-state nuisance negative-control evaluation."""

from __future__ import annotations

import unittest

import evaluate_public_video_tristate_negative_controls as subject
import test_public_video_tristate_contract as contract_test


def sample(timestamp_ms: int, class_name: str | None = None) -> dict[str, object]:
    active = class_name is not None
    return {
        "timestamp_ms": timestamp_ms,
        "semantic_group_counts": {"barrier_structure": 1} if active else {},
        "semantic_class_counts": {class_name: 1} if active else {},
    }


class TristateNegativeControlsTest(unittest.TestCase):
    def test_isolated_spike_passes(self) -> None:
        source = {
            "source_id": "isolated",
            "samples": [sample(0), sample(1000, "traffic cone"), sample(2000)],
        }
        result = subject.evaluate_source(source, contract_test.contract_fixture())
        self.assertTrue(result["passed"])
        self.assertEqual({"traffic cone": 1}, result["active_class_counts"])

    def test_sustained_false_event_fails(self) -> None:
        source = {
            "source_id": "false-event",
            "samples": [
                sample(0, "barricade"),
                sample(1000, "barricade"),
                sample(2000),
                sample(3000),
                sample(4000),
            ],
        }
        result = subject.evaluate_source(source, contract_test.contract_fixture())
        self.assertFalse(result["passed"])
        self.assertEqual([], result["lifecycle"]["intervals"])
        self.assertEqual("uncertain", result["lifecycle"]["terminal_state"])

    def test_terminal_open_event_fails(self) -> None:
        source = {
            "source_id": "terminal-open",
            "samples": [
                sample(0, "traffic cone"),
                sample(1000, "traffic cone"),
                sample(2000, "traffic cone"),
            ],
        }
        result = subject.evaluate_source(source, contract_test.contract_fixture())
        self.assertFalse(result["passed"])
        self.assertEqual("present", result["lifecycle"]["terminal_state"])


if __name__ == "__main__":
    unittest.main()
