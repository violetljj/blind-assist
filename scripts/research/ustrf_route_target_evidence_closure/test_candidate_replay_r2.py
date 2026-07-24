#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from candidate_replay_r2 import (
    TERMINAL_STATES,
    base_terminal_receipt,
    canonical_bytes,
    trace_paths,
)
from validate_candidate_replay_r2 import forbidden_fragments


class CandidateReplayR2ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "stage": "R2-L1-CANDIDATE-REPLAY-R2",
            "candidate_roster": ["C1", "C2", "C3"],
            "expected_scope": {
                "candidate_ledger_traces": 123,
                "discontinuity_resets": 15,
            },
        }

    def test_terminal_states_are_closed_and_exact(self) -> None:
        self.assertEqual(
            TERMINAL_STATES,
            {
                "CANDIDATE_REPLAY_COMPLETE",
                "FAIL_CLOSED_INPUT_BLOCKED",
                "FAIL_CLOSED_EXECUTION_ABORTED",
            },
        )

    def test_terminal_never_grants_profile_or_comparison_authority(self) -> None:
        receipt = base_terminal_receipt(
            "FAIL_CLOSED_EXECUTION_ABORTED",
            self.config,
            {"config_sha256": "a" * 64},
            [],
            [{"candidate_id": "C1", "frame_count": 1}],
            "bounded stop",
        )
        self.assertFalse(receipt["profiles"]["generated"])
        self.assertFalse(receipt["profiles"]["authority"])
        self.assertFalse(
            receipt["candidate_execution"]["partial_trace_profile_authority"]
        )
        self.assertFalse(
            receipt["candidate_execution"]["candidate_comparison_authority"]
        )

    def test_partial_trace_is_recorded_without_profile_authority(self) -> None:
        receipt = base_terminal_receipt(
            "FAIL_CLOSED_EXECUTION_ABORTED",
            self.config,
            {},
            [],
            [{"candidate_id": "C1", "frame_count": 10}],
            "guard",
        )
        self.assertEqual(
            receipt["candidate_execution"]["authoritative_trace_count"], 1
        )
        self.assertEqual(receipt["profiles"]["count"], 0)

    def test_complete_receipt_still_contains_no_profile(self) -> None:
        traces = [{"candidate_id": "C1", "frame_count": 1}] * 123
        receipt = base_terminal_receipt(
            "CANDIDATE_REPLAY_COMPLETE", self.config, {}, [], traces, None
        )
        self.assertEqual(receipt["candidate_execution"]["authoritative_trace_count"], 123)
        self.assertEqual(receipt["profiles"]["count"], 0)

    def test_forbidden_comparison_fragments_are_detected(self) -> None:
        self.assertEqual(forbidden_fragments({"winner": "C1"}), ["$.winner"])
        self.assertEqual(forbidden_fragments({"nested": [{"safe": False}]}), [])

    def test_trace_paths_keep_attempts_and_authority_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempts, authoritative = trace_paths(
                Path(directory),
                "C1",
                {"source_id": "source", "sequence_id": "sequence"},
            )
            self.assertNotEqual(attempts.parent, authoritative.parent.parent)
            self.assertEqual(authoritative.name, "authoritative-receipt.json")
            self.assertEqual(attempts.name, "attempts")

    def test_canonical_bytes_are_key_order_independent(self) -> None:
        self.assertEqual(canonical_bytes({"b": 2, "a": 1}), canonical_bytes({"a": 1, "b": 2}))

    def test_illegal_terminal_state_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            base_terminal_receipt("PARTIAL_SUCCESS", self.config, {}, [], [], "x")


if __name__ == "__main__":
    unittest.main()
