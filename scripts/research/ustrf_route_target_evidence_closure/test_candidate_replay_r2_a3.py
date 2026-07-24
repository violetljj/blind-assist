#!/usr/bin/env python3
from __future__ import annotations

import unittest

from finalize_candidate_replay_r2_a3 import build_terminal


class CandidateReplayFinalizationA3Tests(unittest.TestCase):
    def test_final_terminal_keeps_every_authority_closed(self) -> None:
        parent_terminal = {"terminal_state": "CANDIDATE_REPLAY_COMPLETE"}
        parent_validation = {"status": "VALID"}
        bindings = {
            "parent_terminal_sha256": "a" * 64,
            "parent_validation_sha256": "b" * 64,
        }
        receipt = build_terminal({}, bindings, parent_terminal, parent_validation)
        self.assertTrue(all(value is False for value in receipt["claim_boundary"].values()))
        self.assertEqual(receipt["profiles"], {"generated": False, "count": 0, "authority": False})

    def test_final_scope_is_exact_cartesian_product(self) -> None:
        receipt = build_terminal(
            {},
            {
                "parent_terminal_sha256": "a" * 64,
                "parent_validation_sha256": "b" * 64,
            },
            {"terminal_state": "CANDIDATE_REPLAY_COMPLETE"},
            {"status": "VALID"},
        )
        scope = receipt["verified_scope"]
        self.assertEqual(scope["candidate_count"] * scope["sequence_ledgers"], 123)
        self.assertEqual(scope["authoritative_trace_frames"], 3 * 62229)
        self.assertEqual(scope["authoritative_trace_resets"], 3 * 15)

    def test_four_gib_amendment_is_retained(self) -> None:
        receipt = build_terminal(
            {},
            {
                "parent_terminal_sha256": "a" * 64,
                "parent_validation_sha256": "b" * 64,
            },
            {"terminal_state": "CANDIDATE_REPLAY_COMPLETE"},
            {"status": "VALID"},
        )
        self.assertEqual(
            receipt["verified_scope"]["minimum_available_memory_bytes"], 4 * 1024**3
        )


if __name__ == "__main__":
    unittest.main()
