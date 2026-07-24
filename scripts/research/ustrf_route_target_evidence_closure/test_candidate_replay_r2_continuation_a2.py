#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

from candidate_replay_r2_continuation_a2 import (
    MINIMUM_AVAILABLE_MEMORY_BYTES_A2,
    short_trace_paths,
)


class CandidateReplayContinuationA2Tests(unittest.TestCase):
    def test_user_authorized_guard_is_four_gib(self) -> None:
        self.assertEqual(MINIMUM_AVAILABLE_MEMORY_BYTES_A2, 4 * 1024**3)

    def test_short_path_uses_hash_not_sequence_name(self) -> None:
        descriptor = {
            "source_id": "crowdbot_0410_shared_control",
            "sequence_id": "defaced_" + "very-long-" * 30,
        }
        attempts, authority = short_trace_paths(
            Path("artifacts.local/r2a2"), "C1_CAUSAL_ROUTE_RELATION_FSM", descriptor
        )
        self.assertNotIn("very-long", str(attempts))
        self.assertEqual(len(authority.parent.name), 24)

    def test_short_path_is_deterministic(self) -> None:
        descriptor = {"source_id": "s", "sequence_id": "q"}
        first = short_trace_paths(Path("root"), "C1", descriptor)
        second = short_trace_paths(Path("root"), "C1", descriptor)
        self.assertEqual(first, second)

    def test_source_identity_changes_hash(self) -> None:
        first = short_trace_paths(
            Path("root"), "C1", {"source_id": "s1", "sequence_id": "q"}
        )
        second = short_trace_paths(
            Path("root"), "C1", {"source_id": "s2", "sequence_id": "q"}
        )
        self.assertNotEqual(first, second)

    def test_candidate_namespaces_remain_isolated(self) -> None:
        descriptor = {"source_id": "s", "sequence_id": "q"}
        first = short_trace_paths(Path("root"), "C1", descriptor)
        second = short_trace_paths(Path("root"), "C2", descriptor)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
