#!/usr/bin/env python3
from __future__ import annotations

import unittest

from validate_candidate_replay_r2_memory_guard_a4 import require_memory


class CandidateReplayMemoryGuardA4Tests(unittest.TestCase):
    def test_four_gib_exactly_passes(self) -> None:
        require_memory(4 * 1024**3, 4 * 1024**3)

    def test_one_byte_below_four_gib_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "available_memory_guard"):
            require_memory(4 * 1024**3 - 1, 4 * 1024**3)

    def test_observation_above_four_gib_passes(self) -> None:
        require_memory(6 * 1024**3, 4 * 1024**3)


if __name__ == "__main__":
    unittest.main()
