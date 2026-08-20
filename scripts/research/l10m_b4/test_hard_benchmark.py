from __future__ import annotations

import unittest

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC

from .certify_hard_benchmark import certify
from .hard_benchmark import legal_neighbors, load_benchmark


class HardBenchmarkTest(unittest.TestCase):
    def test_frozen_instance_set_qualifies_without_model_calls(self) -> None:
        result = certify()

        self.assertEqual(result["model_call_count"], 0)
        self.assertEqual(result["terminal"], "B4_HARD_BENCHMARK_QUALIFIED")
        self.assertEqual(len(result["instances"]), 3)
        self.assertTrue(all(row["qualified"] for row in result["instances"]))

    def test_initial_legal_move_count_matches_balanced_operator_surface(self) -> None:
        self.assertEqual(len(legal_neighbors(INITIAL_SPEC)), 8)
        self.assertEqual(len(load_benchmark()["instances"]), 3)


if __name__ == "__main__":
    unittest.main()
