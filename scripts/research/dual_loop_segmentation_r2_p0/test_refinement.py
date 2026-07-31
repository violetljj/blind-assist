from __future__ import annotations

import unittest

from .run_ddrnet_refinement import _candidate_id, _gate


class RefinementTests(unittest.TestCase):
    def test_candidate_identity_is_order_independent(self) -> None:
        left = _candidate_id({"area": 8, "confidence": 0.5})
        right = _candidate_id({"confidence": 0.5, "area": 8})
        self.assertEqual(left, right)

    def test_gate_boundaries_are_inclusive(self) -> None:
        self.assertTrue(_gate(0.06, ">=", 0.06)["passed"])
        self.assertTrue(_gate(2.5, "<=", 2.5)["passed"])
        self.assertFalse(_gate(2.51, "<=", 2.5)["passed"])


if __name__ == "__main__":
    unittest.main()
