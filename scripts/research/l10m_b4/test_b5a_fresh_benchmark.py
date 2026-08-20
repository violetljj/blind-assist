from __future__ import annotations

import unittest

from .certify_fresh_benchmark import certify
from .fresh_benchmark import load_fresh_benchmark


class B5AFreshBenchmarkTest(unittest.TestCase):
    def test_fresh_cohort_has_three_qualified_five_step_landscapes(self) -> None:
        result = certify()

        self.assertEqual(result["model_call_count"], 0)
        self.assertEqual(result["terminal"], "B5_FRESH_HARDER_COHORT_QUALIFIED")
        self.assertEqual(len(result["instances"]), 3)
        self.assertTrue(all(row["qualified"] for row in result["instances"]))
        self.assertTrue(
            all(row["shortest_strict_steps_to_global_optimum"] >= 5 for row in result["instances"])
        )

    def test_instance_identities_do_not_overlap_b4(self) -> None:
        ids = {row["instance_id"] for row in load_fresh_benchmark()["instances"]}

        self.assertEqual(ids, {"obsidian", "coral", "silver"})
        self.assertTrue(ids.isdisjoint({"amber", "cobalt", "jade"}))


if __name__ == "__main__":
    unittest.main()
