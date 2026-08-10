#!/usr/bin/env python3

import unittest

from evaluate_ag_st_unified_student_bonn_boundary import parent_macro


class UnifiedStudentBonnBoundaryTest(unittest.TestCase):
    def test_parent_macro_is_unweighted_across_parents(self) -> None:
        first = {
            "student_average_precision": 0.2,
            "precision_within_tolerance": 0.4,
            "recall_within_tolerance": 0.6,
            "f1_within_tolerance": 0.5,
        }
        second = {
            "student_average_precision": 0.6,
            "precision_within_tolerance": 0.8,
            "recall_within_tolerance": 0.2,
            "f1_within_tolerance": 0.3,
        }
        value = parent_macro({"p1": first, "p2": second})
        self.assertAlmostEqual(0.4, value["student_average_precision"])
        self.assertAlmostEqual(0.6, value["precision_within_tolerance"])
        self.assertAlmostEqual(0.4, value["recall_within_tolerance"])
        self.assertAlmostEqual(0.4, value["f1_within_tolerance"])


if __name__ == "__main__":
    unittest.main()
