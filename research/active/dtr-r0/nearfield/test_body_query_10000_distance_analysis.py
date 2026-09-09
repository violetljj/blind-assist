"""Focused CPU tests for the distance comparison math helpers."""
import unittest

import numpy as np

from body_query_10000_distance_analysis import (
    binary_confusion,
    direction,
    nonempty_recall,
    paired_correctness,
)


class DistanceAnalysisMathTests(unittest.TestCase):
    def test_direction_uses_frozen_tolerance_and_keeps_ties(self):
        self.assertEqual(direction(2e-6), "far_higher")
        self.assertEqual(direction(-2e-6), "near_higher")
        self.assertEqual(direction(1e-6), "tie")
        self.assertEqual(direction(-1e-6), "tie")
        self.assertEqual(direction(0.0), "tie")

    def test_binary_confusion_inclusive_threshold_counts_all_cells(self):
        predicted = np.array([[True, False], [True, False]])
        truth = np.array([[True, True], [False, False]])
        self.assertEqual(binary_confusion(predicted, truth), {"TP": 1, "FP": 1, "FN": 1, "TN": 1})

    def test_nonempty_recall_uses_query_cells_not_frame_any(self):
        probability = np.array([[0.51, 0.49, 0.80], [0.20, 0.90, 0.50]])
        counts = np.array([[1, 0, 0], [0, 2, 0]])
        result = nonempty_recall(probability, counts)
        self.assertEqual(result["positives"], 2)
        self.assertEqual(result["TP"], 2)
        self.assertEqual(result["FN"], 0)
        self.assertEqual(result["recall"], 1.0)

    def test_paired_correctness_keeps_both_wrong_and_ties(self):
        old = np.array([True, True, False, False])
        new = np.array([True, False, True, False])
        self.assertEqual(
            paired_correctness(old, new),
            {"old_only_correct": 1, "new_only_correct": 1, "both_correct": 1, "both_wrong": 1},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
