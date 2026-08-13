from __future__ import annotations

import unittest

import numpy as np

from scripts.research.assistive_geometry.run_ag_obstacle_evidence_tristate_calibration_canary import (
    choose_high_threshold,
    choose_low_threshold,
    gate_folds,
    threshold_stats,
)


class ObstacleEvidenceTristateCalibrationCanaryTest(unittest.TestCase):
    def test_threshold_stats_keeps_middle_unknown(self) -> None:
        scores = np.asarray([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float32)
        truth = np.asarray([False, False, True, True, True])
        result = threshold_stats(scores, truth, 0.2, 0.8)
        self.assertEqual(result["verified_negative_count"], 2)
        self.assertEqual(result["predicted_positive_count"], 2)
        self.assertEqual(result["unknown_count"], 1)
        self.assertEqual(result["false_negative_count"], 0)
        self.assertEqual(result["false_positive_count"], 0)

    def test_threshold_selection_is_parent_worst_case_bounded(self) -> None:
        parents = {
            name: {
                "scores": np.asarray([0.05, 0.10, 0.90, 0.95], dtype=np.float32),
                "truth": np.asarray([False, False, True, True]),
            }
            for name in ("a", "b", "c", "d")
        }
        low, _ = choose_low_threshold(parents)
        high, _ = choose_high_threshold(parents)
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertLess(low, high)

    def test_fold_gate_rejects_single_parent_false_negative(self) -> None:
        good = {
            "false_negative_rate": 0.0,
            "false_positive_rate": 0.0,
            "positive_recall": 0.5,
            "verified_negative_coverage": 0.5,
            "known_coverage": 0.5,
        }
        folds = [
            {
                "held_parent": f"p{index}",
                "held_metrics": dict(good),
            }
            for index in range(6)
        ]
        folds[-1]["held_metrics"]["false_negative_rate"] = 0.02
        gate = gate_folds(folds)
        self.assertFalse(gate["checks"]["all_parents_false_negative_rate_le_0p01"])
        self.assertFalse(gate["pass"])


if __name__ == "__main__":
    unittest.main()
