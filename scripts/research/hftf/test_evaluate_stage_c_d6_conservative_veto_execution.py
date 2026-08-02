import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d6_conservative_veto_execution import (
    execution_stats,
    zero_training_true_veto_threshold,
)


class ConservativeVetoExecutionTest(unittest.TestCase):
    def test_threshold_vetoes_no_calibration_true_alert(self):
        probability = np.asarray(
            [0.1, 0.7, 0.8, 0.9],
            dtype=np.float32,
        )
        target = np.asarray(
            [0.0, 0.0, 1.0, 1.0],
            dtype=np.float32,
        )
        eligible = np.ones(4, dtype=bool)

        threshold = zero_training_true_veto_threshold(
            probability,
            target,
            eligible,
        )
        stats = execution_stats(
            probability,
            target,
            eligible,
            threshold,
        )

        self.assertGreater(threshold, 0.7)
        self.assertEqual(stats["vetoed_true_alert_cells"], 0)
        self.assertEqual(stats["vetoed_false_alert_cells"], 2)

    def test_execution_stats_reports_benefit_and_harm_separately(self):
        stats = execution_stats(
            np.asarray([0.9, 0.8, 0.7, 0.1]),
            np.asarray([1.0, 0.0, 1.0, 0.0]),
            np.ones(4, dtype=bool),
            0.75,
        )

        self.assertEqual(stats["vetoed_false_alert_cells"], 1)
        self.assertEqual(stats["vetoed_true_alert_cells"], 1)
        self.assertAlmostEqual(
            stats["false_alert_veto_coverage"],
            0.5,
        )
        self.assertAlmostEqual(stats["true_alert_veto_rate"], 0.5)
        self.assertEqual(stats["net_correct_veto_cells"], 0)


if __name__ == "__main__":
    unittest.main()
