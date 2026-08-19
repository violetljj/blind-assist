from __future__ import annotations

import unittest

from .evaluation import Arm, ProgressStatus
from .scenario_matrix import PROTOCOL_ID, SCENARIOS, build_matrix, run_matrix


class L10MB0ScenarioMatrixTest(unittest.TestCase):
    def test_matrix_is_new_protocol_and_has_eight_targets(self) -> None:
        self.assertEqual(PROTOCOL_ID, "L10M-B0-B-MATCHED-COUNTERFACTUALS-V1")
        self.assertEqual(tuple(row[0] for row in build_matrix()), SCENARIOS)

    def test_reactive_solvable_preservation_is_reported(self) -> None:
        result = run_matrix()
        self.assertEqual(result["scenario_count"], 6)
        self.assertEqual(result["reactive_solvable_preservation"]["denominator"], 2)
        self.assertIn("stateful_rate", result["reactive_solvable_preservation"])
        self.assertEqual(result["progress_states"], [status.value for status in ProgressStatus])

    def test_uncertain_progress_does_not_count_as_confirmed_stuck(self) -> None:
        result = run_matrix()
        uncertain = next(row for row in result["scenarios"] if row["scenario"] == "uncertain_progress")
        self.assertIsNone(uncertain["arms"][Arm.STATEFUL.value]["stuck_detection_step"])


if __name__ == "__main__":
    unittest.main()
