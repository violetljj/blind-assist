import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_tartanground_paired_relation_pretraining_canary import (
    current_central_body_label,
)


class TartanGroundPairedRelationPretrainingCanaryTest(
    unittest.TestCase
):
    def _record(self, risk_values, known_values):
        risk = np.zeros((3, 3, 6, 6), dtype=np.float32)
        known = np.ones((3, 3, 6, 6), dtype=np.float32)
        for direction, value in risk_values.items():
            risk[0, 1, direction] = value
        for direction, value in known_values.items():
            known[0, 1, direction] = value

        def label(horizon, horizon_index):
            nullable = risk[horizon_index].astype(
                object
            )
            nullable[known[horizon_index] == 0] = None
            return {
                "known_target": known[
                    horizon_index
                ].astype(int).tolist(),
                "risk_score_target_nullable": nullable.tolist(),
            }

        return {
            "labels": {
                horizon: label(horizon, index)
                for index, horizon in enumerate(
                    ("current", "far", "near")
                )
            }
        }

    def test_positive_when_either_central_direction_has_risk(self):
        record = self._record({2: 1.0}, {})
        self.assertEqual(1, current_central_body_label(record))

    def test_negative_when_both_central_directions_are_clear(self):
        record = self._record({}, {})
        self.assertEqual(0, current_central_body_label(record))

    def test_ambiguous_when_clear_truth_is_incomplete(self):
        record = self._record({}, {3: 0.0})
        self.assertIsNone(current_central_body_label(record))


if __name__ == "__main__":
    unittest.main()
