import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d6_sanpo_real_veto_ranking import (
    passed_alertable_pairs,
    phase_ranking,
)


class SanpoRealVetoRankingTest(unittest.TestCase):
    def test_phase_ranking_and_paired_direction(self):
        rows = [
            {
                "parent_event_id": "negative",
                "phase": "negative_event",
                "false_alert_target": 1.0,
                "candidate_p95": 0.9,
                "candidate_max": 0.95,
                "comparator_p95": 0.6,
                "comparator_max": 0.7,
            },
            {
                "parent_event_id": "positive",
                "phase": "positive_alertable",
                "false_alert_target": 0.0,
                "candidate_p95": 0.2,
                "candidate_max": 0.3,
                "comparator_p95": 0.4,
                "comparator_max": 0.5,
            },
            {
                "parent_event_id": "positive",
                "phase": "positive_passed",
                "false_alert_target": 1.0,
                "candidate_p95": 0.8,
                "candidate_max": 0.9,
                "comparator_p95": 0.5,
                "comparator_max": 0.6,
            },
        ]

        ranking = phase_ranking(
            rows,
            {"negative_event", "positive_alertable"},
            "p95",
        )
        pairs = passed_alertable_pairs(rows)

        self.assertEqual(ranking["candidate"]["auroc"], 1.0)
        self.assertEqual(ranking["unit_count"], 2)
        self.assertEqual(pairs["pair_count"], 1)
        self.assertEqual(
            pairs["candidate_passed_score_higher_count"], 1
        )
        self.assertTrue(
            np.isclose(
                pairs["candidate_p95_delta"]["mean"],
                0.6,
            )
        )


if __name__ == "__main__":
    unittest.main()
