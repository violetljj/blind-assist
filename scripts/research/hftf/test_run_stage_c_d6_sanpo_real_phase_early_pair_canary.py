import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_sanpo_real_phase_early_pair_canary import (
    expansion_decision,
    phase_group_weights,
)


def window(session: str, phase: str, target: float) -> dict:
    return {
        "source_session_id": session,
        "phase": phase,
        "false_alert_target": target,
    }


class SanpoRealPhaseEarlyPairCanaryTest(unittest.TestCase):
    def test_phase_group_weights_balance_classes_and_groups(self):
        rows = [
            window("negative", "negative_event", 1.0),
            window("negative", "negative_event", 1.0),
            window("positive-a", "positive_passed", 1.0),
            window("positive-a", "positive_alertable", 0.0),
            window("positive-a", "positive_alertable", 0.0),
            window("positive-b", "positive_alertable", 0.0),
        ]

        weights = phase_group_weights(rows)

        self.assertTrue(np.isclose(weights.sum(), 1.0))
        self.assertTrue(
            np.isclose(
                weights[
                    [row["false_alert_target"] == 1.0 for row in rows]
                ].sum(),
                0.5,
            )
        )
        self.assertTrue(
            np.isclose(
                weights[
                    [row["false_alert_target"] == 0.0 for row in rows]
                ].sum(),
                0.5,
            )
        )
        self.assertTrue(np.isclose(weights[:2].sum(), 0.25))
        self.assertTrue(np.isclose(weights[2], 0.25))
        self.assertTrue(np.isclose(weights[3:5].sum(), 0.25))
        self.assertTrue(np.isclose(weights[5], 0.25))

    def test_expansion_requires_both_primary_increments(self):
        def result(auroc: float, ap: float) -> dict:
            return {
                "event_phase_p95_ranking": {
                    "candidate_auroc_delta": auroc,
                    "candidate_average_precision_delta": ap,
                }
            }

        self.assertTrue(
            expansion_decision(result(0.01, 0.02))[
                "supported_to_expand"
            ]
        )
        self.assertFalse(
            expansion_decision(result(0.01, -0.02))[
                "supported_to_expand"
            ]
        )


if __name__ == "__main__":
    unittest.main()
