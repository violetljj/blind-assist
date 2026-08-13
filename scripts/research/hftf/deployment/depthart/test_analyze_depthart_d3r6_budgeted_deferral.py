from __future__ import annotations

import unittest

import numpy as np

from scripts.research.hftf.deployment.depthart.analyze_depthart_d3r6_budgeted_deferral import (
    budgeted_deferral,
)
from scripts.research.hftf.deployment.depthart.run_depthart_d3r4_selective_router_canary import (
    STATE_CLEAR,
    STATE_OCCUPIED,
    STATE_UNKNOWN,
)


class BudgetedDeferralTest(unittest.TestCase):
    def test_defers_top_score_without_emitting_occupied(self) -> None:
        dataset = {
            "parent_index": np.asarray([0, 0, 0]),
            "hard_evidence": np.asarray([True, True, True]),
            "source_available": np.asarray([True, True, True]),
            "baseline_state": np.asarray([STATE_CLEAR, STATE_CLEAR, STATE_OCCUPIED]),
            "truth_state": np.asarray([STATE_OCCUPIED, STATE_CLEAR, STATE_OCCUPIED]),
        }
        states, actions = budgeted_deferral(
            dataset, np.asarray([0.9, 0.1, 1.0]), 0.0005
        )
        self.assertEqual(states.tolist(), [STATE_UNKNOWN, STATE_CLEAR, STATE_OCCUPIED])
        self.assertEqual(actions["deferred_cell_count"], 1)
        self.assertEqual(actions["occupied_action_count"], 0)


if __name__ == "__main__":
    unittest.main()
