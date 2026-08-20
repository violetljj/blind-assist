from __future__ import annotations

import unittest

from .saturation_audit import INITIAL_SCORE, _summarize


class SaturationSummaryTest(unittest.TestCase):
    def test_late_gain_and_ceiling_metrics(self) -> None:
        ceiling = 0.99
        rows = [
            {
                "first_improvement_generation": 2,
                "best_at_half_budget": ceiling,
                "final_best_score": ceiling,
                "late_realized_gain": 0.0,
                "improvement_generations": [2],
            },
            {
                "first_improvement_generation": None,
                "best_at_half_budget": INITIAL_SCORE,
                "final_best_score": INITIAL_SCORE,
                "late_realized_gain": 0.0,
                "improvement_generations": [],
            },
        ]

        result = _summarize(rows, ceiling)

        self.assertEqual(result["discovery_reach_count"], 1)
        self.assertEqual(result["ceiling_reach_count"], 1)
        self.assertEqual(result["late_strict_improvement_count"], 0)
        self.assertAlmostEqual(
            result["theoretical_headroom_remaining_after_half_budget_total"],
            ceiling - INITIAL_SCORE,
        )


if __name__ == "__main__":
    unittest.main()
