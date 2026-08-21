from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p0_a1_ambiguity_gate import sweep


class SweepTest(unittest.TestCase):
    def test_threshold_grid_is_bounded_and_deterministic(self) -> None:
        self.assertEqual([1.0, 2.0, 3.0], sweep._thresholds([3.0, 1.0, 2.0, 2.0]))
        grid = sweep._thresholds([float(value) for value in range(30)])
        self.assertEqual(11, len(grid))
        self.assertEqual(0.0, grid[0])
        self.assertEqual(29.0, grid[-1])

    def test_gate_never_creates_a_commit(self) -> None:
        rows = [
            {
                "episode_id": "selected",
                "venue_parent_id": "a",
                "truth": "RESOLVABLE",
                "baseline_select": True,
                "baseline_correct": True,
                "features": {"score": 0.9},
            },
            {
                "episode_id": "abstained",
                "venue_parent_id": "b",
                "truth": "AMBIGUOUS",
                "baseline_select": False,
                "baseline_correct": False,
                "features": {"score": 1.0},
            },
        ]
        candidate = sweep._candidate(rows, [("score", "min", 0.8)])
        self.assertEqual(1, candidate["metrics"]["resolvable_commit_coverage_episode_micro"]["numerator"])
        self.assertEqual(0, candidate["metrics"]["ambiguous_false_commit_rate_episode_micro"]["numerator"])

    def test_parent_macro_does_not_treat_frames_as_independent_parents(self) -> None:
        rows = [
            {"episode_id": "a1", "venue_parent_id": "a", "truth": "AMBIGUOUS", "baseline_correct": False},
            {"episode_id": "a2", "venue_parent_id": "a", "truth": "AMBIGUOUS", "baseline_correct": False},
            {"episode_id": "b1", "venue_parent_id": "b", "truth": "AMBIGUOUS", "baseline_correct": False},
        ]
        metrics = sweep._metrics(rows, {"a1", "a2"})
        self.assertAlmostEqual(2 / 3, metrics["ambiguous_false_commit_rate_episode_micro"]["value"])
        self.assertAlmostEqual(0.5, metrics["ambiguous_false_commit_rate_venue_parent_macro"]["value"])


if __name__ == "__main__":
    unittest.main()
