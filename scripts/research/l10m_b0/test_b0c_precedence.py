from __future__ import annotations

import unittest

from .b0c_precedence import PROTOCOL_ID, TERMINAL_ACTIONS, run_episode_b0c, run_matrix_b0c
from .evaluation import Action, Arm
from .scenario_matrix import build_matrix


class L10MB0CPrecedenceTest(unittest.TestCase):
    def test_protocol_and_scenario_count_are_frozen(self) -> None:
        result = run_matrix_b0c()
        self.assertEqual(PROTOCOL_ID, "L10M-B0-C-RECOVERY-PRECEDENCE-V1")
        self.assertEqual(result["scenario_count"], 6)
        self.assertEqual(result["reactive_solvable_preservation"], {"denominator": 2, "stateful_rate": 1.0})

    def test_recovery_arrival_is_terminal_for_both_stateful_arms(self) -> None:
        row = next(item for item in run_matrix_b0c()["scenarios"] if item["scenario"] == "recovery_plus_arrival")
        for arm in (Arm.STATEFUL, Arm.STATEFUL_SAFETY):
            self.assertTrue(row["arms"][arm.value]["success"])
            self.assertEqual(row["arms"][arm.value]["actions"][-1], Action.STOP.value)

    def test_positive_progress_exits_recovery_before_action_selection(self) -> None:
        row = next(item for item in run_matrix_b0c()["scenarios"] if item["scenario"] == "recovery_then_progress")
        for arm in (Arm.STATEFUL, Arm.STATEFUL_SAFETY):
            self.assertEqual(row["arms"][arm.value]["actions"], ["FORWARD", "FORWARD", "FORWARD", "FORWARD"])

    def test_arrival_action_invariant(self) -> None:
        for name, evidence, truth, _ in build_matrix():
            for arm in (Arm.STATEFUL, Arm.STATEFUL_SAFETY):
                stats = run_episode_b0c(arm, evidence, truth)
                for ev, tr, action in zip(evidence, truth, stats.actions):
                    if tr.arrived and ev.quality >= 0.50 and not ev.stale and not ev.conflict:
                        self.assertIn(action, TERMINAL_ACTIONS)


if __name__ == "__main__":
    unittest.main()
