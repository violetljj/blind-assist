from __future__ import annotations

import unittest

from .b0d_arrival_orthogonality import (
    B0C_FROZEN_VERDICT,
    B0D_CONFIRMED_VERDICT,
    PROTOCOL_ID,
    build_b0d_matrix,
    run_matrix_b0d,
)
from .evaluation import Action, Arm
from .scenario_matrix import build_matrix


class L10MB0DArrivalOrthogonalityTest(unittest.TestCase):
    def test_protocol_and_four_case_matrix_are_frozen(self) -> None:
        self.assertEqual(PROTOCOL_ID, "L10M-B0-D-ARRIVAL-STUCK-ORTHOGONALITY-V1")
        self.assertEqual(
            [row[0] for row in build_b0d_matrix()],
            [
                "stuck_without_confirmed_arrival",
                "stuck_with_confirmed_arrival",
                "recovery_with_confirmed_arrival",
                "recovery_unknown_without_arrival",
            ],
        )

    def test_four_cases_match_expected_outcomes_for_both_stateful_arms(self) -> None:
        result = run_matrix_b0d()
        self.assertEqual(result["scenario_count"], 4)
        for row in result["scenarios"]:
            for arm in (Arm.STATEFUL, Arm.STATEFUL_SAFETY):
                self.assertTrue(row["arms"][arm.value]["matches_expected"], row["scenario"])

    def test_terminal_semantics_are_state_independent_within_probe_scope(self) -> None:
        result = run_matrix_b0d()
        self.assertEqual(result["verdict"], B0D_CONFIRMED_VERDICT)
        self.assertTrue(all(result["invariants"].values()))
        self.assertEqual(
            {probe["parent_prepared_action"] for probe in result["state_independence_probes"]},
            {Action.FORWARD.value, Action.RECOVER.value},
        )
        self.assertGreaterEqual(
            {probe["recovery_attempts_before_arrival"] for probe in result["state_independence_probes"]},
            {0, 1, 2},
        )

    def test_b0c_is_interpreted_without_rewriting_its_inputs(self) -> None:
        result = run_matrix_b0d()["b0c_frozen_observation"]
        self.assertEqual(result["verdict"], B0C_FROZEN_VERDICT)
        self.assertEqual(
            result["flipped_scenarios"],
            {
                Arm.STATEFUL.value: ["true_stuck", "recovery_plus_arrival"],
                Arm.STATEFUL_SAFETY.value: ["true_stuck", "recovery_plus_arrival"],
            },
        )
        self.assertEqual(result["reactive_solvable_preservation"], {"denominator": 2, "stateful_rate": 1.0})
        self.assertFalse(result["unknown_progress_accumulates_stuck"])
        self.assertTrue(result["recovery_then_progress_exits_recovery"])
        self.assertFalse(result["b0c_inputs_modified"])
        self.assertFalse(result["b0c_result_modified"])
        true_stuck = next(row for row in build_matrix() if row[0] == "true_stuck")
        self.assertTrue(true_stuck[2][-1].arrived)

    def test_b1_and_structured_search_remain_unstarted(self) -> None:
        self.assertEqual(
            run_matrix_b0d()["execution_boundary"],
            {"b1_started": False, "structured_search_started": False},
        )


if __name__ == "__main__":
    unittest.main()
