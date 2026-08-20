from __future__ import annotations

import unittest

from scripts.research.l10m_b1.evaluator import evaluate_spec
from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, PolicySpec, render_structured
from scripts.research.l10m_b3a.analyze_result import _trajectory
from scripts.research.l10m_b3a.exploration import (
    BALANCED_EXPLORATION_INSTRUCTION,
    admit_balanced_proposal,
    legal_adjacent_moves,
    proposal_move_tokens,
)
from scripts.research.l10m_b3a.protocol import PAIRED_SEEDS, derive_fresh_seeds


class BalancedExplorationTest(unittest.TestCase):
    def test_fresh_seeds_are_hash_derived_and_not_consumed(self) -> None:
        self.assertEqual(PAIRED_SEEDS, (1768, 7368, 1872))
        self.assertEqual(PAIRED_SEEDS, derive_fresh_seeds())
        self.assertTrue(set(PAIRED_SEEDS).isdisjoint({17, 29, 43, 53, 71, 89}))

    def test_instruction_contains_no_oracle_target(self) -> None:
        self.assertNotIn("0.10", BALANCED_EXPLORATION_INSTRUCTION)
        self.assertNotIn("b0110121", BALANCED_EXPLORATION_INSTRUCTION)
        self.assertNotIn("correct", BALANCED_EXPLORATION_INSTRUCTION.lower())

    def test_initial_neighborhood_has_eight_canonical_moves(self) -> None:
        moves = legal_adjacent_moves(INITIAL_SPEC)
        self.assertEqual(len(moves), 8)
        self.assertEqual(len(set(moves)), 8)

    def test_model_untried_direction_is_preferred(self) -> None:
        proposal = PolicySpec(fallback_min_quality=0.50)
        admitted, token, disposition = admit_balanced_proposal(
            INITIAL_SPEC,
            proposal,
            set(),
            seed=PAIRED_SEEDS[0],
            generation=1,
        )
        self.assertEqual(admitted, proposal)
        self.assertEqual(disposition, "MODEL_UNTRIED_DIRECTION")
        self.assertEqual(proposal_move_tokens(INITIAL_SPEC, admitted), {token})

    def test_repeated_model_move_projects_to_untried_coverage(self) -> None:
        proposal = PolicySpec(fallback_min_quality=0.50)
        first, first_token, _ = admit_balanced_proposal(
            INITIAL_SPEC,
            proposal,
            set(),
            seed=PAIRED_SEEDS[0],
            generation=1,
        )
        self.assertEqual(first, proposal)
        second, second_token, disposition = admit_balanced_proposal(
            INITIAL_SPEC,
            proposal,
            {str(first_token)},
            seed=PAIRED_SEEDS[0],
            generation=2,
        )
        self.assertEqual(disposition, "COVERAGE_PROJECTION")
        self.assertNotEqual(second_token, first_token)
        self.assertEqual(proposal_move_tokens(INITIAL_SPEC, second), {second_token})

    def test_full_static_neighborhood_coverage_has_no_repeat(self) -> None:
        attempted: set[str] = set()
        repeated_model_proposal = PolicySpec(fallback_min_quality=0.50)
        for generation in range(1, 9):
            _, token, _ = admit_balanced_proposal(
                INITIAL_SPEC,
                repeated_model_proposal,
                attempted,
                seed=PAIRED_SEEDS[1],
                generation=generation,
            )
            self.assertIsNotNone(token)
            self.assertNotIn(token, attempted)
            attempted.add(str(token))
        self.assertEqual(attempted, set(legal_adjacent_moves(INITIAL_SPEC)))

    def test_balanced_trajectory_recomputes_evaluator_and_operator_integrity(self) -> None:
        attempted: set[str] = set()
        incumbent = INITIAL_SPEC
        best_score = float(evaluate_spec(INITIAL_SPEC)["behavioral_score"])
        repeated_model_proposal = PolicySpec(fallback_min_quality=0.50)
        events = []
        for generation in range(1, 9):
            admitted, token, disposition = admit_balanced_proposal(
                incumbent,
                repeated_model_proposal,
                attempted,
                seed=PAIRED_SEEDS[2],
                generation=generation,
            )
            result = evaluate_spec(admitted)
            score = float(result["behavioral_score"])
            strict_improvement = bool(result["semantic_valid"] and not result["unsafe_candidate"] and score > best_score)
            if token is not None:
                attempted.add(token)
            events.append(
                {
                    "kind": "completion",
                    "seed": PAIRED_SEEDS[2],
                    "arm": "structured_balanced",
                    "generation": generation,
                    "candidate_output": render_structured(admitted).strip(),
                    "operator_move_token": token,
                    "operator_disposition": disposition,
                    **result,
                }
            )
            if strict_improvement:
                incumbent = admitted
                best_score = score

        summary = _trajectory(PAIRED_SEEDS[2], "structured_balanced", events)

        self.assertTrue(summary["operator_integrity"])
        self.assertTrue(summary["discovery_reached"])
        self.assertEqual(summary["semantic_invalid_count"], 0)


if __name__ == "__main__":
    unittest.main()
