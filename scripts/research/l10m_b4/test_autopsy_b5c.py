from __future__ import annotations

import unittest

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, all_specs

from .autopsy_b5c import classify_terminal, shortest_strict_steps
from .hard_benchmark import evaluate_instance, load_benchmark


class B5CAutopsyTest(unittest.TestCase):
    def test_initial_distance_matches_five_step_b4_certificate(self) -> None:
        instance = load_benchmark()["instances"][0]
        scores = {
            spec: float(evaluate_instance(spec, instance)["behavioral_score"])
            for spec in all_specs()
        }

        self.assertEqual(shortest_strict_steps(INITIAL_SPEC, scores, max(scores.values())), 5)

    def test_terminal_priority_is_domain_then_mechanism_then_close(self) -> None:
        mechanism = [{"signature": "waste", "qualified": True}]

        self.assertEqual(
            classify_terminal([{"feature": "fine_turn"}], mechanism)[0],
            "OBSERVABLE_CONDITIONAL_DOMAIN_HYPOTHESIS_IDENTIFIED",
        )
        self.assertEqual(
            classify_terminal([], mechanism)[0],
            "BALANCED_V2_MECHANISM_HYPOTHESIS_IDENTIFIED",
        )
        self.assertEqual(
            classify_terminal([], [{"signature": "waste", "qualified": False}])[0],
            "NO_REPRODUCIBLE_HETEROGENEITY_EXPLANATION_CLOSE_OPERATOR_ADMISSION_ROUTE",
        )


if __name__ == "__main__":
    unittest.main()
