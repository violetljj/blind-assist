from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts.research.goal_copilot_bridge.pilot.evaluator import evaluate_scenarios
from scripts.research.goal_copilot_2a.evaluator import evaluate_condition
from scripts.research.goal_copilot_2a.noise import CORRUPTIONS, SEVERITIES, condition_names


HERE = Path(__file__).resolve().parent
GC1 = HERE.parent / "goal_copilot_bridge" / "pilot"
BASELINE = GC1 / "initial_policy.py"
WINNER = HERE / "frozen_gc1_winner.py"
SCENARIOS = GC1 / "dev_scenarios.json"
PROTOCOL = HERE / "protocol.json"
EXPECTED_WINNER = "24d4e57374dd99363700ae881d18db536e48ec5f79f39e95c5b873e96edbc3a1"


class GoalCopilot2ATest(unittest.TestCase):
    def test_frozen_winner_and_zero_model_protocol(self) -> None:
        self.assertEqual(EXPECTED_WINNER, hashlib.sha256(WINNER.read_bytes()).hexdigest())
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(0, protocol["model_call_budget"])
        self.assertEqual("COMBINED_MODERATE", protocol["primary_condition"])
        self.assertEqual(
            {
                "winner_completion_rate_below": 0.8,
                "or_unsafe_guidance_above": 0,
                "or_premature_completion_above": 0,
                "or_eligible_reacquisition_rate_below": 0.8,
            },
            protocol["gc2b_admission"],
        )
        self.assertFalse(protocol["gc2b_model_calls_authorized_by_gc2a"])

    def test_condition_matrix_is_complete_and_unique(self) -> None:
        names = condition_names()
        self.assertEqual(22, len(names))
        self.assertEqual(22, len(set(names)))
        self.assertEqual("CLEAN", names[0])
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                self.assertIn(f"{corruption}_{severity}", names)
        for severity in SEVERITIES:
            self.assertIn(f"COMBINED_{severity}", names)

    def test_clean_condition_preserves_gc1_behavior(self) -> None:
        for policy in (BASELINE, WINNER):
            gc1 = evaluate_scenarios(policy, SCENARIOS)
            gc2 = evaluate_condition(policy, SCENARIOS, "CLEAN")
            self.assertEqual(gc1["metrics"]["completion_count"], gc2["metrics"]["completion_count"])
            self.assertEqual(gc1["metrics"]["family_completion_counts"], gc2["metrics"]["family_completion_counts"])
            self.assertEqual(gc1["metrics"]["unsafe_guidance"], gc2["metrics"]["unsafe_guidance"])
            self.assertEqual(gc1["metrics"]["premature_completion"], gc2["metrics"]["premature_completion"])
            self.assertEqual(gc1["metrics"]["timeouts"], gc2["metrics"]["timeouts"])
            for old, new in zip(gc1["outcomes"], gc2["outcomes"], strict=True):
                for key in (
                    "scenario_id", "task_family", "goal_completion", "normalized_progress",
                    "wrong_way_actions", "unsafe_guidance", "premature_completion",
                    "total_actions", "timeout", "semantic_validity", "candidate_runtime_error",
                ):
                    self.assertEqual(old[key], new[key], (policy.name, old["scenario_id"], key))

    def test_each_isolated_corruption_is_exercised(self) -> None:
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                result = evaluate_condition(WINNER, SCENARIOS, f"{corruption}_{severity}")
                self.assertGreater(
                    result["metrics"]["corruption_event_counts"][corruption],
                    0,
                    (corruption, severity),
                )

    def test_evaluation_is_deterministic(self) -> None:
        first = evaluate_condition(WINNER, SCENARIOS, "COMBINED_MODERATE")
        second = evaluate_condition(WINNER, SCENARIOS, "COMBINED_MODERATE")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
