import unittest

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_cmp_selective_commitment_v1 import (
    confirmation_verdict,
    gate_decisions,
)


class CmpSelectiveCommitmentRunTest(unittest.TestCase):
    def test_gate_only_demotes_select_and_preserves_raw_input(self) -> None:
        episodes = [{
            "episode_id": "episode-1",
            "candidates": [
                {"candidate_id": "candidate-1", "provider_rank": 1},
                {"candidate_id": "candidate-2", "provider_rank": 2},
            ],
        }]
        raw = [{
            "episode_id": "episode-1", "action": "SELECT", "confidence": 0.90,
            "selected_candidate_ids": ["candidate-2"],
        }]

        gated = gate_decisions(raw, episodes, {
            "policy_id": "TEST", "minimum_brain_confidence": 0.85, "require_provider_rank_1": True,
        })

        self.assertEqual(raw[0]["action"], "SELECT")
        self.assertEqual(gated[0]["action"], "CONTESTED")
        self.assertEqual(gated[0]["selected_candidate_ids"], [])
        self.assertFalse(gated[0]["selective_commitment_gate"]["rank_ok"])

    def test_gate_preserves_qualified_select(self) -> None:
        episodes = [{
            "episode_id": "episode-1",
            "candidates": [{"candidate_id": "candidate-1", "provider_rank": 1}],
        }]
        raw = [{
            "episode_id": "episode-1", "action": "SELECT", "confidence": 0.90,
            "selected_candidate_ids": ["candidate-1"],
        }]

        gated = gate_decisions(raw, episodes, {
            "policy_id": "TEST", "minimum_brain_confidence": 0.85, "require_provider_rank_1": True,
        })

        self.assertEqual(gated, raw)

    def test_confirmation_requires_all_predeclared_conditions(self) -> None:
        def evaluation(wrong: int, correct: int, precision: float) -> dict:
            return {
                "wrong_confident_guidance_all_observations": {"numerator": wrong},
                "outcome_counts": {"CORRECT_GROUNDING": correct},
                "commitment_accuracy": {"value": precision},
            }

        supported = confirmation_verdict(evaluation(5, 20, 0.80), evaluation(2, 17, 0.90))
        failed_retention = confirmation_verdict(evaluation(5, 20, 0.80), evaluation(2, 15, 0.90))

        self.assertEqual(supported["verdict"], "SELECTIVE_COMMITMENT_SUPPORTED")
        self.assertEqual(failed_retention["verdict"], "SELECTIVE_COMMITMENT_NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
