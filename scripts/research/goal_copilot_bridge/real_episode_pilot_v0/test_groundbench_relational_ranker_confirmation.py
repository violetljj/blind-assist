import unittest

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_groundbench_relational_ranker_confirmation import (
    paired_verdict,
)


class GroundBenchRelationalRankerConfirmationTest(unittest.TestCase):
    def test_paired_verdict_requires_rank_and_safe_end_to_end_gain(self) -> None:
        rank0, rank1 = {"recall_at_k": {"1": 30}}, {"recall_at_k": {"1": 34}}
        def evaluation(correct: int, wrong: int) -> dict:
            return {
                "outcome_counts": {"CORRECT_GROUNDING": correct},
                "wrong_confident_guidance_all_observations": {"numerator": wrong},
            }
        supported = paired_verdict(rank0, rank1, evaluation(40, 8), evaluation(43, 8))
        unsafe = paired_verdict(rank0, rank1, evaluation(40, 8), evaluation(43, 9))
        no_brain_gain = paired_verdict(rank0, rank1, evaluation(40, 8), evaluation(40, 7))
        self.assertEqual(supported["verdict"], "RELATIONAL_CANDIDATE_RANKER_SUPPORTED")
        self.assertEqual(unsafe["verdict"], "RELATIONAL_CANDIDATE_RANKER_NOT_SUPPORTED")
        self.assertEqual(no_brain_gain["verdict"], "RELATIONAL_CANDIDATE_RANKER_NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
