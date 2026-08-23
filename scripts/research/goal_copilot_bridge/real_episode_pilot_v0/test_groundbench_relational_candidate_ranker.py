import unittest

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.train_groundbench_relational_candidate_ranker import (
    FEATURE_NAMES,
    candidate_features,
    development_gate,
    expression_signals,
)


class GroundBenchRelationalCandidateRankerTest(unittest.TestCase):
    def test_expression_signals_are_bounded_and_relational(self) -> None:
        signals = expression_signals("the largest red car on the far left")
        self.assertEqual(signals["left"], 1.0)
        self.assertEqual(signals["large"], 1.0)
        self.assertEqual(signals["background"], 0.0)
        self.assertEqual(signals["right"], 0.0)

    def test_feature_vector_has_fixed_schema(self) -> None:
        candidate = {
            "candidate_id": "a", "provider_rank": 2, "proposal_score": 0.7,
            "region": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.6},
        }
        scores = {name: {"a": 0.2} for name in (
            "CROP_100", "CROP_125", "FOCUS_CONTEXT_020", "DUAL_CROP125_FOCUS020",
        )}
        values = candidate_features(candidate, "leftmost large car", scores)
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertGreater(values[FEATURE_NAMES.index("left_affinity")], 0)
        self.assertGreater(values[FEATURE_NAMES.index("large_affinity")], 0)

    def test_gate_requires_rank1_gain_and_noninferior_mrr(self) -> None:
        def metric(rank1: int, mrr: float) -> dict:
            return {"recall_at_k": {"1": rank1}, "mean_reciprocal_rank_given_available": mrr}
        promising = development_gate(metric(30, 0.7), metric(32, 0.71))
        lower_mrr = development_gate(metric(30, 0.7), metric(32, 0.69))
        tied_rank1 = development_gate(metric(30, 0.7), metric(30, 0.75))
        self.assertTrue(promising["confirmation_authorized"])
        self.assertFalse(lower_mrr["confirmation_authorized"])
        self.assertFalse(tied_rank1["confirmation_authorized"])


if __name__ == "__main__":
    unittest.main()
