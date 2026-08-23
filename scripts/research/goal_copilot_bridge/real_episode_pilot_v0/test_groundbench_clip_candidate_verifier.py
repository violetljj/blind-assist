import unittest

from PIL import Image

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_groundbench_clip_candidate_verifier import (
    expanded_box,
    focused_context,
    paired_verdict,
    rerank_episode,
    select_development_variant,
)


class GroundBenchClipCandidateVerifierTest(unittest.TestCase):
    def test_expanded_box_clamps_to_image(self) -> None:
        self.assertEqual(expanded_box([0, 0, 20, 10], 100, 50, 1.25), (0, 0, 22, 11))
        self.assertEqual(expanded_box([99, 49, 100, 50], 100, 50, 1.25), (99, 49, 100, 50))

    def test_focused_context_preserves_candidate(self) -> None:
        image = Image.new("RGB", (20, 20), "white")
        focused = focused_context(image, [5, 5, 15, 15])
        self.assertEqual(focused.getpixel((10, 10)), (255, 255, 255))
        self.assertLess(focused.getpixel((0, 0))[0], 255)

    def test_rerank_is_score_first_and_stable_on_tie(self) -> None:
        episode = {
            "candidates": [
                {"candidate_id": "a", "provider_rank": 1},
                {"candidate_id": "b", "provider_rank": 2},
                {"candidate_id": "c", "provider_rank": 3},
            ]
        }
        ranked = rerank_episode(episode, {"a": 0.1, "b": 0.9, "c": 0.9})
        self.assertEqual([item["candidate_id"] for item in ranked["candidates"]], ["b", "c", "a"])
        self.assertEqual([item["provider_rank"] for item in ranked["candidates"]], [1, 2, 3])

    def test_development_requires_rank1_improvement(self) -> None:
        def metric(rank1: int, mrr: float) -> dict:
            return {"recall_at_k": {"1": rank1}, "mean_reciprocal_rank_given_available": mrr}
        metrics = {
            "PROVIDER_ORDER": metric(10, 0.5),
            "CROP_100": metric(10, 0.6),
            "CROP_125": metric(11, 0.55),
            "FOCUS_CONTEXT_020": metric(9, 0.7),
            "DUAL_CROP125_FOCUS020": metric(11, 0.54),
        }
        selected = select_development_variant(metrics)
        self.assertEqual(selected["selected_variant"], "CROP_125")
        self.assertTrue(selected["confirmation_authorized"])

    def test_confirmation_requires_end_to_end_gain_without_more_wrong(self) -> None:
        rank0, rank1 = {"recall_at_k": {"1": 30}}, {"recall_at_k": {"1": 35}}
        def evaluation(correct: int, wrong: int) -> dict:
            return {
                "outcome_counts": {"CORRECT_GROUNDING": correct},
                "wrong_confident_guidance_all_observations": {"numerator": wrong},
            }
        supported = paired_verdict(rank0, rank1, evaluation(40, 8), evaluation(44, 8))
        unsafe = paired_verdict(rank0, rank1, evaluation(40, 8), evaluation(44, 9))
        self.assertEqual(supported["verdict"], "CLIP_CANDIDATE_VERIFIER_SUPPORTED")
        self.assertEqual(unsafe["verdict"], "CLIP_CANDIDATE_VERIFIER_NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
