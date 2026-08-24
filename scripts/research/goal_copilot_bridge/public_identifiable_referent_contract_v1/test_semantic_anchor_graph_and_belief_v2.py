from __future__ import annotations

import unittest

from .semantic_anchor_graph_and_belief_v2 import (
    ReferentBelief,
    TargetGraph,
    _edit_similarity,
    evaluate,
    frame_observability,
    generate_cohort,
    graph_candidate_scores,
    normalize_text,
)


class SemanticAnchorGraphAndBeliefV2Test(unittest.TestCase):
    def test_normalization_and_partial_lexical_similarity(self) -> None:
        self.assertEqual(normalize_text("Room ３０２"), "ROOM302")
        self.assertGreater(_edit_similarity("302", "30?"), _edit_similarity("302", "320"))

    def test_relational_score_prefers_physical_sign_over_directory(self) -> None:
        frame = next(frame for frame in generate_cohort() if frame.episode_id == "directory_binding")
        scores = graph_candidate_scores(TargetGraph(("ROOM", "302")), frame)
        self.assertGreater(scores["B"]["score"], scores["A"]["score"])
        self.assertGreater(scores["B"]["association"], scores["A"]["association"])

    def test_low_quality_is_unknown_and_does_not_change_belief(self) -> None:
        frame = next(frame for frame in generate_cohort() if frame.episode_id == "unreadable_not_negative")
        belief = ReferentBelief(candidate.candidate_id for candidate in frame.candidates)
        before = dict(belief.probabilities)
        decision = belief.update(frame, graph_candidate_scores(TargetGraph(("ROOM", "302")), frame))
        self.assertEqual(decision["state"], "UNKNOWN")
        self.assertEqual(before, belief.probabilities)
        self.assertLess(frame_observability(frame), 0.28)

    def test_correlated_repeat_is_strongly_downweighted(self) -> None:
        frames = [frame for frame in generate_cohort() if frame.episode_id == "correlated_directory_burst"]
        belief = ReferentBelief(candidate.candidate_id for candidate in frames[0].candidates)
        first = belief.update(frames[0], graph_candidate_scores(TargetGraph(("ROOM", "302")), frames[0]))
        second = belief.update(frames[1], graph_candidate_scores(TargetGraph(("ROOM", "302")), frames[1]))
        self.assertTrue(first["source_novel"])
        self.assertFalse(second["source_novel"])
        self.assertLess(second["update_weight"], first["update_weight"] * 0.1)

    def test_end_to_end_v2_has_fewer_wrong_locks_and_open_set_coverage(self) -> None:
        _, report = evaluate()
        baseline = report["metrics"]["substring_fsm"]
        v2 = report["metrics"]["sage_r_v2"]
        self.assertLess(v2["wrong_locks"], baseline["wrong_locks"])
        self.assertGreater(v2["none_correct"], baseline["none_correct"])
        self.assertEqual(v2["unknown_preserved"], v2["unknown_frames"])

    def test_geometric_jitter_does_not_remove_the_v2_uplift(self) -> None:
        for seed in range(302, 312):
            with self.subTest(seed=seed):
                _, report = evaluate(seed)
                baseline = report["metrics"]["substring_fsm"]
                v2 = report["metrics"]["sage_r_v2"]
                self.assertGreater(v2["correct_terminal_frames"], baseline["correct_terminal_frames"])
                self.assertEqual(v2["wrong_locks"], 0)


if __name__ == "__main__":
    unittest.main()
