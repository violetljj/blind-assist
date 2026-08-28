import unittest

from general_goal_locked_belief import (
    BeliefState,
    CandidateEvidence,
    GoalLockedBeliefController,
    GoalRepresentation,
    StatelessReferenceMatcher,
)


def candidate(candidate_id: str, center_x: float, score: float) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id=candidate_id,
        box_xywh=(center_x - 0.1, 0.3, 0.2, 0.4),
        instance_score=score,
        semantic_score=1.0,
        structure_score=0.9,
        visibility_quality=0.9,
        association_score=0.8,
    )


class GoalLockedBeliefTest(unittest.TestCase):
    def setUp(self) -> None:
        self.goal = GoalRepresentation(
            goal_id="specified-door",
            evidence_modalities=frozenset({"REFERENCE", "STRUCTURE", "GEOMETRY"}),
            reference_id="public-template",
        )

    def test_action_prediction_rejects_wrong_door_and_reacquires(self) -> None:
        controller = GoalLockedBeliefController(self.goal)
        first = controller.step([candidate("target-proposal", 0.5, 0.9)])
        self.assertEqual(BeliefState.TARGET, first.state)

        lost = controller.step(
            [candidate("wrong-door-proposal", 0.5, 0.95)], action_delta=(-0.8, 0.0)
        )
        self.assertEqual(BeliefState.LOST, lost.state)
        self.assertIsNone(lost.selected_candidate_id)

        still_lost = controller.step([candidate("wrong-door-proposal", 0.45, 0.96)])
        self.assertEqual(BeliefState.LOST, still_lost.state)
        self.assertIsNone(still_lost.selected_candidate_id)

        recovered = controller.step(
            [candidate("new-target-proposal", 0.5, 0.88)], action_delta=(0.8, 0.0)
        )
        self.assertEqual(BeliefState.TARGET, recovered.state)
        self.assertEqual("new-target-proposal", recovered.selected_candidate_id)
        self.assertEqual("REFERENCE_GEOMETRY_REACQUIRE", recovered.authority)

    def test_stateless_commits_to_absent_target_distractor(self) -> None:
        decision = StatelessReferenceMatcher().step(
            [candidate("wrong-door-proposal", 0.5, 0.95)]
        )
        self.assertEqual(BeliefState.TARGET, decision.state)
        self.assertEqual("wrong-door-proposal", decision.selected_candidate_id)

    def test_zero_ocr_goal_accepts_missing_text_score(self) -> None:
        decision = GoalLockedBeliefController(self.goal).step(
            [candidate("target-proposal", 0.5, 0.9)]
        )
        self.assertEqual(BeliefState.TARGET, decision.state)


if __name__ == "__main__":
    unittest.main()
