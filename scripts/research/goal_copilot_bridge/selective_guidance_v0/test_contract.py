from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .contract import (
    CandidateCardinality,
    CompletionAuthority,
    CompletionReceipt,
    CurrentFrameObservation,
    EpisodeVisibilityState,
    OutputToken,
    ProviderReceipt,
    RangeBucket,
    decide,
    derive_episode_interaction,
)
from .event_log import EpisodeEventLog


def observation(**overrides):
    values = {
        "goal_contract": {"goal_id": "g1", "goal_text": "find the main entrance"},
        "frame_id": "f1",
        "observed_at_ms": 1_000,
        "decision_at_ms": 1_100,
        "visible_candidate_ids": ("a",),
        "selected_referent": "a",
        "cardinality": CandidateCardinality.UNIQUE,
        "target_visible": True,
        "selection_authorized": True,
        "requested_direction": OutputToken.GUIDE_LEFT,
        "range_bucket": RangeBucket.APPROACHING,
        "provider_receipts": (ProviderReceipt("baseline", "r1", "frozen-v0", "f1"),),
        "latency_ms": 100.0,
    }
    values.update(overrides)
    return CurrentFrameObservation(**values)


class SelectiveGuidanceContractTest(unittest.TestCase):
    def test_stop_is_not_completion(self):
        result = decide(observation(stop_for_safety=True))
        self.assertEqual(OutputToken.STOP_FOR_SAFETY, result.command)
        self.assertNotEqual(OutputToken.COMPLETED_BY_USER, result.status)

    def test_perception_cannot_create_completion(self):
        with self.assertRaisesRegex(ValueError, "completion requires"):
            CompletionReceipt(CompletionAuthority.PERCEPTION, "p1", 1_100)

    def test_contested_candidates_block_direction(self):
        result = decide(observation(
            visible_candidate_ids=("a", "b"),
            selected_referent="a",
            cardinality=CandidateCardinality.SET_VALUED,
            selection_authorized=False,
        ))
        self.assertEqual(OutputToken.CONTESTED, result.status)
        self.assertIsNone(result.command)

    def test_stale_and_not_visible_invalidate_old_direction(self):
        stale = decide(observation(decision_at_ms=3_000))
        not_visible = decide(observation(target_visible=False))
        self.assertEqual(OutputToken.STALE, stale.status)
        self.assertEqual(OutputToken.NOT_VISIBLE, not_visible.status)
        self.assertIsNone(stale.command)
        self.assertIsNone(not_visible.command)
        self.assertIsNone(stale.selected_referent)
        self.assertIsNone(not_visible.selected_referent)

    def test_lost_is_derived_only_by_episode_transition(self):
        found = decide(observation())
        visible = derive_episode_interaction(EpisodeVisibilityState.NEVER_SEEN, found)
        self.assertEqual(EpisodeVisibilityState.VISIBLE, visible.visibility_state)
        self.assertIsNone(visible.event)

        not_visible = decide(observation(target_visible=False))
        lost = derive_episode_interaction(visible.visibility_state, not_visible)
        self.assertEqual(EpisodeVisibilityState.NOT_VISIBLE_AFTER_VISIBLE, lost.visibility_state)
        self.assertEqual(OutputToken.LOST, lost.event)

    def test_never_seen_not_visible_is_not_lost(self):
        not_visible = decide(observation(target_visible=False))
        result = derive_episode_interaction(EpisodeVisibilityState.NEVER_SEEN, not_visible)
        self.assertEqual(EpisodeVisibilityState.NEVER_SEEN, result.visibility_state)
        self.assertIsNone(result.event)

    def test_handoff_requires_later_user_confirmation(self):
        handoff = decide(observation(handoff_ready=True, range_bucket=RangeBucket.NEAR))
        self.assertEqual(OutputToken.HANDOFF_READY, handoff.status)
        completed = decide(observation(
            handoff_ready=True,
            range_bucket=RangeBucket.NEAR,
            completion_receipt=CompletionReceipt(
                CompletionAuthority.USER_EXPLICIT, "user-confirm-1", 1_100
            ),
        ))
        self.assertEqual(OutputToken.COMPLETED_BY_USER, completed.status)

    def test_unique_and_set_valued_have_legal_distinct_behavior(self):
        unique = decide(observation(cardinality=CandidateCardinality.UNIQUE))
        set_valued = decide(observation(
            visible_candidate_ids=("a", "b"),
            selected_referent="b",
            cardinality=CandidateCardinality.SET_VALUED,
            selection_authorized=True,
        ))
        self.assertEqual(OutputToken.FOUND, unique.status)
        self.assertEqual(OutputToken.FOUND, set_valued.status)
        self.assertEqual("a", unique.selected_referent)
        self.assertEqual("b", set_valued.selected_referent)

    def test_abstention_is_a_first_class_output(self):
        result = decide(observation(
            visible_candidate_ids=(),
            selected_referent=None,
            target_visible=None,
            selection_authorized=False,
            requested_direction=None,
            range_bucket=RangeBucket.UNKNOWN,
        ))
        self.assertEqual(OutputToken.ABSTAIN, result.status)
        self.assertIn("RANGE_UNKNOWN", result.tokens)

    def test_event_log_reconstructs_guidance_and_evidence(self):
        obs = observation()
        result = decide(obs)
        with tempfile.TemporaryDirectory() as directory:
            log = EpisodeEventLog(Path(directory) / "events.jsonl")
            log.append(
                episode_id="e1",
                event_id="evt1",
                observation=obs,
                decision=result,
                spoken_command="左前方",
                haptic_command="LEFT_PULSE",
            )
            rows = log.read_all()
        self.assertEqual(1, len(rows))
        self.assertEqual(["a"], rows[0]["visible_candidate_set"])
        self.assertEqual("a", rows[0]["selected_referent"])
        self.assertEqual("FOUND", rows[0]["decision_state"])
        self.assertIn("GUIDE_LEFT", rows[0]["decision_tokens"])
        self.assertEqual("r1", rows[0]["provider_receipts"][0]["receipt_id"])


if __name__ == "__main__":
    unittest.main()
