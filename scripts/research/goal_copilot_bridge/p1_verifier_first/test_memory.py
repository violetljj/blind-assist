from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_verifier_first.core import (
    CandidateEvidence,
    GoalContract,
    initialize_ledger,
    update_ledger,
)
from scripts.research.goal_copilot_bridge.p1_verifier_first.memory import (
    MemoryObservation,
    MemoryPolicy,
    initialize_memory,
    record_observation,
)


def make_ledger():
    goal = GoalContract(
        goal_id="goal",
        reference_mode="UNIQUE",
        goal_predicate="entrance",
        allowed_rebinding=False,
        arrival_predicate="fresh confirmation",
        safety_constraints=("NO_UNVERIFIED_TRANSLATION",),
    )
    return initialize_ledger(goal, "r1", motion_model="STATIC_WORLD")


def candidate(index: int, **changes) -> CandidateEvidence:
    values = {
        "evidence_id": f"e{index}",
        "candidate_id": f"c{index}",
        "entity_hypothesis_id": "r1",
        "proposal_source": "DETECTOR",
        "candidate_region_xyxy": (1.0, 2.0, 10.0, 20.0),
        "appearance_support": 0.9,
        "appearance_contradiction": 0.0,
        "spatial_prediction": "SUPPORTED",
        "parent_slot": "INSUFFICIENT",
        "relational_context": "INSUFFICIENT",
        "distractor_exclusion": "SUPPORTED",
        "current_goal_validity": "SUPPORTED",
    }
    values.update(changes)
    return CandidateEvidence(**values)


def observation(index: int, **changes) -> MemoryObservation:
    values = {
        "evidence_id": f"e{index}",
        "candidate_id": f"c{index}",
        "referent_id": "r1",
        "frame_id": f"f{index}",
        "target_crop_ref": f"sha256:target-{index}",
        "context_crop_ref": f"sha256:context-{index}",
        "full_frame_ref": f"sha256:frame-{index}",
        "orientation_source": "IMU",
        "orientation_yaw_deg": float(index * 10),
        "distance_band": "FAR",
        "viewpoint_bin": "FRONTAL",
        "scale_band": "SMALL",
        "context_anchor_id": "building-a",
    }
    values.update(changes)
    return MemoryObservation(**values)


class AdaptiveMultiViewMemoryTest(unittest.TestCase):
    def test_confirmed_observation_enters_verified_bank(self):
        ledger = update_ledger(
            make_ledger(), sequence_index=1, observability_reason="IN_VIEW_RELIABLE", candidates=(candidate(1),)
        )
        memory = record_observation(initialize_memory("r1"), ledger, observation(1), sequence_index=1)
        self.assertEqual(1, len(memory.verified))
        self.assertEqual(0, len(memory.tentative))
        self.assertEqual("VERIFIED_ADMITTED", memory.receipts[-1].action)

    def test_appearance_only_candidate_is_tentative_and_not_retrievable(self):
        value = candidate(1, spatial_prediction="INSUFFICIENT", distractor_exclusion="INSUFFICIENT")
        ledger = update_ledger(
            make_ledger(), sequence_index=1, observability_reason="IN_VIEW_RELIABLE", candidates=(value,)
        )
        memory = record_observation(initialize_memory("r1"), ledger, observation(1), sequence_index=1)
        self.assertEqual(1, len(memory.tentative))
        self.assertEqual((), memory.retrieval_packet())

    def test_tentative_view_is_promoted_only_after_later_independent_confirmation(self):
        weak = candidate(1, spatial_prediction="INSUFFICIENT", distractor_exclusion="INSUFFICIENT")
        ledger = update_ledger(
            make_ledger(), sequence_index=1, observability_reason="IN_VIEW_RELIABLE", candidates=(weak,)
        )
        tentative_view = observation(1)
        memory = record_observation(initialize_memory("r1"), ledger, tentative_view, sequence_index=1)
        confirmed = candidate(1)
        ledger = update_ledger(
            ledger, sequence_index=2, observability_reason="IN_VIEW_RELIABLE", candidates=(confirmed,)
        )
        memory = record_observation(memory, ledger, tentative_view, sequence_index=2)
        self.assertEqual(1, len(memory.verified))
        self.assertEqual(0, len(memory.tentative))
        self.assertEqual("e1", memory.verified[0].observation.evidence_id)

    def test_out_of_view_event_stops_all_memory_writes(self):
        ledger = update_ledger(
            make_ledger(), sequence_index=1, observability_reason="PREDICTED_OUT_OF_FOV", candidates=()
        )
        memory = record_observation(initialize_memory("r1"), ledger, observation(1), sequence_index=1)
        self.assertEqual(0, len(memory.verified))
        self.assertEqual(0, len(memory.tentative))
        self.assertEqual("OBSERVATION_NOT_WRITTEN", memory.receipts[-1].action)

    def test_occluded_candidate_cannot_enter_tentative_or_verified_memory(self):
        ledger = update_ledger(
            make_ledger(),
            sequence_index=1,
            observability_reason="PREDICTED_OCCLUDED",
            candidates=(candidate(1),),
        )
        memory = record_observation(initialize_memory("r1"), ledger, observation(1), sequence_index=1)
        self.assertEqual(0, len(memory.verified))
        self.assertEqual(0, len(memory.tentative))
        self.assertEqual("OBSERVATION_NOT_WRITTEN", memory.receipts[-1].action)

    def test_redundant_coverage_cell_is_not_saved_twice(self):
        ledger = make_ledger()
        memory = initialize_memory("r1")
        for index in (1, 2):
            ledger = update_ledger(
                ledger, sequence_index=index, observability_reason="IN_VIEW_RELIABLE", candidates=(candidate(index),)
            )
            memory = record_observation(memory, ledger, observation(index), sequence_index=index)
        self.assertEqual(1, len(memory.verified))
        self.assertEqual("VERIFIED_REDUNDANT_DROPPED", memory.receipts[-1].action)

    def test_same_view_cell_with_new_stable_context_is_saved(self):
        ledger = make_ledger()
        memory = initialize_memory("r1")
        for index, context in ((1, "facade"), (2, "door-sign")):
            ledger = update_ledger(
                ledger, sequence_index=index, observability_reason="IN_VIEW_RELIABLE", candidates=(candidate(index),)
            )
            memory = record_observation(
                memory,
                ledger,
                observation(index, context_anchor_id=context),
                sequence_index=index,
            )
        self.assertEqual(2, len(memory.verified))

    def test_new_distance_viewpoint_and_scale_cells_are_saved(self):
        variants = (
            {},
            {"distance_band": "MID", "scale_band": "MEDIUM"},
            {"distance_band": "NEAR", "viewpoint_bin": "RIGHT", "scale_band": "LARGE"},
        )
        ledger = make_ledger()
        memory = initialize_memory("r1")
        for index, changes in enumerate(variants, start=1):
            ledger = update_ledger(
                ledger, sequence_index=index, observability_reason="IN_VIEW_RELIABLE", candidates=(candidate(index),)
            )
            memory = record_observation(memory, ledger, observation(index, **changes), sequence_index=index)
        self.assertEqual(3, len(memory.verified))
        self.assertEqual(3, len(set(memory.coverage_cells)))

    def test_verified_bank_is_bounded_while_preserving_diverse_coverage(self):
        variants = (
            {},
            {"distance_band": "MID", "scale_band": "MEDIUM"},
            {"distance_band": "NEAR", "viewpoint_bin": "RIGHT", "scale_band": "LARGE"},
        )
        ledger = make_ledger()
        memory = initialize_memory("r1")
        policy = MemoryPolicy(max_verified_entries=2)
        for index, changes in enumerate(variants, start=1):
            ledger = update_ledger(
                ledger, sequence_index=index, observability_reason="IN_VIEW_RELIABLE", candidates=(candidate(index),)
            )
            memory = record_observation(
                memory, ledger, observation(index, **changes), sequence_index=index, policy=policy
            )
        self.assertEqual(2, len(memory.verified))
        self.assertIsNotNone(memory.receipts[-1].evicted_evidence_id)

    def test_wrong_referent_cannot_poison_existing_bank(self):
        ledger = update_ledger(
            make_ledger(), sequence_index=1, observability_reason="IN_VIEW_RELIABLE", candidates=(candidate(1),)
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            record_observation(
                initialize_memory("r1"), ledger, observation(1, referent_id="other"), sequence_index=1
            )

    def test_rebinding_requires_a_fresh_memory_bank(self):
        goal = GoalContract(
            goal_id="goal",
            reference_mode="SET_VALUED",
            goal_predicate="entrance",
            allowed_rebinding=True,
            arrival_predicate="fresh confirmation",
            safety_constraints=("NO_UNVERIFIED_TRANSLATION",),
        )
        ledger = initialize_ledger(goal, "r1", motion_model="STATIC_WORLD")
        rebound = candidate(1, entity_hypothesis_id="r2", rebind_candidate=True)
        ledger = update_ledger(
            ledger, sequence_index=1, observability_reason="IN_VIEW_RELIABLE", candidates=(rebound,)
        )
        memory = record_observation(initialize_memory("r1"), ledger, observation(1), sequence_index=1)
        self.assertEqual("REFERENT_REBOUND_REQUIRES_NEW_BANK", memory.receipts[-1].action)
        self.assertEqual(0, len(memory.verified))

    def test_memory_updates_are_immutable_and_receipted(self):
        original = initialize_memory("r1")
        ledger = update_ledger(
            make_ledger(), sequence_index=1, observability_reason="IN_VIEW_RELIABLE", candidates=(candidate(1),)
        )
        updated = record_observation(original, ledger, observation(1), sequence_index=1)
        self.assertEqual((), original.verified)
        self.assertEqual(1, len(updated.verified))
        self.assertEqual(1, updated.receipts[-1].verified_count)


if __name__ == "__main__":
    unittest.main()
