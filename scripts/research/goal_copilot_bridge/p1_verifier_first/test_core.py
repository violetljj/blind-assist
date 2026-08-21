from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_verifier_first.core import (
    CandidateEvidence,
    GoalContract,
    ParentAnchor,
    VerifierPolicy,
    initialize_ledger,
    update_ledger,
)


def contract(mode="UNIQUE", *, rebind=False) -> GoalContract:
    return GoalContract(
        goal_id="goal-opaque",
        reference_mode=mode,
        goal_predicate="building entrance",
        allowed_rebinding=rebind,
        arrival_predicate="fresh task-specific confirmation",
        safety_constraints=("NO_UNVERIFIED_TRANSLATION",),
    )


def candidate(candidate_id="c1", entity_id="r1", **changes) -> CandidateEvidence:
    values = {
        "evidence_id": f"e-{candidate_id}",
        "candidate_id": candidate_id,
        "entity_hypothesis_id": entity_id,
        "proposal_source": "DETECTOR",
        "candidate_region_xyxy": (10.0, 20.0, 40.0, 80.0),
        "appearance_support": 0.95,
        "appearance_contradiction": 0.05,
        "spatial_prediction": "SUPPORTED",
        "parent_slot": "INSUFFICIENT",
        "relational_context": "INSUFFICIENT",
        "distractor_exclusion": "SUPPORTED",
        "current_goal_validity": "SUPPORTED",
    }
    values.update(changes)
    return CandidateEvidence(**values)


class VerifierFirstLedgerTest(unittest.TestCase):
    def ledger(self, *, mode="UNIQUE", rebind=False, parent=False):
        return initialize_ledger(
            contract(mode, rebind=rebind),
            "r1",
            motion_model="PARENT_ATTACHED" if parent else "STATIC_WORLD",
            parent_anchor=ParentAnchor("building-a", "main-door-slot") if parent else None,
        )

    def test_goal_contract_only_allows_set_valued_rebinding(self):
        with self.assertRaisesRegex(ValueError, "SET_VALUED"):
            contract("UNIQUE", rebind=True)

    def test_appearance_is_capped_and_cannot_confirm_identity(self):
        weak = candidate(
            spatial_prediction="INSUFFICIENT",
            distractor_exclusion="INSUFFICIENT",
            appearance_support=1.0,
        )
        result = update_ledger(
            self.ledger(), sequence_index=1, observability_reason="IN_VIEW_RELIABLE", candidates=(weak,)
        )
        self.assertEqual("VERIFYING", result.decision)
        self.assertEqual(0.35, result.hypotheses[0].appearance_contribution)
        self.assertEqual((), result.identity_gallery_evidence_ids)

    def test_prediction_plus_exclusion_confirms_and_updates_gallery(self):
        result = update_ledger(
            self.ledger(), sequence_index=1, observability_reason="IN_VIEW_RELIABLE", candidates=(candidate(),)
        )
        self.assertEqual("CONFIRMED_VISIBLE", result.decision)
        self.assertEqual(("e-c1",), result.identity_gallery_evidence_ids)
        self.assertEqual("H_OTHER_REMAINS_POSSIBLE", result.snapshot()["h_other"])

    def test_parent_slot_can_supply_prediction_without_spatial_geometry(self):
        value = candidate(spatial_prediction="INSUFFICIENT", parent_slot="SUPPORTED")
        result = update_ledger(
            self.ledger(parent=True),
            sequence_index=1,
            observability_reason="IN_VIEW_RELIABLE",
            candidates=(value,),
            parent_context_visible=True,
        )
        self.assertEqual("CONFIRMED_VISIBLE", result.decision)

    def test_multiple_confirmed_hypotheses_are_ambiguous(self):
        result = update_ledger(
            self.ledger(),
            sequence_index=1,
            observability_reason="IN_VIEW_RELIABLE",
            candidates=(candidate("c1", "r1"), candidate("c2", "r2")),
        )
        self.assertEqual("AMBIGUOUS", result.decision)
        self.assertEqual((), result.identity_gallery_evidence_ids)

    def test_known_distractor_is_rejected_and_registered(self):
        result = update_ledger(
            self.ledger(),
            sequence_index=1,
            observability_reason="IN_VIEW_RELIABLE",
            candidates=(candidate("c2", "other", known_distractor=True),),
        )
        self.assertEqual("REJECTED", result.hypotheses[0].status)
        self.assertIn("other", result.distractor_registry)

    def test_out_of_view_and_low_resolution_are_not_negative_evidence(self):
        ledger = update_ledger(
            self.ledger(), sequence_index=1, observability_reason="PREDICTED_OUT_OF_FOV", candidates=()
        )
        self.assertEqual("LATENT_OUT_OF_VIEW", ledger.decision)
        self.assertEqual(0, ledger.reliable_miss_count)
        ledger = update_ledger(
            ledger, sequence_index=2, observability_reason="BELOW_RESOLUTION", candidates=()
        )
        self.assertEqual("VERIFYING", ledger.decision)
        self.assertEqual(0, ledger.reliable_miss_count)

    def test_reliable_in_view_misses_eventually_stale(self):
        ledger = update_ledger(
            self.ledger(), sequence_index=1, observability_reason="IN_VIEW_RELIABLE", candidates=()
        )
        self.assertEqual("VERIFYING", ledger.decision)
        ledger = update_ledger(
            ledger, sequence_index=2, observability_reason="IN_VIEW_RELIABLE", candidates=()
        )
        self.assertEqual("STALE", ledger.decision)
        self.assertEqual(2, ledger.reliable_miss_count)

    def test_unique_goal_does_not_silently_rebind(self):
        result = update_ledger(
            self.ledger(),
            sequence_index=1,
            observability_reason="IN_VIEW_RELIABLE",
            candidates=(candidate("c2", "r2", rebind_candidate=True),),
        )
        self.assertEqual("AMBIGUOUS", result.decision)
        self.assertEqual("r1", result.referent_id)

    def test_set_valued_goal_records_explicit_rebinding(self):
        result = update_ledger(
            self.ledger(mode="SET_VALUED", rebind=True),
            sequence_index=1,
            observability_reason="IN_VIEW_RELIABLE",
            candidates=(candidate("c2", "r2", rebind_candidate=True),),
        )
        self.assertEqual("REBOUND_TO_NEW_VALID_INSTANCE", result.decision)
        self.assertEqual("r2", result.referent_id)
        self.assertEqual("r1", result.evidence_history[-1].prior_referent_id)

    def test_identity_and_current_goal_validity_are_separate(self):
        result = update_ledger(
            self.ledger(),
            sequence_index=1,
            observability_reason="IN_VIEW_RELIABLE",
            candidates=(candidate(current_goal_validity="REJECTED"),),
        )
        self.assertEqual("DISPROVED", result.decision)
        self.assertEqual("REJECTED", result.current_goal_validity)
        self.assertEqual(("e-c1",), result.identity_gallery_evidence_ids)

    def test_hypotheses_are_bounded_but_h_other_is_retained(self):
        values = tuple(
            candidate(
                f"c{index}",
                f"r{index}",
                spatial_prediction="INSUFFICIENT",
                distractor_exclusion="INSUFFICIENT",
            )
            for index in range(6)
        )
        result = update_ledger(
            self.ledger(),
            sequence_index=1,
            observability_reason="IN_VIEW_RELIABLE",
            candidates=values,
            policy=VerifierPolicy(max_hypotheses=3),
        )
        self.assertEqual(3, len(result.hypotheses))
        self.assertEqual("H_OTHER_REMAINS_POSSIBLE", result.h_other)

    def test_evidence_requests_never_authorize_translation(self):
        ledger = update_ledger(
            self.ledger(parent=True),
            sequence_index=1,
            observability_reason="IN_VIEW_RELIABLE",
            candidates=(candidate(spatial_prediction="INSUFFICIENT", distractor_exclusion="INSUFFICIENT"),),
            parent_context_visible=False,
        )
        self.assertEqual("INCLUDE_PARENT_CONTEXT", ledger.evidence_request)
        ledger = update_ledger(
            ledger,
            sequence_index=2,
            observability_reason="OBSERVATION_UNRELIABLE",
            candidates=(),
        )
        self.assertEqual("HOLD_STILL", ledger.evidence_request)
        self.assertNotIn("TRANSLATE", {receipt.decision for receipt in ledger.evidence_history})

    def test_history_is_append_only_across_immutable_ledgers(self):
        first = self.ledger()
        second = update_ledger(
            first, sequence_index=1, observability_reason="IN_VIEW_RELIABLE", candidates=(candidate(),)
        )
        self.assertEqual((), first.evidence_history)
        self.assertEqual(1, len(second.evidence_history))
        with self.assertRaisesRegex(ValueError, "monotonically"):
            update_ledger(second, sequence_index=1, observability_reason="IN_VIEW_RELIABLE", candidates=())


if __name__ == "__main__":
    unittest.main()
