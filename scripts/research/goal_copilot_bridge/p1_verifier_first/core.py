"""Outcome-blind verifier-first referent ledger mechanics.

Candidate sources may propose observations, but only this verifier can update
the active entity hypothesis or identity gallery. The module consumes no image,
pose, evaluator truth, future frame, or model output beyond the explicit public
evidence record defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


TriState = Literal["SUPPORTED", "REJECTED", "INSUFFICIENT"]
ReferenceMode = Literal["UNIQUE", "SET_VALUED", "AMBIGUOUS"]
MotionModel = Literal["STATIC_WORLD", "PARENT_ATTACHED", "INDEPENDENT_DYNAMIC"]
Decision = Literal[
    "CONFIRMED_VISIBLE",
    "LATENT_OUT_OF_VIEW",
    "PREDICTED_OCCLUDED",
    "VERIFYING",
    "AMBIGUOUS",
    "STALE",
    "REBOUND_TO_NEW_VALID_INSTANCE",
    "DISPROVED",
    "NONE",
]
ObservabilityReason = Literal[
    "IN_VIEW_RELIABLE",
    "PREDICTED_OUT_OF_FOV",
    "PREDICTED_OCCLUDED",
    "BELOW_RESOLUTION",
    "OBSERVATION_UNRELIABLE",
]
EvidenceRequest = Literal["NONE", "HOLD_STILL", "ROTATE_SCAN", "INCLUDE_PARENT_CONTEXT"]


@dataclass(frozen=True)
class GoalContract:
    goal_id: str
    reference_mode: ReferenceMode
    goal_predicate: str
    allowed_rebinding: bool
    arrival_predicate: str
    safety_constraints: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.goal_id or not self.goal_predicate or not self.arrival_predicate:
            raise ValueError("goal_id, goal_predicate, and arrival_predicate are required")
        if self.reference_mode != "SET_VALUED" and self.allowed_rebinding:
            raise ValueError("only SET_VALUED goals may allow rebinding")
        if not self.safety_constraints:
            raise ValueError("at least one safety constraint is required")


@dataclass(frozen=True)
class ParentAnchor:
    parent_hypothesis_id: str
    child_slot: str

    def __post_init__(self) -> None:
        if not self.parent_hypothesis_id or not self.child_slot:
            raise ValueError("parent hypothesis and child slot are required")


@dataclass(frozen=True)
class CandidateEvidence:
    evidence_id: str
    candidate_id: str
    entity_hypothesis_id: str
    proposal_source: Literal["DETECTOR", "TRACKER", "MATCHER", "VLM", "GEOMETRY"]
    candidate_region_xyxy: tuple[float, float, float, float]
    appearance_support: float
    appearance_contradiction: float
    spatial_prediction: TriState
    parent_slot: TriState
    relational_context: TriState
    distractor_exclusion: TriState
    current_goal_validity: TriState
    known_distractor: bool = False
    rebind_candidate: bool = False

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.candidate_id or not self.entity_hypothesis_id:
            raise ValueError("opaque evidence, candidate, and entity hypothesis ids are required")
        if not 0.0 <= self.appearance_support <= 1.0:
            raise ValueError("appearance_support must be in [0, 1]")
        if not 0.0 <= self.appearance_contradiction <= 1.0:
            raise ValueError("appearance_contradiction must be in [0, 1]")
        x1, y1, x2, y2 = self.candidate_region_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("candidate region must have positive area")


@dataclass(frozen=True)
class CandidateHypothesis:
    candidate_id: str
    entity_hypothesis_id: str
    status: Literal["CONFIRMED", "VERIFYING", "REJECTED"]
    independent_support_count: int
    appearance_contribution: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceReceipt:
    sequence_index: int
    decision: Decision
    observability_reason: ObservabilityReason
    candidate_ids: tuple[str, ...]
    confirmed_evidence_id: str | None
    prior_referent_id: str
    active_referent_id: str
    negative_evidence_counted: bool
    gallery_updated: bool


@dataclass(frozen=True)
class VerifierPolicy:
    max_hypotheses: int = 4
    appearance_cap: float = 0.35
    appearance_contradiction_limit: float = 0.50
    independent_support_required: int = 2
    reliable_misses_to_stale: int = 2
    maximum_unconfirmed_age: int = 6

    def __post_init__(self) -> None:
        if self.max_hypotheses < 1:
            raise ValueError("max_hypotheses must be positive")
        if not 0.0 <= self.appearance_cap <= 1.0:
            raise ValueError("appearance_cap must be in [0, 1]")
        if self.independent_support_required < 2:
            raise ValueError("identity confirmation requires at least two independent groups")
        if self.reliable_misses_to_stale < 1 or self.maximum_unconfirmed_age < 1:
            raise ValueError("staleness limits must be positive")


@dataclass(frozen=True)
class ReferentLedger:
    goal_contract: GoalContract
    referent_id: str
    motion_model: MotionModel
    parent_anchor: ParentAnchor | None
    decision: Decision
    hypotheses: tuple[CandidateHypothesis, ...]
    distractor_registry: tuple[str, ...]
    identity_gallery_evidence_ids: tuple[str, ...]
    evidence_history: tuple[EvidenceReceipt, ...]
    current_goal_validity: TriState
    reliable_miss_count: int
    last_confirmed_sequence: int
    evidence_request: EvidenceRequest

    @property
    def h_other(self) -> str:
        return "H_OTHER_REMAINS_POSSIBLE"

    def snapshot(self) -> dict:
        return {
            "goal_id": self.goal_contract.goal_id,
            "reference_mode": self.goal_contract.reference_mode,
            "referent_id": self.referent_id,
            "motion_model": self.motion_model,
            "parent_anchor": None if self.parent_anchor is None else {
                "parent_hypothesis_id": self.parent_anchor.parent_hypothesis_id,
                "child_slot": self.parent_anchor.child_slot,
            },
            "decision": self.decision,
            "h_other": self.h_other,
            "hypotheses": [hypothesis.__dict__ for hypothesis in self.hypotheses],
            "distractor_registry": list(self.distractor_registry),
            "identity_gallery_evidence_ids": list(self.identity_gallery_evidence_ids),
            "current_goal_validity": self.current_goal_validity,
            "reliable_miss_count": self.reliable_miss_count,
            "last_confirmed_sequence": self.last_confirmed_sequence,
            "evidence_request": self.evidence_request,
            "directional_guidance_authorized": self.decision == "CONFIRMED_VISIBLE",
        }


def initialize_ledger(
    goal_contract: GoalContract,
    referent_id: str,
    *,
    motion_model: MotionModel,
    parent_anchor: ParentAnchor | None = None,
    sequence_index: int = 0,
) -> ReferentLedger:
    if not referent_id:
        raise ValueError("an established opaque referent_id is required")
    if sequence_index < 0:
        raise ValueError("sequence_index must be non-negative")
    return ReferentLedger(
        goal_contract=goal_contract,
        referent_id=referent_id,
        motion_model=motion_model,
        parent_anchor=parent_anchor,
        decision="NONE",
        hypotheses=(),
        distractor_registry=(),
        identity_gallery_evidence_ids=(),
        evidence_history=(),
        current_goal_validity="INSUFFICIENT",
        reliable_miss_count=0,
        last_confirmed_sequence=sequence_index,
        evidence_request="NONE",
    )


def _assess(candidate: CandidateEvidence, policy: VerifierPolicy) -> CandidateHypothesis:
    predictive = (
        candidate.spatial_prediction,
        candidate.parent_slot,
        candidate.relational_context,
    )
    independent_support_count = sum(
        value == "SUPPORTED" for value in (*predictive, candidate.distractor_exclusion)
    )
    reasons: list[str] = []
    if candidate.known_distractor:
        reasons.append("KNOWN_DISTRACTOR")
    if candidate.distractor_exclusion == "REJECTED":
        reasons.append("EXCLUSION_CONTRADICTION")
    if candidate.appearance_contradiction > policy.appearance_contradiction_limit:
        reasons.append("APPEARANCE_CONTRADICTION")
    if reasons:
        status: Literal["CONFIRMED", "VERIFYING", "REJECTED"] = "REJECTED"
    else:
        prediction_supported = any(value == "SUPPORTED" for value in predictive)
        exclusion_supported = candidate.distractor_exclusion == "SUPPORTED"
        confirmed = (
            prediction_supported
            and exclusion_supported
            and independent_support_count >= policy.independent_support_required
        )
        status = "CONFIRMED" if confirmed else "VERIFYING"
        if not prediction_supported:
            reasons.append("NO_INDEPENDENT_PREDICTION_SUPPORT")
        if not exclusion_supported:
            reasons.append("H_OTHER_NOT_EXCLUDED")
        if independent_support_count < policy.independent_support_required:
            reasons.append("EVIDENCE_BUDGET_INSUFFICIENT")
    return CandidateHypothesis(
        candidate_id=candidate.candidate_id,
        entity_hypothesis_id=candidate.entity_hypothesis_id,
        status=status,
        independent_support_count=independent_support_count,
        appearance_contribution=min(candidate.appearance_support, policy.appearance_cap),
        reasons=tuple(reasons),
    )


def _evidence_request(
    decision: Decision,
    observability_reason: ObservabilityReason,
    *,
    parent_anchor: ParentAnchor | None,
    parent_context_visible: bool,
) -> EvidenceRequest:
    if decision not in {"VERIFYING", "AMBIGUOUS", "STALE"}:
        return "NONE"
    if observability_reason == "OBSERVATION_UNRELIABLE":
        return "HOLD_STILL"
    if parent_anchor is not None and not parent_context_visible:
        return "INCLUDE_PARENT_CONTEXT"
    return "ROTATE_SCAN"


def update_ledger(
    ledger: ReferentLedger,
    *,
    sequence_index: int,
    observability_reason: ObservabilityReason,
    candidates: tuple[CandidateEvidence, ...],
    parent_context_visible: bool = False,
    policy: VerifierPolicy = VerifierPolicy(),
) -> ReferentLedger:
    """Return a new immutable ledger after one causal verification event."""
    previous_sequence = (
        ledger.evidence_history[-1].sequence_index
        if ledger.evidence_history
        else ledger.last_confirmed_sequence
    )
    if sequence_index <= previous_sequence:
        raise ValueError("sequence_index must advance monotonically")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    evidence_ids = [candidate.evidence_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)) or len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("candidate_id and evidence_id must be unique within an event")

    assessed = [(candidate, _assess(candidate, policy)) for candidate in candidates]
    status_order = {"CONFIRMED": 0, "VERIFYING": 1, "REJECTED": 2}
    ordered = sorted(
        assessed,
        key=lambda item: (
            status_order[item[1].status],
            -item[1].independent_support_count,
            item[0].candidate_id,
        ),
    )
    hypotheses = tuple(item[1] for item in ordered[: policy.max_hypotheses])
    confirmed = [item for item in assessed if item[1].status == "CONFIRMED"]
    verifying = [item for item in assessed if item[1].status == "VERIFYING"]
    prior_referent_id = ledger.referent_id
    active_referent_id = prior_referent_id
    goal_validity: TriState = "INSUFFICIENT"
    confirmed_evidence_id: str | None = None
    gallery_updated = False
    negative_evidence_counted = False
    miss_count = ledger.reliable_miss_count
    last_confirmed = ledger.last_confirmed_sequence

    if ledger.goal_contract.reference_mode == "AMBIGUOUS":
        decision: Decision = "AMBIGUOUS"
    elif len(confirmed) > 1:
        decision = "AMBIGUOUS"
    elif len(confirmed) == 1:
        candidate, _ = confirmed[0]
        goal_validity = candidate.current_goal_validity
        if candidate.entity_hypothesis_id != prior_referent_id:
            may_rebind = (
                ledger.goal_contract.reference_mode == "SET_VALUED"
                and ledger.goal_contract.allowed_rebinding
                and candidate.rebind_candidate
                and goal_validity == "SUPPORTED"
            )
            if may_rebind:
                decision = "REBOUND_TO_NEW_VALID_INSTANCE"
                active_referent_id = candidate.entity_hypothesis_id
            else:
                decision = "AMBIGUOUS"
        elif goal_validity == "REJECTED":
            decision = "DISPROVED"
        elif goal_validity == "INSUFFICIENT":
            decision = "VERIFYING"
        else:
            decision = "CONFIRMED_VISIBLE"
        if decision in {"CONFIRMED_VISIBLE", "REBOUND_TO_NEW_VALID_INSTANCE", "DISPROVED", "VERIFYING"}:
            confirmed_evidence_id = candidate.evidence_id
            gallery_updated = candidate.evidence_id not in ledger.identity_gallery_evidence_ids
            last_confirmed = sequence_index
            miss_count = 0
    elif verifying:
        decision = "AMBIGUOUS" if len(verifying) > 1 else "VERIFYING"
    elif observability_reason == "PREDICTED_OUT_OF_FOV":
        decision = "LATENT_OUT_OF_VIEW"
    elif observability_reason == "PREDICTED_OCCLUDED":
        decision = "PREDICTED_OCCLUDED"
    elif observability_reason in {"BELOW_RESOLUTION", "OBSERVATION_UNRELIABLE"}:
        decision = "VERIFYING"
    else:
        negative_evidence_counted = True
        miss_count += 1
        decision = "STALE" if miss_count >= policy.reliable_misses_to_stale else "VERIFYING"

    if sequence_index - last_confirmed > policy.maximum_unconfirmed_age and decision not in {
        "CONFIRMED_VISIBLE", "REBOUND_TO_NEW_VALID_INSTANCE", "DISPROVED"
    }:
        decision = "STALE"

    gallery = ledger.identity_gallery_evidence_ids
    if gallery_updated and confirmed_evidence_id is not None:
        gallery = (*gallery, confirmed_evidence_id)

    distractors = set(ledger.distractor_registry)
    distractors.update(
        candidate.entity_hypothesis_id
        for candidate, hypothesis in assessed
        if hypothesis.status == "REJECTED"
    )
    if confirmed_evidence_id is not None:
        distractors.update(
            candidate.entity_hypothesis_id
            for candidate, hypothesis in assessed
            if hypothesis.status != "CONFIRMED"
        )
    distractors.discard(active_referent_id)

    receipt = EvidenceReceipt(
        sequence_index=sequence_index,
        decision=decision,
        observability_reason=observability_reason,
        candidate_ids=tuple(candidate_ids),
        confirmed_evidence_id=confirmed_evidence_id,
        prior_referent_id=prior_referent_id,
        active_referent_id=active_referent_id,
        negative_evidence_counted=negative_evidence_counted,
        gallery_updated=gallery_updated,
    )
    request = _evidence_request(
        decision,
        observability_reason,
        parent_anchor=ledger.parent_anchor,
        parent_context_visible=parent_context_visible,
    )
    return replace(
        ledger,
        referent_id=active_referent_id,
        decision=decision,
        hypotheses=hypotheses,
        distractor_registry=tuple(sorted(distractors)),
        identity_gallery_evidence_ids=gallery,
        evidence_history=(*ledger.evidence_history, receipt),
        current_goal_validity=goal_validity,
        reliable_miss_count=miss_count,
        last_confirmed_sequence=last_confirmed,
        evidence_request=request,
    )
