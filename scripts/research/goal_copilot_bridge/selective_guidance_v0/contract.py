"""Pure current-frame V0 decision and completion-authority contract.

The module deliberately has no mutable target memory. Every decision is derived
from one frame-bound observation plus an optional user/interaction receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class CandidateCardinality(str, Enum):
    UNIQUE = "UNIQUE"
    SET_VALUED = "SET_VALUED"
    AMBIGUOUS = "AMBIGUOUS"


class OutputToken(str, Enum):
    FOUND = "FOUND"
    CONTESTED = "CONTESTED"
    NOT_VISIBLE = "NOT_VISIBLE"
    LOST = "LOST"
    STALE = "STALE"
    ABSTAIN = "ABSTAIN"
    GUIDE_LEFT = "GUIDE_LEFT"
    GUIDE_RIGHT = "GUIDE_RIGHT"
    GUIDE_STRAIGHT = "GUIDE_STRAIGHT"
    STOP_FOR_SAFETY = "STOP_FOR_SAFETY"
    HANDOFF_READY = "HANDOFF_READY"
    COMPLETED_BY_USER = "COMPLETED_BY_USER"


class RangeBucket(str, Enum):
    FAR = "RANGE_FAR"
    APPROACHING = "RANGE_APPROACHING"
    NEAR = "RANGE_NEAR"
    UNKNOWN = "RANGE_UNKNOWN"


class EpisodeVisibilityState(str, Enum):
    NEVER_SEEN = "NEVER_SEEN"
    VISIBLE = "VISIBLE"
    NOT_VISIBLE_AFTER_VISIBLE = "NOT_VISIBLE_AFTER_VISIBLE"


class CompletionAuthority(str, Enum):
    USER_EXPLICIT = "USER_EXPLICIT"
    TRUSTED_INTERACTION = "TRUSTED_INTERACTION"
    PERCEPTION = "PERCEPTION"
    PROVIDER = "PROVIDER"
    CONTROLLER = "CONTROLLER"


@dataclass(frozen=True)
class ProviderReceipt:
    provider: str
    receipt_id: str
    provenance: str
    frame_id: str


@dataclass(frozen=True)
class CompletionReceipt:
    authority: CompletionAuthority
    event_id: str
    occurred_at_ms: int
    contract_id: str | None = None

    def __post_init__(self) -> None:
        allowed = {
            CompletionAuthority.USER_EXPLICIT,
            CompletionAuthority.TRUSTED_INTERACTION,
        }
        if self.authority not in allowed:
            raise ValueError(
                "completion requires explicit user confirmation or a contracted trusted interaction"
            )
        if self.authority is CompletionAuthority.TRUSTED_INTERACTION and not self.contract_id:
            raise ValueError("trusted interaction completion requires contract_id")
        if not self.event_id:
            raise ValueError("completion receipt event_id is required")


_DIRECTIONAL = {
    OutputToken.GUIDE_LEFT,
    OutputToken.GUIDE_RIGHT,
    OutputToken.GUIDE_STRAIGHT,
}


@dataclass(frozen=True)
class CurrentFrameObservation:
    goal_contract: Mapping[str, Any]
    frame_id: str
    observed_at_ms: int
    decision_at_ms: int
    visible_candidate_ids: tuple[str, ...]
    selected_referent: str | None
    cardinality: CandidateCardinality
    target_visible: bool | None
    selection_authorized: bool
    requested_direction: OutputToken | None
    range_bucket: RangeBucket = RangeBucket.UNKNOWN
    range_uncertainty: float | None = None
    evidence_ttl_ms: int = 1_000
    stop_for_safety: bool = False
    handoff_ready: bool = False
    completion_receipt: CompletionReceipt | None = None
    provider_receipts: tuple[ProviderReceipt, ...] = ()
    latency_ms: float | None = None

    def validate(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id is required")
        if self.decision_at_ms < self.observed_at_ms:
            raise ValueError("decision time precedes observation")
        if self.evidence_ttl_ms < 0:
            raise ValueError("evidence_ttl_ms must be non-negative")
        if self.requested_direction is not None and self.requested_direction not in _DIRECTIONAL:
            raise ValueError("requested_direction must be a GUIDE_* token")
        if self.selected_referent is not None and self.selected_referent not in self.visible_candidate_ids:
            raise ValueError("selected referent must belong to the current visible candidate set")
        if self.range_uncertainty is not None and self.range_uncertainty < 0:
            raise ValueError("range uncertainty must be non-negative")
        for receipt in self.provider_receipts:
            if receipt.frame_id != self.frame_id:
                raise ValueError("provider receipt is not bound to the current frame")


@dataclass(frozen=True)
class GuidanceDecision:
    status: OutputToken
    command: OutputToken | None
    range_bucket: RangeBucket
    selected_referent: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.status in _DIRECTIONAL or self.status is OutputToken.STOP_FOR_SAFETY:
            raise ValueError("status and command responsibilities must remain separate")
        if self.status is OutputToken.LOST:
            raise ValueError("LOST is an episode interaction event, not a current-frame status")
        if self.command is not None and self.command not in _DIRECTIONAL | {OutputToken.STOP_FOR_SAFETY}:
            raise ValueError("invalid guidance command")
        if self.status is OutputToken.COMPLETED_BY_USER and self.command is not None:
            raise ValueError("completed decision cannot issue another command")

    @property
    def tokens(self) -> tuple[str, ...]:
        values = [self.status.value]
        if self.command is not None:
            values.append(self.command.value)
        values.append(self.range_bucket.value)
        return tuple(values)


@dataclass(frozen=True)
class EpisodeInteractionDecision:
    visibility_state: EpisodeVisibilityState
    event: OutputToken | None

    def __post_init__(self) -> None:
        if self.event not in {None, OutputToken.LOST}:
            raise ValueError("episode interaction event must be LOST or absent")


def derive_episode_interaction(
    previous: EpisodeVisibilityState,
    current: GuidanceDecision,
) -> EpisodeInteractionDecision:
    """Derive LOST from a frame-state transition without retaining target identity."""

    if current.status in {OutputToken.FOUND, OutputToken.HANDOFF_READY}:
        return EpisodeInteractionDecision(EpisodeVisibilityState.VISIBLE, None)
    if current.status is OutputToken.NOT_VISIBLE:
        if previous is EpisodeVisibilityState.VISIBLE:
            return EpisodeInteractionDecision(
                EpisodeVisibilityState.NOT_VISIBLE_AFTER_VISIBLE,
                OutputToken.LOST,
            )
        if previous is EpisodeVisibilityState.NOT_VISIBLE_AFTER_VISIBLE:
            return EpisodeInteractionDecision(previous, None)
        return EpisodeInteractionDecision(EpisodeVisibilityState.NEVER_SEEN, None)
    return EpisodeInteractionDecision(previous, None)


def decide(observation: CurrentFrameObservation) -> GuidanceDecision:
    """Return a fail-closed decision using only the supplied current frame."""

    observation.validate()
    if observation.completion_receipt is not None:
        return GuidanceDecision(
            OutputToken.COMPLETED_BY_USER,
            None,
            observation.range_bucket,
            observation.selected_referent,
            "authorized completion receipt",
        )

    age_ms = observation.decision_at_ms - observation.observed_at_ms
    if age_ms > observation.evidence_ttl_ms:
        return GuidanceDecision(
            OutputToken.STALE, None, RangeBucket.UNKNOWN, None, "current-frame evidence expired"
        )
    if observation.target_visible is False:
        return GuidanceDecision(
            OutputToken.NOT_VISIBLE,
            None,
            RangeBucket.UNKNOWN,
            None,
            "target not visible in current frame; LOST requires an episode transition",
        )
    if observation.stop_for_safety:
        return GuidanceDecision(
            OutputToken.FOUND if observation.selected_referent else OutputToken.ABSTAIN,
            OutputToken.STOP_FOR_SAFETY,
            observation.range_bucket,
            observation.selected_referent,
            "safety stop is a control action, not arrival",
        )

    multiple_without_authority = (
        (
            len(observation.visible_candidate_ids) > 1
            or observation.cardinality is CandidateCardinality.SET_VALUED
        )
        and not observation.selection_authorized
    )
    if observation.cardinality is CandidateCardinality.AMBIGUOUS or multiple_without_authority:
        return GuidanceDecision(
            OutputToken.CONTESTED,
            None,
            observation.range_bucket,
            None,
            "candidate authority is contested",
        )
    if observation.selected_referent is None or not observation.selection_authorized:
        return GuidanceDecision(
            OutputToken.ABSTAIN,
            None,
            observation.range_bucket,
            None,
            "no authorized current-frame referent",
        )
    if observation.handoff_ready:
        return GuidanceDecision(
            OutputToken.HANDOFF_READY,
            None,
            observation.range_bucket,
            observation.selected_referent,
            "local handoff range reached; user confirmation still required",
        )
    return GuidanceDecision(
        OutputToken.FOUND,
        observation.requested_direction,
        observation.range_bucket,
        observation.selected_referent,
        "authorized current-frame referent",
    )
