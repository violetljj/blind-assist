"""Adaptive multi-view memory built only from verifier-authorized evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Literal

from .core import ReferentLedger


DistanceBand = Literal["FAR", "MID", "NEAR", "CONTACT"]
ViewpointBin = Literal["LEFT", "FRONTAL", "RIGHT", "REAR_OR_UNKNOWN"]
ScaleBand = Literal["SMALL", "MEDIUM", "LARGE"]
MemoryAction = Literal[
    "TENTATIVE_BUFFERED",
    "VERIFIED_ADMITTED",
    "VERIFIED_REDUNDANT_DROPPED",
    "OBSERVATION_NOT_WRITTEN",
    "REFERENT_REBOUND_REQUIRES_NEW_BANK",
]


@dataclass(frozen=True)
class MemoryObservation:
    evidence_id: str
    candidate_id: str
    referent_id: str
    frame_id: str
    target_crop_ref: str
    context_crop_ref: str
    full_frame_ref: str
    orientation_source: Literal["IMU", "VISUAL", "UNKNOWN"]
    orientation_yaw_deg: float | None
    distance_band: DistanceBand
    viewpoint_bin: ViewpointBin
    scale_band: ScaleBand
    context_anchor_id: str

    def __post_init__(self) -> None:
        required = (
            self.evidence_id,
            self.candidate_id,
            self.referent_id,
            self.frame_id,
            self.target_crop_ref,
            self.context_crop_ref,
            self.full_frame_ref,
            self.context_anchor_id,
        )
        if any(not value for value in required):
            raise ValueError("all observation ids and immutable asset refs are required")
        if self.orientation_source == "IMU" and self.orientation_yaw_deg is None:
            raise ValueError("IMU orientation requires yaw")
        if self.orientation_yaw_deg is not None:
            if not isfinite(self.orientation_yaw_deg) or not -180.0 <= self.orientation_yaw_deg <= 180.0:
                raise ValueError("orientation_yaw_deg must be finite and in [-180, 180]")

    @property
    def coverage_cell(self) -> tuple[DistanceBand, ViewpointBin, ScaleBand]:
        return self.distance_band, self.viewpoint_bin, self.scale_band

    @property
    def memory_signature(self) -> tuple[DistanceBand, ViewpointBin, ScaleBand, str]:
        return (*self.coverage_cell, self.context_anchor_id)


@dataclass(frozen=True)
class MemoryEntry:
    observation: MemoryObservation
    admitted_sequence: int


@dataclass(frozen=True)
class MemoryReceipt:
    sequence_index: int
    evidence_id: str
    action: MemoryAction
    verified_count: int
    tentative_count: int
    evicted_evidence_id: str | None = None


@dataclass(frozen=True)
class MemoryPolicy:
    max_verified_entries: int = 12
    max_tentative_entries: int = 4

    def __post_init__(self) -> None:
        if self.max_verified_entries < 1 or self.max_tentative_entries < 1:
            raise ValueError("memory capacities must be positive")


@dataclass(frozen=True)
class AdaptiveMultiViewMemory:
    referent_id: str
    verified: tuple[MemoryEntry, ...] = ()
    tentative: tuple[MemoryEntry, ...] = ()
    receipts: tuple[MemoryReceipt, ...] = ()

    @property
    def coverage_cells(self) -> tuple[tuple[DistanceBand, ViewpointBin, ScaleBand], ...]:
        return tuple(entry.observation.coverage_cell for entry in self.verified)

    @property
    def memory_signatures(self) -> tuple[tuple[DistanceBand, ViewpointBin, ScaleBand, str], ...]:
        return tuple(entry.observation.memory_signature for entry in self.verified)

    def retrieval_packet(self) -> tuple[dict, ...]:
        """Expose verified views only; tentative observations never become retrieval evidence."""
        return tuple(asdict(entry.observation) for entry in self.verified)


def initialize_memory(referent_id: str) -> AdaptiveMultiViewMemory:
    if not referent_id:
        raise ValueError("an established opaque referent_id is required")
    return AdaptiveMultiViewMemory(referent_id=referent_id)


def _append_receipt(
    memory: AdaptiveMultiViewMemory,
    *,
    sequence_index: int,
    evidence_id: str,
    action: MemoryAction,
    evicted_evidence_id: str | None = None,
) -> AdaptiveMultiViewMemory:
    receipt = MemoryReceipt(
        sequence_index=sequence_index,
        evidence_id=evidence_id,
        action=action,
        verified_count=len(memory.verified),
        tentative_count=len(memory.tentative),
        evicted_evidence_id=evicted_evidence_id,
    )
    return replace(memory, receipts=(*memory.receipts, receipt))


def _eviction_index(
    verified: tuple[MemoryEntry, ...],
    incoming: MemoryObservation,
) -> int:
    """Choose the removal that retains the widest categorical view coverage."""
    choices: list[tuple[tuple[int, int, int], int]] = []
    for index, entry in enumerate(verified):
        remaining = [item.observation for offset, item in enumerate(verified) if offset != index]
        remaining.append(incoming)
        diversity = (
            len({item.distance_band for item in remaining})
            + len({item.viewpoint_bin for item in remaining})
            + len({item.scale_band for item in remaining})
            + len({item.context_anchor_id for item in remaining})
        )
        preserves_initial = 0 if index == 0 else 1
        oldest_first = -entry.admitted_sequence
        choices.append(((diversity, preserves_initial, oldest_first), index))
    return max(choices)[1]


def record_observation(
    memory: AdaptiveMultiViewMemory,
    ledger: ReferentLedger,
    observation: MemoryObservation,
    *,
    sequence_index: int,
    policy: MemoryPolicy = MemoryPolicy(),
) -> AdaptiveMultiViewMemory:
    """Buffer or admit one observation using only the ledger's latest verifier event."""
    if sequence_index < 1:
        raise ValueError("sequence_index must be positive")
    if not ledger.evidence_history or ledger.evidence_history[-1].sequence_index != sequence_index:
        raise ValueError("memory writes require the matching latest verifier event")
    if observation.referent_id != memory.referent_id:
        raise ValueError("observation referent does not match this memory bank")
    if ledger.referent_id != memory.referent_id:
        return _append_receipt(
            memory,
            sequence_index=sequence_index,
            evidence_id=observation.evidence_id,
            action="REFERENT_REBOUND_REQUIRES_NEW_BANK",
        )

    latest = ledger.evidence_history[-1]
    known_candidate = any(
        hypothesis.candidate_id == observation.candidate_id for hypothesis in ledger.hypotheses
    )
    in_view_reliable = latest.observability_reason == "IN_VIEW_RELIABLE"
    verifier_confirmed = latest.confirmed_evidence_id == observation.evidence_id and in_view_reliable

    if verifier_confirmed:
        if observation.evidence_id in {
            entry.observation.evidence_id for entry in memory.verified
        }:
            return _append_receipt(
                memory,
                sequence_index=sequence_index,
                evidence_id=observation.evidence_id,
                action="VERIFIED_REDUNDANT_DROPPED",
            )
        tentative = tuple(
            entry for entry in memory.tentative if entry.observation.evidence_id != observation.evidence_id
        )
        if observation.memory_signature in memory.memory_signatures:
            unchanged = replace(memory, tentative=tentative)
            return _append_receipt(
                unchanged,
                sequence_index=sequence_index,
                evidence_id=observation.evidence_id,
                action="VERIFIED_REDUNDANT_DROPPED",
            )
        verified = list(memory.verified)
        evicted_evidence_id = None
        if len(verified) >= policy.max_verified_entries:
            evicted_index = _eviction_index(memory.verified, observation)
            evicted_evidence_id = verified[evicted_index].observation.evidence_id
            del verified[evicted_index]
        verified.append(MemoryEntry(observation=observation, admitted_sequence=sequence_index))
        updated = replace(memory, verified=tuple(verified), tentative=tentative)
        return _append_receipt(
            updated,
            sequence_index=sequence_index,
            evidence_id=observation.evidence_id,
            action="VERIFIED_ADMITTED",
            evicted_evidence_id=evicted_evidence_id,
        )

    may_buffer = (
        in_view_reliable
        and ledger.decision in {"VERIFYING", "AMBIGUOUS"}
        and known_candidate
    )
    if may_buffer:
        tentative = [
            entry for entry in memory.tentative if entry.observation.evidence_id != observation.evidence_id
        ]
        tentative.append(MemoryEntry(observation=observation, admitted_sequence=sequence_index))
        tentative = tentative[-policy.max_tentative_entries :]
        updated = replace(memory, tentative=tuple(tentative))
        return _append_receipt(
            updated,
            sequence_index=sequence_index,
            evidence_id=observation.evidence_id,
            action="TENTATIVE_BUFFERED",
        )

    return _append_receipt(
        memory,
        sequence_index=sequence_index,
        evidence_id=observation.evidence_id,
        action="OBSERVATION_NOT_WRITTEN",
    )
