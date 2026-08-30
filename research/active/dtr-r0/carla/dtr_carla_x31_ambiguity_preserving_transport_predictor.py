"""Freeze and predict X31 ambiguity-preserving surface-transport ancestry.

X31 changes the temporal representation, not detector, association, route, or
decision thresholds.  X30 reduced every measured world-surface component to a
single quantized centre translation.  A one-frame partial-surface alias could
therefore contradict an otherwise causal motion history and permanently end
the complete track's motion epoch.

This successor retains the supported hypotheses in X27's finite inherited
occupancy-lattice neighbourhood as a transport cone.  Each
direction-consistent branch owns its aligned surface lineage.  Equal motion
signatures coalesce into conservative timestamp and lineage envelopes rather
than beam-pruning feasible ancestry.  A contradiction terminates only that
branch; motion authority ends only when every feasible continuation has ended.
Route risk consumes every still-authorized branch footprint and velocity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_rgbd_model_adapter as adapter  # noqa: E402
import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x24_plan_route_core as route  # noqa: E402
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x29_temporal_occupancy_lineage_predictor as x29  # noqa: E402
import dtr_carla_x30_adaptive_surface_interval_predictor as x30  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X31_AMBIGUITY_PRESERVING_SURFACE_TRANSPORT"
FREEZE_SCHEMA = (
    "blindassist-dtr-carla-x31-ambiguity-preserving-transport-freeze-v1"
)
PREDICTION_SCHEMA = (
    "blindassist-dtr-carla-x31-ambiguity-preserving-transport-predictions-v1"
)
ARM_X31 = "X31_ISSUED_PLAN_SET_VALUED_SURFACE_TRANSPORT_ANCESTRY"
EPSILON = 1e-9

BRANCH_ANCHOR = "OCCUPANCY_PEAK_ANCHORED_BRANCH"
BRANCH_CONTINUATION = "DIRECTION_CONSISTENT_BRANCH_CONTINUATION"
BRANCH_NEUTRAL = "ZERO_SHIFT_SURFACE_SUPPORT_CARRY"


@dataclass(frozen=True)
class SurfaceTransportCandidate:
    shift_cells: x27.Cell
    overlap_cells: int
    zero_overlap_cells: int
    support_cells: int
    center_residual_cells: float
    occupancy_anchored: bool
    is_center_hypothesis: bool


@dataclass(frozen=True)
class BranchEvidence:
    earliest_time_s: float
    time_s: float
    delta_s: float
    shift_cells: x27.Cell
    occupancy_anchored: bool
    overlap_cells: int


@dataclass
class TransportBranch:
    evidence: list[BranchEvidence]
    anchor_times_s: list[float]
    world_lineage: list[frozenset[x27.Cell]]
    last_shift_cells: x27.Cell
    last_state: str


@dataclass
class SurfaceTransportTrack(x29.LineageTrack):
    transport_branches: dict[str, TransportBranch] = field(default_factory=dict)
    branches_created: int = 0
    branches_terminated: int = 0
    ambiguity_frames: int = 0
    maximum_branch_count: int = 0


def fixed_constants() -> dict[str, Any]:
    return {
        **x30.fixed_constants(),
        "representation": "AMBIGUITY_PRESERVING_SET_VALUED_SURFACE_TRANSPORT_ANCESTRY",
        "surface_transport_candidates": (
            "INHERITED_X27_CENTER_NEIGHBOUR_PLUS_LIVE_BRANCH_PREDICTIONS"
        ),
        "surface_transport_support_floor": x27.MINIMUM_ALIGNMENT_CELLS,
        "surface_transport_radius_m": x24.ASSOCIATION_DISTANCE_M,
        "branch_authority_pairs": x27.MINIMUM_AUTHORITY_PAIRS,
        "branch_authority_span_seconds": x24.MINIMUM_FIT_SPAN_S,
        "branch_motion_witness_steps": x27.MINIMUM_AUTHORITY_PAIRS,
        "branch_anchor_rule": (
            "AT_LEAST_ONE_NONZERO_PEAK_BEATS_ZERO_SHIFT_WITH_INHERITED_SUPPORT"
        ),
        "branch_continuation_rule": "STRICTLY_POSITIVE_DOT_WITH_BRANCH_MEDIAN_VELOCITY",
        "branch_conflict_rule": (
            "TERMINATE_ONLY_INCOMPATIBLE_BRANCH_AND_END_EPOCH_ONLY_IF_ALL_BRANCHES_END"
        ),
        "neutral_observation_rule": (
            "ZERO_SHIFT_WITH_INHERITED_SUPPORT_ADDS_PARALLEL_CARRY_WITHOUT_NEW_MOTION_EVIDENCE"
        ),
        "branch_merge_rule": (
            "EXACT_INTERNING_THEN_FUTURE_STABLE_DOMINANCE_THEN_EQUAL_MOTION_SIGNATURE_CONSERVATIVE_ENVELOPE"
        ),
        "ambiguity_envelope": (
            "TIMESTAMP_INTERVALS_PLUS_COMPONENTWISE_LINEAGE_UNION_PRESERVE_EXISTENTIAL_RISK"
        ),
        "route_consumption": "MINIMUM_ENTRY_ACROSS_ALL_AUTHORIZED_TRANSPORT_BRANCHES",
        "confirmation_identity": "STABLE_PARENT_TRACK_ANCESTRY_NOT_BRANCH_VELOCITY_KEY",
        "detector_threshold_change": False,
        "association_threshold_change": False,
        "route_threshold_change": False,
    }


def paths(run_root: Path) -> dict[str, Path]:
    return {
        "x24_freeze": run_root / "freeze-x24.json",
        "x24_predictions": run_root / "predictions-x24.json",
        "freeze": run_root / "freeze-x31.json",
        "predictions": run_root / "predictions-x31.json",
    }


def surface_transport_candidates(
    previous: frozenset[x27.Cell],
    current: frozenset[x27.Cell],
    center_delta_xy: np.ndarray,
    branch_predictions: Sequence[x27.Cell] = (),
) -> list[SurfaceTransportCandidate]:
    """Retain inherited local hypotheses and each live branch prediction."""

    support = min(len(previous), len(current))
    zero_overlap = len(previous & current)
    center_shift, _center_xy = x29.quantized_transport(center_delta_xy)
    center_universe: set[x27.Cell] = set()
    for dx in range(
        -x27.ALIGNMENT_NEIGHBOUR_CELLS,
        x27.ALIGNMENT_NEIGHBOUR_CELLS + 1,
    ):
        for dy in range(
            -x27.ALIGNMENT_NEIGHBOUR_CELLS,
            x27.ALIGNMENT_NEIGHBOUR_CELLS + 1,
        ):
            shift = (center_shift[0] + dx, center_shift[1] + dy)
            if (
                shift != (0, 0)
                and math.hypot(*shift) * x27.LATTICE_CELL_SIZE_M
                <= x24.ASSOCIATION_DISTANCE_M + EPSILON
            ):
                center_universe.add(shift)

    predicted = {
        shift
        for shift in branch_predictions
        if shift != (0, 0)
        and math.hypot(*shift) * x27.LATTICE_CELL_SIZE_M
        <= x24.ASSOCIATION_DISTANCE_M + EPSILON
    }
    universe = center_universe | predicted

    overlaps = {
        shift: len(x27.shifted(previous, shift) & current) for shift in universe
    }
    retained = {
        shift
        for shift, overlap in overlaps.items()
        if overlap >= x27.MINIMUM_ALIGNMENT_CELLS and overlap > zero_overlap
    }
    retained.update(
        shift
        for shift in predicted
        if overlaps[shift] >= x27.MINIMUM_ALIGNMENT_CELLS
    )
    if center_shift != (0, 0):
        retained.add(center_shift)

    values = [
        SurfaceTransportCandidate(
            shift_cells=shift,
            overlap_cells=int(overlaps.get(shift, 0)),
            zero_overlap_cells=zero_overlap,
            support_cells=support,
            center_residual_cells=math.dist(shift, center_shift),
            occupancy_anchored=(
                int(overlaps.get(shift, 0)) >= x27.MINIMUM_ALIGNMENT_CELLS
                and int(overlaps.get(shift, 0)) > zero_overlap
            ),
            is_center_hypothesis=shift == center_shift,
        )
        for shift in retained
    ]
    return sorted(
        values,
        key=lambda value: (
            -value.overlap_cells,
            value.center_residual_cells,
            math.hypot(*value.shift_cells),
            value.shift_cells,
        ),
    )


def evidence_velocity(value: BranchEvidence) -> np.ndarray:
    return (
        np.asarray(value.shift_cells, dtype=np.float64)
        * x27.LATTICE_CELL_SIZE_M
        / value.delta_s
    )


def _trimmed_evidence(
    evidence: Sequence[BranchEvidence], now_s: float
) -> list[BranchEvidence]:
    window = [
        value
        for value in evidence
        if now_s - value.time_s <= x24.TRACK_HISTORY_S + EPSILON
    ]
    return window[-x27.MINIMUM_AUTHORITY_PAIRS :]


def _trimmed_anchor_times(
    anchor_times_s: Sequence[float], now_s: float
) -> list[float]:
    return [
        value
        for value in anchor_times_s
        if now_s - value <= x24.TRACK_HISTORY_S + EPSILON
    ]


def _candidate_consistent(
    evidence: Sequence[BranchEvidence],
    candidate: SurfaceTransportCandidate,
    delta_s: float,
) -> bool:
    if not evidence:
        return True
    reference = np.median(
        np.stack([evidence_velocity(value) for value in evidence]), axis=0
    )
    candidate_velocity = (
        np.asarray(candidate.shift_cells, dtype=np.float64)
        * x27.LATTICE_CELL_SIZE_M
        / delta_s
    )
    return float(np.dot(candidate_velocity, reference)) > EPSILON


def branch_velocity_key(evidence: Sequence[BranchEvidence]) -> x27.Cell:
    velocities_cells_per_s = np.stack(
        [
            np.asarray(value.shift_cells, dtype=np.float64) / value.delta_s
            for value in evidence
        ]
    )
    median = np.rint(np.median(velocities_cells_per_s, axis=0)).astype(np.int64)
    return int(median[0]), int(median[1])


def resolve_branch_authority(
    branch: TransportBranch, now_s: float
) -> tuple[str, np.ndarray]:
    evidence = _trimmed_evidence(branch.evidence, now_s)
    anchor_times_s = _trimmed_anchor_times(branch.anchor_times_s, now_s)
    if (
        len(evidence) < x27.MINIMUM_AUTHORITY_PAIRS
        or evidence[-1].time_s
        - evidence[-x27.MINIMUM_AUTHORITY_PAIRS].earliest_time_s
        < x24.MINIMUM_FIT_SPAN_S - EPSILON
        or not anchor_times_s
    ):
        return x27.UNAUTHORIZED_MOTION, np.zeros(2, dtype=np.float64)
    velocities = np.stack([evidence_velocity(value) for value in evidence])
    median = np.median(velocities, axis=0)
    if float(np.linalg.norm(median)) <= EPSILON or not all(
        float(np.dot(value, median)) > EPSILON for value in velocities
    ):
        return x27.UNAUTHORIZED_MOTION, np.zeros(2, dtype=np.float64)
    return x27.RIGID_DYNAMIC, median


def _branch_priority(branch: TransportBranch, now_s: float) -> tuple[Any, ...]:
    evidence = _trimmed_evidence(branch.evidence, now_s)
    anchor_times_s = _trimmed_anchor_times(branch.anchor_times_s, now_s)
    authority, _velocity = resolve_branch_authority(branch, now_s)
    span = (
        evidence[-1].time_s - evidence[0].earliest_time_s
        if evidence
        else 0.0
    )
    return (
        authority == x27.RIGID_DYNAMIC,
        len(anchor_times_s),
        len(evidence),
        span,
        sum(value.overlap_cells for value in evidence),
        tuple(-value for value in branch.last_shift_cells),
    )


def _extended_branch(
    branch: TransportBranch,
    candidate: SurfaceTransportCandidate,
    *,
    now_s: float,
    delta_s: float,
    current_cells: frozenset[x27.Cell],
) -> TransportBranch:
    evidence = _trimmed_evidence(branch.evidence, now_s)
    anchor_times_s = _trimmed_anchor_times(branch.anchor_times_s, now_s)
    evidence.append(
        BranchEvidence(
            earliest_time_s=now_s,
            time_s=now_s,
            delta_s=delta_s,
            shift_cells=candidate.shift_cells,
            occupancy_anchored=candidate.occupancy_anchored,
            overlap_cells=candidate.overlap_cells,
        )
    )
    if candidate.occupancy_anchored:
        anchor_times_s.append(now_s)
    return TransportBranch(
        evidence=evidence,
        anchor_times_s=anchor_times_s,
        world_lineage=x29.TemporalLineageTracker.advance_lineage(
            branch.world_lineage, current_cells, candidate.shift_cells
        ),
        last_shift_cells=candidate.shift_cells,
        last_state=(
            BRANCH_ANCHOR
            if candidate.occupancy_anchored
            else BRANCH_CONTINUATION
        ),
    )


def _new_branch(
    candidate: SurfaceTransportCandidate,
    *,
    now_s: float,
    delta_s: float,
    previous_cells: frozenset[x27.Cell],
    current_cells: frozenset[x27.Cell],
) -> TransportBranch:
    return TransportBranch(
        evidence=[
            BranchEvidence(
                earliest_time_s=now_s,
                time_s=now_s,
                delta_s=delta_s,
                shift_cells=candidate.shift_cells,
                occupancy_anchored=candidate.occupancy_anchored,
                overlap_cells=candidate.overlap_cells,
            )
        ],
        anchor_times_s=[now_s] if candidate.occupancy_anchored else [],
        world_lineage=[
            x27.shifted(previous_cells, candidate.shift_cells),
            current_cells,
        ][-x29.LINEAGE_OBSERVATIONS :],
        last_shift_cells=candidate.shift_cells,
        last_state=(
            BRANCH_ANCHOR
            if candidate.occupancy_anchored
            else BRANCH_CONTINUATION
        ),
    )


def _evidence_state(branch: TransportBranch) -> tuple[Any, ...]:
    return (
        tuple(
            (
                value.time_s,
                value.earliest_time_s,
                value.delta_s,
                value.shift_cells,
                value.occupancy_anchored,
                value.overlap_cells,
            )
            for value in branch.evidence
        ),
        tuple(branch.anchor_times_s),
    )


def _motion_state(branch: TransportBranch) -> tuple[Any, ...]:
    return (
        _evidence_state(branch),
        branch.last_shift_cells,
        branch.last_state,
    )


def _branch_state_key(branch: TransportBranch) -> str:
    payload = {
        "evidence": _evidence_state(branch),
        "lineage": [sorted(value) for value in branch.world_lineage],
        "last_shift_cells": branch.last_shift_cells,
        "last_state": branch.last_state,
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest().upper()


def _prune_route_footprint_dominance(
    branches: Sequence[TransportBranch],
) -> list[TransportBranch]:
    """Prune only future-stable componentwise lineage subsets."""

    retained: list[TransportBranch] = []
    for branch in branches:
        dominated = False
        for other in branches:
            if branch is other or _motion_state(branch) != _motion_state(other):
                continue
            if len(branch.world_lineage) != len(other.world_lineage):
                continue
            componentwise_subset = all(
                left <= right
                for left, right in zip(
                    branch.world_lineage, other.world_lineage, strict=True
                )
            )
            componentwise_strict = any(
                left < right
                for left, right in zip(
                    branch.world_lineage, other.world_lineage, strict=True
                )
            )
            if componentwise_subset and componentwise_strict:
                dominated = True
                break
        if not dominated:
            retained.append(branch)
    return retained


def _motion_signature(branch: TransportBranch) -> tuple[Any, ...]:
    """State that controls future direction checks, excluding time/geometry."""

    return (
        tuple(
            (value.delta_s, value.shift_cells)
            for value in branch.evidence
        ),
        branch.last_shift_cells,
        branch.last_state,
        len(branch.world_lineage),
    )


def _coalesce_motion_signature_envelopes(
    branches: Sequence[TransportBranch],
) -> list[TransportBranch]:
    """Conservatively envelope timing and geometry for one motion signature.

    Timestamp intervals retain the earliest fit witness and latest live witness.
    Componentwise lineage unions retain every route-relevant cell and stay
    closed under the common future shift/append operation.
    """

    groups: dict[tuple[Any, ...], list[TransportBranch]] = {}
    for branch in branches:
        groups.setdefault(_motion_signature(branch), []).append(branch)

    output: list[TransportBranch] = []
    for group in groups.values():
        if len(group) == 1:
            output.append(group[0])
            continue
        evidence = []
        for ordinal in range(len(group[0].evidence)):
            values = [branch.evidence[ordinal] for branch in group]
            evidence.append(
                BranchEvidence(
                    earliest_time_s=min(
                        value.earliest_time_s for value in values
                    ),
                    time_s=max(value.time_s for value in values),
                    delta_s=values[0].delta_s,
                    shift_cells=values[0].shift_cells,
                    occupancy_anchored=any(
                        value.occupancy_anchored for value in values
                    ),
                    overlap_cells=max(value.overlap_cells for value in values),
                )
            )
        anchor_times_s = [
            value
            for branch in group
            for value in branch.anchor_times_s
        ]
        output.append(
            TransportBranch(
                evidence=evidence,
                anchor_times_s=(
                    [max(anchor_times_s)] if anchor_times_s else []
                ),
                world_lineage=[
                    frozenset().union(
                        *(branch.world_lineage[ordinal] for branch in group)
                    )
                    for ordinal in range(len(group[0].world_lineage))
                ],
                last_shift_cells=group[0].last_shift_cells,
                last_state=group[0].last_state,
            )
        )
    return output


def advance_transport_cone(
    branches: Mapping[str, TransportBranch],
    candidates: Sequence[SurfaceTransportCandidate],
    *,
    now_s: float,
    delta_s: float,
    previous_cells: frozenset[x27.Cell],
    current_cells: frozenset[x27.Cell],
    neutral_supported: bool,
) -> tuple[dict[str, TransportBranch], int, int]:
    """Advance all feasible histories and conservatively envelope equivalents."""

    proposals: list[TransportBranch] = []
    terminated = 0
    created = 0
    for branch in branches.values():
        branch.evidence = _trimmed_evidence(branch.evidence, now_s)
        branch.anchor_times_s = _trimmed_anchor_times(
            branch.anchor_times_s, now_s
        )
        extensions = [
            candidate
            for candidate in candidates
            if _candidate_consistent(branch.evidence, candidate, delta_s)
        ]
        if extensions:
            proposals.extend(
                _extended_branch(
                    branch,
                    candidate,
                    now_s=now_s,
                    delta_s=delta_s,
                    current_cells=current_cells,
                )
                for candidate in extensions
            )
        if neutral_supported and branch.evidence:
            proposals.append(
                TransportBranch(
                    evidence=list(branch.evidence),
                    anchor_times_s=list(branch.anchor_times_s),
                    world_lineage=x29.TemporalLineageTracker.advance_lineage(
                        branch.world_lineage, current_cells, (0, 0)
                    ),
                    last_shift_cells=(0, 0),
                    last_state=BRANCH_NEUTRAL,
                )
            )
        if not extensions and not neutral_supported:
            terminated += 1

    for candidate in candidates:
        if not (candidate.occupancy_anchored or candidate.is_center_hypothesis):
            continue
        proposals.append(
            _new_branch(
                candidate,
                now_s=now_s,
                delta_s=delta_s,
                previous_cells=previous_cells,
                current_cells=current_cells,
            )
        )
        created += 1

    exact = {
        _branch_state_key(proposal): proposal
        for proposal in proposals
        if proposal.evidence
    }
    retained = _prune_route_footprint_dominance(list(exact.values()))
    enveloped = _coalesce_motion_signature_envelopes(retained)
    merged = {_branch_state_key(value): value for value in enveloped}
    return dict(sorted(merged.items())), terminated, created


class AmbiguityPreservingSurfaceTracker(x29.TemporalLineageTracker):
    def __init__(self) -> None:
        self.tracks: dict[str, SurfaceTransportTrack] = {}
        self.next_id = 1

    def update(
        self,
        measurements: Sequence[x30.ComponentMeasurement],
        now_s: float,
        wearer_position_xy: Sequence[float],
    ) -> set[str]:
        wearer = np.asarray(wearer_position_xy, dtype=np.float64).reshape(2)
        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if now_s - value.last_seen_s <= x24.HOLD_WINDOW_S + EPSILON
        }
        prepared = self.prepare(measurements, wearer)
        costs: list[tuple[float, float, float, str, int]] = []
        for track_id, track in self.tracks.items():
            if track.last_center_xy is None or track.last_world_cells is None:
                continue
            for index, measurement in enumerate(prepared):
                if measurement.value.class_id != track.class_id:
                    continue
                delta = measurement.value.position_xy - track.last_center_xy
                distance = float(np.linalg.norm(delta))
                if distance > x24.ASSOCIATION_DISTANCE_M + EPSILON:
                    continue
                alignment = x27.lattice_alignment(
                    track.last_world_cells, measurement.world_cells, delta
                )
                bbox_overlap = (
                    0.0
                    if track.last_bbox is None
                    else x24.bbox_iou(track.last_bbox, measurement.value.bbox)
                )
                costs.append(
                    (
                        -alignment.overlap_fraction,
                        distance,
                        -bbox_overlap,
                        track_id,
                        index,
                    )
                )
        costs.sort()
        assigned_tracks: set[str] = set()
        assigned_measurements: dict[int, str] = {}
        for _overlap, _distance, _bbox, track_id, index in costs:
            if track_id not in assigned_tracks and index not in assigned_measurements:
                assigned_tracks.add(track_id)
                assigned_measurements[index] = track_id

        measured_ids: set[str] = set()
        for index, measurement in enumerate(prepared):
            track_id = assigned_measurements.get(index)
            if track_id is None:
                track_id = f"surface-cone-{self.next_id:06d}"
                self.next_id += 1
                self.tracks[track_id] = SurfaceTransportTrack(
                    track_id=track_id,
                    class_id=measurement.value.class_id,
                    class_name=measurement.value.class_name,
                    world_lineage=[measurement.world_cells],
                    wearer_lineage=[measurement.wearer_cells],
                    measurements=1,
                )
            track = self.tracks[track_id]
            if (
                track.last_center_xy is not None
                and track.last_world_cells is not None
                and track.last_wearer_cells is not None
                and math.isfinite(track.last_measurement_time_s)
            ):
                delta_s = now_s - track.last_measurement_time_s
                x24.require(delta_s > 0.0, f"x31_noncausal_track_time:{track_id}")
                center_delta = measurement.value.position_xy - track.last_center_xy
                world = x27.lattice_alignment(
                    track.last_world_cells, measurement.world_cells, center_delta
                )
                previous_wearer_center = np.mean(
                    np.asarray(tuple(track.last_wearer_cells), dtype=np.float64), axis=0
                )
                current_wearer_center = np.mean(
                    np.asarray(tuple(measurement.wearer_cells), dtype=np.float64), axis=0
                )
                relative = x27.lattice_alignment(
                    track.last_wearer_cells,
                    measurement.wearer_cells,
                    (current_wearer_center - previous_wearer_center)
                    * x27.LATTICE_CELL_SIZE_M,
                )
                wearer_distance = x25.point_polygon_distance(
                    wearer, measurement.value.footprint_xy
                )
                hint = self.motion_hint(world, relative, wearer_distance)
                occupancy_shift_xy = (
                    np.asarray(world.shift_cells, dtype=np.float64)
                    * x27.LATTICE_CELL_SIZE_M
                )
                track.evidence.append(
                    x27.AuthorityEvidence(
                        time_s=now_s,
                        delta_s=delta_s,
                        hint=hint,
                        shift_xy=occupancy_shift_xy,
                        world_alignment=world,
                        wearer_alignment=relative,
                    )
                )
                track.evidence = [
                    value
                    for value in track.evidence
                    if now_s - value.time_s <= x24.TRACK_HISTORY_S + EPSILON
                ]

                candidates = surface_transport_candidates(
                    track.last_world_cells,
                    measurement.world_cells,
                    center_delta,
                    [
                        branch.evidence[-1].shift_cells
                        for branch in track.transport_branches.values()
                        if branch.evidence
                    ],
                )
                zero_overlap = len(track.last_world_cells & measurement.world_cells)
                branches, terminated, created = advance_transport_cone(
                    track.transport_branches,
                    candidates,
                    now_s=now_s,
                    delta_s=delta_s,
                    previous_cells=track.last_world_cells,
                    current_cells=measurement.world_cells,
                    neutral_supported=(
                        zero_overlap >= x27.MINIMUM_ALIGNMENT_CELLS
                    ),
                )
                track.transport_branches = branches
                track.branches_terminated += terminated
                track.branches_created += created
                track.transport_contradictions += terminated
                track.transport_anchors += sum(
                    int(value.occupancy_anchored) for value in candidates
                )
                track.ambiguity_frames += int(len(branches) > 1)
                track.maximum_branch_count = max(
                    track.maximum_branch_count, len(branches)
                )

                authorized = [
                    (key, branch, resolve_branch_authority(branch, now_s)[1])
                    for key, branch in branches.items()
                    if resolve_branch_authority(branch, now_s)[0]
                    == x27.RIGID_DYNAMIC
                ]
                occupancy_authority, occupancy_velocity = x27.resolve_authority(
                    track.evidence, now_s
                )
                previous_authority = track.authority
                if authorized:
                    _best_key, best_branch, best_velocity = max(
                        authorized,
                        key=lambda value: _branch_priority(value[1], now_s),
                    )
                    track.authority = x27.RIGID_DYNAMIC
                    track.authorized_velocity_xy = best_velocity
                    lineage_shift = best_branch.last_shift_cells
                    track.last_transport_shift_cells = best_branch.last_shift_cells
                    track.last_transport_state = best_branch.last_state
                elif branches and occupancy_authority != x27.EGO_CARRIED:
                    track.authority = x27.UNAUTHORIZED_MOTION
                    track.authorized_velocity_xy = np.zeros(2, dtype=np.float64)
                    lineage_shift = candidates[0].shift_cells if candidates else (0, 0)
                    track.last_transport_shift_cells = lineage_shift
                    track.last_transport_state = (
                        candidates[0].is_center_hypothesis
                        and x29.TRANSPORT_CONTINUATION
                        or BRANCH_NEUTRAL
                    )
                else:
                    track.authority = occupancy_authority
                    track.authorized_velocity_xy = occupancy_velocity
                    lineage_shift = (
                        world.shift_cells
                        if world.has_authorized_shift
                        or hint in {x27.STATIC_SCENE, x27.EGO_CARRIED}
                        else (candidates[0].shift_cells if candidates else (0, 0))
                    )
                    track.last_transport_shift_cells = lineage_shift
                    track.last_transport_state = BRANCH_NEUTRAL
                if track.authority != previous_authority:
                    track.authority_transitions += 1

                track.world_lineage = self.advance_lineage(
                    track.world_lineage,
                    measurement.world_cells,
                    lineage_shift,
                )
                track.wearer_lineage = self.advance_lineage(
                    track.wearer_lineage,
                    measurement.wearer_cells,
                    relative.shift_cells,
                )
                track.last_world_alignment = world
                track.last_wearer_alignment = relative
                track.measurements += 1

            track.last_seen_s = now_s
            track.last_measurement_time_s = now_s
            track.last_center_xy = measurement.value.position_xy.copy()
            track.last_world_cells = measurement.world_cells
            track.last_wearer_cells = measurement.wearer_cells
            track.last_bbox = measurement.value.bbox
            track.depth_support = measurement.value.depth_support
            measured_ids.add(track_id)
        return measured_ids

    @staticmethod
    def _track_variant_id(track_id: str, state_key: str) -> str:
        return f"{track_id}::cone-{state_key[:16]}"

    def _row(
        self,
        *,
        track: SurfaceTransportTrack,
        track_id: str,
        parent_track_id: str,
        authority: str,
        velocity: np.ndarray,
        lineage: Sequence[frozenset[x27.Cell]],
        last_shift_cells: x27.Cell,
        branch: TransportBranch | None,
        now_s: float,
        measured_ids: set[str],
        authorized_branch_count: int,
    ) -> dict[str, Any]:
        age_s = now_s - track.last_seen_s
        support_cells, maximum_support = x29.repeated_support(lineage)
        support_valid = len(support_cells) >= x27.MINIMUM_ALIGNMENT_CELLS
        footprint = (
            x29.lineage_footprint(support_cells)
            if support_valid
            else np.zeros((4, 2), dtype=np.float64)
        )
        if authority == x27.RIGID_DYNAMIC and support_valid:
            footprint = footprint + velocity[None, :] * max(0.0, age_s)
        center = (
            np.mean(footprint, axis=0)
            if support_valid
            else np.zeros(2, dtype=np.float64)
        )
        measured = parent_track_id in measured_ids
        evidence = [] if branch is None else _trimmed_evidence(branch.evidence, now_s)
        return {
            "track_id": track_id,
            "parent_track_id": parent_track_id,
            "class_id": track.class_id,
            "class_name": track.class_name,
            "disposition": "MEASURED" if measured else "HOLD",
            "evidence_age_s": max(0.0, age_s),
            "motion_authority": authority,
            "risk_eligible": (
                support_valid
                and authority in {x27.STATIC_SCENE, x27.RIGID_DYNAMIC}
            ),
            "position_forward_m": float(center[0]),
            "position_right_m": float(center[1]),
            "velocity_forward_mps": float(velocity[0]),
            "velocity_right_mps": float(velocity[1]),
            "footprint_xy": [[float(value) for value in row] for row in footprint],
            "footprint_area_m2": (
                x25.polygon_area(footprint) if support_valid else 0.0
            ),
            "lineage_cells": len(support_cells),
            "lineage_observations": len(lineage),
            "maximum_cell_observations": maximum_support,
            "measurements": track.measurements,
            "authority_transitions": track.authority_transitions,
            "transport_lineage_pairs": len(evidence),
            "transport_anchor_pairs": (
                len(_trimmed_anchor_times(branch.anchor_times_s, now_s))
                if branch is not None
                else 0
            ),
            "transport_anchors_total": track.transport_anchors,
            "transport_contradictions": track.transport_contradictions,
            "transport_conflicted": False,
            "transport_state": branch.last_state if branch else track.last_transport_state,
            "transport_shift_cells": list(last_shift_cells) if measured else None,
            "surface_transport_branch_count": len(track.transport_branches),
            "authorized_surface_transport_branches": authorized_branch_count,
            "surface_transport_branches_created": track.branches_created,
            "surface_transport_branches_terminated": track.branches_terminated,
            "surface_transport_ambiguity_frames": track.ambiguity_frames,
            "surface_transport_maximum_branches": track.maximum_branch_count,
            "surface_transport_velocity_key_cells_per_s": (
                list(branch_velocity_key(evidence)) if evidence else None
            ),
            "depth_grid_support": track.depth_support if measured else None,
            "world_lattice_shift_cells": list(last_shift_cells) if measured else None,
        }

    def emitted(self, now_s: float, measured_ids: set[str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for parent_track_id, track in sorted(self.tracks.items()):
            age_s = now_s - track.last_seen_s
            if age_s > x24.HOLD_WINDOW_S + EPSILON:
                continue
            authorized = [
                (key, branch, resolve_branch_authority(branch, now_s)[1])
                for key, branch in track.transport_branches.items()
                if resolve_branch_authority(branch, now_s)[0] == x27.RIGID_DYNAMIC
            ]
            if authorized:
                for key, branch, velocity in authorized:
                    output.append(
                        self._row(
                            track=track,
                            track_id=self._track_variant_id(parent_track_id, key),
                            parent_track_id=parent_track_id,
                            authority=x27.RIGID_DYNAMIC,
                            velocity=velocity,
                            lineage=branch.world_lineage,
                            last_shift_cells=branch.last_shift_cells,
                            branch=branch,
                            now_s=now_s,
                            measured_ids=measured_ids,
                            authorized_branch_count=len(authorized),
                        )
                    )
                continue

            occupancy_authority, occupancy_velocity = x27.resolve_authority(
                track.evidence, now_s
            )
            if track.transport_branches and occupancy_authority != x27.EGO_CARRIED:
                authority = x27.UNAUTHORIZED_MOTION
                velocity = np.zeros(2, dtype=np.float64)
            else:
                authority = occupancy_authority
                velocity = occupancy_velocity
            output.append(
                self._row(
                    track=track,
                    track_id=parent_track_id,
                    parent_track_id=parent_track_id,
                    authority=authority,
                    velocity=velocity,
                    lineage=track.world_lineage,
                    last_shift_cells=track.last_transport_shift_cells,
                    branch=None,
                    now_s=now_s,
                    measured_ids=measured_ids,
                    authorized_branch_count=0,
                )
            )
        return output


def ambiguity_preserving_arm_frame(
    selection: route.RouteSelection,
    *,
    receipt: Mapping[str, Any] | None,
    observation: adapter.FrameObservation,
    wearer_position: tuple[float, float],
    wearer_velocity: tuple[float, float],
    tracks: Sequence[Mapping[str, Any]],
    confirmation: x24.RiskConfirmation,
    sample_period_s: float,
) -> dict[str, Any]:
    """Confirm by stable parent ancestry while routing every branch geometry."""

    if selection.mode_changed:
        confirmation.reset()
    segments = route.build_route_segments(
        selection,
        receipt=receipt,
        now_s=observation.time_s,
        wearer_position_xy=wearer_position,
        wearer_velocity_xy=wearer_velocity,
    )
    entries: dict[str, float | None] = {}
    selected_track_ids: dict[str, str] = {}
    for track in tracks:
        parent_id = str(track.get("parent_track_id", track["track_id"]))
        track_id = str(track["track_id"])
        entry = x30.first_contact_interval_entry_s(
            track["footprint_xy"],
            (track["velocity_forward_mps"], track["velocity_right_mps"]),
            segments,
        )
        previous = entries.get(parent_id)
        if entry is not None and (previous is None or entry < previous):
            entries[parent_id] = float(entry)
            selected_track_ids[parent_id] = track_id
        elif parent_id not in entries:
            entries[parent_id] = None
            selected_track_ids[parent_id] = track_id
    confirmed_parents = confirmation.update(
        entries,
        now_s=observation.time_s,
        sample_period_s=sample_period_s,
    )
    confirmed_entries = [
        float(entries[parent_id])
        for parent_id in confirmed_parents
        if entries.get(parent_id) is not None
    ]
    return {
        "route_mode": selection.mode,
        "authority": selection.authority,
        "plan_receipt_sha256": selection.receipt_sha256,
        "plan_position_residual_m": selection.plan_position_residual_m,
        "plan_velocity_direction_error_degrees": (
            selection.plan_velocity_direction_error_deg
        ),
        "fallback_reason": selection.fallback_reason,
        "route_mode_changed": selection.mode_changed,
        "route_risk": bool(confirmed_entries),
        "minimum_entry_s": min(confirmed_entries) if confirmed_entries else None,
        "candidate_risk_track_ids": sorted(
            selected_track_ids[parent_id]
            for parent_id, entry in entries.items()
            if entry is not None
        ),
        "confirmed_risk_track_ids": sorted(
            selected_track_ids[parent_id]
            for parent_id in confirmed_parents
            if entries.get(parent_id) is not None
        ),
        "candidate_risk_parent_track_ids": sorted(
            parent_id for parent_id, entry in entries.items() if entry is not None
        ),
        "confirmed_risk_parent_track_ids": sorted(
            parent_id
            for parent_id in confirmed_parents
            if entries.get(parent_id) is not None
        ),
    }


def predict_episode(
    episode: adapter.Episode,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: adapter.CameraCalibration,
) -> dict[str, Any]:
    original_tracker = x30.AdaptiveSurfaceLineageTracker
    original_arm_frame = x30.arm_frame
    x30.AdaptiveSurfaceLineageTracker = AmbiguityPreservingSurfaceTracker
    x30.arm_frame = ambiguity_preserving_arm_frame
    try:
        value = x30.predict_episode(episode, candidate_values, calibration)
    finally:
        x30.AdaptiveSurfaceLineageTracker = original_tracker
        x30.arm_frame = original_arm_frame

    for frame in value["frames"]:
        frame["arms"][ARM_X31] = frame["arms"].pop(x30.ARM_X30)
    diagnostics = value["diagnostics"]
    diagnostics["x31_route_mode_counts"] = diagnostics.pop("x30_route_mode_counts")
    diagnostics["surface_transport_ambiguity_frames"] = sum(
        any(
            int(track.get("surface_transport_branch_count", 0)) > 1
            for track in frame["tracks"]
        )
        for frame in value["frames"]
    )
    diagnostics["maximum_surface_transport_branches"] = max(
        (
            int(track.get("surface_transport_branch_count", 0))
            for frame in value["frames"]
            for track in frame["tracks"]
        ),
        default=0,
    )
    old_arm = value["arms"].pop(x30.ARM_X30)
    value["arms"][ARM_X31] = old_arm
    return value


def _algorithm_files() -> dict[str, Path]:
    return {
        "x25_geometry": Path(x25.__file__).resolve(),
        "x27_authority": Path(x27.__file__).resolve(),
        "x29_lineage": Path(x29.__file__).resolve(),
        "x30_surface_interval": Path(x30.__file__).resolve(),
        "x31_predictor": Path(__file__).resolve(),
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(
        not output["freeze"].exists(), f"x31_freeze_exists:{output['freeze']}"
    )
    frozen_x24, contract, _candidates = x24.require_freeze(run_root)
    x24.require(output["x24_predictions"].is_file(), "x31_x24_baseline_missing")
    value = {
        "schema": FREEZE_SCHEMA,
        "status": "FROZEN_TRUTH_BLIND_PENDING_PREDICTION",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "source": {
            "x24_freeze_sha256": x24.sha256_file(output["x24_freeze"]),
            "x24_predictions_sha256": x24.sha256_file(
                output["x24_predictions"]
            ),
            "model_manifest_sha256": contract.manifest_sha256,
            "candidate_aggregate_sha256": frozen_x24["candidates"][
                "aggregate_sha256"
            ],
        },
        "algorithm_files": {
            name: {"path": str(path), "sha256": x24.sha256_file(path)}
            for name, path in _algorithm_files().items()
        },
        "episodes": len(contract.episodes),
        "frames": len(x24.flatten_observations(contract)),
        "fixed_constants": fixed_constants(),
        "arm": ARM_X31,
    }
    x24.write_json_exclusive(output["freeze"], value)
    return {**value, "freeze_sha256": x24.sha256_file(output["freeze"])}


def require_freeze(
    run_root: Path,
) -> tuple[dict[str, Any], adapter.SanitizedModelContract, list[dict[str, Any]]]:
    output = paths(run_root)
    frozen = x24.read_json(output["freeze"].resolve(strict=True), "x31_freeze")
    x24.require(frozen.get("schema") == FREEZE_SCHEMA, "x31_freeze_schema")
    x24.require(
        frozen.get("fixed_constants") == fixed_constants(),
        "x31_constants_drift",
    )
    for name, path in _algorithm_files().items():
        x24.require(
            frozen["algorithm_files"][name]["sha256"] == x24.sha256_file(path),
            f"x31_algorithm_drift:{name}",
        )
    frozen_x24, contract, candidates = x24.require_freeze(run_root)
    x24.require(
        frozen["source"]["x24_freeze_sha256"]
        == x24.sha256_file(output["x24_freeze"]),
        "x31_x24_freeze_drift",
    )
    x24.require(
        frozen["source"]["x24_predictions_sha256"]
        == x24.sha256_file(output["x24_predictions"]),
        "x31_x24_prediction_drift",
    )
    x24.require(
        frozen["source"]["candidate_aggregate_sha256"]
        == frozen_x24["candidates"]["aggregate_sha256"],
        "x31_candidate_drift",
    )
    return frozen, contract, candidates


def predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(
        not output["predictions"].exists(),
        f"x31_predictions_exist:{output['predictions']}",
    )
    frozen, contract, candidate_values = require_freeze(run_root)
    cursor = 0
    episodes: dict[str, Any] = {}
    for episode in contract.episodes:
        count = len(episode.observations)
        episodes[episode.episode_id] = predict_episode(
            episode,
            candidate_values[cursor : cursor + count],
            contract.calibration,
        )
        cursor += count
    value = {
        "schema": PREDICTION_SCHEMA,
        "status": "SEALED_TRUTH_BLIND_PENDING_SCORE",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "arms": [ARM_X31],
        "episodes": episodes,
        "fixed_constants": fixed_constants(),
        "source": {
            "freeze_sha256": x24.sha256_file(output["freeze"]),
            **frozen["source"],
        },
        "claim_boundary": {
            "synthetic_development": True,
            "evaluator_opened": False,
            "current_actor_oracle_used": False,
        },
    }
    x24.write_json_exclusive(output["predictions"], value)
    return {
        **value,
        "predictions_sha256": x24.sha256_file(output["predictions"]),
    }


def _synthetic_candidate(
    shift: x27.Cell,
    *,
    anchored: bool,
    center: bool = False,
    overlap: int = 12,
) -> SurfaceTransportCandidate:
    return SurfaceTransportCandidate(
        shift_cells=shift,
        overlap_cells=overlap,
        zero_overlap_cells=2,
        support_cells=16,
        center_residual_cells=0.0 if center else 1.0,
        occupancy_anchored=anchored,
        is_center_hypothesis=center,
    )


def _exercise_sequence(
    steady_shift: x27.Cell,
    ambiguous: Sequence[SurfaceTransportCandidate],
) -> tuple[dict[str, TransportBranch], int]:
    cells = frozenset((x, y) for x in range(4) for y in range(4))
    branches: dict[str, TransportBranch] = {}
    terminated_total = 0
    for ordinal in range(1, 4):
        branches, terminated, _created = advance_transport_cone(
            branches,
            [_synthetic_candidate(steady_shift, anchored=ordinal == 1, center=True)],
            now_s=ordinal * 0.1,
            delta_s=0.1,
            previous_cells=cells,
            current_cells=x27.shifted(cells, steady_shift),
            neutral_supported=False,
        )
        terminated_total += terminated
    branches, terminated, _created = advance_transport_cone(
        branches,
        ambiguous,
        now_s=0.4,
        delta_s=0.1,
        previous_cells=cells,
        current_cells=x27.shifted(cells, steady_shift),
        neutral_supported=False,
    )
    terminated_total += terminated
    branches, terminated, _created = advance_transport_cone(
        branches,
        [_synthetic_candidate(steady_shift, anchored=False, center=False)],
        now_s=0.5,
        delta_s=0.1,
        previous_cells=cells,
        current_cells=x27.shifted(cells, steady_shift),
        neutral_supported=False,
    )
    return branches, terminated_total + terminated


def _generated_alias_candidates(
    steady_shift: x27.Cell, misleading_center_shift: x27.Cell
) -> list[SurfaceTransportCandidate]:
    previous = frozenset((x, y) for x in range(10) for y in range(6))
    current = x27.shifted(previous, steady_shift)
    candidates = surface_transport_candidates(
        previous,
        current,
        np.asarray(misleading_center_shift, dtype=np.float64)
        * x27.LATTICE_CELL_SIZE_M,
        [steady_shift],
    )
    shifts = {value.shift_cells for value in candidates}
    x24.require(
        steady_shift in shifts and misleading_center_shift in shifts,
        f"x31_live_branch_prediction_unreachable:{steady_shift}:{misleading_center_shift}",
    )
    return candidates


def self_check() -> dict[str, Any]:
    previous = frozenset((x, y) for x in range(6) for y in range(3))
    current = x27.shifted(previous, (-1, 0))
    enumerated = surface_transport_candidates(
        previous,
        current,
        np.asarray((-1.0, -1.0), dtype=np.float64)
        * x27.LATTICE_CELL_SIZE_M,
    )
    x24.require(
        len(enumerated) <= (
            (2 * x27.ALIGNMENT_NEIGHBOUR_CELLS + 1) ** 2
        )
        and any(value.shift_cells == (-1, 0) for value in enumerated),
        "x31_inherited_surface_hypothesis_missing",
    )

    ep03, ep03_terminated = _exercise_sequence(
        (-1, 2),
        _generated_alias_candidates((-1, 2), (-1, -1)),
    )
    x24.require(
        any(
            resolve_branch_authority(branch, 0.5)[0] == x27.RIGID_DYNAMIC
            and float(
                np.dot(
                    resolve_branch_authority(branch, 0.5)[1],
                    np.asarray((-1.0, 2.0)),
                )
            )
            > EPSILON
            for branch in ep03.values()
        ),
        "x31_ep03_compatible_surface_branch_lost",
    )

    ep07, ep07_terminated = _exercise_sequence(
        (-4, 0),
        _generated_alias_candidates((-4, 0), (1, 0)),
    )
    x24.require(
        any(
            resolve_branch_authority(branch, 0.5)[0] == x27.RIGID_DYNAMIC
            and resolve_branch_authority(branch, 0.5)[1][0] < 0.0
            for branch in ep07.values()
        ),
        "x31_ep07_forward_branch_lost_after_alias_rebound",
    )

    abstention_cells = frozenset((x, y) for x in range(4) for y in range(4))
    abstention_branches, _terminated = _exercise_sequence(
        (-4, 0),
        [_synthetic_candidate((-4, 0), anchored=False, center=False)],
    )
    carried, terminated, _created = advance_transport_cone(
        abstention_branches,
        [_synthetic_candidate((1, 0), anchored=False, center=True)],
        now_s=0.6,
        delta_s=0.1,
        previous_cells=abstention_cells,
        current_cells=abstention_cells,
        neutral_supported=True,
    )
    x24.require(
        terminated == 0
        and any(
            resolve_branch_authority(branch, 0.6)[0] == x27.RIGID_DYNAMIC
            and branch.last_state == BRANCH_NEUTRAL
            for branch in carried.values()
        ),
        "x31_zero_surface_abstention_did_not_preserve_authorized_branch",
    )

    parallel, parallel_terminated, _created = advance_transport_cone(
        abstention_branches,
        [_synthetic_candidate((-4, 0), anchored=False, center=True)],
        now_s=0.6,
        delta_s=0.1,
        previous_cells=abstention_cells,
        current_cells=abstention_cells,
        neutral_supported=True,
    )
    x24.require(
        parallel_terminated == 0
        and any(branch.last_state == BRANCH_NEUTRAL for branch in parallel.values())
        and any(
            branch.last_state != BRANCH_NEUTRAL for branch in parallel.values()
        ),
        "x31_zero_abstention_not_parallel_to_compatible_extension",
    )

    reversed_branches, reversed_terminated, _created = advance_transport_cone(
        abstention_branches,
        [_synthetic_candidate((4, 0), anchored=True, center=True)],
        now_s=0.6,
        delta_s=0.1,
        previous_cells=abstention_cells,
        current_cells=x27.shifted(abstention_cells, (4, 0)),
        neutral_supported=False,
    )
    x24.require(
        reversed_terminated >= 1
        and not any(
            resolve_branch_authority(branch, 0.6)[0] == x27.RIGID_DYNAMIC
            for branch in reversed_branches.values()
        ),
        "x31_true_reverse_did_not_end_old_epoch",
    )

    envelope_a = _new_branch(
        _synthetic_candidate((-1, 0), anchored=True, center=True),
        now_s=0.1,
        delta_s=0.1,
        previous_cells=abstention_cells,
        current_cells=x27.shifted(abstention_cells, (-1, 0)),
    )
    shifted_abstention_cells = x27.shifted(abstention_cells, (0, 2))
    envelope_b = _new_branch(
        _synthetic_candidate((-1, 0), anchored=True, center=True),
        now_s=0.2,
        delta_s=0.1,
        previous_cells=shifted_abstention_cells,
        current_cells=x27.shifted(shifted_abstention_cells, (-1, 0)),
    )
    enveloped = _coalesce_motion_signature_envelopes(
        [envelope_a, envelope_b]
    )
    x24.require(
        len(enveloped) == 1
        and enveloped[0].evidence[0].earliest_time_s == 0.1
        and enveloped[0].evidence[0].time_s == 0.2
        and all(
            left <= merged
            for branch in (envelope_a, envelope_b)
            for left, merged in zip(
                branch.world_lineage,
                enveloped[0].world_lineage,
                strict=True,
            )
        ),
        "x31_conservative_ambiguity_envelope_lost_feasible_state",
    )
    return {
        "status": "X31_AMBIGUITY_PRESERVING_TRANSPORT_STRUCTURAL_FALSIFIER_MET",
        "inherited_surface_hypothesis_retained": True,
        "live_branch_prediction_reached_real_generator": True,
        "conservative_ambiguity_envelope_preserved": True,
        "ep03_compatible_branch_retained": True,
        "ep07_alias_rebound_branch_retained": True,
        "zero_surface_abstention_preserved_authority": True,
        "zero_surface_abstention_parallel_to_extension": True,
        "true_reverse_ended_old_epoch": True,
        "ep03_terminated_incompatible_branches": ep03_terminated,
        "ep07_terminated_incompatible_branches": ep07_terminated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "predict"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-root", type=Path, required=True)
    subparsers.add_parser("self-check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        value = freeze(args)
    elif args.command == "predict":
        value = predict(args)
    else:
        value = self_check()
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
