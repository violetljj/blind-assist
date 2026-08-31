"""X33 observation-centric dormant transport reactivation.

X33 separates risk emission lifetime from association memory.  A track that
was already authorized as rigid-dynamic may remain dormant for at most two
observed sample periods beyond X24's unchanged HOLD window (and never beyond
the inherited one-second history).  Dormant tracks emit no rows.  They may be
reactivated only by an observation compatible with a live authorized transport
velocity, using the unchanged association radius or motion-warped occupancy
overlap.  The real pre-loss and reappearance observations then advance X31's
transport evidence across the complete gap.

The representation is inspired by observation-centric reactivation, but all
results on the already consumed C16 source remain same-source Development.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x29_temporal_occupancy_lineage_predictor as x29  # noqa: E402
import dtr_carla_x30_adaptive_surface_interval_predictor as x30  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x32_observation_conditioned_core_predictor as x32  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X33_DORMANT_TRANSPORT_REACTIVATION"
ARM_X33 = "X33_ISSUED_PLAN_DORMANT_TRANSPORT_REACTIVATION"
ASSOCIATION_MEMORY_SAMPLE_PERIODS = 2
EPSILON = x31.EPSILON


def fixed_constants() -> dict[str, Any]:
    return {
        **x32.fixed_constants(),
        "representation": "OBSERVATION_CENTRIC_DORMANT_TRANSPORT_REACTIVATION",
        "risk_hold_seconds": x24.HOLD_WINDOW_S,
        "association_memory_extra_sample_periods": (
            ASSOCIATION_MEMORY_SAMPLE_PERIODS
        ),
        "association_memory_upper_bound_seconds": x24.TRACK_HISTORY_S,
        "dormant_eligibility": "PREVIOUSLY_AUTHORIZED_RIGID_DYNAMIC_ONLY",
        "dormant_emission": False,
        "dormant_motion_hypotheses": "ALL_LIVE_AUTHORIZED_TRANSPORT_BRANCHES",
        "dormant_reactivation_gate": (
            "UNCHANGED_CENTER_RESIDUAL_RADIUS_OR_MOTION_WARPED_OCCUPANCY_OVERLAP"
        ),
        "active_association_precedence": True,
        "detector_threshold_change": False,
        "association_radius_change": False,
        "risk_hold_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


class DormantTransportReactivationTracker(x32.ObservationConditionedCoreTracker):
    def __init__(self) -> None:
        super().__init__()
        self.previous_update_s = float("nan")
        self.sample_period_s = 0.0

    def _association_memory_s(self, now_s: float) -> float:
        if math.isfinite(self.previous_update_s):
            sample_period_s = round(now_s - self.previous_update_s, 6)
            x24.require(sample_period_s > 0.0, "x33_noncausal_update_time")
            self.sample_period_s = sample_period_s
        self.previous_update_s = now_s
        return min(
            x24.TRACK_HISTORY_S,
            x24.HOLD_WINDOW_S
            + ASSOCIATION_MEMORY_SAMPLE_PERIODS * self.sample_period_s,
        )

    @staticmethod
    def _authorized_velocities(
        track: x31.SurfaceTransportTrack, now_s: float
    ) -> list[np.ndarray]:
        values = [
            velocity
            for branch in track.transport_branches.values()
            for authority, velocity in [x31.resolve_branch_authority(branch, now_s)]
            if authority == x27.RIGID_DYNAMIC
        ]
        if not values and track.authority == x27.RIGID_DYNAMIC:
            values = [track.authorized_velocity_xy]
        unique: dict[tuple[float, float], np.ndarray] = {}
        for velocity in values:
            key = tuple(float(round(value, 9)) for value in velocity)
            unique[key] = np.asarray(velocity, dtype=np.float64)
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _dormant_match(
        track: x31.SurfaceTransportTrack,
        measurement: x27.PreparedMeasurement,
        now_s: float,
    ) -> tuple[float, float] | None:
        x24.require(track.last_center_xy is not None, "x33_dormant_center")
        x24.require(track.last_world_cells is not None, "x33_dormant_cells")
        age_s = round(now_s - track.last_measurement_time_s, 6)
        support = min(len(track.last_world_cells), len(measurement.world_cells))
        hypotheses: list[tuple[float, float]] = []
        for velocity in DormantTransportReactivationTracker._authorized_velocities(
            track, now_s
        ):
            predicted_center = track.last_center_xy + velocity * age_s
            residual_m = float(
                np.linalg.norm(measurement.value.position_xy - predicted_center)
            )
            shift = np.rint(
                velocity * age_s / x27.LATTICE_CELL_SIZE_M
            ).astype(np.int64)
            shift_cells = (int(shift[0]), int(shift[1]))
            overlap = len(
                x27.shifted(track.last_world_cells, shift_cells)
                & measurement.world_cells
            )
            overlap_fraction = overlap / support if support else 0.0
            if (
                residual_m <= x24.ASSOCIATION_DISTANCE_M + EPSILON
                or overlap >= x27.MINIMUM_ALIGNMENT_CELLS
            ):
                hypotheses.append((-overlap_fraction, residual_m))
        return min(hypotheses) if hypotheses else None

    def update(
        self,
        measurements: Sequence[x30.ComponentMeasurement],
        now_s: float,
        wearer_position_xy: Sequence[float],
    ) -> set[str]:
        wearer = np.asarray(wearer_position_xy, dtype=np.float64).reshape(2)
        association_memory_s = self._association_memory_s(now_s)
        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if (
                now_s - value.last_seen_s <= x24.HOLD_WINDOW_S + EPSILON
                or (
                    value.authority == x27.RIGID_DYNAMIC
                    and now_s - value.last_seen_s
                    <= association_memory_s + EPSILON
                )
            )
        }
        prepared = self.prepare(measurements, wearer)
        costs: list[tuple[int, float, float, float, str, int]] = []
        dormant_ids: set[str] = set()
        for track_id, track in self.tracks.items():
            if track.last_center_xy is None or track.last_world_cells is None:
                continue
            dormant = now_s - track.last_seen_s > x24.HOLD_WINDOW_S + EPSILON
            if dormant:
                dormant_ids.add(track_id)
            for index, measurement in enumerate(prepared):
                if measurement.value.class_id != track.class_id:
                    continue
                delta = measurement.value.position_xy - track.last_center_xy
                if dormant:
                    match = self._dormant_match(track, measurement, now_s)
                    if match is None:
                        continue
                    overlap_cost, distance = match
                else:
                    distance = float(np.linalg.norm(delta))
                    if distance > x24.ASSOCIATION_DISTANCE_M + EPSILON:
                        continue
                    alignment = x27.lattice_alignment(
                        track.last_world_cells, measurement.world_cells, delta
                    )
                    overlap_cost = -alignment.overlap_fraction
                bbox_overlap = (
                    0.0
                    if track.last_bbox is None
                    else x24.bbox_iou(track.last_bbox, measurement.value.bbox)
                )
                costs.append(
                    (
                        int(dormant),
                        overlap_cost,
                        distance,
                        -bbox_overlap,
                        track_id,
                        index,
                    )
                )
        costs.sort()
        assigned_tracks: set[str] = set()
        assigned_measurements: dict[int, str] = {}
        for _dormant, _overlap, _distance, _bbox, track_id, index in costs:
            if track_id not in assigned_tracks and index not in assigned_measurements:
                assigned_tracks.add(track_id)
                assigned_measurements[index] = track_id

        measured_ids: set[str] = set()
        for index, measurement in enumerate(prepared):
            track_id = assigned_measurements.get(index)
            if track_id is None:
                track_id = f"surface-cone-{self.next_id:06d}"
                self.next_id += 1
                self.tracks[track_id] = x31.SurfaceTransportTrack(
                    track_id=track_id,
                    class_id=measurement.value.class_id,
                    class_name=measurement.value.class_name,
                    world_lineage=[measurement.world_cells],
                    wearer_lineage=[measurement.wearer_cells],
                    measurements=1,
                )
            track = self.tracks[track_id]
            if track_id in dormant_ids:
                track.dormant_reactivations = (
                    int(getattr(track, "dormant_reactivations", 0)) + 1
                )
            if (
                track.last_center_xy is not None
                and track.last_world_cells is not None
                and track.last_wearer_cells is not None
                and math.isfinite(track.last_measurement_time_s)
            ):
                delta_s = round(now_s - track.last_measurement_time_s, 6)
                x24.require(delta_s > 0.0, f"x33_noncausal_track_time:{track_id}")
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

                candidates = x31.surface_transport_candidates(
                    track.last_world_cells,
                    measurement.world_cells,
                    center_delta,
                    [
                        branch.evidence[-1].shift_cells
                        for branch in track.transport_branches.values()
                        if branch.evidence
                        and x31.resolve_branch_authority(branch, now_s)[0]
                        == x27.RIGID_DYNAMIC
                    ],
                )
                zero_overlap = len(track.last_world_cells & measurement.world_cells)
                branches, terminated, created = x31.advance_transport_cone(
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
                    (key, branch, x31.resolve_branch_authority(branch, now_s)[1])
                    for key, branch in branches.items()
                    if x31.resolve_branch_authority(branch, now_s)[0]
                    == x27.RIGID_DYNAMIC
                ]
                occupancy_authority, occupancy_velocity = x27.resolve_authority(
                    track.evidence, now_s
                )
                previous_authority = track.authority
                if authorized:
                    _best_key, best_branch, best_velocity = max(
                        authorized,
                        key=lambda value: x31._branch_priority(value[1], now_s),
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
                        x31.x29.TRANSPORT_CONTINUATION
                        if candidates and candidates[0].is_center_hypothesis
                        else x31.BRANCH_NEUTRAL
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
                    track.last_transport_state = x31.BRANCH_NEUTRAL
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

    def _row(self, **kwargs: Any) -> dict[str, Any]:
        row = super()._row(**kwargs)
        track = kwargs["track"]
        row["dormant_transport_reactivations"] = int(
            getattr(track, "dormant_reactivations", 0)
        )
        row["association_memory_sample_periods"] = (
            ASSOCIATION_MEMORY_SAMPLE_PERIODS
        )
        return row


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    original_tracker = x32.ObservationConditionedCoreTracker
    x32.ObservationConditionedCoreTracker = DormantTransportReactivationTracker
    try:
        value = x32.predict_episode(episode, candidate_values, calibration)
    finally:
        x32.ObservationConditionedCoreTracker = original_tracker
    value["arms"][ARM_X33] = value["arms"].pop(x32.ARM_X32)
    value["diagnostics"]["x33_route_mode_counts"] = value["diagnostics"].pop(
        "x32_route_mode_counts"
    )
    value["diagnostics"]["dormant_transport_reactivations"] = max(
        (
            int(track.get("dormant_transport_reactivations", 0))
            for frame in value["frames"]
            for track in frame["tracks"]
        ),
        default=0,
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X33] = frame["arms"].pop(x32.ARM_X32)
    return value


def self_check() -> dict[str, Any]:
    inherited = x32.self_check()
    tracker = DormantTransportReactivationTracker()
    tracker.previous_update_s = 0.0
    memory_s = tracker._association_memory_s(0.1)
    x24.require(
        abs(memory_s - 0.8) <= EPSILON,
        "x33_two_sample_memory",
    )
    x24.require(
        x24.HOLD_WINDOW_S < memory_s <= x24.TRACK_HISTORY_S,
        "x33_bounded_memory",
    )
    return {
        "status": "X33_DORMANT_TRANSPORT_REACTIVATION_STRUCTURAL_FALSIFIER_MET",
        "x32_structural_status": inherited["status"],
        "risk_hold_unchanged": True,
        "association_radius_unchanged": True,
        "association_memory_bounded": True,
        "active_tracks_precede_dormant_tracks": True,
    }
