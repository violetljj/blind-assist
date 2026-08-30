"""Freeze and predict X29 temporal occupancy-lineage route risk.

X29 separates two causal questions that X28 coupled too tightly:

* motion authority is accumulated from a direction-consistent transport
  lineage, anchored by at least one authorized occupancy translation;
* risk geometry is a sliding lineage of aligned cells that recur in more than
  one observation, rather than a monotonic all-history intersection.

The lineage keeps the most recent four measured supports, aligns every older
support into the current frame, and admits cells observed at least twice.  A
changing silhouette can therefore retain causal support without allowing a
single-frame mask or an unauthorized motion hypothesis into route risk.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
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


EXPERIMENT_ID = "DTR_CARLA_X29_TEMPORAL_OCCUPANCY_LINEAGE"
FREEZE_SCHEMA = "blindassist-dtr-carla-x29-temporal-lineage-freeze-v1"
PREDICTION_SCHEMA = "blindassist-dtr-carla-x29-temporal-lineage-predictions-v1"
ARM_X29 = "X29_ISSUED_PLAN_TEMPORAL_OCCUPANCY_LINEAGE"
EPSILON = 1e-9

TRANSPORT_ANCHOR = "OCCUPANCY_ANCHORED_TRANSPORT"
TRANSPORT_CONTINUATION = "DIRECTIONAL_TRANSPORT_CONTINUATION"
TRANSPORT_NEUTRAL = "NO_QUANTIZED_TRANSPORT"
TRANSPORT_CONFLICTED = "DIRECTION_CONTRADICTION_TERMINATED_EPOCH"

# The authority fit already requires four observations (three pairwise
# translations).  Repeated support means a cell must occur in at least two of
# those aligned observations; a one-frame detector artifact cannot qualify.
LINEAGE_OBSERVATIONS = x24.MINIMUM_FIT_SAMPLES
REPEATED_SUPPORT_OBSERVATIONS = 2


@dataclass(frozen=True)
class TransportEvidence:
    time_s: float
    delta_s: float
    shift_xy: np.ndarray
    occupancy_anchored: bool


def transport_velocity(value: TransportEvidence) -> np.ndarray:
    return value.shift_xy / value.delta_s


def quantized_transport(delta_xy: np.ndarray) -> tuple[x27.Cell, np.ndarray]:
    cells = np.rint(
        np.asarray(delta_xy, dtype=np.float64).reshape(2)
        / x27.LATTICE_CELL_SIZE_M
    ).astype(np.int64)
    shift_cells = (int(cells[0]), int(cells[1]))
    return (
        shift_cells,
        np.asarray(shift_cells, dtype=np.float64) * x27.LATTICE_CELL_SIZE_M,
    )


def direction_consistent_transport(rows: Sequence[TransportEvidence]) -> bool:
    velocities = np.stack([transport_velocity(value) for value in rows])
    median = np.median(velocities, axis=0)
    if float(np.linalg.norm(median)) <= EPSILON:
        return False
    return all(float(np.dot(value, median)) > EPSILON for value in velocities)


def resolve_transport_authority(
    evidence: Sequence[TransportEvidence], now_s: float
) -> tuple[str, np.ndarray]:
    window = [
        value
        for value in evidence
        if now_s - value.time_s <= x24.TRACK_HISTORY_S + EPSILON
    ]
    if (
        len(window) < x27.MINIMUM_AUTHORITY_PAIRS
        or window[-1].time_s
        - window[-x27.MINIMUM_AUTHORITY_PAIRS].time_s
        < x24.MINIMUM_FIT_SPAN_S - EPSILON
        or not any(value.occupancy_anchored for value in window)
        or not direction_consistent_transport(window)
    ):
        return x27.UNAUTHORIZED_MOTION, np.zeros(2, dtype=np.float64)
    return x27.RIGID_DYNAMIC, np.median(
        np.stack([transport_velocity(value) for value in window]), axis=0
    )


@dataclass
class LineageTrack:
    track_id: str
    class_id: int
    class_name: str
    last_seen_s: float = -math.inf
    last_measurement_time_s: float = -math.inf
    last_center_xy: np.ndarray | None = None
    last_world_cells: frozenset[x27.Cell] | None = None
    last_wearer_cells: frozenset[x27.Cell] | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    depth_support: int | None = None
    evidence: list[x27.AuthorityEvidence] = field(default_factory=list)
    transport_evidence: list[TransportEvidence] = field(default_factory=list)
    authority: str = x27.UNAUTHORIZED_MOTION
    authorized_velocity_xy: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    last_world_alignment: x27.LatticeAlignment | None = None
    last_wearer_alignment: x27.LatticeAlignment | None = None
    world_lineage: list[frozenset[x27.Cell]] = field(default_factory=list)
    wearer_lineage: list[frozenset[x27.Cell]] = field(default_factory=list)
    measurements: int = 0
    authority_transitions: int = 0
    transport_contradictions: int = 0
    transport_anchors: int = 0
    transport_conflicted: bool = False
    last_transport_state: str = TRANSPORT_NEUTRAL
    last_transport_shift_cells: x27.Cell = (0, 0)


def fixed_constants() -> dict[str, Any]:
    return {
        **x27.fixed_constants(),
        "representation": "TEMPORAL_ALIGNED_OCCUPANCY_LINEAGE",
        "risk_geometry": "REPEATED_CELL_LINEAGE_ENVELOPE",
        "lineage_observations": LINEAGE_OBSERVATIONS,
        "repeated_support_observations": REPEATED_SUPPORT_OBSERVATIONS,
        "minimum_lineage_cells": x27.MINIMUM_ALIGNMENT_CELLS,
        "lineage_update": "SHIFT_ALL_PRIOR_SUPPORTS_THEN_APPEND_CURRENT",
        "authority_geometry_decoupling": (
            "CORE_SHAPE_CHANGE_NEVER_RESETS_MOTION_EVIDENCE"
        ),
        "motion_authority_representation": (
            "DIRECTION_CONSISTENT_CENTER_TRANSPORT_WITH_OCCUPANCY_ANCHOR"
        ),
        "transport_quantization_cell_m": x27.LATTICE_CELL_SIZE_M,
        "minimum_transport_pairs": x27.MINIMUM_AUTHORITY_PAIRS,
        "minimum_transport_span_seconds": x24.MINIMUM_FIT_SPAN_S,
        "transport_anchor_rule": (
            "AT_LEAST_ONE_AUTHORIZED_NONZERO_OCCUPANCY_TRANSLATION_IN_HISTORY"
        ),
        "transport_continuation_rule": (
            "LOW_OVERLAP_NONZERO_SAME_DIRECTION_EXTENDS_AUTHORITY_LINEAGE"
        ),
        "transport_reset_rule": (
            "DIRECTION_CONTRADICTION_TERMINATES_TRACK_MOTION_EPOCH_UNTIL_IDENTITY_LOSS"
        ),
        "static_conflict_rule": (
            "NONZERO_TRANSPORT_BLOCKS_STATIC_FALLBACK_UNTIL_HISTORY_EXPIRES"
        ),
        "risk_authorities": [x27.STATIC_SCENE, x27.RIGID_DYNAMIC],
    }


def paths(run_root: Path) -> dict[str, Path]:
    return {
        "x24_freeze": run_root / "freeze-x24.json",
        "x24_predictions": run_root / "predictions-x24.json",
        "freeze": run_root / "freeze-x29.json",
        "predictions": run_root / "predictions-x29.json",
    }


def repeated_support(
    lineage: Sequence[frozenset[x27.Cell]],
) -> tuple[frozenset[x27.Cell], int]:
    counts: Counter[x27.Cell] = Counter(
        cell for observation in lineage for cell in observation
    )
    cells = frozenset(
        cell
        for cell, observations in counts.items()
        if observations >= REPEATED_SUPPORT_OBSERVATIONS
    )
    return cells, max(counts.values(), default=0)


def lineage_footprint(cells: frozenset[x27.Cell]) -> np.ndarray:
    x24.require(
        len(cells) >= x27.MINIMUM_ALIGNMENT_CELLS,
        "x29_lineage_support",
    )
    points = np.asarray(sorted(cells), dtype=np.float64) * x27.LATTICE_CELL_SIZE_M
    low = np.min(points, axis=0) - x27.LATTICE_CELL_SIZE_M * 0.5
    high = np.max(points, axis=0) + x27.LATTICE_CELL_SIZE_M * 0.5
    return np.asarray(
        [
            [low[0], low[1]],
            [high[0], low[1]],
            [high[0], high[1]],
            [low[0], high[1]],
        ],
        dtype=np.float64,
    )


class TemporalLineageTracker:
    def __init__(self) -> None:
        self.tracks: dict[str, LineageTrack] = {}
        self.next_id = 1

    @staticmethod
    def prepare(
        measurements: Sequence[x25.FootprintMeasurement],
        wearer_position_xy: np.ndarray,
    ) -> list[x27.PreparedMeasurement]:
        return [
            x27.PreparedMeasurement(
                value=value,
                world_cells=x27.lattice_cells(value.surface_points_xy),
                wearer_cells=x27.lattice_cells(
                    value.surface_points_xy, wearer_position_xy
                ),
            )
            for value in measurements
        ]

    @staticmethod
    def motion_hint(
        world: x27.LatticeAlignment,
        relative: x27.LatticeAlignment,
        wearer_distance_m: float,
    ) -> str:
        if (
            wearer_distance_m <= route.DEFAULT_TUBE_RADIUS_M + EPSILON
            and relative.shift_cells == (0, 0)
            and relative.overlap_cells >= world.overlap_cells
            and world.shift_cells != (0, 0)
        ):
            return x27.EGO_CARRIED
        if world.shift_cells == (0, 0):
            return x27.STATIC_SCENE
        if world.has_authorized_shift:
            return x27.RIGID_DYNAMIC
        return x27.UNAUTHORIZED_MOTION

    @staticmethod
    def advance_lineage(
        lineage: Sequence[frozenset[x27.Cell]],
        current: frozenset[x27.Cell],
        shift_cells: x27.Cell,
    ) -> list[frozenset[x27.Cell]]:
        shifted = [x27.shifted(value, shift_cells) for value in lineage]
        return [*shifted, current][-LINEAGE_OBSERVATIONS:]

    @staticmethod
    def append_transport(
        track: LineageTrack,
        *,
        now_s: float,
        delta_s: float,
        shift_xy: np.ndarray,
        occupancy_anchored: bool,
    ) -> None:
        track.transport_evidence = [
            value
            for value in track.transport_evidence
            if now_s - value.time_s <= x24.TRACK_HISTORY_S + EPSILON
        ]
        if track.transport_conflicted:
            track.last_transport_state = TRANSPORT_CONFLICTED
            return
        if float(np.linalg.norm(shift_xy)) <= EPSILON:
            track.last_transport_state = TRANSPORT_NEUTRAL
            return
        candidate = TransportEvidence(
            time_s=now_s,
            delta_s=delta_s,
            shift_xy=shift_xy,
            occupancy_anchored=occupancy_anchored,
        )
        if track.transport_evidence:
            reference = np.median(
                np.stack(
                    [
                        transport_velocity(value)
                        for value in track.transport_evidence
                    ]
                ),
                axis=0,
            )
            if float(np.dot(transport_velocity(candidate), reference)) <= EPSILON:
                track.transport_evidence = []
                track.transport_contradictions += 1
                track.transport_conflicted = True
                track.last_transport_state = TRANSPORT_CONFLICTED
                return
        track.transport_evidence.append(candidate)
        track.transport_anchors += int(occupancy_anchored)
        track.last_transport_state = (
            TRANSPORT_ANCHOR if occupancy_anchored else TRANSPORT_CONTINUATION
        )

    def update(
        self,
        measurements: Sequence[x25.FootprintMeasurement],
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
                    track.last_world_cells,
                    measurement.world_cells,
                    delta,
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
                track_id = f"temporal-lineage-{self.next_id:06d}"
                self.next_id += 1
                self.tracks[track_id] = LineageTrack(
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
                x24.require(delta_s > 0.0, f"x29_noncausal_track_time:{track_id}")
                center_delta = measurement.value.position_xy - track.last_center_xy
                world = x27.lattice_alignment(
                    track.last_world_cells,
                    measurement.world_cells,
                    center_delta,
                )
                previous_wearer_center = np.mean(
                    np.asarray(tuple(track.last_wearer_cells), dtype=np.float64),
                    axis=0,
                )
                current_wearer_center = np.mean(
                    np.asarray(tuple(measurement.wearer_cells), dtype=np.float64),
                    axis=0,
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
                transport_cells, transport_shift_xy = quantized_transport(
                    center_delta
                )
                track.last_transport_shift_cells = transport_cells
                self.append_transport(
                    track,
                    now_s=now_s,
                    delta_s=delta_s,
                    shift_xy=transport_shift_xy,
                    occupancy_anchored=(
                        hint == x27.RIGID_DYNAMIC and world.has_authorized_shift
                    ),
                )
                previous_authority = track.authority
                transport_authority, transport_velocity_xy = (
                    resolve_transport_authority(track.transport_evidence, now_s)
                )
                occupancy_authority, occupancy_velocity_xy = x27.resolve_authority(
                    track.evidence, now_s
                )
                if transport_authority == x27.RIGID_DYNAMIC:
                    track.authority = transport_authority
                    track.authorized_velocity_xy = transport_velocity_xy
                elif (
                    track.transport_evidence
                    and occupancy_authority != x27.EGO_CARRIED
                ):
                    track.authority = x27.UNAUTHORIZED_MOTION
                    track.authorized_velocity_xy = np.zeros(2, dtype=np.float64)
                else:
                    track.authority = occupancy_authority
                    track.authorized_velocity_xy = occupancy_velocity_xy
                if track.authority != previous_authority:
                    track.authority_transitions += 1
                lineage_shift = (
                    world.shift_cells
                    if world.has_authorized_shift
                    or hint in {x27.STATIC_SCENE, x27.EGO_CARRIED}
                    else transport_cells
                )
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

    def emitted(self, now_s: float, measured_ids: set[str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for track_id, track in sorted(self.tracks.items()):
            age_s = now_s - track.last_seen_s
            if age_s > x24.HOLD_WINDOW_S + EPSILON:
                continue
            support_cells, maximum_support = repeated_support(track.world_lineage)
            support_valid = len(support_cells) >= x27.MINIMUM_ALIGNMENT_CELLS
            footprint = (
                lineage_footprint(support_cells)
                if support_valid
                else np.zeros((4, 2), dtype=np.float64)
            )
            if track.authority == x27.RIGID_DYNAMIC and support_valid:
                footprint = footprint + (
                    track.authorized_velocity_xy[None, :] * max(0.0, age_s)
                )
            center = (
                np.mean(footprint, axis=0)
                if support_valid
                else np.zeros(2, dtype=np.float64)
            )
            world = track.last_world_alignment
            output.append(
                {
                    "track_id": track_id,
                    "class_id": track.class_id,
                    "class_name": track.class_name,
                    "disposition": (
                        "MEASURED" if track_id in measured_ids else "HOLD"
                    ),
                    "evidence_age_s": max(0.0, age_s),
                    "motion_authority": track.authority,
                    "risk_eligible": (
                        support_valid
                        and track.authority
                        in {x27.STATIC_SCENE, x27.RIGID_DYNAMIC}
                    ),
                    "position_forward_m": float(center[0]),
                    "position_right_m": float(center[1]),
                    "velocity_forward_mps": float(
                        track.authorized_velocity_xy[0]
                    ),
                    "velocity_right_mps": float(
                        track.authorized_velocity_xy[1]
                    ),
                    "footprint_xy": [
                        [float(value) for value in row] for row in footprint
                    ],
                    "footprint_area_m2": (
                        x25.polygon_area(footprint) if support_valid else 0.0
                    ),
                    "lineage_cells": len(support_cells),
                    "lineage_observations": len(track.world_lineage),
                    "maximum_cell_observations": maximum_support,
                    "measurements": track.measurements,
                    "authority_transitions": track.authority_transitions,
                    "transport_lineage_pairs": len(track.transport_evidence),
                    "transport_anchor_pairs": sum(
                        int(value.occupancy_anchored)
                        for value in track.transport_evidence
                    ),
                    "transport_anchors_total": track.transport_anchors,
                    "transport_contradictions": track.transport_contradictions,
                    "transport_conflicted": track.transport_conflicted,
                    "transport_state": track.last_transport_state,
                    "transport_shift_cells": (
                        list(track.last_transport_shift_cells)
                        if track_id in measured_ids
                        else None
                    ),
                    "depth_grid_support": (
                        track.depth_support if track_id in measured_ids else None
                    ),
                    "world_lattice_shift_cells": (
                        list(world.shift_cells)
                        if track_id in measured_ids and world is not None
                        else None
                    ),
                }
            )
        return output


def predict_episode(
    episode: adapter.Episode,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: adapter.CameraCalibration,
) -> dict[str, Any]:
    x24.require(
        len(candidate_values) == len(episode.observations),
        f"x29_candidate_count:{episode.episode_id}",
    )
    tracker = TemporalLineageTracker()
    receipt_cache: dict[Path, dict[str, Any]] = {}
    confirmation = x24.RiskConfirmation()
    previous_mode: str | None = None
    frames: list[dict[str, Any]] = []
    route_mode_counts: Counter[str] = Counter()
    authority_track_frames: Counter[str] = Counter()
    measured_frames = emitted_frames = hold_frames = 0
    for ordinal, (observation, candidate_value) in enumerate(
        zip(episode.observations, candidate_values, strict=True)
    ):
        neighbour = (
            episode.observations[ordinal - 1]
            if ordinal
            else episode.observations[ordinal + 1]
        )
        sample_period_s = abs(observation.time_s - neighbour.time_s)
        x24.require(
            sample_period_s > 0.0,
            f"x29_sample_period:{episode.episode_id}:{ordinal}",
        )
        wearer_position, wearer_velocity = x24.wearer_anchor_state(
            observation, episode.route_frame
        )
        measurements = x25.candidate_measurements(
            observation,
            candidate_value,
            calibration,
            episode.route_frame,
        )
        measured_frames += int(bool(measurements))
        measured_ids = tracker.update(
            measurements, observation.time_s, wearer_position
        )
        tracks = tracker.emitted(observation.time_s, measured_ids)
        risk_tracks = [value for value in tracks if bool(value["risk_eligible"])]
        authority_track_frames.update(
            str(value["motion_authority"]) for value in tracks
        )
        emitted_frames += int(bool(tracks))
        hold_frames += int(
            any(track["disposition"] == "HOLD" for track in tracks)
        )
        receipt = x24.load_receipt(observation, receipt_cache)
        selection = route.select_route(
            receipt,
            session_id=observation.navigation_session_id,
            now_s=observation.time_s,
            wearer_position_xy=wearer_position,
            wearer_velocity_xy=wearer_velocity,
            previous_mode=previous_mode,
        )
        previous_mode = selection.mode
        route_mode_counts[selection.mode] += 1
        frames.append(
            {
                "sample_index": observation.sample_index,
                "time_s": observation.time_s,
                "world_frame": observation.world_frame,
                "raw_candidates": len(candidate_value["candidates"]),
                "metric_footprint_measurements": len(measurements),
                "tracks": tracks,
                "risk_eligible_tracks": len(risk_tracks),
                "arms": {
                    ARM_X29: x25.arm_frame(
                        selection,
                        receipt=receipt,
                        observation=observation,
                        wearer_position=wearer_position,
                        wearer_velocity=wearer_velocity,
                        tracks=risk_tracks,
                        confirmation=confirmation,
                        sample_period_s=sample_period_s,
                    )
                },
            }
        )
    return {
        "episode_id": episode.episode_id,
        "frames": frames,
        "diagnostics": {
            "frame_count": len(frames),
            "metric_footprint_measurement_frames": measured_frames,
            "emitted_track_frames": emitted_frames,
            "track_coverage": emitted_frames / max(1, len(frames)),
            "hold_frames": hold_frames,
            "authority_track_frames": dict(sorted(authority_track_frames.items())),
            "authority_transitions": sum(
                value.authority_transitions for value in tracker.tracks.values()
            ),
            "x29_route_mode_counts": dict(sorted(route_mode_counts.items())),
        },
        "arms": {
            ARM_X29: {
                "route_risk_frames": sum(
                    bool(frame["arms"][ARM_X29]["route_risk"])
                    for frame in frames
                ),
                "first_route_risk_time_s": next(
                    (
                        frame["time_s"]
                        for frame in frames
                        if frame["arms"][ARM_X29]["route_risk"]
                    ),
                    None,
                ),
            }
        },
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(
        not output["freeze"].exists(), f"x29_freeze_exists:{output['freeze']}"
    )
    frozen_x24, contract, _candidates = x24.require_freeze(run_root)
    x24.require(output["x24_predictions"].is_file(), "x29_x24_baseline_missing")
    algorithm_files = {
        "x25_geometry": Path(x25.__file__).resolve(),
        "x27_authority": Path(x27.__file__).resolve(),
        "x29_predictor": Path(__file__).resolve(),
    }
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
            for name, path in algorithm_files.items()
        },
        "episodes": len(contract.episodes),
        "frames": len(x24.flatten_observations(contract)),
        "fixed_constants": fixed_constants(),
        "arm": ARM_X29,
    }
    x24.write_json_exclusive(output["freeze"], value)
    return {**value, "freeze_sha256": x24.sha256_file(output["freeze"])}


def require_freeze(
    run_root: Path,
) -> tuple[dict[str, Any], adapter.SanitizedModelContract, list[dict[str, Any]]]:
    output = paths(run_root)
    frozen = x24.read_json(output["freeze"].resolve(strict=True), "x29_freeze")
    x24.require(frozen.get("schema") == FREEZE_SCHEMA, "x29_freeze_schema")
    x24.require(
        frozen.get("fixed_constants") == fixed_constants(),
        "x29_constants_drift",
    )
    algorithm_files = {
        "x25_geometry": Path(x25.__file__).resolve(),
        "x27_authority": Path(x27.__file__).resolve(),
        "x29_predictor": Path(__file__).resolve(),
    }
    for name, path in algorithm_files.items():
        x24.require(
            frozen["algorithm_files"][name]["sha256"] == x24.sha256_file(path),
            f"x29_algorithm_drift:{name}",
        )
    frozen_x24, contract, candidates = x24.require_freeze(run_root)
    x24.require(
        frozen["source"]["x24_freeze_sha256"]
        == x24.sha256_file(output["x24_freeze"]),
        "x29_x24_freeze_drift",
    )
    x24.require(
        frozen["source"]["x24_predictions_sha256"]
        == x24.sha256_file(output["x24_predictions"]),
        "x29_x24_prediction_drift",
    )
    x24.require(
        frozen["source"]["candidate_aggregate_sha256"]
        == frozen_x24["candidates"]["aggregate_sha256"],
        "x29_candidate_drift",
    )
    return frozen, contract, candidates


def predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(
        not output["predictions"].exists(),
        f"x29_predictions_exist:{output['predictions']}",
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
        "arms": [ARM_X29],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "predict"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    value = freeze(args) if args.command == "freeze" else predict(args)
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
