"""Freeze and predict X27 route-frame occupancy-authority risk.

X27 replaces one fitted footprint velocity with a causal four-hypothesis
authority decomposition over a metric occupancy lattice:

* ``EGO_CARRIED`` support is stable in wearer-relative coordinates and is not
  an external obstacle;
* ``STATIC_SCENE`` support is stable in the fixed route frame and carries zero
  world velocity;
* ``RIGID_DYNAMIC`` support wins a repeatable non-zero lattice translation and
  may carry that authorized velocity through HOLD;
* ``UNAUTHORIZED_MOTION`` preserves the latest occupied footprint but may not
  sweep a speculative velocity into the route.

The detector, metric RGB-D projection, quantile OBB, issued-plan gate, route
tube, confirmation duration, route horizon, and HOLD duration remain inherited
from X24/X25.  The evaluator is not imported or opened.
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


EXPERIMENT_ID = "DTR_CARLA_X27_ROUTE_FRAME_OCCUPANCY_AUTHORITY"
FREEZE_SCHEMA = "blindassist-dtr-carla-x27-occupancy-authority-freeze-v1"
PREDICTION_SCHEMA = "blindassist-dtr-carla-x27-occupancy-authority-predictions-v1"
ARM_X27 = "X27_ISSUED_PLAN_OCCUPANCY_AUTHORITY"

EGO_CARRIED = "EGO_CARRIED"
STATIC_SCENE = "STATIC_SCENE"
RIGID_DYNAMIC = "RIGID_DYNAMIC"
UNAUTHORIZED_MOTION = "UNAUTHORIZED_MOTION"
AUTHORITIES = (EGO_CARRIED, STATIC_SCENE, RIGID_DYNAMIC, UNAUTHORIZED_MOTION)

# Four evidence samples span one route-tube radius.  This is derived from the
# inherited geometry and fit contract rather than selected on a scored cohort.
LATTICE_CELL_SIZE_M = route.DEFAULT_TUBE_RADIUS_M / x24.MINIMUM_FIT_SAMPLES
ALIGNMENT_NEIGHBOUR_CELLS = 1
MINIMUM_ALIGNMENT_CELLS = x25.MINIMUM_REGISTRATION_PAIRS
MINIMUM_AUTHORITY_PAIRS = x24.MINIMUM_FIT_SAMPLES - 1
EPSILON = 1e-9

Cell = tuple[int, int]


@dataclass(frozen=True)
class LatticeAlignment:
    shift_cells: Cell
    overlap_cells: int
    zero_overlap_cells: int
    support_cells: int

    @property
    def overlap_fraction(self) -> float:
        return self.overlap_cells / max(1, self.support_cells)

    @property
    def has_authorized_shift(self) -> bool:
        return (
            self.shift_cells != (0, 0)
            and self.overlap_cells > self.zero_overlap_cells
            and self.overlap_cells >= MINIMUM_ALIGNMENT_CELLS
        )


@dataclass(frozen=True)
class AuthorityEvidence:
    time_s: float
    delta_s: float
    hint: str
    shift_xy: np.ndarray
    world_alignment: LatticeAlignment
    wearer_alignment: LatticeAlignment


@dataclass
class OccupancyTrack:
    track_id: str
    class_id: int
    class_name: str
    last_seen_s: float = -math.inf
    last_measurement_time_s: float = -math.inf
    last_center_xy: np.ndarray | None = None
    last_footprint_xy: np.ndarray | None = None
    last_world_cells: frozenset[Cell] | None = None
    last_wearer_cells: frozenset[Cell] | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    depth_support: int | None = None
    evidence: list[AuthorityEvidence] = field(default_factory=list)
    authority: str = UNAUTHORIZED_MOTION
    authorized_velocity_xy: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    last_world_alignment: LatticeAlignment | None = None
    last_wearer_alignment: LatticeAlignment | None = None


@dataclass(frozen=True)
class PreparedMeasurement:
    value: x25.FootprintMeasurement
    world_cells: frozenset[Cell]
    wearer_cells: frozenset[Cell]


def fixed_constants() -> dict[str, Any]:
    return {
        **x24.fixed_constants(),
        "representation": "ROUTE_FRAME_OCCUPANCY_LATTICE_AUTHORITY",
        "footprint_quantiles": [x25.FOOTPRINT_LOW_QUANTILE, x25.FOOTPRINT_HIGH_QUANTILE],
        "occupancy_lattice_cell_m": LATTICE_CELL_SIZE_M,
        "alignment_neighbour_cells": ALIGNMENT_NEIGHBOUR_CELLS,
        "minimum_alignment_cells": MINIMUM_ALIGNMENT_CELLS,
        "minimum_authority_pairs": MINIMUM_AUTHORITY_PAIRS,
        "authority_states": list(AUTHORITIES),
        "alignment_tie_break": "MAX_OVERLAP_THEN_MINIMUM_TRANSLATION_STATIC_FIRST",
        "ego_carried_rule": "WEARER_RELATIVE_ZERO_SHIFT_AND_WORLD_NONZERO_SHIFT_WITHIN_ROUTE_TUBE",
        "dynamic_rule": "REPEATED_DIRECTION_CONSISTENT_NONZERO_OCCUPANCY_TRANSLATION",
        "unauthorized_hold_rule": "PRESERVE_DIAGNOSTIC_OCCUPANCY_WITHOUT_ROUTE_RISK_AUTHORITY",
        "route_time_step_seconds": x25.ROUTE_TIME_STEP_S,
        "route_entry_bisection_iterations": x25.BISECTION_ITERATIONS,
    }


def paths(run_root: Path) -> dict[str, Path]:
    return {
        "x24_freeze": run_root / "freeze-x24.json",
        "x24_predictions": run_root / "predictions-x24.json",
        "freeze": run_root / "freeze-x27.json",
        "predictions": run_root / "predictions-x27.json",
    }


def lattice_cells(points_xy: np.ndarray, origin_xy: np.ndarray | None = None) -> frozenset[Cell]:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if origin_xy is not None:
        points = points - np.asarray(origin_xy, dtype=np.float64).reshape(1, 2)
    quantized = np.rint(points / LATTICE_CELL_SIZE_M).astype(np.int64)
    return frozenset((int(row[0]), int(row[1])) for row in quantized)


def shifted(cells: frozenset[Cell], delta: Cell) -> frozenset[Cell]:
    dx, dy = delta
    return frozenset((x + dx, y + dy) for x, y in cells)


def lattice_alignment(
    previous: frozenset[Cell],
    current: frozenset[Cell],
    initial_shift_xy: np.ndarray,
) -> LatticeAlignment:
    support = min(len(previous), len(current))
    zero_overlap = len(previous & current)
    initial = np.rint(
        np.asarray(initial_shift_xy, dtype=np.float64).reshape(2) / LATTICE_CELL_SIZE_M
    ).astype(np.int64)
    candidates: set[Cell] = {(0, 0)}
    for dx in range(-ALIGNMENT_NEIGHBOUR_CELLS, ALIGNMENT_NEIGHBOUR_CELLS + 1):
        for dy in range(-ALIGNMENT_NEIGHBOUR_CELLS, ALIGNMENT_NEIGHBOUR_CELLS + 1):
            value = (int(initial[0] + dx), int(initial[1] + dy))
            if math.hypot(*value) * LATTICE_CELL_SIZE_M <= x24.ASSOCIATION_DISTANCE_M + EPSILON:
                candidates.add(value)
    scored = [
        (
            len(shifted(previous, value) & current),
            -math.hypot(*value),
            -abs(value[0]),
            -abs(value[1]),
            value,
        )
        for value in candidates
    ]
    overlap, _negative_norm, _negative_x, _negative_y, best = max(scored)
    return LatticeAlignment(
        shift_cells=best,
        overlap_cells=int(overlap),
        zero_overlap_cells=int(zero_overlap),
        support_cells=int(support),
    )


def alignment_velocity(evidence: AuthorityEvidence) -> np.ndarray:
    return evidence.shift_xy / evidence.delta_s


def direction_consistent(rows: Sequence[AuthorityEvidence]) -> bool:
    velocities = np.stack([alignment_velocity(value) for value in rows])
    median = np.median(velocities, axis=0)
    if float(np.linalg.norm(median)) <= EPSILON:
        return False
    return all(float(np.dot(value, median)) > EPSILON for value in velocities)


def resolve_authority(
    evidence: Sequence[AuthorityEvidence], now_s: float
) -> tuple[str, np.ndarray]:
    window = [value for value in evidence if now_s - value.time_s <= x24.TRACK_HISTORY_S + EPSILON]
    if not window:
        return UNAUTHORIZED_MOTION, np.zeros(2, dtype=np.float64)
    trailing = [window[-1]]
    for value in reversed(window[:-1]):
        if value.hint != trailing[-1].hint:
            break
        trailing.append(value)
    trailing.reverse()
    if len(trailing) >= MINIMUM_AUTHORITY_PAIRS and trailing[-1].hint == EGO_CARRIED:
        return EGO_CARRIED, np.zeros(2, dtype=np.float64)
    if (
        len(trailing) >= MINIMUM_AUTHORITY_PAIRS
        and trailing[-1].hint == RIGID_DYNAMIC
        and trailing[-1].time_s - trailing[-MINIMUM_AUTHORITY_PAIRS].time_s
        >= x24.MINIMUM_FIT_SPAN_S - EPSILON
        and direction_consistent(trailing)
    ):
        velocity = np.median(
            np.stack([alignment_velocity(value) for value in trailing]), axis=0
        )
        return RIGID_DYNAMIC, velocity
    if len(trailing) >= MINIMUM_AUTHORITY_PAIRS and trailing[-1].hint == STATIC_SCENE:
        return STATIC_SCENE, np.zeros(2, dtype=np.float64)
    return UNAUTHORIZED_MOTION, np.zeros(2, dtype=np.float64)


class OccupancyAuthorityTracker:
    def __init__(self) -> None:
        self.tracks: dict[str, OccupancyTrack] = {}
        self.next_id = 1

    def prepare(
        self,
        measurements: Sequence[x25.FootprintMeasurement],
        wearer_position_xy: np.ndarray,
    ) -> list[PreparedMeasurement]:
        return [
            PreparedMeasurement(
                value=value,
                world_cells=lattice_cells(value.surface_points_xy),
                wearer_cells=lattice_cells(value.surface_points_xy, wearer_position_xy),
            )
            for value in measurements
        ]

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
                alignment = lattice_alignment(track.last_world_cells, measurement.world_cells, delta)
                overlap = alignment.overlap_fraction
                bbox_overlap = (
                    0.0
                    if track.last_bbox is None
                    else x24.bbox_iou(track.last_bbox, measurement.value.bbox)
                )
                costs.append((-overlap, distance, -bbox_overlap, track_id, index))
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
                track_id = f"occupancy-authority-{self.next_id:06d}"
                self.next_id += 1
                self.tracks[track_id] = OccupancyTrack(
                    track_id=track_id,
                    class_id=measurement.value.class_id,
                    class_name=measurement.value.class_name,
                )
            track = self.tracks[track_id]
            if (
                track.last_center_xy is not None
                and track.last_world_cells is not None
                and track.last_wearer_cells is not None
                and math.isfinite(track.last_measurement_time_s)
            ):
                delta_s = now_s - track.last_measurement_time_s
                x24.require(delta_s > 0.0, f"x27_noncausal_track_time:{track_id}")
                center_delta = measurement.value.position_xy - track.last_center_xy
                world = lattice_alignment(
                    track.last_world_cells, measurement.world_cells, center_delta
                )
                wearer_delta = np.mean(
                    np.asarray(tuple(measurement.wearer_cells), dtype=np.float64), axis=0
                ) - np.mean(np.asarray(tuple(track.last_wearer_cells), dtype=np.float64), axis=0)
                relative = lattice_alignment(
                    track.last_wearer_cells,
                    measurement.wearer_cells,
                    wearer_delta * LATTICE_CELL_SIZE_M,
                )
                wearer_distance = x25.point_polygon_distance(
                    wearer, measurement.value.footprint_xy
                )
                if (
                    wearer_distance <= route.DEFAULT_TUBE_RADIUS_M + EPSILON
                    and relative.shift_cells == (0, 0)
                    and relative.overlap_cells >= world.overlap_cells
                    and world.shift_cells != (0, 0)
                ):
                    hint = EGO_CARRIED
                elif world.shift_cells == (0, 0):
                    hint = STATIC_SCENE
                elif world.has_authorized_shift:
                    hint = RIGID_DYNAMIC
                else:
                    hint = UNAUTHORIZED_MOTION
                shift_xy = np.asarray(world.shift_cells, dtype=np.float64) * LATTICE_CELL_SIZE_M
                track.evidence.append(
                    AuthorityEvidence(
                        time_s=now_s,
                        delta_s=delta_s,
                        hint=hint,
                        shift_xy=shift_xy,
                        world_alignment=world,
                        wearer_alignment=relative,
                    )
                )
                track.evidence = [
                    value
                    for value in track.evidence
                    if now_s - value.time_s <= x24.TRACK_HISTORY_S + EPSILON
                ]
                track.last_world_alignment = world
                track.last_wearer_alignment = relative
                track.authority, track.authorized_velocity_xy = resolve_authority(
                    track.evidence, now_s
                )
            track.last_seen_s = now_s
            track.last_measurement_time_s = now_s
            track.last_center_xy = measurement.value.position_xy.copy()
            track.last_footprint_xy = measurement.value.footprint_xy.copy()
            track.last_world_cells = measurement.world_cells
            track.last_wearer_cells = measurement.wearer_cells
            track.last_bbox = measurement.value.bbox
            track.depth_support = measurement.value.depth_support
            measured_ids.add(track_id)
        return measured_ids

    def emitted(self, now_s: float, measured_ids: set[str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for track_id, track in sorted(self.tracks.items()):
            if track.last_footprint_xy is None:
                continue
            age_s = now_s - track.last_seen_s
            if age_s > x24.HOLD_WINDOW_S + EPSILON:
                continue
            translation = (
                track.authorized_velocity_xy * max(0.0, age_s)
                if track.authority == RIGID_DYNAMIC
                else np.zeros(2, dtype=np.float64)
            )
            footprint = track.last_footprint_xy + translation[None, :]
            center = np.mean(footprint, axis=0)
            world = track.last_world_alignment
            relative = track.last_wearer_alignment
            output.append(
                {
                    "track_id": track_id,
                    "class_id": track.class_id,
                    "class_name": track.class_name,
                    "disposition": "MEASURED" if track_id in measured_ids else "HOLD",
                    "evidence_age_s": max(0.0, age_s),
                    "motion_authority": track.authority,
                    "risk_eligible": track.authority in {STATIC_SCENE, RIGID_DYNAMIC},
                    "position_forward_m": float(center[0]),
                    "position_right_m": float(center[1]),
                    "velocity_forward_mps": float(track.authorized_velocity_xy[0]),
                    "velocity_right_mps": float(track.authorized_velocity_xy[1]),
                    "footprint_xy": [[float(value) for value in row] for row in footprint],
                    "footprint_area_m2": x25.polygon_area(footprint),
                    "depth_grid_support": track.depth_support if track_id in measured_ids else None,
                    "world_lattice_shift_cells": (
                        list(world.shift_cells) if track_id in measured_ids and world is not None else None
                    ),
                    "world_lattice_overlap_cells": (
                        world.overlap_cells if track_id in measured_ids and world is not None else None
                    ),
                    "world_lattice_zero_overlap_cells": (
                        world.zero_overlap_cells if track_id in measured_ids and world is not None else None
                    ),
                    "wearer_lattice_shift_cells": (
                        list(relative.shift_cells)
                        if track_id in measured_ids and relative is not None
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
    x24.require(len(candidate_values) == len(episode.observations), f"x27_candidate_count:{episode.episode_id}")
    tracker = OccupancyAuthorityTracker()
    receipt_cache: dict[Path, dict[str, Any]] = {}
    confirmation = x24.RiskConfirmation()
    previous_mode: str | None = None
    frames: list[dict[str, Any]] = []
    route_mode_counts: Counter[str] = Counter()
    authority_track_frames: Counter[str] = Counter()
    measured_frames = emitted_frames = hold_frames = 0
    for ordinal, (observation, candidate_value) in enumerate(zip(episode.observations, candidate_values)):
        neighbour = episode.observations[ordinal - 1] if ordinal else episode.observations[ordinal + 1]
        sample_period_s = abs(observation.time_s - neighbour.time_s)
        x24.require(sample_period_s > 0.0, f"x27_sample_period:{episode.episode_id}:{ordinal}")
        wearer_position, wearer_velocity = x24.wearer_anchor_state(observation, episode.route_frame)
        measurements = x25.candidate_measurements(
            observation, candidate_value, calibration, episode.route_frame
        )
        measured_frames += int(bool(measurements))
        measured_ids = tracker.update(measurements, observation.time_s, wearer_position)
        tracks = tracker.emitted(observation.time_s, measured_ids)
        risk_tracks = [value for value in tracks if bool(value["risk_eligible"])]
        authority_track_frames.update(str(value["motion_authority"]) for value in tracks)
        emitted_frames += int(bool(tracks))
        hold_frames += int(any(track["disposition"] == "HOLD" for track in tracks))
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
                    ARM_X27: x25.arm_frame(
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
            "x27_route_mode_counts": dict(sorted(route_mode_counts.items())),
        },
        "arms": {
            ARM_X27: {
                "route_risk_frames": sum(bool(frame["arms"][ARM_X27]["route_risk"]) for frame in frames),
                "first_route_risk_time_s": next(
                    (frame["time_s"] for frame in frames if frame["arms"][ARM_X27]["route_risk"]),
                    None,
                ),
            }
        },
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(not output["freeze"].exists(), f"x27_freeze_exists:{output['freeze']}")
    frozen_x24, contract, _candidates = x24.require_freeze(run_root)
    x24.require(output["x24_predictions"].is_file(), "x27_x24_baseline_missing")
    algorithm_files = {
        "x25_geometry": Path(x25.__file__).resolve(),
        "x27_predictor": Path(__file__).resolve(),
    }
    value = {
        "schema": FREEZE_SCHEMA,
        "status": "FROZEN_TRUTH_BLIND_PENDING_PREDICTION",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "source": {
            "x24_freeze_sha256": x24.sha256_file(output["x24_freeze"]),
            "x24_predictions_sha256": x24.sha256_file(output["x24_predictions"]),
            "model_manifest_sha256": contract.manifest_sha256,
            "candidate_aggregate_sha256": frozen_x24["candidates"]["aggregate_sha256"],
        },
        "algorithm_files": {
            name: {"path": str(path), "sha256": x24.sha256_file(path)}
            for name, path in algorithm_files.items()
        },
        "episodes": len(contract.episodes),
        "frames": len(x24.flatten_observations(contract)),
        "fixed_constants": fixed_constants(),
        "arm": ARM_X27,
    }
    x24.write_json_exclusive(output["freeze"], value)
    return {**value, "freeze_sha256": x24.sha256_file(output["freeze"])}


def require_freeze(
    run_root: Path,
) -> tuple[dict[str, Any], adapter.SanitizedModelContract, list[dict[str, Any]]]:
    output = paths(run_root)
    frozen = x24.read_json(output["freeze"].resolve(strict=True), "x27_freeze")
    x24.require(frozen.get("schema") == FREEZE_SCHEMA, "x27_freeze_schema")
    x24.require(frozen.get("fixed_constants") == fixed_constants(), "x27_constants_drift")
    algorithm_files = {
        "x25_geometry": Path(x25.__file__).resolve(),
        "x27_predictor": Path(__file__).resolve(),
    }
    for name, path in algorithm_files.items():
        x24.require(
            frozen["algorithm_files"][name]["sha256"] == x24.sha256_file(path),
            f"x27_algorithm_drift:{name}",
        )
    frozen_x24, contract, candidates = x24.require_freeze(run_root)
    x24.require(
        frozen["source"]["x24_freeze_sha256"] == x24.sha256_file(output["x24_freeze"]),
        "x27_x24_freeze_drift",
    )
    x24.require(
        frozen["source"]["x24_predictions_sha256"]
        == x24.sha256_file(output["x24_predictions"]),
        "x27_x24_prediction_drift",
    )
    x24.require(
        frozen["source"]["candidate_aggregate_sha256"]
        == frozen_x24["candidates"]["aggregate_sha256"],
        "x27_candidate_drift",
    )
    return frozen, contract, candidates


def predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(not output["predictions"].exists(), f"x27_predictions_exist:{output['predictions']}")
    frozen, contract, candidate_values = require_freeze(run_root)
    cursor = 0
    episodes: dict[str, Any] = {}
    for episode in contract.episodes:
        count = len(episode.observations)
        episodes[episode.episode_id] = predict_episode(
            episode, candidate_values[cursor : cursor + count], contract.calibration
        )
        cursor += count
    value = {
        "schema": PREDICTION_SCHEMA,
        "status": "SEALED_TRUTH_BLIND_PENDING_SCORE",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "arms": [ARM_X27],
        "episodes": episodes,
        "fixed_constants": fixed_constants(),
        "source": {"freeze_sha256": x24.sha256_file(output["freeze"]), **frozen["source"]},
        "claim_boundary": {
            "synthetic_development": True,
            "evaluator_opened": False,
            "current_actor_oracle_used": False,
        },
    }
    x24.write_json_exclusive(output["predictions"], value)
    return {**value, "predictions_sha256": x24.sha256_file(output["predictions"])}


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
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
