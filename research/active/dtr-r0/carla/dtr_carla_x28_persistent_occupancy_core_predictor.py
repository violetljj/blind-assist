"""Freeze and predict X28 persistent occupancy-core route risk.

X28 keeps X27's four motion-authority states but separates the latest visible
footprint from the support allowed to enter route risk.  A track earns a
``persistent_core`` only from metric lattice cells that survive consecutive
causal alignments in the selected authority frame.  Authority changes,
identity/support breaks, or an undersized core reset the epoch.  Only STATIC
and RIGID_DYNAMIC cores enter future route risk; EGO_CARRIED and
UNAUTHORIZED_MOTION remain available for association and diagnostics.
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


EXPERIMENT_ID = "DTR_CARLA_X28_PERSISTENT_OCCUPANCY_CORE"
FREEZE_SCHEMA = "blindassist-dtr-carla-x28-persistent-core-freeze-v1"
PREDICTION_SCHEMA = "blindassist-dtr-carla-x28-persistent-core-predictions-v1"
ARM_X28 = "X28_ISSUED_PLAN_PERSISTENT_OCCUPANCY_CORE"
EPSILON = 1e-9


@dataclass
class CoreTrack:
    track_id: str
    class_id: int
    class_name: str
    last_seen_s: float = -math.inf
    last_measurement_time_s: float = -math.inf
    last_center_xy: np.ndarray | None = None
    last_world_cells: frozenset[x27.Cell] | None = None
    last_wearer_cells: frozenset[x27.Cell] | None = None
    world_core_cells: frozenset[x27.Cell] | None = None
    wearer_core_cells: frozenset[x27.Cell] | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    depth_support: int | None = None
    evidence: list[x27.AuthorityEvidence] = field(default_factory=list)
    authority: str = x27.UNAUTHORIZED_MOTION
    authorized_velocity_xy: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    last_world_alignment: x27.LatticeAlignment | None = None
    last_wearer_alignment: x27.LatticeAlignment | None = None
    epoch_measurements: int = 1
    authority_breaks: int = 0


def fixed_constants() -> dict[str, Any]:
    return {
        **x27.fixed_constants(),
        "representation": "PERSISTENT_ROUTE_FRAME_OCCUPANCY_CORE_AUTHORITY",
        "risk_geometry": "AXIS_ALIGNED_LATTICE_CORE_EXPANDED_BY_HALF_CELL",
        "core_minimum_cells": x27.MINIMUM_ALIGNMENT_CELLS,
        "core_update": "INTERSECTION_AFTER_AUTHORITY_FRAME_ALIGNMENT",
        "authority_break": "RESET_CORE_AND_EVIDENCE_EPOCH",
        "risk_authorities": [x27.STATIC_SCENE, x27.RIGID_DYNAMIC],
    }


def paths(run_root: Path) -> dict[str, Path]:
    return {
        "x24_freeze": run_root / "freeze-x24.json",
        "x24_predictions": run_root / "predictions-x24.json",
        "freeze": run_root / "freeze-x28.json",
        "predictions": run_root / "predictions-x28.json",
    }


def core_footprint(cells: frozenset[x27.Cell]) -> np.ndarray:
    x24.require(len(cells) >= x27.MINIMUM_ALIGNMENT_CELLS, "x28_core_support")
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


def aligned_core(
    previous: frozenset[x27.Cell],
    current: frozenset[x27.Cell],
    shift_cells: x27.Cell,
) -> frozenset[x27.Cell]:
    return x27.shifted(previous, shift_cells) & current


class PersistentCoreTracker:
    def __init__(self) -> None:
        self.tracks: dict[str, CoreTrack] = {}
        self.next_id = 1

    @staticmethod
    def prepare(
        measurements: Sequence[x25.FootprintMeasurement], wearer_position_xy: np.ndarray
    ) -> list[x27.PreparedMeasurement]:
        return [
            x27.PreparedMeasurement(
                value=value,
                world_cells=x27.lattice_cells(value.surface_points_xy),
                wearer_cells=x27.lattice_cells(value.surface_points_xy, wearer_position_xy),
            )
            for value in measurements
        ]

    @staticmethod
    def reset_epoch(
        track: CoreTrack,
        measurement: x27.PreparedMeasurement,
        *,
        count_break: bool,
    ) -> None:
        track.world_core_cells = measurement.world_cells
        track.wearer_core_cells = measurement.wearer_cells
        track.evidence.clear()
        track.authority = x27.UNAUTHORIZED_MOTION
        track.authorized_velocity_xy = np.zeros(2, dtype=np.float64)
        track.epoch_measurements = 1
        if count_break:
            track.authority_breaks += 1

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
                    track.last_world_cells, measurement.world_cells, delta
                )
                bbox_overlap = (
                    0.0
                    if track.last_bbox is None
                    else x24.bbox_iou(track.last_bbox, measurement.value.bbox)
                )
                costs.append(
                    (-alignment.overlap_fraction, distance, -bbox_overlap, track_id, index)
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
            is_new = track_id is None
            if is_new:
                track_id = f"persistent-core-{self.next_id:06d}"
                self.next_id += 1
                self.tracks[track_id] = CoreTrack(
                    track_id=track_id,
                    class_id=measurement.value.class_id,
                    class_name=measurement.value.class_name,
                )
            track = self.tracks[track_id]
            if is_new:
                self.reset_epoch(track, measurement, count_break=False)
            elif (
                track.last_center_xy is not None
                and track.last_world_cells is not None
                and track.last_wearer_cells is not None
                and track.world_core_cells is not None
                and track.wearer_core_cells is not None
                and math.isfinite(track.last_measurement_time_s)
            ):
                delta_s = now_s - track.last_measurement_time_s
                x24.require(delta_s > 0.0, f"x28_noncausal_track_time:{track_id}")
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
                if (
                    wearer_distance <= route.DEFAULT_TUBE_RADIUS_M + EPSILON
                    and relative.shift_cells == (0, 0)
                    and relative.overlap_cells >= world.overlap_cells
                    and world.shift_cells != (0, 0)
                ):
                    hint = x27.EGO_CARRIED
                    next_world_core = aligned_core(
                        track.world_core_cells, measurement.world_cells, world.shift_cells
                    )
                    next_wearer_core = track.wearer_core_cells & measurement.wearer_cells
                elif world.shift_cells == (0, 0):
                    hint = x27.STATIC_SCENE
                    next_world_core = track.world_core_cells & measurement.world_cells
                    next_wearer_core = aligned_core(
                        track.wearer_core_cells,
                        measurement.wearer_cells,
                        relative.shift_cells,
                    )
                elif world.has_authorized_shift:
                    hint = x27.RIGID_DYNAMIC
                    next_world_core = aligned_core(
                        track.world_core_cells, measurement.world_cells, world.shift_cells
                    )
                    next_wearer_core = aligned_core(
                        track.wearer_core_cells,
                        measurement.wearer_cells,
                        relative.shift_cells,
                    )
                else:
                    hint = x27.UNAUTHORIZED_MOTION
                    next_world_core = frozenset()
                    next_wearer_core = frozenset()

                previous_hint = track.evidence[-1].hint if track.evidence else None
                core_valid = (
                    len(next_world_core) >= x27.MINIMUM_ALIGNMENT_CELLS
                    and (
                        hint != x27.EGO_CARRIED
                        or len(next_wearer_core) >= x27.MINIMUM_ALIGNMENT_CELLS
                    )
                )
                if hint == x27.UNAUTHORIZED_MOTION or not core_valid or (
                    previous_hint is not None and previous_hint != hint
                ):
                    self.reset_epoch(track, measurement, count_break=True)
                else:
                    shift_xy = (
                        np.asarray(world.shift_cells, dtype=np.float64)
                        * x27.LATTICE_CELL_SIZE_M
                    )
                    track.world_core_cells = next_world_core
                    track.wearer_core_cells = next_wearer_core
                    track.evidence.append(
                        x27.AuthorityEvidence(
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
                    track.epoch_measurements += 1
                    track.authority, track.authorized_velocity_xy = x27.resolve_authority(
                        track.evidence, now_s
                    )
                track.last_world_alignment = world
                track.last_wearer_alignment = relative

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
            if track.world_core_cells is None:
                continue
            age_s = now_s - track.last_seen_s
            if age_s > x24.HOLD_WINDOW_S + EPSILON:
                continue
            core_valid = len(track.world_core_cells) >= x27.MINIMUM_ALIGNMENT_CELLS
            if core_valid:
                footprint = core_footprint(track.world_core_cells)
            else:
                footprint = np.zeros((4, 2), dtype=np.float64)
            if track.authority == x27.RIGID_DYNAMIC and core_valid:
                footprint = footprint + track.authorized_velocity_xy[None, :] * max(0.0, age_s)
            center = np.mean(footprint, axis=0) if core_valid else np.zeros(2, dtype=np.float64)
            world = track.last_world_alignment
            output.append(
                {
                    "track_id": track_id,
                    "class_id": track.class_id,
                    "class_name": track.class_name,
                    "disposition": "MEASURED" if track_id in measured_ids else "HOLD",
                    "evidence_age_s": max(0.0, age_s),
                    "motion_authority": track.authority,
                    "risk_eligible": (
                        core_valid
                        and track.authority in {x27.STATIC_SCENE, x27.RIGID_DYNAMIC}
                    ),
                    "position_forward_m": float(center[0]),
                    "position_right_m": float(center[1]),
                    "velocity_forward_mps": float(track.authorized_velocity_xy[0]),
                    "velocity_right_mps": float(track.authorized_velocity_xy[1]),
                    "footprint_xy": [[float(value) for value in row] for row in footprint],
                    "footprint_area_m2": x25.polygon_area(footprint) if core_valid else 0.0,
                    "persistent_core_cells": len(track.world_core_cells),
                    "epoch_measurements": track.epoch_measurements,
                    "authority_breaks": track.authority_breaks,
                    "depth_grid_support": track.depth_support if track_id in measured_ids else None,
                    "world_lattice_shift_cells": (
                        list(world.shift_cells) if track_id in measured_ids and world is not None else None
                    ),
                }
            )
        return output


def predict_episode(
    episode: adapter.Episode,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: adapter.CameraCalibration,
) -> dict[str, Any]:
    x24.require(len(candidate_values) == len(episode.observations), f"x28_candidate_count:{episode.episode_id}")
    tracker = PersistentCoreTracker()
    receipt_cache: dict[Path, dict[str, Any]] = {}
    confirmation = x24.RiskConfirmation()
    previous_mode: str | None = None
    frames: list[dict[str, Any]] = []
    route_mode_counts: Counter[str] = Counter()
    authority_track_frames: Counter[str] = Counter()
    measured_frames = emitted_frames = hold_frames = authority_breaks = 0
    for ordinal, (observation, candidate_value) in enumerate(zip(episode.observations, candidate_values)):
        neighbour = episode.observations[ordinal - 1] if ordinal else episode.observations[ordinal + 1]
        sample_period_s = abs(observation.time_s - neighbour.time_s)
        x24.require(sample_period_s > 0.0, f"x28_sample_period:{episode.episode_id}:{ordinal}")
        wearer_position, wearer_velocity = x24.wearer_anchor_state(observation, episode.route_frame)
        measurements = x25.candidate_measurements(
            observation, candidate_value, calibration, episode.route_frame
        )
        measured_frames += int(bool(measurements))
        measured_ids = tracker.update(measurements, observation.time_s, wearer_position)
        tracks = tracker.emitted(observation.time_s, measured_ids)
        risk_tracks = [value for value in tracks if bool(value["risk_eligible"])]
        authority_track_frames.update(str(value["motion_authority"]) for value in tracks)
        authority_breaks = sum(value.authority_breaks for value in tracker.tracks.values())
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
                    ARM_X28: x25.arm_frame(
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
            "authority_breaks": authority_breaks,
            "x28_route_mode_counts": dict(sorted(route_mode_counts.items())),
        },
        "arms": {
            ARM_X28: {
                "route_risk_frames": sum(bool(frame["arms"][ARM_X28]["route_risk"]) for frame in frames),
                "first_route_risk_time_s": next(
                    (frame["time_s"] for frame in frames if frame["arms"][ARM_X28]["route_risk"]),
                    None,
                ),
            }
        },
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(not output["freeze"].exists(), f"x28_freeze_exists:{output['freeze']}")
    frozen_x24, contract, _candidates = x24.require_freeze(run_root)
    x24.require(output["x24_predictions"].is_file(), "x28_x24_baseline_missing")
    algorithm_files = {
        "x25_geometry": Path(x25.__file__).resolve(),
        "x27_authority": Path(x27.__file__).resolve(),
        "x28_predictor": Path(__file__).resolve(),
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
        "arm": ARM_X28,
    }
    x24.write_json_exclusive(output["freeze"], value)
    return {**value, "freeze_sha256": x24.sha256_file(output["freeze"])}


def require_freeze(
    run_root: Path,
) -> tuple[dict[str, Any], adapter.SanitizedModelContract, list[dict[str, Any]]]:
    output = paths(run_root)
    frozen = x24.read_json(output["freeze"].resolve(strict=True), "x28_freeze")
    x24.require(frozen.get("schema") == FREEZE_SCHEMA, "x28_freeze_schema")
    x24.require(frozen.get("fixed_constants") == fixed_constants(), "x28_constants_drift")
    algorithm_files = {
        "x25_geometry": Path(x25.__file__).resolve(),
        "x27_authority": Path(x27.__file__).resolve(),
        "x28_predictor": Path(__file__).resolve(),
    }
    for name, path in algorithm_files.items():
        x24.require(
            frozen["algorithm_files"][name]["sha256"] == x24.sha256_file(path),
            f"x28_algorithm_drift:{name}",
        )
    frozen_x24, contract, candidates = x24.require_freeze(run_root)
    x24.require(
        frozen["source"]["x24_freeze_sha256"] == x24.sha256_file(output["x24_freeze"]),
        "x28_x24_freeze_drift",
    )
    x24.require(
        frozen["source"]["x24_predictions_sha256"]
        == x24.sha256_file(output["x24_predictions"]),
        "x28_x24_prediction_drift",
    )
    x24.require(
        frozen["source"]["candidate_aggregate_sha256"]
        == frozen_x24["candidates"]["aggregate_sha256"],
        "x28_candidate_drift",
    )
    return frozen, contract, candidates


def predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(not output["predictions"].exists(), f"x28_predictions_exist:{output['predictions']}")
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
        "arms": [ARM_X28],
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
