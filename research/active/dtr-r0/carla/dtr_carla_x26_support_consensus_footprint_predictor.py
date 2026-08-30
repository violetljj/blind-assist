"""Freeze and predict X26 support-consensus footprint transport.

The footprint itself is X25's metric quantile OBB.  Motion authority comes
from its opposing support boundaries: rigid translation moves both supports
together, whereas partial visibility normally erodes one boundary and leaves
the other stable.  Per axis X26 carries the smaller-magnitude support delta,
then fits the unchanged causal robust-motion window and sweeps the transported
footprint against the issued route.
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


EXPERIMENT_ID = "DTR_CARLA_X26_SUPPORT_CONSENSUS_FOOTPRINT_TRACK"
FREEZE_SCHEMA = "blindassist-dtr-carla-x26-support-consensus-freeze-v1"
PREDICTION_SCHEMA = "blindassist-dtr-carla-x26-support-consensus-predictions-v1"
ARM_X26 = "X26_ISSUED_PLAN_SUPPORT_CONSENSUS_FOOTPRINT"
EPSILON = 1e-9


@dataclass
class SupportConsensusTrack:
    track_id: str
    class_id: int
    class_name: str
    history: list[tuple[float, np.ndarray]] = field(default_factory=list)
    last_seen_s: float = -math.inf
    last_raw_center_xy: np.ndarray | None = None
    last_support_bounds_xy: np.ndarray | None = None
    last_transport_anchor_xy: np.ndarray | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    footprint_offsets_xy: np.ndarray | None = None
    position_xy: np.ndarray | None = None
    velocity_xy: np.ndarray | None = None
    state_time_s: float = -math.inf
    depth_support: int | None = None
    selected_support_delta_xy: np.ndarray | None = None


def fixed_constants() -> dict[str, Any]:
    return {
        **x24.fixed_constants(),
        "representation": "SUPPORT_CONSENSUS_RIGID_2D_QUANTILE_OBB",
        "footprint_quantiles": [x25.FOOTPRINT_LOW_QUANTILE, x25.FOOTPRINT_HIGH_QUANTILE],
        "kinematic_authority": "MINIMUM_MAGNITUDE_DELTA_OF_OPPOSING_AXIS_SUPPORTS",
        "route_time_step_seconds": x25.ROUTE_TIME_STEP_S,
        "route_entry_bisection_iterations": x25.BISECTION_ITERATIONS,
    }


def paths(run_root: Path) -> dict[str, Path]:
    return {
        "x24_freeze": run_root / "freeze-x24.json",
        "x24_predictions": run_root / "predictions-x24.json",
        "freeze": run_root / "freeze-x26.json",
        "predictions": run_root / "predictions-x26.json",
    }


def support_bounds(footprint_xy: np.ndarray) -> np.ndarray:
    footprint = np.asarray(footprint_xy, dtype=np.float64).reshape(-1, 2)
    return np.vstack((np.min(footprint, axis=0), np.max(footprint, axis=0)))


def support_consensus_delta(previous_bounds: np.ndarray, current_bounds: np.ndarray) -> np.ndarray:
    deltas = np.asarray(current_bounds, dtype=np.float64) - np.asarray(previous_bounds, dtype=np.float64)
    return np.asarray(
        [
            deltas[0, axis] if abs(deltas[0, axis]) <= abs(deltas[1, axis]) else deltas[1, axis]
            for axis in range(2)
        ],
        dtype=np.float64,
    )


class SupportConsensusFootprintTracker:
    def __init__(self) -> None:
        self.tracks: dict[str, SupportConsensusTrack] = {}
        self.next_id = 1

    def update(self, measurements: Sequence[x25.FootprintMeasurement], now_s: float) -> set[str]:
        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if now_s - value.last_seen_s <= x24.HOLD_WINDOW_S + EPSILON
        }
        raw_centers = [np.mean(value.footprint_xy, axis=0) for value in measurements]
        costs: list[tuple[float, float, str, int]] = []
        for track_id, track in self.tracks.items():
            if track.last_raw_center_xy is None:
                continue
            for index, measurement in enumerate(measurements):
                if measurement.class_id != track.class_id:
                    continue
                distance = float(np.linalg.norm(track.last_raw_center_xy - raw_centers[index]))
                if distance <= x24.ASSOCIATION_DISTANCE_M + EPSILON:
                    overlap = 0.0 if track.last_bbox is None else x24.bbox_iou(track.last_bbox, measurement.bbox)
                    costs.append((distance, -overlap, track_id, index))
        costs.sort()
        assigned_tracks: set[str] = set()
        assigned_measurements: dict[int, str] = {}
        for _distance, _overlap, track_id, index in costs:
            if track_id not in assigned_tracks and index not in assigned_measurements:
                assigned_tracks.add(track_id)
                assigned_measurements[index] = track_id

        measured_ids: set[str] = set()
        for index, measurement in enumerate(measurements):
            track_id = assigned_measurements.get(index)
            if track_id is None:
                track_id = f"support-footprint-{self.next_id:06d}"
                self.next_id += 1
                self.tracks[track_id] = SupportConsensusTrack(
                    track_id=track_id,
                    class_id=measurement.class_id,
                    class_name=measurement.class_name,
                )
            track = self.tracks[track_id]
            bounds = support_bounds(measurement.footprint_xy)
            raw_center = raw_centers[index]
            selected_delta = np.zeros(2, dtype=np.float64)
            if track.last_support_bounds_xy is None or track.last_transport_anchor_xy is None:
                transport_anchor = raw_center.copy()
            else:
                selected_delta = support_consensus_delta(track.last_support_bounds_xy, bounds)
                transport_anchor = track.last_transport_anchor_xy + selected_delta
            track.history.append((now_s, transport_anchor.copy()))
            track.history = [
                row for row in track.history if now_s - row[0] <= x24.TRACK_HISTORY_S + EPSILON
            ]
            track.last_seen_s = now_s
            track.last_raw_center_xy = raw_center.copy()
            track.last_support_bounds_xy = bounds.copy()
            track.last_transport_anchor_xy = transport_anchor.copy()
            track.last_bbox = measurement.bbox
            track.footprint_offsets_xy = measurement.footprint_xy - transport_anchor[None, :]
            track.depth_support = measurement.depth_support
            track.selected_support_delta_xy = selected_delta
            motion = x24.robust_motion(track.history, now_s)
            if motion is not None:
                track.position_xy, track.velocity_xy = motion
                track.state_time_s = now_s
            measured_ids.add(track_id)
        return measured_ids

    def emitted(self, now_s: float, measured_ids: set[str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for track_id, track in sorted(self.tracks.items()):
            if track.position_xy is None or track.velocity_xy is None or track.footprint_offsets_xy is None:
                continue
            age_s = now_s - track.last_seen_s
            if age_s > x24.HOLD_WINDOW_S + EPSILON:
                continue
            position = track.position_xy + track.velocity_xy * max(0.0, now_s - track.state_time_s)
            footprint = position[None, :] + track.footprint_offsets_xy
            output.append(
                {
                    "track_id": track_id,
                    "class_id": track.class_id,
                    "class_name": track.class_name,
                    "disposition": "MEASURED" if track_id in measured_ids else "HOLD",
                    "evidence_age_s": max(0.0, age_s),
                    "position_forward_m": float(position[0]),
                    "position_right_m": float(position[1]),
                    "velocity_forward_mps": float(track.velocity_xy[0]),
                    "velocity_right_mps": float(track.velocity_xy[1]),
                    "footprint_xy": [[float(value) for value in row] for row in footprint],
                    "footprint_area_m2": x25.polygon_area(footprint),
                    "depth_grid_support": track.depth_support if track_id in measured_ids else None,
                    "selected_support_delta_xy": (
                        [float(value) for value in track.selected_support_delta_xy]
                        if track_id in measured_ids and track.selected_support_delta_xy is not None
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
    x24.require(len(candidate_values) == len(episode.observations), f"x26_candidate_count:{episode.episode_id}")
    tracker = SupportConsensusFootprintTracker()
    receipt_cache: dict[Path, dict[str, Any]] = {}
    confirmation = x24.RiskConfirmation()
    previous_mode: str | None = None
    frames: list[dict[str, Any]] = []
    route_mode_counts: Counter[str] = Counter()
    measured_frames = emitted_frames = hold_frames = 0
    for ordinal, (observation, candidate_value) in enumerate(zip(episode.observations, candidate_values)):
        neighbour = episode.observations[ordinal - 1] if ordinal else episode.observations[ordinal + 1]
        sample_period_s = abs(observation.time_s - neighbour.time_s)
        x24.require(sample_period_s > 0.0, f"x26_sample_period:{episode.episode_id}:{ordinal}")
        measurements = x25.candidate_measurements(observation, candidate_value, calibration, episode.route_frame)
        measured_frames += int(bool(measurements))
        measured_ids = tracker.update(measurements, observation.time_s)
        tracks = tracker.emitted(observation.time_s, measured_ids)
        emitted_frames += int(bool(tracks))
        hold_frames += int(any(track["disposition"] == "HOLD" for track in tracks))
        wearer_position, wearer_velocity = x24.wearer_anchor_state(observation, episode.route_frame)
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
                "arms": {
                    ARM_X26: x25.arm_frame(
                        selection,
                        receipt=receipt,
                        observation=observation,
                        wearer_position=wearer_position,
                        wearer_velocity=wearer_velocity,
                        tracks=tracks,
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
            "x26_route_mode_counts": dict(sorted(route_mode_counts.items())),
        },
        "arms": {
            ARM_X26: {
                "route_risk_frames": sum(bool(frame["arms"][ARM_X26]["route_risk"]) for frame in frames),
                "first_route_risk_time_s": next(
                    (frame["time_s"] for frame in frames if frame["arms"][ARM_X26]["route_risk"]),
                    None,
                ),
            }
        },
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(not output["freeze"].exists(), f"x26_freeze_exists:{output['freeze']}")
    frozen_x24, contract, _candidates = x24.require_freeze(run_root)
    x24.require(output["x24_predictions"].is_file(), "x26_x24_baseline_missing")
    algorithm_files = {
        "x25_geometry": Path(x25.__file__).resolve(),
        "x26_predictor": Path(__file__).resolve(),
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
        "arm": ARM_X26,
    }
    x24.write_json_exclusive(output["freeze"], value)
    return {**value, "freeze_sha256": x24.sha256_file(output["freeze"])}


def require_freeze(run_root: Path) -> tuple[dict[str, Any], adapter.SanitizedModelContract, list[dict[str, Any]]]:
    output = paths(run_root)
    frozen = x24.read_json(output["freeze"].resolve(strict=True), "x26_freeze")
    x24.require(frozen.get("schema") == FREEZE_SCHEMA, "x26_freeze_schema")
    x24.require(frozen.get("fixed_constants") == fixed_constants(), "x26_constants_drift")
    algorithm_files = {"x25_geometry": Path(x25.__file__).resolve(), "x26_predictor": Path(__file__).resolve()}
    for name, path in algorithm_files.items():
        x24.require(frozen["algorithm_files"][name]["sha256"] == x24.sha256_file(path), f"x26_algorithm_drift:{name}")
    frozen_x24, contract, candidates = x24.require_freeze(run_root)
    x24.require(frozen["source"]["x24_freeze_sha256"] == x24.sha256_file(output["x24_freeze"]), "x26_x24_freeze_drift")
    x24.require(frozen["source"]["x24_predictions_sha256"] == x24.sha256_file(output["x24_predictions"]), "x26_x24_prediction_drift")
    x24.require(frozen["source"]["candidate_aggregate_sha256"] == frozen_x24["candidates"]["aggregate_sha256"], "x26_candidate_drift")
    return frozen, contract, candidates


def predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(not output["predictions"].exists(), f"x26_predictions_exist:{output['predictions']}")
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
        "arms": [ARM_X26],
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
