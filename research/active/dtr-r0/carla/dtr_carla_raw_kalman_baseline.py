"""Truth-blind classical raw-measurement baselines for DTR CARLA reckoning.

The tracker consumes the same detector/depth metric measurements as X24 but
does not consume X24/X73/X94 tracks or risk decisions.  It is deliberately
small: class-aware nearest-neighbour association, a constant-velocity Kalman
filter, route-tube collision math, and an optional 0.60 s event hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import dtr_carla_rgbd_model_adapter as adapter
import dtr_carla_x24_plan_adherent_predictor as x24
import dtr_carla_x24_plan_route_core as route


EXPERIMENT_ID = "DTR_CARLA_BASELINE_RECKONING_RAW_KALMAN"
PREDICTION_SCHEMA = "blindassist-dtr-carla-raw-kalman-baselines-v1"
ARM_RADIAL = "RAW_KALMAN_RADIAL_TTC"
ARM_ROUTE = "RAW_KALMAN_CV_ROUTE_TUBE"
ARM_HYSTERESIS = "RAW_KALMAN_CV_ROUTE_TUBE_HYSTERESIS_0P60S"
ARMS = (ARM_RADIAL, ARM_ROUTE, ARM_HYSTERESIS)

ASSOCIATION_GATE_M = 1.50
TRACK_MAX_AGE_S = 0.60
EVENT_HOLD_S = 0.60
MINIMUM_TRACK_HITS = 2
POSITION_MEASUREMENT_SD_M = 0.35
INITIAL_POSITION_SD_M = 0.50
INITIAL_VELOCITY_SD_MPS = 3.00
PROCESS_ACCELERATION_SD_MPS2 = 2.00
EPSILON = 1e-9


def fixed_constants() -> dict[str, Any]:
    return {
        "association": "class_aware_greedy_nearest_predicted_position",
        "association_gate_m": ASSOCIATION_GATE_M,
        "track_max_age_s": TRACK_MAX_AGE_S,
        "minimum_track_hits": MINIMUM_TRACK_HITS,
        "position_measurement_sd_m": POSITION_MEASUREMENT_SD_M,
        "initial_position_sd_m": INITIAL_POSITION_SD_M,
        "initial_velocity_sd_mps": INITIAL_VELOCITY_SD_MPS,
        "process_acceleration_sd_mps2": PROCESS_ACCELERATION_SD_MPS2,
        "motion_model": "constant_velocity_kalman_filter",
        "route_horizon_s": route.DEFAULT_ROUTE_HORIZON_S,
        "route_tube_radius_m": route.DEFAULT_TUBE_RADIUS_M,
        "minimum_closing_speed_mps": route.DEFAULT_MIN_CLOSING_SPEED_MPS,
        "event_hold_s": EVENT_HOLD_S,
        "route_contract": "X24_issued_plan_adherence_else_observed_cv",
    }


@dataclass
class KalmanTrack:
    track_id: str
    class_id: int
    class_name: str
    state: np.ndarray
    covariance: np.ndarray
    state_time_s: float
    last_seen_s: float
    hits: int
    last_bbox: tuple[float, float, float, float]
    depth_support: int


def _transition(delta_s: float) -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, delta_s, 0.0],
            [0.0, 1.0, 0.0, delta_s],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _process_noise(delta_s: float) -> np.ndarray:
    dt2 = delta_s * delta_s
    base = np.asarray(
        [
            [dt2 * dt2 / 4.0, 0.0, dt2 * delta_s / 2.0, 0.0],
            [0.0, dt2 * dt2 / 4.0, 0.0, dt2 * delta_s / 2.0],
            [dt2 * delta_s / 2.0, 0.0, dt2, 0.0],
            [0.0, dt2 * delta_s / 2.0, 0.0, dt2],
        ],
        dtype=np.float64,
    )
    return base * PROCESS_ACCELERATION_SD_MPS2**2


class RawKalmanTracker:
    def __init__(self) -> None:
        self.tracks: dict[str, KalmanTrack] = {}
        self.next_id = 1

    @staticmethod
    def _predict(track: KalmanTrack, now_s: float) -> None:
        delta_s = now_s - track.state_time_s
        if delta_s < -EPSILON:
            raise ValueError("kalman_time_reversed")
        if delta_s <= EPSILON:
            return
        transition = _transition(delta_s)
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + _process_noise(delta_s)
        track.state_time_s = now_s

    @staticmethod
    def _correct(track: KalmanTrack, measurement: x24.Measurement) -> None:
        observation = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        noise = np.eye(2, dtype=np.float64) * POSITION_MEASUREMENT_SD_M**2
        innovation = measurement.position_xy - observation @ track.state
        innovation_covariance = observation @ track.covariance @ observation.T + noise
        gain = track.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        track.state = track.state + gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        # Joseph form keeps the covariance symmetric and positive semidefinite.
        residual = identity - gain @ observation
        track.covariance = residual @ track.covariance @ residual.T + gain @ noise @ gain.T

    @staticmethod
    def _new_track(track_id: str, measurement: x24.Measurement, now_s: float) -> KalmanTrack:
        covariance = np.diag(
            [
                INITIAL_POSITION_SD_M**2,
                INITIAL_POSITION_SD_M**2,
                INITIAL_VELOCITY_SD_MPS**2,
                INITIAL_VELOCITY_SD_MPS**2,
            ]
        ).astype(np.float64)
        return KalmanTrack(
            track_id=track_id,
            class_id=measurement.class_id,
            class_name=measurement.class_name,
            state=np.asarray(
                [measurement.position_xy[0], measurement.position_xy[1], 0.0, 0.0],
                dtype=np.float64,
            ),
            covariance=covariance,
            state_time_s=now_s,
            last_seen_s=now_s,
            hits=1,
            last_bbox=measurement.bbox,
            depth_support=measurement.depth_support,
        )

    def update(self, measurements: Sequence[x24.Measurement], now_s: float) -> set[str]:
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if now_s - track.last_seen_s <= TRACK_MAX_AGE_S + EPSILON
        }
        for track in self.tracks.values():
            self._predict(track, now_s)

        costs: list[tuple[float, float, str, int]] = []
        for track_id, track in self.tracks.items():
            for index, measurement in enumerate(measurements):
                if measurement.class_id != track.class_id:
                    continue
                distance = float(np.linalg.norm(track.state[:2] - measurement.position_xy))
                if distance <= ASSOCIATION_GATE_M + EPSILON:
                    overlap = x24.bbox_iou(track.last_bbox, measurement.bbox)
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
                track_id = f"kalman-{self.next_id:06d}"
                self.next_id += 1
                self.tracks[track_id] = self._new_track(track_id, measurement, now_s)
            else:
                track = self.tracks[track_id]
                self._correct(track, measurement)
                track.last_seen_s = now_s
                track.hits += 1
                track.last_bbox = measurement.bbox
                track.depth_support = measurement.depth_support
            measured_ids.add(track_id)
        return measured_ids

    def emitted(self, now_s: float, measured_ids: set[str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for track_id, track in sorted(self.tracks.items()):
            age_s = now_s - track.last_seen_s
            if track.hits < MINIMUM_TRACK_HITS or age_s > TRACK_MAX_AGE_S + EPSILON:
                continue
            output.append(
                {
                    "track_id": track_id,
                    "class_id": track.class_id,
                    "class_name": track.class_name,
                    "disposition": "MEASURED" if track_id in measured_ids else "PREDICTED_HOLD",
                    "evidence_age_s": max(0.0, age_s),
                    "position_forward_m": float(track.state[0]),
                    "position_right_m": float(track.state[1]),
                    "velocity_forward_mps": float(track.state[2]),
                    "velocity_right_mps": float(track.state[3]),
                    "depth_grid_support": track.depth_support if track_id in measured_ids else None,
                    "hits": track.hits,
                }
            )
        return output


@dataclass
class EventHold:
    last_risk_time_s: float | None = None
    last_entry_s: float | None = None
    route_identity: tuple[str, str] | None = None

    def update(
        self,
        *,
        raw_risk: bool,
        raw_entry_s: float | None,
        now_s: float,
        selection: route.RouteSelection,
    ) -> tuple[bool, float | None]:
        identity = (selection.mode, str(selection.receipt_sha256 or ""))
        if raw_risk:
            self.last_risk_time_s = now_s
            self.last_entry_s = raw_entry_s
            self.route_identity = identity
            return True, raw_entry_s
        if (
            self.last_risk_time_s is not None
            and now_s - self.last_risk_time_s <= EVENT_HOLD_S + EPSILON
            and identity == self.route_identity
            and not selection.mode_changed
        ):
            elapsed = now_s - self.last_risk_time_s
            entry = None if self.last_entry_s is None else max(0.0, self.last_entry_s - elapsed)
            return True, entry
        self.last_risk_time_s = None
        self.last_entry_s = None
        self.route_identity = None
        return False, None


def _prediction_row(
    selection: route.RouteSelection,
    *,
    risk: bool,
    entry_s: float | None,
    candidate_ids: Sequence[str],
    raw_risk: bool | None = None,
) -> dict[str, Any]:
    row = {
        "route_mode": selection.mode,
        "authority": selection.authority,
        "plan_receipt_sha256": selection.receipt_sha256,
        "route_mode_changed": selection.mode_changed,
        "route_risk": bool(risk),
        "minimum_entry_s": entry_s if risk else None,
        "candidate_risk_track_ids": sorted(candidate_ids),
    }
    if raw_risk is not None:
        row["raw_route_risk"] = bool(raw_risk)
    return row


def predict_episode(
    episode: adapter.Episode,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: adapter.CameraCalibration,
) -> dict[str, Any]:
    x24.require(len(candidate_values) == len(episode.observations), "raw_kalman_candidate_count")
    tracker = RawKalmanTracker()
    event_hold = EventHold()
    receipt_cache: dict[Path, dict[str, Any]] = {}
    previous_mode: str | None = None
    frames: list[dict[str, Any]] = []
    measurement_frames = emitted_frames = predicted_hold_frames = 0

    for observation, candidate_value in zip(episode.observations, candidate_values):
        measurements = x24.candidate_measurements(
            observation, candidate_value, calibration, episode.route_frame
        )
        measurement_frames += int(bool(measurements))
        measured_ids = tracker.update(measurements, observation.time_s)
        tracks = tracker.emitted(observation.time_s, measured_ids)
        emitted_frames += int(bool(tracks))
        predicted_hold_frames += int(any(row["disposition"] == "PREDICTED_HOLD" for row in tracks))
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

        radial_segments = (
            route.RouteSegment(
                start_offset_s=0.0,
                end_offset_s=route.DEFAULT_ROUTE_HORIZON_S,
                start_position_xy=wearer_position,
                velocity_xy=wearer_velocity,
            ),
        )
        radial_entries: dict[str, float | None] = {}
        route_entries: dict[str, float | None] = {}
        for track in tracks:
            track_id = str(track["track_id"])
            position = (track["position_forward_m"], track["position_right_m"])
            velocity = (track["velocity_forward_mps"], track["velocity_right_mps"])
            radial_entries[track_id] = route.first_metric_tube_entry_s(
                position, velocity, radial_segments
            )
            route_entries[track_id] = route.first_selected_route_entry_s(
                selection,
                receipt=receipt,
                now_s=observation.time_s,
                wearer_position_xy=wearer_position,
                wearer_velocity_xy=wearer_velocity,
                target_position_xy=position,
                target_velocity_xy=velocity,
            )

        radial_hits = {key: value for key, value in radial_entries.items() if value is not None}
        route_hits = {key: value for key, value in route_entries.items() if value is not None}
        radial_entry = min(radial_hits.values()) if radial_hits else None
        route_entry = min(route_hits.values()) if route_hits else None
        held_risk, held_entry = event_hold.update(
            raw_risk=bool(route_hits),
            raw_entry_s=route_entry,
            now_s=observation.time_s,
            selection=selection,
        )
        arms = {
            ARM_RADIAL: _prediction_row(
                selection,
                risk=bool(radial_hits),
                entry_s=radial_entry,
                candidate_ids=tuple(radial_hits),
            ),
            ARM_ROUTE: _prediction_row(
                selection,
                risk=bool(route_hits),
                entry_s=route_entry,
                candidate_ids=tuple(route_hits),
            ),
            ARM_HYSTERESIS: _prediction_row(
                selection,
                risk=held_risk,
                entry_s=held_entry,
                candidate_ids=tuple(route_hits),
                raw_risk=bool(route_hits),
            ),
        }
        frames.append(
            {
                "sample_index": observation.sample_index,
                "time_s": observation.time_s,
                "world_frame": observation.world_frame,
                "raw_candidates": len(candidate_value["candidates"]),
                "metric_measurements": len(measurements),
                "tracks": tracks,
                "arms": arms,
            }
        )

    return {
        "episode_id": episode.episode_id,
        "frames": frames,
        "diagnostics": {
            "frame_count": len(frames),
            "measurement_frames": measurement_frames,
            "emitted_track_frames": emitted_frames,
            "predicted_hold_frames": predicted_hold_frames,
        },
        "arms": {
            arm: {
                "route_risk_frames": sum(bool(frame["arms"][arm]["route_risk"]) for frame in frames)
            }
            for arm in ARMS
        },
    }


__all__ = [
    "ARMS",
    "ARM_HYSTERESIS",
    "ARM_RADIAL",
    "ARM_ROUTE",
    "EventHold",
    "RawKalmanTracker",
    "fixed_constants",
    "predict_episode",
]
