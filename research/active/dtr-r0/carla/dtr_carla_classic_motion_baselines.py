"""Frozen classical-motion and tiny learned references for final reckoning.

The functions here are causal and evaluator-blind.  The finite-difference arm
uses only consecutive associated metric measurements.  CTRV estimates turn
rate only from past measured Kalman velocity headings.  The tiny logistic arm
uses a fixed eight-feature interface and is fit once on FIT_ONLY rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

import dtr_carla_x24_plan_adherent_predictor as x24
import dtr_carla_x24_plan_route_core as route


ARM_FINITE_DIFFERENCE_CV = "FINITE_DIFFERENCE_CV_ROUTE_TUBE"
ARM_CAUSAL_CTRV = "CAUSAL_CTRV_ROUTE_TUBE"
ARM_TINY_LEARNED = "TINY_LEARNED_PREDICTOR"

ASSOCIATION_GATE_M = 1.50
TRACK_MAX_AGE_S = 0.60
MINIMUM_TRACK_HITS = 2
TURN_HISTORY_S = 0.60
MINIMUM_TURN_SAMPLES = 3
MINIMUM_SPEED_FOR_HEADING_MPS = 0.10
MAXIMUM_ABSOLUTE_YAW_RATE_RAD_S = math.radians(120.0)
CTRV_INTEGRATION_STEP_S = 0.05
TINY_FEATURE_NAMES = (
    "radial_entry_or_horizon_s",
    "cv_entry_or_horizon_s",
    "ctrv_entry_or_horizon_s",
    "minimum_track_distance_m_clipped",
    "maximum_track_speed_mps_clipped",
    "confirmed_track_count_clipped",
    "current_measurement_fraction",
    "issued_plan_mode",
)
TINY_L2 = 0.10
TINY_LEARNING_RATE = 0.05
TINY_STEPS = 500
TINY_THRESHOLD = 0.50
EPSILON = 1e-9


def fixed_constants() -> dict[str, Any]:
    return {
        "association_gate_m": ASSOCIATION_GATE_M,
        "track_max_age_s": TRACK_MAX_AGE_S,
        "minimum_track_hits": MINIMUM_TRACK_HITS,
        "finite_difference_velocity": "latest_two_associated_measurements",
        "turn_history_s": TURN_HISTORY_S,
        "minimum_turn_samples": MINIMUM_TURN_SAMPLES,
        "minimum_speed_for_heading_mps": MINIMUM_SPEED_FOR_HEADING_MPS,
        "maximum_absolute_yaw_rate_rad_s": MAXIMUM_ABSOLUTE_YAW_RATE_RAD_S,
        "ctrv_integration_step_s": CTRV_INTEGRATION_STEP_S,
        "tiny_feature_names": list(TINY_FEATURE_NAMES),
        "tiny_l2": TINY_L2,
        "tiny_learning_rate": TINY_LEARNING_RATE,
        "tiny_steps": TINY_STEPS,
        "tiny_threshold": TINY_THRESHOLD,
    }


@dataclass
class FiniteDifferenceTrack:
    track_id: str
    class_id: int
    class_name: str
    position_xy: np.ndarray
    velocity_xy: np.ndarray
    last_seen_s: float
    hits: int
    last_bbox: tuple[float, float, float, float]


class CausalFiniteDifferenceTracker:
    def __init__(self) -> None:
        self.tracks: dict[str, FiniteDifferenceTrack] = {}
        self.next_id = 1

    def update(self, measurements: Sequence[x24.Measurement], now_s: float) -> list[dict[str, Any]]:
        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if now_s - value.last_seen_s <= TRACK_MAX_AGE_S + EPSILON
        }
        costs: list[tuple[float, float, str, int]] = []
        for track_id, track in self.tracks.items():
            predicted = track.position_xy + track.velocity_xy * max(0.0, now_s - track.last_seen_s)
            for index, measurement in enumerate(measurements):
                if measurement.class_id != track.class_id:
                    continue
                distance = float(np.linalg.norm(predicted - measurement.position_xy))
                if distance <= ASSOCIATION_GATE_M + EPSILON:
                    costs.append(
                        (distance, -x24.bbox_iou(track.last_bbox, measurement.bbox), track_id, index)
                    )
        costs.sort()
        assigned_tracks: set[str] = set()
        assigned_measurements: dict[int, str] = {}
        for _distance, _overlap, track_id, index in costs:
            if track_id not in assigned_tracks and index not in assigned_measurements:
                assigned_tracks.add(track_id)
                assigned_measurements[index] = track_id

        measured_ids: list[str] = []
        for index, measurement in enumerate(measurements):
            track_id = assigned_measurements.get(index)
            if track_id is None:
                track_id = f"fdcv-{self.next_id:06d}"
                self.next_id += 1
                self.tracks[track_id] = FiniteDifferenceTrack(
                    track_id,
                    measurement.class_id,
                    measurement.class_name,
                    measurement.position_xy.copy(),
                    np.zeros(2, dtype=np.float64),
                    now_s,
                    1,
                    measurement.bbox,
                )
            else:
                track = self.tracks[track_id]
                delta_s = now_s - track.last_seen_s
                if delta_s <= EPSILON:
                    raise ValueError("finite_difference_nonpositive_delta")
                track.velocity_xy = (measurement.position_xy - track.position_xy) / delta_s
                track.position_xy = measurement.position_xy.copy()
                track.last_seen_s = now_s
                track.hits += 1
                track.last_bbox = measurement.bbox
            measured_ids.append(track_id)

        # No event or missing-measurement persistence: emit only tracks measured now.
        return [
            {
                "track_id": track_id,
                "class_id": self.tracks[track_id].class_id,
                "class_name": self.tracks[track_id].class_name,
                "position_forward_m": float(self.tracks[track_id].position_xy[0]),
                "position_right_m": float(self.tracks[track_id].position_xy[1]),
                "velocity_forward_mps": float(self.tracks[track_id].velocity_xy[0]),
                "velocity_right_mps": float(self.tracks[track_id].velocity_xy[1]),
                "hits": self.tracks[track_id].hits,
            }
            for track_id in measured_ids
            if self.tracks[track_id].hits >= MINIMUM_TRACK_HITS
        ]


@dataclass
class CausalTurnHistory:
    headings: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    def update(self, tracks: Sequence[Mapping[str, Any]], now_s: float) -> dict[str, float]:
        live = {str(track["track_id"]) for track in tracks}
        self.headings = {key: value for key, value in self.headings.items() if key in live}
        output: dict[str, float] = {}
        for track in tracks:
            track_id = str(track["track_id"])
            speed = math.hypot(
                float(track["velocity_forward_mps"]),
                float(track["velocity_right_mps"]),
            )
            rows = self.headings.setdefault(track_id, [])
            if str(track.get("disposition", "MEASURED")) == "MEASURED" and speed >= MINIMUM_SPEED_FOR_HEADING_MPS:
                heading = math.atan2(
                    float(track["velocity_right_mps"]),
                    float(track["velocity_forward_mps"]),
                )
                rows.append((now_s, heading))
            rows[:] = [row for row in rows if now_s - row[0] <= TURN_HISTORY_S + EPSILON]
            if len(rows) < MINIMUM_TURN_SAMPLES:
                output[track_id] = 0.0
                continue
            unwrapped = np.unwrap(np.asarray([row[1] for row in rows], dtype=np.float64))
            times = np.asarray([row[0] for row in rows], dtype=np.float64)
            centered = times - times.mean()
            denominator = float(centered @ centered)
            yaw_rate = 0.0 if denominator <= EPSILON else float(centered @ (unwrapped - unwrapped.mean()) / denominator)
            output[track_id] = max(
                -MAXIMUM_ABSOLUTE_YAW_RATE_RAD_S,
                min(MAXIMUM_ABSOLUTE_YAW_RATE_RAD_S, yaw_rate),
            )
        return output


def ctrv_position(
    position_xy: Sequence[float],
    velocity_xy: Sequence[float],
    yaw_rate_rad_s: float,
    delta_s: float,
) -> tuple[float, float]:
    x, y = map(float, position_xy)
    vx, vy = map(float, velocity_xy)
    speed = math.hypot(vx, vy)
    if speed <= EPSILON or abs(yaw_rate_rad_s) <= 1e-6:
        return x + vx * delta_s, y + vy * delta_s
    heading = math.atan2(vy, vx)
    next_heading = heading + yaw_rate_rad_s * delta_s
    radius = speed / yaw_rate_rad_s
    return (
        x + radius * (math.sin(next_heading) - math.sin(heading)),
        y - radius * (math.cos(next_heading) - math.cos(heading)),
    )


def _wearer_at(segments: Sequence[route.RouteSegment], offset_s: float) -> tuple[float, float]:
    for segment in segments:
        if segment.start_offset_s - EPSILON <= offset_s <= segment.end_offset_s + EPSILON:
            local = max(0.0, offset_s - segment.start_offset_s)
            return (
                segment.start_position_xy[0] + segment.velocity_xy[0] * local,
                segment.start_position_xy[1] + segment.velocity_xy[1] * local,
            )
    final = segments[-1]
    local = max(0.0, final.end_offset_s - final.start_offset_s)
    return (
        final.start_position_xy[0] + final.velocity_xy[0] * local,
        final.start_position_xy[1] + final.velocity_xy[1] * local,
    )


def first_ctrv_route_entry_s(
    *,
    target_position_xy: Sequence[float],
    target_velocity_xy: Sequence[float],
    yaw_rate_rad_s: float,
    route_segments: Sequence[route.RouteSegment],
) -> float | None:
    horizon = max(segment.end_offset_s for segment in route_segments)
    previous_distance: float | None = None
    steps = int(math.ceil(horizon / CTRV_INTEGRATION_STEP_S))
    for index in range(steps + 1):
        offset = min(horizon, index * CTRV_INTEGRATION_STEP_S)
        target = ctrv_position(target_position_xy, target_velocity_xy, yaw_rate_rad_s, offset)
        wearer = _wearer_at(route_segments, offset)
        distance = math.dist(target, wearer)
        if distance <= route.DEFAULT_TUBE_RADIUS_M + EPSILON:
            if previous_distance is None:
                next_offset = min(horizon, offset + CTRV_INTEGRATION_STEP_S)
                next_target = ctrv_position(
                    target_position_xy,
                    target_velocity_xy,
                    yaw_rate_rad_s,
                    next_offset,
                )
                next_wearer = _wearer_at(route_segments, next_offset)
                closing = (
                    distance - math.dist(next_target, next_wearer)
                ) / max(EPSILON, next_offset - offset)
            else:
                closing = (previous_distance - distance) / CTRV_INTEGRATION_STEP_S
            if closing + EPSILON >= route.DEFAULT_MIN_CLOSING_SPEED_MPS:
                return offset
        previous_distance = distance
    return None


def tiny_feature_vector(
    *,
    radial_entry_s: float | None,
    cv_entry_s: float | None,
    ctrv_entry_s: float | None,
    tracks: Sequence[Mapping[str, Any]],
    current_measurement_count: int,
    issued_plan_mode: bool,
) -> np.ndarray:
    horizon = route.DEFAULT_ROUTE_HORIZON_S + 1.0
    distances = [
        math.hypot(float(row["position_forward_m"]), float(row["position_right_m"]))
        for row in tracks
    ]
    speeds = [
        math.hypot(float(row["velocity_forward_mps"]), float(row["velocity_right_mps"]))
        for row in tracks
    ]
    return np.asarray(
        [
            horizon if radial_entry_s is None else min(horizon, float(radial_entry_s)),
            horizon if cv_entry_s is None else min(horizon, float(cv_entry_s)),
            horizon if ctrv_entry_s is None else min(horizon, float(ctrv_entry_s)),
            min(20.0, min(distances, default=20.0)),
            min(10.0, max(speeds, default=0.0)),
            min(10.0, float(len(tracks))),
            min(1.0, float(current_measurement_count) / max(1, len(tracks))),
            float(bool(issued_plan_mode)),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class TinyLogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    def probability(self, features: Sequence[float]) -> float:
        vector = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        score = float(vector @ self.weights + self.bias)
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, score))))

    def route_risk(self, features: Sequence[float]) -> bool:
        return self.probability(features) >= TINY_THRESHOLD

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": list(TINY_FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": TINY_THRESHOLD,
            "fit_constants": {
                "l2": TINY_L2,
                "learning_rate": TINY_LEARNING_RATE,
                "steps": TINY_STEPS,
            },
        }


def fit_tiny_logistic(features: Sequence[Sequence[float]], labels: Sequence[bool]) -> TinyLogisticModel:
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(TINY_FEATURE_NAMES):
        raise ValueError("tiny_feature_shape")
    if len(matrix) != len(target) or len(matrix) == 0:
        raise ValueError("tiny_label_shape")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("tiny_nonfinite")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < EPSILON] = 1.0
    normalized = (matrix - mean) / scale
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(TINY_STEPS):
        logits = np.clip(normalized @ weights + bias, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        error = probability - target
        weights -= TINY_LEARNING_RATE * (
            normalized.T @ error / len(matrix) + TINY_L2 * weights
        )
        bias -= TINY_LEARNING_RATE * float(error.mean())
    return TinyLogisticModel(mean, scale, weights, bias)


__all__ = [
    "ARM_CAUSAL_CTRV",
    "ARM_FINITE_DIFFERENCE_CV",
    "ARM_TINY_LEARNED",
    "CausalFiniteDifferenceTracker",
    "CausalTurnHistory",
    "TinyLogisticModel",
    "ctrv_position",
    "first_ctrv_route_entry_s",
    "fit_tiny_logistic",
    "fixed_constants",
    "tiny_feature_vector",
]
