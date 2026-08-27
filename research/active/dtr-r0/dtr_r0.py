"""Dependency-free mechanics for the DTR-R0 dynamic route-risk experiment.

Coordinates
-----------
All motion is represented on a two-dimensional ground plane.  World ``x/y``
are arbitrary metric axes.  A yaw of zero points along world ``+x`` and
positive yaw rotates counter-clockwise toward world ``+y``.  A detector
observation is expressed in the sensor frame as ``forward_m/left_m``.

``EgoPose.body_yaw_rad`` defines the wearer's route direction while
``sensor_yaw_rad`` defines the camera direction.  Keeping them separate is
what makes a head turn a camera motion rather than target motion.  Transforming
each causal observation to world coordinates before fitting velocity also
compensates ego translation and rotation.

This module intentionally contains no dataset-specific truth access.  An arm
sees only the frames passed to ``step`` up to the current timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import statistics
from typing import Any, Iterable, Mapping, Optional, Sequence


EPSILON = 1e-9


class Arm(str, Enum):
    """The three matched baselines and the route-intersection challenger."""

    B0_DETECTION = "B0_detection_reminder"
    B1_DISTANCE = "B1_distance_gate"
    B2_RADIAL_TTC = "B2_radial_ttc"
    C_ROUTE_INTERSECTION = "C_route_intersection"


class Signal(str, Enum):
    """Shared lifecycle outputs.

    UNKNOWN is an availability result, not a negative prediction.  In
    particular it never clears a previously active event.
    """

    ONSET = "ONSET"
    HOLD = "HOLD"
    CLEAR = "CLEAR"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def norm(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass(frozen=True)
class EgoPose:
    x_m: float
    y_m: float
    body_yaw_rad: float
    sensor_yaw_rad: float

    @property
    def position(self) -> Vec2:
        return Vec2(self.x_m, self.y_m)

    @property
    def route_unit(self) -> Vec2:
        return Vec2(math.cos(self.body_yaw_rad), math.sin(self.body_yaw_rad))

    def local_to_world(self, forward_m: float, left_m: float) -> Vec2:
        cosine = math.cos(self.sensor_yaw_rad)
        sine = math.sin(self.sensor_yaw_rad)
        return Vec2(
            self.x_m + forward_m * cosine - left_m * sine,
            self.y_m + forward_m * sine + left_m * cosine,
        )

    def world_to_local(self, point: Vec2) -> tuple[float, float]:
        delta = point - self.position
        cosine = math.cos(self.sensor_yaw_rad)
        sine = math.sin(self.sensor_yaw_rad)
        return (
            delta.x * cosine + delta.y * sine,
            -delta.x * sine + delta.y * cosine,
        )


@dataclass(frozen=True)
class Observation:
    track_id: str
    forward_m: float
    left_m: float
    radius_m: float = 0.30


@dataclass(frozen=True)
class CausalFrame:
    time_s: float
    ego_pose: Optional[EgoPose]
    observations: tuple[Observation, ...] = ()
    # None means detector availability is unknown. A real-input adapter sets an
    # explicit count so B0 remains evaluable even when metric projection fails.
    person_detection_count: Optional[int] = None


@dataclass(frozen=True)
class WorldObservation:
    time_s: float
    track_id: str
    position: Vec2
    radius_m: float


@dataclass(frozen=True)
class TrackEstimate:
    track_id: str
    time_s: float
    position: Vec2
    velocity_mps: Vec2
    radius_m: float
    sample_count: int


@dataclass(frozen=True)
class DTRConfig:
    distance_gate_m: float = 3.0
    radial_ttc_max_s: float = 3.0
    minimum_closing_speed_mps: float = 0.10
    track_window_s: float = 1.50
    minimum_track_span_s: float = 0.20
    route_horizon_s: float = 3.00
    route_half_width_m: float = 0.65
    nominal_wearer_speed_mps: float = 1.00
    clear_grace_s: float = 0.50

    def __post_init__(self) -> None:
        if not 1.50 <= self.route_horizon_s <= 3.00:
            raise ValueError("route horizon must remain inside 1.5--3.0 seconds")
        if self.track_window_s <= 0.0 or self.minimum_track_span_s <= 0.0:
            raise ValueError("track window and minimum span must be positive")
        if self.route_half_width_m <= 0.0:
            raise ValueError("route half width must be positive")


@dataclass(frozen=True)
class Prediction:
    time_s: float
    signal: Signal
    raw_alert: Optional[bool]
    reason: str
    track_id: Optional[str] = None
    diagnostic: Mapping[str, float | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "signal": self.signal.value,
            "raw_alert": self.raw_alert,
            "reason": self.reason,
            "track_id": self.track_id,
            "diagnostic": dict(self.diagnostic),
        }


class EventLifecycle:
    """Map a ternary alert decision onto the shared event lifecycle.

    A known negative must remain continuous for ``clear_grace_s`` before an
    active event clears.  UNKNOWN is not negative evidence: it preserves the
    active state and restarts the grace observation window.
    """

    def __init__(self, clear_grace_s: float) -> None:
        if clear_grace_s < 0.0 or not math.isfinite(clear_grace_s):
            raise ValueError("clear grace must be finite and non-negative")
        self.clear_grace_s = clear_grace_s
        self.active = False
        self._clear_candidate_since_s: Optional[float] = None

    def update(self, time_s: float, raw_alert: Optional[bool]) -> Signal:
        if raw_alert is None:
            self._clear_candidate_since_s = None
            return Signal.UNKNOWN
        if raw_alert:
            self._clear_candidate_since_s = None
            if self.active:
                return Signal.HOLD
            self.active = True
            return Signal.ONSET
        if not self.active:
            self._clear_candidate_since_s = None
            return Signal.CLEAR
        if self._clear_candidate_since_s is None:
            self._clear_candidate_since_s = time_s
        if time_s - self._clear_candidate_since_s + EPSILON < self.clear_grace_s:
            return Signal.HOLD
        self.active = False
        self._clear_candidate_since_s = None
        return Signal.CLEAR


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def estimate_constant_velocity(
    observations: Sequence[WorldObservation],
    *,
    minimum_span_s: float = 0.20,
) -> Optional[TrackEstimate]:
    """Fit ``position = intercept + velocity * time`` by least squares.

    Only the supplied causal samples are used.  Returning ``None`` for a short
    or degenerate track is deliberate: unavailable motion is UNKNOWN, not a
    zero-velocity assumption.
    """

    if len(observations) < 2:
        return None
    ordered = sorted(observations, key=lambda item: item.time_s)
    if ordered[-1].time_s - ordered[0].time_s + EPSILON < minimum_span_s:
        return None
    if any(item.track_id != ordered[-1].track_id for item in ordered):
        raise ValueError("constant-velocity fit received mixed track ids")

    mean_t = sum(item.time_s for item in ordered) / len(ordered)
    denominator = sum((item.time_s - mean_t) ** 2 for item in ordered)
    if denominator <= EPSILON:
        return None
    mean_x = sum(item.position.x for item in ordered) / len(ordered)
    mean_y = sum(item.position.y for item in ordered) / len(ordered)
    vx = sum(
        (item.time_s - mean_t) * (item.position.x - mean_x) for item in ordered
    ) / denominator
    vy = sum(
        (item.time_s - mean_t) * (item.position.y - mean_y) for item in ordered
    ) / denominator
    current_t = ordered[-1].time_s
    intercept_x = mean_x - vx * mean_t
    intercept_y = mean_y - vy * mean_t
    return TrackEstimate(
        track_id=ordered[-1].track_id,
        time_s=current_t,
        position=Vec2(intercept_x + vx * current_t, intercept_y + vy * current_t),
        velocity_mps=Vec2(vx, vy),
        radius_m=ordered[-1].radius_m,
        sample_count=len(ordered),
    )


def _minimum_separation(
    relative_position: Vec2,
    relative_velocity: Vec2,
    start_s: float,
    end_s: float,
) -> tuple[float, float]:
    speed_squared = relative_velocity.dot(relative_velocity)
    if speed_squared <= EPSILON:
        best_t = start_s
    else:
        unconstrained = -relative_position.dot(relative_velocity) / speed_squared
        best_t = min(end_s, max(start_s, unconstrained))
    distance = (relative_position + relative_velocity * best_t).norm()
    return distance, best_t


class DTRR0Arm:
    """Stateful causal runner for one DTR-R0 arm."""

    def __init__(self, arm: Arm, config: Optional[DTRConfig] = None) -> None:
        self.arm = arm
        self.config = config or DTRConfig()
        self.lifecycle = EventLifecycle(self.config.clear_grace_s)
        self._tracks: dict[str, list[WorldObservation]] = {}
        self._last_time_s: Optional[float] = None

    def step(self, frame: CausalFrame) -> Prediction:
        self._validate_frame(frame)
        self._last_time_s = frame.time_s

        if self.arm is Arm.B0_DETECTION:
            detection_count = frame.person_detection_count
            if detection_count is None:
                detection_count = len(frame.observations) if frame.observations else None
            if detection_count is None:
                return self._prediction(
                    frame.time_s, None, "person_detector_unavailable"
                )
            first = frame.observations[0] if frame.observations else None
            return self._prediction(
                frame.time_s,
                detection_count > 0,
                (
                    "current_person_detection_present"
                    if detection_count > 0
                    else "current_person_detection_absent"
                ),
                first.track_id if first is not None else None,
                {"person_detection_count": float(detection_count)},
            )
        if not frame.observations:
            return self._prediction(frame.time_s, None, "missing_current_metric_track")
        if self.arm is Arm.B1_DISTANCE:
            nearest = min(
                frame.observations,
                key=lambda item: math.hypot(item.forward_m, item.left_m),
            )
            distance = math.hypot(nearest.forward_m, nearest.left_m)
            return self._prediction(
                frame.time_s,
                distance <= self.config.distance_gate_m,
                "distance_gate_evaluated",
                nearest.track_id,
                {"distance_m": distance},
            )

        if frame.ego_pose is None:
            return self._prediction(frame.time_s, None, "missing_ego_pose")
        world_observations = self._append_world_observations(frame)
        estimates = self._current_estimates(world_observations)
        if not estimates:
            return self._prediction(frame.time_s, None, "insufficient_causal_track")
        if self.arm is Arm.B2_RADIAL_TTC:
            return self._radial_ttc(frame, estimates)
        if self.arm is Arm.C_ROUTE_INTERSECTION:
            return self._route_intersection(frame, estimates)
        raise AssertionError(f"unsupported arm: {self.arm}")

    def _validate_frame(self, frame: CausalFrame) -> None:
        if not math.isfinite(frame.time_s):
            raise ValueError("frame time must be finite")
        if self._last_time_s is not None and frame.time_s <= self._last_time_s:
            raise ValueError("frame times must be strictly increasing")
        if frame.ego_pose is not None and not _finite(
            frame.ego_pose.x_m,
            frame.ego_pose.y_m,
            frame.ego_pose.body_yaw_rad,
            frame.ego_pose.sensor_yaw_rad,
        ):
            raise ValueError("ego pose must contain finite values")
        seen: set[str] = set()
        if frame.person_detection_count is not None:
            if frame.person_detection_count < 0:
                raise ValueError("person detection count must be non-negative")
            if frame.person_detection_count < len(frame.observations):
                raise ValueError(
                    "person detection count cannot be smaller than metric observations"
                )
        for observation in frame.observations:
            if not observation.track_id or observation.track_id in seen:
                raise ValueError("track ids must be non-empty and unique per frame")
            seen.add(observation.track_id)
            if not _finite(
                observation.forward_m,
                observation.left_m,
                observation.radius_m,
            ) or observation.radius_m < 0.0:
                raise ValueError("observations must contain finite metric values")

    def _append_world_observations(
        self, frame: CausalFrame
    ) -> list[WorldObservation]:
        assert frame.ego_pose is not None
        current: list[WorldObservation] = []
        cutoff = frame.time_s - self.config.track_window_s
        for observation in frame.observations:
            world = WorldObservation(
                time_s=frame.time_s,
                track_id=observation.track_id,
                position=frame.ego_pose.local_to_world(
                    observation.forward_m, observation.left_m
                ),
                radius_m=observation.radius_m,
            )
            history = self._tracks.setdefault(observation.track_id, [])
            history.append(world)
            history[:] = [sample for sample in history if sample.time_s >= cutoff]
            current.append(world)
        return current

    def _current_estimates(
        self, current: Sequence[WorldObservation]
    ) -> list[TrackEstimate]:
        estimates: list[TrackEstimate] = []
        for observation in current:
            estimate = estimate_constant_velocity(
                self._tracks[observation.track_id],
                minimum_span_s=self.config.minimum_track_span_s,
            )
            if estimate is not None:
                estimates.append(estimate)
        return estimates

    def _wearer_velocity(self, pose: EgoPose) -> Vec2:
        return pose.route_unit * self.config.nominal_wearer_speed_mps

    def _radial_ttc(
        self, frame: CausalFrame, estimates: Sequence[TrackEstimate]
    ) -> Prediction:
        assert frame.ego_pose is not None
        wearer_velocity = self._wearer_velocity(frame.ego_pose)
        evaluated: list[tuple[TrackEstimate, float, float]] = []
        for estimate in estimates:
            relative_position = estimate.position - frame.ego_pose.position
            distance = relative_position.norm()
            if distance <= EPSILON:
                evaluated.append((estimate, 0.0, 0.0))
                continue
            relative_velocity = estimate.velocity_mps - wearer_velocity
            radial_speed = relative_position.dot(relative_velocity) / distance
            if radial_speed < -self.config.minimum_closing_speed_mps:
                ttc = -distance / radial_speed
                evaluated.append((estimate, ttc, radial_speed))
        if not evaluated:
            return self._prediction(
                frame.time_s,
                False,
                "no_radially_closing_track",
            )
        best = min(evaluated, key=lambda item: item[1])
        return self._prediction(
            frame.time_s,
            0.0 <= best[1] <= self.config.radial_ttc_max_s,
            "radial_ttc_evaluated",
            best[0].track_id,
            {"ttc_s": best[1], "radial_speed_mps": best[2]},
        )

    def _route_intersection(
        self, frame: CausalFrame, estimates: Sequence[TrackEstimate]
    ) -> Prediction:
        assert frame.ego_pose is not None
        wearer_velocity = self._wearer_velocity(frame.ego_pose)
        candidates: list[tuple[TrackEstimate, float, float, float]] = []
        for estimate in estimates:
            distance, future_s = _minimum_separation(
                estimate.position - frame.ego_pose.position,
                estimate.velocity_mps - wearer_velocity,
                0.0,
                self.config.route_horizon_s,
            )
            threshold = self.config.route_half_width_m + estimate.radius_m
            candidates.append((estimate, distance, future_s, threshold))
        best = min(candidates, key=lambda item: item[1])
        return self._prediction(
            frame.time_s,
            best[1] <= best[3],
            "route_tube_intersection_evaluated",
            best[0].track_id,
            {
                "minimum_separation_m": best[1],
                "intersection_threshold_m": best[3],
                "future_s": best[2],
                "horizon_start_s": 0.0,
                "horizon_end_s": self.config.route_horizon_s,
            },
        )

    def _prediction(
        self,
        time_s: float,
        raw_alert: Optional[bool],
        reason: str,
        track_id: Optional[str] = None,
        diagnostic: Optional[Mapping[str, float | str]] = None,
    ) -> Prediction:
        return Prediction(
            time_s=time_s,
            signal=self.lifecycle.update(time_s, raw_alert),
            raw_alert=raw_alert,
            reason=reason,
            track_id=track_id,
            diagnostic=diagnostic or {},
        )


def run_arm(
    frames: Iterable[CausalFrame],
    arm: Arm,
    config: Optional[DTRConfig] = None,
) -> list[Prediction]:
    runner = DTRR0Arm(arm, config)
    return [runner.step(frame) for frame in frames]


def pose_from_dict(value: Optional[Mapping[str, Any]]) -> Optional[EgoPose]:
    if value is None:
        return None
    return EgoPose(
        x_m=float(value["x_m"]),
        y_m=float(value["y_m"]),
        body_yaw_rad=float(value["body_yaw_rad"]),
        sensor_yaw_rad=float(value["sensor_yaw_rad"]),
    )


def frame_from_dict(value: Mapping[str, Any]) -> CausalFrame:
    detection_count = value.get("person_detection_count")
    return CausalFrame(
        time_s=float(value["time_s"]),
        ego_pose=pose_from_dict(value.get("ego_pose")),
        observations=tuple(
            Observation(
                track_id=str(item["track_id"]),
                forward_m=float(item["forward_m"]),
                left_m=float(item["left_m"]),
                radius_m=float(item.get("radius_m", 0.30)),
            )
            for item in value.get("observations", [])
        ),
        person_detection_count=(
            int(detection_count) if detection_count is not None else None
        ),
    )


def _ratio(numerator: int | float, denominator: int | float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_event_metrics(
    episodes: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[Prediction]],
    *,
    clear_grace_s: float = 0.50,
) -> dict[str, Any]:
    """Compute event-level metrics without converting UNKNOWN to negative.

    Truth is consumed only here, after predictions already exist.  A critical
    event is recalled when at least one ONSET/HOLD falls between the start of
    its complete-path warning interval and the end of the event. Segment counts
    are ONSET counts, not alert-frame counts.
    """

    critical_count = 0
    recalled_count = 0
    noncritical_count = 0
    irrelevant_segments = 0
    critical_segments = 0
    leads: list[float] = []
    unknown_frames = 0
    total_frames = 0
    stable_clear_eligible = 0
    stable_clear_success = 0
    clear_delays: list[float] = []
    total_observation_s = 0.0

    for episode in episodes:
        episode_id = str(episode["episode_id"])
        episode_predictions = list(predictions[episode_id])
        if len(episode_predictions) != len(episode["frames"]):
            raise ValueError(f"prediction/frame mismatch for {episode_id}")
        truth = episode["truth"]
        frame_times = [float(frame["time_s"]) for frame in episode["frames"]]
        if frame_times:
            total_observation_s += max(frame_times) - min(frame_times)
        critical = bool(truth["critical_event"])
        onsets = [item for item in episode_predictions if item.signal is Signal.ONSET]
        unknown_frames += sum(
            item.signal is Signal.UNKNOWN for item in episode_predictions
        )
        total_frames += len(episode_predictions)

        if critical:
            critical_count += 1
            warning_start = float(truth["warning_start_s"])
            event_end = float(truth["event_end_s"])
            hits = [
                item
                for item in episode_predictions
                if warning_start <= item.time_s <= event_end
                and item.signal in (Signal.ONSET, Signal.HOLD)
            ]
            if hits:
                recalled_count += 1
                leads.append(float(truth["event_start_s"]) - hits[0].time_s)
            critical_segments += len(onsets)
            irrelevant_segments += sum(
                not (warning_start <= item.time_s <= event_end) for item in onsets
            )

            exit_time = truth.get("exit_time_s")
            if exit_time is not None:
                first_clear = next(
                    (
                        item
                        for item in episode_predictions
                        if item.time_s >= float(exit_time)
                        and item.signal is Signal.CLEAR
                    ),
                    None,
                )
                if first_clear is not None:
                    clear_delays.append(first_clear.time_s - float(exit_time))
                post_exit = [
                    item
                    for item in episode_predictions
                    if item.time_s >= float(exit_time) + clear_grace_s
                ]
                if post_exit:
                    stable_clear_eligible += 1
                    if all(item.signal is Signal.CLEAR for item in post_exit):
                        stable_clear_success += 1
        else:
            noncritical_count += 1
            irrelevant_segments += len(onsets)

    median_lead = statistics.median(leads) if leads else None
    total_observation_minutes = total_observation_s / 60.0
    return {
        "episode_count": len(episodes),
        "critical_event_count": critical_count,
        "noncritical_episode_count": noncritical_count,
        "critical_event_recalled_count": recalled_count,
        "critical_event_recall": _ratio(recalled_count, critical_count),
        "irrelevant_alert_segments": irrelevant_segments,
        "irrelevant_alert_segments_per_noncritical_episode": _ratio(
            irrelevant_segments, noncritical_count
        ),
        "total_observation_minutes": total_observation_minutes,
        "false_alerts_per_minute": _ratio(
            irrelevant_segments, total_observation_minutes
        ),
        "median_first_alert_lead_s": median_lead,
        "alert_segments_on_critical_events": critical_segments,
        "mean_alert_segments_per_critical_event": _ratio(
            critical_segments, critical_count
        ),
        "stable_clear_eligible_events": stable_clear_eligible,
        "stable_clear_success_events": stable_clear_success,
        "stable_clear_rate": _ratio(stable_clear_success, stable_clear_eligible),
        "median_clear_delay_s": (
            statistics.median(clear_delays) if clear_delays else None
        ),
        "unknown_frames": unknown_frames,
        "total_frames": total_frames,
        "known_frame_coverage": _ratio(total_frames - unknown_frames, total_frames),
    }
