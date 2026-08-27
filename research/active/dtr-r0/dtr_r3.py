"""Causal DTR-R3 route/model ablations.

R3 compares three deliberately fixed alternatives to R1/R2:

``A``
    A causal constant-turn-rate-and-velocity (CTRV) wearer route plus the
    component-wise median of all causal pairwise target velocities.
``B``
    A straight wearer route plus a distributional decision over every causal
    pairwise target-velocity hypothesis.
``C``
    The curved CTRV route plus the same distributional decision, fused with
    R2's existing imminent R0 route-intersection guard.

The distributional decision is deliberately label-free and fixed: strictly
more than half of the velocity hypotheses must enter the route tube.  A tie is
not a majority.  ``entry_support`` and ``decision_score`` are ordinal
diagnostics, not calibrated probabilities.

The A/B/C arms are a coupled performance comparison, not a factorial ablation:
A changes route and target summarization relative to B, while C changes route
and adds the imminent guard relative to B.  Their differences cannot identify
the causal contribution of one component in isolation.

Only current and past inputs are accepted.  ``CausalFrame.ego_pose`` supplies
the current pose; callers replaying a chunk may additionally pass causal
``TimedEgoPose`` history.  For sources whose observations are already in a
fixed-world ego-relative frame (for example a JRDB diagnostic), callers can
pass an identity pose on every frame.  R3-C optionally accepts a separate
``guard_frame`` so the unchanged R0/R2 guard can continue to consume a legacy
relative-frame adapter while the distributional branch consumes true poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import statistics
from typing import Iterable, Mapping, Sequence

from dtr_r0 import (
    Arm,
    CausalFrame,
    DTRConfig,
    DTRR0Arm,
    EgoPose,
    Prediction,
    Vec2,
)
from dtr_r1 import FROZEN_R1_CONFIG, R1Config, RiskEventLifecycle
from dtr_r2 import FROZEN_R2_CONFIG, R2Config


EPSILON = 1e-9
FIXED_SUPPORT_THRESHOLD = 0.50


class R3Arm(str, Enum):
    """The three matched R3 ablations."""

    A_CURVED_ROBUST_CV = "A_curved_ctrv_robust_target_cv"
    B_STRAIGHT_DISTRIBUTIONAL = "B_straight_distributional_occupancy"
    C_CURVED_DISTRIBUTIONAL_GUARDED = (
        "C_curved_distributional_with_r2_imminent_guard"
    )


@dataclass(frozen=True)
class R3Config:
    """R3-only mechanics; R1 owns target history and route-tube constants."""

    ego_history_window_s: float = 0.50
    minimum_ego_span_s: float = 0.20
    route_sample_step_s: float = 0.10

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                self.ego_history_window_s,
                self.minimum_ego_span_s,
                self.route_sample_step_s,
            )
        ):
            raise ValueError("R3 history spans and route step must be finite and positive")
        if self.minimum_ego_span_s > self.ego_history_window_s + EPSILON:
            raise ValueError("minimum ego span cannot exceed the ego history window")

    def to_dict(self) -> dict[str, float | str]:
        values: dict[str, float | str] = {
            "ego_history_window_s": self.ego_history_window_s,
            "minimum_ego_span_s": self.minimum_ego_span_s,
            "route_sample_step_s": self.route_sample_step_s,
            "wearer_motion": "causal_constant_turn_rate_and_forward_velocity",
            "target_motion": "all_causal_pairwise_constant_velocity_hypotheses",
            "support_decision": "strict_majority_entry_support_gt_0.5",
            "support_threshold": FIXED_SUPPORT_THRESHOLD,
        }
        canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
        values["fingerprint_sha256"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        return values


FROZEN_R3_CONFIG = R3Config()


@dataclass(frozen=True)
class TimedEgoPose:
    """One caller-owned current-or-past pose sample."""

    time_s: float
    pose: EgoPose


@dataclass(frozen=True)
class WorldTargetObservation:
    time_s: float
    track_id: str
    position: Vec2
    radius_m: float


@dataclass(frozen=True)
class EgoMotionEstimate:
    forward_speed_mps: float
    yaw_rate_radps: float
    sample_count: int
    span_s: float


@dataclass(frozen=True)
class RouteSample:
    future_s: float
    position: Vec2


@dataclass(frozen=True)
class RouteSegment:
    start_s: float
    span_s: float
    start_position: Vec2
    velocity: Vec2


@dataclass(frozen=True)
class TrajectoryEvidence:
    entry_time_s: float | None
    minimum_separation_m: float
    minimum_separation_time_s: float


@dataclass(frozen=True)
class TrackDecision:
    track_id: str
    raw_alert: bool
    urgent: bool
    decision_score: float
    entry_support: float
    entry_time_s: float | None
    minimum_separation_m: float
    minimum_separation_time_s: float
    velocity_hypotheses: int
    entering_hypotheses: int
    robust_velocity_mps: Vec2


def _finite_pose(pose: EgoPose) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            pose.x_m,
            pose.y_m,
            pose.body_yaw_rad,
            pose.sensor_yaw_rad,
        )
    )


def _same_pose(left: EgoPose, right: EgoPose) -> bool:
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=EPSILON)
        for a, b in (
            (left.x_m, right.x_m),
            (left.y_m, right.y_m),
            (left.body_yaw_rad, right.body_yaw_rad),
            (left.sensor_yaw_rad, right.sensor_yaw_rad),
        )
    )


def _fitted_slope(times: Sequence[float], values: Sequence[float]) -> float | None:
    if len(times) < 2:
        return None
    mean_time = sum(times) / len(times)
    denominator = sum((time_s - mean_time) ** 2 for time_s in times)
    if denominator <= EPSILON:
        return None
    mean_value = sum(values) / len(values)
    return sum(
        (time_s - mean_time) * (value - mean_value)
        for time_s, value in zip(times, values)
    ) / denominator


def _unwrapped_body_yaws(samples: Sequence[TimedEgoPose]) -> list[float]:
    output = [samples[0].pose.body_yaw_rad]
    for sample in samples[1:]:
        raw_yaw = sample.pose.body_yaw_rad
        delta = (raw_yaw - output[-1] + math.pi) % (2.0 * math.pi) - math.pi
        output.append(output[-1] + delta)
    return output


def _estimate_ego_motion(
    history: Sequence[TimedEgoPose],
    minimum_span_s: float,
) -> EgoMotionEstimate | None:
    if len(history) < 2:
        return None
    ordered = sorted(history, key=lambda item: item.time_s)
    span_s = ordered[-1].time_s - ordered[0].time_s
    if span_s + EPSILON < minimum_span_s:
        return None
    times = [item.time_s for item in ordered]
    velocity_x = _fitted_slope(times, [item.pose.x_m for item in ordered])
    velocity_y = _fitted_slope(times, [item.pose.y_m for item in ordered])
    yaw_rate = _fitted_slope(times, _unwrapped_body_yaws(ordered))
    if velocity_x is None or velocity_y is None or yaw_rate is None:
        return None
    current_yaw = ordered[-1].pose.body_yaw_rad
    forward_speed = max(
        0.0,
        velocity_x * math.cos(current_yaw) + velocity_y * math.sin(current_yaw),
    )
    return EgoMotionEstimate(
        forward_speed_mps=forward_speed,
        yaw_rate_radps=yaw_rate,
        sample_count=len(ordered),
        span_s=span_s,
    )


def _route_samples(
    current_pose: EgoPose,
    motion: EgoMotionEstimate,
    *,
    curved: bool,
    horizon_s: float,
    step_s: float,
) -> list[RouteSample]:
    yaw_rate = motion.yaw_rate_radps if curved else 0.0
    speed = motion.forward_speed_mps
    yaw = current_pose.body_yaw_rad
    if speed <= EPSILON or abs(yaw_rate) <= 1e-6:
        return [
            RouteSample(0.0, current_pose.position),
            RouteSample(
                horizon_s,
                Vec2(
                    current_pose.x_m + speed * horizon_s * math.cos(yaw),
                    current_pose.y_m + speed * horizon_s * math.sin(yaw),
                ),
            ),
        ]

    sample_times = [0.0]
    next_time = step_s
    while next_time < horizon_s - EPSILON:
        sample_times.append(next_time)
        next_time += step_s
    sample_times.append(horizon_s)

    samples: list[RouteSample] = []
    for future_s in sample_times:
        if abs(yaw_rate) <= 1e-6:
            position = Vec2(
                current_pose.x_m + speed * future_s * math.cos(yaw),
                current_pose.y_m + speed * future_s * math.sin(yaw),
            )
        else:
            future_yaw = yaw + yaw_rate * future_s
            position = Vec2(
                current_pose.x_m
                + speed / yaw_rate * (math.sin(future_yaw) - math.sin(yaw)),
                current_pose.y_m
                - speed / yaw_rate * (math.cos(future_yaw) - math.cos(yaw)),
            )
        samples.append(RouteSample(future_s=future_s, position=position))
    return samples


def _route_segments(route: Sequence[RouteSample]) -> tuple[RouteSegment, ...]:
    return tuple(
        RouteSegment(
            start_s=left.future_s,
            span_s=right.future_s - left.future_s,
            start_position=left.position,
            velocity=(right.position - left.position)
            * (1.0 / (right.future_s - left.future_s)),
        )
        for left, right in zip(route, route[1:])
    )


def _pairwise_velocities(
    history: Sequence[WorldTargetObservation],
    minimum_span_s: float,
) -> list[Vec2]:
    ordered = sorted(history, key=lambda item: item.time_s)
    hypotheses: list[Vec2] = []
    for left_index, left in enumerate(ordered[:-1]):
        for right in ordered[left_index + 1 :]:
            span_s = right.time_s - left.time_s
            if span_s + EPSILON < minimum_span_s:
                continue
            hypotheses.append((right.position - left.position) * (1.0 / span_s))
    return hypotheses


def _linear_entry_s(
    position: Vec2,
    velocity: Vec2,
    radius_m: float,
    horizon_s: float,
    minimum_closing_speed_mps: float,
) -> float | None:
    """R1's entry rule, applied to one locally linear route segment."""

    distance = position.norm()
    speed_squared = velocity.dot(velocity)
    if distance <= radius_m + EPSILON:
        if distance <= EPSILON:
            closing_speed = velocity.norm()
        else:
            closing_speed = -position.dot(velocity) / distance
        return 0.0 if closing_speed + EPSILON >= minimum_closing_speed_mps else None
    if speed_squared <= EPSILON:
        return None
    b = 2.0 * position.dot(velocity)
    c = position.dot(position) - radius_m * radius_m
    discriminant = b * b - 4.0 * speed_squared * c
    if discriminant < 0.0:
        return None
    root = (-b - math.sqrt(max(0.0, discriminant))) / (2.0 * speed_squared)
    if root < -EPSILON or root > horizon_s + EPSILON:
        return None
    entry_s = max(0.0, root)
    entry_position = position + velocity * entry_s
    entry_distance = max(EPSILON, entry_position.norm())
    inward_speed = -entry_position.dot(velocity) / entry_distance
    if inward_speed + EPSILON < minimum_closing_speed_mps:
        return None
    return entry_s


def _trajectory_evidence(
    target_position: Vec2,
    target_velocity: Vec2,
    route: Sequence[RouteSegment],
    tube_radius_m: float,
    minimum_closing_speed_mps: float,
) -> TrajectoryEvidence:
    """Evaluate target CV against a piecewise-linear CTRV route approximation."""

    first_entry_s: float | None = None
    minimum_separation_m = math.inf
    minimum_separation_time_s = 0.0
    for segment in route:
        span_s = segment.span_s
        relative_start = (
            target_position
            + target_velocity * segment.start_s
            - segment.start_position
        )
        relative_velocity = target_velocity - segment.velocity

        speed_squared = relative_velocity.dot(relative_velocity)
        if speed_squared <= EPSILON:
            closest_local_s = 0.0
        else:
            closest_local_s = min(
                span_s,
                max(
                    0.0,
                    -relative_start.dot(relative_velocity) / speed_squared,
                ),
            )
        separation = (
            relative_start + relative_velocity * closest_local_s
        ).norm()
        if separation < minimum_separation_m:
            minimum_separation_m = separation
            minimum_separation_time_s = segment.start_s + closest_local_s

        if first_entry_s is None:
            local_entry_s = _linear_entry_s(
                relative_start,
                relative_velocity,
                tube_radius_m,
                span_s,
                minimum_closing_speed_mps,
            )
            if local_entry_s is not None:
                first_entry_s = segment.start_s + local_entry_s

    return TrajectoryEvidence(
        entry_time_s=first_entry_s,
        minimum_separation_m=minimum_separation_m,
        minimum_separation_time_s=minimum_separation_time_s,
    )


def _ordinal_decision_score(
    raw_alert: bool,
    entry_support: float,
    entry_time_s: float | None,
    horizon_s: float,
) -> float:
    if not raw_alert or entry_time_s is None:
        return 0.0
    urgency = 1.0 - min(1.0, max(0.0, entry_time_s / horizon_s))
    return entry_support * (0.5 + 0.5 * urgency)


class DTRR3Arm:
    """Stateful causal runner for one R3 ablation."""

    def __init__(
        self,
        arm: R3Arm | str,
        r1_config: R1Config = FROZEN_R1_CONFIG,
        r3_config: R3Config = FROZEN_R3_CONFIG,
        r0_config: DTRConfig | None = None,
        r2_config: R2Config = FROZEN_R2_CONFIG,
    ) -> None:
        self.arm = R3Arm(arm)
        self.r1_config = r1_config
        self.r3_config = r3_config
        self.r0_config = r0_config or DTRConfig()
        self.r2_config = r2_config
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                r1_config.history_window_s,
                r1_config.minimum_pair_span_s,
                r1_config.route_horizon_s,
                r1_config.route_half_width_m,
            )
        ):
            raise ValueError("R1 history and route constants must be finite and positive")
        if r1_config.minimum_pair_span_s > r1_config.history_window_s + EPSILON:
            raise ValueError("minimum target span cannot exceed target history window")
        if (
            not math.isfinite(r1_config.minimum_closing_speed_mps)
            or r1_config.minimum_closing_speed_mps < 0.0
        ):
            raise ValueError("minimum closing speed must be finite and non-negative")
        if not 0.0 < r2_config.imminent_horizon_fraction <= 1.0:
            raise ValueError("imminent horizon fraction must be in (0, 1]")

        self.lifecycle = RiskEventLifecycle(r1_config.clear_grace_s)
        self._ego_history: list[TimedEgoPose] = []
        self._tracks: dict[str, list[WorldTargetObservation]] = {}
        self._last_time_s: float | None = None
        self._guard = (
            DTRR0Arm(Arm.C_ROUTE_INTERSECTION, self.r0_config)
            if self.arm is R3Arm.C_CURVED_DISTRIBUTIONAL_GUARDED
            else None
        )

    def step(
        self,
        frame: CausalFrame,
        guard_frame: CausalFrame | None = None,
        *,
        ego_pose_history: Sequence[
            TimedEgoPose | tuple[float, EgoPose]
        ] = (),
    ) -> Prediction:
        """Consume one causal frame and optionally the R2 guard's legacy frame.

        ``ego_pose_history`` may contain current and/or past samples, but any
        future timestamp is rejected.  Normally sequential calls need only put
        the current pose in ``frame`` because the runner retains the last 0.5 s.
        """

        self._validate_frame(frame)
        if guard_frame is not None and not math.isclose(
            guard_frame.time_s, frame.time_s, rel_tol=0.0, abs_tol=EPSILON
        ):
            raise ValueError("guard frame time must equal the R3 frame time")

        supplied_history = self._coerce_ego_history(
            ego_pose_history, current_time_s=frame.time_s
        )
        effective_pose = self._resolve_current_pose(frame, supplied_history)
        effective_frame = (
            frame
            if effective_pose is frame.ego_pose
            else CausalFrame(
                time_s=frame.time_s,
                ego_pose=effective_pose,
                observations=frame.observations,
                person_detection_count=frame.person_detection_count,
            )
        )
        self._last_time_s = frame.time_s
        self._merge_ego_history(
            frame.time_s,
            effective_pose,
            supplied_history,
        )

        guard_prediction: Prediction | None = None
        guard_active = False
        guard_boundary_s = (
            self.r0_config.route_horizon_s
            * self.r2_config.imminent_horizon_fraction
        )
        if self._guard is not None:
            selected_guard_frame = guard_frame or effective_frame
            guard_prediction = self._guard.step(selected_guard_frame)
            guard_future_s = guard_prediction.diagnostic.get("future_s")
            guard_active = bool(
                guard_prediction.raw_alert is True
                and isinstance(guard_future_s, (int, float))
                and float(guard_future_s) <= guard_boundary_s + EPSILON
            )

        decision, unknown_reason, motion = self._early_decision(effective_frame)
        if self.arm is not R3Arm.C_CURVED_DISTRIBUTIONAL_GUARDED:
            if decision is None:
                return self._prediction(
                    frame.time_s,
                    None,
                    unknown_reason,
                    diagnostic=self._unknown_diagnostic(),
                )
            return self._prediction_from_track(frame.time_s, decision, motion)

        return self._guarded_prediction(
            frame.time_s,
            decision,
            unknown_reason,
            motion,
            guard_prediction,
            guard_active,
            guard_boundary_s,
        )

    def _validate_frame(self, frame: CausalFrame) -> None:
        if not math.isfinite(frame.time_s):
            raise ValueError("frame time must be finite")
        if self._last_time_s is not None and frame.time_s <= self._last_time_s:
            raise ValueError("frame times must be strictly increasing")
        if frame.ego_pose is not None and not _finite_pose(frame.ego_pose):
            raise ValueError("ego pose must contain finite values")
        seen: set[str] = set()
        for observation in frame.observations:
            if not observation.track_id or observation.track_id in seen:
                raise ValueError("track ids must be non-empty and unique per frame")
            seen.add(observation.track_id)
            if not all(
                math.isfinite(value)
                for value in (
                    observation.forward_m,
                    observation.left_m,
                    observation.radius_m,
                )
            ) or observation.radius_m < 0.0:
                raise ValueError("observations must contain finite metric values")

    def _coerce_ego_history(
        self,
        history: Sequence[TimedEgoPose | tuple[float, EgoPose]],
        *,
        current_time_s: float,
    ) -> list[TimedEgoPose]:
        samples: list[TimedEgoPose] = []
        for value in history:
            sample = (
                value
                if isinstance(value, TimedEgoPose)
                else TimedEgoPose(time_s=float(value[0]), pose=value[1])
            )
            if not math.isfinite(sample.time_s) or not _finite_pose(sample.pose):
                raise ValueError("ego history must contain finite times and poses")
            if sample.time_s > current_time_s + EPSILON:
                raise ValueError("future ego pose is not a causal R3 input")
            samples.append(sample)
        samples.sort(key=lambda item: item.time_s)
        for left, right in zip(samples, samples[1:]):
            if math.isclose(left.time_s, right.time_s, rel_tol=0.0, abs_tol=EPSILON):
                if not _same_pose(left.pose, right.pose):
                    raise ValueError("conflicting ego poses share one timestamp")
        return samples

    def _resolve_current_pose(
        self,
        frame: CausalFrame,
        supplied_history: Sequence[TimedEgoPose],
    ) -> EgoPose | None:
        supplied_current = next(
            (
                sample.pose
                for sample in reversed(supplied_history)
                if math.isclose(
                    sample.time_s, frame.time_s, rel_tol=0.0, abs_tol=EPSILON
                )
            ),
            None,
        )
        if frame.ego_pose is not None and supplied_current is not None:
            if not _same_pose(frame.ego_pose, supplied_current):
                raise ValueError("frame pose conflicts with supplied current ego pose")
        return frame.ego_pose if frame.ego_pose is not None else supplied_current

    def _merge_ego_history(
        self,
        current_time_s: float,
        current_pose: EgoPose | None,
        supplied_history: Sequence[TimedEgoPose],
    ) -> None:
        merged = list(self._ego_history)
        merged.extend(supplied_history)
        if current_pose is not None:
            merged.append(TimedEgoPose(current_time_s, current_pose))
        merged.sort(key=lambda item: item.time_s)
        deduplicated: list[TimedEgoPose] = []
        for sample in merged:
            if deduplicated and math.isclose(
                deduplicated[-1].time_s,
                sample.time_s,
                rel_tol=0.0,
                abs_tol=EPSILON,
            ):
                if not _same_pose(deduplicated[-1].pose, sample.pose):
                    raise ValueError("conflicting ego poses share one timestamp")
                continue
            deduplicated.append(sample)
        cutoff_s = current_time_s - self.r3_config.ego_history_window_s
        self._ego_history = [
            sample
            for sample in deduplicated
            if sample.time_s + EPSILON >= cutoff_s
            and sample.time_s <= current_time_s + EPSILON
        ]

    def _early_decision(
        self,
        frame: CausalFrame,
    ) -> tuple[TrackDecision | None, str, EgoMotionEstimate | None]:
        if frame.ego_pose is None:
            return None, "missing_ego_pose", None
        if not frame.observations:
            return None, "missing_current_metric_track", None

        # Preserve target samples even while the independent ego-motion branch
        # is still warming up.  Otherwise the first usable ego estimate would
        # incorrectly discard an already-causal target displacement.
        cutoff_s = frame.time_s - self.r1_config.history_window_s
        current: list[WorldTargetObservation] = []
        for observation in frame.observations:
            target = WorldTargetObservation(
                time_s=frame.time_s,
                track_id=observation.track_id,
                position=frame.ego_pose.local_to_world(
                    observation.forward_m, observation.left_m
                ),
                radius_m=observation.radius_m,
            )
            history = self._tracks.setdefault(observation.track_id, [])
            history.append(target)
            history[:] = [item for item in history if item.time_s + EPSILON >= cutoff_s]
            current.append(target)

        motion = _estimate_ego_motion(
            self._ego_history, self.r3_config.minimum_ego_span_s
        )
        if motion is None:
            return None, "insufficient_causal_ego_motion", None

        curved = self.arm is not R3Arm.B_STRAIGHT_DISTRIBUTIONAL
        route = _route_samples(
            frame.ego_pose,
            motion,
            curved=curved,
            horizon_s=self.r1_config.route_horizon_s,
            step_s=self.r3_config.route_sample_step_s,
        )
        route_segments = _route_segments(route)

        decisions: list[TrackDecision] = []
        distributional = self.arm is not R3Arm.A_CURVED_ROBUST_CV
        for target in current:
            hypotheses = _pairwise_velocities(
                self._tracks[target.track_id],
                self.r1_config.minimum_pair_span_s,
            )
            if not hypotheses:
                continue
            decisions.append(
                self._evaluate_track(
                    target,
                    hypotheses,
                    route_segments,
                    distributional=distributional,
                )
            )
        if not decisions:
            return None, "insufficient_causal_target_motion", motion
        best = max(
            decisions,
            key=lambda item: (
                item.raw_alert,
                item.decision_score,
                item.entry_support,
                -(
                    item.entry_time_s
                    if item.entry_time_s is not None
                    else math.inf
                ),
                -item.minimum_separation_m,
            ),
        )
        return best, "evaluated", motion

    def _evaluate_track(
        self,
        target: WorldTargetObservation,
        hypotheses: Sequence[Vec2],
        route: Sequence[RouteSegment],
        *,
        distributional: bool,
    ) -> TrackDecision:
        tube_radius_m = self.r1_config.route_half_width_m + target.radius_m
        robust_velocity = Vec2(
            statistics.median(item.x for item in hypotheses),
            statistics.median(item.y for item in hypotheses),
        )

        if distributional:
            evidence = [
                _trajectory_evidence(
                    target.position,
                    velocity,
                    route,
                    tube_radius_m,
                    self.r1_config.minimum_closing_speed_mps,
                )
                for velocity in hypotheses
            ]
            entries = [
                item.entry_time_s
                for item in evidence
                if item.entry_time_s is not None
            ]
            entering_hypotheses = len(entries)
            entry_support = entering_hypotheses / len(hypotheses)
            # Integer arithmetic makes the frozen strict-majority boundary
            # exact: a 1/2, 2/4, ... tie cannot alert.
            raw_alert = entering_hypotheses * 2 > len(hypotheses)
            entry_time_s = statistics.median(entries) if entries else None
            minimum_separation_m = statistics.median(
                item.minimum_separation_m for item in evidence
            )
            minimum_separation_time_s = statistics.median(
                item.minimum_separation_time_s for item in evidence
            )
        else:
            robust_evidence = _trajectory_evidence(
                target.position,
                robust_velocity,
                route,
                tube_radius_m,
                self.r1_config.minimum_closing_speed_mps,
            )
            raw_alert = robust_evidence.entry_time_s is not None
            entering_hypotheses = int(raw_alert)
            entry_support = float(raw_alert)
            entry_time_s = robust_evidence.entry_time_s
            minimum_separation_m = robust_evidence.minimum_separation_m
            minimum_separation_time_s = (
                robust_evidence.minimum_separation_time_s
            )

        urgent_boundary_s = self.r1_config.route_horizon_s / 2.0
        urgent = bool(
            raw_alert
            and entry_time_s is not None
            and entry_time_s <= urgent_boundary_s + EPSILON
        )
        return TrackDecision(
            track_id=target.track_id,
            raw_alert=raw_alert,
            urgent=urgent,
            decision_score=_ordinal_decision_score(
                raw_alert,
                entry_support,
                entry_time_s,
                self.r1_config.route_horizon_s,
            ),
            entry_support=entry_support,
            entry_time_s=entry_time_s,
            minimum_separation_m=minimum_separation_m,
            minimum_separation_time_s=minimum_separation_time_s,
            velocity_hypotheses=len(hypotheses),
            entering_hypotheses=entering_hypotheses,
            robust_velocity_mps=robust_velocity,
        )

    def _track_diagnostic(
        self,
        decision: TrackDecision,
        motion: EgoMotionEstimate | None,
    ) -> dict[str, float | str]:
        distributional = self.arm is not R3Arm.A_CURVED_ROBUST_CV
        return {
            "decision_score": decision.decision_score,
            "decision_score_kind": "ordinal_not_probability",
            "entry_support": decision.entry_support,
            "entry_time_s": (
                decision.entry_time_s
                if decision.entry_time_s is not None
                else "none"
            ),
            "median_entry_s": (
                decision.entry_time_s
                if decision.entry_time_s is not None
                else "none"
            ),
            "minimum_separation_m": decision.minimum_separation_m,
            "minimum_separation_time_s": decision.minimum_separation_time_s,
            "velocity_hypotheses": float(decision.velocity_hypotheses),
            "entering_hypotheses": float(decision.entering_hypotheses),
            "support_threshold": (
                FIXED_SUPPORT_THRESHOLD if distributional else "not_applied"
            ),
            "support_rule": (
                "strict_majority_entry_support_gt_0.5"
                if distributional
                else "robust_cv_entry"
            ),
            "entry_support_kind": (
                "pairwise_trajectory_fraction"
                if distributional
                else "binary_robust_cv_entry_not_ensemble_support"
            ),
            "decision_rule": (
                "strict_majority_pairwise_trajectory_entry"
                if distributional
                else "componentwise_median_target_cv_entry"
            ),
            "route_model": (
                "causal_ctrv_curved"
                if self.arm is not R3Arm.B_STRAIGHT_DISTRIBUTIONAL
                else "causal_forward_speed_straight"
            ),
            "route_history_s": self.r3_config.ego_history_window_s,
            "route_sample_step_s": self.r3_config.route_sample_step_s,
            "route_tube_radius_without_target_m": self.r1_config.route_half_width_m,
            "robust_target_velocity_x_mps": decision.robust_velocity_mps.x,
            "robust_target_velocity_y_mps": decision.robust_velocity_mps.y,
            "ego_forward_speed_mps": (
                motion.forward_speed_mps if motion is not None else "unknown"
            ),
            "ego_yaw_rate_radps": (
                motion.yaw_rate_radps if motion is not None else "unknown"
            ),
            "ego_motion_samples": (
                float(motion.sample_count) if motion is not None else "unknown"
            ),
            "ego_motion_span_s": motion.span_s if motion is not None else "unknown",
        }

    def _unknown_diagnostic(self) -> dict[str, float | str]:
        distributional = self.arm is not R3Arm.A_CURVED_ROBUST_CV
        return {
            "decision_score": 0.0,
            "decision_score_kind": "ordinal_not_probability",
            "entry_support": "unknown",
            "entry_time_s": "none",
            "median_entry_s": "none",
            "minimum_separation_m": "unknown",
            "minimum_separation_time_s": "unknown",
            "support_threshold": (
                FIXED_SUPPORT_THRESHOLD if distributional else "not_applied"
            ),
            "support_rule": (
                "strict_majority_entry_support_gt_0.5"
                if distributional
                else "robust_cv_entry"
            ),
        }

    def _prediction_from_track(
        self,
        time_s: float,
        decision: TrackDecision,
        motion: EgoMotionEstimate | None,
    ) -> Prediction:
        reason = (
            "curved_ctrv_robust_target_cv_evaluated"
            if self.arm is R3Arm.A_CURVED_ROBUST_CV
            else "straight_distributional_occupancy_evaluated"
        )
        return self._prediction(
            time_s,
            decision.raw_alert,
            reason,
            decision.track_id,
            urgent=decision.urgent,
            diagnostic=self._track_diagnostic(decision, motion),
        )

    def _guarded_prediction(
        self,
        time_s: float,
        decision: TrackDecision | None,
        unknown_reason: str,
        motion: EgoMotionEstimate | None,
        guard_prediction: Prediction | None,
        guard_active: bool,
        guard_boundary_s: float,
    ) -> Prediction:
        assert guard_prediction is not None
        if decision is not None and decision.raw_alert or guard_active:
            raw_alert: bool | None = True
        elif decision is not None:
            raw_alert = False
        else:
            raw_alert = None

        early_urgent = decision.urgent if decision is not None else False
        urgent = early_urgent or guard_active
        diagnostic = (
            self._track_diagnostic(decision, motion)
            if decision is not None
            else self._unknown_diagnostic()
        )
        guard_future = guard_prediction.diagnostic.get("future_s")
        guard_separation = guard_prediction.diagnostic.get("minimum_separation_m")
        guard_score = 0.0
        if guard_active and isinstance(guard_future, (int, float)):
            guard_urgency = 1.0 - min(
                1.0,
                max(0.0, float(guard_future) / self.r0_config.route_horizon_s),
            )
            guard_score = 0.5 + 0.5 * guard_urgency

        early_entry = decision.entry_time_s if decision is not None else None
        guard_owns_entry = bool(
            guard_active
            and isinstance(guard_future, (int, float))
            and (early_entry is None or float(guard_future) < early_entry)
        )
        if guard_owns_entry:
            diagnostic["entry_time_s"] = float(guard_future)
            diagnostic["median_entry_s"] = float(guard_future)
            if isinstance(guard_separation, (int, float)):
                diagnostic["minimum_separation_m"] = float(guard_separation)
        diagnostic["decision_score"] = max(
            float(diagnostic["decision_score"]), guard_score
        )
        diagnostic.update(
            {
                "guard_boundary_s": guard_boundary_s,
                "imminent_guard_active": str(guard_active).lower(),
                "r0_guard_raw_alert": str(guard_prediction.raw_alert).lower(),
                "r0_guard_entry_time_s": (
                    guard_future if isinstance(guard_future, (int, float)) else "none"
                ),
                "r0_guard_minimum_separation_m": (
                    guard_separation
                    if isinstance(guard_separation, (int, float))
                    else "unknown"
                ),
            }
        )
        return self._prediction(
            time_s,
            raw_alert,
            (
                "imminent_route_guard"
                if guard_active and (decision is None or not decision.raw_alert)
                else "curved_distributional_occupancy_evaluated"
                if decision is not None
                else unknown_reason
            ),
            (
                guard_prediction.track_id
                if guard_active and (decision is None or not decision.raw_alert)
                else decision.track_id if decision is not None else None
            ),
            urgent=urgent,
            diagnostic=diagnostic,
        )

    def _prediction(
        self,
        time_s: float,
        raw_alert: bool | None,
        reason: str,
        track_id: str | None = None,
        *,
        urgent: bool = False,
        diagnostic: Mapping[str, float | str] | None = None,
    ) -> Prediction:
        return Prediction(
            time_s=time_s,
            signal=self.lifecycle.update(time_s, raw_alert, urgent=urgent),
            raw_alert=raw_alert,
            reason=reason,
            track_id=track_id,
            diagnostic=diagnostic or {},
        )


def run_r3_arm(
    frames: Iterable[CausalFrame],
    arm: R3Arm | str,
    r1_config: R1Config = FROZEN_R1_CONFIG,
    r3_config: R3Config = FROZEN_R3_CONFIG,
    r0_config: DTRConfig | None = None,
    r2_config: R2Config = FROZEN_R2_CONFIG,
    *,
    guard_frames: Iterable[CausalFrame | None] | None = None,
) -> list[Prediction]:
    """Run an R3 arm, checking that an optional guard stream is aligned."""

    runner = DTRR3Arm(arm, r1_config, r3_config, r0_config, r2_config)
    if guard_frames is None:
        return [runner.step(frame) for frame in frames]

    sentinel = object()
    guard_iterator = iter(guard_frames)
    output: list[Prediction] = []
    for frame in frames:
        guard_frame = next(guard_iterator, sentinel)
        if guard_frame is sentinel:
            raise ValueError("guard frame stream ended before the R3 frame stream")
        output.append(runner.step(frame, guard_frame))
    if next(guard_iterator, sentinel) is not sentinel:
        raise ValueError("guard frame stream is longer than the R3 frame stream")
    return output
