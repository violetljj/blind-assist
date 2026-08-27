"""Dependency-free DTR-R1 robust future-occupancy mechanics.

R0 commits to one least-squares constant-velocity forecast.  R1 instead keeps
the causal pairwise velocity hypotheses implicit in a short relative-motion
track.  Their component-wise median is the robust Theil-Sen motion estimate;
the full hypothesis set supplies an occupancy-support confidence.  One noisy
displacement therefore cannot manufacture an event.

The relative track is formed after rotating the detector observation into the
world frame and subtracting the current ego position.  It consequently
contains both target and wearer motion without assuming a nominal walking
speed.  The same representation also accepts already ego-relative tracks such
as JRDB's processed 3-D labels.

Truth and future observations are never consumed here.  Complexity is
quadratic only in the short causal history (normally tens of samples).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from typing import Iterable, Sequence

from dtr_r0 import (
    Arm,
    CausalFrame,
    Prediction,
    Signal,
    Vec2,
)


EPSILON = 1e-9


@dataclass(frozen=True)
class R1Config:
    history_window_s: float = 1.50
    minimum_pair_span_s: float = 0.20
    route_horizon_s: float = 3.00
    route_half_width_m: float = 0.65
    minimum_closing_speed_mps: float = 0.10
    clear_grace_s: float = 0.50

    def to_dict(self) -> dict[str, float | str]:
        values: dict[str, float | str] = {
            "history_window_s": self.history_window_s,
            "minimum_pair_span_s": self.minimum_pair_span_s,
            "route_horizon_s": self.route_horizon_s,
            "route_half_width_m": self.route_half_width_m,
            "minimum_closing_speed_mps": self.minimum_closing_speed_mps,
            "clear_grace_s": self.clear_grace_s,
            "velocity_hypothesis": "all_causal_pairwise_slopes",
            "occupancy_consensus": "theil_sen_velocity_with_vote_support",
            "urgent_boundary": "half_route_horizon",
        }
        canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
        values["fingerprint_sha256"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        return values


FROZEN_R1_CONFIG = R1Config()


@dataclass(frozen=True)
class RelativeObservation:
    time_s: float
    track_id: str
    position: Vec2
    radius_m: float


@dataclass(frozen=True)
class OccupancyConsensus:
    track_id: str
    raw_alert: bool
    urgent: bool
    support: float
    median_entry_s: float | None
    velocity_hypotheses: int
    entering_hypotheses: int
    radius_m: float
    horizon_s: float

    @property
    def risk_score(self) -> float:
        if not self.raw_alert or self.median_entry_s is None:
            return 0.0
        # Support owns forecast confidence; normalized time-to-entry owns
        # urgency.  This score is descriptive rather than another threshold.
        urgency = 1.0 - min(
            1.0, max(0.0, self.median_entry_s / self.horizon_s)
        )
        return self.support * (0.5 + 0.5 * urgency)


class RiskEventLifecycle:
    """Stable ONSET/HOLD/ESCALATE/CLEAR lifecycle for one risk stream."""

    def __init__(self, clear_grace_s: float) -> None:
        if clear_grace_s < 0.0 or not math.isfinite(clear_grace_s):
            raise ValueError("clear grace must be finite and non-negative")
        self.clear_grace_s = clear_grace_s
        self.active = False
        self.escalated = False
        self._clear_candidate_since_s: float | None = None

    def update(
        self,
        time_s: float,
        raw_alert: bool | None,
        *,
        urgent: bool = False,
    ) -> Signal:
        if raw_alert is None:
            self._clear_candidate_since_s = None
            return Signal.UNKNOWN
        if raw_alert:
            self._clear_candidate_since_s = None
            if not self.active:
                self.active = True
                self.escalated = False
                return Signal.ONSET
            if urgent and not self.escalated:
                self.escalated = True
                return Signal.ESCALATE
            return Signal.HOLD
        if not self.active:
            self._clear_candidate_since_s = None
            return Signal.CLEAR
        if self._clear_candidate_since_s is None:
            self._clear_candidate_since_s = time_s
        if time_s - self._clear_candidate_since_s + EPSILON < self.clear_grace_s:
            return Signal.HOLD
        self.active = False
        self.escalated = False
        self._clear_candidate_since_s = None
        return Signal.CLEAR


def _first_tube_entry_s(
    position: Vec2,
    velocity: Vec2,
    radius_m: float,
    horizon_s: float,
    minimum_closing_speed_mps: float,
) -> float | None:
    """Return first future entry into a circular time-aligned route tube.

    A target already inside the tube is actionable only while it is still
    closing.  That makes stable side-by-side or parallel occupancy different
    from a new route incursion without inventing a separate class rule.
    """

    distance = position.norm()
    speed_squared = velocity.dot(velocity)
    if distance <= radius_m + EPSILON:
        if distance <= EPSILON:
            closing_speed = velocity.norm()
        else:
            closing_speed = -position.dot(velocity) / distance
        return 0.0 if closing_speed >= minimum_closing_speed_mps else None
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


def _velocity_hypotheses(
    history: Sequence[RelativeObservation],
    minimum_span_s: float,
) -> list[Vec2]:
    hypotheses: list[Vec2] = []
    ordered = sorted(history, key=lambda item: item.time_s)
    for left_index, left in enumerate(ordered[:-1]):
        for right in ordered[left_index + 1 :]:
            span_s = right.time_s - left.time_s
            if span_s + EPSILON < minimum_span_s:
                continue
            hypotheses.append((right.position - left.position) * (1.0 / span_s))
    return hypotheses


def occupancy_consensus(
    history: Sequence[RelativeObservation],
    config: R1Config = FROZEN_R1_CONFIG,
) -> OccupancyConsensus | None:
    if not history:
        return None
    current = max(history, key=lambda item: item.time_s)
    hypotheses = _velocity_hypotheses(history, config.minimum_pair_span_s)
    if not hypotheses:
        return None
    tube_radius_m = config.route_half_width_m + current.radius_m
    entry_votes = [
        _first_tube_entry_s(
            current.position,
            velocity,
            tube_radius_m,
            config.route_horizon_s,
            config.minimum_closing_speed_mps,
        )
        for velocity in hypotheses
    ]
    finite_entries = [value for value in entry_votes if value is not None]
    support = len(finite_entries) / len(entry_votes)
    robust_velocity = Vec2(
        statistics.median(item.x for item in hypotheses),
        statistics.median(item.y for item in hypotheses),
    )
    robust_entry_s = _first_tube_entry_s(
        current.position,
        robust_velocity,
        tube_radius_m,
        config.route_horizon_s,
        config.minimum_closing_speed_mps,
    )
    raw_alert = robust_entry_s is not None
    median_entry_s = robust_entry_s
    urgent = bool(
        raw_alert
        and median_entry_s is not None
        and median_entry_s <= config.route_horizon_s / 2.0 + EPSILON
    )
    return OccupancyConsensus(
        track_id=current.track_id,
        raw_alert=raw_alert,
        urgent=urgent,
        support=support,
        median_entry_s=median_entry_s,
        velocity_hypotheses=len(entry_votes),
        entering_hypotheses=len(finite_entries),
        radius_m=tube_radius_m,
        horizon_s=config.route_horizon_s,
    )


class DTRR1Arm:
    """Causal multi-target robust route-occupancy runner."""

    def __init__(self, config: R1Config = FROZEN_R1_CONFIG) -> None:
        self.arm = Arm.D_R1_OCCUPANCY_CONSENSUS
        self.config = config
        self.lifecycle = RiskEventLifecycle(self.config.clear_grace_s)
        self._tracks: dict[str, list[RelativeObservation]] = {}
        self._last_time_s: float | None = None

    def step(self, frame: CausalFrame) -> Prediction:
        self._validate_frame(frame)
        self._last_time_s = frame.time_s
        if frame.ego_pose is None:
            return self._prediction(frame.time_s, None, "missing_ego_pose")
        if not frame.observations:
            return self._prediction(
                frame.time_s, None, "missing_current_metric_track"
            )

        cutoff_s = frame.time_s - self.config.history_window_s
        current_consensus: list[OccupancyConsensus] = []
        for observation in frame.observations:
            target_world = frame.ego_pose.local_to_world(
                observation.forward_m, observation.left_m
            )
            relative = RelativeObservation(
                time_s=frame.time_s,
                track_id=observation.track_id,
                position=target_world - frame.ego_pose.position,
                radius_m=observation.radius_m,
            )
            history = self._tracks.setdefault(observation.track_id, [])
            history.append(relative)
            history[:] = [item for item in history if item.time_s >= cutoff_s]
            consensus = occupancy_consensus(history, self.config)
            if consensus is not None:
                current_consensus.append(consensus)

        if not current_consensus:
            return self._prediction(
                frame.time_s, None, "insufficient_causal_relative_track"
            )
        best = max(
            current_consensus,
            key=lambda item: (
                item.raw_alert,
                item.risk_score,
                item.support,
                -(
                    item.median_entry_s
                    if item.median_entry_s is not None
                    else math.inf
                ),
            ),
        )
        return self._prediction(
            frame.time_s,
            best.raw_alert,
            "robust_route_occupancy_consensus_evaluated",
            best.track_id,
            urgent=best.urgent,
            diagnostic={
                "occupancy_support": best.support,
                "risk_score": best.risk_score,
                "median_entry_s": (
                    best.median_entry_s
                    if best.median_entry_s is not None
                    else "none"
                ),
                "velocity_hypotheses": float(best.velocity_hypotheses),
                "entering_hypotheses": float(best.entering_hypotheses),
                "route_tube_radius_m": best.radius_m,
            },
        )

    def _validate_frame(self, frame: CausalFrame) -> None:
        if not math.isfinite(frame.time_s):
            raise ValueError("frame time must be finite")
        if self._last_time_s is not None and frame.time_s <= self._last_time_s:
            raise ValueError("frame times must be strictly increasing")
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

    def _prediction(
        self,
        time_s: float,
        raw_alert: bool | None,
        reason: str,
        track_id: str | None = None,
        *,
        urgent: bool = False,
        diagnostic: dict[str, float | str] | None = None,
    ) -> Prediction:
        return Prediction(
            time_s=time_s,
            signal=self.lifecycle.update(time_s, raw_alert, urgent=urgent),
            raw_alert=raw_alert,
            reason=reason,
            track_id=track_id,
            diagnostic=diagnostic or {},
        )


def run_r1_arm(
    frames: Iterable[CausalFrame],
    config: R1Config = FROZEN_R1_CONFIG,
) -> list[Prediction]:
    runner = DTRR1Arm(config)
    return [runner.step(frame) for frame in frames]
