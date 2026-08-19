"""Deterministic controlled-evidence evaluator for L10M-B0.

This module evaluates policy mechanics only. It deliberately does not load
RGB, depth, detector, or learned-model outputs and must not be interpreted as
an end-to-end BlindAssist result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Arm(str, Enum):
    REACTIVE = "reactive"
    STATEFUL = "stateful"
    STATEFUL_SAFETY = "stateful_safety"


class Action(str, Enum):
    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STOP = "STOP"
    RECOVER = "RECOVER"


class Hazard(str, Enum):
    LOW = "LOW"
    HIGH = "HIGH"


@dataclass(frozen=True)
class Evidence:
    episode_id: str
    step: int
    alignment: float  # negative = target left, positive = target right
    center_hazard: Hazard
    quality: float
    stale: bool = False
    conflict: bool = False
    target_visible: bool = True

    def validate(self) -> None:
        if not self.episode_id or self.step < 0:
            raise ValueError("invalid evidence identity")
        if not -1.0 <= self.alignment <= 1.0:
            raise ValueError("alignment must be in [-1, 1]")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")


@dataclass(frozen=True)
class Truth:
    episode_id: str
    step: int
    progress: float
    arrived: bool = False
    unsafe_forward: bool = False


@dataclass
class Belief:
    progress_direction: str = "UNKNOWN"
    progress_magnitude: float = 0.0
    progress_confidence: float = 0.0
    stuck_count: int = 0
    last_action: Action | None = None
    action_repeat_count: int = 0
    arrival_evidence: float = 0.0


@dataclass
class EpisodeStats:
    success: bool = False
    unsafe_actions: int = 0
    false_arrivals: int = 0
    stuck_detection_step: int | None = None
    recovery_success: int = 0
    unknown_steps: int = 0
    oscillations: int = 0
    excess_actions: int = 0
    actions: list[Action] = field(default_factory=list)


def _proposal(evidence: Evidence, belief: Belief | None) -> Action:
    if not evidence.target_visible or evidence.quality < 0.35:
        return Action.STOP
    if abs(evidence.alignment) > 0.20:
        return Action.LEFT if evidence.alignment < 0 else Action.RIGHT
    if belief is not None and belief.stuck_count >= 2:
        return Action.RECOVER
    return Action.FORWARD


def _shield(action: Action, evidence: Evidence, belief: Belief) -> Action:
    """Evidence-bounded safety contract; never changes evaluator truth."""
    if action == Action.FORWARD and (
        evidence.center_hazard == Hazard.HIGH
        or evidence.conflict
        or evidence.stale
        or evidence.quality < 0.50
    ):
        return Action.STOP
    if action in {Action.LEFT, Action.RIGHT} and evidence.conflict:
        return Action.STOP
    if action == Action.FORWARD and belief.stuck_count >= 3:
        return Action.RECOVER
    return action


def _update_belief(belief: Belief, evidence: Evidence, truth: Truth, action: Action) -> None:
    observable = evidence.quality >= 0.50 and not evidence.stale and not evidence.conflict
    if observable:
        belief.progress_direction = "POSITIVE" if truth.progress > 0 else "NONE"
        belief.progress_magnitude = truth.progress
        belief.progress_confidence = evidence.quality
        if action == Action.FORWARD and truth.progress <= 0:
            belief.stuck_count += 1
        elif truth.progress > 0:
            belief.stuck_count = 0
    else:
        belief.progress_direction = "UNKNOWN"
        belief.progress_confidence = 0.0
    if action == belief.last_action:
        belief.action_repeat_count += 1
    else:
        belief.action_repeat_count = 1
    belief.last_action = action
    belief.arrival_evidence = evidence.quality if observable and abs(evidence.alignment) <= 0.10 else 0.0


def run_episode(arm: Arm, evidence_rows: list[Evidence], truth_rows: list[Truth]) -> EpisodeStats:
    if len(evidence_rows) != len(truth_rows) or not evidence_rows:
        raise ValueError("evidence and truth must be non-empty and aligned")
    belief = Belief()
    stats = EpisodeStats()
    previous_direction: Action | None = None
    for evidence, truth in zip(evidence_rows, truth_rows):
        evidence.validate()
        if evidence.episode_id != truth.episode_id or evidence.step != truth.step:
            raise ValueError("evidence/truth identity mismatch")
        action = _proposal(evidence, belief if arm != Arm.REACTIVE else None)
        if arm == Arm.STATEFUL_SAFETY:
            action = _shield(action, evidence, belief)
        if action == Action.STOP and evidence.quality < 0.50:
            stats.unknown_steps += 1
        stats.unsafe_actions += int(action == Action.FORWARD and truth.unsafe_forward)
        if action == Action.STOP and truth.arrived is False and evidence.quality >= 0.50:
            stats.false_arrivals += 1
        if previous_direction in {Action.LEFT, Action.RIGHT} and action in {Action.LEFT, Action.RIGHT} and action != previous_direction:
            stats.oscillations += 1
        previous_direction = action
        stats.actions.append(action)
        before_stuck = belief.stuck_count
        if arm != Arm.REACTIVE:
            _update_belief(belief, evidence, truth, action)
            if belief.stuck_count >= 2 and before_stuck < 2 and stats.stuck_detection_step is None:
                stats.stuck_detection_step = evidence.step
            if action == Action.RECOVER and truth.progress > 0:
                stats.recovery_success += 1
        # Termination is evidence-bounded: evaluator truth alone cannot turn
        # stale/low-quality evidence into an ARRIVED decision.
        arrival_supported = (
            evidence.quality >= 0.50
            and not evidence.stale
            and not evidence.conflict
            and evidence.target_visible
            and abs(evidence.alignment) <= 0.10
        )
        if truth.arrived and arrival_supported and action in {Action.STOP, Action.FORWARD}:
            stats.success = True
    stats.excess_actions = max(0, len(stats.actions) - 1)
    return stats


def summarize(results: dict[Arm, Iterable[EpisodeStats]]) -> dict[str, dict[str, float | None]]:
    """Return the B0 behavioral vector; no composite score is produced."""
    output: dict[str, dict[str, float | None]] = {}
    for arm, rows_iter in results.items():
        rows = list(rows_iter)
        n = len(rows)
        if not n:
            raise ValueError("each arm requires at least one episode")
        detections = [r.stuck_detection_step for r in rows if r.stuck_detection_step is not None]
        output[arm.value] = {
            "task_success": sum(r.success for r in rows) / n,
            "unsafe_action_rate": sum(r.unsafe_actions for r in rows) / n,
            "false_arrival_rate": sum(r.false_arrivals for r in rows) / n,
            "stuck_detection_latency": sum(detections) / len(detections) if detections else None,
            "recovery_success": sum(r.recovery_success for r in rows) / n,
            "UNKNOWN_rate": sum(r.unknown_steps for r in rows) / sum(len(r.actions) for r in rows),
            "oscillation_rate": sum(r.oscillations for r in rows) / n,
            "excess_action": sum(r.excess_actions for r in rows) / n,
        }
    return output
