from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class State(str, Enum):
    SEARCH = "SEARCH"
    TARGET_FOUND = "TARGET_FOUND"
    LOCKED = "LOCKED"
    LOST = "LOST"
    NEAR = "NEAR"
    TASK_COMPLETE = "TASK_COMPLETE"


class Action(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FORWARD = "FORWARD"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class Candidate:
    proposal_id: str
    center_x: float
    scale: float
    text_score: float
    appearance_score: float
    structure_score: float
    completion_score: float


@dataclass(frozen=True)
class Decision:
    state: State
    action: Action
    selected_id: str | None
    belief: float


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _logit(probability: float) -> float:
    p = _clamp(probability, 0.02, 0.98)
    return math.log(p / (1.0 - p))


def _direction(center_x: float) -> Action:
    if center_x < -0.14:
        return Action.LEFT
    if center_x > 0.14:
        return Action.RIGHT
    return Action.FORWARD


def _identity_score(candidate: Candidate) -> float:
    return (
        0.46 * candidate.text_score
        + 0.34 * candidate.appearance_score
        + 0.20 * candidate.structure_score
    )


class GreedyFrameController:
    """Per-frame goal grounding baseline with no temporal belief."""

    name = "B0_frame_greedy"

    def reset(self) -> None:
        self.state = State.SEARCH

    def step(self, candidates: Iterable[Candidate]) -> Decision:
        candidates = list(candidates)
        if not candidates:
            self.state = State.SEARCH
            return Decision(self.state, Action.RIGHT, None, 0.0)
        best = max(candidates, key=_identity_score)
        score = _identity_score(best)
        if score < 0.61:
            self.state = State.SEARCH
            return Decision(self.state, Action.RIGHT, None, score)
        if (
            best.scale >= 0.72
            and abs(best.center_x) <= 0.18
            and best.completion_score >= 0.62
        ):
            self.state = State.TASK_COMPLETE
            return Decision(self.state, Action.COMPLETE, best.proposal_id, score)
        self.state = State.LOCKED
        return Decision(self.state, _direction(best.center_x), best.proposal_id, score)


class StickyIoUController:
    """A conventional sticky local tracker; useful but not goal-belief aware."""

    name = "B1_sticky_local"

    def reset(self) -> None:
        self.state = State.SEARCH
        self.last_center: float | None = None
        self.last_scale = 0.0
        self.misses = 0
        self.near_hits = 0

    def step(self, candidates: Iterable[Candidate]) -> Decision:
        candidates = list(candidates)
        eligible = [candidate for candidate in candidates if _identity_score(candidate) >= 0.54]
        best: Candidate | None = None
        if eligible and self.last_center is not None:
            local = [
                candidate
                for candidate in eligible
                if abs(candidate.center_x - self.last_center) <= 0.32
                and abs(candidate.scale - self.last_scale) <= 0.38
            ]
            if local:
                best = max(local, key=lambda item: _identity_score(item) - 0.30 * abs(item.center_x - self.last_center))
        if best is None and eligible:
            best = max(eligible, key=_identity_score)

        if best is None:
            self.misses += 1
            self.near_hits = 0
            if self.last_center is not None and self.misses <= 3:
                self.state = State.LOST
                return Decision(self.state, _direction(self.last_center), None, 0.0)
            self.state = State.SEARCH
            self.last_center = None
            return Decision(self.state, Action.RIGHT, None, 0.0)

        score = _identity_score(best)
        self.misses = 0
        self.last_center = best.center_x
        self.last_scale = best.scale
        self.near_hits = self.near_hits + 1 if (
            best.scale >= 0.70
            and abs(best.center_x) <= 0.18
            and best.completion_score >= 0.58
        ) else 0
        if self.near_hits >= 2:
            self.state = State.TASK_COMPLETE
            return Decision(self.state, Action.COMPLETE, best.proposal_id, score)
        self.state = State.NEAR if best.scale >= 0.66 else State.LOCKED
        return Decision(self.state, _direction(best.center_x), best.proposal_id, score)


class GoalLockController:
    """L10-R0: goal evidence belief, dual memory, and active reacquisition."""

    name = "L10_R0_goal_lock"

    acquire_belief = 3.2
    retain_belief = 0.3

    def reset(self) -> None:
        self.state = State.SEARCH
        self.belief = -2.2
        self.confirm_hits = 0
        self.misses = 0
        self.near_hits = 0
        self.last_center: float | None = None
        self.velocity = 0.0
        self.prototype_text = 0.5
        self.prototype_appearance = 0.5
        self.prototype_structure = 0.5
        self.search_phase = 0
        self.active_id: str | None = None
        self.prototype_ready = False

    def _candidate_evidence(self, candidate: Candidate) -> float:
        semantic = (
            0.47 * _logit(candidate.text_score)
            + 0.33 * _logit(candidate.appearance_score)
            + 0.20 * _logit(candidate.structure_score)
        )
        memory_agreement = 0.0
        if self.prototype_ready:
            memory_agreement = 0.9 * (
                1.0
                - abs(candidate.text_score - self.prototype_text)
                - 0.7 * abs(candidate.appearance_score - self.prototype_appearance)
                - 0.4 * abs(candidate.structure_score - self.prototype_structure)
            )
        motion = 0.0
        if self.last_center is not None:
            predicted = self.last_center + self.velocity
            residual = abs(candidate.center_x - predicted)
            # Broad enough to reacquire after an occlusion; still penalizes a distant confuser.
            motion = 1.0 - min(1.7, 2.3 * residual)
        contradiction = 0.0
        if candidate.text_score < 0.40:
            contradiction += 1.5
        if candidate.appearance_score < 0.42:
            contradiction += 0.9
        return semantic + memory_agreement + motion - contradiction

    def _acquire_gate(self, candidate: Candidate) -> bool:
        initial = (
            candidate.text_score >= 0.66
            and candidate.appearance_score >= 0.60
            and candidate.structure_score >= 0.63
        )
        reacquire = (
            self.prototype_ready
            and candidate.text_score >= 0.50
            and candidate.appearance_score >= 0.62
            and candidate.structure_score >= 0.62
        )
        return initial or reacquire

    def _search_action(self) -> Action:
        if self.last_center is not None and self.misses <= 5:
            predicted = _clamp(self.last_center + self.velocity, -1.0, 1.0)
            return _direction(predicted)
        # Expanding deterministic sweep: 4 right, 8 left, then repeat.
        phase = self.search_phase % 12
        self.search_phase += 1
        return Action.RIGHT if phase < 4 else Action.LEFT

    def _update_prototype(self, candidate: Candidate) -> None:
        alpha = 0.18
        self.prototype_text = (1.0 - alpha) * self.prototype_text + alpha * candidate.text_score
        self.prototype_appearance = (1.0 - alpha) * self.prototype_appearance + alpha * candidate.appearance_score
        self.prototype_structure = (1.0 - alpha) * self.prototype_structure + alpha * candidate.structure_score

    def step(self, candidates: Iterable[Candidate]) -> Decision:
        candidates = list(candidates)
        ranked = sorted(
            ((self._candidate_evidence(candidate), candidate) for candidate in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        best_evidence, best = ranked[0] if ranked else (-2.0, None)

        if self.state in {State.LOCKED, State.NEAR} and self.active_id is not None:
            active = next(
                (
                    (evidence, candidate)
                    for evidence, candidate in ranked
                    if candidate.proposal_id == self.active_id
                ),
                None,
            )
            # A current lock cannot silently jump to another instance. A new
            # proposal is admitted only after the state has explicitly become LOST.
            if active is not None:
                best_evidence, best = active
            elif best is not None and best.proposal_id != self.active_id:
                best_evidence, best = -2.0, None

        if (
            best is not None
            and self.state in {State.SEARCH, State.TARGET_FOUND}
            and self.active_id is not None
            and best.proposal_id != self.active_id
        ):
            # Belief belongs to one candidate hypothesis, not to the whole frame.
            self.belief = -1.4
            self.confirm_hits = 0
        if best is not None and self.state in {State.SEARCH, State.TARGET_FOUND}:
            self.active_id = best.proposal_id

        if best is None or best_evidence < -0.1:
            self.misses += 1
            self.confirm_hits = 0
            self.near_hits = 0
            # Missing expected evidence is negative, but a short gap must not erase identity.
            self.belief = 0.90 * self.belief - (0.55 if self.misses <= 3 else 0.85)
            if self.state in {State.TARGET_FOUND, State.LOCKED, State.NEAR} or (
                self.state is State.LOST and self.misses <= 10
            ):
                self.state = State.LOST
            else:
                self.state = State.SEARCH
                if self.misses > 10:
                    self.last_center = None
                    self.velocity = 0.0
            return Decision(self.state, self._search_action(), None, self.belief)

        previous_center = self.last_center
        self.misses = 0
        self.belief = _clamp(0.72 * self.belief + best_evidence, -6.0, 8.0)
        acquire_ok = self._acquire_gate(best)
        self.confirm_hits = self.confirm_hits + 1 if best_evidence >= 1.0 and acquire_ok else 0

        if self.state in {State.SEARCH, State.TARGET_FOUND, State.LOST}:
            if self.confirm_hits >= 2 and self.belief >= self.acquire_belief:
                self.state = State.LOCKED
                self.active_id = best.proposal_id
            else:
                self.state = State.TARGET_FOUND
        elif self.belief < self.retain_belief:
            self.state = State.LOST
        else:
            self.state = State.LOCKED

        if self.state in {State.LOCKED, State.NEAR}:
            self._update_prototype(best)
            self.prototype_ready = True
            if previous_center is not None:
                instantaneous = best.center_x - previous_center
                self.velocity = 0.65 * self.velocity + 0.35 * instantaneous
            self.last_center = best.center_x

        near_evidence = (
            best.scale >= 0.68
            and abs(best.center_x) <= 0.17
            and best.text_score >= 0.68
            and best.appearance_score >= 0.62
            and best.structure_score >= 0.64
            and best.completion_score >= 0.62
        )
        self.near_hits = self.near_hits + 1 if near_evidence and self.state is State.LOCKED else 0
        if self.near_hits >= 3 and self.belief >= 4.4:
            self.state = State.TASK_COMPLETE
            return Decision(self.state, Action.COMPLETE, best.proposal_id, self.belief)
        if self.state is State.LOCKED and best.scale >= 0.62:
            self.state = State.NEAR
        return Decision(self.state, _direction(best.center_x), best.proposal_id, self.belief)


CONTROLLERS = (GreedyFrameController, StickyIoUController, GoalLockController)
