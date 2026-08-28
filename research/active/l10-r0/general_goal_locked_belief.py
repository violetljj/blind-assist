from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot
from typing import Iterable


REFERENCE_THRESHOLD = 0.55
REFERENCE_MARGIN = 0.04
CONTINUATION_INSTANCE_THRESHOLD = 0.45
CONTINUATION_GEOMETRY_RADIUS = 0.28
REACQUIRE_GEOMETRY_RADIUS = 0.24
MAX_LOST_FRAMES = 12


class BeliefState(str, Enum):
    UNKNOWN = "UNKNOWN"
    TARGET = "TARGET"
    LOST = "LOST"


@dataclass(frozen=True)
class GoalRepresentation:
    goal_id: str
    evidence_modalities: frozenset[str]
    reference_id: str | None = None

    def __post_init__(self) -> None:
        allowed = {"SEMANTIC", "REFERENCE", "TEXT", "STRUCTURE", "GEOMETRY"}
        unknown = set(self.evidence_modalities) - allowed
        if unknown:
            raise ValueError(f"Unknown goal evidence modalities: {sorted(unknown)}")


@dataclass(frozen=True)
class CandidateEvidence:
    """Provider-neutral evidence. candidate_id is opaque and is not evaluator truth."""

    candidate_id: str
    box_xywh: tuple[float, float, float, float]
    instance_score: float
    semantic_score: float = 0.0
    text_score: float | None = None
    structure_score: float = 0.0
    visibility_quality: float = 1.0
    association_score: float = 0.0

    def __post_init__(self) -> None:
        x, y, width, height = self.box_xywh
        if width <= 0.0 or height <= 0.0:
            raise ValueError("Candidate boxes must have positive width and height")
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            raise ValueError("Candidate boxes must use normalized [0, 1] coordinates")
        scores = (
            self.instance_score,
            self.semantic_score,
            self.structure_score,
            self.visibility_quality,
            self.association_score,
        )
        if not all(0.0 <= value <= 1.0 for value in scores):
            raise ValueError("Evidence scores must be in [0, 1]")
        if self.text_score is not None and not 0.0 <= self.text_score <= 1.0:
            raise ValueError("text_score must be None or in [0, 1]")

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.box_xywh
        return x + width / 2.0, y + height / 2.0


@dataclass(frozen=True)
class BeliefDecision:
    state: BeliefState
    selected_candidate_id: str | None
    action: str
    authority: str | None
    instance_score: float | None = None
    reference_margin: float | None = None
    geometry_error: float | None = None


def _runner_up(scores: Iterable[float]) -> float:
    ordered = sorted(scores, reverse=True)
    return ordered[1] if len(ordered) > 1 else 0.0


def _direction(center_x: float) -> str:
    if center_x < 0.4:
        return "LEFT"
    if center_x > 0.6:
        return "RIGHT"
    return "CENTER"


class StatelessReferenceMatcher:
    def step(self, candidates: list[CandidateEvidence]) -> BeliefDecision:
        if not candidates:
            return BeliefDecision(BeliefState.UNKNOWN, None, "SEARCH", None)
        scores = [candidate.instance_score for candidate in candidates]
        selected = max(candidates, key=lambda candidate: candidate.instance_score)
        margin = selected.instance_score - _runner_up(scores)
        if selected.instance_score < REFERENCE_THRESHOLD or margin < REFERENCE_MARGIN:
            return BeliefDecision(
                BeliefState.UNKNOWN,
                None,
                "SEARCH_REFERENCE_INSUFFICIENT",
                None,
                selected.instance_score,
                margin,
            )
        return BeliefDecision(
            BeliefState.TARGET,
            selected.candidate_id,
            f"NAVIGATE_{_direction(selected.center[0])}",
            "REFERENCE_STATELESS",
            selected.instance_score,
            margin,
        )


class GoalLockedBeliefController:
    """Reference-bound belief with action-conditioned image-plane prediction."""

    def __init__(self, goal: GoalRepresentation):
        if "REFERENCE" not in goal.evidence_modalities:
            raise ValueError("This G0 controller requires REFERENCE evidence")
        self.goal = goal
        self.state = BeliefState.UNKNOWN
        self.predicted_center: tuple[float, float] | None = None
        self.lost_frames = 0
        self.bound_once = False

    @staticmethod
    def _geometry_error(
        candidate: CandidateEvidence, predicted_center: tuple[float, float]
    ) -> float:
        return hypot(
            candidate.center[0] - predicted_center[0],
            candidate.center[1] - predicted_center[1],
        )

    def _predict(self, action_delta: tuple[float, float]) -> None:
        if self.predicted_center is None:
            return
        self.predicted_center = (
            self.predicted_center[0] + action_delta[0],
            self.predicted_center[1] + action_delta[1],
        )

    def _cold_start(self, candidates: list[CandidateEvidence]) -> BeliefDecision:
        baseline = StatelessReferenceMatcher().step(candidates)
        if baseline.selected_candidate_id is None:
            return baseline
        selected = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == baseline.selected_candidate_id
        )
        self.state = BeliefState.TARGET
        self.predicted_center = selected.center
        self.lost_frames = 0
        self.bound_once = True
        return BeliefDecision(
            BeliefState.TARGET,
            selected.candidate_id,
            baseline.action,
            "REFERENCE_COLD_START",
            baseline.instance_score,
            baseline.reference_margin,
            0.0,
        )

    def _mark_lost(self) -> BeliefDecision:
        self.state = BeliefState.LOST
        self.lost_frames += 1
        if self.lost_frames > MAX_LOST_FRAMES:
            self.state = BeliefState.UNKNOWN
            self.predicted_center = None
            self.bound_once = False
            return BeliefDecision(BeliefState.UNKNOWN, None, "SEARCH_RESET", None)
        return BeliefDecision(BeliefState.LOST, None, "SEARCH_BOUND_INSTANCE", None)

    def step(
        self,
        candidates: list[CandidateEvidence],
        action_delta: tuple[float, float] = (0.0, 0.0),
    ) -> BeliefDecision:
        self._predict(action_delta)
        if self.state == BeliefState.UNKNOWN or self.predicted_center is None:
            return self._cold_start(candidates)

        if self.state == BeliefState.LOST:
            px, py = self.predicted_center
            if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0):
                return self._mark_lost()

        ranked = sorted(
            (
                (self._geometry_error(candidate, self.predicted_center), candidate)
                for candidate in candidates
            ),
            key=lambda row: (
                row[0],
                -row[1].association_score,
                -row[1].instance_score,
                row[1].candidate_id,
            ),
        )
        if not ranked:
            return self._mark_lost()

        geometry_error, selected = ranked[0]
        radius = (
            REACQUIRE_GEOMETRY_RADIUS
            if self.state == BeliefState.LOST
            else CONTINUATION_GEOMETRY_RADIUS
        )
        if geometry_error > radius:
            return self._mark_lost()
        if selected.instance_score < CONTINUATION_INSTANCE_THRESHOLD:
            return self._mark_lost()
        if self.state == BeliefState.LOST:
            scores = [candidate.instance_score for candidate in candidates]
            margin = selected.instance_score - _runner_up(scores)
            if selected.instance_score < REFERENCE_THRESHOLD or margin < REFERENCE_MARGIN:
                return self._mark_lost()
            authority = "REFERENCE_GEOMETRY_REACQUIRE"
        else:
            margin = None
            authority = "BOUND_TEMPORAL_GEOMETRY"

        self.state = BeliefState.TARGET
        self.predicted_center = selected.center
        self.lost_frames = 0
        return BeliefDecision(
            BeliefState.TARGET,
            selected.candidate_id,
            f"NAVIGATE_{_direction(selected.center[0])}",
            authority,
            selected.instance_score,
            margin,
            geometry_error,
        )
