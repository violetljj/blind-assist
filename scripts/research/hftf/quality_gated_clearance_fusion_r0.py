"""Deterministic quality-gated clearance filtering; no model or optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence


class State(str, Enum):
    CLEAR = "CLEAR"
    OCCUPIED = "OCCUPIED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Evidence:
    timestamp_ns: int
    clearance_m: Sequence[Optional[float]]
    geometry_valid: Sequence[bool]
    tof_valid: bool
    teacher_age_s: Optional[float]
    disagreement: Optional[float]


@dataclass(frozen=True)
class Output:
    clearance_m: tuple[Optional[float], Optional[float], Optional[float]]
    state: tuple[State, State, State]
    quality: str


MAX_GAP_NS = 500_000_000
STALE_TEACHER_S = 0.5
HIGH_DISAGREEMENT = 0.20
CLEAR_CONFIRMATIONS = 2
CLEARANCE_THRESHOLD_M = 1.5


class Filter:
    def __init__(self) -> None:
        self._previous: Optional[Output] = None
        self._previous_timestamp_ns: Optional[int] = None
        self._clear_streak = [0, 0, 0]

    def reset(self) -> None:
        self._previous = None
        self._previous_timestamp_ns = None
        self._clear_streak = [0, 0, 0]

    def update(self, evidence: Evidence) -> Output:
        if len(evidence.clearance_m) != 3 or len(evidence.geometry_valid) != 3:
            return self._unknown(evidence.timestamp_ns, "INVALID_SHAPE")
        if self._previous_timestamp_ns is not None:
            gap = evidence.timestamp_ns - self._previous_timestamp_ns
            if not 0 < gap <= MAX_GAP_NS:
                return self._unknown(evidence.timestamp_ns, "TIMESTAMP_GAP")
        stale = evidence.teacher_age_s is None or evidence.teacher_age_s > STALE_TEACHER_S
        disagreement_high = evidence.disagreement is None or evidence.disagreement > HIGH_DISAGREEMENT
        hard_unknown = any(not valid or value is None for valid, value in zip(evidence.geometry_valid, evidence.clearance_m)) or stale or disagreement_high
        if hard_unknown:
            return self._unknown(evidence.timestamp_ns, "QUALITY_UNKNOWN")

        alpha = 0.6 if evidence.tof_valid and float(evidence.disagreement) <= 0.10 else 0.3
        values: list[Optional[float]] = []
        states: list[State] = []
        previous_values = self._previous.clearance_m if self._previous else (None, None, None)
        previous_states = self._previous.state if self._previous else (State.UNKNOWN,) * 3
        for band, raw in enumerate(evidence.clearance_m):
            assert raw is not None
            prior = previous_values[band]
            filtered = float(raw) if prior is None else alpha * float(raw) + (1.0 - alpha) * float(prior)
            values.append(filtered)
            if filtered <= CLEARANCE_THRESHOLD_M:
                self._clear_streak[band] = 0
                states.append(State.OCCUPIED)
            else:
                self._clear_streak[band] += 1
                states.append(State.CLEAR if previous_states[band] == State.CLEAR or self._clear_streak[band] >= CLEAR_CONFIRMATIONS else State.UNKNOWN)
        output = Output(tuple(values), tuple(states), "HIGH" if alpha == 0.6 else "MEDIUM")
        self._previous, self._previous_timestamp_ns = output, evidence.timestamp_ns
        return output

    def _unknown(self, timestamp_ns: int, quality: str) -> Output:
        self._previous_timestamp_ns = timestamp_ns
        self._clear_streak = [0, 0, 0]
        output = Output((None, None, None), (State.UNKNOWN,) * 3, quality)
        self._previous = output
        return output
