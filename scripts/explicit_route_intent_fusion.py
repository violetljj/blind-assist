#!/usr/bin/env python3
"""Deterministic route-risk lifecycle for externally supplied route intent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RouteRiskSample:
    timestamp_ms: int
    route_valid: bool
    intersection_fraction: float | None


def decode_route_risk_lifecycle(samples: Iterable[RouteRiskSample], *, threshold: float = 1.0 / 3.0,
                                open_consecutive: int = 2, clear_consecutive: int = 2,
                                expected_step_ms: int = 1000) -> list[dict[str, Any]]:
    if open_consecutive < 1 or clear_consecutive < 1:
        raise ValueError("consecutive counts must be positive")
    rows = sorted(samples, key=lambda sample: sample.timestamp_ms)
    transitions: list[dict[str, Any]] = []
    state = "context_attention"
    active_run = 0
    clear_run = 0
    previous_timestamp: int | None = None
    for sample in rows:
        contiguous = previous_timestamp is not None and sample.timestamp_ms - previous_timestamp == expected_step_ms
        if not contiguous:
            active_run = 0
            clear_run = 0
        previous_timestamp = sample.timestamp_ms
        if not sample.route_valid or sample.intersection_fraction is None:
            active_run = 0
            clear_run = 0
            continue
        active = float(sample.intersection_fraction) >= threshold
        if state != "intervention_needed":
            active_run = active_run + 1 if active else 0
            clear_run = 0
            if active_run >= open_consecutive:
                state = "intervention_needed"
                transitions.append({"state": state, "timestamp_ms": sample.timestamp_ms})
                active_run = 0
        else:
            clear_run = clear_run + 1 if not active else 0
            active_run = 0
            if clear_run >= clear_consecutive:
                state = "route_clear"
                transitions.append({"state": state, "timestamp_ms": sample.timestamp_ms})
                state = "context_attention"
                clear_run = 0
    return transitions


def first_intervention_timestamp(transitions: Iterable[dict[str, Any]]) -> int | None:
    return next((int(row["timestamp_ms"]) for row in transitions
                 if row.get("state") == "intervention_needed"), None)
