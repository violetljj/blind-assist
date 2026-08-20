"""Frozen deterministic perception-corruption model for GC2-A."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from scripts.research.goal_copilot_bridge.pilot.task_api import Observation

CORRUPTIONS = (
    "TARGET_DROPOUT",
    "BEARING_JITTER",
    "FALSE_TARGET",
    "NEARNESS_ERROR",
    "TRACKING_COLLAPSE",
    "DELAYED_EVIDENCE",
)
SEVERITIES = ("MILD", "MODERATE", "STRESS")

_JITTER = {
    "MILD": (-2.0, 1.0, 2.0, -1.0),
    "MODERATE": (-6.0, 4.0, 7.0, -5.0),
    "STRESS": (-12.0, 9.0, 15.0, -10.0),
}
_NEARNESS = {
    "MILD": (-0.05, 0.08, -0.03, 0.06),
    "MODERATE": (0.25, -0.15, 0.30, -0.20),
    "STRESS": (0.65, -0.35, 0.55, -0.40),
}
_DELAY = {"MILD": 1, "MODERATE": 1, "STRESS": 2}


def condition_names() -> tuple[str, ...]:
    isolated = tuple(f"{name}_{severity}" for name in CORRUPTIONS for severity in SEVERITIES)
    combined = tuple(f"COMBINED_{severity}" for severity in SEVERITIES)
    return ("CLEAN", *isolated, *combined)


def _active(condition: str) -> tuple[tuple[str, ...], str | None]:
    if condition == "CLEAN":
        return (), None
    for severity in SEVERITIES:
        if condition == f"COMBINED_{severity}":
            return CORRUPTIONS, severity
        suffix = f"_{severity}"
        if condition.endswith(suffix):
            corruption = condition[: -len(suffix)]
            if corruption in CORRUPTIONS:
                return (corruption,), severity
    raise ValueError(f"unknown condition: {condition}")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def corrupt_observation(
    current: Observation,
    history: list[Observation],
    *,
    condition: str,
    scenario_index: int,
    action_index: int,
) -> tuple[Observation, tuple[str, ...]]:
    active, severity = _active(condition)
    if not active or severity is None:
        return current, ()
    applied: list[str] = []
    original_visible = current.target_visible
    observed = current

    if "DELAYED_EVIDENCE" in active and history:
        delay = _DELAY[severity]
        use_delay = severity != "MILD" or (action_index + scenario_index) % 4 == 0
        if use_delay:
            observed = history[max(0, len(history) - delay)]
            applied.append("DELAYED_EVIDENCE")

    if "TARGET_DROPOUT" in active and observed.target_visible:
        phase = (action_index + scenario_index) % {"MILD": 7, "MODERATE": 5, "STRESS": 4}[severity]
        hidden_phases = {"MILD": {0}, "MODERATE": {0, 1}, "STRESS": {0, 1, 2}}[severity]
        if phase in hidden_phases:
            observed = replace(
                observed,
                target_visible=False,
                target_confidence=0.0,
                target_relative_scale=None,
                relative_nearness=None,
            )
            applied.append("TARGET_DROPOUT")

    if "BEARING_JITTER" in active and observed.target_bearing is not None:
        delta = _JITTER[severity][(action_index + scenario_index) % 4]
        observed = replace(observed, target_bearing=observed.target_bearing + delta)
        applied.append("BEARING_JITTER")

    if "FALSE_TARGET" in active and not original_visible:
        period = {"MILD": 7, "MODERATE": 4, "STRESS": 2}[severity]
        if (action_index + scenario_index) % period == 0:
            sign = -1.0 if scenario_index % 2 == 0 else 1.0
            bearing = sign * {"MILD": 8.0, "MODERATE": 14.0, "STRESS": 20.0}[severity]
            observed = replace(
                observed,
                target_visible=True,
                target_bearing=bearing,
                target_relative_scale={"MILD": 0.18, "MODERATE": 0.30, "STRESS": 0.45}[severity],
                target_confidence={"MILD": 0.52, "MODERATE": 0.66, "STRESS": 0.78}[severity],
                relative_nearness={"MILD": 0.20, "MODERATE": 0.42, "STRESS": 0.72}[severity],
                interaction_ready=False,
            )
            applied.append("FALSE_TARGET")

    if "NEARNESS_ERROR" in active and observed.relative_nearness is not None:
        delta = _NEARNESS[severity][(action_index + scenario_index) % 4]
        observed = replace(observed, relative_nearness=_clamp(observed.relative_nearness + delta))
        applied.append("NEARNESS_ERROR")

    if "TRACKING_COLLAPSE" in active:
        phase = (action_index + scenario_index) % {"MILD": 7, "MODERATE": 5, "STRESS": 3}[severity]
        collapse = phase == 0 or (severity == "STRESS" and phase == 1)
        if collapse:
            quality = {"MILD": 0.42, "MODERATE": 0.16, "STRESS": 0.0}[severity]
            updates = {"tracking_quality": quality}
            if severity in {"MODERATE", "STRESS"} and observed.target_visible:
                updates.update(target_visible=False, target_confidence=0.0)
            observed = replace(observed, **updates)
            applied.append("TRACKING_COLLAPSE")

    return observed, tuple(applied)


def corruption_counts(events: Iterable[tuple[str, ...]]) -> dict[str, int]:
    counts = {name: 0 for name in CORRUPTIONS}
    for event in events:
        for name in event:
            counts[name] += 1
    return counts
