"""Deterministic B4 harder-instance construction over frozen B1 mechanics."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from scripts.research.l10m_b0.evaluation import Action, Evidence, Hazard, Truth
from scripts.research.l10m_b1.evaluator import HiddenStep, evaluate_spec
from scripts.research.l10m_b1.policy_space import (
    FALLBACK_ACTIONS,
    QUALITY_FLOORS,
    RECOVERY_ACTIONS,
    STUCK_RESPONSES,
    TURN_THRESHOLDS,
    PolicySpec,
)


BENCHMARK_PATH = Path(__file__).with_name("hard_benchmark_v1.json")
EXPECTED_BENCHMARK_ID = "L10M-B4-HARD-FRESH-COHORT-V1"
MOTIF_KEYS = {
    "fine_turn",
    "wide_forward",
    "lost_left",
    "lost_right",
    "quality_040_left",
    "quality_040_right",
    "quality_055_left",
    "quality_055_right",
    "recovery_left",
    "recovery_right",
}


def load_benchmark(path: Path = BENCHMARK_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark_id") != EXPECTED_BENCHMARK_ID:
        raise ValueError("unexpected B4 benchmark identity")
    instances = payload.get("instances")
    if not isinstance(instances, list) or len(instances) != 3:
        raise ValueError("B4 harder cohort requires exactly three instances")
    ids = [row.get("instance_id") for row in instances]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("B4 instance identities must be unique and non-empty")
    for row in instances:
        motifs = row.get("motifs")
        if not isinstance(motifs, dict) or set(motifs) != MOTIF_KEYS:
            raise ValueError(f"unexpected motif schema for {row.get('instance_id')}")
        if any(not isinstance(value, int) or value < 0 for value in motifs.values()):
            raise ValueError("motif counts must be non-negative integers")
    return payload


def _step(
    episode_id: str,
    index: int,
    *,
    alignment: float,
    quality: float,
    accepted: Iterable[Action],
    progress_signal: float | None = 0.2,
    arrived: bool = False,
    target_visible: bool = True,
    hazard: Hazard = Hazard.LOW,
) -> HiddenStep:
    evidence = Evidence(
        episode_id=episode_id,
        step=index,
        alignment=alignment,
        center_hazard=hazard,
        quality=quality,
        target_visible=target_visible,
        progress_signal=progress_signal,
    )
    truth = Truth(
        episode_id=episode_id,
        step=index,
        progress=1.0 if arrived else 0.2,
        arrived=arrived,
        unsafe_forward=False,
    )
    return HiddenStep(evidence=evidence, truth=truth, accepted_actions=frozenset(accepted))


def _single_step_episodes(
    instance_id: str,
    motif: str,
    count: int,
    *,
    alignment: float,
    quality: float,
    accepted: Action,
    target_visible: bool = True,
    hazard: Hazard = Hazard.LOW,
) -> list[list[HiddenStep]]:
    return [
        [
            _step(
                f"{instance_id}-{motif}-{index:02d}",
                0,
                alignment=alignment,
                quality=quality,
                accepted=[accepted],
                target_visible=target_visible,
                progress_signal=None if not target_visible else 0.2,
                hazard=hazard,
            )
        ]
        for index in range(count)
    ]


def _recovery_episodes(
    instance_id: str, direction: Action, count: int
) -> list[list[HiddenStep]]:
    episodes: list[list[HiddenStep]] = []
    for index in range(count):
        episode_id = f"{instance_id}-recovery-{direction.value.lower()}-{index:02d}"
        episodes.append(
            [
                _step(
                    episode_id,
                    0,
                    alignment=0.0,
                    quality=0.95,
                    accepted=[Action.FORWARD],
                    progress_signal=0.0,
                ),
                _step(
                    episode_id,
                    1,
                    alignment=0.0,
                    quality=0.95,
                    accepted=[Action.FORWARD],
                    progress_signal=0.0,
                ),
                _step(
                    episode_id,
                    2,
                    alignment=0.0,
                    quality=0.95,
                    accepted=[direction],
                    progress_signal=0.0,
                    hazard=Hazard.HIGH,
                ),
            ]
        )
    return episodes


def build_instance(instance: dict[str, Any]) -> list[list[HiddenStep]]:
    instance_id = str(instance["instance_id"])
    motifs = instance["motifs"]
    episodes: list[list[HiddenStep]] = [
        [
            _step(
                f"{instance_id}-arrival-anchor",
                0,
                alignment=0.0,
                quality=0.95,
                accepted=[Action.FORWARD],
            ),
            _step(
                f"{instance_id}-arrival-anchor",
                1,
                alignment=0.0,
                quality=0.95,
                accepted=[Action.FORWARD, Action.STOP],
                progress_signal=1.0,
                arrived=True,
            ),
        ]
    ]
    # Alternating signs prevent a direction-specific shortcut; threshold is the
    # only field controlling these motifs.
    for index in range(motifs["fine_turn"]):
        alignment = -0.15 if index % 2 == 0 else 0.15
        direction = Action.LEFT if alignment < 0 else Action.RIGHT
        episodes += _single_step_episodes(
            instance_id,
            f"fine-turn-{index:02d}",
            1,
            alignment=alignment,
            quality=0.95,
            accepted=direction,
        )
    for index in range(motifs["wide_forward"]):
        alignment = -0.25 if index % 2 == 0 else 0.25
        episodes += _single_step_episodes(
            instance_id,
            f"wide-forward-{index:02d}",
            1,
            alignment=alignment,
            quality=0.95,
            accepted=Action.FORWARD,
        )
    for direction in (Action.LEFT, Action.RIGHT):
        suffix = direction.value.lower()
        episodes += _single_step_episodes(
            instance_id,
            f"lost-{suffix}",
            motifs[f"lost_{suffix}"],
            alignment=0.0,
            quality=0.95,
            accepted=direction,
            target_visible=False,
            hazard=Hazard.HIGH,
        )
        for quality_name, quality in (("040", 0.40), ("055", 0.55)):
            episodes += _single_step_episodes(
                instance_id,
                f"quality-{quality_name}-{suffix}",
                motifs[f"quality_{quality_name}_{suffix}"],
                alignment=0.0,
                quality=quality,
                accepted=direction,
                hazard=Hazard.HIGH,
            )
        episodes += _recovery_episodes(
            instance_id, direction, motifs[f"recovery_{suffix}"]
        )
    return episodes


def evaluate_instance(spec: PolicySpec, instance: dict[str, Any]) -> dict[str, object]:
    return evaluate_spec(spec, build_instance(instance))


def legal_neighbors(spec: PolicySpec) -> tuple[PolicySpec, ...]:
    neighbors: set[PolicySpec] = set()
    numeric = (
        ("action_selection_turn_threshold", TURN_THRESHOLDS),
        ("fallback_min_quality", QUALITY_FLOORS),
    )
    for field, values in numeric:
        index = values.index(getattr(spec, field))
        for next_index in (index - 1, index + 1):
            if 0 <= next_index < len(values):
                neighbors.add(replace(spec, **{field: values[next_index]}))
    categorical = (
        ("fallback_action", FALLBACK_ACTIONS),
        ("stuck_response", STUCK_RESPONSES),
        ("recovery_transition_action", RECOVERY_ACTIONS),
    )
    for field, values in categorical:
        for value in values:
            if value != getattr(spec, field):
                neighbors.add(replace(spec, **{field: value}))
    return tuple(sorted(neighbors, key=repr))
