"""Frozen B1 policy evaluator layered on the closed B0 state semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scripts.research.l10m_b0.b0c_precedence import _arrival_supported
from scripts.research.l10m_b0.evaluation import (
    Action,
    Belief,
    Evidence,
    Hazard,
    ProgressStatus,
    Truth,
    _shield,
    _update_belief,
)

from .policy_space import PolicySpec


COHORT_PATH = Path(__file__).with_name("hidden_cohort_v1.json")


@dataclass(frozen=True)
class HiddenStep:
    evidence: Evidence
    truth: Truth
    accepted_actions: frozenset[Action]


def load_hidden_cohort(path: Path = COHORT_PATH) -> list[list[HiddenStep]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cohort_id") != "L10M-B1-HIDDEN-SYNTHETIC-COHORT-V1":
        raise ValueError("unexpected B1 hidden cohort identity")
    episodes: list[list[HiddenStep]] = []
    seen_ids: set[str] = set()
    for episode in payload.get("episodes", []):
        episode_id = episode["episode_id"]
        if episode_id in seen_ids:
            raise ValueError("duplicate hidden episode identity")
        seen_ids.add(episode_id)
        rows: list[HiddenStep] = []
        for index, step in enumerate(episode["steps"]):
            evidence = Evidence(
                episode_id=episode_id,
                step=index,
                alignment=float(step["alignment"]),
                center_hazard=Hazard(step["hazard"]),
                quality=float(step["quality"]),
                stale=bool(step.get("stale", False)),
                conflict=bool(step.get("conflict", False)),
                target_visible=bool(step.get("target_visible", True)),
                progress_signal=step.get("progress_signal"),
            )
            evidence.validate()
            truth = Truth(
                episode_id=episode_id,
                step=index,
                progress=float(step["progress"]),
                arrived=bool(step["arrived"]),
                unsafe_forward=bool(step["unsafe_forward"]),
            )
            accepted = frozenset(Action(value) for value in step["accepted_actions"])
            if not accepted:
                raise ValueError("each hidden step needs at least one accepted action")
            rows.append(HiddenStep(evidence, truth, accepted))
        if not rows:
            raise ValueError("hidden episode is empty")
        episodes.append(rows)
    if not episodes:
        raise ValueError("hidden cohort is empty")
    return episodes


def evaluate_spec(spec: PolicySpec, cohort: list[list[HiddenStep]] | None = None) -> dict[str, object]:
    """Evaluate one candidate without allowing it to edit state, truth, or safety."""
    spec.validate()
    episodes = load_hidden_cohort() if cohort is None else cohort
    arrival_episodes = 0
    successful_arrival_episodes = 0
    action_count = 0
    accepted_action_count = 0
    unsafe_actions = 0
    false_arrivals = 0
    oscillations = 0
    terminal_violations = 0
    unknown_stuck_increments = 0
    ledger = []

    for rows in episodes:
        belief = Belief()
        previous_direction: Action | None = None
        episode_arrived = any(row.truth.arrived for row in rows)
        episode_success = False
        actions: list[str] = []
        for row in rows:
            evidence = row.evidence
            truth = row.truth
            terminal = _arrival_supported(evidence, truth)

            # These transitions are part of the B0 freeze, not the candidate.
            if evidence.progress_status == ProgressStatus.POSITIVE:
                belief.stuck_count = 0
            action = Action.STOP if terminal else spec.propose(evidence, belief)
            if action != Action.STOP:
                action = _shield(action, evidence, belief)

            action_count += 1
            accepted_action_count += int(action in row.accepted_actions)
            unsafe_actions += int(action == Action.FORWARD and truth.unsafe_forward)
            if (
                action == Action.STOP
                and not truth.arrived
                and evidence.quality >= 0.50
                and evidence.center_hazard == Hazard.LOW
                and abs(evidence.alignment) <= 0.10
            ):
                false_arrivals += 1
            if (
                previous_direction in {Action.LEFT, Action.RIGHT}
                and action in {Action.LEFT, Action.RIGHT}
                and action != previous_direction
            ):
                oscillations += 1
            previous_direction = action
            if terminal and action not in {Action.STOP, Action.FORWARD}:
                terminal_violations += 1
            episode_success |= terminal and action in {Action.STOP, Action.FORWARD}

            before_stuck = belief.stuck_count
            _update_belief(belief, evidence, action)
            if evidence.progress_status == ProgressStatus.UNKNOWN and belief.stuck_count > before_stuck:
                unknown_stuck_increments += 1
            actions.append(action.value)

        arrival_episodes += int(episode_arrived)
        successful_arrival_episodes += int(episode_arrived and episode_success)
        ledger.append({"episode_id": rows[0].evidence.episode_id, "actions": actions, "success": episode_success})

    arrival_success_rate = successful_arrival_episodes / arrival_episodes
    action_agreement_rate = accepted_action_count / action_count
    unsafe_action_rate = unsafe_actions / action_count
    false_arrival_rate = false_arrivals / action_count
    oscillation_rate = oscillations / len(episodes)
    semantic_valid = terminal_violations == 0 and unknown_stuck_increments == 0
    unsafe_candidate = unsafe_actions > 0
    behavioral_score = (
        0.60 * arrival_success_rate
        + 0.40 * action_agreement_rate
        - 1.00 * unsafe_action_rate
        - 0.20 * false_arrival_rate
        - 0.05 * oscillation_rate
    )
    return {
        "semantic_valid": semantic_valid,
        "unsafe_candidate": unsafe_candidate,
        "behavioral_score": behavioral_score,
        "behavioral_vector": {
            "arrival_success_rate": arrival_success_rate,
            "action_agreement_rate": action_agreement_rate,
            "unsafe_action_rate": unsafe_action_rate,
            "false_arrival_rate": false_arrival_rate,
            "oscillation_rate": oscillation_rate,
        },
        "invariant_counts": {
            "terminal_violations": terminal_violations,
            "unknown_stuck_increments": unknown_stuck_increments,
        },
        "episode_ledger": ledger,
    }
