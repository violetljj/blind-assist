"""GC2-A hidden-state evaluator with imperfect candidate-visible observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.pilot.evaluator import (
    ALLOWED_ACTIONS,
    MOTION_SAFETY_FIELDS,
    load_candidate,
)
from scripts.research.goal_copilot_bridge.pilot.task_api import Observation
from scripts.research.goal_copilot_2a.noise import corrupt_observation, corruption_counts


def _observation(values: list[Any]) -> Observation:
    if len(values) != len(Observation.__dataclass_fields__):
        raise ValueError("observation shape mismatch")
    return Observation(*values)


def _call(function: Any, *args: Any) -> tuple[bool, Any]:
    try:
        return True, function(*args)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_episode(
    functions: dict[str, Any], scenario: dict[str, Any], condition: str, scenario_index: int
) -> dict[str, Any]:
    hidden_step = 0
    previous_observed: Observation | None = None
    true_history: list[Observation] = []
    belief = 0.0
    last_seen_bearing: float | None = None
    true_seen = False
    true_loss = False
    true_reacquired = False
    observed_loss_events = 0
    unsafe = 0
    premature = 0
    wrong_way = 0
    actions = 0
    completed = False
    runtime_error: str | None = None
    max_progress = 0.0
    corruption_events: list[tuple[str, ...]] = []
    action_trace: list[str] = []

    for action_index in range(scenario["max_actions"]):
        step = scenario["steps"][hidden_step]
        true_observation = _observation(step["observation"])
        max_progress = max(max_progress, float(step["progress"]))
        if true_observation.target_visible:
            if true_loss:
                true_reacquired = True
            true_seen = True
        elif true_seen:
            true_loss = True

        observation, applied = corrupt_observation(
            true_observation,
            true_history,
            condition=condition,
            scenario_index=scenario_index,
            action_index=action_index,
        )
        corruption_events.append(applied)
        true_history.append(true_observation)
        if not observation.target_visible and previous_observed and previous_observed.target_visible:
            observed_loss_events += 1
        if observation.target_visible:
            last_seen_bearing = observation.target_bearing

        ok, value = _call(functions["update_task_belief"], belief, observation)
        if not ok or not isinstance(value, (int, float)) or isinstance(value, bool):
            runtime_error = str(value)
            break
        belief = float(value)
        ok, value = _call(functions["detect_progress"], previous_observed, observation)
        if not ok or not isinstance(value, bool):
            runtime_error = str(value)
            break
        ok, value = _call(functions["propose_actions"], scenario["task_family"], observation, belief)
        if not ok or not isinstance(value, (tuple, list)) or not value:
            runtime_error = str(value)
            break
        proposals = tuple(value)
        if any(action not in ALLOWED_ACTIONS for action in proposals):
            runtime_error = "proposal contains an unknown action"
            break
        if not observation.target_visible and last_seen_bearing is not None:
            ok, recovery = _call(functions["recover_target"], last_seen_bearing)
            if not ok or recovery not in {"SCAN_LEFT", "SCAN_RIGHT"}:
                runtime_error = str(recovery)
                break
            proposals = (recovery, *proposals)
        ok, completion_claim = _call(
            functions["decide_completion"], scenario["task_family"], observation, belief
        )
        if not ok or not isinstance(completion_claim, bool):
            runtime_error = str(completion_claim)
            break
        ok, action = _call(functions["select_action"], proposals, observation, completion_claim)
        if not ok or action not in ALLOWED_ACTIONS:
            runtime_error = str(action)
            break

        actions += 1
        action_trace.append(action)
        safety_field = MOTION_SAFETY_FIELDS.get(action)
        if safety_field and not getattr(observation, safety_field):
            unsafe += 1
        expected = step["expected_action"]
        is_final = hidden_step == len(scenario["steps"]) - 1
        if action in {"COMPLETE", "INTERACT"} and not (
            is_final and action == expected and completion_claim
        ):
            premature += 1
        if is_final and action == expected and completion_claim:
            completed = True
            max_progress = 1.0
            break
        if action == expected and not is_final:
            hidden_step += 1
        elif action != "STOP":
            wrong_way += 1
        previous_observed = observation

    alignment_trace = [action for action in action_trace if action in {"ALIGN_LEFT", "ALIGN_RIGHT"}]
    reversals = sum(a != b for a, b in zip(alignment_trace, alignment_trace[1:]))
    return {
        "scenario_id": scenario["id"],
        "task_family": scenario["task_family"],
        "condition": condition,
        "goal_completion": completed,
        "normalized_progress": max_progress,
        "true_reacquisition_required": true_loss,
        "true_reacquisition_success": (not true_loss) or true_reacquired,
        "observed_loss_events": observed_loss_events,
        "alignment_direction_reversals": reversals,
        "wrong_way_actions": wrong_way,
        "unsafe_guidance": unsafe,
        "premature_completion": premature,
        "total_actions": actions,
        "timeout": not completed and actions >= scenario["max_actions"],
        "semantic_validity": runtime_error is None,
        "candidate_runtime_error": runtime_error,
        "corruption_events": corruption_counts(corruption_events),
        "action_trace": action_trace,
    }


def evaluate_condition(policy_path: Path, scenario_path: Path, condition: str) -> dict[str, Any]:
    functions, complexity = load_candidate(policy_path)
    scenarios = json.loads(scenario_path.read_text(encoding="utf-8"))["scenarios"]
    outcomes = [run_episode(functions, scenario, condition, index) for index, scenario in enumerate(scenarios)]
    family_totals: dict[str, int] = {}
    family_completion: dict[str, int] = {}
    for outcome in outcomes:
        family = outcome["task_family"]
        family_totals[family] = family_totals.get(family, 0) + 1
        family_completion[family] = family_completion.get(family, 0) + int(outcome["goal_completion"])
    eligible = [outcome for outcome in outcomes if outcome["true_reacquisition_required"]]
    counts = {name: sum(item["corruption_events"][name] for item in outcomes) for name in outcomes[0]["corruption_events"]}
    metrics = {
        "completion_count": sum(item["goal_completion"] for item in outcomes),
        "scenario_count": len(outcomes),
        "completion_rate": sum(item["goal_completion"] for item in outcomes) / len(outcomes),
        "family_completion_counts": dict(sorted(family_completion.items())),
        "family_completion_rates": {family: family_completion[family] / family_totals[family] for family in sorted(family_totals)},
        "normalized_progress_total": sum(float(item["normalized_progress"]) for item in outcomes),
        "eligible_reacquisition_count": len(eligible),
        "eligible_reacquisition_rate": (
            sum(item["true_reacquisition_success"] for item in eligible) / len(eligible) if eligible else 1.0
        ),
        "observed_loss_events": sum(item["observed_loss_events"] for item in outcomes),
        "alignment_direction_reversals": sum(item["alignment_direction_reversals"] for item in outcomes),
        "wrong_way_actions": sum(item["wrong_way_actions"] for item in outcomes),
        "unsafe_guidance": sum(item["unsafe_guidance"] for item in outcomes),
        "premature_completion": sum(item["premature_completion"] for item in outcomes),
        "total_actions": sum(item["total_actions"] for item in outcomes),
        "timeouts": sum(item["timeout"] for item in outcomes),
        "semantic_validity": all(item["semantic_validity"] for item in outcomes),
        "candidate_digest": complexity["candidate_digest"],
        "candidate_complexity_ast_nodes": complexity["ast_nodes"],
        "corruption_event_counts": counts,
    }
    return {"condition": condition, "metrics": metrics, "outcomes": outcomes}
