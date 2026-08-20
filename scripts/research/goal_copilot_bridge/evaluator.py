"""BlindAssist-owned closed-loop evaluator for GOAL-COPILOT-1."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from task_api import CANDIDATE_FUNCTIONS, Action, Observation

MODULE_DIR = Path(__file__).resolve().parent
SEALED_SCENARIOS = MODULE_DIR / "sealed_scenarios.json"
REQUIRED_FUNCTIONS = set(CANDIDATE_FUNCTIONS)
ALLOWED_NODES = {
    ast.Module,
    ast.Expr,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.If,
    ast.Return,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.BoolOp,
    ast.UnaryOp,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Is,
    ast.IsNot,
    ast.Tuple,
    ast.List,
    ast.Subscript,
    ast.Attribute,
}
ALLOWED_ACTIONS = {action.value for action in Action}
MOTION_SAFETY_FIELDS = {
    "FORWARD": "forward_free",
    "ALIGN_LEFT": "left_free",
    "ALIGN_RIGHT": "right_free",
}


class CandidateContractError(ValueError):
    """Candidate source is outside the explicitly searchable policy surface."""


class EvaluationInfrastructureError(RuntimeError):
    """The BA-owned evaluator or sealed material cannot produce an assessment."""


def load_candidate(source_path: Path) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise CandidateContractError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise CandidateContractError("dunder names are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CandidateContractError("dunder attributes are not allowed")

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    defined = {node.name for node in functions}
    if defined != REQUIRED_FUNCTIONS or len(functions) != len(REQUIRED_FUNCTIONS):
        raise CandidateContractError(
            f"candidate functions must be exactly {sorted(REQUIRED_FUNCTIONS)}; got {sorted(defined)}"
        )
    if any(not isinstance(node, (ast.Expr, ast.FunctionDef)) for node in tree.body):
        raise CandidateContractError("module-level state is not allowed")

    namespace: dict[str, Any] = {"__builtins__": {}}
    exec(compile(tree, str(source_path), "exec"), namespace, namespace)
    return {name: namespace[name] for name in REQUIRED_FUNCTIONS}


def _observation(values: list[Any]) -> Observation:
    if len(values) != len(Observation.__dataclass_fields__):
        raise EvaluationInfrastructureError("sealed observation shape does not match task API")
    return Observation(*values)


def _candidate_call(function: Any, *args: Any) -> tuple[bool, Any]:
    try:
        return True, function(*args)
    except Exception as exc:  # Candidate failure is a semantic rejection, not BA infra.
        return False, f"{type(exc).__name__}: {exc}"


def _evaluate_scenario(functions: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    node_id = scenario["start"]
    previous_observation: Observation | None = None
    belief = 0.0
    last_seen_bearing: float | None = None
    max_progress = 0.0
    total_actions = 0
    wrong_way_actions = 0
    unsafe_guidance = 0
    premature_completion = 0
    recovery_steps = 0
    ever_acquired = False
    continuity_visible_steps = 0
    continuity_observation_steps = 0
    had_loss = False
    reacquired = False
    semantic_validity = True
    runtime_error: str | None = None
    completed = False

    for _ in range(scenario["max_actions"]):
        node = scenario["nodes"][node_id]
        observation = _observation(node["observation"])
        max_progress = max(max_progress, float(node["progress"]))
        if ever_acquired:
            continuity_observation_steps += 1
            if observation.target_visible:
                continuity_visible_steps += 1
        if observation.target_visible:
            if had_loss:
                reacquired = True
            last_seen_bearing = observation.target_bearing
            ever_acquired = True
        elif ever_acquired:
            had_loss = True

        ok, belief_result = _candidate_call(functions["update_task_belief"], belief, observation)
        if not ok or not isinstance(belief_result, (int, float)):
            semantic_validity = False
            runtime_error = str(belief_result)
            break
        belief = float(belief_result)

        ok, progress_result = _candidate_call(
            functions["detect_progress"], previous_observation, observation
        )
        if not ok or not isinstance(progress_result, bool):
            semantic_validity = False
            runtime_error = str(progress_result)
            break

        ok, proposals_result = _candidate_call(
            functions["propose_actions"], scenario["task_family"], observation, belief
        )
        if not ok or not isinstance(proposals_result, (tuple, list)):
            semantic_validity = False
            runtime_error = str(proposals_result)
            break
        proposals = tuple(proposals_result)
        if not proposals or any(action not in ALLOWED_ACTIONS for action in proposals):
            semantic_validity = False
            runtime_error = "proposal contains no action or an unknown action"
            break

        if not observation.target_visible and last_seen_bearing is not None:
            ok, recovery_action = _candidate_call(functions["recover_target"], last_seen_bearing)
            if not ok or recovery_action not in {"SCAN_LEFT", "SCAN_RIGHT"}:
                semantic_validity = False
                runtime_error = str(recovery_action)
                break
            proposals = (recovery_action, *proposals)
            recovery_steps += 1

        ok, completion_claim = _candidate_call(
            functions["decide_completion"], scenario["task_family"], observation, belief
        )
        if not ok or not isinstance(completion_claim, bool):
            semantic_validity = False
            runtime_error = str(completion_claim)
            break

        ok, action = _candidate_call(
            functions["select_action"], proposals, observation, completion_claim
        )
        if not ok or action not in ALLOWED_ACTIONS:
            semantic_validity = False
            runtime_error = str(action)
            break
        total_actions += 1

        safety_field = MOTION_SAFETY_FIELDS.get(action)
        if safety_field and not getattr(observation, safety_field):
            unsafe_guidance += 1

        completion_actions = set(node.get("completion_actions", []))
        if action in {"COMPLETE", "INTERACT"}:
            if action in completion_actions and completion_claim:
                completed = True
                max_progress = 1.0
                break
            premature_completion += 1

        next_node = node.get("transitions", {}).get(action)
        if next_node is None:
            if action != "STOP":
                wrong_way_actions += 1
        else:
            node_id = next_node
        previous_observation = observation

    return {
        "scenario_id": scenario["id"],
        "task_family": scenario["task_family"],
        "goal_completion": completed,
        "normalized_progress": max_progress,
        "reacquisition_required": had_loss,
        "reacquisition_success": (not had_loss) or reacquired,
        "tracking_continuity": (
            continuity_visible_steps / continuity_observation_steps
            if continuity_observation_steps
            else 1.0
        ),
        "wrong_way_actions": wrong_way_actions,
        "unsafe_guidance": unsafe_guidance,
        "premature_completion": premature_completion,
        "recovery_steps": recovery_steps,
        "total_actions": total_actions,
        "timeout": not completed and total_actions >= scenario["max_actions"],
        "semantic_validity": semantic_validity,
        "candidate_runtime_error": runtime_error,
    }


def _aggregate(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(outcomes)
    required_recovery = [outcome for outcome in outcomes if outcome["reacquisition_required"]]
    return {
        "goal_completion": sum(outcome["goal_completion"] for outcome in outcomes) / count,
        "normalized_progress": sum(outcome["normalized_progress"] for outcome in outcomes) / count,
        "reacquisition_success": (
            sum(outcome["reacquisition_success"] for outcome in required_recovery)
            / len(required_recovery)
            if required_recovery
            else None
        ),
        "tracking_continuity": sum(outcome["tracking_continuity"] for outcome in outcomes) / count,
        "wrong_way_actions": sum(outcome["wrong_way_actions"] for outcome in outcomes),
        "unsafe_guidance": sum(outcome["unsafe_guidance"] for outcome in outcomes),
        "premature_completion": sum(outcome["premature_completion"] for outcome in outcomes),
        "recovery_steps": sum(outcome["recovery_steps"] for outcome in outcomes),
        "total_actions": sum(outcome["total_actions"] for outcome in outcomes),
        "timeout": sum(outcome["timeout"] for outcome in outcomes),
        "semantic_validity": all(outcome["semantic_validity"] for outcome in outcomes),
    }


def evaluate_candidate(source_path: Path) -> dict[str, Any]:
    functions = load_candidate(source_path)
    try:
        sealed = json.loads(SEALED_SCENARIOS.read_text(encoding="utf-8"))
        scenarios = sealed["scenarios"]
        if {scenario["task_family"] for scenario in scenarios} != {
            "FIND_AND_REACH",
            "TRACK_AND_REACQUIRE",
            "FIND_ALIGN_INTERACT",
        }:
            raise EvaluationInfrastructureError("sealed task-family coverage is incomplete")
        outcomes = [_evaluate_scenario(functions, scenario) for scenario in scenarios]
    except EvaluationInfrastructureError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationInfrastructureError(f"sealed evaluator material is invalid: {exc}") from exc

    metrics = _aggregate(outcomes)
    hard_gate_pass = (
        metrics["unsafe_guidance"] == 0
        and metrics["premature_completion"] == 0
        and metrics["semantic_validity"]
    )
    if not hard_gate_pass:
        assessment = "REJECT"
        reason = "SAFETY_PREMATURE_OR_SEMANTIC_HARD_GATE_FAILED"
    elif metrics["goal_completion"] < 1.0:
        assessment = "REJECT"
        reason = "GOAL_COMPLETION_NOT_ESTABLISHED_ALL_FAMILIES"
    else:
        assessment = "ACCEPT"
        reason = "MOCK_POLICY_PASSES_FROZEN_V0_MECHANICAL_GATE"
    return {
        "assessment": assessment,
        "reason": reason,
        "hard_gate_pass": hard_gate_pass,
        "metrics": metrics,
        "outcomes": outcomes,
        "observation_schema": list(Observation.__dataclass_fields__),
        "claim_ceiling": "bridge_mechanics_only_no_model_or_scientific_result",
    }
