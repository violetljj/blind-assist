"""BlindAssist-owned deterministic closed-loop dev evaluator.

The candidate language is deliberately not general Python. It is a bounded,
straight-line decision language expressed as six Python functions. No calls,
imports, loops, assignment, mutation, I/O, or module state are admitted.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .task_api import Action, CANDIDATE_SIGNATURES, Observation, TaskFamily
except ImportError:  # Standalone inside an exported SearchTaskBundle.
    from task_api import Action, CANDIDATE_SIGNATURES, Observation, TaskFamily

HERE = Path(__file__).resolve().parent
DEV_SCENARIOS = HERE / "dev_scenarios.json"
MAX_SOURCE_BYTES = 65_536
MAX_AST_NODES = 4_000
ALLOWED_ACTIONS = {action.value for action in Action}
OBSERVATION_FIELDS = set(Observation.__dataclass_fields__)
MOTION_SAFETY_FIELDS = {
    "FORWARD": "forward_free",
    "ALIGN_LEFT": "left_free",
    "ALIGN_RIGHT": "right_free",
}
ALLOWED_NODES = {
    ast.Module, ast.Expr, ast.FunctionDef, ast.arguments, ast.arg, ast.If,
    ast.Return, ast.Compare, ast.Name, ast.Load, ast.Constant, ast.BoolOp,
    ast.UnaryOp, ast.And, ast.Or, ast.Not, ast.USub, ast.Eq, ast.NotEq,
    ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Is, ast.IsNot, ast.Tuple,
    ast.List, ast.Subscript, ast.Attribute,
}


class CandidateContractError(ValueError):
    """Candidate source is outside the frozen searchable policy surface."""


class EvaluationInfrastructureError(RuntimeError):
    """The BA-owned evaluator cannot produce an authoritative assessment."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_candidate(source_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = source_path.read_bytes()
    if len(raw) > MAX_SOURCE_BYTES:
        raise CandidateContractError("candidate source exceeds 65536 bytes")
    source = raw.decode("utf-8")
    tree = ast.parse(source, filename=str(source_path))
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise CandidateContractError("candidate AST exceeds 4000 nodes")
    for node in nodes:
        if type(node) not in ALLOWED_NODES:
            raise CandidateContractError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise CandidateContractError("dunder names are not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr not in OBSERVATION_FIELDS:
                raise CandidateContractError(f"attribute is outside Observation: {node.attr}")

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    defined = {node.name for node in functions}
    required = set(CANDIDATE_SIGNATURES)
    if defined != required or len(functions) != len(required):
        raise CandidateContractError(
            f"candidate functions must be exactly {sorted(required)}; got {sorted(defined)}"
        )
    if any(not isinstance(node, (ast.Expr, ast.FunctionDef)) for node in tree.body):
        raise CandidateContractError("module-level state is not allowed")
    for function in functions:
        args = function.args
        if function.decorator_list or args.vararg or args.kwarg or args.kwonlyargs or args.defaults:
            raise CandidateContractError(f"non-frozen signature: {function.name}")
        if len(args.args) != CANDIDATE_SIGNATURES[function.name]:
            raise CandidateContractError(f"wrong arity: {function.name}")

    namespace: dict[str, Any] = {"__builtins__": {}}
    exec(compile(tree, str(source_path), "exec"), namespace, namespace)
    functions_by_name = {name: namespace[name] for name in required}
    return functions_by_name, {
        "candidate_digest": _sha256_bytes(raw),
        "source_bytes": len(raw),
        "ast_nodes": len(nodes),
    }


def _observation(values: list[Any]) -> Observation:
    if len(values) != len(Observation.__dataclass_fields__):
        raise EvaluationInfrastructureError("scenario observation shape mismatch")
    return Observation(*values)


def _call(function: Any, *args: Any) -> tuple[bool, Any]:
    try:
        return True, function(*args)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _evaluate_scenario(functions: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    step_index = 0
    previous: Observation | None = None
    belief = 0.0
    last_seen_bearing: float | None = None
    had_loss = False
    reacquired = False
    unsafe = 0
    premature = 0
    wrong_way = 0
    actions = 0
    runtime_error: str | None = None
    completed = False
    max_progress = 0.0

    for _ in range(scenario["max_actions"]):
        step = scenario["steps"][step_index]
        observation = _observation(step["observation"])
        max_progress = max(max_progress, float(step["progress"]))
        if observation.target_visible:
            if had_loss:
                reacquired = True
            last_seen_bearing = observation.target_bearing
        elif last_seen_bearing is not None:
            had_loss = True

        ok, value = _call(functions["update_task_belief"], belief, observation)
        if not ok or not isinstance(value, (int, float)) or isinstance(value, bool):
            runtime_error = str(value)
            break
        belief = float(value)
        ok, value = _call(functions["detect_progress"], previous, observation)
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
        safety_field = MOTION_SAFETY_FIELDS.get(action)
        if safety_field and not getattr(observation, safety_field):
            unsafe += 1

        expected = step["expected_action"]
        is_final = step_index == len(scenario["steps"]) - 1
        if action in {"COMPLETE", "INTERACT"} and not (
            is_final and action == expected and completion_claim
        ):
            premature += 1
        if is_final and action == expected and completion_claim:
            completed = True
            max_progress = 1.0
            break
        if action == expected and not is_final:
            step_index += 1
        elif action != "STOP":
            wrong_way += 1
        previous = observation

    return {
        "scenario_id": scenario["id"],
        "task_family": scenario["task_family"],
        "goal_completion": completed,
        "normalized_progress": max_progress,
        "reacquisition_required": had_loss,
        "reacquisition_success": (not had_loss) or reacquired,
        "wrong_way_actions": wrong_way,
        "unsafe_guidance": unsafe,
        "premature_completion": premature,
        "total_actions": actions,
        "timeout": not completed and actions >= scenario["max_actions"],
        "semantic_validity": runtime_error is None,
        "candidate_runtime_error": runtime_error,
    }


def evaluate_payload(source_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    functions, complexity = load_candidate(source_path)
    try:
        scenarios = payload["scenarios"]
        families = {scenario["task_family"] for scenario in scenarios}
        if families != {family.value for family in TaskFamily}:
            raise EvaluationInfrastructureError("task-family coverage is incomplete")
        outcomes = [_evaluate_scenario(functions, scenario) for scenario in scenarios]
    except EvaluationInfrastructureError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationInfrastructureError(f"scenario material is invalid: {exc}") from exc

    family_counts: dict[str, int] = {}
    family_totals: dict[str, int] = {}
    for outcome in outcomes:
        family = outcome["task_family"]
        family_totals[family] = family_totals.get(family, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + int(outcome["goal_completion"])
    family_rates = {key: family_counts[key] / family_totals[key] for key in sorted(family_totals)}
    required_recovery = [item for item in outcomes if item["reacquisition_required"]]
    reacquisition_rate = (
        sum(item["reacquisition_success"] for item in required_recovery) / len(required_recovery)
        if required_recovery else 1.0
    )
    completion_count = sum(item["goal_completion"] for item in outcomes)
    progress_total = sum(float(item["normalized_progress"]) for item in outcomes)
    unsafe = sum(item["unsafe_guidance"] for item in outcomes)
    premature = sum(item["premature_completion"] for item in outcomes)
    semantic = all(item["semantic_validity"] for item in outcomes)
    timeouts = sum(item["timeout"] for item in outcomes)
    completed_actions = sum(item["total_actions"] for item in outcomes if item["goal_completion"])
    hard_gate = unsafe == 0 and premature == 0 and semantic
    guidance_score = (
        min(family_rates.values()) * 1_000_000
        + completion_count * 10_000
        + reacquisition_rate * 100
        + progress_total
        - timeouts * 0.01
        - completed_actions * 0.0001
        - complexity["ast_nodes"] * 0.0000001
        if hard_gate else -1_000_000_000.0
    )
    failed = [
        {"scenario_id": item["scenario_id"], "family": item["task_family"],
         "progress": item["normalized_progress"], "timeout": item["timeout"]}
        for item in outcomes if not item["goal_completion"]
    ]
    metrics = {
        "combined_score": guidance_score,
        "validity": 1 if hard_gate else 0,
        "hard_gate_pass": hard_gate,
        "completion_count": completion_count,
        "scenario_count": len(outcomes),
        "minimum_family_completion_rate": min(family_rates.values()),
        "family_completion_counts": family_counts,
        "family_completion_rates": family_rates,
        "reacquisition_success": reacquisition_rate,
        "normalized_progress_total": progress_total,
        "unsafe_guidance": unsafe,
        "premature_completion": premature,
        "timeouts": timeouts,
        "actions_on_completed_scenarios": completed_actions,
        "semantic_validity": semantic,
        "candidate_complexity_ast_nodes": complexity["ast_nodes"],
        "candidate_digest": complexity["candidate_digest"],
    }
    return {"metrics": metrics, "outcomes": outcomes, "failed_scenarios": failed}


def evaluate_scenarios(source_path: Path, scenario_path: Path) -> dict[str, Any]:
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    return evaluate_payload(source_path, payload)


def evaluate(program_path: str) -> dict[str, Any]:
    """SkyDiscover-compatible dev entry point; dev metrics are guidance only."""
    try:
        result = evaluate_scenarios(Path(program_path), DEV_SCENARIOS)
        return {
            **result["metrics"],
            "artifacts": {
                "feedback": json.dumps(result["failed_scenarios"], sort_keys=True),
                "authority": "DEV_GUIDANCE_ONLY_BLINDASSIST_RETAINS_ACCEPTANCE",
            },
        }
    except CandidateContractError as exc:
        return {
            "combined_score": -1_000_000_000.0,
            "validity": 0,
            "error": str(exc),
            "artifacts": {"feedback": str(exc), "failure_stage": "candidate_isolation"},
        }
