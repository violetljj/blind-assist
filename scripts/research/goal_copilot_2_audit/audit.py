"""Read-only GC2 failure autopsy, observability bounds, and RGB grounding audit.

This module does not alter the frozen GC2 evaluator or noise implementation.  It
replays only consumed development scenarios and accepts an already locked public
development winner.  Held-out material is neither an input nor an accepted path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from scripts.research.goal_copilot_bridge.pilot.evaluator import (
    ALLOWED_ACTIONS,
    MOTION_SAFETY_FIELDS,
    load_candidate,
)
from scripts.research.goal_copilot_bridge.pilot.task_api import Observation
from scripts.research.goal_copilot_2a import noise as frozen_noise


CORRUPTIONS = frozen_noise.CORRUPTIONS
MODERATE = "MODERATE"
_JITTER = (-6.0, 4.0, 7.0, -5.0)
_NEARNESS = (0.25, -0.15, 0.30, -0.20)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _observation(values: list[Any]) -> Observation:
    return Observation(*values)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def corrupt_moderate_subset(
    current: Observation,
    history: list[Observation],
    *,
    active: Iterable[str],
    scenario_index: int,
    action_index: int,
) -> tuple[Observation, tuple[str, ...]]:
    """Mirror frozen COMBINED_MODERATE while allowing diagnostic ablations."""
    enabled = set(active)
    unknown = enabled.difference(CORRUPTIONS)
    if unknown:
        raise ValueError(f"unknown corruptions: {sorted(unknown)}")
    applied: list[str] = []
    original_visible = current.target_visible
    observed = current

    if "DELAYED_EVIDENCE" in enabled and history:
        observed = history[max(0, len(history) - 1)]
        applied.append("DELAYED_EVIDENCE")

    if "TARGET_DROPOUT" in enabled and observed.target_visible:
        phase = (action_index + scenario_index) % 5
        if phase in {0, 1}:
            observed = replace(
                observed,
                target_visible=False,
                target_confidence=0.0,
                target_relative_scale=None,
                relative_nearness=None,
            )
            applied.append("TARGET_DROPOUT")

    if "BEARING_JITTER" in enabled and observed.target_bearing is not None:
        observed = replace(
            observed,
            target_bearing=observed.target_bearing + _JITTER[(action_index + scenario_index) % 4],
        )
        applied.append("BEARING_JITTER")

    if "FALSE_TARGET" in enabled and not original_visible:
        if (action_index + scenario_index) % 4 == 0:
            observed = replace(
                observed,
                target_visible=True,
                target_bearing=(-14.0 if scenario_index % 2 == 0 else 14.0),
                target_relative_scale=0.30,
                target_confidence=0.66,
                relative_nearness=0.42,
                interaction_ready=False,
            )
            applied.append("FALSE_TARGET")

    if "NEARNESS_ERROR" in enabled and observed.relative_nearness is not None:
        observed = replace(
            observed,
            relative_nearness=_clamp(
                observed.relative_nearness + _NEARNESS[(action_index + scenario_index) % 4]
            ),
        )
        applied.append("NEARNESS_ERROR")

    if "TRACKING_COLLAPSE" in enabled:
        if (action_index + scenario_index) % 5 == 0:
            updates: dict[str, Any] = {"tracking_quality": 0.16}
            if observed.target_visible:
                updates.update(target_visible=False, target_confidence=0.0)
            observed = replace(observed, **updates)
            applied.append("TRACKING_COLLAPSE")

    return observed, tuple(applied)


def _call(function: Any, *args: Any) -> tuple[bool, Any]:
    try:
        return True, function(*args)
    except Exception as exc:  # pragma: no cover - diagnostic preservation path
        return False, f"{type(exc).__name__}: {exc}"


def run_episode(
    functions: dict[str, Any],
    scenario: dict[str, Any],
    scenario_index: int,
    active: Iterable[str],
) -> dict[str, Any]:
    hidden_step = 0
    previous_observed: Observation | None = None
    true_history: list[Observation] = []
    belief = 0.0
    last_seen_bearing: float | None = None
    completed = False
    unsafe = premature = wrong_way = 0
    runtime_error: str | None = None
    trace: list[dict[str, Any]] = []

    for action_index in range(scenario["max_actions"]):
        step = scenario["steps"][hidden_step]
        true_observation = _observation(step["observation"])
        observation, applied = corrupt_moderate_subset(
            true_observation,
            true_history,
            active=active,
            scenario_index=scenario_index,
            action_index=action_index,
        )
        true_history.append(true_observation)
        if observation.target_visible:
            last_seen_bearing = observation.target_bearing

        belief_before = belief
        ok, value = _call(functions["update_task_belief"], belief, observation)
        if not ok or not isinstance(value, (int, float)) or isinstance(value, bool):
            runtime_error = str(value)
            break
        belief = float(value)
        ok, _ = _call(functions["detect_progress"], previous_observed, observation)
        if not ok:
            runtime_error = "detect_progress failed"
            break
        ok, proposals_value = _call(
            functions["propose_actions"], scenario["task_family"], observation, belief
        )
        if not ok or not isinstance(proposals_value, (tuple, list)) or not proposals_value:
            runtime_error = str(proposals_value)
            break
        proposals = tuple(proposals_value)
        if any(action not in ALLOWED_ACTIONS for action in proposals):
            runtime_error = "proposal contains an unknown action"
            break
        recovery_injected = False
        if not observation.target_visible and last_seen_bearing is not None:
            ok, recovery = _call(functions["recover_target"], last_seen_bearing)
            if not ok or recovery not in {"SCAN_LEFT", "SCAN_RIGHT"}:
                runtime_error = str(recovery)
                break
            proposals = (recovery, *proposals)
            recovery_injected = True
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

        expected = step["expected_action"]
        is_final = hidden_step == len(scenario["steps"]) - 1
        safety_field = MOTION_SAFETY_FIELDS.get(action)
        if safety_field and not getattr(observation, safety_field):
            unsafe += 1
        if action in {"COMPLETE", "INTERACT"} and not (
            is_final and action == expected and completion_claim
        ):
            premature += 1
        aligned = action == expected
        trace.append(
            {
                "action_index": action_index,
                "hidden_step": hidden_step,
                "expected_action": expected,
                "action": action,
                "aligned": aligned,
                "belief_before": belief_before,
                "belief_after": belief,
                "completion_claim": completion_claim,
                "recovery_injected": recovery_injected,
                "applied_corruptions": list(applied),
                "true_observation": asdict(true_observation),
                "observed": asdict(observation),
            }
        )
        if is_final and aligned and completion_claim:
            completed = True
            break
        if aligned and not is_final:
            hidden_step += 1
        elif action != "STOP":
            wrong_way += 1
        previous_observed = observation

    return {
        "scenario_id": scenario["id"],
        "task_family": scenario["task_family"],
        "goal_completion": completed,
        "final_hidden_step": hidden_step,
        "wrong_way_actions": wrong_way,
        "unsafe_guidance": unsafe,
        "premature_completion": premature,
        "timeout": not completed and len(trace) >= scenario["max_actions"],
        "semantic_validity": runtime_error is None,
        "candidate_runtime_error": runtime_error,
        "trace": trace,
    }


def _summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = {}
    for outcome in outcomes:
        families.setdefault(outcome["task_family"], []).append(outcome)
    return {
        "completion_count": sum(item["goal_completion"] for item in outcomes),
        "scenario_count": len(outcomes),
        "family_completion_counts": {
            family: sum(item["goal_completion"] for item in items)
            for family, items in sorted(families.items())
        },
        "wrong_way_actions": sum(item["wrong_way_actions"] for item in outcomes),
        "unsafe_guidance": sum(item["unsafe_guidance"] for item in outcomes),
        "premature_completion": sum(item["premature_completion"] for item in outcomes),
        "timeouts": sum(item["timeout"] for item in outcomes),
        "semantic_validity": all(item["semantic_validity"] for item in outcomes),
    }


def _classify_first_divergence(item: dict[str, Any]) -> str:
    expected, action = item["expected_action"], item["action"]
    observed = item["observed"]
    applied = set(item["applied_corruptions"])
    if not observed["target_visible"] and "TRACKING_COLLAPSE" in applied:
        return "TRACKING_COLLAPSE_FORCED_TARGET_LOSS"
    if not observed["target_visible"] and "TARGET_DROPOUT" in applied:
        return "TARGET_DROPOUT_FORCED_TARGET_LOSS"
    if "DELAYED_EVIDENCE" in applied and observed != item["true_observation"]:
        return "STALE_EVIDENCE_WRONG_STATE"
    if "FALSE_TARGET" in applied and action != expected:
        return "FALSE_TARGET_DIRECTIONAL_DIVERSION"
    if expected in {"ALIGN_LEFT", "ALIGN_RIGHT"} and action == "FORWARD" and "BEARING_JITTER" in applied:
        return "BEARING_JITTER_ALIGNMENT_BYPASS"
    if action in {"ALIGN_LEFT", "ALIGN_RIGHT", "SCAN_LEFT", "SCAN_RIGHT"} and action != expected:
        return "DIRECTIONAL_DIVERSION_OR_RECOVERY_LOCK"
    if expected in {"COMPLETE", "INTERACT"}:
        return "COMPLETION_EVIDENCE_NOT_ADMITTED"
    if action == "STOP":
        return "EVIDENCE_GATE_STOP"
    return "OTHER_WRONG_ACTION"


def _failure_autopsy(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = []
    classes: Counter[str] = Counter()
    for outcome in outcomes:
        first = next((item for item in outcome["trace"] if not item["aligned"]), None)
        classification = "NO_ACTION_DIVERGENCE" if first is None else _classify_first_divergence(first)
        classes[classification] += 1
        episodes.append(
            {
                "scenario_id": outcome["scenario_id"],
                "task_family": outcome["task_family"],
                "completed": outcome["goal_completion"],
                "last_aligned_action_index": (
                    -1 if first is None else int(first["action_index"]) - 1
                ),
                "first_divergence": first,
                "classification": classification,
                "final_hidden_step": outcome["final_hidden_step"],
            }
        )
    return {"classification_counts": dict(sorted(classes.items())), "episodes": episodes}


def _oracle_paths(scenarios: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    paths: list[list[dict[str, Any]]] = []
    for scenario_index, scenario in enumerate(scenarios):
        true_history: list[Observation] = []
        path: list[dict[str, Any]] = []
        for action_index, step in enumerate(scenario["steps"]):
            truth = _observation(step["observation"])
            observed, _ = corrupt_moderate_subset(
                truth,
                true_history,
                active=CORRUPTIONS,
                scenario_index=scenario_index,
                action_index=action_index,
            )
            true_history.append(truth)
            path.append(
                {
                    "observation": observed,
                    "expected_action": step["expected_action"],
                    "task_family": scenario["task_family"],
                }
            )
        paths.append(path)
    return paths


def _history_upper_bound(paths: list[list[dict[str, Any]]]) -> dict[str, Any]:
    mapping: dict[bytes, str] = {}
    conflicts: list[dict[str, Any]] = []
    for path in paths:
        observations: list[dict[str, Any]] = []
        actions: list[str] = []
        for item in path:
            observations.append(asdict(item["observation"]))
            key = _canonical(
                {
                    "task_family": item["task_family"],
                    "observations": observations,
                    "past_actions": actions,
                }
            )
            previous = mapping.get(key)
            if previous is not None and previous != item["expected_action"]:
                conflicts.append({"actions": sorted({previous, item["expected_action"]})})
            mapping[key] = item["expected_action"]
            actions.append(item["expected_action"])
    return {
        "completion_upper_bound": 0 if conflicts else len(paths),
        "scenario_count": len(paths),
        "history_action_conflict_count": len(conflicts),
        "lookup_states": len(mapping),
        "scope": "CONSUMED_FIXED_SCENARIOS_ONLY_NOT_GENERALIZATION",
    }


def _observation_predicate(observation: Observation) -> str:
    terms = [
        f"observation.{name} == {value!r}"
        for name, value in asdict(observation).items()
    ]
    return " and ".join(terms)


def _surface_lookup_source(paths: list[list[dict[str, Any]]]) -> tuple[str, list[str]]:
    conflicts: list[str] = []
    initial: dict[str, tuple[int, str]] = {}
    transitions: list[tuple[int, int, str]] = []
    state = 1
    for path in paths:
        previous_state = 0
        for position, item in enumerate(path):
            current_state = state
            state += 1
            if position == 0:
                predicate = _observation_predicate(item["observation"])
                prior = initial.get(predicate)
                value = (current_state, item["expected_action"])
                if prior is not None and prior[1] != value[1]:
                    conflicts.append("identical initial observation requires different action")
                initial[predicate] = value
            else:
                transitions.append((previous_state, current_state, item["expected_action"]))
            previous_state = current_state

    update_lines = ["def update_task_belief(previous_progress, observation):"]
    for predicate, (current_state, _) in initial.items():
        update_lines.extend([f"    if previous_progress == 0 and {predicate}:", f"        return {current_state}"])
    for prior, current, _ in transitions:
        update_lines.extend([f"    if previous_progress == {prior}:", f"        return {current}"])
    update_lines.append("    return -1")

    action_by_state: dict[int, str] = {value[0]: value[1] for value in initial.values()}
    action_by_state.update({current: action for _, current, action in transitions})
    propose_lines = ["def propose_actions(task_family, observation, task_belief):"]
    for code, action in sorted(action_by_state.items()):
        propose_lines.extend([f"    if task_belief == {code}:", f"        return ({action!r},)"])
    propose_lines.append("    return ('STOP',)")

    completion_states = sorted(
        code for code, action in action_by_state.items() if action in {"COMPLETE", "INTERACT"}
    )
    completion_lines = ["def decide_completion(task_family, observation, task_belief):"]
    for code in completion_states:
        completion_lines.extend([f"    if task_belief == {code}:", "        return True"])
    completion_lines.append("    return False")
    remainder = [
        "def select_action(proposals, observation, completion_claim):",
        "    return proposals[-1]",
        "",
        "def detect_progress(previous_observation, observation):",
        "    return False",
        "",
        "def recover_target(last_seen_bearing):",
        "    if last_seen_bearing is not None and last_seen_bearing < 0:",
        "        return 'SCAN_LEFT'",
        "    return 'SCAN_RIGHT'",
    ]
    source = "\n\n".join(("\n".join(update_lines), "\n".join(propose_lines), "\n".join(remainder), "\n".join(completion_lines))) + "\n"
    return source, conflicts


def _surface_upper_bound(scenarios: list[dict[str, Any]], paths: list[list[dict[str, Any]]]) -> dict[str, Any]:
    source, conflicts = _surface_lookup_source(paths)
    if conflicts:
        return {
            "completion_upper_bound": 0,
            "scenario_count": len(scenarios),
            "contract_valid": False,
            "conflicts": conflicts,
        }
    with tempfile.TemporaryDirectory(prefix="gc2-surface-upper-") as temp_dir:
        candidate_path = Path(temp_dir) / "policy.py"
        candidate_path.write_text(source, encoding="utf-8")
        functions, complexity = load_candidate(candidate_path)
        outcomes = [
            run_episode(functions, scenario, index, CORRUPTIONS)
            for index, scenario in enumerate(scenarios)
        ]
    return {
        "completion_upper_bound": sum(item["goal_completion"] for item in outcomes),
        "scenario_count": len(outcomes),
        "contract_valid": True,
        "candidate_complexity_ast_nodes": complexity["ast_nodes"],
        "lookup_oracle_only": True,
        "scope": "FINITE_CONSUMED_SCENARIO_MEMORIZATION_NOT_A_POLICY_CLAIM",
    }


def _public_rgb_proxy(trace_path: Path, device_trace_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    present = [bool(row.get("detections")) for row in rows]
    absent_runs: list[int] = []
    run = 0
    for value in present:
        if value:
            if run:
                absent_runs.append(run)
                run = 0
        else:
            run += 1
    if run:
        absent_runs.append(run)
    confidences = [
        float(detection["confidence"])
        for row in rows
        for detection in row.get("detections", [])
    ]
    device_rows = [
        json.loads(line)
        for line in device_trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    neutral = [
        row for row in device_rows
        if row.get("branch_id") == "PRODUCTION_SEMANTIC_WITH_OBJECT_DETECTOR_TEMPORAL_GEOMETRY_NEUTRALIZED"
    ]
    latency = [float(row["timing"]["detector_total_ms"]) for row in neutral if row.get("timing")]
    return {
        "source_authority": "DEVELOPMENT_ONLY_BURNED_REAL_WORLD_RGB_PROCESSED_ON_ANDROID_NOT_PHONE_CAPTURE",
        "frame_count": len(rows),
        "frames_with_any_detection": sum(present),
        "frames_without_any_detection": len(rows) - sum(present),
        "longest_any-detection-absent_run_frames": max(absent_runs, default=0),
        "detection_confidence_median": statistics.median(confidences) if confidences else None,
        "detector_total_ms_median": statistics.median(latency) if latency else None,
        "detector_total_ms_max": max(latency, default=None),
        "limitations": [
            "no goal-target identity truth",
            "no phone-camera capture",
            "no calibrated target bearing",
            "no tracking-id loss and reacquisition truth",
            "no relative-nearness truth",
            "no capture-to-Goal-Copilot-observation timestamp",
        ],
    }


def audit(
    scenario_path: Path,
    winner_path: Path,
    public_rgb_trace: Path,
    device_trace: Path,
) -> dict[str, Any]:
    scenarios = json.loads(scenario_path.read_text(encoding="utf-8"))["scenarios"]
    functions, complexity = load_candidate(winner_path)
    full_outcomes = [
        run_episode(functions, scenario, index, CORRUPTIONS)
        for index, scenario in enumerate(scenarios)
    ]
    ablations: dict[str, Any] = {"ALL_MODERATE": _summarize(full_outcomes)}
    for removed in CORRUPTIONS:
        active = tuple(name for name in CORRUPTIONS if name != removed)
        outcomes = [
            run_episode(functions, scenario, index, active)
            for index, scenario in enumerate(scenarios)
        ]
        ablations[f"WITHOUT_{removed}"] = _summarize(outcomes)

    paths = _oracle_paths(scenarios)
    hidden_count = sum(len(item["steps"]) <= item["max_actions"] for item in scenarios)
    history_bound = _history_upper_bound(paths)
    surface_bound = _surface_upper_bound(scenarios, paths)
    grounding = _public_rgb_proxy(public_rgb_trace, device_trace)
    grounding["status"] = "REAL_PHONE_RGB_NOISE_GROUNDING_NOT_EVALUABLE"
    grounding["reason"] = (
        "Available device traces are public/burned RGB replay and lack the target identity, "
        "tracking, nearness, bearing, and timestamp contract needed to calibrate GC2 moderate."
    )

    result = {
        "schema_version": 1,
        "audit_id": "GOAL-COPILOT-2-OBSERVABILITY-AND-REALITY-AUDIT",
        "status": "COMPLETE_ZERO_MODEL_CONSUMED_DIAGNOSTIC",
        "model_calls": 0,
        "heldout_accessed": False,
        "inputs": {
            "scenario_sha256": _sha256(scenario_path),
            "winner_sha256": _sha256(winner_path),
            "winner_ast_nodes": complexity["ast_nodes"],
            "public_rgb_trace_sha256": _sha256(public_rgb_trace),
            "device_trace_sha256": _sha256(device_trace),
        },
        "failure_autopsy": _failure_autopsy(full_outcomes),
        "counterfactual_ablations": ablations,
        "observability_upper_bounds": {
            "A_hidden_state_oracle": {
                "completion_upper_bound": hidden_count,
                "scenario_count": len(scenarios),
            },
            "B_full_noisy_history_lookup": history_bound,
            "C_current_six_function_surface_lookup": surface_bound,
        },
        "real_rgb_grounding": grounding,
        "decision": "A_STOP_SYNTHETIC_MODERATE_AS_OPTIMIZATION_TARGET_AND_MOVE_TO_REAL_RGB_EVIDENCE",
        "decision_basis": [
            "hidden, full-history, and current-surface diagnostic upper bounds are all high on the consumed fixed set",
            "the searched winner still fails every moderate episode, so the result does not identify a representation bottleneck",
            "single-corruption ablations must be interpreted as simulator mechanism diagnostics only",
            "GC2 moderate cannot be calibrated to real phone RGB with the available evidence contract",
        ],
        "selected_route": (
            "A separately scoped real-phone RGB target-evidence capture/audit with target identity, detector/tracker "
            "outputs, camera/timing metadata, and no policy search"
        ),
        "next_execution_authorized": False,
        "next_required_contract": (
            "Freeze the phone-capture source, goal-target identity/truth, detector/tracker fields, timing mapping, "
            "privacy handling, bounded clip roster, and diagnostic-only claim ceiling before collection or replay."
        ),
        "not_authorized": [
            "GC2-C",
            "held-out opening",
            "additional Sky or model calls",
            "larger search budget",
            "representation ladder on consumed scenarios",
            "product or safety claim",
        ],
        "claim_ceiling": "consumed_symbolic_diagnostic_plus_non_phone_public_rgb_device_proxy_only",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--public-rgb-trace", type=Path, required=True)
    parser.add_argument("--device-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit result: {args.output}")
    result = audit(
        args.scenarios.resolve(),
        args.winner.resolve(),
        args.public_rgb_trace.resolve(),
        args.device_trace.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(result))
    print(json.dumps({"status": result["status"], "decision": result["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
