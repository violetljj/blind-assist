"""Run one sealed V0 current-frame episode over the frozen ABotN action graph."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.core import (
    Attribution,
    EpisodeState,
    Policy,
    State,
    apply_observation,
    stop_episode,
)


SCHEMA = "blindassist_abotn_v0_closed_loop_run_v0"
PUBLIC_SCHEMA = "blindassist_abotn_v0_action_graph_public_v0"
PIXEL_SCHEMA = "blindassist_abotn_webgl_action_graph_pixels_v0"
QUALIFICATION_SCHEMA = "blindassist_abotn_v0_action_graph_pixel_qualification_v0"
MAX_PROVIDER_OBSERVATIONS = Policy().max_instructions + Policy().max_consecutive_unreliable


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _audit_call_mechanics(call_dir: Path) -> dict[str, Any]:
    required = (
        call_dir / "brain-prompt.txt",
        call_dir / "brain-output-schema.json",
        call_dir / "brain-input.jpg",
        call_dir / "observation.json",
        call_dir / "completion.json",
    )
    if any(not path.is_file() for path in required):
        raise ValueError("provider call is missing a required immutable artifact")
    completed_item_types: list[str] = []
    external_action_events = 0
    for stdout_path in sorted(call_dir.glob("attempt-*-stdout.jsonl")):
        for line in stdout_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            event_type = str(event.get("type") or "")
            item_type = str(event.get("item", {}).get("type") or "")
            if event_type == "item.completed":
                completed_item_types.append(item_type)
                if item_type != "agent_message":
                    external_action_events += 1
            if "tool" in event_type.lower() or "command" in event_type.lower():
                external_action_events += 1
    completion = json.loads((call_dir / "completion.json").read_text(encoding="utf-8"))
    passed = (
        completion.get("status") == "RUN_SUCCESS"
        and completed_item_types
        and external_action_events == 0
    )
    return {
        "observation_sha256": _sha256(call_dir / "observation.json"),
        "prompt_sha256": _sha256(call_dir / "brain-prompt.txt"),
        "rendered_provider_input_sha256": _sha256(call_dir / "brain-input.jpg"),
        "completed_item_types": completed_item_types,
        "external_action_event_count": external_action_events,
        "provider_completion_status": completion.get("status"),
        "pass": passed,
    }


def _failure_class(state: EpisodeState, *, arrival: bool, action_exhausted: bool) -> str | None:
    if state.state == State.COMPLETE.value and arrival:
        return None
    if state.state == State.COMPLETE.value and not arrival:
        return "CONTROL_POLICY_BOTTLENECK_FALSE_ARRIVAL"
    if state.reliable_observation_count == 0 or state.consecutive_unreliable >= Policy().max_consecutive_unreliable:
        return "CURRENT_FRAME_GROUNDING_BOTTLENECK"
    if action_exhausted:
        return "CONTROL_POLICY_BOTTLENECK_ACTION_EXHAUSTED"
    return "CONTROL_POLICY_BOTTLENECK"


def run(
    *, public_graph_path: Path, private_truth_path: Path, freeze_path: Path,
    pixel_receipt_path: Path, qualification_path: Path, output_dir: Path,
    codex_exe: Path, grounding_dino: Path,
) -> dict[str, Any]:
    from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.cli import _append_event
    from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.provider_adapter import (
        ProviderAdapterError,
        ground_current_frame,
        preflight_provider,
    )
    from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.responsive_replay_runner import (
        _action_for_event,
    )

    public = json.loads(public_graph_path.read_text(encoding="utf-8"))
    pixels = json.loads(pixel_receipt_path.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if public.get("schema_version") != PUBLIC_SCHEMA or public.get("private_truth_access") is not False:
        raise ValueError("public action graph is not eligible")
    if pixels.get("schema_version") != PIXEL_SCHEMA or pixels.get("terminal") != "ABOTN_WEBGL_ACTION_GRAPH_PIXELS_PASS":
        raise ValueError("pixel cohort is not eligible")
    if (
        qualification.get("schema_version") != QUALIFICATION_SCHEMA
        or qualification.get("terminal") != "ABOTN_V0_ACTION_GRAPH_PIXELS_QUALIFIED_FOR_ONE_CLOSED_LOOP_RUN"
        or qualification.get("one_shot_provider_execution_authorized") is not True
    ):
        raise ValueError("one-shot provider execution is not qualified")
    expected_inputs = qualification["inputs"]
    if (
        expected_inputs["public_graph_sha256"] != _sha256(public_graph_path)
        or expected_inputs["private_truth_sha256"] != _sha256(private_truth_path)
        or expected_inputs["freeze_receipt_sha256"] != _sha256(freeze_path)
        or expected_inputs["pixel_receipt_sha256"] != _sha256(pixel_receipt_path)
    ):
        raise ValueError("qualified input binding drift")
    frames = {frame["observation_id"]: frame for frame in pixels["frames"]}
    nodes = {node["node_id"]: node for node in public["nodes"]}
    pixel_root = pixel_receipt_path.parent
    if set(frames) != set(nodes):
        raise ValueError("public graph/pixel node mismatch")
    for node_id, frame in frames.items():
        path = pixel_root / frame["path"]
        if not path.is_file() or _sha256(path) != frame["sha256"]:
            raise ValueError(f"qualified pixel changed: {node_id}")

    # Provider and machine preflight intentionally precede formal run creation.
    provider_lock = preflight_provider(codex_exe=codex_exe, model_dir=grounding_dino)
    if output_dir.exists():
        raise ValueError("formal one-shot output already exists; replay is forbidden")
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "provider-lock.json", provider_lock)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL_ONE_SHOT_STARTED",
        "public_graph_sha256": _sha256(public_graph_path),
        "private_truth_sha256": _sha256(private_truth_path),
        "freeze_receipt_sha256": _sha256(freeze_path),
        "pixel_receipt_sha256": _sha256(pixel_receipt_path),
        "qualification_sha256": _sha256(qualification_path),
        "provider_lock_sha256": _sha256(output_dir / "provider-lock.json"),
        "frozen_budget": {
            "episodes": 1,
            "provider_observations_maximum": MAX_PROVIDER_OBSERVATIONS,
            "brain_attempts_per_observation_maximum": 2,
        },
        "provider_private_truth_access": False,
        "retry_rule": "UNCHANGED_PROVIDER_INTERNAL_SCHEMA_RETRY_ONLY_MAX_TWO_ATTEMPTS",
        "rerun_rule": "NO_EPISODE_OR_OBSERVATION_RERUN_AFTER_FORMAL_START",
        "claim_ceiling": "UNOFFICIAL_RENDERER_ONE_TASK_CLOSED_LOOP_ENGINEERING_ONLY",
    }
    _atomic_json(output_dir / "run-manifest.json", manifest)
    journal = {
        "schema_version": "blindassist_abotn_v0_closed_loop_journal_v0",
        "status": "ACTIVE",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_calls_dispatched": 0,
        "provider_calls_completed": 0,
        "provider_calls_in_doubt": 0,
        "brain_attempts_dispatched": 0,
    }
    journal_path = output_dir / "provider-journal.json"
    _atomic_json(journal_path, journal)
    events_path = output_dir / "events.jsonl"
    episode_id = str(public["episode_id"])
    state = EpisodeState.start(
        episode_id=episode_id,
        location_id="abotn-scene-20260227163550",
        goal_name=str(public["goal_contract"]["target_name"]),
        started_at_ms=_now_ms(),
    )
    current = str(public["start_node_id"])
    trajectory: list[dict[str, Any]] = []
    call_audits: list[dict[str, Any]] = []
    action_exhausted = False
    _append_event(events_path, {
        "event_type": "EPISODE_STARTED",
        "episode_id": episode_id,
        "start_node_id": current,
        "started_at_ms": state.started_at_ms,
    })
    while state.state not in {State.COMPLETE.value, State.ABSTAIN.value}:
        if state.observation_count >= MAX_PROVIDER_OBSERVATIONS:
            stop = stop_episode(
                state,
                stopped_at_ms=_now_ms(),
                attribution=Attribution.INTERACTION_OR_CONTROL_BOTTLENECK,
                reason="FROZEN_PROVIDER_OBSERVATION_BUDGET_EXHAUSTED",
            )
            _append_event(events_path, stop)
            action_exhausted = True
            break
        node = nodes[current]
        frame = frames[current]
        observation_id = f"{episode_id}-closed-loop-o{state.observation_count + 1:03d}"
        call_dir = output_dir / "provider-calls" / observation_id
        journal.update({
            "status": "DISPATCHING",
            "active_observation_id": observation_id,
            "provider_calls_dispatched": journal["provider_calls_dispatched"] + 1,
            "provider_calls_in_doubt": 1,
        })
        _atomic_json(journal_path, journal)
        try:
            observation = ground_current_frame(
                provider_lock=provider_lock,
                call_dir=call_dir,
                episode_id=episode_id,
                goal_name=state.goal_name,
                image_path=pixel_root / frame["path"],
                frame_id=current,
                observation_id=observation_id,
                captured_at_ms=_now_ms(),
            )
        except ProviderAdapterError as error:
            completion_path = call_dir / "completion.json"
            completion = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else {}
            in_doubt = completion.get("status") == "IN_DOUBT" or not completion_path.is_file()
            journal.update({
                "status": "SEALED_PROVIDER_IN_DOUBT" if in_doubt else "SEALED_PROVIDER_FAILED",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "active_observation_id": observation_id,
                "provider_calls_in_doubt": 1 if in_doubt else 0,
                "failure": str(error),
            })
            _atomic_json(journal_path, journal)
            receipt = {
                "schema_version": SCHEMA,
                "closed_at_utc": datetime.now(timezone.utc).isoformat(),
                "terminal": "ABOTN_V0_CLOSED_LOOP_PROVIDER_IN_DOUBT" if in_doubt else "ABOTN_V0_CLOSED_LOOP_PROVIDER_FAILED",
                "provider_calls_dispatched": journal["provider_calls_dispatched"],
                "provider_calls_completed": journal["provider_calls_completed"],
                "provider_calls_in_doubt": journal["provider_calls_in_doubt"],
                "rerun_authorized": False,
                "claim_ceiling": manifest["claim_ceiling"],
            }
            _atomic_json(output_dir / "terminal-receipt.json", receipt)
            return receipt
        audit = _audit_call_mechanics(call_dir)
        if not audit["pass"]:
            raise ValueError(f"provider call mechanics audit failed: {audit}")
        call_audits.append({"observation_id": observation_id, "node_id": current, **audit})
        attempts = len(list(call_dir.glob("attempt-*-dispatch.json")))
        journal.update({
            "status": "ACTIVE",
            "active_observation_id": None,
            "provider_calls_completed": journal["provider_calls_completed"] + 1,
            "provider_calls_in_doubt": 0,
            "brain_attempts_dispatched": journal["brain_attempts_dispatched"] + attempts,
        })
        _atomic_json(journal_path, journal)
        result = apply_observation(state, observation, Policy())
        state = result.state
        event = dict(result.event) | {"environment_node_id": current}
        action = _action_for_event(event)
        event["environment_action"] = action
        _append_event(events_path, event)
        trajectory.append({
            "observation_id": observation_id,
            "node_id": current,
            "pose_index": node["pose_index"],
            "viewport_yaw_index": node["viewport_yaw_index"],
            "p0_status": event["p0_status"],
            "from_state": event["from_state"],
            "to_state": event["to_state"],
            "action": action,
        })
        if action is None:
            break
        target = node["actions"].get(action)
        _append_event(events_path, {
            "event_type": "ENVIRONMENT_ACTION_APPLIED",
            "episode_id": episode_id,
            "action": action,
            "from_node_id": current,
            "to_node_id": target,
            "rule": "FROZEN_ABOTN_ACTION_GRAPH_LOOKUP",
        })
        if target is None:
            stop = stop_episode(
                state,
                stopped_at_ms=_now_ms(),
                attribution=Attribution.INTERACTION_OR_CONTROL_BOTTLENECK,
                reason="FROZEN_ABOTN_ACTION_GRAPH_EDGE_UNAVAILABLE",
            )
            _append_event(events_path, stop)
            action_exhausted = True
            break
        current = str(target)

    # Evaluator-private truth is opened only after every provider call is terminal.
    private = json.loads(private_truth_path.read_text(encoding="utf-8"))
    private_by_node = {node["node_id"]: node for node in private["nodes"]}
    terminal_truth = private_by_node[current]
    private_literals = [
        "endpoint_xy",
        "distance_to_goal_m",
        str(private["endpoint_xy"][0]),
        str(private["endpoint_xy"][1]),
    ]
    literal_hits = []
    for call_dir in sorted((output_dir / "provider-calls").iterdir()):
        prompt = (call_dir / "brain-prompt.txt").read_text(encoding="utf-8")
        literal_hits.extend(value for value in private_literals if value in prompt)
    if literal_hits:
        raise ValueError(f"private arrival truth leaked to provider prompt: {sorted(set(literal_hits))}")
    arrival = bool(terminal_truth["arrival"])
    completed = state.state == State.COMPLETE.value and arrival
    false_arrival = state.state == State.COMPLETE.value and not arrival
    failure_class = _failure_class(state, arrival=arrival, action_exhausted=action_exhausted)
    evaluation = {
        "schema_version": "blindassist_abotn_v0_closed_loop_evaluation_v0",
        "episode_id": episode_id,
        "terminal_control_state": state.state,
        "terminal_node_id": current,
        "terminal_pose_index": nodes[current]["pose_index"],
        "terminal_viewport_yaw_index": nodes[current]["viewport_yaw_index"],
        "terminal_distance_to_goal_m": terminal_truth["distance_to_goal_m"],
        "terminal_metric_arrival": arrival,
        "episode_completion": completed,
        "false_arrival": false_arrival,
        "failure_class": failure_class,
        "selection_accuracy": "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING",
        "wrong_target_confirmation": "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING",
        "observation_count": state.observation_count,
        "reliable_observation_count": state.reliable_observation_count,
        "instruction_count": state.instruction_count,
        "rescan_count": state.rescan_count,
        "provider_private_truth_access": False,
    }
    _atomic_json(output_dir / "evaluation.json", evaluation)
    journal.update({
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_sha256": _sha256(output_dir / "evaluation.json"),
    })
    _atomic_json(journal_path, journal)
    receipt = {
        "schema_version": SCHEMA,
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": "ABOTN_V0_CLOSED_LOOP_ENGINEERING_RUN_COMPLETE",
        "provider": {
            "identity": "FROZEN_GROUNDING_DINO_TINY_PLUS_CODEX_TERRA_V0",
            "observation_calls": journal["provider_calls_completed"],
            "brain_attempts": journal["brain_attempts_dispatched"],
            "in_doubt": journal["provider_calls_in_doubt"],
        },
        "episode": evaluation,
        "action_state_trajectory": trajectory,
        "provider_call_audits": call_audits,
        "provider_private_truth_literal_hits": sorted(set(literal_hits)),
        "teacher_calls": 0,
        "baseline_episode_runs": 1,
        "rerun_authorized": False,
        "claim_ceiling": "UNOFFICIAL_RENDERER_SINGLE_TASK_CLOSED_LOOP_ENGINEERING_ONLY_NO_REAL_USER_PRODUCT_SAFETY_OR_SCIENTIFIC_CONFIRMATION",
        "next_action": (
            "STOP_AND_ATTRIBUTE_FIRST_FAILURE_WITHOUT_TUNING"
            if not completed
            else "REQUIRE_NEW_AUTHORIZATION_BEFORE_ANY_BROADER_COHORT"
        ),
        "artifact_sha256": {
            # The manifest is finalized after this receipt and therefore is
            # intentionally excluded from this non-circular artifact map.
            "provider-lock.json": _sha256(output_dir / "provider-lock.json"),
            "provider-journal.json": _sha256(journal_path),
            "events.jsonl": _sha256(events_path),
            "evaluation.json": _sha256(output_dir / "evaluation.json"),
        },
    }
    _atomic_json(output_dir / "terminal-receipt.json", receipt)
    manifest.update({
        "status": "SEALED_ENGINEERING_RUN_COMPLETE",
        "terminal_receipt_sha256": _sha256(output_dir / "terminal-receipt.json"),
    })
    _atomic_json(output_dir / "run-manifest.json", manifest)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-graph", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--pixel-receipt", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codex-exe", type=Path, default=Path("E:/codex-tools/bin/codex.exe"))
    parser.add_argument("--grounding-dino", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(
        public_graph_path=args.public_graph.resolve(),
        private_truth_path=args.private_truth.resolve(),
        freeze_path=args.freeze_receipt.resolve(),
        pixel_receipt_path=args.pixel_receipt.resolve(),
        qualification_path=args.qualification.resolve(),
        output_dir=args.output_dir.resolve(),
        codex_exe=args.codex_exe.resolve(),
        grounding_dino=args.grounding_dino.resolve(),
    )
    print(json.dumps({
        "terminal": receipt["terminal"],
        "provider": receipt.get("provider"),
        "episode": receipt.get("episode"),
        "next_action": receipt.get("next_action"),
    }, ensure_ascii=False, indent=2))
    return 0 if receipt["terminal"] == "ABOTN_V0_CLOSED_LOOP_ENGINEERING_RUN_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
