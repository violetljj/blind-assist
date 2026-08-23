"""Freeze and run one ABotN POI task through the official waypoint evaluator.

The provider boundary is deliberately narrower than the official Observation:
only the current true-front RGB image and public POI name are used. Evaluator
position, heading, target position, distance, maps, and history stay private.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.core import (
    EpisodeState,
    Policy,
    State,
    apply_observation,
)
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_abotn_v0_closed_loop import (
    HANDOFF_DISTANCE_LIMIT_M,
    HANDOFF_READY,
    _apply_termination_mode,
    _audit_call_mechanics,
)


SCHEMA = "blindassist_abotn_official_waypoint_handoff_v1"
FREEZE_SCHEMA = f"{SCHEMA}_freeze_v0"
RUN_SCHEMA = f"{SCHEMA}_run_v0"
OFFICIAL_COMMIT = "2a0aefb56f1e2d315bba924239e9e8ad9dca9d92"
MAX_STEPS = 15
FORWARD_STEP_M = 2.0
TURN_DEG = 12.0
PRIVATE_FIELD_NAMES = (
    "target_position",
    "distance_to_goal",
    "goal_world",
    "position",
    "rotation",
    "heading",
    "history_images",
    "history_poses",
    "occ_map",
    "height_map",
    "meta_data",
)


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


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return result.stdout.strip()


def _goal_name(annotation_path: Path) -> str:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    value = annotation.get("label", {}).get("extend", {}).get("goal_label")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("selected annotation has no public POI goal label")
    return value.strip()


def _action_prediction(action: str | None) -> tuple[np.ndarray, np.ndarray, bool]:
    """Map the frozen V0 control action into the official local waypoint API."""

    if action == "FORWARD":
        waypoint = np.array([[FORWARD_STEP_M, 0.0]], dtype=np.float32)
        direction = np.array([[1.0, 0.0]], dtype=np.float32)
        return waypoint, direction, False
    if action in {"TURN_LEFT", "TURN_RIGHT"}:
        angle = math.radians(TURN_DEG if action == "TURN_LEFT" else -TURN_DEG)
        waypoint = np.array([[0.0, 0.0]], dtype=np.float32)
        direction = np.array([[math.cos(angle), math.sin(angle)]], dtype=np.float32)
        return waypoint, direction, False
    if action == "RESCAN_HOLD":
        angle = math.radians(TURN_DEG)
        return (
            np.array([[0.0, 0.0]], dtype=np.float32),
            np.array([[math.cos(angle), math.sin(angle)]], dtype=np.float32),
            False,
        )
    if action is None:
        # The official protocol exposes only an `arrive` stop bit. Here it is
        # transport for ABSTAIN/HANDOFF stop, never a completion claim.
        return (
            np.array([[0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
            True,
        )
    raise ValueError(f"unsupported frozen control action: {action}")


def _action_for_event(event: Mapping[str, Any]) -> str | None:
    if event.get("to_state") in {State.COMPLETE.value, State.ABSTAIN.value, HANDOFF_READY}:
        return None
    if event.get("to_state") in {State.RESCAN.value, State.ARRIVAL_CONFIRM.value}:
        return "RESCAN_HOLD"
    candidate = event.get("candidate")
    if event.get("to_state") != State.ADVANCE_AND_REOBSERVE.value or not isinstance(candidate, Mapping):
        raise ValueError("non-terminal control event cannot map to an official waypoint")
    center_x = float(candidate["center_x"])
    policy = Policy()
    if center_x < policy.center_left:
        return "TURN_LEFT"
    if center_x > policy.center_right:
        return "TURN_RIGHT"
    return "FORWARD"


class _CanonicalViewRenderer:
    """Repair the pinned renderer/evaluator current-view ordering mismatch."""

    def __init__(self, delegate: Any):
        self.delegate = delegate

    def render_at_pose(self, *args: Any, **kwargs: Any) -> list[Any]:
        images = list(self.delegate.render_at_pose(*args, **kwargs))
        if len(images) != 3:
            raise ValueError("official POI run requires exactly three current views")
        # GaussianRenderer: left,right,front. Evaluator: left,front,right.
        return [images[0], images[2], images[1]]


def freeze(
    *, repo_root: Path, official_repo: Path, annotations_root: Path,
    maps_root: Path, scene_id: str, task_id: str, point_cloud: Path,
    provider_lock_path: Path, output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError("freeze output already exists")
    if _git_value(official_repo, "rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise ValueError("pinned official evaluator commit drift")
    annotation = annotations_root / scene_id / f"{task_id}.json"
    if not annotation.is_file() or not point_cloud.is_file() or not provider_lock_path.is_file():
        raise ValueError("a frozen input is missing")
    goal_name = _goal_name(annotation)
    occ_map = maps_root / scene_id / "map" / "occ_map.png"
    height_map = maps_root / scene_id / "map" / "occ_map_height.tiff"
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": "ABOTN_OFFICIAL_WAYPOINT_HANDOFF_V1_FROZEN",
        "selection": {
            "scene_id": scene_id,
            "task_id": task_id,
            "episode_id": f"abotn-{scene_id}-{task_id.replace('_', '-')}",
            "goal_name": goal_name,
            "rule": "EXPLICIT_FRESH_CACHED_SCENE_TASK_AFTER_TRAJ_0_AND_TRAJ_1_CONSUMED",
        },
        "inputs": {
            "annotation_path": str(annotation.resolve()),
            "annotation_sha256": _sha256(annotation),
            "point_cloud_path": str(point_cloud.resolve()),
            "point_cloud_sha256": _sha256(point_cloud),
            "provider_lock_path": str(provider_lock_path.resolve()),
            "provider_lock_sha256": _sha256(provider_lock_path),
            "official_repo_path": str(official_repo.resolve()),
            "official_repo_commit": OFFICIAL_COMMIT,
            "blindassist_repo_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        },
        "execution": {
            "official_evaluator": "PoiGoalEvaluator",
            "official_arrive_threshold_m": 2.0,
            "max_steps": MAX_STEPS,
            "forward_step_m": FORWARD_STEP_M,
            "turn_degrees": TURN_DEG,
            "rescan_motion": "IN_PLACE_LEFT_SWEEP_ONE_FROZEN_TURN_STEP",
            "current_views": ["left", "front", "right"],
            "provider_view": "front",
            "renderer_retries": 0,
            "provider_schema_attempts_per_observation_maximum": 2,
            "teacher_calls": 0,
            "episode_reruns": 0,
            "completion_authority_receipt_present": False,
        },
        "firewall": {
            "provider_visible": ["current_true_front_rgb", "poi_name"],
            "provider_private_truth_access": False,
            "withheld_fields": list(PRIVATE_FIELD_NAMES),
            "history_enabled": False,
            "map_enabled_for_provider": False,
        },
        "substrate": {
            "bounded_yaw_graph_removed": True,
            "waypoint_execution": "OFFICIAL_CONTINUOUS_LOCAL_WAYPOINT_TRANSFORM",
            "official_renderer_view_order_adapter": "LEFT_RIGHT_FRONT_TO_LEFT_FRONT_RIGHT",
            "occ_map_available": occ_map.is_file(),
            "height_map_available": height_map.is_file(),
            "collision_claim": "NOT_EVALUABLE_MAP_NOT_CACHED" if not occ_map.is_file() else "OFFICIAL_EVALUATOR",
        },
        "termination_contract": {
            "mode": "HANDOFF_V1",
            "handoff_distance_limit_m": HANDOFF_DISTANCE_LIMIT_M,
            "claim_boundary": "HANDOFF_READY_IS_NOT_ARRIVED_OR_COMPLETED",
            "official_arrive_bit_usage": "STOP_TRANSPORT_ONLY_NO_COMPLETION_CLAIM",
        },
        "claim_ceiling": "ONE_FRESH_OFFICIAL_RENDERER_WAYPOINT_TASK_ENGINEERING_ONLY",
        "rerun_rule": "NO_TASK_OR_OBSERVATION_RERUN_AFTER_FORMAL_START",
    }
    _atomic_json(output_path, payload)
    return payload


def _load_official(official_repo: Path) -> dict[str, Any]:
    sys.path.insert(0, str(official_repo))
    from abotn_evaluator.interface.poi_goal import BasePoiGoalAgent
    from abotn_evaluator.interface.point_goal import WaypointPrediction
    from abotn_evaluator.poi_goal.evaluator import PoiGoalEvalConfig, PoiGoalEvaluator
    from abotn_evaluator.render_client import GaussianRenderer
    from abotn_evaluator.scene import GaussianScene
    return {
        "BasePoiGoalAgent": BasePoiGoalAgent,
        "WaypointPrediction": WaypointPrediction,
        "PoiGoalEvalConfig": PoiGoalEvalConfig,
        "PoiGoalEvaluator": PoiGoalEvaluator,
        "GaussianRenderer": GaussianRenderer,
        "GaussianScene": GaussianScene,
    }


def _agent_class(official: Mapping[str, Any]) -> type:
    from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.provider_adapter import (
        ground_current_frame,
    )

    BasePoiGoalAgent = official["BasePoiGoalAgent"]
    WaypointPrediction = official["WaypointPrediction"]

    class CurrentFrameWaypointAgent(BasePoiGoalAgent):
        def __init__(self, *, provider_lock: Mapping[str, Any], output_dir: Path, episode_id: str):
            self.provider_lock = provider_lock
            self.output_dir = output_dir
            self.episode_id = episode_id
            self.events_path = output_dir / "events.jsonl"
            self.journal_path = output_dir / "provider-journal.json"
            self.state: EpisodeState | None = None
            self.trajectory: list[dict[str, Any]] = []
            self.call_audits: list[dict[str, Any]] = []
            self.goal_name = ""

        def reset(self) -> None:
            self.state = None
            self.trajectory = []
            self.call_audits = []

        def predict(self, observation: Any) -> Any:
            # Deliberate public projection. No other Observation field is read.
            goal_name = str(observation.poi_name)
            step_count = int(observation.step_count)
            front = observation.images["front"]
            if self.state is None:
                self.goal_name = goal_name
                self.state = EpisodeState.start(
                    episode_id=self.episode_id,
                    location_id=f"abotn-scene-{self.episode_id}",
                    goal_name=goal_name,
                    started_at_ms=_now_ms(),
                )
                _append_jsonl(self.events_path, {
                    "event_type": "EPISODE_STARTED",
                    "episode_id": self.episode_id,
                    "goal_name": goal_name,
                    "started_at_ms": self.state.started_at_ms,
                })
            if goal_name != self.goal_name:
                raise ValueError("public POI name changed during frozen task")
            frame_id = f"official-step-{step_count:03d}-front"
            observation_id = f"{self.episode_id}-o{step_count + 1:03d}"
            frame_path = self.output_dir / "current-front-frames" / f"{frame_id}.png"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            front.save(frame_path)
            journal = json.loads(self.journal_path.read_text(encoding="utf-8"))
            journal.update({
                "status": "DISPATCHING",
                "active_observation_id": observation_id,
                "provider_calls_dispatched": journal["provider_calls_dispatched"] + 1,
                "provider_calls_in_doubt": 1,
            })
            _atomic_json(self.journal_path, journal)
            call_dir = self.output_dir / "provider-calls" / observation_id
            provider_observation = ground_current_frame(
                provider_lock=self.provider_lock,
                call_dir=call_dir,
                episode_id=self.episode_id,
                goal_name=goal_name,
                image_path=frame_path,
                frame_id=frame_id,
                observation_id=observation_id,
                captured_at_ms=_now_ms(),
            )
            audit = _audit_call_mechanics(call_dir)
            if not audit["pass"]:
                raise ValueError(f"provider call mechanics audit failed: {audit}")
            attempts = len(list(call_dir.glob("attempt-*-dispatch.json")))
            journal.update({
                "status": "ACTIVE",
                "active_observation_id": None,
                "provider_calls_completed": journal["provider_calls_completed"] + 1,
                "provider_calls_in_doubt": 0,
                "brain_attempts_dispatched": journal["brain_attempts_dispatched"] + attempts,
            })
            _atomic_json(self.journal_path, journal)
            self.call_audits.append({"observation_id": observation_id, **audit})
            result = apply_observation(self.state, provider_observation, Policy())
            self.state, event = _apply_termination_mode(
                result.state, result.event, termination_mode="HANDOFF_V1"
            )
            action = _action_for_event(event)
            event = dict(event) | {
                "environment_action": action,
                "official_step_count": step_count,
                "official_stop_bit_is_completion_claim": False,
            }
            _append_jsonl(self.events_path, event)
            self.trajectory.append({
                "observation_id": observation_id,
                "official_step_count": step_count,
                "p0_status": event["p0_status"],
                "from_state": event["from_state"],
                "to_state": event["to_state"],
                "action": action,
            })
            waypoint, directions, stop_request = _action_prediction(action)
            return WaypointPrediction(
                waypoint=waypoint,
                directions=directions,
                arrive=stop_request,
                extra={"control_state": self.state.state, "completion_claim": False},
            )

    return CurrentFrameWaypointAgent


def run(*, freeze_path: Path, output_dir: Path, render_url: str) -> dict[str, Any]:
    from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.provider_adapter import preflight_provider

    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("official waypoint freeze schema mismatch")
    if output_dir.exists():
        raise ValueError("formal output already exists; rerun is forbidden")
    inputs = frozen["inputs"]
    annotation = Path(inputs["annotation_path"])
    point_cloud = Path(inputs["point_cloud_path"])
    provider_lock_path = Path(inputs["provider_lock_path"])
    official_repo = Path(inputs["official_repo_path"])
    for path, expected in (
        (annotation, inputs["annotation_sha256"]),
        (point_cloud, inputs["point_cloud_sha256"]),
        (provider_lock_path, inputs["provider_lock_sha256"]),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen input drift: {path}")
    if _git_value(official_repo, "rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise ValueError("official evaluator commit drift")
    provider_lock = json.loads(provider_lock_path.read_text(encoding="utf-8"))
    live_lock = preflight_provider(
        codex_exe=Path(provider_lock["codex"]["executable"]),
        model_dir=Path(provider_lock["grounding_dino"]["model_dir"]),
    )
    if live_lock != provider_lock:
        raise ValueError("live provider identity differs from frozen provider")

    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "provider-lock.json", provider_lock)
    _atomic_json(output_dir / "run-manifest.json", {
        "schema_version": RUN_SCHEMA,
        "status": "FORMAL_ONE_SHOT_STARTED",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": _sha256(freeze_path),
        "render_url": render_url,
        "provider_private_truth_access": False,
        "rerun_rule": frozen["rerun_rule"],
        "claim_ceiling": frozen["claim_ceiling"],
    })
    _atomic_json(output_dir / "provider-journal.json", {
        "schema_version": f"{SCHEMA}_provider_journal_v0",
        "status": "ACTIVE",
        "provider_calls_dispatched": 0,
        "provider_calls_completed": 0,
        "provider_calls_in_doubt": 0,
        "brain_attempts_dispatched": 0,
    })

    official = _load_official(official_repo)
    scene = official["GaussianScene"](
        local_data_path=str(annotation.parents[1]),
        local_map_path=str(point_cloud.parents[3]),
    )
    selected = scene.get_episode_by_id(frozen["selection"]["scene_id"])
    if selected is None:
        raise ValueError("frozen scene missing from official scene loader")
    selected.tasks = [task for task in selected.tasks if task.task_id == frozen["selection"]["task_id"]]
    if len(selected.tasks) != 1:
        raise ValueError("frozen task missing or duplicated")
    scene.episodes = [selected]
    delegate = official["GaussianRenderer"](
        render_url=render_url,
        num_views=3,
        max_retries=0,
    )
    renderer = _CanonicalViewRenderer(delegate)
    config = official["PoiGoalEvalConfig"](
        render_url=render_url,
        max_steps=MAX_STEPS,
        provide_history=False,
        provide_occ_map=False,
        provide_height_map=False,
        save_render_images=True,
    )
    Agent = _agent_class(official)
    agent = Agent(
        provider_lock=provider_lock,
        output_dir=output_dir,
        episode_id=frozen["selection"]["episode_id"],
    )
    evaluator = official["PoiGoalEvaluator"](
        scene=scene,
        renderer=renderer,
        config=config,
        output_dir=str(output_dir / "official-evaluator"),
    )
    try:
        results = evaluator.evaluate(agent)
    except Exception as error:
        journal = json.loads((output_dir / "provider-journal.json").read_text(encoding="utf-8"))
        # A missing terminal completion means the dispatched call remains in doubt.
        active = journal.get("active_observation_id")
        completion = output_dir / "provider-calls" / str(active) / "completion.json" if active else None
        in_doubt = bool(active and (completion is None or not completion.is_file()))
        journal.update({
            "status": "SEALED_PROVIDER_IN_DOUBT" if in_doubt else "SEALED_RUN_FAILED",
            "provider_calls_in_doubt": 1 if in_doubt else 0,
            "failure_type": type(error).__name__,
            "failure": str(error),
        })
        _atomic_json(output_dir / "provider-journal.json", journal)
        receipt = {
            "schema_version": RUN_SCHEMA,
            "terminal": "ABOTN_OFFICIAL_WAYPOINT_PROVIDER_IN_DOUBT" if in_doubt else "ABOTN_OFFICIAL_WAYPOINT_RUN_FAILED",
            "rerun_authorized": False,
            "failure_type": type(error).__name__,
            "failure": str(error),
            "claim_ceiling": frozen["claim_ceiling"],
        }
        _atomic_json(output_dir / "terminal-receipt.json", receipt)
        return receipt
    if len(results) != 1:
        raise ValueError("official evaluator did not return exactly one frozen task result")

    private_hits: list[str] = []
    for call_dir in sorted((output_dir / "provider-calls").iterdir()):
        prompt = (call_dir / "brain-prompt.txt").read_text(encoding="utf-8")
        private_hits.extend(name for name in PRIVATE_FIELD_NAMES if name in prompt)
    if private_hits:
        raise ValueError(f"evaluator-private field name leaked to provider prompt: {sorted(set(private_hits))}")
    result = results[0]
    state = agent.state
    if state is None:
        raise ValueError("official evaluator returned without a provider observation")
    official_success = bool(result.get("success"))
    handoff_ready = state.state == HANDOFF_READY
    distance = float(result["distance_to_goal"])
    if handoff_ready and distance <= HANDOFF_DISTANCE_LIMIT_M:
        failure_class = None
    elif handoff_ready:
        failure_class = "CONTROL_POLICY_BOTTLENECK_PREMATURE_HANDOFF"
    elif official_success:
        failure_class = "METRIC_ARRIVAL_WITHOUT_VISUAL_HANDOFF"
    elif state.reliable_observation_count == 0:
        failure_class = "CURRENT_FRAME_GROUNDING_BOTTLENECK"
    else:
        failure_class = "CONTROL_POLICY_BOTTLENECK"
    evaluation = {
        "schema_version": f"{SCHEMA}_evaluation_v0",
        "episode_id": frozen["selection"]["episode_id"],
        "official_result": result,
        "terminal_control_state": state.state,
        "terminal_metric_arrival": official_success,
        "handoff_ready": handoff_ready,
        "handoff_within_frozen_distance_limit": handoff_ready and distance <= HANDOFF_DISTANCE_LIMIT_M,
        "episode_completion": False,
        "completion_authority_receipt_present": False,
        "provider_private_truth_access": False,
        "private_field_name_hits": sorted(set(private_hits)),
        "failure_class": failure_class,
        "selection_accuracy": "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING",
        "collision_outcome": frozen["substrate"]["collision_claim"],
    }
    _atomic_json(output_dir / "evaluation.json", evaluation)
    journal = json.loads((output_dir / "provider-journal.json").read_text(encoding="utf-8"))
    journal.update({"status": "COMPLETED", "active_observation_id": None})
    _atomic_json(output_dir / "provider-journal.json", journal)
    receipt = {
        "schema_version": RUN_SCHEMA,
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": "ABOTN_OFFICIAL_WAYPOINT_HANDOFF_V1_ENGINEERING_RUN_COMPLETE",
        "provider": {
            "observation_calls": journal["provider_calls_completed"],
            "brain_attempts": journal["brain_attempts_dispatched"],
            "in_doubt": journal["provider_calls_in_doubt"],
        },
        "episode": evaluation,
        "action_state_trajectory": agent.trajectory,
        "provider_call_audits": agent.call_audits,
        "teacher_calls": 0,
        "baseline_episode_runs": 1,
        "rerun_authorized": False,
        "claim_ceiling": frozen["claim_ceiling"],
        "next_action": "STOP_AND_ATTRIBUTE_FIRST_FAILURE_WITHOUT_TUNING",
    }
    _atomic_json(output_dir / "terminal-receipt.json", receipt)
    manifest = json.loads((output_dir / "run-manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "status": "SEALED_ENGINEERING_RUN_COMPLETE",
        "terminal_receipt_sha256": _sha256(output_dir / "terminal-receipt.json"),
    })
    _atomic_json(output_dir / "run-manifest.json", manifest)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--repo-root", type=Path, required=True)
    freeze_parser.add_argument("--official-repo", type=Path, required=True)
    freeze_parser.add_argument("--annotations-root", type=Path, required=True)
    freeze_parser.add_argument("--maps-root", type=Path, required=True)
    freeze_parser.add_argument("--scene-id", required=True)
    freeze_parser.add_argument("--task-id", required=True)
    freeze_parser.add_argument("--point-cloud", type=Path, required=True)
    freeze_parser.add_argument("--provider-lock", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--freeze", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--render-url", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = freeze(
            repo_root=args.repo_root.resolve(),
            official_repo=args.official_repo.resolve(),
            annotations_root=args.annotations_root.resolve(),
            maps_root=args.maps_root.resolve(),
            scene_id=args.scene_id,
            task_id=args.task_id,
            point_cloud=args.point_cloud.resolve(),
            provider_lock_path=args.provider_lock.resolve(),
            output_path=args.output.resolve(),
        )
        print(json.dumps({"terminal": payload["terminal"], "selection": payload["selection"]}, ensure_ascii=False, indent=2))
        return 0
    receipt = run(
        freeze_path=args.freeze.resolve(),
        output_dir=args.output_dir.resolve(),
        render_url=args.render_url,
    )
    print(json.dumps({
        "terminal": receipt["terminal"],
        "provider": receipt.get("provider"),
        "episode": receipt.get("episode"),
        "next_action": receipt.get("next_action"),
    }, ensure_ascii=False, indent=2))
    return 0 if receipt["terminal"] == "ABOTN_OFFICIAL_WAYPOINT_HANDOFF_V1_ENGINEERING_RUN_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
