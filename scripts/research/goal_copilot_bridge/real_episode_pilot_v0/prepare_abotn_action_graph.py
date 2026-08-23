"""Freeze an ABotN renderer graph for the existing V0 current-frame actions."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PUBLIC_SCHEMA = "blindassist_abotn_v0_action_graph_public_v0"
PRIVATE_SCHEMA = "blindassist_abotn_v0_action_graph_private_v0"
FREEZE_SCHEMA = "blindassist_abotn_v0_action_graph_freeze_v0"
ACTIONS = ("TURN_LEFT", "TURN_RIGHT", "FORWARD", "RESCAN_HOLD")
VIEWPORT_YAWS = (-2, -1, 0, 1, 2)
VIEWPORT_YAW_STEP_DEG = 12.0
FORWARD_TRAVEL_M = 2.0
ARRIVE_THRESHOLD_M = 2.0
MAX_INSTRUCTIONS = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


DEFAULT_EPISODE_ID = "abotn-20260227163550-traj-0"


def _node_id(episode_id: str, pose_index: int, yaw_index: int) -> str:
    yaw_label = "z" if yaw_index == 0 else (f"l{yaw_index}" if yaw_index > 0 else f"r{abs(yaw_index)}")
    return f"{episode_id}-p{pose_index:03d}-yaw-{yaw_label}"


def _step_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    return math.hypot(float(second["x"]) - float(first["x"]), float(second["y"]) - float(first["y"]))


def _forward_targets(trajectory: Sequence[Mapping[str, Any]]) -> list[int | None]:
    targets: list[int | None] = []
    for index in range(len(trajectory)):
        travelled = 0.0
        target = None
        for other in range(index + 1, len(trajectory)):
            travelled += _step_distance(trajectory[other - 1], trajectory[other])
            if travelled >= FORWARD_TRAVEL_M:
                target = other
                break
        targets.append(target)
    return targets


def _shortest_arrival_steps(
    start: str, nodes: Mapping[str, Mapping[str, Any]], arrivals: set[str]
) -> int | None:
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        node_id, steps = queue.popleft()
        if node_id in arrivals:
            return steps
        for target in nodes[node_id]["actions"].values():
            if target not in seen:
                seen.add(target)
                queue.append((target, steps + 1))
    return None


def build_graph(
    annotation: Mapping[str, Any], *, episode_id: str = DEFAULT_EPISODE_ID,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trajectory = annotation.get("trajectory")
    extension = annotation.get("label", {}).get("extend", {})
    endpoint = extension.get("end_point")
    goal_name = str(extension.get("goal_label") or "").strip()
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        raise ValueError("ABotN trajectory is missing")
    if not isinstance(endpoint, list) or len(endpoint) != 2 or not goal_name:
        raise ValueError("ABotN metric endpoint or named goal is missing")
    forward_targets = _forward_targets(trajectory)
    public_nodes = []
    private_nodes = []
    for pose_index, pose in enumerate(trajectory):
        distance = math.hypot(float(pose["x"]) - float(endpoint[0]), float(pose["y"]) - float(endpoint[1]))
        for yaw_index in VIEWPORT_YAWS:
            node_id = _node_id(episode_id, pose_index, yaw_index)
            actions: dict[str, str] = {}
            if yaw_index < max(VIEWPORT_YAWS):
                actions["TURN_LEFT"] = _node_id(episode_id, pose_index, yaw_index + 1)
            if yaw_index > min(VIEWPORT_YAWS):
                actions["TURN_RIGHT"] = _node_id(episode_id, pose_index, yaw_index - 1)
            if forward_targets[pose_index] is not None:
                actions["FORWARD"] = _node_id(
                    episode_id, int(forward_targets[pose_index]), yaw_index
                )
            if pose_index + 1 < len(trajectory):
                actions["RESCAN_HOLD"] = _node_id(episode_id, pose_index + 1, yaw_index)
            public_nodes.append(
                {
                    "node_id": node_id,
                    "pose_index": pose_index,
                    "viewport_yaw_index": yaw_index,
                    "viewport_yaw_offset_deg": yaw_index * VIEWPORT_YAW_STEP_DEG,
                    "source_camera": {
                        "position": [float(pose["x"]), float(pose["y"]), float(pose["z"])],
                        "euler_radians": [
                            float(pose["roll"]),
                            float(pose["pitch"]),
                            float(pose["yaw"]) + math.radians(yaw_index * VIEWPORT_YAW_STEP_DEG),
                        ],
                    },
                    "actions": actions,
                    "rendered_frame_path": f"frames/frame-{len(public_nodes):03d}.png",
                }
            )
            private_nodes.append(
                {
                    "node_id": node_id,
                    "pose_index": pose_index,
                    "distance_to_goal_m": distance,
                    "arrival": distance < ARRIVE_THRESHOLD_M,
                }
            )
    public = {
        "schema_version": PUBLIC_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_id": episode_id,
        "goal_contract": {
            "goal_type": "NAMED_POI",
            "target_name": goal_name,
            "instruction": str(annotation.get("instruction") or f"前往{goal_name}"),
        },
        "action_contract": {
            "actions": list(ACTIONS),
            "viewport_yaw_indices": list(VIEWPORT_YAWS),
            "viewport_yaw_step_deg": VIEWPORT_YAW_STEP_DEG,
            "forward_source_path_travel_m": FORWARD_TRAVEL_M,
            "forward_rule": "FIRST_LATER_SOURCE_POSE_AT_OR_BEYOND_FIXED_PATH_TRAVEL",
            "rescan_rule": "NEXT_SOURCE_POSE_FRESH_REOBSERVATION",
            "maximum_instructions": MAX_INSTRUCTIONS,
            "provider_outcome_dependent": False,
        },
        "start_node_id": _node_id(episode_id, 0, 0),
        "nodes": public_nodes,
        "private_truth_access": False,
    }
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_id": public["episode_id"],
        "public_graph_sha256": _canonical_hash(public),
        "endpoint_xy": [float(endpoint[0]), float(endpoint[1])],
        "arrive_threshold_m": ARRIVE_THRESHOLD_M,
        "arrival_rule": "distance_to_goal_m < arrive_threshold_m",
        "nodes": private_nodes,
    }
    nodes_by_id = {node["node_id"]: node for node in public_nodes}
    arrivals = {node["node_id"] for node in private_nodes if node["arrival"]}
    shortest = _shortest_arrival_steps(public["start_node_id"], nodes_by_id, arrivals)
    serialized_public = json.dumps(public, ensure_ascii=False)
    forbidden = (
        "endpoint_xy",
        "distance_to_goal_m",
        json.dumps([float(endpoint[0]), float(endpoint[1])]),
    )
    hits = [value for value in forbidden if value in serialized_public]
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_graph_sha256": private["public_graph_sha256"],
        "private_truth_sha256": _canonical_hash(private),
        "pose_count": len(trajectory),
        "node_count": len(public_nodes),
        "edge_count_by_action": {
            action: sum(action in node["actions"] for node in public_nodes) for action in ACTIONS
        },
        "arrival_node_count": len(arrivals),
        "shortest_start_to_arrival_steps": shortest,
        "maximum_instructions": MAX_INSTRUCTIONS,
        "start_reaches_arrival_within_budget": shortest is not None and shortest <= MAX_INSTRUCTIONS,
        "public_private_literal_hits": hits,
        "provider_calls_before_freeze": 0,
        "teacher_calls_before_freeze": 0,
        "baseline_calls_before_freeze": 0,
        "render_calls_before_freeze": 0,
        "terminal": (
            "ABOTN_V0_ACTION_GRAPH_FROZEN_ELIGIBLE"
            if shortest is not None and shortest <= MAX_INSTRUCTIONS and not hits
            else "ABOTN_V0_ACTION_GRAPH_FROZEN_NOT_ELIGIBLE"
        ),
        "claim_ceiling": "DETERMINISTIC_RENDERER_ACTION_GRAPH_MECHANICS_ONLY",
    }
    return public, private, freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--selection-rule")
    parser.add_argument("--excluded-episode-id", action="append", default=[])
    args = parser.parse_args(argv)
    annotation_path = args.annotation.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError("output directory already exists; refusing replay")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    episode_id = args.episode_id or f"abotn-{annotation_path.parent.name}-{annotation_path.stem.replace('_', '-')}"
    public, private, freeze = build_graph(annotation, episode_id=episode_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "public-graph.json", public)
    _atomic_json(output_dir / "evaluator-private.json", private)
    freeze["inputs"] = {
        "annotation_path": str(annotation_path),
        "annotation_sha256": _sha256(annotation_path),
        "public_graph_file_sha256": _sha256(output_dir / "public-graph.json"),
        "private_truth_file_sha256": _sha256(output_dir / "evaluator-private.json"),
    }
    freeze["episode_selection"] = {
        "episode_id": episode_id,
        "rule": args.selection_rule,
        "excluded_episode_ids": args.excluded_episode_id,
        "selection_depended_on_pixels_or_provider_outcome": False,
    }
    _atomic_json(output_dir / "freeze-receipt.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    return 0 if freeze["terminal"] == "ABOTN_V0_ACTION_GRAPH_FROZEN_ELIGIBLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
