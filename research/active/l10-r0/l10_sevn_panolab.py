#!/usr/bin/env python3
"""Bridge SEVN 1.0 metadata into the L10-PanoLab one-step action contract.

Materialization reads the original SEVN coordinate/label HDF5 files and graph
pickle, then emits a dependency-free public cohort plus separated evaluator
truth. Replay uses only those JSON files.  Image payload is deliberately not
required: the first integration gate proves source identity, exact address-door
annotations, graph actions, and truth separation before paying the 1.86 GB
low-resolution image download.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from l10_panolab import ACTIONS, StepResult, require, sha256_file, utc_now


PANO_WIDTH_PX = 224.0
VIEWPORT_WIDTH_PX = 84.0
VIEWPORT_FOV_DEGREES = 360.0 * VIEWPORT_WIDTH_PX / PANO_WIDTH_PX
PAN_DEGREES = 67.5
PAN_START_OFFSET_DEGREES = 90.0
HEADING_QUANTUM_DEGREES = 22.5
SCENARIOS = (
    "PAN_LEFT_ADDRESS_DOOR_RECOVERY",
    "PAN_RIGHT_ADDRESS_DOOR_RECOVERY",
    "APPROACH_ADDRESS_DOOR_RECOVERY",
)
BINDING_STATES = {"CORRECT_UNIQUE", "WRONG_UNIQUE", "SET_VALUED", "NOT_VISIBLE"}
SOURCE_SPECS = {
    "coord.hdf5": {"bytes": 419832, "md5": "9d2d6f17aa31c9cb8309195cf2da00f9"},
    "label.hdf5": {"bytes": 351434, "md5": "36e24d0ee0b2f351282f3b84aa5d5d56"},
    "graph.pkl": {"bytes": 2793330, "md5": "f68be888e8f3b964544678070689b777"},
}


def wrap360(value: float) -> float:
    return value % 360.0


def signed_delta_degrees(a: float, b: float) -> float:
    """Return the shortest signed rotation from a to b."""
    return (b - a + 180.0) % 360.0 - 180.0


def quantize_heading(value: float) -> float:
    return wrap360(round(value / HEADING_QUANTUM_DEGREES) * HEADING_QUANTUM_DEGREES)


def label_heading_degrees(x_min: float, x_max: float, panorama_angle_degrees: float) -> float:
    center = (x_min + x_max) / 2.0
    label_relative = (PANO_WIDTH_PX - center) * 360.0 / PANO_WIDTH_PX - 180.0
    return wrap360(label_relative + panorama_angle_degrees)


def label_half_width_degrees(x_min: float, x_max: float) -> float:
    return max(0.0, x_max - x_min) * 180.0 / PANO_WIDTH_PX


def label_fully_visible(
    x_min: float,
    x_max: float,
    panorama_angle_degrees: float,
    viewport_headings: list[float],
) -> bool:
    center = label_heading_degrees(x_min, x_max, panorama_angle_degrees)
    half_width = label_half_width_degrees(x_min, x_max)
    half_view = VIEWPORT_FOV_DEGREES / 2.0
    return any(
        abs(signed_delta_degrees(viewport, center)) + half_width < half_view
        for viewport in viewport_headings
    )


def canonical_text(value: Any) -> str | None:
    try:
        import pandas as pd  # type: ignore

        if pd.isna(value):
            return None
    except ImportError:
        if value is None:
            return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def md5_file(path: Path) -> str:
    import hashlib

    digest = hashlib.md5()  # noqa: S324 - verifies the upstream Zenodo receipt only.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_upstream_files(metadata_dir: Path) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for name, expected in SOURCE_SPECS.items():
        path = metadata_dir / name
        require(path.is_file(), f"missing SEVN upstream file: {path}")
        actual_bytes = path.stat().st_size
        actual_md5 = md5_file(path)
        require(actual_bytes == expected["bytes"], f"{name}: unexpected byte size")
        require(actual_md5 == expected["md5"], f"{name}: Zenodo MD5 mismatch")
        receipts[name] = {
            "bytes": actual_bytes,
            "md5": actual_md5,
            "zenodo_content_url": f"https://zenodo.org/api/records/3526490/files/{name}/content",
        }
    return receipts


def address_key(row: Any) -> tuple[str, str] | None:
    number = canonical_text(row["house_number"])
    street = canonical_text(row["street_name"])
    if number is None or street is None:
        return None
    return street, number


def row_public(row: Any) -> dict[str, Any]:
    return {
        "frame_id": int(row.name),
        "house_number": canonical_text(row["house_number"]),
        "street_name": canonical_text(row["street_name"]),
    }


def heading_between(coords: Any, start_frame: int, end_frame: int) -> float:
    start = coords.loc[start_frame]
    end = coords.loc[end_frame]
    return wrap360(math.degrees(math.atan2(float(end.y - start.y), float(end.x - start.x))))


def distance_xy_m(coords: Any, start_frame: int, end_frame: int) -> float:
    start = coords.loc[start_frame]
    end = coords.loc[end_frame]
    return math.hypot(float(end.x - start.x), float(end.y - start.y))


def truth_for_view(
    labels_by_frame: dict[int, list[dict[str, Any]]],
    coords: Any,
    frame_id: int,
    viewport_headings: list[float],
    target_address: tuple[str, str],
) -> dict[str, Any]:
    panorama_angle = float(coords.loc[frame_id].angle)
    visible_doors = []
    visible_target_doors = []
    target_text_visible = False
    for row in labels_by_frame.get(frame_id, []):
        if not label_fully_visible(
            float(row["x_min"]),
            float(row["x_max"]),
            panorama_angle,
            viewport_headings,
        ):
            continue
        row_address = (canonical_text(row["street_name"]), canonical_text(row["house_number"]))
        if row["obj_type"] == "door":
            visible_doors.append(row)
            if row_address == target_address:
                visible_target_doors.append(row)
        elif row["obj_type"] == "house_number" and row_address == target_address:
            target_text_visible = True

    if len(visible_target_doors) == 1:
        binding_state = "CORRECT_UNIQUE"
    elif len(visible_target_doors) > 1:
        binding_state = "SET_VALUED"
    elif len(visible_doors) == 1:
        binding_state = "WRONG_UNIQUE"
    else:
        binding_state = "NOT_VISIBLE"
    return {
        "binding_state": binding_state,
        "target_visible": len(visible_target_doors) > 0,
        "target_match_count": len(visible_target_doors),
        "visible_door_count": len(visible_doors),
        "target_house_number_visible": target_text_visible,
    }


def observation(
    observation_id: str,
    coords: Any,
    frame_id: int,
    viewport_headings: list[float],
    image_payload_available: bool,
) -> dict[str, Any]:
    row = coords.loc[frame_id]
    return {
        "observation_id": observation_id,
        "frame_id": frame_id,
        "camera_pose": {
            "local_xyz_m": [round(float(row.x), 6), round(float(row.y), 6), round(float(row.z), 6)],
            "panorama_angle_degrees": round(wrap360(float(row.angle)), 6),
            "pose_authority": "SEVN_ORB_SLAM2_LOCAL_COORDINATES",
        },
        "viewport_headings_degrees": [round(wrap360(value), 6) for value in viewport_headings],
        "horizontal_fov_degrees": VIEWPORT_FOV_DEGREES,
        "image_asset": {
            "container": "images.hdf5",
            "dataset": "images",
            "frame_lookup_dataset": "frames",
            "frame_id": frame_id,
            "payload_available": image_payload_available,
        },
    }


def public_edge(
    action: str,
    start_id: str,
    destination_id: str,
    executed: bool,
    movement_distance_m: float,
) -> dict[str, Any]:
    return {
        "action": action,
        "to_observation_id": destination_id,
        "action_executed": executed,
        "movement_distance_m": round(movement_distance_m, 6),
        "provider_transition": "SEVN_GRAPH_EDGE" if movement_distance_m > 0 else "SEVN_SAME_PANORAMA_VIEW",
        "unavailable_reason": None if executed else (
            "SEVN_HAS_NO_LATERAL_SIDESTEP_SEMANTICS" if action.startswith("SIDESTEP")
            else "ACTION_NOT_AVAILABLE_FOR_THIS_FROZEN_EPISODE"
        ),
        "before_observation_id": start_id,
    }


def dataframe_rows_by_frame(labels: Any) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    for frame_id, row in labels.iterrows():
        rows.setdefault(int(frame_id), []).append({
            "obj_type": canonical_text(row.obj_type),
            "house_number": canonical_text(row.house_number),
            "street_name": canonical_text(row.street_name),
            "is_goal": bool(row.is_goal),
            "x_min": int(row.x_min),
            "x_max": int(row.x_max),
            "y_min": int(row.y_min),
            "y_max": int(row.y_max),
        })
    return rows


def candidate_goals(labels: Any, coords: Any, graph: Any) -> list[Any]:
    goals = labels[(labels.obj_type == "door") & labels.is_goal].copy()
    goals = goals[goals.house_number.notna() & goals.street_name.notna()]
    goals = goals[goals.index.isin(coords.index) & goals.index.isin(graph.nodes)]
    goals["_frame"] = goals.index.astype(int)
    goals["_street"] = goals.street_name.astype(str)
    goals["_number"] = goals.house_number.astype(str)
    return [row for _, row in goals.sort_values(["_street", "_number", "_frame", "x_min"]).iterrows()]


def build_episode(
    sequence: int,
    scenario: str,
    goal: Any,
    start_frame: int,
    start_heading: float,
    goal_frame: int,
    goal_heading: float,
    coords: Any,
    labels_by_frame: dict[int, list[dict[str, Any]]],
    image_payload_available: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    episode_id = f"SEVN{sequence:03d}"
    target_address = address_key(goal)
    require(target_address is not None, f"{episode_id}: target address missing")
    start_id = f"{episode_id}_START"
    observation_specs: dict[str, tuple[int, list[float]]] = {
        start_id: (start_frame, [start_heading]),
        f"{episode_id}_PAN_LEFT": (start_frame, [start_heading + PAN_DEGREES]),
        f"{episode_id}_PAN_RIGHT": (start_frame, [start_heading - PAN_DEGREES]),
        f"{episode_id}_SWEEP": (start_frame, [start_heading + value for value in (0.0, 90.0, 180.0, 270.0)]),
    }
    approach_available = scenario == "APPROACH_ADDRESS_DOOR_RECOVERY"
    if approach_available:
        observation_specs[f"{episode_id}_APPROACH"] = (goal_frame, [goal_heading])

    observations = {
        observation_id: observation(
            observation_id,
            coords,
            frame_id,
            headings,
            image_payload_available,
        )
        for observation_id, (frame_id, headings) in observation_specs.items()
    }
    transition_row: dict[str, dict[str, Any]] = {}
    for action in ACTIONS:
        destination_id = start_id
        executed = action in {"HOLD", "PAN_LEFT", "PAN_RIGHT", "SWEEP"}
        movement = 0.0
        if action in {"PAN_LEFT", "PAN_RIGHT", "SWEEP"}:
            destination_id = f"{episode_id}_{action}"
        elif action == "APPROACH" and approach_available:
            destination_id = f"{episode_id}_APPROACH"
            executed = True
            movement = distance_xy_m(coords, start_frame, goal_frame)
        transition_row[action] = public_edge(action, start_id, destination_id, executed, movement)

    episode = {
        "episode_id": episode_id,
        "poi_id": f"SEVN:{target_address[0]}:{target_address[1]}",
        "mission": {"street_name": target_address[0], "house_number": target_address[1]},
        "start_observation_id": start_id,
        "transitions": {start_id: transition_row},
    }
    truth_observations = {
        observation_id: truth_for_view(labels_by_frame, coords, frame_id, headings, target_address)
        for observation_id, (frame_id, headings) in observation_specs.items()
    }
    target_truth = {
        "scenario_class": scenario,
        "target_address": {"street_name": target_address[0], "house_number": target_address[1]},
        "target_door_annotation": {
            **row_public(goal),
            "x_min": int(goal.x_min),
            "x_max": int(goal.x_max),
            "y_min": int(goal.y_min),
            "y_max": int(goal.y_max),
            "annotation_authority": "SEVN_HUMAN_DOOR_POLYGON_WITH_ADDRESS",
        },
        "observations": truth_observations,
    }
    return episode, target_truth, observations


def select_materialized_cohort(labels: Any, coords: Any, graph: Any, per_scenario: int, image_payload_available: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    labels_by_frame = dataframe_rows_by_frame(labels)
    goals = candidate_goals(labels, coords, graph)
    selected: list[tuple[str, Any, int, float, int, float]] = []
    used_addresses: set[tuple[str, str]] = set()

    for scenario in SCENARIOS[:2]:
        for goal in goals:
            target_address = address_key(goal)
            if target_address is None or target_address in used_addresses:
                continue
            goal_frame = int(goal.name)
            target_heading = label_heading_degrees(goal.x_min, goal.x_max, float(coords.loc[goal_frame].angle))
            offset = -PAN_START_OFFSET_DEGREES if scenario.startswith("PAN_LEFT") else PAN_START_OFFSET_DEGREES
            start_heading = wrap360(target_heading + offset)
            destination_heading = wrap360(start_heading + (PAN_DEGREES if scenario.startswith("PAN_LEFT") else -PAN_DEGREES))
            before = truth_for_view(labels_by_frame, coords, goal_frame, [start_heading], target_address)
            after = truth_for_view(labels_by_frame, coords, goal_frame, [destination_heading], target_address)
            if before["binding_state"] == "CORRECT_UNIQUE" or after["binding_state"] != "CORRECT_UNIQUE":
                continue
            selected.append((scenario, goal, goal_frame, start_heading, goal_frame, destination_heading))
            used_addresses.add(target_address)
            if sum(item[0] == scenario for item in selected) == per_scenario:
                break

    scenario = SCENARIOS[2]
    for goal in goals:
        target_address = address_key(goal)
        if target_address is None or target_address in used_addresses:
            continue
        goal_frame = int(goal.name)
        candidates = []
        for neighbor in graph.neighbors(goal_frame):
            neighbor = int(neighbor)
            if neighbor not in coords.index:
                continue
            heading = quantize_heading(heading_between(coords, neighbor, goal_frame))
            before = truth_for_view(labels_by_frame, coords, neighbor, [heading], target_address)
            after = truth_for_view(labels_by_frame, coords, goal_frame, [heading], target_address)
            if before["binding_state"] == "CORRECT_UNIQUE" or after["binding_state"] != "CORRECT_UNIQUE":
                continue
            candidates.append((distance_xy_m(coords, neighbor, goal_frame), neighbor, heading))
        if not candidates:
            continue
        _, start_frame, heading = min(candidates)
        selected.append((scenario, goal, start_frame, heading, goal_frame, heading))
        used_addresses.add(target_address)
        if sum(item[0] == scenario for item in selected) == per_scenario:
            break

    counts = Counter(item[0] for item in selected)
    require(all(counts[scenario] == per_scenario for scenario in SCENARIOS),
            f"insufficient SEVN candidates for frozen panel: {dict(counts)}")

    episodes = []
    truth_episodes: dict[str, Any] = {}
    observations: dict[str, Any] = {}
    for sequence, item in enumerate(selected, start=1):
        scenario, goal, start_frame, start_heading, goal_frame, goal_heading = item
        episode, episode_truth, episode_observations = build_episode(
            sequence,
            scenario,
            goal,
            start_frame,
            start_heading,
            goal_frame,
            goal_heading,
            coords,
            labels_by_frame,
            image_payload_available,
        )
        episodes.append(episode)
        truth_episodes[episode["episode_id"]] = episode_truth
        observations.update(episode_observations)

    return {
        "observations": observations,
        "episodes": episodes,
        "scenario_counts": dict(counts),
    }, {"episodes": truth_episodes}


def validate_public(public: dict[str, Any], per_scenario: int) -> None:
    require(public.get("schema") == "blindassist-l10-sevn-panolab-public-cohort-v1", "unexpected SEVN public schema")
    require(public.get("provider") == "SEVN 1.0 / Zenodo 3526490", "unexpected SEVN provider")
    observations = public.get("observations")
    episodes = public.get("episodes")
    require(isinstance(observations, dict) and observations, "SEVN observations must be nonempty")
    require(isinstance(episodes, list) and len(episodes) == per_scenario * len(SCENARIOS), "unexpected SEVN panel size")
    forbidden = {"binding_state", "target_visible", "target_door_annotation", "visible_door_count"}
    for observation_id, row in observations.items():
        require(row.get("observation_id") == observation_id, f"{observation_id}: ID mismatch")
        require(not forbidden.intersection(row), f"{observation_id}: evaluator truth leaked")
    for episode in episodes:
        start = episode["start_observation_id"]
        require(start in observations, f"{episode['episode_id']}: start observation missing")
        transitions = episode["transitions"][start]
        require(tuple(transitions) == ACTIONS, f"{episode['episode_id']}: action set mismatch")
        for action, edge in transitions.items():
            destination = edge["to_observation_id"]
            require(destination in observations, f"{episode['episode_id']}/{action}: destination missing")
            if not edge["action_executed"]:
                require(destination == start and edge["movement_distance_m"] == 0.0,
                        f"{episode['episode_id']}/{action}: unavailable action changed state")


def validate_truth(truth: dict[str, Any], public: dict[str, Any], per_scenario: int) -> None:
    require(truth.get("schema") == "blindassist-l10-sevn-panolab-evaluator-truth-v1", "unexpected SEVN truth schema")
    episode_truth = truth.get("episodes")
    public_ids = {episode["episode_id"] for episode in public["episodes"]}
    require(isinstance(episode_truth, dict) and set(episode_truth) == public_ids, "SEVN truth/public episode mismatch")
    counts = Counter(row["scenario_class"] for row in episode_truth.values())
    require(all(counts[scenario] == per_scenario for scenario in SCENARIOS), "SEVN truth scenario count mismatch")
    for episode in public["episodes"]:
        episode_id = episode["episode_id"]
        rows = episode_truth[episode_id]["observations"]
        reachable = {edge["to_observation_id"] for edge in episode["transitions"][episode["start_observation_id"]].values()}
        require(reachable.issubset(rows), f"{episode_id}: truth missing reachable observation")
        for observation_id in reachable:
            require(rows[observation_id]["binding_state"] in BINDING_STATES,
                    f"{episode_id}/{observation_id}: invalid binding state")


class SEVNPanoLab:
    def __init__(self, public_manifest: dict[str, Any], per_scenario: int):
        validate_public(public_manifest, per_scenario)
        self._manifest = copy.deepcopy(public_manifest)
        self._observations = self._manifest["observations"]
        self._episodes = {episode["episode_id"]: episode for episode in self._manifest["episodes"]}
        self._episode: dict[str, Any] | None = None
        self._current: str | None = None
        self._stepped = False

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self._episodes)

    def reset(self, episode_id: str) -> dict[str, Any]:
        require(episode_id in self._episodes, f"unknown SEVN episode: {episode_id}")
        self._episode = self._episodes[episode_id]
        self._current = self._episode["start_observation_id"]
        self._stepped = False
        return self._observation()

    def _observation(self) -> dict[str, Any]:
        require(self._episode is not None and self._current is not None, "reset must be called first")
        return {"episode_id": self._episode["episode_id"], "poi_id": self._episode["poi_id"],
                "mission": copy.deepcopy(self._episode["mission"]), **copy.deepcopy(self._observations[self._current])}

    def step(self, action: str) -> StepResult:
        require(action in ACTIONS, f"unsupported action: {action}")
        require(self._episode is not None and self._current is not None, "reset must be called first")
        require(not self._stepped, "SEVN adapter permits exactly one step")
        before = self._observation()
        edge = copy.deepcopy(self._episode["transitions"][self._current][action])
        self._current = edge["to_observation_id"]
        self._stepped = True
        return StepResult(before=before, after=self._observation(), action_receipt=edge, done=True)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(row["after"]["binding_state"] == "CORRECT_UNIQUE" for row in rows)
    wrong = sum(row["after"]["binding_state"] == "WRONG_UNIQUE" for row in rows)
    visible = sum(row["after"]["target_visible"] for row in rows)
    text_visible = sum(row["after"]["target_house_number_visible"] for row in rows)
    return {
        "episode_count": total,
        "correct_unique": correct,
        "correct_unique_rate": round(correct / total, 6),
        "wrong_unique": wrong,
        "wrong_unique_rate": round(wrong / total, 6),
        "target_visible": visible,
        "target_visible_rate": round(visible / total, 6),
        "target_house_number_visible": text_visible,
        "target_house_number_visible_rate": round(text_visible / total, 6),
        "action_executed": sum(row["action_receipt"]["action_executed"] for row in rows),
    }


def replay(public: dict[str, Any], truth: dict[str, Any], action_by_episode: dict[str, str], per_scenario: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    environment = SEVNPanoLab(public, per_scenario)
    rows = []
    for episode_id in environment.episode_ids:
        before_observation = environment.reset(episode_id)
        action = action_by_episode[episode_id]
        stepped = environment.step(action)
        episode_truth = truth["episodes"][episode_id]
        before = copy.deepcopy(episode_truth["observations"][before_observation["observation_id"]])
        after = copy.deepcopy(episode_truth["observations"][stepped.after["observation_id"]])
        rows.append({
            "episode_id": episode_id,
            "scenario_class": episode_truth["scenario_class"],
            "action": action,
            "before": before,
            "after": after,
            "action_receipt": stepped.action_receipt,
        })
    return rows, summarize(rows)


def outcome_key(score: dict[str, Any], action: str) -> tuple[int, int, int, int, int]:
    return (
        int(score["binding_state"] == "CORRECT_UNIQUE"),
        int(score["target_visible"]),
        int(score["target_house_number_visible"]),
        -int(score["binding_state"] == "WRONG_UNIQUE"),
        -ACTIONS.index(action),
    )


def run_replay(public: dict[str, Any], truth: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    per_scenario = int(protocol["cohort"]["episodes_per_scenario"])
    validate_public(public, per_scenario)
    validate_truth(truth, public, per_scenario)
    fixed = {}
    for action in ACTIONS:
        _, fixed[action] = replay(public, truth, {episode["episode_id"]: action for episode in public["episodes"]}, per_scenario)

    oracle_actions = {}
    for episode in public["episodes"]:
        episode_id = episode["episode_id"]
        start = episode["start_observation_id"]
        candidates = []
        for action in ACTIONS:
            destination = episode["transitions"][start][action]["to_observation_id"]
            score = truth["episodes"][episode_id]["observations"][destination]
            candidates.append((outcome_key(score, action), action))
        oracle_actions[episode_id] = max(candidates)[1]
    oracle_rows, oracle = replay(public, truth, oracle_actions, per_scenario)
    image_payload_available = bool(public["image_payload"]["available"])
    if oracle["correct_unique_rate"] < 1.0:
        decision = "L10_SEVN_METADATA_ACTION_ADAPTER_NOT_READY"
    elif image_payload_available:
        decision = "L10_SEVN_ACTION_ADAPTER_READY_FOR_PIXEL_REPLAY"
    else:
        decision = "L10_SEVN_METADATA_ACTION_ADAPTER_READY_IMAGE_PAYLOAD_PENDING"
    return {
        "schema": "blindassist-l10-sevn-panolab-development-result-v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "claim_scope": "SEVN_1_0_HUMAN_ANNOTATION_AND_GRAPH_METADATA_INTEGRATION_DEVELOPMENT",
        "provider": public["provider"],
        "cohort": {
            "episode_count": len(public["episodes"]),
            "distinct_address_count": len({episode["poi_id"] for episode in public["episodes"]}),
            "scenario_counts": dict(Counter(row["scenario_class"] for row in truth["episodes"].values())),
        },
        "image_payload": copy.deepcopy(public["image_payload"]),
        "fixed_policy_metrics": fixed,
        "oracle_metrics": oracle,
        "oracle_action_distribution": dict(Counter(oracle_actions.values())),
        "episode_results": oracle_rows,
        "non_claims": copy.deepcopy(protocol["non_claims"]),
    }


def materialize(args: argparse.Namespace) -> None:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise SystemExit("materialize requires project-local pandas<2 and PyTables; replay is dependency-free") from exc
    receipts = verify_upstream_files(args.metadata_dir)
    labels = pd.read_hdf(args.metadata_dir / "label.hdf5", key="df", mode="r")
    coords = pd.read_hdf(args.metadata_dir / "coord.hdf5", key="df", mode="r")
    with (args.metadata_dir / "graph.pkl").open("rb") as handle:
        graph = pickle.load(handle)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(protocol.get("schema") == "blindassist-l10-sevn-panolab-protocol-v1", "unexpected SEVN protocol")
    per_scenario = int(protocol["cohort"]["episodes_per_scenario"])
    image_path = args.metadata_dir / "images.hdf5"
    image_payload_available = image_path.is_file()
    if image_payload_available:
        require(image_path.stat().st_size == 1861417787, "images.hdf5: unexpected byte size")
        require(md5_file(image_path) == "f8e570b8232efba23dfc53cc9c9d0b2c",
                "images.hdf5: Zenodo MD5 mismatch")
    cohort, evaluator = select_materialized_cohort(labels, coords, graph, per_scenario, image_payload_available)
    public = {
        "schema": "blindassist-l10-sevn-panolab-public-cohort-v1",
        "generated_at_utc": utc_now(),
        "provider": "SEVN 1.0 / Zenodo 3526490",
        "license": "MIT",
        "source_receipts": receipts,
        "image_payload": {
            "available": image_payload_available,
            "expected_file": "images.hdf5",
            "expected_bytes": 1861417787,
            "expected_md5": "f8e570b8232efba23dfc53cc9c9d0b2c",
            "status": "AVAILABLE" if image_payload_available else "NOT_DOWNLOADED_1_86_GB",
        },
        "action_set": list(ACTIONS),
        "observation_contract": "PUBLIC_MISSION_POSE_VIEWPORT_AND_IMAGE_LOCATOR_ONLY_NO_SEVN_LABELS",
        "observations": cohort["observations"],
        "episodes": cohort["episodes"],
    }
    truth = {
        "schema": "blindassist-l10-sevn-panolab-evaluator-truth-v1",
        "generated_at_utc": utc_now(),
        "truth_authority": "SEVN_HUMAN_DOOR_AND_TEXT_ANNOTATIONS_PLUS_FROZEN_VIEWPORT_GEOMETRY",
        "episodes": evaluator["episodes"],
    }
    validate_public(public, per_scenario)
    validate_truth(truth, public, per_scenario)
    args.source_out.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.truth_out.write_text(json.dumps(truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source": str(args.source_out),
        "source_sha256": sha256_file(args.source_out),
        "truth": str(args.truth_out),
        "truth_sha256": sha256_file(args.truth_out),
        "episode_count": len(public["episodes"]),
        "scenario_counts": cohort["scenario_counts"],
        "image_payload": public["image_payload"],
    }, ensure_ascii=False, indent=2))


def replay_command(args: argparse.Namespace) -> None:
    public = json.loads(args.source.read_text(encoding="utf-8"))
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = run_replay(public, truth, protocol)
    result["inputs"] = {
        "source_path": str(args.source),
        "source_sha256": sha256_file(args.source),
        "truth_path": str(args.truth),
        "truth_sha256": sha256_file(args.truth),
        "protocol_path": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "cohort": result["cohort"],
        "oracle": result["oracle_metrics"],
        "oracle_action_distribution": result["oracle_action_distribution"],
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--metadata-dir", type=Path, required=True)
    materialize_parser.add_argument("--protocol", type=Path, required=True)
    materialize_parser.add_argument("--source-out", type=Path, required=True)
    materialize_parser.add_argument("--truth-out", type=Path, required=True)
    materialize_parser.set_defaults(func=materialize)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--source", type=Path, required=True)
    replay_parser.add_argument("--truth", type=Path, required=True)
    replay_parser.add_argument("--protocol", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.set_defaults(func=replay_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
