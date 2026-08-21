"""One-shot action-responsive Mapillary replay for Last-10m engineering sanity.

Preparation is provider-outcome blind: it chooses one already-reviewed goal from
local Mapillary metadata, expands only the selected real capture sequence, and
freezes action edges from GPS/heading/timestamp geometry.  Execution then reuses
the unchanged P0 adapter once per visited observation.  No candidate, bbox,
feature, identity, or prior provider output is used by the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests
from PIL import Image

from .cli import _append_event, _atomic_json, _failed_provider_observation, _read_json, _sha256_file
from .core import Attribution, ContractError, EpisodeState, Policy, State, apply_observation, stop_episode
from .provider_adapter import ProviderAdapterError, ground_current_frame, preflight_provider


PUBLIC_MANIFEST = "responsive-replay-public.json"
TRUTH_SIDECAR = "responsive-replay-evaluator-truth.json"
FREEZE_RECEIPT = "responsive-replay-freeze-receipt.json"
EXECUTION_MODE = "ACTION_RESPONSIVE_MAPILLARY_POSE_AND_VIEWPORT_REPLAY"
CLAIM_CEILING = (
    "DETERMINISTIC_VIEWPORT_ENGINEERING_MECHANICS_ONLY_NO_REAL_BLIND_USER_BUILDING_ENTRANCE_NAVIGATION_"
    "PRODUCT_OR_SAFETY_CONFIRMATION"
)
ACTIONS = ("TURN_LEFT", "TURN_RIGHT", "FORWARD", "RESCAN_HOLD")
MAPILLARY_FIELDS = (
    "id,computed_geometry,captured_at,computed_compass_angle,compass_angle,"
    "camera_type,camera_parameters,width,height,sequence,thumb_1024_url"
)
ARRIVAL_DISTANCE_M = 8.0
ARRIVAL_BEARING_ERROR_DEG = 30.0
MAX_EPISODES = 6
VIEWPORT_YAWS = (-2, -1, 0, 1, 2)
VIEWPORT_YAW_STEP_DEG = 12.0
VIEWPORT_SHIFT_FRACTION = 0.10


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _signed_angle(to_deg: float, from_deg: float) -> float:
    return (float(to_deg) - float(from_deg) + 540.0) % 360.0 - 180.0


def _distance_and_bearing(first: Sequence[float], second: Sequence[float]) -> tuple[float, float]:
    lon1, lat1 = map(float, first)
    lon2, lat2 = map(float, second)
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    north = (lat2 - lat1) * 111_320.0
    east = (lon2 - lon1) * 111_320.0 * math.cos(mean_lat)
    return math.hypot(east, north), math.degrees(math.atan2(east, north)) % 360.0


def _mapillary_get(session: requests.Session, url: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
    response = session.get(url, params=dict(params), timeout=45)
    if response.status_code != 200:
        raise ContractError(f"Mapillary metadata request failed with HTTP {response.status_code}")
    value = response.json()
    if not isinstance(value, Mapping) or "error" in value:
        raise ContractError("Mapillary metadata response is invalid")
    return value


def _osm_anchor(node_id: str) -> dict[str, Any]:
    if not node_id.startswith("node/") or not node_id[5:].isdigit():
        raise ContractError("selected target anchor is not an OSM node")
    response = requests.get(
        f"https://api.openstreetmap.org/api/0.6/node/{node_id[5:]}.json",
        headers={"User-Agent": "BlindAssist-action-responsive-engineering-replay"},
        timeout=45,
    )
    if response.status_code != 200:
        raise ContractError(f"OSM anchor request failed with HTTP {response.status_code}")
    elements = response.json().get("elements", [])
    if len(elements) != 1:
        raise ContractError("OSM anchor response is not singular")
    element = elements[0]
    tags = element.get("tags") if isinstance(element.get("tags"), Mapping) else {}
    if not str(tags.get("entrance", "")):
        raise ContractError("selected OSM anchor is not tagged as an entrance")
    return {
        "anchor_id": node_id,
        "coordinates": [float(element["lon"]), float(element["lat"])],
        "osm_version": int(element["version"]),
        "entrance_tag": str(tags["entrance"]),
        "retrieved_at_ms": _now_ms(),
        "source_url": f"https://www.openstreetmap.org/node/{node_id[5:]}",
    }


def _local_inventory(brain_cohort: Path, metadata_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    brain = _read_json(brain_cohort)
    if brain.get("claim_ceiling") != "SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY":
        raise ContractError("brain cohort is not the existing reviewed Silver-B Development bank")
    unique_goal_by_frame: dict[str, str] = {}
    for episode in brain.get("episodes", []):
        evaluator = episode.get("evaluator_episode", {})
        if evaluator.get("goal_reference_resolution") != "UNIQUE":
            continue
        frame_ids = evaluator.get("observation_window", {}).get("frame_ids", [])
        if len(frame_ids) == 1:
            unique_goal_by_frame[str(frame_ids[0])] = str(evaluator["goal_spec"]["target_name"])

    records: dict[str, dict[str, Any]] = {}
    metadata_files = sorted(metadata_root.rglob("mapillary_metadata.json"))
    for path in metadata_files:
        value = _read_json(path)
        for raw in value.get("images", []):
            if not isinstance(raw, Mapping):
                continue
            image_id = str(raw.get("id", ""))
            image_path = Path(str(raw.get("path", "")))
            if image_id and image_path.is_file() and image_id not in records:
                records[image_id] = dict(raw) | {"metadata_path": str(path.resolve())}

    goal_by_anchor: dict[str, str] = {}
    for frame_id, goal in unique_goal_by_frame.items():
        item = records.get(frame_id)
        if item and str(item.get("target_anchor_id", "")):
            goal_by_anchor[str(item["target_anchor_id"])] = goal
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records.values():
        anchor = str(item.get("target_anchor_id", ""))
        if anchor in goal_by_anchor:
            grouped.setdefault(anchor, []).append(item)
    viable = [(len(items), anchor, goal_by_anchor[anchor], items) for anchor, items in grouped.items() if len(items) >= 5]
    if not viable:
        raise ContractError("no reviewed Mapillary anchor has enough local pose/sequence support")
    _, anchor_id, goal_name, selected = sorted(viable, key=lambda item: (-item[0], item[1]))[0]
    sequences = Counter(str(item.get("sequence_id", "")) for item in selected if item.get("sequence_id"))
    if not sequences:
        raise ContractError("selected Mapillary anchor has no sequence metadata")
    sequence_id = sorted(sequences.items(), key=lambda item: (-item[1], item[0]))[0][0]
    inventory = {
        "metadata_file_count": len(metadata_files),
        "local_mapillary_image_count": len(records),
        "reviewed_unique_frame_count": len(unique_goal_by_frame),
        "mapped_anchor_count": len(grouped),
        "selection_rule": "MAX_LOCAL_POSE_FRAME_COUNT_THEN_LEXICAL_ANCHOR_AND_SEQUENCE",
        "selected_anchor_id": anchor_id,
        "selected_goal_name": goal_name,
        "selected_sequence_id": sequence_id,
        "selected_anchor_local_frame_count": len(selected),
        "brain_cohort_path": str(brain_cohort.resolve()),
        "brain_cohort_sha256": _sha256_file(brain_cohort),
    }
    return inventory, selected


def _fetch_sequence(token: str, sequence_id: str, anchor: Mapping[str, Any]) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers["Authorization"] = f"OAuth {token}"
    sequence = _mapillary_get(
        session,
        "https://graph.mapillary.com/image_ids",
        {"sequence_id": sequence_id},
    )
    image_ids = [str(item["id"]) for item in sequence.get("data", []) if isinstance(item, Mapping) and item.get("id")]
    if not image_ids:
        raise ContractError("selected Mapillary sequence is empty")
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for chunk in _chunks(image_ids, 40):
        response = _mapillary_get(
            session,
            "https://graph.mapillary.com/",
            {"ids": ",".join(chunk), "fields": MAPILLARY_FIELDS},
        )
        for image_id, value in response.items():
            if isinstance(value, Mapping):
                raw_by_id[str(image_id)] = value

    target = anchor["coordinates"]
    normalized = []
    for image_id in image_ids:
        item = raw_by_id.get(image_id)
        if not item:
            continue
        geometry = item.get("computed_geometry")
        coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
        heading = item.get("computed_compass_angle", item.get("compass_angle"))
        parameters = item.get("camera_parameters")
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 2
            or not isinstance(heading, (int, float))
            or str(item.get("camera_type", "")).lower() not in {"perspective", "planar"}
            or not isinstance(parameters, list)
            or not parameters
            or not item.get("thumb_1024_url")
        ):
            continue
        distance, target_bearing = _distance_and_bearing(coordinates, target)
        error = _signed_angle(target_bearing, float(heading))
        if not (1.0 <= distance <= 45.0 and abs(error) <= 65.0):
            continue
        normalized.append(
            {
                "frame_id": image_id,
                "sequence_index": image_ids.index(image_id),
                "captured_at_ms": int(item.get("captured_at") or 0),
                "coordinates": [float(coordinates[0]), float(coordinates[1])],
                "heading_deg": float(heading) % 360.0,
                "heading_kind": "COMPUTED_COMPASS_ANGLE" if item.get("computed_compass_angle") is not None else "COMPASS_ANGLE",
                "target_distance_m": distance,
                "target_bearing_deg": target_bearing,
                "target_bearing_error_deg": error,
                "source_width": int(item.get("width") or 0),
                "source_height": int(item.get("height") or 0),
                "download_url": str(item["thumb_1024_url"]),
            }
        )
    if len(normalized) < 10:
        raise ContractError("selected Mapillary sequence has insufficient anchor-facing pose coverage")
    return normalized


def _best_edge(node: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]], action: str) -> str | None:
    candidates: list[tuple[tuple[float, ...], str]] = []
    for other in nodes:
        if other["frame_id"] == node["frame_id"]:
            continue
        movement, move_bearing = _distance_and_bearing(node["coordinates"], other["coordinates"])
        heading_delta = _signed_angle(other["heading_deg"], node["heading_deg"])
        distance_delta = float(other["target_distance_m"]) - float(node["target_distance_m"])
        time_delta = abs(int(other["captured_at_ms"]) - int(node["captured_at_ms"])) / 1000.0
        move_error = abs(_signed_angle(move_bearing, node["heading_deg"]))
        if action == "TURN_LEFT" and movement <= 3.0 and -50.0 <= heading_delta <= -8.0 and abs(distance_delta) <= 3.0:
            score = (abs(heading_delta + 20.0), movement, time_delta, str(other["frame_id"]))
        elif action == "TURN_RIGHT" and movement <= 3.0 and 8.0 <= heading_delta <= 50.0 and abs(distance_delta) <= 3.0:
            score = (abs(heading_delta - 20.0), movement, time_delta, str(other["frame_id"]))
        elif (
            action == "FORWARD"
            and 0.6 <= movement <= 5.0
            and move_error <= 45.0
            and distance_delta <= -0.35
            and abs(heading_delta) <= 40.0
        ):
            score = (abs(movement - 2.0), move_error, abs(heading_delta), str(other["frame_id"]))
        elif (
            action == "RESCAN_HOLD"
            and movement <= 2.5
            and abs(heading_delta) <= 12.0
            and abs(distance_delta) <= 2.5
        ):
            score = (movement, abs(heading_delta), time_delta, str(other["frame_id"]))
        else:
            continue
        candidates.append((score, str(other["frame_id"])))
    return min(candidates)[1] if candidates else None


def _build_edges(nodes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    return {
        str(node["frame_id"]): {
            action: target
            for action in ACTIONS
            if (target := _best_edge(node, nodes, action)) is not None
        }
        for node in nodes
    }


def _pose_arrival(node: Mapping[str, Any]) -> bool:
    return (
        float(node["target_distance_m"]) <= ARRIVAL_DISTANCE_M
        and abs(float(node["target_bearing_error_deg"])) <= ARRIVAL_BEARING_ERROR_DEG
    )


def _can_reach(start: str, targets: set[str], edges: Mapping[str, Mapping[str, str]]) -> bool:
    queue = deque([start])
    seen = {start}
    while queue:
        current = queue.popleft()
        if current in targets:
            return True
        for target in edges.get(current, {}).values():
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return False


def _choose_starts(nodes: Sequence[Mapping[str, Any]], edges: Mapping[str, Mapping[str, str]]) -> list[str]:
    arrivals = {str(item["frame_id"]) for item in nodes if _pose_arrival(item)}
    candidates = sorted(
        (
            item
            for item in nodes
            if 15.0 <= float(item["target_distance_m"]) <= 35.0
            and abs(float(item["target_bearing_error_deg"])) <= 55.0
            and _can_reach(str(item["frame_id"]), arrivals, edges)
        ),
        key=lambda item: (-float(item["target_distance_m"]), int(item["captured_at_ms"]), str(item["frame_id"])),
    )
    if len(candidates) < 5:
        raise ContractError("pose graph cannot provide five fixed start states with an arrival path")
    count = min(MAX_EPISODES, len(candidates))
    indices = sorted({round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)})
    starts = [str(candidates[index]["frame_id"]) for index in indices]
    if len(starts) < 5:
        raise ContractError("deterministic start thinning produced too few episodes")
    return starts


def _view_frame_id(base_frame_id: str, yaw: int) -> str:
    label = "z" if yaw == 0 else (f"p{yaw}" if yaw > 0 else f"m{abs(yaw)}")
    return f"{base_frame_id}--viewport-{label}"


def _expand_viewport_graph(
    base_nodes: Sequence[Mapping[str, Any]], base_edges: Mapping[str, Mapping[str, str]]
) -> list[dict[str, Any]]:
    expanded = []
    for base in base_nodes:
        base_id = str(base["frame_id"])
        for yaw in VIEWPORT_YAWS:
            actions: dict[str, str] = {}
            if yaw > min(VIEWPORT_YAWS):
                actions["TURN_LEFT"] = _view_frame_id(base_id, yaw - 1)
            if yaw < max(VIEWPORT_YAWS):
                actions["TURN_RIGHT"] = _view_frame_id(base_id, yaw + 1)
            if base_edges.get(base_id, {}).get("FORWARD"):
                actions["FORWARD"] = _view_frame_id(base_edges[base_id]["FORWARD"], yaw)
            if base_edges.get(base_id, {}).get("RESCAN_HOLD"):
                actions["RESCAN_HOLD"] = _view_frame_id(base_edges[base_id]["RESCAN_HOLD"], yaw)
            expanded.append(
                {
                    **{key: value for key, value in base.items() if key != "download_url"},
                    "frame_id": _view_frame_id(base_id, yaw),
                    "source_frame_id": base_id,
                    "viewport_yaw_index": yaw,
                    "viewport_yaw_offset_deg": yaw * VIEWPORT_YAW_STEP_DEG,
                    "heading_deg": (float(base["heading_deg"]) + yaw * VIEWPORT_YAW_STEP_DEG) % 360.0,
                    "target_bearing_error_deg": float(base["target_bearing_error_deg"]) - yaw * VIEWPORT_YAW_STEP_DEG,
                    "actions": actions,
                }
            )
    return expanded


def _materialize_viewports(base_nodes: list[dict[str, Any]], expanded: list[dict[str, Any]], output_dir: Path) -> None:
    source_dir = output_dir / "source-images"
    image_dir = output_dir / "images"
    source_dir.mkdir(parents=True, exist_ok=False)
    image_dir.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    source_by_id: dict[str, Path] = {}
    source_hash_by_id: dict[str, str] = {}
    for item in base_nodes:
        path = source_dir / f"{item['frame_id']}.jpg"
        response = session.get(item.pop("download_url"), timeout=60)
        if response.status_code != 200 or not response.content:
            raise ContractError(f"Mapillary image download failed for {item['frame_id']}")
        path.write_bytes(response.content)
        source_by_id[str(item["frame_id"])] = path
        source_hash_by_id[str(item["frame_id"])] = _sha256_file(path)
    for item in expanded:
        source_id = str(item["source_frame_id"])
        with Image.open(source_by_id[source_id]) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            shift = -int(round(int(item["viewport_yaw_index"]) * VIEWPORT_SHIFT_FRACTION * width))
            rendered = Image.new("RGB", (width, height), (32, 32, 32))
            rendered.paste(image, (shift, 0))
        path = image_dir / f"{item['frame_id']}.jpg"
        rendered.save(path, format="JPEG", quality=95, subsampling=0)
        item["image_path"] = str(path.resolve())
        item["image_sha256"] = _sha256_file(path)
        item["source_image_sha256"] = source_hash_by_id[source_id]
        item["source_url"] = f"https://www.mapillary.com/app/?focus=photo&pKey={source_id}"


def prepare_responsive_replay(*, brain_cohort: Path, metadata_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise ContractError("responsive replay output directory must not already exist")
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN") or os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise ContractError("Mapillary access token is unavailable")
    inventory, _ = _local_inventory(brain_cohort, metadata_root)
    anchor = _osm_anchor(inventory["selected_anchor_id"])
    base_nodes = _fetch_sequence(token, inventory["selected_sequence_id"], anchor)
    base_nodes.sort(key=lambda item: (int(item["sequence_index"]), str(item["frame_id"])))
    base_edges = _build_edges(base_nodes)
    base_action_counts = {action: sum(action in value for value in base_edges.values()) for action in ACTIONS}
    native_turn_reachable_starts = [
        item for item in base_nodes
        if 15.0 <= float(item["target_distance_m"]) <= 35.0
        and (
            (
                float(item["target_bearing_error_deg"]) >= 10.0
                and "TURN_RIGHT" in base_edges[str(item["frame_id"])]
            )
            or (
                float(item["target_bearing_error_deg"]) <= -10.0
                and "TURN_LEFT" in base_edges[str(item["frame_id"])]
            )
        )
    ]
    if native_turn_reachable_starts:
        # The fallback is intentionally used only because the selected approach
        # segment has no reliable same-position turn captures.
        raise ContractError("unexpected native turn coverage; viewport fallback selection rule must be re-audited")
    arrivals = {str(item["frame_id"]) for item in base_nodes if _pose_arrival(item)}
    if len(arrivals) < 2 or not any(base_edges.get(item, {}).get("RESCAN_HOLD") in arrivals for item in arrivals):
        raise ContractError("responsive graph lacks two real near-arrival observations joined by RESCAN_HOLD")
    starts = _choose_starts(base_nodes, base_edges)
    nodes = _expand_viewport_graph(base_nodes, base_edges)
    edges = {str(item["frame_id"]): item["actions"] for item in nodes}
    action_counts = {action: sum(action in value for value in edges.values()) for action in ACTIONS}
    if any(value == 0 for value in action_counts.values()):
        raise ContractError(f"responsive graph lacks an action family: {action_counts}")
    expanded_arrivals = {str(item["frame_id"]) for item in nodes if _pose_arrival(item)}

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        _materialize_viewports(base_nodes, nodes, output_dir)
        public_nodes = []
        for item in nodes:
            public_nodes.append(
                {
                    key: item[key]
                    for key in (
                        "frame_id", "source_frame_id", "sequence_index", "captured_at_ms", "coordinates", "heading_deg", "heading_kind",
                        "target_distance_m", "target_bearing_deg", "target_bearing_error_deg", "source_width", "source_height",
                        "viewport_yaw_index", "viewport_yaw_offset_deg", "image_path", "image_sha256", "source_image_sha256", "source_url",
                    )
                }
                | {"actions": item["actions"]}
            )
        public = {
            "schema_version": 1,
            "milestone_id": "BLINDASSIST_LAST_10M_REGROUNDING_V0",
            "execution_mode": EXECUTION_MODE,
            "goal_name": inventory["selected_goal_name"],
            "location_id": "mapillary-responsive-site-01",
            "sequence_id": inventory["selected_sequence_id"],
            "action_set": list(ACTIONS),
            "nodes": public_nodes,
            "episodes": [
                {"episode_id": f"responsive-site-01-e{index:02d}", "start_frame_id": _view_frame_id(frame_id, 0)}
                for index, frame_id in enumerate(starts, start=1)
            ],
            "source": {
                "name": "Mapillary",
                "attribution": "Mapillary contributors; image IDs and source links retained",
                "license_note": "Mapillary Terms; local research use and attribution obligations retained; no imagery redistributed",
                "metadata_authority": "COMPUTED_GEOMETRY_COMPUTED_COMPASS_ANGLE_SEQUENCE_CAPTURE_TIME_PLUS_FIXED_VIEWPORT_TRANSFORM",
            },
            "transition_rule": {
                "provider_outcome_blind": True,
                "selection_inputs": ["GPS", "computed heading", "capture timestamp", "sequence order", "OSM target anchor", "fixed yaw viewport index"],
                "turn_max_translation_m": 3.0,
                "turn_rule": "SAME_REAL_CAPTURE_DETERMINISTIC_HORIZONTAL_VIEWPORT_TRANSLATION",
                "viewport_yaw_step_deg": VIEWPORT_YAW_STEP_DEG,
                "viewport_shift_fraction": VIEWPORT_SHIFT_FRACTION,
                "forward_translation_m": [0.6, 5.0],
                "rescan_hold_max_translation_m": 2.5,
            },
            "claim_ceiling": CLAIM_CEILING,
        }
        truth = {
            "schema_version": 1,
            "authority": "EVALUATOR_ONLY_NOT_PROVIDER_VISIBLE",
            "anchor": anchor,
            "arrival_rule": {
                "maximum_anchor_distance_m": ARRIVAL_DISTANCE_M,
                "maximum_absolute_bearing_error_deg": ARRIVAL_BEARING_ERROR_DEG,
                "meaning": "OSM_MAIN_ENTRANCE_POSE_PROXY_NOT_ACCESSIBILITY_OR_SAFETY_TRUTH",
            },
            "wrong_target_confirmation_evaluability": "NOT_EVALUABLE_EXACT_FRAME_REGION_TRUTH_UNAVAILABLE",
            "public_manifest_sha256": _canonical_hash(public),
        }
        freeze = {
            "schema_version": 1,
            "status": "RESPONSIVE_REPLAY_AND_ROSTER_FROZEN_BEFORE_FORMAL_PROVIDER_OBSERVATIONS",
            "frozen_at_ms": _now_ms(),
            "formal_provider_observation_count": 0,
            "selection_inventory": inventory,
            "source_priority_decision": "D_MINIMAL_OUTCOME_INDEPENDENT_VIEWPORT_FALLBACK_AFTER_MAPILLARY_NATIVE_TURN_GAP",
            "base_real_frame_count": len(base_nodes),
            "node_count": len(nodes),
            "episode_count": len(starts),
            "arrival_node_count": len(expanded_arrivals),
            "base_pose_action_edge_counts": base_action_counts,
            "action_edge_counts": action_counts,
            "public_manifest_sha256": truth["public_manifest_sha256"],
            "pre_formal_mechanical_fixes": [
                {
                    "failure": "INITIAL_PREPARE_FOUND_ONLY_ONE_NODE_AT_6M_AND_NO_2M_RESCAN_EDGE",
                    "provider_observations_before_fix": 0,
                    "fix": "USE_8M_LAST_10M_POSE_PROXY_AND_2_5M_ADJACENT_SEQUENCE_RESCAN_BOUND",
                    "evidence": "two consecutive real sequence frames are 2.220m apart at 7.77m and 5.56m from the frozen OSM entrance",
                },
                {
                    "failure": "RESPONSIVE_SCENES_V0_NATIVE_APPROACH_STARTS_HAVE_NO_RELIABLE_TURN_EDGE",
                    "provider_observations_before_fix": 0,
                    "fix": "ADD_FIXED_FIVE_STATE_HORIZONTAL_VIEWPORT_YAW_GRAPH_FOR_TURNS_ONLY",
                    "evidence": "all six 15m-to-35m pose-reachable starts had FORWARD support but zero same-position native turn support",
                }
            ],
            "claim_ceiling": CLAIM_CEILING,
        }
        _atomic_json(output_dir / PUBLIC_MANIFEST, public)
        _atomic_json(output_dir / TRUTH_SIDECAR, truth)
        _atomic_json(output_dir / FREEZE_RECEIPT, freeze)
        return freeze
    except Exception:
        # Keep any partial acquisition for diagnosis; it is not a frozen roster.
        raise


def _action_for_event(event: Mapping[str, Any]) -> str | None:
    if event.get("to_state") in {State.COMPLETE.value, State.ABSTAIN.value}:
        return None
    if event.get("to_state") in {State.RESCAN.value, State.ARRIVAL_CONFIRM.value}:
        return "RESCAN_HOLD"
    candidate = event.get("candidate")
    if event.get("to_state") != State.ADVANCE_AND_REOBSERVE.value or not isinstance(candidate, Mapping):
        raise ContractError("non-terminal control event cannot be mapped to an environment action")
    center_x = float(candidate["center_x"])
    policy = Policy()
    if center_x < policy.center_left:
        return "TURN_LEFT"
    if center_x > policy.center_right:
        return "TURN_RIGHT"
    return "FORWARD"


def _verdict(summaries: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    completed = sum(item["episode_completion"] for item in summaries)
    false_arrivals = sum(item["false_arrival"] for item in summaries)
    grounding_failures = sum(item["failure_class"] == "CURRENT_FRAME_GROUNDING_BOTTLENECK" for item in summaries)
    control_failures = sum(item["failure_class"] == "CONTROL_POLICY_BOTTLENECK" for item in summaries)
    if completed and false_arrivals == 0:
        return "RESPONSIVE_REGROUNDING_MECHANICALLY_WORKS", "at least one frozen start reached pose-proxy ARRIVAL without false arrival"
    if grounding_failures > control_failures:
        return "CURRENT_FRAME_GROUNDING_BOTTLENECK", "current-frame P0 unreliability dominated terminal failures"
    return "CONTROL_POLICY_BOTTLENECK", "reliable current-frame candidates did not produce pose-proxy completion"


def execute_responsive_replay(
    *, scene_dir: Path, run_dir: Path, codex_exe: Path, model_dir: Path
) -> dict[str, Any]:
    public = _read_json(scene_dir / PUBLIC_MANIFEST)
    truth = _read_json(scene_dir / TRUTH_SIDECAR)
    freeze = _read_json(scene_dir / FREEZE_RECEIPT)
    if truth.get("public_manifest_sha256") != _canonical_hash(public):
        raise ContractError("responsive public/truth binding drift")
    if freeze.get("public_manifest_sha256") != truth["public_manifest_sha256"]:
        raise ContractError("responsive freeze receipt binding drift")
    if freeze.get("formal_provider_observation_count") != 0:
        raise ContractError("roster was not frozen before provider observations")
    if run_dir.exists():
        raise ContractError("formal responsive run directory already exists; one-shot replay cannot resume or rerun")
    nodes = {str(item["frame_id"]): item for item in public.get("nodes", [])}
    episodes = public.get("episodes", [])
    if not 5 <= len(episodes) <= 15 or len(public.get("action_set", [])) != 4:
        raise ContractError("responsive roster size or action contract is invalid")
    for node in nodes.values():
        image = Path(str(node["image_path"]))
        if not image.is_file() or _sha256_file(image) != node["image_sha256"]:
            raise ContractError("frozen responsive frame is missing or changed")

    # Required machine/provider preflight happens before the formal run root exists.
    provider_lock = preflight_provider(codex_exe=codex_exe, model_dir=model_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    lock = run_dir / "responsive-run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    _atomic_json(run_dir / "provider-lock.json", provider_lock)
    _atomic_json(
        run_dir / "run-manifest.json",
        {
            "schema_version": 1,
            "status": "FORMAL_ONE_SHOT_STARTED",
            "started_at_ms": _now_ms(),
            "scene_public_sha256": truth["public_manifest_sha256"],
            "provider_lock_sha256": _canonical_hash(provider_lock),
            "retry_rule": "UNCHANGED_PROVIDER_INTERNAL_SCHEMA_RETRY_ONLY_MAX_TWO_ATTEMPTS",
            "rerun_rule": "NO_EPISODE_OR_OBSERVATION_RERUN_AFTER_FORMAL_START",
            "claim_ceiling": CLAIM_CEILING,
        },
    )
    summaries: list[dict[str, Any]] = []
    abnormal_events: list[dict[str, Any]] = []
    try:
        for ordinal, episode in enumerate(episodes, start=1):
            episode_id = str(episode["episode_id"])
            directory = run_dir / "episodes" / episode_id
            directory.mkdir(parents=True, exist_ok=False)
            events_path = directory / "events.jsonl"
            started_at_ms = _now_ms()
            state = EpisodeState.start(
                episode_id=episode_id,
                location_id=str(public["location_id"]),
                goal_name=str(public["goal_name"]),
                started_at_ms=started_at_ms,
            )
            current = str(episode["start_frame_id"])
            trajectory: list[dict[str, Any]] = []
            direction_commands = 0
            rescan_actions = 0
            exhausted = False
            failure_state: str | None = None
            _append_event(events_path, {"event_type": "EPISODE_STARTED", "episode_id": episode_id, "start_frame_id": current, "started_at_ms": started_at_ms})
            while state.state not in {State.COMPLETE.value, State.ABSTAIN.value}:
                node = nodes[current]
                observation_id = f"{episode_id}-o{state.observation_count + 1:03d}"
                call_dir = directory / "provider_calls" / observation_id
                try:
                    observation = ground_current_frame(
                        provider_lock=provider_lock,
                        call_dir=call_dir,
                        episode_id=episode_id,
                        goal_name=state.goal_name,
                        image_path=Path(str(node["image_path"])),
                        frame_id=current,
                        observation_id=observation_id,
                        captured_at_ms=_now_ms(),
                    )
                except ProviderAdapterError as error:
                    abnormal_events.append({"episode_id": episode_id, "observation_id": observation_id, "error": str(error)})
                    observation = _failed_provider_observation(
                        episode_id=episode_id,
                        observation_id=observation_id,
                        frame_id=current,
                        frame_sha256=str(node["image_sha256"]),
                        captured_at_ms=_now_ms(),
                        reason=str(error),
                    )
                result = apply_observation(state, observation, Policy())
                state = result.state
                event = dict(result.event) | {
                    "environment_frame_id": current,
                    "target_distance_m": node["target_distance_m"],
                    "target_bearing_error_deg": node["target_bearing_error_deg"],
                }
                action = _action_for_event(event)
                event["environment_action"] = action
                _append_event(events_path, event)
                trajectory.append(
                    {
                        "observation_id": observation_id,
                        "frame_id": current,
                        "from_state": event["from_state"],
                        "to_state": event["to_state"],
                        "p0_status": event["p0_status"],
                        "action": action,
                        "target_distance_m": node["target_distance_m"],
                        "target_bearing_error_deg": node["target_bearing_error_deg"],
                    }
                )
                if action is None:
                    break
                if action == "RESCAN_HOLD":
                    rescan_actions += 1
                else:
                    direction_commands += 1
                next_frame = node.get("actions", {}).get(action)
                action_event = {
                    "event_type": "ENVIRONMENT_ACTION_APPLIED",
                    "episode_id": episode_id,
                    "action": action,
                    "from_frame_id": current,
                    "to_frame_id": next_frame,
                    "rule": "FROZEN_POSE_GRAPH_LOOKUP",
                }
                _append_event(events_path, action_event)
                if not next_frame:
                    exhausted = True
                    failure_state = state.state
                    stop = stop_episode(
                        state,
                        stopped_at_ms=_now_ms(),
                        attribution=Attribution.INTERACTION_OR_CONTROL_BOTTLENECK,
                        reason="FROZEN_RESPONSIVE_GRAPH_ACTION_UNAVAILABLE",
                    )
                    _append_event(events_path, dict(stop) | {"environment_terminal_state": "EXHAUSTED"})
                    break
                current = str(next_frame)

            final_node = nodes[current]
            pose_arrival = _pose_arrival(final_node)
            completed = state.state == State.COMPLETE.value and pose_arrival
            false_arrival = state.state == State.COMPLETE.value and not pose_arrival
            if completed:
                failure_class = None
            elif false_arrival:
                failure_class = "CONTROL_POLICY_BOTTLENECK"
                failure_state = failure_state or State.ARRIVAL_CONFIRM.value
            elif state.reliable_observation_count == 0 or state.consecutive_unreliable >= Policy().max_consecutive_unreliable:
                failure_class = "CURRENT_FRAME_GROUNDING_BOTTLENECK"
                failure_state = failure_state or state.state
            else:
                failure_class = "CONTROL_POLICY_BOTTLENECK"
                failure_state = failure_state or state.state
            terminal_state = "EXHAUSTED" if exhausted else state.state
            summary = {
                "schema_version": 1,
                "episode_id": episode_id,
                "start_frame_id": episode["start_frame_id"],
                "terminal_frame_id": current,
                "terminal_state": terminal_state,
                "control_terminal_state": state.state,
                "episode_completion": completed,
                "false_arrival": false_arrival,
                "wrong_target_confirmation": None,
                "wrong_target_confirmation_evaluability": truth["wrong_target_confirmation_evaluability"],
                "first_reliable_grounding_latency_ms": (
                    state.first_discovery_at_ms - state.started_at_ms if state.first_discovery_at_ms is not None else None
                ),
                "observation_count": state.observation_count,
                "reliable_observation_count": state.reliable_observation_count,
                "direction_command_count": direction_commands,
                "rescan_count": rescan_actions,
                "abstention": state.state == State.ABSTAIN.value and not exhausted,
                "exhausted": exhausted,
                "failure_class": failure_class,
                "failure_state": failure_state,
                "terminal_pose_proxy_arrival": pose_arrival,
                "action_state_trajectory": trajectory,
                "claim_ceiling": CLAIM_CEILING,
            }
            _atomic_json(directory / "episode_summary.json", summary)
            _append_event(events_path, {"event_type": "EPISODE_ADJUDICATED", **summary})
            summaries.append(summary)
            print(
                f"episode {ordinal}/{len(episodes)} {episode_id} terminal={terminal_state} "
                f"completion={completed} observations={state.observation_count}",
                flush=True,
            )

        verdict, verdict_reason = _verdict(summaries)
        failure_counts = Counter(item["failure_class"] for item in summaries if item["failure_class"])
        failure_state_counts = Counter(item["failure_state"] for item in summaries if item["failure_state"])
        latencies = [int(item["first_reliable_grounding_latency_ms"]) for item in summaries if item["first_reliable_grounding_latency_ms"] is not None]
        result = {
            "schema_version": 1,
            "milestone_id": "BLINDASSIST_LAST_10M_REGROUNDING_V0",
            "status": "RESPONSIVE_ENGINEERING_SANITY_COMPLETE",
            "execution_mode": EXECUTION_MODE,
            "episode_count": len(summaries),
            "episode_completion_count": sum(item["episode_completion"] for item in summaries),
            "false_arrival_count": sum(item["false_arrival"] for item in summaries),
            "wrong_target_confirmation_count": None,
            "wrong_target_confirmation_evaluability": truth["wrong_target_confirmation_evaluability"],
            "first_reliable_grounding_latency_ms": {
                "count": len(latencies),
                "median": sorted(latencies)[len(latencies) // 2] if latencies else None,
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
            "observation_count": sum(item["observation_count"] for item in summaries),
            "reliable_observation_count": sum(item["reliable_observation_count"] for item in summaries),
            "direction_command_count": sum(item["direction_command_count"] for item in summaries),
            "rescan_count": sum(item["rescan_count"] for item in summaries),
            "abstention_count": sum(item["abstention"] for item in summaries),
            "exhausted_trajectory_count": sum(item["exhausted"] for item in summaries),
            "failure_class_counts": dict(sorted(failure_counts.items())),
            "failure_state_counts": dict(sorted(failure_state_counts.items())),
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "episodes": summaries,
            "provider_lock_sha256": _canonical_hash(provider_lock),
            "scene_public_sha256": truth["public_manifest_sha256"],
            "formal_abnormal_events": abnormal_events,
            "formal_post_outcome_fixes_or_reruns": [],
            "claim_ceiling": CLAIM_CEILING,
        }
        _atomic_json(run_dir / "responsive-result.json", result)
        _atomic_json(
            run_dir / "execution-receipt.json",
            {
                "schema_version": 1,
                "status": "FORMAL_ONE_SHOT_COMPLETED_AND_SEALED",
                "completed_at_ms": _now_ms(),
                "episode_count": len(summaries),
                "provider_observation_count": result["observation_count"],
                "abnormal_events": abnormal_events,
                "post_outcome_fixes_or_reruns": [],
                "result_sha256": _canonical_hash(result),
                "claim_ceiling": CLAIM_CEILING,
            },
        )
        return result
    finally:
        if lock.exists():
            lock.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--brain-cohort", type=Path, required=True)
    prepare.add_argument("--metadata-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--scene-dir", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--codex-exe", type=Path, default=Path("E:/codex-tools/bin/codex.exe"))
    run.add_argument("--model-dir", type=Path, default=Path("artifacts.local/models/grounding-dino-tiny-a2bb814"))
    args = parser.parse_args(argv)
    try:
        result = (
            prepare_responsive_replay(
                brain_cohort=args.brain_cohort,
                metadata_root=args.metadata_root,
                output_dir=args.output_dir,
            )
            if args.command == "prepare"
            else execute_responsive_replay(
                scene_dir=args.scene_dir,
                run_dir=args.run_dir,
                codex_exe=args.codex_exe,
                model_dir=args.model_dir,
            )
        )
    except (ContractError, ProviderAdapterError, OSError, requests.RequestException, json.JSONDecodeError) as error:
        parser.exit(2, f"fail-closed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
