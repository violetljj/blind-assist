"""Pure protocol/evidence helpers for the CARLA C2 rich-scene source."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_ID = "DTR_CARLA_C2_RICH_MULTILAYOUT_OCCLUSION_SOURCE_V2"
MODEL_TOP_LEVEL_ALLOWLIST = {
    "schema_version",
    "episode_id",
    "sample_index",
    "world_frame",
    "time_s",
    "timestamp_s",
    "wearable_rgb",
    "metric_depth",
    "camera",
    "wearer_pose_current",
    "navigation",
    "frame_alignment",
}
FORBIDDEN_MODEL_KEYS = {
    "actor",
    "actor_id",
    "actor_ids",
    "actors",
    "bbox",
    "bounding_box",
    "collision_polygons_xy",
    "contact",
    "contact_label",
    "current_actors",
    "current_contact",
    "expected_outcome",
    "future_contact_within_horizon",
    "instance",
    "instance_segmentation",
    "instance_visibility",
    "layout_id",
    "layout_role",
    "occlusion",
    "occlusion_label",
    "realized_future",
    "responsible_asset",
    "responsible_assets",
    "role",
    "scenario_role",
    "semantic_role",
    "track_id",
    "truth",
    "twin_role",
    "velocity",
    "witness",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _rounded_alignment_value(value: Any, digits: int = 5) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {
            key: _rounded_alignment_value(child, digits)
            for key, child in sorted(value.items())
        }
    if isinstance(value, list):
        return [_rounded_alignment_value(child, digits) for child in value]
    return value


def build_rgbd_alignment_receipt(
    wearable_by_episode: dict[str, list[dict[str, Any]]],
    depth_by_episode: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Prove a deterministic cross-server RGB/depth replay alignment."""

    if set(wearable_by_episode) != set(depth_by_episode):
        raise ValueError("RGB/depth episode sets differ")
    episodes: list[dict[str, Any]] = []
    for episode_id in sorted(wearable_by_episode):
        wearable_rows = wearable_by_episode[episode_id]
        depth_rows = depth_by_episode[episode_id]
        if not wearable_rows or len(wearable_rows) != len(depth_rows):
            raise ValueError(f"RGB/depth frame counts differ for {episode_id}")
        offsets: set[int] = set()
        projection: list[dict[str, Any]] = []
        wearable_source_frames: list[int] = []
        depth_source_frames: list[int] = []
        for wearable, depth in zip(wearable_rows, depth_rows, strict=True):
            sample_index = int(wearable["sample_index"])
            if sample_index != int(depth["sample_index"]):
                raise ValueError(f"RGB/depth sample indices differ for {episode_id}")
            time_s = round(float(wearable["time_s"]), 8)
            if time_s != round(float(depth["time_s"]), 8):
                raise ValueError(f"RGB/depth logical timestamps differ for {episode_id}")
            camera = _rounded_alignment_value(wearable["camera_transform"])
            wearer = _rounded_alignment_value(wearable["wearer_transform"])
            if camera != _rounded_alignment_value(depth["camera_transform"]):
                raise ValueError(f"RGB/depth camera replay differs for {episode_id}")
            if wearer != _rounded_alignment_value(depth["wearer_transform"]):
                raise ValueError(f"RGB/depth wearer replay differs for {episode_id}")
            wearable_frame = int(wearable["world_frame"])
            depth_frame = int(depth["world_frame"])
            offsets.add(depth_frame - wearable_frame)
            wearable_source_frames.append(wearable_frame)
            depth_source_frames.append(depth_frame)
            projection.append(
                {
                    "episode_id": episode_id,
                    "sample_index": sample_index,
                    "time_s": time_s,
                    "world_frame": wearable_frame,
                    "camera_world_transform": camera,
                    "wearer_pose_current": wearer,
                }
            )
        if len(offsets) != 1:
            raise ValueError(f"RGB/depth source-frame offset varies for {episode_id}")
        if any(
            right != left + 1
            for left, right in zip(
                wearable_source_frames, wearable_source_frames[1:]
            )
        ) or any(
            right != left + 1
            for left, right in zip(depth_source_frames, depth_source_frames[1:])
        ):
            raise ValueError(f"RGB/depth source frames are not contiguous for {episode_id}")
        episodes.append(
            {
                "episode_id": episode_id,
                "frames": len(wearable_rows),
                "wearable_source_world_frame_first": wearable_source_frames[0],
                "wearable_source_world_frame_last": wearable_source_frames[-1],
                "depth_source_world_frame_first": depth_source_frames[0],
                "depth_source_world_frame_last": depth_source_frames[-1],
                "depth_minus_wearable_source_world_frame_offset": next(iter(offsets)),
                "alignment_projection_sha256": sha256_bytes(
                    canonical_json_bytes(projection)
                ),
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": (
            "dtr-c2-model-rgbd-deterministic-replay-alignment-receipt-v1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "authority": "DETERMINISTIC_REPLAY_ALIGNMENT_VERIFIED",
        "world_frame_rule": (
            "world_frame equals wearable_rgb.source_world_frame; metric_depth is "
            "mapped into that namespace by the verified per-episode source offset"
        ),
        "matching_keys": ["episode_id", "sample_index", "time_s"],
        "verified_equal_fields": [
            "camera_world_transform",
            "wearer_pose_current",
        ],
        "episodes": episodes,
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    return receipt


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def trajectory_segments(trajectory: dict[str, Any]) -> list[dict[str, float]]:
    values = sorted(trajectory["segments"], key=lambda item: float(item["start_s"]))
    if not values or abs(float(values[0]["start_s"])) > 1e-9:
        raise ValueError("every trajectory must begin at t=0")
    starts = [float(item["start_s"]) for item in values]
    if starts != sorted(set(starts)):
        raise ValueError("trajectory segment starts must be unique and ordered")
    return values


def trajectory_position(trajectory: dict[str, Any], time_s: float) -> tuple[float, float]:
    forward = float(trajectory["start_forward_m"])
    right = float(trajectory["start_right_m"])
    values = trajectory_segments(trajectory)
    for index, segment in enumerate(values):
        start_s = float(segment["start_s"])
        if time_s <= start_s:
            break
        end_s = (
            min(time_s, float(values[index + 1]["start_s"]))
            if index + 1 < len(values)
            else time_s
        )
        if end_s <= start_s:
            continue
        elapsed = end_s - start_s
        forward += float(segment["velocity_forward_mps"]) * elapsed
        right += float(segment["velocity_right_mps"]) * elapsed
        if end_s >= time_s:
            break
    return forward, right


def trajectory_velocity(trajectory: dict[str, Any], time_s: float) -> tuple[float, float]:
    selected = trajectory_segments(trajectory)[0]
    for segment in trajectory_segments(trajectory):
        if float(segment["start_s"]) <= time_s + 1e-9:
            selected = segment
        else:
            break
    return (
        float(selected["velocity_forward_mps"]),
        float(selected["velocity_right_mps"]),
    )


def trajectory_prefix_equal(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    end_s: float,
    sample_s: float,
    tolerance: float = 1e-9,
) -> bool:
    steps = int(round(end_s / sample_s))
    for index in range(steps + 1):
        time_s = index * sample_s
        if math.dist(
            trajectory_position(first, time_s), trajectory_position(second, time_s)
        ) > tolerance:
            return False
        if time_s < end_s - tolerance and math.dist(
            trajectory_velocity(first, time_s), trajectory_velocity(second, time_s)
        ) > tolerance:
            return False
    return True


def build_plan_receipt(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    payload = {
        "schema_version": "dtr-c2-plan-receipt-v1",
        "plan_id": str(plan["plan_id"]),
        "session_id": str(plan["session_id"]),
        "issued_at_s": float(plan["issued_at_s"]),
        "expires_at_s": float(plan["expires_at_s"]),
        "coordinate_frame": "LAYOUT_FORWARD_RIGHT",
        "time_parameterized_waypoints": [
            {
                "time_s": float(item["time_s"]),
                "forward_m": float(item["forward_m"]),
                "right_m": float(item["right_m"]),
            }
            for item in plan["time_parameterized_waypoints"]
        ],
    }
    return {**payload, "receipt_sha256": sha256_json(payload)}


def plan_waypoints_world(
    receipt: dict[str, Any] | None, anchor: dict[str, Any]
) -> list[dict[str, float]]:
    if receipt is None:
        return []
    center = [float(value) for value in anchor["center_xy_m"]]
    forward = [float(value) for value in anchor["forward_xy"]]
    right = [float(value) for value in anchor["right_xy"]]
    values: list[dict[str, float]] = []
    for waypoint in receipt["time_parameterized_waypoints"]:
        local_forward = float(waypoint["forward_m"])
        local_right = float(waypoint["right_m"])
        values.append(
            {
                "time_s": float(waypoint["time_s"]),
                "x_m": center[0]
                + forward[0] * local_forward
                + right[0] * local_right,
                "y_m": center[1]
                + forward[1] * local_forward
                + right[1] * local_right,
            }
        )
    return values


def materialize_layout_assets(
    protocol: dict[str, Any], layout_id: str
) -> list[dict[str, Any]]:
    templates = protocol["asset_templates"]
    layout = protocol["layouts"][layout_id]
    values: list[dict[str, Any]] = []
    for instance in layout["assets"]:
        template_name = str(instance["template"])
        if template_name not in templates:
            raise ValueError(f"unknown asset template: {template_name}")
        merged = dict(templates[template_name])
        merged.update(instance)
        merged["template"] = template_name
        values.append(merged)
    return values


def layout_receipt(protocol: dict[str, Any], layout_id: str) -> dict[str, Any]:
    layout = protocol["layouts"][layout_id]
    payload = {
        "schema_version": "dtr-c2-layout-receipt-v1",
        "layout_id": layout_id,
        "anchor": layout["anchor"],
        "witness": layout["witness"],
        "weather": layout["weather"],
        "assets": materialize_layout_assets(protocol, layout_id),
    }
    return {**payload, "receipt_sha256": sha256_json(payload)}


def camera_intrinsics(width: int, height: int, fov_degrees: float) -> list[list[float]]:
    focal = float(width) / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return [
        [focal, 0.0, float(width) / 2.0],
        [0.0, focal, float(height) / 2.0],
        [0.0, 0.0, 1.0],
    ]


def contiguous_runs(indices: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(int(value) for value in indices))
    if not ordered:
        return []
    runs: list[list[int]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return runs


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    sx, sy = start
    ex, ey = end
    px, py = point
    dx, dy = ex - sx, ey - sy
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.dist(point, start)
    ratio = ((px - sx) * dx + (py - sy) * dy) / denominator
    ratio = min(1.0, max(0.0, ratio))
    return math.hypot(px - (sx + ratio * dx), py - (sy + ratio * dy))


def _point_in_convex_polygon(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    signs: list[float] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        signs.append(
            (end[0] - start[0]) * (point[1] - start[1])
            - (end[1] - start[1]) * (point[0] - start[0])
        )
    return all(value >= -1e-7 for value in signs) or all(
        value <= 1e-7 for value in signs
    )


def point_polygon_distance(point: tuple[float, float], polygon: list[list[float]]) -> float:
    values = [(float(item[0]), float(item[1])) for item in polygon]
    if len(values) < 3:
        raise ValueError("a contact polygon requires at least three vertices")
    if _point_in_convex_polygon(point, values):
        return 0.0
    return min(
        _point_segment_distance(point, values[index], values[(index + 1) % len(values)])
        for index in range(len(values))
    )


def contact_union(
    wearer_xy: tuple[float, float],
    polygons: dict[str, list[list[float]]],
    *,
    wearer_radius_m: float,
) -> tuple[bool, float, list[str]]:
    distances = {
        key: point_polygon_distance(wearer_xy, polygon)
        for key, polygon in polygons.items()
    }
    if not distances:
        return False, math.inf, []
    minimum = min(distances.values())
    contacts = sorted(
        key for key, distance in distances.items() if distance <= wearer_radius_m + 1e-9
    )
    return bool(contacts), minimum, contacts


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _on_segment(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return (
        min(first[0], second[0]) - 1e-7 <= point[0] <= max(first[0], second[0]) + 1e-7
        and min(first[1], second[1]) - 1e-7
        <= point[1]
        <= max(first[1], second[1]) + 1e-7
        and abs(_orientation(first, second, point)) <= 1e-7
    )


def segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    values = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    if values[0] * values[1] < 0.0 and values[2] * values[3] < 0.0:
        return True
    return (
        (abs(values[0]) <= 1e-7 and _on_segment(c, a, b))
        or (abs(values[1]) <= 1e-7 and _on_segment(d, a, b))
        or (abs(values[2]) <= 1e-7 and _on_segment(a, c, d))
        or (abs(values[3]) <= 1e-7 and _on_segment(b, c, d))
    )


def line_intersects_polygon(
    start: tuple[float, float],
    end: tuple[float, float],
    polygon: list[list[float]],
) -> bool:
    points = [(float(value[0]), float(value[1])) for value in polygon]
    return any(
        segments_intersect(start, end, points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def angle_delta_degrees(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def forbidden_model_paths(value: Any, prefix: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            lowered = str(key).lower()
            if lowered in FORBIDDEN_MODEL_KEYS or lowered.endswith("_actor_id"):
                failures.append(path)
            failures.extend(forbidden_model_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(forbidden_model_paths(child, f"{prefix}[{index}]"))
    return failures


def validate_model_record(record: dict[str, Any]) -> None:
    extra = sorted(set(record) - MODEL_TOP_LEVEL_ALLOWLIST)
    if extra:
        raise ValueError(f"model record has non-allowlisted top-level keys: {extra}")
    missing = sorted(MODEL_TOP_LEVEL_ALLOWLIST - set(record))
    if missing:
        raise ValueError(f"model record is missing required keys: {missing}")
    forbidden = forbidden_model_paths(record)
    if forbidden:
        raise ValueError(f"model record contains evaluator truth: {forbidden[:10]}")


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected C2 protocol identity")
    capture = protocol["capture"]
    resolution = [int(value) for value in capture["resolution"]]
    required_resolution = [int(value) for value in protocol["admission"]["required_resolution"]]
    if resolution != [1280, 720] or resolution != required_resolution:
        raise ValueError("C2 formal sensor resolution must be exactly 1280x720")
    expected_k = camera_intrinsics(1280, 720, float(capture["fov_degrees"]))
    calibration = capture["camera_calibration"]
    if any(
        abs(float(actual) - float(expected)) > 1e-9
        for actual, expected in zip(
            calibration["focal_length_pixels"],
            [expected_k[0][0], expected_k[1][1]],
            strict=True,
        )
    ):
        raise ValueError("frozen camera focal length does not match width/FOV")
    if [float(value) for value in calibration["principal_point"]] != [640.0, 360.0]:
        raise ValueError("frozen principal point does not match 1280x720")

    layouts = protocol.get("layouts", {})
    scenarios = list(protocol.get("scenarios", []))
    if len(layouts) != int(protocol["admission"]["expected_layout_count"]):
        raise ValueError("layout count does not match admission contract")
    if len(scenarios) != int(protocol["admission"]["expected_episode_count"]):
        raise ValueError("episode count does not match admission contract")
    episode_ids = [str(value["episode_id"]) for value in scenarios]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode IDs must be unique")

    known_trajectories = set(protocol["trajectory_library"])
    minimum_assets = int(
        protocol["admission"]["minimum_active_assets_per_layout_excluding_wearer"]
    )
    preferred_blueprints = {
        str(protocol["wearer"]["blueprint_candidates"][0])
    }
    for layout_id, layout in layouts.items():
        anchor = layout["anchor"]
        forward = [float(value) for value in anchor["forward_xy"]]
        right = [float(value) for value in anchor["right_xy"]]
        if abs(math.hypot(*forward) - 1.0) > 1e-3:
            raise ValueError(f"{layout_id} forward axis is not normalized")
        if abs(math.hypot(*right) - 1.0) > 1e-3:
            raise ValueError(f"{layout_id} right axis is not normalized")
        if abs(forward[0] * right[0] + forward[1] * right[1]) > 1e-3:
            raise ValueError(f"{layout_id} axes are not orthogonal")
        assets = materialize_layout_assets(protocol, layout_id)
        if len(assets) < minimum_assets:
            raise ValueError(f"{layout_id} has only {len(assets)} active assets")
        keys = [str(value["asset_key"]) for value in assets]
        tracks = [str(value["track_id"]) for value in assets]
        if len(keys) != len(set(keys)) or len(tracks) != len(set(tracks)):
            raise ValueError(f"{layout_id} asset keys and track IDs must be unique")
        for asset in assets:
            if not any(name in asset for name in ("fixed_pose", "trajectory", "trajectory_key")):
                raise ValueError(f"{layout_id}/{asset['asset_key']} has no pose source")
            if "trajectory" in asset and str(asset["trajectory"]) not in known_trajectories:
                raise ValueError(f"unknown trajectory on {layout_id}/{asset['asset_key']}")
            preferred_blueprints.add(str(asset["blueprint_candidates"][0]))
        receipt = layout_receipt(protocol, layout_id)
        if not receipt["receipt_sha256"]:
            raise ValueError(f"{layout_id} receipt is empty")

    scenario_map = {str(value["episode_id"]): value for value in scenarios}
    for scenario in scenarios:
        layout_id = str(scenario["layout_id"])
        if layout_id not in layouts:
            raise ValueError(f"unknown layout: {layout_id}")
        if str(scenario["wearer_trajectory"]) not in known_trajectories:
            raise ValueError(f"unknown wearer trajectory: {scenario['wearer_trajectory']}")
        active = {
            str(value["asset_key"]): value
            for value in materialize_layout_assets(protocol, layout_id)
        }
        for key, trajectory_name in scenario["asset_trajectories"].items():
            if str(key) not in active:
                raise ValueError(f"{scenario['episode_id']} overrides unknown asset {key}")
            if str(trajectory_name) not in known_trajectories:
                raise ValueError(f"{scenario['episode_id']} uses unknown trajectory")
        navigation_session_id = str(scenario.get("navigation_session_id", ""))
        if not navigation_session_id:
            raise ValueError(f"{scenario['episode_id']} lacks a navigation session")
        receipt = build_plan_receipt(scenario.get("issued_plan"))
        if receipt is not None and receipt["session_id"] != navigation_session_id:
            raise ValueError(
                f"{scenario['episode_id']} plan/navigation session mismatch"
            )

    for pair in protocol.get("twin_contracts", []):
        first = scenario_map[str(pair["a"])]
        second = scenario_map[str(pair["b"])]
        if first["layout_id"] != second["layout_id"]:
            raise ValueError("a twin pair must use the same layout")
        end_s = float(pair["identical_before_s"])
        sample_s = float(protocol["environment"]["sample_seconds"])
        if not trajectory_prefix_equal(
            protocol["trajectory_library"][first["wearer_trajectory"]],
            protocol["trajectory_library"][second["wearer_trajectory"]],
            end_s=end_s,
            sample_s=sample_s,
        ):
            raise ValueError("twin wearer prefixes differ")
        keys = set(first["asset_trajectories"]) | set(second["asset_trajectories"])
        for key in keys:
            if key not in first["asset_trajectories"] or key not in second["asset_trajectories"]:
                raise ValueError(f"twin trajectory override set differs for {key}")
            if not trajectory_prefix_equal(
                protocol["trajectory_library"][first["asset_trajectories"][key]],
                protocol["trajectory_library"][second["asset_trajectories"][key]],
                end_s=end_s,
                sample_s=sample_s,
            ):
                raise ValueError(f"twin prefix differs for {key}")

    minimum_unique = int(
        protocol["admission"]["minimum_unique_actual_blueprints_across_pack"]
    )
    if len(preferred_blueprints) < minimum_unique:
        raise ValueError(
            f"protocol has only {len(preferred_blueprints)} distinct preferred blueprints"
        )
    if bool(protocol["model_contract"].get("include_current_actors")):
        raise ValueError("current_actors must be disabled for the C2 model contract")
    if not protocol.get("occlusion_contracts"):
        raise ValueError("C2 requires a track-before-occlusion contract")
