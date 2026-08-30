"""Evaluate the mandatory C12 direct instance-only raster probe.

The input root is the instance shard itself: it must directly contain
``result.json`` and ``episodes/ep_XX``.  This evaluator does not start CARLA,
does not modify the candidate protocol, and does not create a formal evidence
tree.  It writes only the caller-named decision receipt.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import dtr_carla_c2_rich_scene as c2
import join_dtr_carla_c2_rich_scene as join
import materialize_dtr_carla_c12_x31_opaque_skin_protocol as c12


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "dtr_carla_c12_x31_opaque_skin_protocol.json"
PROBE_MET = c12.RASTER_PROBE_GATE_STATUS
PROBE_NOT_MET = "DTR_CARLA_C12_INSTANCE_RASTER_PROBE_GATE_NOT_MET"
RAW_COMPLETE = "DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE"
EXPECTED_FRAME_SCHEMA = "dtr-c2-raw-shard-frame-v1"
EXPECTED_RESULT_SCHEMA = "dtr-carla-c2-raw-shard-result-v1"
SCRIPTED_POSE_TOLERANCE_M = 1e-4
ANGLE_TOLERANCE_DEGREES = 1e-3
LOCAL_POSITION_TOLERANCE_M = 1e-9
BBOX_TOLERANCE_M = 1e-4
FRACTION_TOLERANCE = 1e-12
FAILURE_REPORT_LIMIT = 200


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank_jsonl_line:{path}:{line_number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"non_object_jsonl:{path}:{line_number}")
            values.append(value)
    return values


def serialized_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def angle_delta_degrees(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def normalized_xy(values: Iterable[Any]) -> tuple[float, float]:
    x, y = (float(value) for value in values)
    magnitude = math.hypot(x, y)
    require(magnitude > 0.0, "zero_length_anchor_axis")
    return x / magnitude, y / magnitude


def asset_trajectory(
    protocol: Mapping[str, Any],
    scenario: Mapping[str, Any],
    asset: Mapping[str, Any],
) -> Mapping[str, Any]:
    if "trajectory_key" in asset:
        name = str(scenario["asset_trajectories"][str(asset["trajectory_key"])])
    else:
        name = str(asset["trajectory"])
    return protocol["trajectory_library"][name]


def expected_asset_state(
    protocol: Mapping[str, Any],
    scenario: Mapping[str, Any],
    asset: Mapping[str, Any],
    time_s: float,
) -> dict[str, Any]:
    """Pure equivalent of the capture path's local XY/yaw trajectory math."""

    layout = protocol["layouts"][str(scenario["layout_id"])]
    anchor = layout["anchor"]
    center_x, center_y = (float(value) for value in anchor["center_xy_m"])
    forward_x, forward_y = normalized_xy(anchor["forward_xy"])
    right_x, right_y = normalized_xy(anchor["right_xy"])
    trajectory = asset_trajectory(protocol, scenario, asset)
    local_forward, local_right = c2.trajectory_position(dict(trajectory), time_s)
    velocity_forward, velocity_right = c2.trajectory_velocity(dict(trajectory), time_s)
    x = center_x + forward_x * local_forward + right_x * local_right
    y = center_y + forward_y * local_forward + right_y * local_right
    velocity_x = forward_x * velocity_forward + right_x * velocity_right
    velocity_y = forward_y * velocity_forward + right_y * velocity_right
    base_yaw = math.degrees(math.atan2(forward_y, forward_x))
    if "yaw_offset_degrees" in trajectory:
        yaw = base_yaw + float(trajectory["yaw_offset_degrees"])
    elif math.hypot(velocity_x, velocity_y) > 1e-9:
        yaw = math.degrees(math.atan2(velocity_y, velocity_x))
    else:
        yaw = base_yaw
    return {
        "local_forward_m": local_forward,
        "local_right_m": local_right,
        "x": x,
        "y": y,
        "yaw": yaw,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
    }


def convex_hull_from_bbox_vertices(vertices: Iterable[Mapping[str, Any]]) -> list[list[float]]:
    unique: dict[tuple[float, float], tuple[float, float]] = {}
    for vertex in vertices:
        key = (round(float(vertex["x"]), 6), round(float(vertex["y"]), 6))
        unique[key] = (float(vertex["x"]), float(vertex["y"]))
    points = sorted(unique.values())
    require(len(points) >= 3, "degenerate_proxy_bbox_vertices")

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return [[float(x), float(y)] for x, y in lower[:-1] + upper[:-1]]


def polygons_equal(
    first: Iterable[Iterable[Any]],
    second: Iterable[Iterable[Any]],
    *,
    tolerance: float,
) -> bool:
    first_values = [tuple(float(item) for item in point) for point in first]
    second_values = [tuple(float(item) for item in point) for point in second]
    return len(first_values) == len(second_values) and all(
        math.dist(left, right) <= tolerance
        for left, right in zip(first_values, second_values, strict=True)
    )


class Gate:
    def __init__(self, categories: Iterable[str]) -> None:
        self.categories = {name: True for name in categories}
        self.failure_count = 0
        self.failures: list[dict[str, Any]] = []

    def check(self, condition: bool, category: str, reason: str, **context: Any) -> None:
        require(category in self.categories, f"unknown_gate_category:{category}")
        if condition:
            return
        self.categories[category] = False
        self.failure_count += 1
        if len(self.failures) < FAILURE_REPORT_LIMIT:
            self.failures.append({"category": category, "reason": reason, **context})


def float_close(first: Any, second: Any, tolerance: float) -> bool:
    return abs(float(first) - float(second)) <= tolerance


def check_bbox(
    gate: Gate,
    category: str,
    value: Mapping[str, Any],
    *,
    expected_location: tuple[float, float, float],
    expected_extent: tuple[float, float, float],
    episode_id: str,
    sample_index: int,
    asset_key: str,
) -> None:
    location = value["location"]
    extent = value["extent"]
    for axis, expected in zip(("x", "y", "z"), expected_location, strict=True):
        gate.check(
            float_close(location[axis], expected, BBOX_TOLERANCE_M),
            category,
            "bbox_location_drift",
            episode_id=episode_id,
            sample_index=sample_index,
            asset_key=asset_key,
            axis=axis,
            expected=expected,
            actual=location[axis],
        )
    for axis, expected in zip(("x", "y", "z"), expected_extent, strict=True):
        gate.check(
            float_close(extent[axis], expected, BBOX_TOLERANCE_M),
            category,
            "bbox_extent_drift",
            episode_id=episode_id,
            sample_index=sample_index,
            asset_key=asset_key,
            axis=axis,
            expected=expected,
            actual=extent[axis],
        )


def check_actor_pose(
    gate: Gate,
    category: str,
    protocol: Mapping[str, Any],
    scenario: Mapping[str, Any],
    asset: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    expected_blueprint: str,
) -> None:
    episode_id = str(scenario["episode_id"])
    sample_index = int(row["sample_index"])
    asset_key = str(asset["asset_key"])
    actors = row["actors"]
    gate.check(
        asset_key in actors,
        category,
        "actor_missing",
        episode_id=episode_id,
        sample_index=sample_index,
        asset_key=asset_key,
    )
    if asset_key not in actors:
        return
    actor = actors[asset_key]
    expected = expected_asset_state(protocol, scenario, asset, float(row["time_s"]))
    for field, wanted in (
        ("asset_key", asset_key),
        ("track_id", str(asset["track_id"])),
        ("role", str(asset["role"])),
        ("kind", str(asset["kind"])),
        ("actual_blueprint", expected_blueprint),
    ):
        gate.check(
            str(actor.get(field)) == wanted,
            category,
            "actor_identity_drift",
            episode_id=episode_id,
            sample_index=sample_index,
            asset_key=asset_key,
            field=field,
            expected=wanted,
            actual=actor.get(field),
        )
    local = actor["local_position"]
    for field, wanted in (
        ("forward_m", expected["local_forward_m"]),
        ("right_m", expected["local_right_m"]),
    ):
        gate.check(
            float_close(local[field], wanted, LOCAL_POSITION_TOLERANCE_M),
            category,
            "local_position_drift",
            episode_id=episode_id,
            sample_index=sample_index,
            asset_key=asset_key,
            field=field,
            expected=wanted,
            actual=local[field],
        )
    command = actor["scripted_command_transform"]
    for field, wanted in (("x", expected["x"]), ("y", expected["y"])):
        gate.check(
            float_close(command[field], wanted, SCRIPTED_POSE_TOLERANCE_M),
            category,
            "scripted_command_position_drift",
            episode_id=episode_id,
            sample_index=sample_index,
            asset_key=asset_key,
            field=field,
            expected=wanted,
            actual=command[field],
        )
    gate.check(
        angle_delta_degrees(float(command["yaw"]), float(expected["yaw"]))
        <= ANGLE_TOLERANCE_DEGREES,
        category,
        "scripted_command_yaw_drift",
        episode_id=episode_id,
        sample_index=sample_index,
        asset_key=asset_key,
        expected=expected["yaw"],
        actual=command["yaw"],
    )
    velocity = actor["command_velocity"]
    for field, wanted in (
        ("x", expected["velocity_x"]),
        ("y", expected["velocity_y"]),
        ("z", 0.0),
    ):
        gate.check(
            float_close(velocity[field], wanted, SCRIPTED_POSE_TOLERANCE_M),
            category,
            "command_velocity_drift",
            episode_id=episode_id,
            sample_index=sample_index,
            asset_key=asset_key,
            field=field,
            expected=wanted,
            actual=velocity[field],
        )
    actual = actor["transform"]
    planar_error = math.hypot(
        float(actual["x"]) - float(command["x"]),
        float(actual["y"]) - float(command["y"]),
    )
    position_error = math.dist(
        (float(actual["x"]), float(actual["y"]), float(actual["z"])),
        (float(command["x"]), float(command["y"]), float(command["z"])),
    )
    angle_error = max(
        angle_delta_degrees(float(actual[field]), float(command[field]))
        for field in ("pitch", "yaw", "roll")
    )
    residual = actor["scripted_pose_residual"]
    gate.check(
        planar_error <= SCRIPTED_POSE_TOLERANCE_M
        and position_error <= SCRIPTED_POSE_TOLERANCE_M
        and angle_error <= ANGLE_TOLERANCE_DEGREES,
        category,
        "actual_pose_not_scripted_command",
        episode_id=episode_id,
        sample_index=sample_index,
        asset_key=asset_key,
        planar_error_m=planar_error,
        position_error_m=position_error,
        angle_error_degrees=angle_error,
    )
    gate.check(
        float_close(
            residual["planar_position_error_m"], planar_error, SCRIPTED_POSE_TOLERANCE_M
        )
        and float_close(
            residual["position_error_m"], position_error, SCRIPTED_POSE_TOLERANCE_M
        )
        and float_close(
            residual["angle_error_degrees"], angle_error, ANGLE_TOLERANCE_DEGREES
        ),
        category,
        "recorded_pose_residual_mismatch",
        episode_id=episode_id,
        sample_index=sample_index,
        asset_key=asset_key,
    )


def check_manifest_asset(
    gate: Gate,
    category: str,
    manifest_by_key: Mapping[str, Mapping[str, Any]],
    asset: Mapping[str, Any],
    *,
    episode_id: str,
    expected_blueprint: str,
    expected_extent: tuple[float, float, float],
) -> None:
    asset_key = str(asset["asset_key"])
    gate.check(
        asset_key in manifest_by_key,
        category,
        "manifest_asset_missing",
        episode_id=episode_id,
        asset_key=asset_key,
    )
    if asset_key not in manifest_by_key:
        return
    value = manifest_by_key[asset_key]
    expected = {
        "asset_key": asset_key,
        "track_id": str(asset["track_id"]),
        "role": str(asset["role"]),
        "kind": str(asset["kind"]),
        "template": str(asset["template"]),
        "actual_blueprint": expected_blueprint,
        "fallback_index": 0,
        "simulate_physics_disabled": True,
        "collisions_enabled": False,
        "bbox_nonzero": True,
    }
    for field, wanted in expected.items():
        gate.check(
            value.get(field) == wanted,
            category,
            "manifest_field_drift",
            episode_id=episode_id,
            asset_key=asset_key,
            field=field,
            expected=wanted,
            actual=value.get(field),
        )
    gate.check(
        value.get("candidate_blueprints") == [expected_blueprint],
        category,
        "manifest_blueprint_candidates_drift",
        episode_id=episode_id,
        asset_key=asset_key,
        expected=[expected_blueprint],
        actual=value.get("candidate_blueprints"),
    )
    extent = value["bbox_extent"]
    for axis, wanted in zip(("x", "y", "z"), expected_extent, strict=True):
        gate.check(
            float_close(extent[axis], wanted, BBOX_TOLERANCE_M),
            category,
            "manifest_bbox_extent_drift",
            episode_id=episode_id,
            asset_key=asset_key,
            axis=axis,
            expected=wanted,
            actual=extent[axis],
        )


def load_and_validate_candidate(protocol_path: Path) -> tuple[dict[str, Any], str]:
    base_path = c12.DEFAULT_BASE.resolve(strict=True)
    require(c2.sha256_file(base_path) == c12.PARENT_PROTOCOL_SHA256, "parent_file_hash")
    base = read_json(base_path)
    require(
        c2.sha256_json(base) == c12.PARENT_PROTOCOL_CANONICAL_SHA256,
        "parent_canonical_hash",
    )
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    expected = c12.materialize(base)
    require(protocol == expected, "candidate_not_exact_c12_materialization")
    require(
        protocol_bytes == c12.serialized_protocol(expected),
        "candidate_serialization_not_exact_c12_materialization",
    )
    c12.validate_c12(protocol, base)
    return protocol, c2.sha256_file(protocol_path)


def evaluate(
    capture_root: Path,
    protocol_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any], int]:
    protocol, protocol_sha256 = load_and_validate_candidate(protocol_path)
    protocol_sha256_before = c2.sha256_file(protocol_path)
    result_path = capture_root / "result.json"
    episodes_root = capture_root / "episodes"
    require(result_path.is_file(), f"direct_instance_result_missing:{result_path}")
    require(episodes_root.is_dir(), f"direct_instance_episodes_missing:{episodes_root}")
    require(not (capture_root / "shards").exists(), "capture_root_must_be_direct_instance_shard")
    result = read_json(result_path)
    scenario_by_episode = {
        str(value["episode_id"]): value for value in protocol["scenarios"]
    }
    contract_by_episode: dict[str, Mapping[str, Any]] = {}
    for contract in protocol["occlusion_contracts"]:
        require(len(contract["episodes"]) == 1, "c12_probe_contract_not_single_episode")
        episode_id = str(contract["episodes"][0])
        require(episode_id not in contract_by_episode, "c12_probe_duplicate_episode_contract")
        contract_by_episode[episode_id] = contract
    require(set(scenario_by_episode) == set(c12.EPISODE_IDS), "c12_probe_scenario_set")
    require(set(contract_by_episode) == set(c12.EPISODE_IDS), "c12_probe_contract_set")

    categories = (
        "direct_instance_capture_identity",
        "capture_file_integrity",
        "exact_panel_manifest",
        "exact_firetruck_proxy_manifest",
        "exact_panel_poses",
        "unchanged_firetruck_proxy_poses",
        "firetruck_proxy_los_authority",
        "zero_pixel_run_count_and_duration",
        "pre_trackability",
        "post_reappearance",
    )
    gate = Gate(categories)
    gate.check(
        result.get("schema_version") == EXPECTED_RESULT_SCHEMA,
        "direct_instance_capture_identity",
        "result_schema",
        actual=result.get("schema_version"),
    )
    gate.check(
        result.get("status") == RAW_COMPLETE,
        "direct_instance_capture_identity",
        "raw_capture_not_complete",
        actual=result.get("status"),
    )
    gate.check(
        result.get("sensor") == "instance",
        "direct_instance_capture_identity",
        "not_instance_only",
        actual=result.get("sensor"),
    )
    gate.check(
        result.get("protocol_sha256") == protocol_sha256,
        "direct_instance_capture_identity",
        "protocol_hash_mismatch",
        expected=protocol_sha256,
        actual=result.get("protocol_sha256"),
    )
    gate.check(
        result.get("experiment_id") == protocol["experiment_id"],
        "direct_instance_capture_identity",
        "experiment_id_mismatch",
        actual=result.get("experiment_id"),
    )
    gate.check(
        result.get("map") == protocol["environment"]["map"],
        "direct_instance_capture_identity",
        "map_mismatch",
        actual=result.get("map"),
    )
    gate.check(
        float_close(
            result.get("scripted_pose_planar_position_tolerance_m", math.inf),
            SCRIPTED_POSE_TOLERANCE_M,
            1e-12,
        ),
        "direct_instance_capture_identity",
        "scripted_pose_tolerance_drift",
        actual=result.get("scripted_pose_planar_position_tolerance_m"),
    )
    gate.check(
        result.get("scripted_pose_application") == "atomic_batch_before_sensor_tick",
        "direct_instance_capture_identity",
        "scripted_pose_application_drift",
        actual=result.get("scripted_pose_application"),
    )

    result_episode_by_id = {
        str(value["episode_id"]): value for value in result.get("episodes", [])
    }
    gate.check(
        set(result_episode_by_id) == set(c12.EPISODE_IDS),
        "direct_instance_capture_identity",
        "result_episode_set",
        actual=sorted(result_episode_by_id),
    )
    rows_by_episode: dict[str, list[dict[str, Any]]] = {}
    episode_reports: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    frames_hashes: dict[str, str] = {}

    for episode_id in c12.EPISODE_IDS:
        scenario = scenario_by_episode[episode_id]
        layout_id = str(scenario["layout_id"])
        episode_root = episodes_root / episode_id
        frames_path = episode_root / "frames.jsonl"
        asset_manifest_path = episode_root / "asset_manifest.json"
        episode_manifest_path = episode_root / "manifest.json"
        require(frames_path.is_file(), f"frames_missing:{episode_id}")
        require(asset_manifest_path.is_file(), f"asset_manifest_missing:{episode_id}")
        require(episode_manifest_path.is_file(), f"episode_manifest_missing:{episode_id}")
        frames_hashes[episode_id] = c2.sha256_file(frames_path)
        manifest_hashes[episode_id] = c2.sha256_file(asset_manifest_path)
        episode_manifest = read_json(episode_manifest_path)
        gate.check(
            episode_manifest.get("frames_sha256") == frames_hashes[episode_id]
            and episode_manifest.get("asset_manifest_sha256")
            == manifest_hashes[episode_id]
            and episode_manifest.get("sensor") == "instance"
            and episode_manifest.get("episode_id") == episode_id
            and episode_manifest.get("layout_id") == layout_id,
            "capture_file_integrity",
            "episode_manifest_mismatch",
            episode_id=episode_id,
        )
        rows = read_jsonl(frames_path)
        rows_by_episode[episode_id] = rows
        expected_frames = (
            int(
                round(
                    float(protocol["layouts"][layout_id]["duration_seconds"])
                    / float(protocol["environment"]["sample_seconds"])
                )
            )
            + 1
        )
        gate.check(
            len(rows) == expected_frames,
            "capture_file_integrity",
            "frame_count",
            episode_id=episode_id,
            expected=expected_frames,
            actual=len(rows),
        )
        result_episode = result_episode_by_id.get(episode_id)
        gate.check(
            result_episode is not None
            and result_episode.get("sensor") == "instance"
            and result_episode.get("layout_id") == layout_id
            and int(result_episode.get("frames", -1)) == expected_frames,
            "direct_instance_capture_identity",
            "result_episode_identity",
            episode_id=episode_id,
        )
        layout_assets = c2.materialize_layout_assets(protocol, layout_id)
        asset_by_key = {str(value["asset_key"]): value for value in layout_assets}
        panel_assets = [
            asset_by_key[c12.panel_key(layout_id, ordinal)]
            for ordinal in range(1, c12.PANEL_COUNT_PER_LAYOUT + 1)
        ]
        proxy_key = f"{layout_id}_occluder"
        proxy_asset = asset_by_key[proxy_key]
        expected_scripted_keys = sorted(
            str(value["asset_key"])
            for value in layout_assets
            if value.get("scripted_pose_authority") is True
        )
        if result_episode is not None:
            gate.check(
                result_episode.get("scripted_pose_authority_assets")
                == expected_scripted_keys,
                "direct_instance_capture_identity",
                "scripted_asset_set",
                episode_id=episode_id,
                expected=expected_scripted_keys,
                actual=result_episode.get("scripted_pose_authority_assets"),
            )

        manifest = read_json(asset_manifest_path)
        require(isinstance(manifest, list), f"asset_manifest_not_list:{episode_id}")
        manifest_by_key = {str(value["asset_key"]): value for value in manifest}
        expected_actor_keys = {"wearer", *asset_by_key}
        gate.check(
            len(manifest_by_key) == len(manifest)
            and set(manifest_by_key) == expected_actor_keys,
            "capture_file_integrity",
            "asset_manifest_key_set",
            episode_id=episode_id,
            expected_count=len(expected_actor_keys),
            actual_count=len(manifest),
        )
        for panel in panel_assets:
            check_manifest_asset(
                gate,
                "exact_panel_manifest",
                manifest_by_key,
                panel,
                episode_id=episode_id,
                expected_blueprint=c12.PANEL_BLUEPRINT,
                expected_extent=c12.PANEL_EVIDENCE_BBOX_EXTENT,
            )
        check_manifest_asset(
            gate,
            "exact_firetruck_proxy_manifest",
            manifest_by_key,
            proxy_asset,
            episode_id=episode_id,
            expected_blueprint=c12.PROXY_BLUEPRINT,
            expected_extent=c12.PROXY_BBOX_EXTENT,
        )

        for sample_index, row in enumerate(rows):
            gate.check(
                row.get("schema_version") == EXPECTED_FRAME_SCHEMA
                and row.get("sensor") == "instance"
                and row.get("episode_id") == episode_id
                and row.get("layout_id") == layout_id
                and int(row.get("sample_index", -1)) == sample_index
                and float_close(
                    row.get("time_s", math.inf),
                    sample_index * float(protocol["environment"]["sample_seconds"]),
                    1e-9,
                ),
                "capture_file_integrity",
                "frame_identity_or_sequence",
                episode_id=episode_id,
                sample_index=sample_index,
            )
            actors = row["actors"]
            gate.check(
                set(actors) == expected_actor_keys,
                "capture_file_integrity",
                "frame_actor_key_set",
                episode_id=episode_id,
                sample_index=sample_index,
            )
            for panel in panel_assets:
                check_actor_pose(
                    gate,
                    "exact_panel_poses",
                    protocol,
                    scenario,
                    panel,
                    row,
                    expected_blueprint=c12.PANEL_BLUEPRINT,
                )
                panel_key = str(panel["asset_key"])
                if panel_key in actors:
                    check_bbox(
                        gate,
                        "exact_panel_poses",
                        actors[panel_key]["bounding_box"],
                        expected_location=c12.PANEL_EVIDENCE_BBOX_LOCATION,
                        expected_extent=c12.PANEL_EVIDENCE_BBOX_EXTENT,
                        episode_id=episode_id,
                        sample_index=sample_index,
                        asset_key=panel_key,
                    )
            check_actor_pose(
                gate,
                "unchanged_firetruck_proxy_poses",
                protocol,
                scenario,
                proxy_asset,
                row,
                expected_blueprint=c12.PROXY_BLUEPRINT,
            )
            if proxy_key in actors:
                check_bbox(
                    gate,
                    "unchanged_firetruck_proxy_poses",
                    actors[proxy_key]["bounding_box"],
                    expected_location=c12.PROXY_BBOX_LOCATION,
                    expected_extent=c12.PROXY_BBOX_EXTENT,
                    episode_id=episode_id,
                    sample_index=sample_index,
                    asset_key=proxy_key,
                )
                polygons = row["truth"]["collision_polygons_xy"]
                gate.check(
                    proxy_key in polygons
                    and all(str(panel["asset_key"]) not in polygons for panel in panel_assets),
                    "firetruck_proxy_los_authority",
                    "proxy_or_panel_collision_polygon_authority",
                    episode_id=episode_id,
                    sample_index=sample_index,
                )
                if proxy_key in polygons:
                    computed_proxy_polygon = convex_hull_from_bbox_vertices(
                        actors[proxy_key]["bounding_box"]["world_vertices"]
                    )
                    gate.check(
                        polygons_equal(
                            polygons[proxy_key],
                            computed_proxy_polygon,
                            tolerance=BBOX_TOLERANCE_M,
                        ),
                        "firetruck_proxy_los_authority",
                        "proxy_truth_polygon_not_firetruck_bbox",
                        episode_id=episode_id,
                        sample_index=sample_index,
                    )

        contract = contract_by_episode[episode_id]
        target_key = str(contract["target_asset"])
        for row in rows:
            visibility = row["instance_visibility"][target_key]
            pixels = int(visibility["pixels"])
            fraction = float(visibility["pixel_fraction"])
            expected_fraction = pixels / (
                int(protocol["capture"]["resolution"][0])
                * int(protocol["capture"]["resolution"][1])
            )
            gate.check(
                pixels >= 0
                and float_close(fraction, expected_fraction, FRACTION_TOLERANCE)
                and bool(visibility["visible"]) == (pixels > 0),
                "capture_file_integrity",
                "instance_visibility_inconsistent",
                episode_id=episode_id,
                sample_index=row["sample_index"],
                target_key=target_key,
            )
        runs = join.physical_occlusion_runs(
            rows,
            target_key=target_key,
            occluder_key=proxy_key,
            fov_degrees=float(protocol["capture"]["fov_degrees"]),
            hidden_fraction=float(contract["complete_occlusion_pixel_fraction"]),
        )
        run_reports: list[dict[str, Any]] = []
        for run in runs:
            pre = join.consecutive_trackable_before(
                rows,
                end_index_exclusive=run[0],
                target_key=target_key,
                minimum_fraction=float(contract["minimum_trackable_pixel_fraction"]),
            )
            post = join.consecutive_trackable_after(
                rows,
                start_index_inclusive=run[-1] + 1,
                target_key=target_key,
                minimum_fraction=float(contract["minimum_trackable_pixel_fraction"]),
            )
            run_reports.append(
                {
                    "sample_indices": run,
                    "frames": len(run),
                    "duration_seconds": len(run)
                    * float(protocol["environment"]["sample_seconds"]),
                    "pre_track_sample_indices": pre,
                    "pre_track_frames": len(pre),
                    "post_reappearance_sample_indices": post,
                    "post_reappearance_frames": len(post),
                    "all_target_pixels_exactly_zero": all(
                        int(rows[index]["instance_visibility"][target_key]["pixels"])
                        == 0
                        for index in run
                    ),
                }
            )
        exactly_one = len(run_reports) == 1
        gate.check(
            exactly_one,
            "zero_pixel_run_count_and_duration",
            "physical_zero_pixel_run_count",
            episode_id=episode_id,
            actual=len(run_reports),
        )
        selected = run_reports[0] if exactly_one else None
        if selected is not None:
            gate.check(
                6 <= int(selected["frames"]) <= 13
                and 0.6 - 1e-9
                <= float(selected["duration_seconds"])
                <= 1.3 + 1e-9
                and bool(selected["all_target_pixels_exactly_zero"]),
                "zero_pixel_run_count_and_duration",
                "physical_zero_pixel_run_duration",
                episode_id=episode_id,
                run=selected,
            )
            gate.check(
                int(selected["pre_track_frames"]) >= 10,
                "pre_trackability",
                "insufficient_consecutive_pre_track_frames",
                episode_id=episode_id,
                actual=selected["pre_track_frames"],
            )
            gate.check(
                int(selected["post_reappearance_frames"]) >= 8,
                "post_reappearance",
                "insufficient_consecutive_post_reappearance_frames",
                episode_id=episode_id,
                actual=selected["post_reappearance_frames"],
            )
        else:
            gate.check(False, "pre_trackability", "no_unique_physical_zero_pixel_run", episode_id=episode_id)
            gate.check(False, "post_reappearance", "no_unique_physical_zero_pixel_run", episode_id=episode_id)
        episode_reports.append(
            {
                "episode_id": episode_id,
                "layout_id": layout_id,
                "contract_id": str(contract["contract_id"]),
                "target_asset": target_key,
                "line_of_sight_proxy_asset": proxy_key,
                "line_of_sight_proxy_blueprint": c12.PROXY_BLUEPRINT,
                "physical_zero_pixel_runs": run_reports,
                "selected": selected,
                "passed": (
                    selected is not None
                    and 6 <= int(selected["frames"]) <= 13
                    and int(selected["pre_track_frames"]) >= 10
                    and int(selected["post_reappearance_frames"]) >= 8
                    and bool(selected["all_target_pixels_exactly_zero"])
                ),
            }
        )

    combined_manifest_path = capture_root / "episode_asset_manifests.json"
    gate.check(
        combined_manifest_path.is_file()
        and result.get("episode_asset_manifests_sha256")
        == (c2.sha256_file(combined_manifest_path) if combined_manifest_path.is_file() else None),
        "capture_file_integrity",
        "combined_asset_manifest_hash",
    )
    gate.check(
        result.get("checks", {}).get("all_captured_frames_match_authoritative_scripted_pose")
        is True
        and result.get("checks", {}).get("zero_blueprint_fallbacks") is True
        and result.get("checks", {}).get("all_spawned_assets_have_nonzero_bbox") is True,
        "direct_instance_capture_identity",
        "raw_capture_required_checks",
        actual=result.get("checks"),
    )

    gate_met = all(gate.categories.values()) and all(
        value["passed"] for value in episode_reports
    )
    receipt_core = {
        "schema_version": "dtr-carla-c12-instance-raster-probe-receipt-v1",
        "status": PROBE_MET if gate_met else PROBE_NOT_MET,
        "gate_met": gate_met,
        "authority": {
            "purpose": "C12_INSTANCE_ONLY_RASTER_REACHABILITY_PRELAUNCH_GATE",
            "probe_is_not_formal_source_or_algorithm_evidence": True,
            "probe_pixels_must_not_be_reused_as_formal_capture": True,
            "formal_launch_gate_satisfied_by_this_receipt": gate_met,
            "formal_launch_started": False,
            "detector_model_x24_x31_route_or_scorer_opened": False,
        },
        "candidate_protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
            "cohort_id": protocol["cohort_id"],
            "capture_seed": protocol["capture"]["seed"],
            "exact_c12_materialization_validated": True,
            "unchanged_after_evaluation": c2.sha256_file(protocol_path)
            == protocol_sha256_before,
        },
        "direct_instance_capture": {
            "root": str(capture_root),
            "result_sha256": c2.sha256_file(result_path),
            "sensor": result.get("sensor"),
            "status": result.get("status"),
            "frames_sha256_by_episode": frames_hashes,
            "asset_manifest_sha256_by_episode": manifest_hashes,
        },
        "frozen_gate": {
            "episode_count": 8,
            "exact_physical_zero_pixel_runs_per_episode": 1,
            "minimum_complete_occlusion_frames": 6,
            "maximum_complete_occlusion_frames": 13,
            "minimum_complete_occlusion_seconds": 0.6,
            "maximum_complete_occlusion_seconds": 1.3,
            "minimum_pre_track_frames": 10,
            "minimum_post_reappearance_frames": 8,
            "minimum_trackable_pixel_fraction": 0.0002,
            "complete_occlusion_pixel_fraction": 0.0,
            "line_of_sight_proxy_blueprint": c12.PROXY_BLUEPRINT,
            "panel_blueprint": c12.PANEL_BLUEPRINT,
            "panel_count_per_layout": c12.PANEL_COUNT_PER_LAYOUT,
            "panel_fallback_index": 0,
        },
        "unchanged_join_logic": {
            "module": str(Path(join.__file__).resolve()),
            "module_sha256": c2.sha256_file(Path(join.__file__).resolve()),
            "functions": [
                "physical_occlusion_runs",
                "consecutive_trackable_before",
                "consecutive_trackable_after",
            ],
            "logic_copied_or_modified": False,
        },
        "checks": gate.categories,
        "failure_count": gate.failure_count,
        "failures_truncated": gate.failure_count > len(gate.failures),
        "failures": gate.failures,
        "episodes": episode_reports,
        "orientation_boundary": {
            "yaw_offset_degrees_by_layout": c12.PANEL_YAW_OFFSET_BY_LAYOUT,
            "exact_scripted_yaw_validated": True,
            "front_back_material_orientation_is_decided_only_by_raster_result": True,
        },
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": c2.sha256_json(receipt_core),
    }
    output_bytes = serialized_json(receipt)
    require(output_path.resolve() != protocol_path.resolve(), "output_must_not_be_protocol")
    require(output_path.parent.is_dir(), "caller_must_precreate_ignored_output_parent")
    if output_path.exists() and output_path.read_bytes() != output_bytes:
        raise FileExistsError(f"existing probe receipt differs: {output_path}")
    if not output_path.exists():
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_bytes(output_bytes)
        temporary.replace(output_path)
    require(c2.sha256_file(protocol_path) == protocol_sha256_before, "protocol_changed")
    return receipt, 0 if gate_met else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-root",
        type=Path,
        required=True,
        help="Direct instance shard containing result.json and episodes/ (not its parent).",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Caller-selected ignored JSON receipt path; its parent must already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_root = args.capture_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    output_path = args.output.resolve()
    receipt, exit_code = evaluate(capture_root, protocol_path, output_path)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "gate_met": receipt["gate_met"],
                "failure_count": receipt["failure_count"],
                "output": str(output_path),
                "formal_launch_started": False,
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
