"""Pure validation and per-map C2 compilation for the CARLA C4 scene bank."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from dtr_carla_c2_rich_scene import (
    EXPERIMENT_ID as C2_EXPERIMENT_ID,
    canonical_json_bytes,
    point_polygon_distance,
    trajectory_prefix_equal,
    validate_protocol as validate_c2_protocol,
)


C4_EXPERIMENT_ID = "DTR_CARLA_C4_MULTIMAP_WORLD_PACK_V1"
ASSET_SCHEMA = "dtr-c4-asset-registry-v1"
SCENE_SCHEMA = "dtr-c4-scene-registry-v1"
COMPILED_SCHEMA = "dtr-c4-per-map-c2-protocol-index-v1"
RECEIPT_SCHEMA = "dtr-c4-static-compiler-receipt-v1"
FORMAL_STATUS = "STATIC_COMPILE_ADMITTED"
DYNAMIC_FOOTPRINT_SCHEMA = "dtr-c4-captured-dynamic-footprint-receipt-v1"
C3_FOOTPRINT_SOURCE_RUN_ID = "c3-dynamic-risk-20260830-165705"
C3_FOOTPRINT_SOURCE_EVIDENCE_SHA256 = (
    "4681346104C1F870A3CBAD7865F0369B9AC7236CC847DC352AEE01A5D6767574"
)
C3_FOOTPRINT_SOURCE_FRAMES_SHA256 = (
    "F6BAFBBA2A3D5B0DF9B3DD915E928D5C8EFBB1085782793CD2E8A24A72019EC4"
)
C3_FOOTPRINT_CANONICAL_SHA256 = (
    "847209CCD6A23317C1760B7C505EA3C2FC933FD880D176E085336995E169D7EE"
)

PACKAGED_MAPS = {
    "Carla/Maps/Town01",
    "Carla/Maps/Town02",
    "Carla/Maps/Town03_Opt",
    "Carla/Maps/Town04",
    "Carla/Maps/Town05",
    "Carla/Maps/Town10HD_Opt",
}
REQUIRED_SCENE_CLASSES = {
    "narrow_alley",
    "mall_exit",
    "parking_lot",
    "bus_stop",
    "construction_zone",
    "rainy_night",
    "backlight",
    "crowded_pedestrians",
}
CLASS_PARAMETER_KEYS = {
    "narrow_alley": {"clear_width_m", "occlusion_mode"},
    "mall_exit": {"exit_width_m", "release_rate_targets_per_s"},
    "parking_lot": {"parking_aisle_width_m", "parked_vehicle_rows"},
    "bus_stop": {"stop_length_m", "boarding_zone_width_m"},
    "construction_zone": {"chicane_width_m", "barrier_count"},
    "rainy_night": {"minimum_precipitation", "minimum_wetness"},
    "backlight": {
        "maximum_camera_sun_delta_degrees",
        "maximum_sun_altitude_degrees",
    },
    "crowded_pedestrians": {
        "minimum_dynamic_pedestrians",
        "minimum_dynamic_targets",
    },
}
IDENTITY_FORBIDDEN_TOKENS = {
    "contact",
    "safe",
    "outcome",
    "collision",
    "positive",
    "negative",
    "hit",
    "miss",
}
WEATHER_PARAMETER_KEYS = {
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "sun_azimuth_angle",
    "sun_altitude_angle",
    "fog_density",
    "wetness",
}
XODR_SHA256_BY_MAP = {
    "Carla/Maps/Town01": "97a7f6ac67812567e5c8ee0599cd823b23f80f30f3f97c502212e38b72e2b709",
    "Carla/Maps/Town02": "953c05f17def231239ffcadba3307628a82d0d335b67b4e29c3098f7aed1dd7d",
    "Carla/Maps/Town03_Opt": "abf177bcd9b66cfefb7ac4919f6456815f1384f4229f1343c8a6a90bbb7713c9",
    "Carla/Maps/Town04": "4dddd1560ab7812b0e89cd4a523d8b6f2b93947eafb496d9eeb83cc772116c85",
    "Carla/Maps/Town05": "7edc53b0be840e177598da7a4ed9571146d7b2595ce280d154c90f92e8a48851",
}


def load_json(path: Path) -> dict[str, Any]:
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        folded: dict[str, str] = {}
        for key, value in pairs:
            normalized = key.casefold()
            if normalized in folded:
                raise ValueError(
                    f"duplicate JSON key: {folded[normalized]!r} and {key!r}"
                )
            folded[normalized] = key
            output[key] = value
        return output

    value = json.loads(
        path.read_text(encoding="utf-8-sig"), object_pairs_hook=no_duplicate_keys
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def materialized_json_sha256(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    payload = text.replace("\n", os.linesep).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    _require(not missing and not extra, f"{label} keys differ: missing={missing} extra={extra}")


def _casefold_unique(values: list[str], label: str) -> None:
    normalized = [value.casefold() for value in values]
    _require(len(normalized) == len(set(normalized)), f"{label} are not unique")


def _identity_is_neutral(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold())
    return not any(token in normalized for token in IDENTITY_FORBIDDEN_TOKENS)


def _has_nonzero_motion(trajectory: dict[str, Any]) -> bool:
    return any(
        abs(float(segment["velocity_forward_mps"])) > 1e-9
        or abs(float(segment["velocity_right_mps"])) > 1e-9
        for segment in trajectory["segments"]
    )


def _trajectory_position(
    trajectory: dict[str, Any], time_s: float
) -> tuple[float, float]:
    forward = float(trajectory["start_forward_m"])
    right = float(trajectory["start_right_m"])
    segments = trajectory["segments"]
    for index, segment in enumerate(segments):
        start_s = float(segment["start_s"])
        next_start_s = (
            float(segments[index + 1]["start_s"])
            if index + 1 < len(segments)
            else time_s
        )
        elapsed_s = max(0.0, min(time_s, next_start_s) - start_s)
        forward += float(segment["velocity_forward_mps"]) * elapsed_s
        right += float(segment["velocity_right_mps"]) * elapsed_s
    return forward, right


def _captured_footprint_polygon(
    footprint: dict[str, Any],
    actor_position: tuple[float, float],
    yaw_degrees: float,
) -> list[list[float]]:
    center_forward = float(footprint["center_forward_m"])
    center_right = float(footprint["center_right_m"])
    extent_forward = float(footprint["extent_forward_m"])
    extent_right = float(footprint["extent_right_m"])
    radians = math.radians(float(yaw_degrees))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    polygon: list[list[float]] = []
    for local_forward, local_right in (
        (-extent_forward, -extent_right),
        (-extent_forward, extent_right),
        (extent_forward, extent_right),
        (extent_forward, -extent_right),
    ):
        offset_forward = center_forward + local_forward
        offset_right = center_right + local_right
        polygon.append(
            [
                float(actor_position[0])
                + cosine * offset_forward
                - sine * offset_right,
                float(actor_position[1])
                + sine * offset_forward
                + cosine * offset_right,
            ]
        )
    return polygon


def _angle_delta_degrees(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _expected_engine_ini_path(carla_map: str) -> str:
    suffix = carla_map.removeprefix("Carla/Maps/")
    return f"/Game/Carla/Maps/{suffix}.{suffix}"


def _expected_startup_map_argument(carla_map: str) -> str:
    return _expected_engine_ini_path(carla_map)


def validate_asset_registry(
    registry: dict[str, Any], source_c3_registry: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    _exact_keys(
        registry,
        {
            "schema_version",
            "registry_id",
            "carla_version",
            "description",
            "source_registry",
            "scripted_actor_policy",
            "dynamic_footprint_receipt",
            "assets",
            "claim_boundary",
        },
        "C4 asset registry",
    )
    _require(registry["schema_version"] == ASSET_SCHEMA, "unexpected C4 asset schema")
    _require(str(registry["carla_version"]) == "0.9.16", "C4 must freeze CARLA 0.9.16")
    source = registry["source_registry"]
    _exact_keys(
        source,
        {"registry_id", "reuse_policy", "evidence_scope"},
        "C4 source registry",
    )
    _require(
        source["registry_id"] == source_c3_registry.get("registry_id"),
        "C4 source registry identity differs from supplied C3 registry",
    )
    _require(
        source["reuse_policy"] == "exact_source_asset_and_blueprint_match",
        "C4 must require exact source asset and blueprint reuse",
    )
    policy = registry["scripted_actor_policy"]
    _exact_keys(
        policy,
        {
            "simulate_physics",
            "engine_collisions_enabled",
            "autopilot",
            "traffic_manager",
            "risk_contact_authority",
        },
        "C4 scripted actor policy",
    )
    _require(
        not any(bool(policy[key]) for key in ("simulate_physics", "engine_collisions_enabled", "autopilot", "traffic_manager")),
        "C4 scripted actors must disable physics, collisions, autopilot, and traffic manager",
    )
    _require(
        policy["risk_contact_authority"] == "evaluator_collision_polygons_xy",
        "C4 contact authority must be evaluator collision geometry",
    )
    footprint_receipt = registry["dynamic_footprint_receipt"]
    _exact_keys(
        footprint_receipt,
        {
            "schema_version",
            "source_run_id",
            "source_sealed_evidence_manifest_sha256",
            "source_instance_frames_path",
            "source_instance_frames_sha256",
            "wearer_body_radius_m",
            "local_bbox_canonical_sha256",
            "local_bbox_by_asset_id",
        },
        "C4 dynamic footprint receipt",
    )
    _require(
        footprint_receipt["schema_version"] == DYNAMIC_FOOTPRINT_SCHEMA,
        "unexpected C4 dynamic footprint schema",
    )
    _require(
        footprint_receipt["source_run_id"] == C3_FOOTPRINT_SOURCE_RUN_ID,
        "C4 footprint source run differs",
    )
    _require(
        footprint_receipt["source_sealed_evidence_manifest_sha256"]
        == C3_FOOTPRINT_SOURCE_EVIDENCE_SHA256,
        "C4 footprint source evidence manifest differs",
    )
    _require(
        footprint_receipt["source_instance_frames_path"]
        == "shards/instance/episodes/c3_town10_e01/frames.jsonl"
        and footprint_receipt["source_instance_frames_sha256"]
        == C3_FOOTPRINT_SOURCE_FRAMES_SHA256,
        "C4 footprint source frame receipt differs",
    )
    _require(
        float(footprint_receipt["wearer_body_radius_m"]) == 0.45,
        "C4 footprint wearer radius differs",
    )
    rows = registry["assets"]
    _require(isinstance(rows, list) and len(rows) == 40, "C4 must register exactly 40 assets")
    required = {
        "asset_id",
        "source_asset_id",
        "blueprint_id",
        "kind",
        "role_family",
        "mobility",
        "risk_participation",
        "collision_relevant",
        "surface_class",
    }
    source_assets = source_c3_registry.get("assets")
    _require(isinstance(source_assets, dict), "supplied C3 registry has no asset object")
    assets: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"C4 asset[{index}] must be an object")
        _exact_keys(row, required, f"C4 asset[{index}]")
        asset_id = str(row["asset_id"])
        _require(asset_id, f"C4 asset[{index}] has empty asset_id")
        _require(asset_id not in assets, f"duplicate C4 asset_id: {asset_id}")
        source_id = str(row["source_asset_id"])
        _require(source_id in source_assets, f"unknown C3 source asset: {source_id}")
        source_row = source_assets[source_id]
        for key in (
            "blueprint_id",
            "kind",
            "role_family",
            "risk_participation",
            "collision_relevant",
        ):
            _require(
                row[key] == source_row[key],
                f"C4 asset {asset_id} differs from C3 source on {key}",
            )
        mobility = str(row["mobility"])
        expected_mobility = (
            "wearer"
            if str(row["role_family"]) == "wearer"
            else "dynamic"
            if bool(row["risk_participation"])
            else "static"
        )
        _require(mobility == expected_mobility, f"C4 asset {asset_id} mobility differs")
        blueprint = str(row["blueprint_id"])
        _require(
            blueprint.startswith(("walker.", "vehicle.", "static.prop."))
            and not any(token in blueprint for token in ("*", "?", "[", "]")),
            f"C4 asset {asset_id} has invalid blueprint identity",
        )
        _require(str(row["surface_class"]) in {"sidewalk", "sidewalk_or_road", "road", "road_or_shoulder", "ground"}, f"C4 asset {asset_id} has invalid surface")
        assets[asset_id] = row
    _casefold_unique(list(assets), "C4 asset IDs")
    _casefold_unique([str(value["source_asset_id"]) for value in rows], "C4 source asset IDs")
    _casefold_unique([str(value["blueprint_id"]) for value in rows], "C4 blueprint IDs")
    mobility_counts = {
        key: sum(str(value["mobility"]) == key for value in rows)
        for key in ("wearer", "dynamic", "static")
    }
    _require(
        mobility_counts == {"wearer": 1, "dynamic": 16, "static": 23},
        f"C4 asset mobility counts differ: {mobility_counts}",
    )
    footprints = footprint_receipt["local_bbox_by_asset_id"]
    _require(
        footprint_receipt["local_bbox_canonical_sha256"]
        == C3_FOOTPRINT_CANONICAL_SHA256
        and sha256_json(footprints) == C3_FOOTPRINT_CANONICAL_SHA256,
        "C4 captured dynamic footprint content differs",
    )
    dynamic_asset_ids = {
        asset_id
        for asset_id, asset in assets.items()
        if str(asset["mobility"]) == "dynamic"
    }
    _require(
        isinstance(footprints, dict) and set(footprints) == dynamic_asset_ids,
        "C4 captured dynamic footprint set differs",
    )
    footprint_keys = {
        "center_forward_m",
        "center_right_m",
        "extent_forward_m",
        "extent_right_m",
    }
    for asset_id in sorted(dynamic_asset_ids):
        footprint = footprints[asset_id]
        _require(
            isinstance(footprint, dict),
            f"C4 dynamic footprint {asset_id} must be an object",
        )
        _exact_keys(footprint, footprint_keys, f"C4 dynamic footprint {asset_id}")
        _require(
            all(math.isfinite(float(footprint[key])) for key in footprint_keys),
            f"C4 dynamic footprint {asset_id} is not finite",
        )
        _require(
            float(footprint["extent_forward_m"]) > 0.0
            and float(footprint["extent_right_m"]) > 0.0,
            f"C4 dynamic footprint {asset_id} has invalid extents",
        )
        assets[asset_id] = {
            **assets[asset_id],
            "_captured_dynamic_footprint": copy.deepcopy(footprint),
        }
    return assets


def _validate_trajectory(name: str, trajectory: dict[str, Any]) -> None:
    _exact_keys(
        trajectory,
        {"start_forward_m", "start_right_m", "yaw_offset_degrees", "segments"},
        f"trajectory {name}",
    )
    for key in ("start_forward_m", "start_right_m", "yaw_offset_degrees"):
        _require(math.isfinite(float(trajectory[key])), f"trajectory {name} {key} is not finite")
    segments = trajectory["segments"]
    _require(isinstance(segments, list) and segments, f"trajectory {name} has no segments")
    starts: list[float] = []
    for index, segment in enumerate(segments):
        _exact_keys(
            segment,
            {"start_s", "velocity_forward_mps", "velocity_right_mps"},
            f"trajectory {name} segment[{index}]",
        )
        starts.append(float(segment["start_s"]))
        _require(
            all(
                math.isfinite(float(segment[key]))
                for key in ("start_s", "velocity_forward_mps", "velocity_right_mps")
            ),
            f"trajectory {name} segment[{index}] is not finite",
        )
    _require(starts[0] == 0.0 and starts == sorted(set(starts)), f"trajectory {name} segment starts differ")


def _validate_weather(scene_id: str, scene: dict[str, Any]) -> None:
    weather = scene["weather"]
    _exact_keys(weather, {"preset", "parameters"}, f"scene {scene_id} weather")
    _require(bool(str(weather["preset"])), f"scene {scene_id} has empty weather preset")
    parameters = weather["parameters"]
    _exact_keys(parameters, WEATHER_PARAMETER_KEYS, f"scene {scene_id} weather parameters")
    for key in WEATHER_PARAMETER_KEYS - {"sun_azimuth_angle", "sun_altitude_angle"}:
        value = float(parameters[key])
        _require(0.0 <= value <= 100.0, f"scene {scene_id} weather {key} is out of range")
    _require(0.0 <= float(parameters["sun_azimuth_angle"]) <= 360.0, f"scene {scene_id} sun azimuth is out of range")
    _require(-90.0 <= float(parameters["sun_altitude_angle"]) <= 90.0, f"scene {scene_id} sun altitude is out of range")


def validate_scene_registry(
    registry: dict[str, Any], assets: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    _exact_keys(
        registry,
        {
            "schema_version",
            "registry_id",
            "asset_registry_id",
            "description",
            "map_inventory",
            "required_scene_classes",
            "capture_contract",
            "trajectories",
            "scenes",
            "claim_boundary",
        },
        "C4 scene registry",
    )
    _require(registry["schema_version"] == SCENE_SCHEMA, "unexpected C4 scene schema")
    _require(
        registry["asset_registry_id"] == "DTR_CARLA_C4_MULTIMAP_ASSET_REGISTRY_V1",
        "C4 scene registry points at another asset registry",
    )
    inventory = registry["map_inventory"]
    _require(isinstance(inventory, list) and len(inventory) == 6, "C4 map inventory must contain six packaged maps")
    map_bindings: dict[str, dict[str, Any]] = {}
    for index, binding in enumerate(inventory):
        _exact_keys(
            binding,
            {"carla_map", "engine_ini_map_object_path", "package_status", "cold_start_status"},
            f"map inventory[{index}]",
        )
        carla_map = str(binding["carla_map"])
        _require(carla_map in PACKAGED_MAPS, f"C4 references an unapproved packaged map: {carla_map}")
        _require(carla_map not in map_bindings, f"duplicate C4 map binding: {carla_map}")
        _require(binding["engine_ini_map_object_path"] == _expected_engine_ini_path(carla_map), f"C4 engine ini binding differs for {carla_map}")
        _require(binding["package_status"] == "PACKAGED_UMAP", f"C4 map is not inventory-backed: {carla_map}")
        _require(binding["cold_start_status"] in {"TASK_OWNED_ENGINE_INI_PROBED", "C2_C3_CAPTURED", "UNPROBED"}, f"C4 map has unknown cold-start status: {carla_map}")
        map_bindings[carla_map] = binding
    _require(set(map_bindings) == PACKAGED_MAPS, "C4 map inventory differs from the packaged allowlist")
    _require(
        set(registry["required_scene_classes"]) == REQUIRED_SCENE_CLASSES,
        "C4 required scene-class set differs",
    )
    capture = registry["capture_contract"]
    _exact_keys(
        capture,
        {
            "resolution",
            "sensor_order",
            "model_modalities",
            "evaluator_modalities",
            "render_backend",
            "render_quality_level",
            "sample_seconds",
            "duration_seconds",
            "showcase_time_s",
            "capture_seed_base",
            "fresh_server_per_sensor",
        },
        "C4 capture contract",
    )
    _require(capture["resolution"] == [1280, 720], "C4 formal sensors must stay 1280x720")
    _require(capture["sensor_order"] == ["instance", "wearable", "depth", "witness"], "C4 sensor order differs")
    _require(capture["model_modalities"] == ["wearable", "depth"], "C4 model modalities differ")
    _require(capture["evaluator_modalities"] == ["instance", "witness"], "C4 evaluator modalities differ")
    _require(capture["render_backend"] == "dx12" and capture["render_quality_level"] == "Epic", "C4 render contract differs")
    _require(float(capture["sample_seconds"]) == 0.05, "C4 sample interval differs")
    _require(float(capture["duration_seconds"]) == 4.0, "C4 duration differs")
    _require(bool(capture["fresh_server_per_sensor"]), "C4 must retain fresh-server capture")

    trajectories = registry["trajectories"]
    _require(isinstance(trajectories, dict) and trajectories, "C4 has no trajectories")
    for name, trajectory in trajectories.items():
        _validate_trajectory(str(name), trajectory)

    scenes = registry["scenes"]
    _require(isinstance(scenes, dict) and len(scenes) == 8, "C4 must contain exactly eight layouts")
    _casefold_unique([str(value) for value in scenes], "C4 layout IDs")
    used_classes: set[str] = set()
    used_maps: set[str] = set()
    used_assets = {next(key for key, value in assets.items() if value["mobility"] == "wearer")}
    all_episode_ids: list[str] = []
    outcomes_by_episode_slot: dict[str, set[str]] = defaultdict(set)
    total_dynamic = 0
    total_static = 0
    runtime_anchor_validation_count = 0
    risk_corridor_dynamic_episode_checks = 0
    contact_geometry_checks = 0
    safe_geometry_checks = 0
    contact_terminal_separations: list[float] = []
    safe_minimum_separations: list[float] = []
    contact_primary_minimum_footprint_clearances: list[float] = []
    safe_primary_minimum_footprint_clearances: list[float] = []
    nonprimary_minimum_footprint_clearances: list[float] = []
    background_pair_minimum_center_clearances: list[float] = []
    scene_keys = {
        "status",
        "scenario_class",
        "map",
        "display_name",
        "weather",
        "anchor",
        "witness",
        "class_parameters",
        "actors",
        "episodes",
        "counterfactual_contract",
        "admission",
    }
    for scene_id, scene in scenes.items():
        _exact_keys(scene, scene_keys, f"scene {scene_id}")
        _require(scene["status"] == FORMAL_STATUS, f"scene {scene_id} is not statically admitted")
        scenario_class = str(scene["scenario_class"])
        _require(scenario_class in REQUIRED_SCENE_CLASSES, f"scene {scene_id} has unknown class")
        _require(scenario_class not in used_classes, f"duplicate C4 scene class: {scenario_class}")
        used_classes.add(scenario_class)
        carla_map = str(scene["map"])
        _require(carla_map in map_bindings, f"scene {scene_id} uses an unbound map")
        used_maps.add(carla_map)
        for label, identity in (
            ("layout_id", str(scene_id)),
            ("display_name", str(scene["display_name"])),
        ):
            _require(_identity_is_neutral(identity), f"scene {scene_id} {label} leaks evaluator outcome")
        _validate_weather(str(scene_id), scene)
        anchor = scene["anchor"]
        _exact_keys(anchor, {"center_xy_m", "forward_xy", "right_xy", "source"}, f"scene {scene_id} anchor")
        _require(len(anchor["center_xy_m"]) == 2 and len(anchor["forward_xy"]) == 2 and len(anchor["right_xy"]) == 2, f"scene {scene_id} anchor dimensions differ")
        center = [float(value) for value in anchor["center_xy_m"]]
        forward = [float(value) for value in anchor["forward_xy"]]
        right = [float(value) for value in anchor["right_xy"]]
        _require(all(math.isfinite(value) for value in [*center, *forward, *right]), f"scene {scene_id} anchor is not finite")
        _require(abs(math.hypot(*forward) - 1.0) <= 1e-3, f"scene {scene_id} forward axis is not normalized")
        _require(abs(math.hypot(*right) - 1.0) <= 1e-3, f"scene {scene_id} right axis is not normalized")
        _require(abs(forward[0] * right[0] + forward[1] * right[1]) <= 1e-3, f"scene {scene_id} anchor axes are not orthogonal")
        source = anchor["source"]
        source_kind = str(source.get("kind", ""))
        if source_kind == "opendrive_waypoint_receipt":
            _exact_keys(
                source,
                {
                    "kind",
                    "map",
                    "xodr_sha256",
                    "sample_distance_m",
                    "filtered_waypoint_index",
                    "road_id",
                    "section_id",
                    "lane_id",
                    "s_m",
                    "runtime_validation_required",
                },
                f"scene {scene_id} anchor source",
            )
            _require(carla_map in XODR_SHA256_BY_MAP, f"scene {scene_id} has no frozen OpenDRIVE authority")
            _require(str(source["map"]) == carla_map, f"scene {scene_id} OpenDRIVE map differs")
            _require(
                str(source["xodr_sha256"]).casefold() == XODR_SHA256_BY_MAP[carla_map],
                f"scene {scene_id} OpenDRIVE SHA-256 differs",
            )
            _require(float(source["sample_distance_m"]) == 5.0, f"scene {scene_id} OpenDRIVE sampling differs")
            _require(int(source["filtered_waypoint_index"]) >= 0, f"scene {scene_id} waypoint index is negative")
            _require(int(source["road_id"]) >= 0, f"scene {scene_id} road ID is negative")
            _require(int(source["section_id"]) >= 0, f"scene {scene_id} section ID is negative")
            _require(int(source["lane_id"]) != 0, f"scene {scene_id} lane ID is zero")
            _require(float(source["s_m"]) >= 0.0, f"scene {scene_id} waypoint s is negative")
            _require(bool(source["runtime_validation_required"]), f"scene {scene_id} OpenDRIVE anchor must retain runtime validation")
        elif source_kind == "c3_captured_anchor":
            _exact_keys(source, {"kind", "spawn_point_index", "runtime_validation_required"}, f"scene {scene_id} anchor source")
            _require(carla_map == "Carla/Maps/Town10HD_Opt", f"scene {scene_id} captured anchor is not Town10")
            _require(int(source["spawn_point_index"]) >= 0, f"scene {scene_id} spawn index is negative")
            _require(not bool(source["runtime_validation_required"]), f"scene {scene_id} captured anchor remains unvalidated")
        else:
            raise ValueError(f"scene {scene_id} anchor source is not auditable")
        if bool(source["runtime_validation_required"]):
            runtime_anchor_validation_count += 1
        _require(set(scene["class_parameters"]) == CLASS_PARAMETER_KEYS[scenario_class], f"scene {scene_id} class parameters differ")

        actors = scene["actors"]
        _require(isinstance(actors, list) and actors, f"scene {scene_id} has no actors")
        instance_ids = [str(value.get("instance_id", "")) for value in actors]
        track_ids = [str(value.get("track_id", "")) for value in actors]
        _casefold_unique(instance_ids, f"scene {scene_id} instance IDs")
        _casefold_unique(track_ids, f"scene {scene_id} track IDs")
        actor_by_id: dict[str, dict[str, Any]] = {}
        dynamic_actors: list[dict[str, Any]] = []
        static_actors: list[dict[str, Any]] = []
        trajectory_key_actors: set[str] = set()
        for index, actor in enumerate(actors):
            _require(isinstance(actor, dict), f"scene {scene_id} actor[{index}] must be an object")
            required = {"instance_id", "track_id", "asset_id", "role"}
            placements = {key for key in ("trajectory_ref", "trajectory_key", "fixed_pose") if key in actor}
            _require(set(actor) == required | placements and len(placements) == 1, f"scene {scene_id} actor[{index}] fields differ")
            instance_id = str(actor["instance_id"])
            asset_id = str(actor["asset_id"])
            _require(asset_id in assets, f"scene {scene_id} references unknown asset {asset_id}")
            asset = assets[asset_id]
            _require(asset["mobility"] != "wearer", f"scene {scene_id} explicitly places wearer")
            used_assets.add(asset_id)
            actor_by_id[instance_id] = actor
            if asset["mobility"] == "dynamic":
                _require("fixed_pose" not in placements, f"scene {scene_id}/{instance_id} dynamic target is fixed")
                dynamic_actors.append(actor)
                if "trajectory_ref" in actor:
                    reference = str(actor["trajectory_ref"])
                    _require(reference in trajectories and _has_nonzero_motion(trajectories[reference]), f"scene {scene_id}/{instance_id} has invalid dynamic trajectory")
                else:
                    trajectory_key_actors.add(instance_id)
            else:
                _require(placements == {"fixed_pose"}, f"scene {scene_id}/{instance_id} static support is not fixed")
                static_actors.append(actor)
        _require(trajectory_key_actors == {"target_primary"}, f"scene {scene_id} primary trajectory-key set differs")

        admission = scene["admission"]
        _exact_keys(admission, {"dynamic_target_count", "static_support_count", "total_actors_including_wearer", "minimum_visible_frames_per_dynamic_target_per_episode", "risk_corridor_threshold_m"}, f"scene {scene_id} admission")
        _require(int(admission["dynamic_target_count"]) == len(dynamic_actors), f"scene {scene_id} dynamic target count differs")
        _require(int(admission["static_support_count"]) == len(static_actors), f"scene {scene_id} static support count differs")
        _require(int(admission["total_actors_including_wearer"]) == len(actors) + 1, f"scene {scene_id} total actor count differs")
        _require(int(admission["minimum_visible_frames_per_dynamic_target_per_episode"]) == 10, f"scene {scene_id} visibility gate differs")
        _require(float(admission["risk_corridor_threshold_m"]) == 3.0, f"scene {scene_id} risk corridor is not 3m")
        minimum_dynamic_targets = 12 if scenario_class == "crowded_pedestrians" else 8
        _require(
            len(dynamic_actors) >= minimum_dynamic_targets,
            f"scene {scene_id} must have at least {minimum_dynamic_targets} dynamic risk participants",
        )
        total_dynamic += len(dynamic_actors)
        total_static += len(static_actors)

        episodes = scene["episodes"]
        _require(isinstance(episodes, list) and len(episodes) == 2, f"scene {scene_id} must have two episodes")
        episode_by_id: dict[str, dict[str, Any]] = {}
        sample_seconds = float(capture["sample_seconds"])
        duration_seconds = float(capture["duration_seconds"])
        sample_times = [
            index * sample_seconds
            for index in range(int(round(duration_seconds / sample_seconds)) + 1)
        ]
        for episode in episodes:
            _exact_keys(episode, {"episode_id", "navigation_session_id", "expected_outcome", "expected_responsible_assets", "wearer_trajectory", "asset_trajectories", "issued_plan"}, f"scene {scene_id} episode")
            episode_id = str(episode["episode_id"])
            plan = episode["issued_plan"]
            _exact_keys(plan, {"plan_id", "session_id", "issued_at_s", "expires_at_s", "time_parameterized_waypoints"}, f"scene {scene_id}/{episode_id} plan")
            for label, identity in (
                ("episode_id", episode_id),
                ("navigation_session_id", str(episode["navigation_session_id"])),
                ("plan_id", str(plan["plan_id"])),
                ("plan_session_id", str(plan["session_id"])),
            ):
                _require(_identity_is_neutral(identity), f"scene {scene_id} {label} leaks evaluator outcome")
            _require(str(plan["session_id"]) == str(episode["navigation_session_id"]), f"scene {scene_id}/{episode_id} plan session differs")
            _require(float(plan["issued_at_s"]) == 0.0, f"scene {scene_id}/{episode_id} plan issue time differs")
            _require(float(plan["expires_at_s"]) == duration_seconds, f"scene {scene_id}/{episode_id} plan expiry differs")
            outcome = str(episode["expected_outcome"])
            _require(outcome in {"CONTACT", "SAFE"}, f"scene {scene_id}/{episode_id} outcome differs")
            responsible = list(episode["expected_responsible_assets"])
            _require(responsible == (["target_primary"] if outcome == "CONTACT" else []), f"scene {scene_id}/{episode_id} responsibility differs")
            wearer_reference = str(episode["wearer_trajectory"])
            _require(wearer_reference in trajectories, f"scene {scene_id}/{episode_id} wearer trajectory is unknown")
            wearer_trajectory = trajectories[wearer_reference]
            _require(_has_nonzero_motion(wearer_trajectory), f"scene {scene_id}/{episode_id} wearer has zero motion")
            plan_waypoints = plan["time_parameterized_waypoints"]
            _require(isinstance(plan_waypoints, list) and len(plan_waypoints) >= 3, f"scene {scene_id}/{episode_id} plan waypoints differ")
            plan_times: list[float] = []
            for waypoint_index, waypoint in enumerate(plan_waypoints):
                _exact_keys(waypoint, {"time_s", "forward_m", "right_m"}, f"scene {scene_id}/{episode_id} plan waypoint[{waypoint_index}]")
                waypoint_time = float(waypoint["time_s"])
                plan_times.append(waypoint_time)
                expected_forward, expected_right = _trajectory_position(wearer_trajectory, waypoint_time)
                _require(
                    math.isclose(float(waypoint["forward_m"]), expected_forward, abs_tol=1e-6)
                    and math.isclose(float(waypoint["right_m"]), expected_right, abs_tol=1e-6),
                    f"scene {scene_id}/{episode_id} plan does not match wearer trajectory",
                )
            _require(
                plan_times == sorted(set(plan_times))
                and plan_times[0] == 0.0
                and plan_times[-1] == duration_seconds,
                f"scene {scene_id}/{episode_id} plan timing differs",
            )
            overrides = episode["asset_trajectories"]
            _require(set(overrides) == trajectory_key_actors, f"scene {scene_id}/{episode_id} override set differs")
            for actor_id, reference in overrides.items():
                _require(actor_id in actor_by_id and reference in trajectories, f"scene {scene_id}/{episode_id} override is unknown")
                _require(_has_nonzero_motion(trajectories[reference]), f"scene {scene_id}/{episode_id} override has zero motion")

            wearer_positions = [
                _trajectory_position(wearer_trajectory, time_s)
                for time_s in sample_times
            ]
            sampled_by_actor: dict[str, list[tuple[float, float]]] = {}
            separations_by_actor: dict[str, list[float]] = {}
            footprint_clearances_by_actor: dict[str, list[float]] = {}
            for actor in dynamic_actors:
                actor_id = str(actor["instance_id"])
                trajectory_reference = str(
                    overrides[actor_id]
                    if "trajectory_key" in actor
                    else actor["trajectory_ref"]
                )
                actor_trajectory = trajectories[trajectory_reference]
                actor_positions = [
                    _trajectory_position(actor_trajectory, time_s)
                    for time_s in sample_times
                ]
                _require(
                    all(math.isfinite(value) for point in actor_positions for value in point),
                    f"scene {scene_id}/{episode_id}/{actor_id} has non-finite sampled geometry",
                )
                separations = [
                    math.dist(wearer_position, actor_position)
                    for wearer_position, actor_position in zip(wearer_positions, actor_positions)
                ]
                footprint = assets[str(actor["asset_id"])][
                    "_captured_dynamic_footprint"
                ]
                footprint_clearances = [
                    point_polygon_distance(
                        wearer_position,
                        _captured_footprint_polygon(
                            footprint,
                            actor_position,
                            float(actor_trajectory["yaw_offset_degrees"]),
                        ),
                    )
                    - 0.45
                    for wearer_position, actor_position in zip(
                        wearer_positions, actor_positions
                    )
                ]
                minimum_footprint_clearance = min(footprint_clearances)
                _require(
                    minimum_footprint_clearance
                    <= float(admission["risk_corridor_threshold_m"]),
                    f"scene {scene_id}/{episode_id}/{actor_id} never enters the 3m risk corridor",
                )
                if actor_id != "target_primary":
                    _require(
                        minimum_footprint_clearance > 0.05,
                        f"scene {scene_id}/{episode_id}/{actor_id} creates an unregistered contact",
                    )
                    nonprimary_minimum_footprint_clearances.append(
                        minimum_footprint_clearance
                    )
                sampled_by_actor[actor_id] = actor_positions
                separations_by_actor[actor_id] = separations
                footprint_clearances_by_actor[actor_id] = footprint_clearances
                risk_corridor_dynamic_episode_checks += 1

            trajectory_signatures = {
                actor_id: tuple(
                    (round(point[0], 6), round(point[1], 6))
                    for point in actor_positions
                )
                for actor_id, actor_positions in sampled_by_actor.items()
            }
            _require(
                len(set(trajectory_signatures.values())) == len(trajectory_signatures),
                f"scene {scene_id}/{episode_id} has completely overlapping dynamic trajectories",
            )
            initial_positions = {
                actor_id: actor_positions[0]
                for actor_id, actor_positions in sampled_by_actor.items()
            }
            initial_actor_ids = sorted(initial_positions)
            _require(
                all(
                    math.dist(initial_positions[first], initial_positions[second]) > 0.05
                    for first_index, first in enumerate(initial_actor_ids)
                    for second in initial_actor_ids[first_index + 1 :]
                ),
                f"scene {scene_id}/{episode_id} has overlapping dynamic spawn centers",
            )
            background_actor_ids = sorted(
                actor_id
                for actor_id in sampled_by_actor
                if actor_id.startswith("dynamic_")
            )
            for first_index, first in enumerate(background_actor_ids):
                for second in background_actor_ids[first_index + 1 :]:
                    minimum_pair_clearance = min(
                        math.dist(first_position, second_position)
                        for first_position, second_position in zip(
                            sampled_by_actor[first], sampled_by_actor[second]
                        )
                    )
                    _require(
                        minimum_pair_clearance >= 1.0,
                        f"scene {scene_id}/{episode_id}/{first}/{second} background trajectories are too close",
                    )
                    background_pair_minimum_center_clearances.append(
                        minimum_pair_clearance
                    )
            primary_separations = separations_by_actor["target_primary"]
            primary_footprint_clearances = footprint_clearances_by_actor[
                "target_primary"
            ]
            if outcome == "CONTACT":
                contact_geometry_checks += 1
                contact_terminal_separations.append(primary_separations[-1])
                contact_primary_minimum_footprint_clearances.append(
                    min(primary_footprint_clearances)
                )
                _require(
                    primary_separations[-1] <= 0.5
                    and min(primary_footprint_clearances) <= 0.0,
                    f"scene {scene_id}/{episode_id} CONTACT geometry does not converge",
                )
            else:
                safe_geometry_checks += 1
                safe_minimum_separations.append(min(primary_separations))
                safe_primary_minimum_footprint_clearances.append(
                    min(primary_footprint_clearances)
                )
                _require(
                    min(primary_separations) >= 1.0
                    and min(primary_footprint_clearances) > 0.05,
                    f"scene {scene_id}/{episode_id} SAFE geometry enters contact clearance",
                )
            episode_by_id[episode_id] = episode
            all_episode_ids.append(episode_id)
            slot = episode_id.rsplit("_", 1)[-1]
            outcomes_by_episode_slot[slot].add(outcome)
        _require({str(value["expected_outcome"]) for value in episodes} == {"CONTACT", "SAFE"}, f"scene {scene_id} is not an outcome pair")

        counterfactual = scene["counterfactual_contract"]
        _exact_keys(counterfactual, {"a", "b", "primary_instance_id", "occluder_instance_id", "identical_before_s", "allowed_difference"}, f"scene {scene_id} counterfactual")
        pair_ids = {str(counterfactual["a"]), str(counterfactual["b"])}
        _require(pair_ids == set(episode_by_id), f"scene {scene_id} counterfactual pair differs")
        primary_id = str(counterfactual["primary_instance_id"])
        occluder_id = str(counterfactual["occluder_instance_id"])
        _require(primary_id in trajectory_key_actors, f"scene {scene_id} primary target differs")
        _require(occluder_id in actor_by_id and assets[str(actor_by_id[occluder_id]["asset_id"])]["mobility"] == "dynamic", f"scene {scene_id} occluder differs")
        first = episode_by_id[str(counterfactual["a"])]
        second = episode_by_id[str(counterfactual["b"])]
        first_trajectory = trajectories[str(first["asset_trajectories"][primary_id])]
        second_trajectory = trajectories[str(second["asset_trajectories"][primary_id])]
        _require(float(counterfactual["identical_before_s"]) == 2.0, f"scene {scene_id} frozen branch time differs")
        _require(str(counterfactual["allowed_difference"]) == "primary_velocity_after_2s", f"scene {scene_id} allowed counterfactual difference differs")
        _require(first["wearer_trajectory"] == second["wearer_trajectory"], f"scene {scene_id} twin wearer trajectories differ")
        _require(first["issued_plan"] == second["issued_plan"], f"scene {scene_id} twin issued plans differ")
        _require(
            trajectory_prefix_equal(
                first_trajectory,
                second_trajectory,
                end_s=float(counterfactual["identical_before_s"]),
                sample_s=float(capture["sample_seconds"]),
            ),
            f"scene {scene_id} target prefixes differ before the frozen boundary",
        )
        _require(
            not trajectory_prefix_equal(
                first_trajectory,
                second_trajectory,
                end_s=float(capture["duration_seconds"]),
                sample_s=float(capture["sample_seconds"]),
            ),
            f"scene {scene_id} target twins never branch after the frozen boundary",
        )

        parameters = scene["class_parameters"]
        weather_parameters = scene["weather"]["parameters"]
        if scenario_class == "narrow_alley":
            _require(float(parameters["clear_width_m"]) <= 4.0, "narrow-alley width is not narrow")
        elif scenario_class == "rainy_night":
            _require("Night" in str(scene["weather"]["preset"]), "rainy-night preset is not a night preset")
            _require(float(weather_parameters["precipitation"]) >= float(parameters["minimum_precipitation"]), "rainy-night precipitation gate differs")
            _require(float(weather_parameters["wetness"]) >= float(parameters["minimum_wetness"]), "rainy-night wetness gate differs")
            _require(float(weather_parameters["sun_altitude_angle"]) < 0.0, "rainy-night sun is above the horizon")
        elif scenario_class == "backlight":
            heading = math.degrees(math.atan2(forward[1], forward[0])) % 360.0
            sun_azimuth = float(weather_parameters["sun_azimuth_angle"])
            _require(_angle_delta_degrees(heading, sun_azimuth) <= float(parameters["maximum_camera_sun_delta_degrees"]), "backlight sun is not camera-aligned")
            _require(float(weather_parameters["sun_altitude_angle"]) <= float(parameters["maximum_sun_altitude_degrees"]), "backlight sun is too high")
        elif scenario_class == "crowded_pedestrians":
            dynamic_walkers = sum(assets[str(value["asset_id"])]["kind"] == "walker" for value in dynamic_actors)
            _require(dynamic_walkers >= int(parameters["minimum_dynamic_pedestrians"]), "crowded layout has too few dynamic pedestrians")
            _require(len(dynamic_actors) >= int(parameters["minimum_dynamic_targets"]), "crowded layout has too few dynamic targets")

    _casefold_unique(all_episode_ids, "C4 episode IDs")
    _require(used_classes == REQUIRED_SCENE_CLASSES, "C4 scene-class coverage differs")
    _require(used_maps == PACKAGED_MAPS and len(used_maps) >= 4, "C4 multimap coverage differs")
    _require(used_assets == set(assets), f"C4 scene bank does not exercise every registered asset: {sorted(set(assets) - used_assets)}")
    _require(all(values == {"CONTACT", "SAFE"} for values in outcomes_by_episode_slot.values()), "C4 episode suffix leaks an outcome class")
    _require(contact_geometry_checks == len(scenes), "C4 CONTACT geometry check count differs")
    _require(safe_geometry_checks == len(scenes), "C4 SAFE geometry check count differs")
    _require(
        risk_corridor_dynamic_episode_checks == total_dynamic * 2,
        "C4 per-episode dynamic risk-corridor check count differs",
    )
    return {
        "layout_count": len(scenes),
        "map_count": len(used_maps),
        "maps": sorted(used_maps),
        "scene_classes": sorted(used_classes),
        "episode_count": len(all_episode_ids),
        "dynamic_target_placements": total_dynamic,
        "static_support_placements": total_static,
        "nonwearer_actor_placements": total_dynamic + total_static,
        "actor_placements_including_wearers": total_dynamic + total_static + len(scenes),
        "runtime_anchor_validation_count": runtime_anchor_validation_count,
        "risk_corridor_dynamic_episode_checks": risk_corridor_dynamic_episode_checks,
        "contact_geometry_checks": contact_geometry_checks,
        "safe_geometry_checks": safe_geometry_checks,
        "maximum_contact_terminal_separation_m": max(contact_terminal_separations),
        "minimum_safe_separation_m": min(safe_minimum_separations),
        "maximum_contact_primary_footprint_clearance_m": max(
            contact_primary_minimum_footprint_clearances
        ),
        "minimum_safe_primary_footprint_clearance_m": min(
            safe_primary_minimum_footprint_clearances
        ),
        "minimum_nonprimary_footprint_clearance_m": min(
            nonprimary_minimum_footprint_clearances
        ),
        "minimum_background_pair_center_clearance_m": min(
            background_pair_minimum_center_clearances
        ),
        "map_bindings": map_bindings,
    }


def validate_registry_bundle(
    asset_registry: dict[str, Any],
    scene_registry: dict[str, Any],
    source_c3_registry: dict[str, Any],
) -> dict[str, Any]:
    assets = validate_asset_registry(asset_registry, source_c3_registry)
    scene_report = validate_scene_registry(scene_registry, assets)
    return {
        "registered_asset_count": len(assets),
        "registered_dynamic_asset_types": sum(value["mobility"] == "dynamic" for value in assets.values()),
        "registered_static_asset_types": sum(value["mobility"] == "static" for value in assets.values()),
        **scene_report,
    }


def _compile_asset_templates(
    assets: dict[str, dict[str, Any]], source_c3_registry: dict[str, Any]
) -> dict[str, Any]:
    templates: dict[str, Any] = {}
    source_assets = source_c3_registry["assets"]
    for asset_id, asset in assets.items():
        source = source_assets[str(asset["source_asset_id"])]
        templates[asset_id] = {
            "kind": str(asset["kind"]),
            "blueprint_candidates": [str(asset["blueprint_id"])],
            "surface_offset_m": float(source["surface_policy"]["surface_offset_m"]),
            "collision_relevant": bool(asset["collision_relevant"]),
            "collisions_enabled": False,
            "c4_source_asset_id": str(asset["source_asset_id"]),
            "c4_risk_participation": bool(asset["risk_participation"]),
        }
    return templates


def compile_multimap(
    base_c2_protocol: dict[str, Any],
    source_c3_registry: dict[str, Any],
    asset_registry: dict[str, Any],
    scene_registry: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    report = validate_registry_bundle(asset_registry, scene_registry, source_c3_registry)
    _require(base_c2_protocol.get("experiment_id") == C2_EXPERIMENT_ID, "C4 compiler requires the frozen C2 protocol family")
    assets = {str(value["asset_id"]): value for value in asset_registry["assets"]}
    source_assets = source_c3_registry["assets"]
    wearer_id = next(key for key, value in assets.items() if value["mobility"] == "wearer")
    wearer = assets[wearer_id]
    capture = scene_registry["capture_contract"]
    map_bindings = {str(value["carla_map"]): value for value in scene_registry["map_inventory"]}
    scenes_by_map: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for scene_id, scene in scene_registry["scenes"].items():
        scenes_by_map[str(scene["map"])].append((str(scene_id), scene))
    protocols: dict[str, dict[str, Any]] = {}
    index_entries: list[dict[str, Any]] = []
    evaluator_outcomes: dict[str, Any] = {}
    for map_index, carla_map in enumerate(sorted(scenes_by_map)):
        protocol = copy.deepcopy(base_c2_protocol)
        binding = map_bindings[carla_map]
        map_name = carla_map.rsplit("/", 1)[-1]
        protocol_id = map_name.casefold()
        protocol["environment"]["map"] = carla_map
        protocol["capture"]["seed"] = int(capture["capture_seed_base"]) + map_index
        protocol["capture"]["render_backend"] = str(capture["render_backend"])
        protocol["capture"]["render_quality_level"] = str(capture["render_quality_level"])
        protocol["objective"] = f"Capture the C4 static multimap scene group for {map_name}."
        protocol["asset_templates"] = _compile_asset_templates(assets, source_c3_registry)
        wearer_source = source_assets[str(wearer["source_asset_id"])]
        protocol["wearer"] = {
            "asset_key": "wearer",
            "track_id": "m_00",
            "role": "wearer",
            "kind": str(wearer["kind"]),
            "blueprint_candidates": [str(wearer["blueprint_id"])],
            "surface_offset_m": float(wearer_source["surface_policy"]["surface_offset_m"]),
            "collisions_enabled": False,
            "collision_relevant": False,
            "scripted_invincible": True,
        }
        protocol["trajectory_library"] = copy.deepcopy(scene_registry["trajectories"])
        protocol["layouts"] = {}
        protocol["scenarios"] = []
        protocol["occlusion_contracts"] = []
        protocol["twin_contracts"] = []
        used_blueprints = {str(wearer["blueprint_id"])}
        layout_asset_counts: list[int] = []
        entry_layouts: list[dict[str, Any]] = []
        entry_episodes: list[dict[str, Any]] = []
        for scene_id, scene in sorted(scenes_by_map[carla_map]):
            dynamic_target_ids = [
                str(actor["instance_id"])
                for actor in scene["actors"]
                if assets[str(actor["asset_id"])]["mobility"] == "dynamic"
            ]
            compiled_actors: list[dict[str, Any]] = []
            for actor in scene["actors"]:
                asset_id = str(actor["asset_id"])
                used_blueprints.add(str(assets[asset_id]["blueprint_id"]))
                compiled = {
                    "asset_key": str(actor["instance_id"]),
                    "track_id": str(actor["track_id"]),
                    "role": str(actor["role"]),
                    "template": asset_id,
                }
                if "fixed_pose" in actor:
                    compiled["fixed_pose"] = copy.deepcopy(actor["fixed_pose"])
                elif "trajectory_ref" in actor:
                    compiled["trajectory"] = str(actor["trajectory_ref"])
                else:
                    compiled["trajectory_key"] = str(actor["instance_id"])
                compiled_actors.append(compiled)
            layout_asset_counts.append(len(compiled_actors))
            protocol["layouts"][scene_id] = {
                "display_name": str(scene["display_name"]),
                "weather": str(scene["weather"]["preset"]),
                "duration_seconds": float(capture["duration_seconds"]),
                "showcase_time_s": float(capture["showcase_time_s"]),
                "anchor": copy.deepcopy(scene["anchor"]),
                "witness": copy.deepcopy(scene["witness"]),
                "assets": compiled_actors,
                "c4_scenario_class": str(scene["scenario_class"]),
                "c4_weather_parameters": copy.deepcopy(scene["weather"]["parameters"]),
            }
            episode_ids: list[str] = []
            for episode_index, episode in enumerate(scene["episodes"]):
                compiled_episode = copy.deepcopy(episode)
                compiled_episode["layout_id"] = scene_id
                compiled_episode["scenario_role"] = "counterfactual_pair_member"
                compiled_episode["twin_role"] = f"variant_{episode_index + 1}"
                protocol["scenarios"].append(compiled_episode)
                episode_id = str(episode["episode_id"])
                episode_ids.append(episode_id)
                evaluator_outcomes[episode_id] = {
                    "expected_outcome": str(episode["expected_outcome"]),
                    "expected_responsible_assets": list(episode["expected_responsible_assets"]),
                }
                entry_episodes.append(
                    {
                        "episode_id": episode_id,
                        "layout_id": scene_id,
                        "dynamic_target_ids": list(dynamic_target_ids),
                        "minimum_visible_frames_per_dynamic_target": int(
                            scene["admission"]["minimum_visible_frames_per_dynamic_target_per_episode"]
                        ),
                        "risk_corridor_threshold_m": float(
                            scene["admission"]["risk_corridor_threshold_m"]
                        ),
                    }
                )
            counterfactual = scene["counterfactual_contract"]
            required_outcomes = {
                str(value["episode_id"]): str(value["expected_outcome"])
                for value in scene["episodes"]
            }
            protocol["occlusion_contracts"].append(
                {
                    "contract_id": f"{scene_id}_occlusion_pair",
                    "episodes": episode_ids,
                    "target_asset": str(counterfactual["primary_instance_id"]),
                    "occluder_asset": str(counterfactual["occluder_instance_id"]),
                    "minimum_pre_track_frames": 10,
                    "minimum_post_reappearance_frames": 10,
                    "minimum_trackable_pixel_fraction": 0.0002,
                    "complete_occlusion_pixel_fraction": 0.0,
                    "minimum_complete_occlusion_seconds": 0.3,
                    "maximum_complete_occlusion_seconds": 0.6,
                    "pair_identical_through_seconds": float(counterfactual["identical_before_s"]),
                    "required_outcomes": required_outcomes,
                }
            )
            protocol["twin_contracts"].append(
                {
                    "family": "track_then_physical_occlusion_outcome_twin",
                    "a": str(counterfactual["a"]),
                    "b": str(counterfactual["b"]),
                    "identical_before_s": float(counterfactual["identical_before_s"]),
                    "allowed_difference": str(counterfactual["allowed_difference"]),
                }
            )
            entry_layouts.append(
                {
                    "layout_id": scene_id,
                    "scene_class": str(scene["scenario_class"]),
                    "episode_ids": episode_ids,
                    "weather_preset": str(scene["weather"]["preset"]),
                    "weather_parameters": copy.deepcopy(scene["weather"]["parameters"]),
                    "anchor": copy.deepcopy(scene["anchor"]),
                }
            )
        protocol["admission"] = {
            "expected_episode_count": len(protocol["scenarios"]),
            "expected_layout_count": len(protocol["layouts"]),
            "minimum_active_assets_per_layout_excluding_wearer": min(layout_asset_counts),
            "minimum_unique_actual_blueprints_across_pack": len(used_blueprints),
            "required_resolution": [1280, 720],
            "required_model_modalities": ["wearable", "depth"],
            "required_evaluator_modalities": ["instance", "witness"],
            "require_zero_blueprint_fallbacks": True,
        }
        protocol["c4_compatibility"] = {
            "schema_version": COMPILED_SCHEMA,
            "experiment_id": C4_EXPERIMENT_ID,
            "protocol_id": protocol_id,
            "carla_map": carla_map,
            "engine_ini_map_object_path": str(binding["engine_ini_map_object_path"]),
            "cold_start_status": str(binding["cold_start_status"]),
            "weather_override_authority": "frozen_c4_weather_parameters_materialized_by_capture",
        }
        protocol["claim_boundary"] = [
            "This is a C2-compatible static C4 protocol candidate; noncaptured anchors remain runtime-unverified.",
            "Frozen c4_weather_parameters are materialized by the capture path after copying the named preset.",
            "Outcome truth is evaluator-only and absent from model-visible identifiers.",
        ]
        validate_c2_protocol(protocol)
        protocols[protocol_id] = protocol
        index_entries.append(
            {
                "protocol_id": protocol_id,
                "protocol_path": f"{protocol_id}.c2-protocol.json",
                "protocol_sha256": materialized_json_sha256(protocol),
                "carla_map": carla_map,
                "startup_map_argument": _expected_startup_map_argument(carla_map),
                "engine_ini_map_object_path": str(binding["engine_ini_map_object_path"]),
                "cold_start_status": str(binding["cold_start_status"]),
                "layout_ids": list(protocol["layouts"]),
                "episodes": entry_episodes,
                "layout_count": len(protocol["layouts"]),
                "episode_count": len(protocol["scenarios"]),
                "unique_registered_blueprints_in_protocol": len(used_blueprints),
                "layouts": entry_layouts,
            }
        )
    model_visible_index = [
        {
            "protocol_id": value["protocol_id"],
            "carla_map": value["carla_map"],
            "layout_ids": [layout["layout_id"] for layout in value["layouts"]],
            "episode_ids": [episode_id for layout in value["layouts"] for episode_id in layout["episode_ids"]],
        }
        for value in index_entries
    ]
    _require(
        all(
            _identity_is_neutral(identity)
            for value in model_visible_index
            for identity in [value["protocol_id"], *value["layout_ids"], *value["episode_ids"]]
        ),
        "compiled C4 model-visible index leaks evaluator outcomes",
    )
    compiled_index = {
        "schema_version": COMPILED_SCHEMA,
        "experiment_id": C4_EXPERIMENT_ID,
        "capture": {
            "resolution": list(capture["resolution"]),
            "sensor_order": list(capture["sensor_order"]),
        },
        "registries": {
            "asset_registry": {
                "path": "dtr_carla_c4_asset_registry.json",
                "sha256": materialized_json_sha256(asset_registry),
            },
            "scene_registry": {
                "path": "dtr_carla_c4_scene_registry.json",
                "sha256": materialized_json_sha256(scene_registry),
            },
        },
        "protocols": index_entries,
        "admission": {
            "expected_map_count": len(protocols),
            "expected_protocol_count": len(protocols),
            "expected_layout_count": report["layout_count"],
            "expected_episode_count": report["episode_count"],
            "expected_sensor_count": len(capture["sensor_order"]),
            "expected_shard_count": len(protocols) * len(capture["sensor_order"]),
            "expected_dynamic_target_placements": report["dynamic_target_placements"],
            "expected_dynamic_target_episode_geometry_checks": report[
                "risk_corridor_dynamic_episode_checks"
            ],
            "minimum_dynamic_targets_per_non_crowded_layout": 8,
            "minimum_dynamic_targets_crowded_layout": 12,
        },
        "model_visible_index": model_visible_index,
        "evaluator_outcomes": evaluator_outcomes,
        "claim_boundary": list(scene_registry["claim_boundary"]),
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "experiment_id": C4_EXPERIMENT_ID,
        "source_c3_registry_content_sha256": sha256_json(source_c3_registry),
        "c4_asset_registry_sha256": materialized_json_sha256(asset_registry),
        "c4_scene_registry_sha256": materialized_json_sha256(scene_registry),
        "base_c2_protocol_content_sha256": sha256_json(base_c2_protocol),
        "compiled_index_sha256": materialized_json_sha256(compiled_index),
        "protocol_sha256_by_id": {
            key: materialized_json_sha256(value) for key, value in sorted(protocols.items())
        },
        "counts": {
            **{key: value for key, value in report.items() if key != "map_bindings"},
            "compiled_protocol_count": len(protocols),
        },
        "checks": {
            "all_c4_assets_exactly_match_c3_source": True,
            "all_six_map_groups_compile_as_c2_protocols": len(protocols) == 6,
            "all_eight_requested_scene_classes_present": report["layout_count"] == 8,
            "formal_resolution_is_1280x720": capture["resolution"] == [1280, 720],
            "model_visible_identifiers_are_outcome_neutral": True,
            "engine_ini_map_bindings_frozen": all(
                value["engine_ini_map_object_path"] == _expected_engine_ini_path(value["carla_map"])
                for value in index_entries
            ),
            "opendrive_waypoint_receipts_frozen": report["runtime_anchor_validation_count"] == 7,
            "all_dynamic_risk_participants_enter_3m_per_episode": report[
                "risk_corridor_dynamic_episode_checks"
            ]
            == report["dynamic_target_placements"] * 2,
            "contact_safe_geometry_gate_passed": report["contact_geometry_checks"] == 8
            and report["safe_geometry_checks"] == 8,
            "dynamic_density_gate_passed": report["dynamic_target_placements"] >= 68,
            "wearer_is_scripted_invincible": all(
                value["wearer"].get("scripted_invincible") is True
                for value in protocols.values()
            ),
            "frozen_weather_parameters_are_capture_materialized": all(
                value["c4_compatibility"]["weather_override_authority"]
                == "frozen_c4_weather_parameters_materialized_by_capture"
                for value in protocols.values()
            ),
        },
    }
    return protocols, compiled_index, receipt
