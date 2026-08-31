"""Capture one formal 1280x720 sensor shard for the CARLA C2 rich-scene source."""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
import time
from pathlib import Path
from typing import Any

import carla
import numpy as np

from dtr_carla_c2_rich_scene import (
    build_plan_receipt,
    camera_intrinsics,
    contact_union,
    layout_receipt,
    materialize_layout_assets,
    sha256_file,
    trajectory_position,
    trajectory_velocity,
    validate_protocol,
    write_json_atomic,
    write_jsonl,
)


SENSOR_TYPES = {
    "instance": "sensor.camera.instance_segmentation",
    "wearable": "sensor.camera.rgb",
    "depth": "sensor.camera.depth",
    "witness": "sensor.camera.rgb",
}

SCRIPTED_POSE_PLANAR_POSITION_TOLERANCE_M = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sensor", choices=tuple(SENSOR_TYPES), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--rpc-timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def connect(
    host: str,
    port: int,
    attempts: int = 30,
    rpc_timeout_seconds: float = 120.0,
) -> carla.Client:
    last_error: Exception | None = None
    for _ in range(attempts):
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        try:
            client.get_server_version()
            client.set_timeout(rpc_timeout_seconds)
            return client
        except Exception as exc:
            last_error = exc
            time.sleep(2.0)
    raise RuntimeError(f"CARLA server did not become ready: {last_error}")


def copy_settings(settings: carla.WorldSettings) -> dict[str, Any]:
    return {
        "synchronous_mode": bool(settings.synchronous_mode),
        "fixed_delta_seconds": settings.fixed_delta_seconds,
        "no_rendering_mode": bool(settings.no_rendering_mode),
        "substepping": bool(settings.substepping),
        "max_substep_delta_time": settings.max_substep_delta_time,
        "max_substeps": settings.max_substeps,
        "deterministic_ragdolls": bool(settings.deterministic_ragdolls),
    }


def apply_settings(world: carla.World, values: dict[str, Any]) -> None:
    settings = world.get_settings()
    for name, value in values.items():
        setattr(settings, name, value)
    world.apply_settings(settings)


def transform_dict(transform: carla.Transform) -> dict[str, float]:
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "pitch": float(transform.rotation.pitch),
        "yaw": float(transform.rotation.yaw),
        "roll": float(transform.rotation.roll),
    }


def angle_error_degrees(actual: float, expected: float) -> float:
    return abs((float(actual) - float(expected) + 180.0) % 360.0 - 180.0)


def transform_residual(
    actual: carla.Transform, expected: carla.Transform
) -> dict[str, float]:
    dx = float(actual.location.x) - float(expected.location.x)
    dy = float(actual.location.y) - float(expected.location.y)
    dz = float(actual.location.z) - float(expected.location.z)
    return {
        "planar_position_error_m": math.hypot(dx, dy),
        "position_error_m": math.sqrt(dx * dx + dy * dy + dz * dz),
        "angle_error_degrees": max(
            angle_error_degrees(actual.rotation.pitch, expected.rotation.pitch),
            angle_error_degrees(actual.rotation.yaw, expected.rotation.yaw),
            angle_error_degrees(actual.rotation.roll, expected.rotation.roll),
        ),
    }


def vector_dict(x: float, y: float, z: float = 0.0) -> dict[str, float]:
    return {"x": float(x), "y": float(y), "z": float(z)}


def bbox_dict(actor: carla.Actor, transform: carla.Transform) -> dict[str, Any]:
    bbox = actor.bounding_box
    vertices = bbox.get_world_vertices(transform)
    return {
        "location": vector_dict(bbox.location.x, bbox.location.y, bbox.location.z),
        "extent": vector_dict(bbox.extent.x, bbox.extent.y, bbox.extent.z),
        "rotation": {
            "pitch": float(bbox.rotation.pitch),
            "yaw": float(bbox.rotation.yaw),
            "roll": float(bbox.rotation.roll),
        },
        "world_vertices": [vector_dict(value.x, value.y, value.z) for value in vertices],
    }


def bbox_nonzero(actor: carla.Actor) -> bool:
    extent = actor.bounding_box.extent
    return max(float(extent.x), float(extent.y), float(extent.z)) > 1e-4


def polygon_from_bbox(actor: carla.Actor, transform: carla.Transform) -> list[list[float]]:
    unique: dict[tuple[float, float], tuple[float, float]] = {}
    for vertex in actor.bounding_box.get_world_vertices(transform):
        key = (round(float(vertex.x), 6), round(float(vertex.y), 6))
        unique[key] = (float(vertex.x), float(vertex.y))
    points = sorted(unique.values())
    if len(points) < 3:
        raise RuntimeError(f"{actor.type_id} has a degenerate XY bbox")

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
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise RuntimeError(f"{actor.type_id} has a degenerate XY bbox hull")
    return [[value[0], value[1]] for value in hull]


def await_frame(
    sensor_queue: queue.Queue[carla.SensorData],
    expected_frame: int,
    sensor_name: str,
    timeout_seconds: float = 30.0,
) -> carla.SensorData:
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {sensor_name} frame {expected_frame}")
        try:
            value = sensor_queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError(
                f"timed out waiting for {sensor_name} frame {expected_frame}"
            ) from exc
        if value.frame < expected_frame:
            continue
        if value.frame > expected_frame:
            raise RuntimeError(
                f"{sensor_name} skipped frame {expected_frame} and returned {value.frame}"
            )
        return value


def sensor_blueprint(
    world: carla.World,
    type_id: str,
    width: int,
    height: int,
    fov_degrees: float,
    sensor_tick: float,
) -> carla.ActorBlueprint:
    blueprint = world.get_blueprint_library().find(type_id)
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", f"{fov_degrees:.6f}")
    blueprint.set_attribute("sensor_tick", f"{sensor_tick:.9f}")
    if type_id == "sensor.camera.rgb" and blueprint.has_attribute(
        "enable_postprocess_effects"
    ):
        blueprint.set_attribute("enable_postprocess_effects", "true")
    return blueprint


def normalized_anchor(layout: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor = layout["anchor"]
    center = np.asarray(anchor["center_xy_m"], dtype=np.float64)
    forward = np.asarray(anchor["forward_xy"], dtype=np.float64)
    right = np.asarray(anchor["right_xy"], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right /= np.linalg.norm(right)
    return center, forward, right


def local_to_world(
    local: tuple[float, float],
    center: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return center + forward * float(local[0]) + right * float(local[1])


def local_velocity_to_world(
    local: tuple[float, float], forward: np.ndarray, right: np.ndarray
) -> np.ndarray:
    return forward * float(local[0]) + right * float(local[1])


def road_surface_z(world_map: carla.Map, xy: np.ndarray, offset_m: float) -> float:
    waypoint = world_map.get_waypoint(
        carla.Location(x=float(xy[0]), y=float(xy[1]), z=0.0),
        project_to_road=True,
        lane_type=carla.LaneType.Any,
    )
    if waypoint is None:
        raise RuntimeError(f"no surface waypoint near {xy.tolist()}")
    return float(waypoint.transform.location.z) + float(offset_m)


def trajectory_for_asset(
    asset: dict[str, Any], scenario: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any] | None:
    if "trajectory_key" in asset:
        name = str(scenario["asset_trajectories"][str(asset["trajectory_key"])])
        return protocol["trajectory_library"][name]
    if "trajectory" in asset:
        return protocol["trajectory_library"][str(asset["trajectory"])]
    return None


def pose_for_asset(
    asset: dict[str, Any],
    scenario: dict[str, Any],
    protocol: dict[str, Any],
    time_s: float,
    *,
    world_map: carla.Map,
    center: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    trajectory = trajectory_for_asset(asset, scenario, protocol)
    base_yaw = math.degrees(math.atan2(float(forward[1]), float(forward[0])))
    if trajectory is None:
        fixed = asset["fixed_pose"]
        local = (float(fixed["forward_m"]), float(fixed["right_m"]))
        velocity_local = (0.0, 0.0)
        yaw = base_yaw + float(fixed.get("yaw_offset_degrees", 0.0))
    else:
        local = trajectory_position(trajectory, time_s)
        velocity_local = trajectory_velocity(trajectory, time_s)
        if "yaw_offset_degrees" in trajectory:
            yaw = base_yaw + float(trajectory["yaw_offset_degrees"])
        else:
            velocity_world = local_velocity_to_world(velocity_local, forward, right)
            yaw = (
                math.degrees(
                    math.atan2(float(velocity_world[1]), float(velocity_world[0]))
                )
                if float(np.linalg.norm(velocity_world)) > 1e-9
                else base_yaw
            )
    xy = local_to_world(local, center, forward, right)
    velocity_world = local_velocity_to_world(velocity_local, forward, right)
    z = road_surface_z(world_map, xy, float(asset["surface_offset_m"]))
    return {
        "local": local,
        "velocity_world": velocity_world,
        "transform": carla.Transform(
            carla.Location(x=float(xy[0]), y=float(xy[1]), z=z),
            carla.Rotation(yaw=yaw),
        ),
    }


def pose_for_wearer(
    scenario: dict[str, Any],
    protocol: dict[str, Any],
    time_s: float,
    *,
    world_map: carla.Map,
    center: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    trajectory = protocol["trajectory_library"][str(scenario["wearer_trajectory"])]
    local = trajectory_position(trajectory, time_s)
    velocity_local = trajectory_velocity(trajectory, time_s)
    velocity_world = local_velocity_to_world(velocity_local, forward, right)
    base_yaw = math.degrees(math.atan2(float(forward[1]), float(forward[0])))
    xy = local_to_world(local, center, forward, right)
    z = road_surface_z(
        world_map, xy, float(protocol["wearer"]["surface_offset_m"])
    )
    return {
        "local": local,
        "velocity_world": velocity_world,
        "transform": carla.Transform(
            carla.Location(x=float(xy[0]), y=float(xy[1]), z=z),
            carla.Rotation(yaw=base_yaw),
        ),
    }


def resolve_blueprint(
    library: carla.BlueprintLibrary, candidates: list[str]
) -> tuple[carla.ActorBlueprint, int]:
    for index, candidate in enumerate(candidates):
        matches = [value for value in library.filter(candidate) if value.id == candidate]
        if matches:
            return matches[0], index
    raise RuntimeError(f"none of the frozen blueprint candidates exist: {candidates}")


def deterministic_blueprint_attributes(
    blueprint: carla.ActorBlueprint,
    role_name: str,
    *,
    scripted_invincible: bool,
) -> dict[str, str]:
    applied: dict[str, str] = {}
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)
        applied["role_name"] = role_name
    if blueprint.has_attribute("is_invincible"):
        value = "true" if scripted_invincible else "false"
        blueprint.set_attribute("is_invincible", value)
        applied["is_invincible"] = value
    for name in ("color", "driver_id"):
        if blueprint.has_attribute(name):
            values = list(blueprint.get_attribute(name).recommended_values)
            if values:
                blueprint.set_attribute(name, str(values[0]))
                applied[name] = str(values[0])
    return applied


def spawn_asset(
    world: carla.World,
    asset: dict[str, Any],
    initial_pose: dict[str, Any],
    unique_index: int,
) -> tuple[carla.Actor, dict[str, Any]]:
    blueprint, fallback_index = resolve_blueprint(
        world.get_blueprint_library(), list(asset["blueprint_candidates"])
    )
    attributes = deterministic_blueprint_attributes(
        blueprint,
        f"dtr_c2_{asset['asset_key']}",
        scripted_invincible=bool(asset.get("scripted_invincible", False)),
    )
    actor = world.try_spawn_actor(blueprint, initial_pose["transform"])
    spawn_strategy = "scene_pose"
    if actor is None:
        source = initial_pose["transform"]
        hidden = carla.Transform(
            carla.Location(
                x=float(source.location.x),
                y=float(source.location.y),
                z=float(source.location.z) + 30.0 + unique_index,
            ),
            source.rotation,
        )
        actor = world.try_spawn_actor(blueprint, hidden)
        spawn_strategy = "elevated_then_teleported"
    if actor is None:
        raise RuntimeError(
            f"failed to spawn {asset['asset_key']} from {asset['blueprint_candidates']}"
        )
    simulate_physics_disabled = False
    try:
        actor.set_simulate_physics(False)
        simulate_physics_disabled = True
    except Exception as error:
        if bool(asset.get("scripted_pose_authority", False)):
            raise RuntimeError(
                f"failed to disable physics for authoritative scripted asset "
                f"{asset['asset_key']}"
            ) from error
    collision_manifest: dict[str, Any] = {}
    if "collisions_enabled" in asset:
        collisions_enabled = bool(asset["collisions_enabled"])
        actor.set_collisions(collisions_enabled)
        collision_manifest["collisions_enabled"] = collisions_enabled
    if str(asset["kind"]) == "walker":
        try:
            actor.apply_control(carla.WalkerControl(speed=0.0))
        except Exception:
            pass
    actor.set_transform(initial_pose["transform"])
    extent = actor.bounding_box.extent
    return actor, {
        "asset_key": str(asset["asset_key"]),
        "track_id": str(asset["track_id"]),
        "role": str(asset["role"]),
        "kind": str(asset["kind"]),
        "template": str(asset.get("template", "wearer")),
        "candidate_blueprints": list(asset["blueprint_candidates"]),
        "actual_blueprint": str(actor.type_id),
        "fallback_index": int(fallback_index),
        "spawn_strategy": spawn_strategy,
        "simulate_physics_disabled": simulate_physics_disabled,
        "attributes": attributes,
        "carla_actor_id": int(actor.id),
        "bbox_extent": vector_dict(extent.x, extent.y, extent.z),
        "bbox_nonzero": bbox_nonzero(actor),
        **collision_manifest,
    }


def look_at_transform(
    layout: dict[str, Any],
    *,
    center: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
) -> carla.Transform:
    witness = layout["witness"]
    source_xy = local_to_world(
        (float(witness["forward_m"]), float(witness["right_m"])),
        center,
        forward,
        right,
    )
    target_xy = local_to_world(
        (
            float(witness["look_at_forward_m"]),
            float(witness["look_at_right_m"]),
        ),
        center,
        forward,
        right,
    )
    dx = float(target_xy[0] - source_xy[0])
    dy = float(target_xy[1] - source_xy[1])
    dz = float(witness["look_at_z_m"] - witness["z_m"])
    return carla.Transform(
        carla.Location(
            x=float(source_xy[0]), y=float(source_xy[1]), z=float(witness["z_m"])
        ),
        carla.Rotation(
            pitch=math.degrees(math.atan2(dz, math.hypot(dx, dy))),
            yaw=math.degrees(math.atan2(dy, dx)),
        ),
    )


def save_image(image: carla.Image, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save_to_disk(str(path))
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"CARLA did not materialize image: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "world_frame": int(image.frame),
        "sensor_timestamp_s": float(image.timestamp),
        "width": int(image.width),
        "height": int(image.height),
    }


def batched_instance_metrics(
    image: carla.Image, actors: dict[str, carla.Actor]
) -> dict[str, dict[str, Any]]:
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (image.height, image.width, 4)
    )
    instance_ids = bgra[:, :, 1].astype(np.uint32) | (
        bgra[:, :, 0].astype(np.uint32) << 8
    )
    total_pixels = int(image.width) * int(image.height)
    values: dict[str, dict[str, Any]] = {}
    for key, actor in actors.items():
        if key == "wearer":
            continue
        ys, xs = np.nonzero(instance_ids == int(actor.id))
        pixels = int(xs.size)
        values[key] = {
            "pixels": pixels,
            "pixel_fraction": float(pixels / total_pixels),
            "visible": pixels > 0,
            "bbox_uv_normalized": (
                [
                    float(xs.min() / image.width),
                    float(ys.min() / image.height),
                    float((xs.max() + 1) / image.width),
                    float((ys.max() + 1) / image.height),
                ]
                if pixels
                else None
            ),
        }
    return values


def apply_scene(
    wearer: carla.Actor,
    actors: dict[str, carla.Actor],
    assets: list[dict[str, Any]],
    scenario: dict[str, Any],
    protocol: dict[str, Any],
    time_s: float,
    *,
    world_map: carla.Map,
    center: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    apply_transforms: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    wearer_pose = pose_for_wearer(
        scenario,
        protocol,
        time_s,
        world_map=world_map,
        center=center,
        forward=forward,
        right=right,
    )
    if apply_transforms:
        wearer.set_transform(wearer_pose["transform"])
        try:
            wearer.apply_control(carla.WalkerControl(speed=0.0))
        except Exception:
            pass
    poses: dict[str, dict[str, Any]] = {}
    for asset in assets:
        key = str(asset["asset_key"])
        pose = pose_for_asset(
            asset,
            scenario,
            protocol,
            time_s,
            world_map=world_map,
            center=center,
            forward=forward,
            right=right,
        )
        if apply_transforms:
            actors[key].set_transform(pose["transform"])
            if str(asset["kind"]) == "walker":
                try:
                    actors[key].apply_control(carla.WalkerControl(speed=0.0))
                except Exception:
                    pass
        poses[key] = pose
    return wearer_pose, poses


def tick_scripted_scene(
    client: carla.Client,
    world: carla.World,
    sensor_queue: queue.Queue[carla.SensorData],
    sensor_name: str,
    wearer: carla.Actor,
    actors: dict[str, carla.Actor],
    wearer_pose: dict[str, Any],
    poses: dict[str, dict[str, Any]],
    pose_authority_keys: set[str],
) -> tuple[
    int,
    carla.SensorData,
    dict[str, carla.Transform],
    dict[str, dict[str, float]],
]:
    labels = ["wearer", *sorted(actors)]
    commands = [carla.command.ApplyTransform(wearer.id, wearer_pose["transform"])]
    commands.extend(
        carla.command.ApplyTransform(actors[key].id, poses[key]["transform"])
        for key in labels[1:]
    )
    previous_frame = int(world.get_snapshot().frame)
    responses = client.apply_batch_sync(commands, do_tick=True)
    if len(responses) != len(commands):
        raise RuntimeError(
            f"scripted pose batch response count differs: {len(responses)} != {len(commands)}"
        )
    failures = [
        {"actor": label, "error": str(response.error)}
        for label, response in zip(labels, responses, strict=True)
        if response.has_error()
    ]
    if failures:
        raise RuntimeError(f"scripted pose batch failed: {json.dumps(failures)}")

    snapshot = world.get_snapshot()
    world_frame = int(snapshot.frame)
    if world_frame != previous_frame + 1:
        raise RuntimeError(
            f"scripted pose batch advanced unexpected frame: {previous_frame} -> {world_frame}"
        )
    image = await_frame(sensor_queue, world_frame, sensor_name)

    expected = {"wearer": wearer_pose["transform"]}
    expected.update({key: poses[key]["transform"] for key in sorted(actors)})
    actor_by_label = {"wearer": wearer, **actors}
    actual: dict[str, carla.Transform] = {}
    residuals: dict[str, dict[str, float]] = {}
    pose_failures: list[dict[str, Any]] = []
    required_labels = {"wearer", *pose_authority_keys}
    for label in labels:
        actor_snapshot = snapshot.find(actor_by_label[label].id)
        if actor_snapshot is None:
            raise RuntimeError(f"snapshot omitted task-owned actor {label}")
        actual_transform = actor_snapshot.get_transform()
        actual[label] = actual_transform
        residual = transform_residual(actual_transform, expected[label])
        residuals[label] = residual
        if (
            label in required_labels
            and residual["planar_position_error_m"]
            > SCRIPTED_POSE_PLANAR_POSITION_TOLERANCE_M
        ):
            pose_failures.append(
                {
                    "actor": label,
                    **residual,
                    "expected": transform_dict(expected[label]),
                    "actual": transform_dict(actual_transform),
                }
            )
    if pose_failures:
        raise RuntimeError(
            "captured frame differs from authoritative scripted pose: "
            + json.dumps(pose_failures[:5], sort_keys=True)
        )
    return world_frame, image, actual, residuals


def weather_parameter(name: str) -> carla.WeatherParameters:
    value = getattr(carla.WeatherParameters, name, None)
    if value is None:
        raise ValueError(f"unknown CARLA weather preset: {name}")
    return carla.WeatherParameters(
        cloudiness=float(value.cloudiness),
        precipitation=float(value.precipitation),
        precipitation_deposits=float(value.precipitation_deposits),
        wind_intensity=float(value.wind_intensity),
        sun_azimuth_angle=float(value.sun_azimuth_angle),
        sun_altitude_angle=float(value.sun_altitude_angle),
        fog_density=float(value.fog_density),
        fog_distance=float(value.fog_distance),
        fog_falloff=float(value.fog_falloff),
        wetness=float(value.wetness),
        scattering_intensity=float(value.scattering_intensity),
        mie_scattering_scale=float(value.mie_scattering_scale),
        rayleigh_scattering_scale=float(value.rayleigh_scattering_scale),
        dust_storm=float(value.dust_storm),
    )


def layout_weather_parameter(layout: dict[str, Any]) -> carla.WeatherParameters:
    """Materialize the preset plus optional frozen C4 weather overrides."""

    weather = weather_parameter(str(layout["weather"]))
    overrides = layout.get("c4_weather_parameters")
    if overrides is None:
        return weather
    if not isinstance(overrides, dict):
        raise ValueError("c4_weather_parameters must be an object")
    supported = {
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "wind_intensity",
        "sun_azimuth_angle",
        "sun_altitude_angle",
        "fog_density",
        "wetness",
    }
    unknown = sorted(set(overrides) - supported)
    if unknown:
        raise ValueError(f"unsupported C4 weather parameters: {unknown}")
    for name, raw_value in overrides.items():
        setattr(weather, name, float(raw_value))
    return weather


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    output_root = args.output_root.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    sensor_name = str(args.sensor)
    if sensor_name not in protocol["capture"]["sensor_order"]:
        raise ValueError(f"sensor is not frozen by the protocol: {sensor_name}")
    if not output_root.is_dir():
        raise FileNotFoundError(f"runner must reserve output root first: {output_root}")
    shard_root = output_root / "shards" / sensor_name
    if shard_root.exists():
        raise FileExistsError(f"refusing shard overwrite: {shard_root}")
    shard_root.mkdir(parents=True)
    protocol_snapshot = output_root / "frozen_protocol.json"
    if protocol_snapshot.exists():
        if sha256_file(protocol_snapshot) != sha256_file(protocol_path):
            raise RuntimeError("frozen protocol changed between sensor shards")
    else:
        protocol_snapshot.write_bytes(protocol_path.read_bytes())

    random.seed(int(protocol["capture"]["seed"]))
    np.random.seed(int(protocol["capture"]["seed"]))
    fixed_delta = float(protocol["environment"]["fixed_delta_seconds"])
    width, height = map(int, protocol["capture"]["resolution"])
    fov_degrees = float(protocol["capture"]["fov_degrees"])
    expected_total_frames = sum(
        int(round(float(protocol["layouts"][scenario["layout_id"]]["duration_seconds"]) / fixed_delta))
        + 1
        for scenario in protocol["scenarios"]
    )
    receipts = {
        str(value["episode_id"]): build_plan_receipt(value.get("issued_plan"))
        for value in protocol["scenarios"]
    }

    client = connect(
        args.host,
        args.port,
        rpc_timeout_seconds=args.rpc_timeout_seconds,
    )
    if client.get_client_version() != protocol["environment"]["carla_version"]:
        raise RuntimeError(f"unexpected CARLA client version: {client.get_client_version()}")
    if client.get_server_version() != protocol["environment"]["carla_version"]:
        raise RuntimeError(f"unexpected CARLA server version: {client.get_server_version()}")
    world = client.get_world()
    if world.get_map().name != protocol["environment"]["map"]:
        raise RuntimeError(
            f"required default map is unavailable; refusing dynamic load: {world.get_map().name}"
        )
    shared = [
        actor
        for actor in world.get_actors()
        if actor.type_id.startswith(("vehicle.", "walker.", "sensor."))
    ]
    if shared:
        raise RuntimeError("refusing a shared CARLA world")

    original_settings = copy_settings(world.get_settings())
    original_weather = world.get_weather()
    wearer: carla.Actor | None = None
    sensor: carla.Sensor | None = None
    owned: list[carla.Actor] = []
    scene_actors: dict[str, carla.Actor] = {}
    sensor_queue: queue.Queue[carla.SensorData] = queue.Queue()
    payload_inventory: list[dict[str, Any]] = []
    episode_manifests: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    sensor_alignment_ok = True
    max_scripted_pose_planar_position_error_m = 0.0

    try:
        apply_settings(
            world,
            {
                "synchronous_mode": False,
                "fixed_delta_seconds": None,
                "no_rendering_mode": False,
                "substepping": True,
                "max_substep_delta_time": 0.01,
                "max_substeps": 10,
                "deterministic_ragdolls": True,
            },
        )
        first_scenario = protocol["scenarios"][0]
        first_layout = protocol["layouts"][first_scenario["layout_id"]]
        center, forward, right = normalized_anchor(first_layout)
        initial_wearer_pose = pose_for_wearer(
            first_scenario,
            protocol,
            0.0,
            world_map=world.get_map(),
            center=center,
            forward=forward,
            right=right,
        )
        wearer_spec = dict(protocol["wearer"])
        wearer_spec["template"] = "wearer"
        wearer, wearer_manifest = spawn_asset(world, wearer_spec, initial_wearer_pose, 0)
        owned.append(wearer)

        blueprint = sensor_blueprint(
            world,
            SENSOR_TYPES[sensor_name],
            width,
            height,
            fov_degrees,
            fixed_delta,
        )
        relative = protocol["capture"]["wearable_relative_transform"]
        relative_transform = carla.Transform(
            carla.Location(
                x=float(relative["x_m"]),
                y=float(relative["y_m"]),
                z=float(relative["z_m"]),
            ),
            carla.Rotation(
                pitch=float(relative["pitch_degrees"]),
                yaw=float(relative["yaw_degrees"]),
                roll=float(relative["roll_degrees"]),
            ),
        )
        if sensor_name == "witness":
            sensor = world.spawn_actor(
                blueprint,
                look_at_transform(
                    first_layout, center=center, forward=forward, right=right
                ),
            )
        else:
            sensor = world.spawn_actor(
                blueprint,
                relative_transform,
                attach_to=wearer,
                attachment_type=carla.AttachmentType.Rigid,
            )
        sensor.listen(sensor_queue.put)
        owned.append(sensor)

        warmup = 0
        deadline = time.monotonic() + 120.0
        while warmup < 5:
            try:
                sensor_queue.get(timeout=0.25)
                warmup += 1
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{sensor_name} camera warmup timed out")

        apply_settings(
            world,
            {
                "synchronous_mode": True,
                "fixed_delta_seconds": fixed_delta,
                "no_rendering_mode": False,
                "substepping": True,
                "max_substep_delta_time": 0.01,
                "max_substeps": 10,
                "deterministic_ragdolls": True,
            },
        )
        while True:
            try:
                sensor_queue.get_nowait()
            except queue.Empty:
                break
        aligned = False
        for _ in range(8):
            frame = world.tick(30.0)
            try:
                await_frame(sensor_queue, frame, sensor_name, timeout_seconds=5.0)
                aligned = True
                break
            except TimeoutError:
                continue
        if not aligned:
            raise TimeoutError(f"could not phase-align the {sensor_name} camera")

        unique_actual_blueprints = {str(wearer.type_id)}
        all_fallback_indices = [int(wearer_manifest["fallback_index"])]
        all_bbox_nonzero = [bool(wearer_manifest["bbox_nonzero"])]
        total_spawn_counter = 1

        for scenario_index, scenario in enumerate(protocol["scenarios"]):
            episode_id = str(scenario["episode_id"])
            layout_id = str(scenario["layout_id"])
            layout = protocol["layouts"][layout_id]
            assets = materialize_layout_assets(protocol, layout_id)
            pose_authority_keys = {
                str(asset["asset_key"])
                for asset in assets
                if bool(asset.get("scripted_pose_authority", False))
            }
            center, forward, right = normalized_anchor(layout)

            if scene_actors:
                actor_ids = sorted(int(value.id) for value in scene_actors.values())
                client.apply_batch_sync(
                    [carla.command.DestroyActor(value) for value in actor_ids],
                    do_tick=True,
                )
                owned = [value for value in owned if int(value.id) not in set(actor_ids)]
                scene_actors.clear()
                while True:
                    try:
                        sensor_queue.get_nowait()
                    except queue.Empty:
                        break

            world.set_weather(layout_weather_parameter(layout))
            initial_wearer_pose = pose_for_wearer(
                scenario,
                protocol,
                0.0,
                world_map=world.get_map(),
                center=center,
                forward=forward,
                right=right,
            )
            wearer.set_transform(initial_wearer_pose["transform"])
            if sensor_name == "witness":
                sensor.set_transform(
                    look_at_transform(
                        layout, center=center, forward=forward, right=right
                    )
                )

            active_manifest: list[dict[str, Any]] = [
                {**wearer_manifest, "episode_id": episode_id, "layout_id": layout_id}
            ]
            for asset in assets:
                initial_pose = pose_for_asset(
                    asset,
                    scenario,
                    protocol,
                    0.0,
                    world_map=world.get_map(),
                    center=center,
                    forward=forward,
                    right=right,
                )
                actor, manifest = spawn_asset(
                    world, asset, initial_pose, total_spawn_counter
                )
                total_spawn_counter += 1
                scene_actors[str(asset["asset_key"])] = actor
                owned.append(actor)
                enriched_manifest = {
                    **manifest,
                    "episode_id": episode_id,
                    "layout_id": layout_id,
                }
                active_manifest.append(enriched_manifest)
                unique_actual_blueprints.add(str(actor.type_id))
                all_fallback_indices.append(int(manifest["fallback_index"]))
                all_bbox_nonzero.append(bool(manifest["bbox_nonzero"]))

            apply_scene(
                wearer,
                scene_actors,
                assets,
                scenario,
                protocol,
                0.0,
                world_map=world.get_map(),
                center=center,
                forward=forward,
                right=right,
            )
            for _ in range(2):
                reset_frame = world.tick(30.0)
                await_frame(sensor_queue, reset_frame, sensor_name)
                apply_scene(
                    wearer,
                    scene_actors,
                    assets,
                    scenario,
                    protocol,
                    0.0,
                    world_map=world.get_map(),
                    center=center,
                    forward=forward,
                    right=right,
                )

            episode_root = shard_root / "episodes" / episode_id
            payload_root = episode_root / "payload"
            payload_root.mkdir(parents=True)
            duration_s = float(layout["duration_seconds"])
            frame_count = int(round(duration_s / fixed_delta)) + 1
            records: list[dict[str, Any]] = []
            episode_payloads: list[dict[str, Any]] = []
            receipt = receipts[episode_id]
            frozen_layout_receipt = layout_receipt(protocol, layout_id)

            for sample_index in range(frame_count):
                time_s = sample_index * fixed_delta
                wearer_pose, poses = apply_scene(
                    wearer,
                    scene_actors,
                    assets,
                    scenario,
                    protocol,
                    time_s,
                    world_map=world.get_map(),
                    center=center,
                    forward=forward,
                    right=right,
                    apply_transforms=False,
                )
                (
                    world_frame,
                    image,
                    actual_transforms,
                    pose_residuals,
                ) = tick_scripted_scene(
                    client,
                    world,
                    sensor_queue,
                    sensor_name,
                    wearer,
                    scene_actors,
                    wearer_pose,
                    poses,
                    pose_authority_keys,
                )
                sensor_alignment_ok = sensor_alignment_ok and int(image.frame) == int(
                    world_frame
                )
                if len(image.raw_data) <= 0:
                    raise RuntimeError(
                        f"empty {sensor_name} payload at {episode_id}/{sample_index}"
                    )
                actor_states: dict[str, dict[str, Any]] = {}
                collision_polygons: dict[str, list[list[float]]] = {}
                wearer_transform = actual_transforms["wearer"]
                max_scripted_pose_planar_position_error_m = max(
                    max_scripted_pose_planar_position_error_m,
                    pose_residuals["wearer"]["planar_position_error_m"],
                )
                actor_states["wearer"] = {
                    "track_id": str(protocol["wearer"]["track_id"]),
                    "asset_key": "wearer",
                    "role": "wearer",
                    "kind": "walker",
                    "actual_blueprint": str(wearer.type_id),
                    "carla_actor_id": int(wearer.id),
                    "transform": transform_dict(wearer_transform),
                    "scripted_command_transform": transform_dict(
                        wearer_pose["transform"]
                    ),
                    "scripted_pose_residual": pose_residuals["wearer"],
                    "local_position": {
                        "forward_m": float(wearer_pose["local"][0]),
                        "right_m": float(wearer_pose["local"][1]),
                    },
                    "command_velocity": vector_dict(
                        wearer_pose["velocity_world"][0],
                        wearer_pose["velocity_world"][1],
                    ),
                    "bounding_box": bbox_dict(wearer, wearer_transform),
                }
                for asset in assets:
                    key = str(asset["asset_key"])
                    actor = scene_actors[key]
                    pose = poses[key]
                    scripted_transform = pose["transform"]
                    actual_transform = actual_transforms[key]
                    if key in pose_authority_keys:
                        max_scripted_pose_planar_position_error_m = max(
                            max_scripted_pose_planar_position_error_m,
                            pose_residuals[key]["planar_position_error_m"],
                        )
                    actor_states[key] = {
                        "track_id": str(asset["track_id"]),
                        "asset_key": key,
                        "role": str(asset["role"]),
                        "kind": str(asset["kind"]),
                        "actual_blueprint": str(actor.type_id),
                        "carla_actor_id": int(actor.id),
                        "transform": transform_dict(actual_transform),
                        "scripted_command_transform": transform_dict(
                            scripted_transform
                        ),
                        "scripted_pose_residual": pose_residuals[key],
                        "local_position": {
                            "forward_m": float(pose["local"][0]),
                            "right_m": float(pose["local"][1]),
                        },
                        "command_velocity": vector_dict(
                            pose["velocity_world"][0], pose["velocity_world"][1]
                        ),
                        "bounding_box": bbox_dict(actor, actual_transform),
                    }
                    if bool(asset["collision_relevant"]):
                        collision_polygons[key] = polygon_from_bbox(
                            actor, actual_transform
                        )

                wearer_xy = local_to_world(wearer_pose["local"], center, forward, right)
                current_contact, minimum_distance, responsible = contact_union(
                    (float(wearer_xy[0]), float(wearer_xy[1])),
                    collision_polygons,
                    wearer_radius_m=float(
                        protocol["route_contract"]["wearer_body_radius_m"]
                    ),
                )
                file_path = payload_root / f"{sample_index:06d}.png"
                payload = save_image(image, file_path)
                payload.update(
                    {
                        "sensor": sensor_name,
                        "episode_id": episode_id,
                        "sample_index": sample_index,
                        "relative_path": str(file_path.relative_to(shard_root)).replace(
                            "\\", "/"
                        ),
                    }
                )
                payload_inventory.append(payload)
                episode_payloads.append(payload)
                visibility = (
                    batched_instance_metrics(image, {"wearer": wearer, **scene_actors})
                    if sensor_name == "instance"
                    else None
                )
                records.append(
                    {
                        "schema_version": "dtr-c2-raw-shard-frame-v1",
                        "sensor": sensor_name,
                        "episode_id": episode_id,
                        "layout_id": layout_id,
                        "sample_index": sample_index,
                        "time_s": time_s,
                        "sensor_timestamp_s": float(image.timestamp),
                        "world_frame": int(world_frame),
                        "sensor_path": payload["relative_path"],
                        "sensor_payload_bytes": int(payload["bytes"]),
                        "sensor_payload_sha256": str(payload["sha256"]),
                        "camera_transform": transform_dict(image.transform),
                        "wearer_transform": transform_dict(wearer_transform),
                        "layout_receipt_sha256": frozen_layout_receipt[
                            "receipt_sha256"
                        ],
                        "plan_receipt_sha256": (
                            receipt["receipt_sha256"] if receipt else None
                        ),
                        "actors": actor_states,
                        "instance_visibility": visibility,
                        "truth": {
                            "scenario_role": str(scenario["scenario_role"]),
                            "twin_role": str(scenario["twin_role"]),
                            "expected_outcome": str(scenario["expected_outcome"]),
                            "expected_responsible_assets": list(
                                scenario["expected_responsible_assets"]
                            ),
                            "current_contact": bool(current_contact),
                            "minimum_distance_m": float(minimum_distance),
                            "responsible_assets": responsible,
                            "collision_polygons_xy": collision_polygons,
                        },
                    }
                )

            horizon_s = float(protocol["route_contract"]["future_horizon_seconds"])
            for index, record in enumerate(records):
                current_time = float(record["time_s"])
                future_contacts = [
                    value
                    for value in records[index:]
                    if float(value["time_s"]) - current_time <= horizon_s + 1e-9
                    and bool(value["truth"]["current_contact"])
                ]
                record["truth"]["future_contact_within_horizon"] = bool(
                    future_contacts
                )
                record["truth"]["realized_time_to_contact_seconds"] = (
                    float(future_contacts[0]["time_s"]) - current_time
                    if future_contacts
                    else None
                )

            frames_path = episode_root / "frames.jsonl"
            inventory_path = episode_root / "payload_inventory.json"
            asset_manifest_path = episode_root / "asset_manifest.json"
            write_jsonl(frames_path, records)
            write_json_atomic(inventory_path, episode_payloads)
            write_json_atomic(asset_manifest_path, active_manifest)
            contact_rows = [
                value for value in records if bool(value["truth"]["current_contact"])
            ]
            observed_responsible = sorted(
                {
                    str(key)
                    for value in contact_rows
                    for key in value["truth"]["responsible_assets"]
                }
            )
            observed_outcome = "CONTACT" if contact_rows else "SAFE"
            summary = {
                "episode_id": episode_id,
                "layout_id": layout_id,
                "sensor": sensor_name,
                "frames": len(records),
                "duration_seconds": duration_s,
                "scripted_pose_authority_assets": sorted(pose_authority_keys),
                "active_asset_count_including_wearer": len(active_manifest),
                "unique_actual_blueprints": len(
                    {str(value["actual_blueprint"]) for value in active_manifest}
                ),
                "expected_outcome": str(scenario["expected_outcome"]),
                "observed_outcome": observed_outcome,
                "expected_responsible_assets": sorted(
                    str(value) for value in scenario["expected_responsible_assets"]
                ),
                "observed_responsible_assets": observed_responsible,
                "first_contact_time_s": (
                    float(contact_rows[0]["time_s"]) if contact_rows else None
                ),
                "minimum_distance_m": min(
                    float(value["truth"]["minimum_distance_m"]) for value in records
                ),
                "target_visible_frames": (
                    sum(
                        bool(value["instance_visibility"]["target_primary"]["visible"])
                        for value in records
                    )
                    if sensor_name == "instance" and layout_id == "layout_01"
                    else None
                ),
                "layout_receipt_sha256": frozen_layout_receipt["receipt_sha256"],
                "plan_receipt_sha256": receipt["receipt_sha256"] if receipt else None,
            }
            summary_path = episode_root / "summary.json"
            write_json_atomic(summary_path, summary)
            write_json_atomic(
                episode_root / "manifest.json",
                {
                    "schema_version": "dtr-c2-raw-shard-episode-manifest-v1",
                    "episode_id": episode_id,
                    "layout_id": layout_id,
                    "sensor": sensor_name,
                    "frames": len(records),
                    "frames_sha256": sha256_file(frames_path),
                    "summary_sha256": sha256_file(summary_path),
                    "payload_count": len(episode_payloads),
                    "payload_inventory_sha256": sha256_file(inventory_path),
                    "asset_manifest_sha256": sha256_file(asset_manifest_path),
                },
            )
            episode_manifests.extend(active_manifest)
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)

        calibration = {
            "schema_version": "dtr-c2-camera-calibration-v1",
            "sensor": sensor_name,
            "sensor_type": SENSOR_TYPES[sensor_name],
            "width": width,
            "height": height,
            "fov_degrees": fov_degrees,
            "K": camera_intrinsics(width, height, fov_degrees),
            "sensor_tick_seconds": fixed_delta,
            "wearable_relative_transform": (
                protocol["capture"]["wearable_relative_transform"]
                if sensor_name != "witness"
                else None
            ),
            "depth_codec": (
                protocol["capture"]["camera_calibration"]["depth_codec"]
                if sensor_name == "depth"
                else None
            ),
        }
        calibration_path = shard_root / "camera_calibration.json"
        inventory_path = shard_root / "payload_inventory.json"
        manifest_path = shard_root / "episode_asset_manifests.json"
        write_json_atomic(calibration_path, calibration)
        write_json_atomic(inventory_path, payload_inventory)
        write_json_atomic(manifest_path, episode_manifests)
        checks = {
            "all_expected_episodes_captured": len(summaries)
            == int(protocol["admission"]["expected_episode_count"]),
            "all_expected_outcomes_match": all(
                value["expected_outcome"] == value["observed_outcome"]
                for value in summaries
            ),
            "all_expected_responsible_sets_match": all(
                value["expected_responsible_assets"]
                == value["observed_responsible_assets"]
                for value in summaries
            ),
            "all_formal_payloads_are_1280x720": all(
                int(value["width"]) == 1280 and int(value["height"]) == 720
                for value in payload_inventory
            ),
            "all_raw_payloads_materialized": len(payload_inventory)
            == expected_total_frames,
            "sensor_world_frames_aligned": sensor_alignment_ok,
            "all_captured_frames_match_authoritative_scripted_pose": (
                max_scripted_pose_planar_position_error_m
                <= SCRIPTED_POSE_PLANAR_POSITION_TOLERANCE_M
            ),
            "minimum_active_assets_met": all(
                int(value["active_asset_count_including_wearer"]) - 1
                >= int(
                    protocol["admission"][
                        "minimum_active_assets_per_layout_excluding_wearer"
                    ]
                )
                for value in summaries
            ),
            "minimum_unique_blueprints_met": len(unique_actual_blueprints)
            >= int(
                protocol["admission"][
                    "minimum_unique_actual_blueprints_across_pack"
                ]
            ),
            "zero_blueprint_fallbacks": all(value == 0 for value in all_fallback_indices),
            "all_spawned_assets_have_nonzero_bbox": all(all_bbox_nonzero),
            "calibration_matches_formal_resolution": calibration["K"]
            == camera_intrinsics(1280, 720, fov_degrees),
        }
        if sensor_name == "instance":
            pair_summaries = [
                value for value in summaries if value["layout_id"] == "layout_01"
            ]
            checks["pair_target_has_visible_and_hidden_samples"] = all(
                0 < int(value["target_visible_frames"]) < int(value["frames"])
                for value in pair_summaries
            )
        result = {
            "schema_version": "dtr-carla-c2-raw-shard-result-v1",
            "experiment_id": protocol["experiment_id"],
            "status": (
                "DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE"
                if all(checks.values())
                else "DTR_CARLA_C2_RAW_SHARD_CAPTURE_NOT_EVALUABLE"
            ),
            "sensor": sensor_name,
            "protocol_sha256": sha256_file(protocol_path),
            "capture_script_sha256": sha256_file(Path(__file__).resolve()),
            "helper_module_sha256": sha256_file(
                Path(__file__).with_name("dtr_carla_c2_rich_scene.py")
            ),
            "carla_client_version": client.get_client_version(),
            "carla_server_version": client.get_server_version(),
            "map": world.get_map().name,
            "checks": checks,
            "episodes": summaries,
            "payload_count": len(payload_inventory),
            "payload_inventory_sha256": sha256_file(inventory_path),
            "episode_asset_manifest_count": len(episode_manifests),
            "episode_asset_manifests_sha256": sha256_file(manifest_path),
            "unique_actual_blueprint_count": len(unique_actual_blueprints),
            "unique_actual_blueprints": sorted(unique_actual_blueprints),
            "calibration_sha256": sha256_file(calibration_path),
            "scripted_pose_application": "atomic_batch_before_sensor_tick",
            "scripted_pose_planar_position_tolerance_m": (
                SCRIPTED_POSE_PLANAR_POSITION_TOLERANCE_M
            ),
            "maximum_scripted_pose_planar_position_error_m": (
                max_scripted_pose_planar_position_error_m
            ),
        }
        write_json_atomic(shard_root / "result.json", result)
        return 0 if all(checks.values()) else 2
    finally:
        if sensor is not None:
            try:
                if sensor.is_listening:
                    sensor.stop()
            except Exception:
                pass
        if owned:
            try:
                client.apply_batch_sync(
                    [carla.command.DestroyActor(actor.id) for actor in reversed(owned)],
                    do_tick=bool(world.get_settings().synchronous_mode),
                )
            except Exception as exc:
                print(f"WARNING actor cleanup failed: {exc}", file=sys.stderr)
        try:
            apply_settings(world, original_settings)
            world.set_weather(original_weather)
        except Exception as exc:
            print(f"WARNING world restore failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
