"""Capture one fresh-server DTR-CARLA-C1 complex-scene sensor shard.

The host profile admits exactly one long-lived GPU camera per CARLA process.
This entrypoint therefore materializes one of instance, wearable RGB, wearable
depth, or witness RGB. A later truth-separating join compares actual actor
state across all four fresh-server replays before it can declare completion.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import carla
import numpy as np

from dtr_carla_c1_complex import (
    build_plan_receipt,
    contact_union,
    plan_authority,
    sha256_file,
    trajectory_position,
    trajectory_prefix_equal,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sensor", choices=tuple(SENSOR_TYPES), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    return parser.parse_args()


def connect(host: str, port: int, attempts: int = 30) -> carla.Client:
    last_error: Exception | None = None
    for _ in range(attempts):
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        try:
            client.get_server_version()
            client.set_timeout(120.0)
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


def polygon_from_bbox(actor: carla.Actor, transform: carla.Transform) -> list[list[float]]:
    unique: dict[tuple[float, float], tuple[float, float]] = {}
    for vertex in actor.bounding_box.get_world_vertices(transform):
        key = (round(float(vertex.x), 6), round(float(vertex.y), 6))
        unique[key] = (float(vertex.x), float(vertex.y))
    points = sorted(unique.values())
    if len(points) < 3:
        raise RuntimeError(
            f"{actor.type_id} must expose at least three distinct XY bbox points, got {len(points)}"
        )

    # Walkers can carry a pitched/rolled local bbox, whose eight 3-D corners
    # project to more than four distinct XY points.  Score against the convex
    # footprint of every projected corner instead of assuming a flat OBB.
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
        raise RuntimeError(f"{actor.type_id} produced a degenerate XY bbox hull")
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


def normalized_anchor(protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor = protocol["environment"]["anchor"]
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
    return float(waypoint.transform.location.z) + offset_m


def trajectory_for_asset(
    asset: dict[str, Any], scenario: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any] | None:
    reference = asset.get("trajectory_ref")
    if reference == "wearer":
        name = str(scenario["wearer_trajectory"])
    elif reference:
        name = str(scenario["asset_trajectories"][str(reference)])
    else:
        return None
    return protocol["trajectory_library"][name]


def asset_pose(
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
        velocity_world = local_velocity_to_world(velocity_local, forward, right)
        yaw = (
            math.degrees(math.atan2(float(velocity_world[1]), float(velocity_world[0])))
            if float(np.linalg.norm(velocity_world)) > 1e-9
            else base_yaw
        )
    xy = local_to_world(local, center, forward, right)
    velocity_world = local_velocity_to_world(velocity_local, forward, right)
    z = (
        float(protocol["environment"]["walker_origin_z_m"])
        if str(asset["kind"]) == "walker"
        else road_surface_z(world_map, xy, float(asset["surface_offset_m"]))
    )
    return {
        "local": local,
        "velocity_world": velocity_world,
        "transform": carla.Transform(
            carla.Location(x=float(xy[0]), y=float(xy[1]), z=z),
            carla.Rotation(yaw=yaw),
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
    blueprint: carla.ActorBlueprint, role_name: str
) -> dict[str, str]:
    applied: dict[str, str] = {}
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)
        applied["role_name"] = role_name
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "false")
        applied["is_invincible"] = "false"
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
    index: int,
) -> tuple[carla.Actor, dict[str, Any]]:
    blueprint, fallback_index = resolve_blueprint(
        world.get_blueprint_library(), list(asset["blueprint_candidates"])
    )
    attributes = deterministic_blueprint_attributes(
        blueprint, f"dtr_c1_{asset['asset_key']}"
    )
    actor = world.try_spawn_actor(blueprint, initial_pose["transform"])
    spawn_strategy = "scene_pose"
    if actor is None:
        source = initial_pose["transform"]
        hidden = carla.Transform(
            carla.Location(
                x=float(source.location.x),
                y=float(source.location.y),
                z=float(source.location.z) + 20.0 + index,
            ),
            source.rotation,
        )
        actor = world.try_spawn_actor(blueprint, hidden)
        spawn_strategy = "elevated_then_teleported"
    if actor is None:
        raise RuntimeError(
            f"failed to spawn {asset['asset_key']} from {asset['blueprint_candidates']}"
        )
    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass
    if str(asset["kind"]) == "walker":
        try:
            actor.apply_control(carla.WalkerControl(speed=0.0))
        except Exception:
            pass
    actor.set_transform(initial_pose["transform"])
    return actor, {
        "asset_key": str(asset["asset_key"]),
        "track_id": str(asset["track_id"]),
        "scenario_role": str(asset["role"]),
        "kind": str(asset["kind"]),
        "candidate_blueprints": list(asset["blueprint_candidates"]),
        "actual_blueprint": str(actor.type_id),
        "fallback_index": int(fallback_index),
        "spawn_strategy": spawn_strategy,
        "attributes": attributes,
        "carla_actor_id": int(actor.id),
    }


def apply_scene(
    actors: dict[str, carla.Actor],
    scenario: dict[str, Any],
    protocol: dict[str, Any],
    time_s: float,
    *,
    world_map: carla.Map,
    center: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
) -> dict[str, dict[str, Any]]:
    poses: dict[str, dict[str, Any]] = {}
    for asset in protocol["asset_cluster"]:
        key = str(asset["asset_key"])
        pose = asset_pose(
            asset,
            scenario,
            protocol,
            time_s,
            world_map=world_map,
            center=center,
            forward=forward,
            right=right,
        )
        actors[key].set_transform(pose["transform"])
        if str(asset["kind"]) == "walker":
            try:
                actors[key].apply_control(carla.WalkerControl(speed=0.0))
            except Exception:
                pass
        poses[key] = pose
    return poses


def instance_metrics(image: carla.Image, actor_id: int) -> dict[str, Any]:
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (image.height, image.width, 4)
    )
    instance_ids = bgra[:, :, 1].astype(np.uint32) | (
        bgra[:, :, 0].astype(np.uint32) << 8
    )
    ys, xs = np.nonzero(instance_ids == int(actor_id))
    pixels = int(xs.size)
    return {
        "pixels": pixels,
        "visible": pixels >= 16,
        "bbox_uv": (
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if pixels
            else None
        ),
    }


def look_at_transform(
    protocol: dict[str, Any],
    *,
    center: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
) -> carla.Transform:
    witness = protocol["capture"]["witness"]
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
            x=float(source_xy[0]),
            y=float(source_xy[1]),
            z=float(witness["z_m"]),
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
        "width": int(image.width),
        "height": int(image.height),
    }


def write_episode_payload_inventory(
    episode_root: Path, values: list[dict[str, Any]]
) -> Path:
    path = episode_root / "payload_inventory.json"
    write_json_atomic(path, values)
    return path


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
    environment = protocol["environment"]
    fixed_delta = float(environment["sample_seconds"])
    duration_s = float(environment["duration_seconds"])
    frame_count = int(round(duration_s / fixed_delta)) + 1
    center, forward, right = normalized_anchor(protocol)
    receipts = {
        str(value["episode_id"]): build_plan_receipt(value.get("issued_plan"))
        for value in protocol["scenarios"]
    }

    client = connect(args.host, args.port)
    if client.get_client_version() != environment["carla_version"]:
        raise RuntimeError(f"unexpected CARLA client version: {client.get_client_version()}")
    if client.get_server_version() != environment["carla_version"]:
        raise RuntimeError(f"unexpected CARLA server version: {client.get_server_version()}")
    world = client.get_world()
    if world.get_map().name != environment["map"]:
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
    actors: dict[str, carla.Actor] = {}
    owned: list[carla.Actor] = []
    sensor: carla.Sensor | None = None
    sensor_queue: queue.Queue[carla.SensorData] = queue.Queue()
    asset_manifest: list[dict[str, Any]] = []
    dynamic_spawn_history: list[dict[str, Any]] = []
    payload_inventory: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    sensor_alignment_ok = True

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
        world.set_weather(carla.WeatherParameters.ClearNoon)
        first = protocol["scenarios"][0]
        for index, asset in enumerate(protocol["asset_cluster"]):
            pose = asset_pose(
                asset,
                first,
                protocol,
                0.0,
                world_map=world.get_map(),
                center=center,
                forward=forward,
                right=right,
            )
            actor, manifest = spawn_asset(world, asset, pose, index)
            actors[str(asset["asset_key"])] = actor
            owned.append(actor)
            asset_manifest.append(manifest)
            if asset.get("trajectory_ref") and asset["asset_key"] != "wearer":
                dynamic_spawn_history.append({**manifest, "episode_id": "ep_01"})

        capture = protocol["capture"]
        resolution_key = (
            "witness_resolution" if sensor_name == "witness" else "wearable_resolution"
        )
        width, height = map(int, capture[resolution_key])
        blueprint = sensor_blueprint(
            world,
            SENSOR_TYPES[sensor_name],
            width,
            height,
            float(capture["fov_degrees"]),
            fixed_delta,
        )
        if sensor_name == "witness":
            sensor = world.spawn_actor(
                blueprint,
                look_at_transform(protocol, center=center, forward=forward, right=right),
            )
        else:
            relative = capture["wearable_relative_transform"]
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
            sensor = world.spawn_actor(
                blueprint,
                relative_transform,
                attach_to=actors["wearer"],
                attachment_type=carla.AttachmentType.Rigid,
            )
        sensor.listen(sensor_queue.put)
        owned.append(sensor)

        warmup = 0
        deadline = time.monotonic() + 90.0
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
        phase_aligned = False
        for _ in range(8):
            frame = world.tick(30.0)
            try:
                await_frame(sensor_queue, frame, sensor_name, timeout_seconds=3.0)
                phase_aligned = True
                break
            except TimeoutError:
                continue
        if not phase_aligned:
            raise TimeoutError(f"could not phase-align the {sensor_name} camera")

        dynamic_assets = [
            value
            for value in protocol["asset_cluster"]
            if value.get("trajectory_ref") and value["asset_key"] != "wearer"
        ]
        for scenario_index, scenario in enumerate(protocol["scenarios"]):
            episode_id = str(scenario["episode_id"])
            if scenario_index > 0:
                old_ids = {int(actors[str(value["asset_key"])].id) for value in dynamic_assets}
                client.apply_batch_sync(
                    [carla.command.DestroyActor(value) for value in sorted(old_ids)],
                    do_tick=True,
                )
                owned = [value for value in owned if int(value.id) not in old_ids]
                while True:
                    try:
                        sensor_queue.get_nowait()
                    except queue.Empty:
                        break
                for asset_index, asset in enumerate(dynamic_assets):
                    pose = asset_pose(
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
                        world,
                        asset,
                        pose,
                        len(protocol["asset_cluster"]) + scenario_index * 10 + asset_index,
                    )
                    actors[str(asset["asset_key"])] = actor
                    owned.append(actor)
                    dynamic_spawn_history.append({**manifest, "episode_id": episode_id})
            episode_root = shard_root / "episodes" / episode_id
            episode_root.mkdir(parents=True)
            payload_root = episode_root / "payload"
            payload_root.mkdir()
            records: list[dict[str, Any]] = []
            receipt = receipts[episode_id]
            apply_scene(
                actors,
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
                    actors,
                    scenario,
                    protocol,
                    0.0,
                    world_map=world.get_map(),
                    center=center,
                    forward=forward,
                    right=right,
                )

            for sample_index in range(frame_count):
                time_s = sample_index * fixed_delta
                poses = apply_scene(
                    actors,
                    scenario,
                    protocol,
                    time_s,
                    world_map=world.get_map(),
                    center=center,
                    forward=forward,
                    right=right,
                )
                world_frame = world.tick(30.0)
                image = await_frame(sensor_queue, world_frame, sensor_name)
                sensor_alignment_ok = sensor_alignment_ok and int(image.frame) == int(world_frame)
                if len(image.raw_data) <= 0:
                    raise RuntimeError(f"empty {sensor_name} payload at {episode_id}/{sample_index}")
                snapshot = world.get_snapshot()
                actor_states: dict[str, dict[str, Any]] = {}
                collision_polygons: dict[str, list[list[float]]] = {}
                for asset in protocol["asset_cluster"]:
                    key = str(asset["asset_key"])
                    actor = actors[key]
                    actor_snapshot = snapshot.find(actor.id)
                    if actor_snapshot is None:
                        raise RuntimeError(f"snapshot omitted task-owned actor {key}")
                    actual_transform = actor_snapshot.get_transform()
                    pose = poses[key]
                    velocity = pose["velocity_world"]
                    actor_states[key] = {
                        "track_id": str(asset["track_id"]),
                        "asset_key": key,
                        "scenario_role": str(asset["role"]),
                        "kind": str(asset["kind"]),
                        "actual_blueprint": str(actor.type_id),
                        "carla_actor_id": int(actor.id),
                        "transform": transform_dict(actual_transform),
                        "local_position": {
                            "forward_m": float(pose["local"][0]),
                            "right_m": float(pose["local"][1]),
                        },
                        "command_velocity": vector_dict(velocity[0], velocity[1]),
                        "bounding_box": bbox_dict(actor, actual_transform),
                    }
                    if bool(asset["collision_relevant"]) and key != "wearer":
                        collision_polygons[key] = polygon_from_bbox(actor, actual_transform)

                wearer_local = poses["wearer"]["local"]
                wearer_xy = local_to_world(wearer_local, center, forward, right)
                current_contact, minimum_distance, responsible = contact_union(
                    (float(wearer_xy[0]), float(wearer_xy[1])),
                    collision_polygons,
                    wearer_radius_m=float(protocol["route_contract"]["wearer_body_radius_m"]),
                )
                file_path = payload_root / f"{sample_index:06d}.png"
                payload = save_image(image, file_path)
                payload.update(
                    {
                        "sensor": sensor_name,
                        "episode_id": episode_id,
                        "sample_index": sample_index,
                    }
                )
                payload_inventory.append(payload)
                visibility = (
                    {
                        key: instance_metrics(image, actor.id)
                        for key, actor in actors.items()
                        if key != "wearer"
                    }
                    if sensor_name == "instance"
                    else None
                )
                records.append(
                    {
                        "schema_version": "dtr-c1-raw-shard-frame-v1",
                        "sensor": sensor_name,
                        "episode_id": episode_id,
                        "sample_index": sample_index,
                        "time_s": time_s,
                        "world_frame": int(world_frame),
                        "sensor_path": str(file_path.relative_to(shard_root)).replace("\\", "/"),
                        "sensor_payload_bytes": int(payload["bytes"]),
                        "sensor_payload_sha256": str(payload["sha256"]),
                        "camera_transform": transform_dict(image.transform),
                        "plan_receipt_sha256": receipt["receipt_sha256"] if receipt else None,
                        "plan_authority": plan_authority(
                            receipt,
                            session_id=str(scenario["plan_session_id"]),
                            time_s=time_s,
                        ),
                        "actors": actor_states,
                        "instance_visibility": visibility,
                        "truth": {
                            "scenario_role": str(scenario["scenario_role"]),
                            "twin_role": str(scenario["twin_role"]),
                            "expected_contact": bool(scenario["expected_contact"]),
                            "current_contact": bool(current_contact),
                            "minimum_distance_m": float(minimum_distance),
                            "responsible_asset": responsible,
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
                record["truth"]["future_contact_within_horizon"] = bool(future_contacts)
                record["truth"]["realized_time_to_contact_seconds"] = (
                    float(future_contacts[0]["time_s"]) - current_time
                    if future_contacts
                    else None
                )
            frames_path = episode_root / "frames.jsonl"
            write_jsonl(frames_path, records)
            contact_times = [
                float(value["time_s"])
                for value in records
                if bool(value["truth"]["current_contact"])
            ]
            primary_visible_frames = (
                sum(
                    bool(value["instance_visibility"]["target_primary"]["visible"])
                    for value in records
                )
                if sensor_name == "instance"
                else None
            )
            summary = {
                "episode_id": episode_id,
                "sensor": sensor_name,
                "frames": len(records),
                "expected_contact": bool(scenario["expected_contact"]),
                "observed_contact": bool(contact_times),
                "first_contact_time_s": contact_times[0] if contact_times else None,
                "last_contact_time_s": contact_times[-1] if contact_times else None,
                "minimum_distance_m": min(
                    float(value["truth"]["minimum_distance_m"]) for value in records
                ),
                "primary_visible_frames": primary_visible_frames,
                "plan_authority_at_start": plan_authority(
                    receipt,
                    session_id=str(scenario["plan_session_id"]),
                    time_s=0.0,
                ),
                "plan_receipt_sha256": receipt["receipt_sha256"] if receipt else None,
            }
            summary_path = episode_root / "summary.json"
            write_json_atomic(summary_path, summary)
            episode_payloads = [
                value for value in payload_inventory if value["episode_id"] == episode_id
            ]
            episode_inventory_path = write_episode_payload_inventory(
                episode_root, episode_payloads
            )
            write_json_atomic(
                episode_root / "manifest.json",
                {
                    "schema_version": "dtr-c1-raw-shard-episode-manifest-v1",
                    "episode_id": episode_id,
                    "sensor": sensor_name,
                    "frames": len(records),
                    "frames_sha256": sha256_file(frames_path),
                    "summary_sha256": sha256_file(summary_path),
                    "payload_count": len(episode_payloads),
                    "payload_inventory_sha256": sha256_file(episode_inventory_path),
                },
            )
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)

        role_counts = Counter(value["scenario_role"] for value in asset_manifest)
        required_counts = Counter(protocol["admission"]["required_role_counts"])
        twin_checks: dict[str, bool] = {}
        for pair in protocol["twin_contracts"]:
            first_scenario = next(
                value for value in protocol["scenarios"] if value["episode_id"] == pair["a"]
            )
            second_scenario = next(
                value for value in protocol["scenarios"] if value["episode_id"] == pair["b"]
            )
            identical_before_s = float(pair["identical_before_s"])
            wearer_equal = trajectory_prefix_equal(
                protocol["trajectory_library"][first_scenario["wearer_trajectory"]],
                protocol["trajectory_library"][second_scenario["wearer_trajectory"]],
                end_s=identical_before_s,
                sample_s=fixed_delta,
            )
            assets_equal = all(
                trajectory_prefix_equal(
                    protocol["trajectory_library"][first_scenario["asset_trajectories"][key]],
                    protocol["trajectory_library"][second_scenario["asset_trajectories"][key]],
                    end_s=identical_before_s,
                    sample_s=fixed_delta,
                )
                for key in first_scenario["asset_trajectories"]
            )
            twin_checks[str(pair["family"])] = wearer_equal and assets_equal

        occluder_asset = next(
            value for value in protocol["asset_cluster"] if value["role"] == "physical_occluder"
        )
        occluder_extent = actors[str(occluder_asset["asset_key"])].bounding_box.extent
        checks = {
            "all_eight_episodes_captured": len(summaries) == 8,
            "all_frame_counts_match": all(value["frames"] == frame_count for value in summaries),
            "all_expected_contact_relations_match": all(
                value["expected_contact"] == value["observed_contact"] for value in summaries
            ),
            "programmatic_asset_roles_admitted": role_counts == required_counts,
            "all_actual_blueprints_recorded": all(
                bool(value["actual_blueprint"]) for value in asset_manifest
            ),
            "physical_occluder_has_nonzero_bbox": min(
                float(occluder_extent.x),
                float(occluder_extent.y),
                float(occluder_extent.z),
            ) > 0.0,
            "all_raw_payloads_materialized": len(payload_inventory) == 8 * frame_count,
            "sensor_world_frames_aligned": sensor_alignment_ok,
            "all_twin_prefix_contracts_hold": all(twin_checks.values()),
            "same_plan_receipts_are_byte_identical": (
                receipts["ep_01"] == receipts["ep_02"]
                and receipts["ep_03"] == receipts["ep_04"]
                and receipts["ep_07"] == receipts["ep_08"]
            ),
            "expired_and_missing_plan_cases_present": (
                next(value for value in summaries if value["episode_id"] == "ep_05")[
                    "plan_authority_at_start"
                ]
                == "EXPIRED"
                and next(value for value in summaries if value["episode_id"] == "ep_06")[
                    "plan_authority_at_start"
                ]
                == "NO_PLAN"
            ),
            "dynamic_actors_respawned_for_each_episode": len(dynamic_spawn_history)
            == len(dynamic_assets) * len(protocol["scenarios"]),
        }
        if sensor_name == "instance":
            straight_episode_ids = {
                str(value["episode_id"])
                for value in protocol["scenarios"]
                if value["wearer_trajectory"] == "wearer_straight"
            }
            checks["primary_visibility_varies_in_straight_route_episodes"] = all(
                0 < int(value["primary_visible_frames"]) < frame_count
                for value in summaries
                if value["episode_id"] in straight_episode_ids
            )
        asset_manifest_path = shard_root / "asset_manifest.json"
        payload_inventory_path = shard_root / "payload_inventory.json"
        dynamic_spawn_history_path = shard_root / "dynamic_spawn_history.json"
        write_json_atomic(asset_manifest_path, asset_manifest)
        write_json_atomic(payload_inventory_path, payload_inventory)
        write_json_atomic(dynamic_spawn_history_path, dynamic_spawn_history)
        result = {
            "schema_version": "dtr-carla-c1-raw-shard-result-v1",
            "experiment_id": protocol["experiment_id"],
            "status": (
                "DTR_CARLA_C1_RAW_SHARD_CAPTURE_COMPLETE"
                if all(checks.values())
                else "DTR_CARLA_C1_RAW_SHARD_CAPTURE_NOT_EVALUABLE"
            ),
            "sensor": sensor_name,
            "protocol_sha256": sha256_file(protocol_path),
            "capture_script_sha256": sha256_file(Path(__file__).resolve()),
            "helper_module_sha256": sha256_file(
                Path(__file__).with_name("dtr_carla_c1_complex.py")
            ),
            "carla_client_version": client.get_client_version(),
            "carla_server_version": client.get_server_version(),
            "map": world.get_map().name,
            "checks": checks,
            "twin_checks": twin_checks,
            "episodes": summaries,
            "asset_count": len(asset_manifest),
            "asset_manifest_sha256": sha256_file(asset_manifest_path),
            "payload_count": len(payload_inventory),
            "payload_inventory_sha256": sha256_file(payload_inventory_path),
            "dynamic_spawn_history_count": len(dynamic_spawn_history),
            "dynamic_spawn_history_sha256": sha256_file(dynamic_spawn_history_path),
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
