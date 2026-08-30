"""Materialize one CARLA N1 native-traffic behavior trace and compact RGB preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import carla


PLAN_SCHEMA = "dtr-carla-n1-natural-dynamics-plan-v1"
RESULT_STATUS = "DTR_CARLA_N1_NATURAL_DYNAMICS_MATERIALIZED"
FIXED_DELTA_SECONDS = 0.05
PREVIEW_COUNT = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=26000)
    parser.add_argument("--traffic-manager-port", type=int, default=26003)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def write_json_atomic(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing output overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            temporary = Path(stream.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected N1 plan schema")
    if plan.get("evidence_role") != "synthetic_development_input_plan":
        raise ValueError("unexpected N1 evidence role")
    if plan.get("environment", {}).get("carla_version") != "0.9.16":
        raise ValueError("N1 requires CARLA 0.9.16")
    if plan.get("environment", {}).get("map") != "Carla/Maps/Town10HD_Opt":
        raise ValueError("N1 pilot is bound to Town10HD_Opt")
    fingerprint = str(plan.get("plan_fingerprint_sha256", ""))
    unhashed = dict(plan)
    unhashed.pop("plan_fingerprint_sha256", None)
    if fingerprint != canonical_sha256(unhashed):
        raise ValueError("N1 plan fingerprint differs")
    vehicles = list(plan.get("vehicle_intents", []))
    walkers = list(plan.get("walker_intents", []))
    events = list(plan.get("tail_events", []))
    if len(vehicles) != 6 or len(walkers) < 12:
        raise ValueError("N1 pilot actor denominator differs")
    profiles = {str(value["behavior_profile"]) for value in vehicles}
    if profiles != {"cautious", "nominal", "assertive"}:
        raise ValueError("N1 traffic profile coverage differs")
    if {str(value["type"]) for value in events} != {
        "occluded_jaywalk",
        "sudden_brake",
        "reverse_pullout",
        "door_open",
    }:
        raise ValueError("N1 long-tail event coverage differs")


def connect(host: str, port: int) -> carla.Client:
    last_error: Exception | None = None
    for _ in range(30):
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        try:
            client.get_server_version()
            client.set_timeout(120.0)
            return client
        except Exception as error:  # pragma: no cover - depends on server startup
            last_error = error
            time.sleep(2.0)
    raise RuntimeError(f"CARLA server did not become ready: {last_error}")


def copy_world_settings(settings: carla.WorldSettings) -> dict[str, Any]:
    return {
        "synchronous_mode": bool(settings.synchronous_mode),
        "fixed_delta_seconds": settings.fixed_delta_seconds,
        "no_rendering_mode": bool(settings.no_rendering_mode),
        "substepping": bool(settings.substepping),
        "max_substep_delta_time": settings.max_substep_delta_time,
        "max_substeps": settings.max_substeps,
        "deterministic_ragdolls": bool(settings.deterministic_ragdolls),
    }


def apply_world_settings(world: carla.World, values: dict[str, Any]) -> None:
    settings = world.get_settings()
    for name, value in values.items():
        setattr(settings, name, value)
    world.apply_settings(settings)


def vector_dict(value: carla.Vector3D) -> dict[str, float]:
    return {"x": float(value.x), "y": float(value.y), "z": float(value.z)}


def transform_dict(value: carla.Transform) -> dict[str, float]:
    return {
        "x": float(value.location.x),
        "y": float(value.location.y),
        "z": float(value.location.z),
        "pitch": float(value.rotation.pitch),
        "yaw": float(value.rotation.yaw),
        "roll": float(value.rotation.roll),
    }


def speed_mps(actor: carla.Actor) -> float:
    value = actor.get_velocity()
    return math.sqrt(float(value.x) ** 2 + float(value.y) ** 2 + float(value.z) ** 2)


def planar_distance(first: carla.Location, second: carla.Location) -> float:
    return math.hypot(float(first.x) - float(second.x), float(first.y) - float(second.y))


def stable_index(label: str, size: int) -> int:
    if size <= 0:
        raise ValueError("cannot index an empty sequence")
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def await_image(
    image_queue: queue.Queue[carla.Image], expected_frame: int, timeout_s: float = 30.0
) -> carla.Image:
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for RGB frame {expected_frame}")
        try:
            value = image_queue.get(timeout=remaining)
        except queue.Empty as error:
            raise TimeoutError(
                f"timed out waiting for RGB frame {expected_frame}"
            ) from error
        if int(value.frame) < expected_frame:
            continue
        if int(value.frame) > expected_frame:
            raise RuntimeError(
                f"RGB sensor skipped frame {expected_frame} and returned {value.frame}"
            )
        return value


def look_at_transform(
    source: carla.Location, target: carla.Location
) -> carla.Transform:
    dx = float(target.x) - float(source.x)
    dy = float(target.y) - float(source.y)
    dz = float(target.z) - float(source.z)
    return carla.Transform(
        source,
        carla.Rotation(
            pitch=math.degrees(math.atan2(dz, math.hypot(dx, dy))),
            yaw=math.degrees(math.atan2(dy, dx)),
        ),
    )


def blueprint_with_attributes(
    library: carla.BlueprintLibrary,
    blueprint_id: str,
    actor_id: str,
    rng: random.Random,
) -> carla.ActorBlueprint:
    blueprint = library.find(blueprint_id)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", actor_id)
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "false")
    for name in ("color", "driver_id"):
        if blueprint.has_attribute(name):
            values = list(blueprint.get_attribute(name).recommended_values)
            if values:
                blueprint.set_attribute(name, str(values[rng.randrange(len(values))]))
    return blueprint


def select_vehicle_spawn(
    spawn_points: list[carla.Transform],
    focus: carla.Location,
    desired_distance_m: float,
    used: set[int],
) -> list[tuple[int, carla.Transform]]:
    ranked = [
        (
            abs(planar_distance(value.location, focus) - desired_distance_m),
            planar_distance(value.location, focus),
            index,
            value,
        )
        for index, value in enumerate(spawn_points)
        if index not in used and planar_distance(value.location, focus) <= 120.0
    ]
    ranked.sort(key=lambda value: (value[0], value[1], value[2]))
    return [(value[2], value[3]) for value in ranked]


def sample_navigation(
    world: carla.World, focus: carla.Location, count: int = 2048
) -> list[carla.Location]:
    values: dict[tuple[float, float, float], carla.Location] = {}
    for _ in range(count):
        location = world.get_random_location_from_navigation()
        if location is None or planar_distance(location, focus) > 90.0:
            continue
        key = (
            round(float(location.x), 3),
            round(float(location.y), 3),
            round(float(location.z), 3),
        )
        values.setdefault(key, location)
    result = list(values.values())
    result.sort(key=lambda value: (float(value.x), float(value.y), float(value.z)))
    if len(result) < 80:
        raise RuntimeError(f"insufficient local navigation samples: {len(result)}")
    return result


def desired_walker_point(
    intent: dict[str, Any],
    focus: carla.Location,
    forward: tuple[float, float],
    right: tuple[float, float],
) -> carla.Location:
    spawn = intent["spawn_intent"]
    forward_m = float(spawn["forward_offset_m"])
    right_m = float(spawn["right_offset_m"])
    return carla.Location(
        x=float(focus.x) + forward[0] * forward_m + right[0] * right_m,
        y=float(focus.y) + forward[1] * forward_m + right[1] * right_m,
        z=float(focus.z),
    )


def choose_navigation_point(
    navigation: list[carla.Location],
    desired: carla.Location,
    used: set[int],
) -> tuple[int, carla.Location]:
    ranked = sorted(
        (
            planar_distance(value, desired),
            index,
            value,
        )
        for index, value in enumerate(navigation)
        if index not in used
    )
    if not ranked:
        raise RuntimeError("no unused local navigation point remains")
    return ranked[0][1], ranked[0][2]


def configure_vehicle(
    traffic_manager: carla.TrafficManager,
    actor: carla.Vehicle,
    intent: dict[str, Any],
) -> None:
    policy = intent["intent"]
    profile = str(intent["behavior_profile"])
    traffic_manager.set_desired_speed(actor, float(policy["target_speed_mps"]) * 3.6)
    traffic_manager.distance_to_leading_vehicle(
        actor,
        max(
            1.0,
            float(policy["target_speed_mps"])
            * float(policy["following_distance_s"]),
        ),
    )
    traffic_manager.auto_lane_change(actor, True)
    lane_change = 100.0 * float(policy["lane_change_probability"])
    traffic_manager.random_left_lanechange_percentage(actor, lane_change / 2.0)
    traffic_manager.random_right_lanechange_percentage(actor, lane_change / 2.0)
    ignore_lights = {"cautious": 0.0, "nominal": 1.0, "assertive": 4.0}[profile]
    traffic_manager.ignore_lights_percentage(actor, ignore_lights)
    if hasattr(traffic_manager, "ignore_signs_percentage"):
        traffic_manager.ignore_signs_percentage(actor, ignore_lights)
    traffic_manager.update_vehicle_lights(actor, True)


def actor_state(actor: carla.Actor, actor_id: str, kind: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "actor_id": actor_id,
        "carla_actor_id": int(actor.id),
        "type_id": str(actor.type_id),
        "kind": kind,
        "transform": transform_dict(actor.get_transform()),
        "velocity": vector_dict(actor.get_velocity()),
        "acceleration": vector_dict(actor.get_acceleration()),
        "angular_velocity": vector_dict(actor.get_angular_velocity()),
    }
    if kind == "vehicle":
        control = actor.get_control()
        state["control"] = {
            "throttle": float(control.throttle),
            "steer": float(control.steer),
            "brake": float(control.brake),
            "hand_brake": bool(control.hand_brake),
            "reverse": bool(control.reverse),
            "gear": int(control.gear),
        }
    else:
        control = actor.get_control()
        state["control"] = {
            "direction": vector_dict(control.direction),
            "speed": float(control.speed),
            "jump": bool(control.jump),
        }
    return state


def event_observation(actor: carla.Actor) -> dict[str, Any]:
    return {
        "transform": transform_dict(actor.get_transform()),
        "speed_mps": speed_mps(actor),
    }


def main() -> int:
    args = parse_args()
    plan_path = args.plan.resolve(strict=True)
    output_root = args.output_root.resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    for name in (
        "actor_manifest.json",
        "behavior_trace.jsonl",
        "event_receipts.json",
        "result.json",
    ):
        if (output_root / name).exists():
            raise FileExistsError(f"refusing output overwrite: {output_root / name}")

    seed = int(plan["master_seed"])
    random.seed(seed)
    blueprint_rng = random.Random(int(plan["subsystem_seeds"]["traffic"]))
    traffic_seed = int(plan["subsystem_seeds"]["traffic"]) % (2**31 - 1)
    pedestrian_seed = int(plan["subsystem_seeds"]["crowd"]) % (2**31 - 1)
    client = connect(args.host, args.port)
    if client.get_client_version() != "0.9.16":
        raise RuntimeError(f"unexpected CARLA client: {client.get_client_version()}")
    if client.get_server_version() != "0.9.16":
        raise RuntimeError(f"unexpected CARLA server: {client.get_server_version()}")
    world = client.get_world()
    if world.get_map().name != plan["environment"]["map"]:
        raise RuntimeError(f"unexpected CARLA map: {world.get_map().name}")
    shared = [
        actor
        for actor in world.get_actors()
        if actor.type_id.startswith(("vehicle.", "walker.", "sensor.", "controller."))
    ]
    if shared:
        raise RuntimeError("refusing a shared CARLA world")

    original_settings = copy_world_settings(world.get_settings())
    original_weather = world.get_weather()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    owned: list[carla.Actor] = []
    vehicle_actors: dict[str, carla.Vehicle] = {}
    walker_actors: dict[str, carla.Walker] = {}
    walker_controllers: dict[str, carla.WalkerAIController] = {}
    actor_metadata: dict[str, dict[str, Any]] = {}
    sensor: carla.Sensor | None = None
    recorder_started = False
    image_queue: queue.Queue[carla.Image] = queue.Queue()
    cleanup_errors: list[str] = []

    focus_values = plan["focus"]["anchor"]["center_xy_m"]
    focus = carla.Location(x=float(focus_values[0]), y=float(focus_values[1]), z=0.2)
    forward_values = plan["focus"]["anchor"]["forward_xy"]
    right_values = plan["focus"]["anchor"]["right_xy"]
    forward = (float(forward_values[0]), float(forward_values[1]))
    right = (float(right_values[0]), float(right_values[1]))
    destination_points: dict[str, carla.Location] = {}
    event_runtime: dict[str, dict[str, Any]] = {
        str(event["event_id"]): {
            "plan": event,
            "started": False,
            "ended": False,
            "samples": [],
            "api_success": False,
        }
        for event in plan["tail_events"]
    }
    reroute_receipts: list[dict[str, Any]] = []
    fired_reroutes: set[str] = set()
    preview_paths: list[Path] = []
    trace_record_count = 0

    try:
        apply_world_settings(
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
        camera_blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_blueprint.set_attribute("image_size_x", "1280")
        camera_blueprint.set_attribute("image_size_y", "720")
        camera_blueprint.set_attribute("fov", "90")
        camera_blueprint.set_attribute("sensor_tick", f"{FIXED_DELTA_SECONDS:.6f}")
        if camera_blueprint.has_attribute("enable_postprocess_effects"):
            camera_blueprint.set_attribute("enable_postprocess_effects", "true")
        camera_source = carla.Location(
            x=float(focus.x) - 14.0 * forward[0] - 10.0 * right[0],
            y=float(focus.y) - 14.0 * forward[1] - 10.0 * right[1],
            z=8.0,
        )
        camera_target = carla.Location(
            x=float(focus.x) + 4.0 * forward[0] + 4.0 * right[0],
            y=float(focus.y) + 4.0 * forward[1] + 4.0 * right[1],
            z=1.0,
        )
        sensor = world.spawn_actor(
            camera_blueprint, look_at_transform(camera_source, camera_target)
        )
        sensor.listen(image_queue.put)
        owned.append(sensor)

        warmup_deadline = time.monotonic() + 120.0
        warmup_images = 0
        while warmup_images < 5:
            remaining = warmup_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("N1 RGB camera asynchronous warmup timed out")
            try:
                image_queue.get(timeout=min(0.5, remaining))
                warmup_images += 1
            except queue.Empty:
                continue

        apply_world_settings(
            world,
            {
                "synchronous_mode": True,
                "fixed_delta_seconds": FIXED_DELTA_SECONDS,
                "no_rendering_mode": False,
                "substepping": True,
                "max_substep_delta_time": 0.01,
                "max_substeps": 10,
                "deterministic_ragdolls": True,
            },
        )
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(traffic_seed)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.global_percentage_speed_difference(0.0)
        world.set_pedestrians_seed(pedestrian_seed)
        crossing_values = [
            float(value["intent"]["crossing_propensity"])
            for value in plan["walker_intents"]
        ]
        world.set_pedestrians_cross_factor(sum(crossing_values) / len(crossing_values))
        while True:
            try:
                image_queue.get_nowait()
            except queue.Empty:
                break
        aligned = False
        for _ in range(8):
            alignment_frame = world.tick(30.0)
            try:
                await_image(image_queue, alignment_frame, timeout_s=5.0)
                aligned = True
                break
            except TimeoutError:
                continue
        if not aligned:
            raise TimeoutError("N1 RGB camera could not phase-align in synchronous mode")

        spawn_points = list(world.get_map().get_spawn_points())
        used_vehicle_spawns: set[int] = set()
        for ordinal, intent in enumerate(plan["vehicle_intents"]):
            actor_id = str(intent["actor_id"])
            blueprint = blueprint_with_attributes(
                world.get_blueprint_library(),
                str(intent["blueprint_id"]),
                actor_id,
                blueprint_rng,
            )
            desired_distance = 8.0 + abs(
                float(intent["spawn_intent"]["longitudinal_offset_m"])
            )
            actor: carla.Vehicle | None = None
            selected_index = -1
            selected_transform: carla.Transform | None = None
            for candidate_index, transform in select_vehicle_spawn(
                spawn_points, focus, desired_distance, used_vehicle_spawns
            ):
                actor = world.try_spawn_actor(blueprint, transform)
                if actor is not None:
                    selected_index = candidate_index
                    selected_transform = transform
                    used_vehicle_spawns.add(candidate_index)
                    break
            if actor is None or selected_transform is None:
                raise RuntimeError(f"failed to spawn plan vehicle {actor_id}")
            owned.append(actor)
            vehicle_actors[actor_id] = actor
            actor.set_autopilot(True, args.traffic_manager_port)
            configure_vehicle(traffic_manager, actor, intent)
            actor_metadata[actor_id] = {
                "actor_id": actor_id,
                "kind": "vehicle",
                "behavior_profile": str(intent["behavior_profile"]),
                "blueprint_id": str(actor.type_id),
                "carla_actor_id": int(actor.id),
                "spawn_point_index": selected_index,
                "spawn_transform": transform_dict(selected_transform),
                "intent": intent["intent"],
            }

        navigation = sample_navigation(world, focus)
        used_navigation: set[int] = set()
        for intent in plan["walker_intents"]:
            actor_id = str(intent["actor_id"])
            blueprint = blueprint_with_attributes(
                world.get_blueprint_library(),
                str(intent["blueprint_id"]),
                actor_id,
                blueprint_rng,
            )
            desired = desired_walker_point(intent, focus, forward, right)
            ranked = sorted(
                (
                    planar_distance(location, desired),
                    index,
                    location,
                )
                for index, location in enumerate(navigation)
                if index not in used_navigation
            )
            actor: carla.Walker | None = None
            selected_index = -1
            selected_location: carla.Location | None = None
            for _, candidate_index, location in ranked:
                actor = world.try_spawn_actor(blueprint, carla.Transform(location))
                if actor is not None:
                    selected_index = candidate_index
                    selected_location = location
                    used_navigation.add(candidate_index)
                    break
            if actor is None or selected_location is None:
                raise RuntimeError(f"failed to spawn plan walker {actor_id}")
            owned.append(actor)
            walker_actors[actor_id] = actor
            actor_metadata[actor_id] = {
                "actor_id": actor_id,
                "kind": "pedestrian",
                "behavior_profile": str(intent["behavior_profile"]),
                "blueprint_id": str(actor.type_id),
                "carla_actor_id": int(actor.id),
                "navigation_sample_index": selected_index,
                "spawn_transform": transform_dict(actor.get_transform()),
                "intent": intent["intent"],
            }

        settle_frame = world.tick(30.0)
        await_image(image_queue, settle_frame)
        controller_blueprint = world.get_blueprint_library().find(
            "controller.ai.walker"
        )
        for actor_id in sorted(walker_actors):
            controller = world.spawn_actor(
                controller_blueprint,
                carla.Transform(),
                attach_to=walker_actors[actor_id],
            )
            owned.append(controller)
            walker_controllers[actor_id] = controller

        destination_ids = {
            str(value["intent"]["effective_destination_id"])
            for value in plan["walker_intents"]
        }
        destination_ids.update(
            str(value["to_destination_id"])
            for value in plan["group_reroute_interactions"]
        )
        local_navigation = [
            value for value in navigation if 12.0 <= planar_distance(value, focus) <= 75.0
        ]
        for destination_id in sorted(destination_ids):
            destination_points[destination_id] = local_navigation[
                stable_index(destination_id, len(local_navigation))
            ]
        walker_intent_by_id = {
            str(value["actor_id"]): value for value in plan["walker_intents"]
        }
        for actor_id, controller in walker_controllers.items():
            intent = walker_intent_by_id[actor_id]
            controller.start()
            destination_id = str(intent["intent"]["effective_destination_id"])
            controller.go_to_location(destination_points[destination_id])
            controller.set_max_speed(float(intent["intent"]["walking_speed_mps"]))

        for _ in range(4):
            warmup_frame = world.tick(30.0)
            await_image(image_queue, warmup_frame)

        actor_manifest = {
            "schema_version": "dtr-carla-n1-actor-manifest-v1",
            "plan_id": plan["plan_id"],
            "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
            "map": world.get_map().name,
            "fixed_delta_seconds": FIXED_DELTA_SECONDS,
            "traffic_manager_port": int(args.traffic_manager_port),
            "subsystem_seeds": plan["subsystem_seeds"],
            "actors": [actor_metadata[key] for key in sorted(actor_metadata)],
            "destination_points": {
                key: vector_dict(value) for key, value in sorted(destination_points.items())
            },
            "camera": {
                "width": 1280,
                "height": 720,
                "fov_degrees": 90.0,
                "transform": transform_dict(sensor.get_transform()),
            },
        }
        write_json_atomic(output_root / "actor_manifest.json", actor_manifest)

        recorder_path = output_root / "carla_recorder.log"
        client.start_recorder(str(recorder_path), True)
        recorder_started = True
        frame_count = int(round(float(plan["duration_seconds"]) / FIXED_DELTA_SECONDS)) + 1
        preview_samples = {
            int(round(index * (frame_count - 1) / (PREVIEW_COUNT - 1)))
            for index in range(PREVIEW_COUNT)
        }
        preview_root = output_root / "preview"
        preview_root.mkdir(parents=True, exist_ok=False)
        trace_path = output_root / "behavior_trace.jsonl"
        initial_locations = {
            actor_id: actor.get_location()
            for actor_id, actor in {**vehicle_actors, **walker_actors}.items()
        }
        maximum_speed = {actor_id: 0.0 for actor_id in initial_locations}
        maximum_displacement = {actor_id: 0.0 for actor_id in initial_locations}
        close_encounters: set[tuple[str, str]] = set()

        with trace_path.open("x", encoding="utf-8", newline="\n") as trace_stream:
            for sample_index in range(frame_count):
                time_s = sample_index * FIXED_DELTA_SECONDS

                for interaction in plan["group_reroute_interactions"]:
                    interaction_id = str(interaction["interaction_id"])
                    if (
                        interaction_id in fired_reroutes
                        or time_s + 1e-9 < float(interaction["trigger_time_s"])
                    ):
                        continue
                    destination_id = str(interaction["to_destination_id"])
                    applied: list[str] = []
                    for actor_id in interaction["participant_actor_ids"]:
                        actor_id = str(actor_id)
                        controller = walker_controllers.get(actor_id)
                        if controller is None:
                            continue
                        controller.go_to_location(destination_points[destination_id])
                        applied.append(actor_id)
                    fired_reroutes.add(interaction_id)
                    reroute_receipts.append(
                        {
                            "interaction_id": interaction_id,
                            "planned_trigger_time_s": float(
                                interaction["trigger_time_s"]
                            ),
                            "applied_time_s": time_s,
                            "participant_actor_ids": applied,
                            "destination_id": destination_id,
                        }
                    )

                for runtime in event_runtime.values():
                    event = runtime["plan"]
                    trigger = float(event["trigger_time_s"])
                    end = trigger + float(event["duration_s"])
                    actor_id = str(event["primary_actor_id"])
                    actor = vehicle_actors.get(actor_id) or walker_actors.get(actor_id)
                    if actor is None:
                        raise RuntimeError(f"event actor is unavailable: {actor_id}")
                    if not runtime["started"] and time_s + 1e-9 >= trigger:
                        runtime["started"] = True
                        runtime["applied_time_s"] = time_s
                        runtime["before"] = event_observation(actor)
                        event_type = str(event["type"])
                        effect = event["intent_effect"]
                        if event_type == "occluded_jaywalk":
                            direction_sign = (
                                -1.0
                                if str(effect["crossing_direction"]) == "right_to_left"
                                else 1.0
                            )
                            current = actor.get_location()
                            desired = carla.Location(
                                x=float(current.x) + direction_sign * right[0] * 12.0,
                                y=float(current.y) + direction_sign * right[1] * 12.0,
                                z=float(current.z),
                            )
                            crossing_target = min(
                                navigation,
                                key=lambda value: planar_distance(value, desired),
                            )
                            runtime["crossing_target"] = vector_dict(crossing_target)
                            controller = walker_controllers[actor_id]
                            controller.go_to_location(crossing_target)
                            controller.set_max_speed(float(effect["crossing_speed_mps"]))
                        elif event_type == "sudden_brake":
                            actor.set_autopilot(False, args.traffic_manager_port)
                            actor.apply_control(carla.VehicleControl(brake=1.0))
                        elif event_type == "reverse_pullout":
                            actor.set_autopilot(False, args.traffic_manager_port)
                            actor.apply_control(
                                carla.VehicleControl(
                                    throttle=min(
                                        0.65,
                                        0.25
                                        + float(effect["reverse_speed_mps"]) / 5.0,
                                    ),
                                    steer=max(
                                        -1.0,
                                        min(
                                            1.0,
                                            float(effect["steering_degrees"]) / 45.0,
                                        ),
                                    ),
                                    reverse=True,
                                )
                            )
                        elif event_type == "door_open":
                            side = str(effect["door_side"])
                            door = carla.VehicleDoor.FL if side == "left" else carla.VehicleDoor.FR
                            actor.open_door(door)
                        runtime["api_success"] = True

                    if runtime["started"] and not runtime["ended"]:
                        event_type = str(event["type"])
                        effect = event["intent_effect"]
                        if event_type == "sudden_brake":
                            actor.apply_control(carla.VehicleControl(brake=1.0))
                        elif event_type == "reverse_pullout":
                            actor.apply_control(
                                carla.VehicleControl(
                                    throttle=min(
                                        0.65,
                                        0.25
                                        + float(effect["reverse_speed_mps"]) / 5.0,
                                    ),
                                    steer=max(
                                        -1.0,
                                        min(
                                            1.0,
                                            float(effect["steering_degrees"]) / 45.0,
                                        ),
                                    ),
                                    reverse=True,
                                )
                            )
                        runtime["samples"].append(
                            {"time_s": time_s, **event_observation(actor)}
                        )

                    if runtime["started"] and not runtime["ended"] and time_s >= end:
                        event_type = str(event["type"])
                        if event_type == "occluded_jaywalk":
                            controller = walker_controllers[actor_id]
                            destination_id = str(
                                walker_intent_by_id[actor_id]["intent"][
                                    "effective_destination_id"
                                ]
                            )
                            controller.go_to_location(destination_points[destination_id])
                            controller.set_max_speed(
                                float(
                                    walker_intent_by_id[actor_id]["intent"][
                                        "walking_speed_mps"
                                    ]
                                )
                            )
                        elif event_type in {"sudden_brake", "reverse_pullout"}:
                            actor.apply_control(carla.VehicleControl(brake=1.0))
                            actor.set_autopilot(True, args.traffic_manager_port)
                            configure_vehicle(
                                traffic_manager,
                                actor,
                                next(
                                    value
                                    for value in plan["vehicle_intents"]
                                    if value["actor_id"] == actor_id
                                ),
                            )
                        elif event_type == "door_open":
                            side = str(event["intent_effect"]["door_side"])
                            door = carla.VehicleDoor.FL if side == "left" else carla.VehicleDoor.FR
                            actor.close_door(door)
                        runtime["ended"] = True
                        runtime["ended_time_s"] = time_s
                        runtime["after"] = event_observation(actor)

                world_frame = world.tick(30.0)
                image = await_image(image_queue, world_frame)
                if sample_index in preview_samples:
                    preview_path = preview_root / f"sample_{sample_index:04d}.png"
                    image.save_to_disk(str(preview_path))
                    if not preview_path.is_file() or preview_path.stat().st_size <= 0:
                        raise RuntimeError(f"preview image did not materialize: {preview_path}")
                    preview_paths.append(preview_path)

                actors = {**vehicle_actors, **walker_actors}
                states: dict[str, dict[str, Any]] = {}
                for actor_id in sorted(actors):
                    kind = "vehicle" if actor_id in vehicle_actors else "pedestrian"
                    actor = actors[actor_id]
                    states[actor_id] = actor_state(actor, actor_id, kind)
                    maximum_speed[actor_id] = max(maximum_speed[actor_id], speed_mps(actor))
                    maximum_displacement[actor_id] = max(
                        maximum_displacement[actor_id],
                        planar_distance(actor.get_location(), initial_locations[actor_id]),
                    )
                walker_ids = sorted(walker_actors)
                for first_index, first_id in enumerate(walker_ids):
                    for second_id in walker_ids[first_index + 1 :]:
                        if (
                            planar_distance(
                                walker_actors[first_id].get_location(),
                                walker_actors[second_id].get_location(),
                            )
                            <= 2.0
                        ):
                            close_encounters.add((first_id, second_id))
                record = {
                    "schema_version": "dtr-carla-n1-behavior-trace-frame-v1",
                    "sample_index": sample_index,
                    "time_s": round(time_s, 9),
                    "world_frame": int(world_frame),
                    "actors": states,
                }
                trace_stream.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                    + "\n"
                )
                trace_record_count += 1

        client.stop_recorder()
        recorder_started = False
        if not recorder_path.is_file() or recorder_path.stat().st_size <= 0:
            raise RuntimeError("CARLA recorder log did not materialize")

        event_receipts: list[dict[str, Any]] = []
        motion_effects: dict[str, bool] = {}
        for event_id in sorted(event_runtime):
            runtime = event_runtime[event_id]
            event = runtime["plan"]
            samples = runtime["samples"]
            event_type = str(event["type"])
            before = runtime.get("before")
            after = runtime.get("after")
            displacement = 0.0
            speed_drop = 0.0
            if before is not None and after is not None:
                first = carla.Location(
                    x=float(before["transform"]["x"]),
                    y=float(before["transform"]["y"]),
                    z=float(before["transform"]["z"]),
                )
                second = carla.Location(
                    x=float(after["transform"]["x"]),
                    y=float(after["transform"]["y"]),
                    z=float(after["transform"]["z"]),
                )
                displacement = planar_distance(first, second)
                minimum_speed = min(
                    [float(value["speed_mps"]) for value in samples]
                    or [float(after["speed_mps"])]
                )
                speed_drop = max(0.0, float(before["speed_mps"]) - minimum_speed)
            if event_type == "occluded_jaywalk":
                motion_effects[event_type] = displacement >= 0.5
            elif event_type == "sudden_brake":
                motion_effects[event_type] = speed_drop >= 0.25
            elif event_type == "reverse_pullout":
                motion_effects[event_type] = displacement >= 0.25
            else:
                motion_effects[event_type] = bool(runtime["api_success"])
            event_receipts.append(
                {
                    "event_id": event_id,
                    "type": event_type,
                    "primary_actor_id": str(event["primary_actor_id"]),
                    "planned_trigger_time_s": float(event["trigger_time_s"]),
                    "applied_time_s": runtime.get("applied_time_s"),
                    "ended_time_s": runtime.get("ended_time_s"),
                    "api_success": bool(runtime["api_success"]),
                    "before": before,
                    "after": after,
                    "observed_displacement_m": displacement,
                    "observed_speed_drop_mps": speed_drop,
                    "observed_effect": bool(motion_effects[event_type]),
                }
            )
        event_document = {
            "schema_version": "dtr-carla-n1-event-receipts-v1",
            "plan_id": plan["plan_id"],
            "tail_events": event_receipts,
            "group_reroutes": reroute_receipts,
        }
        write_json_atomic(output_root / "event_receipts.json", event_document)

        moving_vehicles = [
            actor_id
            for actor_id in sorted(vehicle_actors)
            if maximum_speed[actor_id] >= 0.5 and maximum_displacement[actor_id] >= 1.0
        ]
        moving_walkers = [
            actor_id
            for actor_id in sorted(walker_actors)
            if maximum_displacement[actor_id] >= 0.5
        ]
        profiles = sorted(
            {
                str(actor_metadata[actor_id]["behavior_profile"])
                for actor_id in vehicle_actors
            }
        )
        checks = {
            "plan_fingerprint_verified": True,
            "town10hd_map_verified": world.get_map().name
            == "Carla/Maps/Town10HD_Opt",
            "actor_denominator_exact": len(vehicle_actors)
            == len(plan["vehicle_intents"])
            and len(walker_actors) == len(plan["walker_intents"]),
            "three_traffic_profiles_realized": profiles
            == ["assertive", "cautious", "nominal"],
            "native_vehicle_motion_observed": len(moving_vehicles) >= 3,
            "native_walker_motion_observed": len(moving_walkers) >= 6,
            "group_reroutes_executed": len(reroute_receipts)
            == len(plan["group_reroute_interactions"]),
            "crowd_close_encounters_observed": len(close_encounters) > 0,
            "all_tail_event_apis_applied": all(
                bool(value["api_success"]) for value in event_receipts
            ),
            "tail_event_effects_observed": all(motion_effects.values()),
            "trace_complete": trace_record_count == frame_count,
            "preview_complete": len(preview_paths) == PREVIEW_COUNT,
            "carla_recorder_materialized": recorder_path.is_file()
            and recorder_path.stat().st_size > 0,
        }
        failed = [name for name, value in checks.items() if not value]
        if failed:
            raise RuntimeError(f"N1 materialization checks failed: {failed}")
        result = {
            "schema_version": "dtr-carla-n1-natural-dynamics-result-v1",
            "status": RESULT_STATUS,
            "plan_id": plan["plan_id"],
            "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
            "plan_file_sha256": sha256_file(plan_path),
            "map": world.get_map().name,
            "fixed_delta_seconds": FIXED_DELTA_SECONDS,
            "duration_seconds": float(plan["duration_seconds"]),
            "frame_count": frame_count,
            "vehicle_count": len(vehicle_actors),
            "walker_count": len(walker_actors),
            "traffic_profiles": profiles,
            "moving_vehicle_count": len(moving_vehicles),
            "moving_vehicle_actor_ids": moving_vehicles,
            "moving_walker_count": len(moving_walkers),
            "moving_walker_actor_ids": moving_walkers,
            "walker_close_encounter_pair_count": len(close_encounters),
            "walker_close_encounter_pairs": [list(value) for value in sorted(close_encounters)],
            "group_reroute_count": len(reroute_receipts),
            "tail_event_count": len(event_receipts),
            "tail_event_effects": motion_effects,
            "maximum_speed_mps": maximum_speed,
            "maximum_displacement_m": maximum_displacement,
            "trace": {
                "path": str(trace_path),
                "records": trace_record_count,
                "sha256": sha256_file(trace_path),
            },
            "actor_manifest": {
                "path": str(output_root / "actor_manifest.json"),
                "sha256": sha256_file(output_root / "actor_manifest.json"),
            },
            "event_receipts": {
                "path": str(output_root / "event_receipts.json"),
                "sha256": sha256_file(output_root / "event_receipts.json"),
            },
            "carla_recorder": {
                "path": str(recorder_path),
                "bytes": recorder_path.stat().st_size,
                "sha256": sha256_file(recorder_path),
            },
            "preview": {
                "count": len(preview_paths),
                "files": [
                    {
                        "path": str(value),
                        "bytes": value.stat().st_size,
                        "sha256": sha256_file(value),
                    }
                    for value in preview_paths
                ],
            },
            "checks": checks,
            "claim_boundary": [
                *plan["claim_boundary"],
                "This result records one native-policy CARLA realization and is not cross-cold-start pixel determinism.",
                "The frozen behavior trace is evaluator-side Development evidence until a separate truth-blind multi-modal replay is completed.",
            ],
        }
        write_json_atomic(output_root / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        if recorder_started:
            try:
                client.stop_recorder()
            except Exception as error:  # pragma: no cover - cleanup path
                cleanup_errors.append(f"recorder: {error}")
        for controller in walker_controllers.values():
            try:
                controller.stop()
            except Exception as error:  # pragma: no cover - cleanup path
                cleanup_errors.append(f"controller {controller.id}: {error}")
        if sensor is not None:
            try:
                sensor.stop()
            except Exception as error:  # pragma: no cover - cleanup path
                cleanup_errors.append(f"sensor {sensor.id}: {error}")
        actor_ids = [int(value.id) for value in reversed(owned) if value.is_alive]
        if actor_ids:
            try:
                client.apply_batch_sync(
                    [carla.command.DestroyActor(value) for value in actor_ids], False
                )
            except Exception as error:  # pragma: no cover - cleanup path
                cleanup_errors.append(f"destroy: {error}")
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception as error:  # pragma: no cover - cleanup path
            cleanup_errors.append(f"traffic manager: {error}")
        try:
            world.set_weather(original_weather)
            apply_world_settings(world, original_settings)
        except Exception as error:  # pragma: no cover - cleanup path
            cleanup_errors.append(f"world restore: {error}")
        if cleanup_errors:
            print(
                json.dumps(
                    {"cleanup_warnings": cleanup_errors},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    raise SystemExit(main())
