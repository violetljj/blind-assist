"""Replay one frozen N1 behavior trace into a joined C2-compatible four-modal pack."""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import carla
import numpy as np

from capture_dtr_carla_c2_rich_scene import (
    SENSOR_TYPES,
    apply_settings,
    await_frame,
    batched_instance_metrics,
    bbox_dict,
    bbox_nonzero,
    connect,
    copy_settings,
    polygon_from_bbox,
    road_surface_z,
    save_image,
    sensor_blueprint,
    transform_dict,
    transform_residual,
    vector_dict,
    weather_parameter,
)
from dtr_carla_c2_rich_scene import (
    camera_intrinsics,
    contact_union,
    sha256_file,
    validate_model_record,
    write_json_atomic,
    write_jsonl,
)
from dtr_carla_n2_frozen_trace_replay import (
    EXPERIMENT_ID,
    SENSOR_ORDER,
    active_tail_event_ids,
    actor_roster,
    build_alignment_receipt,
    door_event,
    read_json,
    safe_relative,
    scan_model_tree,
    seal_tree,
    verify_source_bundle,
)


POSITION_TOLERANCE_M = 1e-4
ANGLE_TOLERANCE_DEGREES = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=26100)
    return parser.parse_args()


def carla_transform(value: dict[str, Any]) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=float(value["x"]), y=float(value["y"]), z=float(value["z"])
        ),
        carla.Rotation(
            pitch=float(value["pitch"]),
            yaw=float(value["yaw"]),
            roll=float(value["roll"]),
        ),
    )


def deterministic_attributes(
    blueprint: carla.ActorBlueprint, role_name: str
) -> dict[str, str]:
    applied: dict[str, str] = {}
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)
        applied["role_name"] = role_name
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "true")
        applied["is_invincible"] = "true"
    for name in ("color", "driver_id"):
        if blueprint.has_attribute(name):
            values = list(blueprint.get_attribute(name).recommended_values)
            if values:
                blueprint.set_attribute(name, str(values[0]))
                applied[name] = str(values[0])
    return applied


def spawn_physics_off_actor(
    world: carla.World,
    *,
    actor_id: str,
    type_id: str,
    kind: str,
    initial_transform: carla.Transform,
    fallback_height_index: int,
) -> tuple[carla.Actor, dict[str, Any]]:
    blueprint = world.get_blueprint_library().find(type_id)
    attributes = deterministic_attributes(blueprint, f"dtr_n2_{actor_id}")
    actor = world.try_spawn_actor(blueprint, initial_transform)
    spawn_strategy = "trace_initial_transform"
    if actor is None:
        elevated = carla.Transform(
            carla.Location(
                x=float(initial_transform.location.x),
                y=float(initial_transform.location.y),
                z=float(initial_transform.location.z) + 40.0 + fallback_height_index,
            ),
            initial_transform.rotation,
        )
        actor = world.try_spawn_actor(blueprint, elevated)
        spawn_strategy = "elevated_then_teleported"
    if actor is None:
        raise RuntimeError(f"failed to spawn exact replay blueprint {type_id}/{actor_id}")
    try:
        actor.set_simulate_physics(False)
        actor.set_collisions(False)
        if kind == "pedestrian":
            actor.apply_control(carla.WalkerControl(speed=0.0))
        elif kind == "vehicle":
            actor.set_autopilot(False)
            actor.apply_control(carla.VehicleControl())
        actor.set_transform(initial_transform)
    except Exception:
        actor.destroy()
        raise
    extent = actor.bounding_box.extent
    return actor, {
        "actor_id": actor_id,
        "kind": kind,
        "requested_blueprint": type_id,
        "actual_blueprint": str(actor.type_id),
        "carla_actor_id": int(actor.id),
        "spawn_strategy": spawn_strategy,
        "simulate_physics_disabled": True,
        "collisions_disabled": True,
        "attributes": attributes,
        "bbox_extent": vector_dict(extent.x, extent.y, extent.z),
        "bbox_nonzero": bbox_nonzero(actor),
    }


def relative_sensor_transform(value: dict[str, Any]) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=float(value["x_m"]),
            y=float(value["y_m"]),
            z=float(value["z_m"]),
        ),
        carla.Rotation(
            pitch=float(value["pitch_degrees"]),
            yaw=float(value["yaw_degrees"]),
            roll=float(value["roll_degrees"]),
        ),
    )


def transform_max_difference(
    first: carla.Transform, second: carla.Transform
) -> tuple[float, float]:
    residual = transform_residual(first, second)
    return float(residual["position_error_m"]), float(residual["angle_error_degrees"])


def write_contact_sheet(
    output_path: Path,
    modality_paths: dict[str, dict[int, Path]],
    event_samples: list[int],
) -> None:
    magick = shutil.which("magick")
    if magick is None:
        raise RuntimeError("ImageMagick is required for the N2 contact sheet")
    sources: list[str] = []
    for sample_index in event_samples:
        for sensor_name in SENSOR_ORDER:
            sources.append(str(modality_paths[sensor_name][sample_index]))
    subprocess.run(
        [
            magick,
            "montage",
            *sources,
            "-thumbnail",
            "480x270",
            "-tile",
            f"{len(SENSOR_ORDER)}x{len(event_samples)}",
            "-geometry",
            "+5+5",
            str(output_path),
        ],
        check=True,
    )
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError("N2 contact sheet was not materialized")


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=True)
    protocol = read_json(protocol_path)
    bundle = verify_source_bundle(protocol, source_root)
    frozen_protocol = output_root / "frozen_protocol.json"
    if not frozen_protocol.is_file() or sha256_file(frozen_protocol) != sha256_file(
        protocol_path
    ):
        raise RuntimeError("runner did not freeze the requested N2 protocol")
    for reserved in ("model", "evaluator", "result.json"):
        if (output_root / reserved).exists():
            raise FileExistsError(f"refusing N2 output overwrite: {output_root / reserved}")

    source_receipt_path = output_root / "source_bundle_receipt.json"
    write_json_atomic(source_receipt_path, bundle["receipt"])
    capture = protocol["capture"]
    environment = protocol["environment"]
    episode = protocol["episode"]
    fixed_delta = float(environment["fixed_delta_seconds"])
    width, height = map(int, capture["resolution"])
    fov = float(capture["fov_degrees"])
    source_rows = bundle["rows"]
    source_roster = actor_roster(bundle["actor_manifest"])

    client = connect(args.host, args.port)
    if client.get_client_version() != environment["carla_version"]:
        raise RuntimeError(f"unexpected CARLA client version: {client.get_client_version()}")
    if client.get_server_version() != environment["carla_version"]:
        raise RuntimeError(f"unexpected CARLA server version: {client.get_server_version()}")
    world = client.get_world()
    if world.get_map().name != environment["map"]:
        raise RuntimeError(f"unexpected CARLA map: {world.get_map().name}")
    shared = [
        actor
        for actor in world.get_actors()
        if actor.type_id.startswith(("vehicle.", "walker.", "sensor.", "controller."))
    ]
    if shared:
        raise RuntimeError("refusing a shared CARLA world")

    original_settings = copy_settings(world.get_settings())
    original_weather = world.get_weather()
    owned: list[carla.Actor] = []
    actors: dict[str, carla.Actor] = {}
    sensors: dict[str, carla.Sensor] = {}
    queues: dict[str, queue.Queue[carla.SensorData]] = {
        name: queue.Queue() for name in SENSOR_ORDER
    }
    wearer: carla.Actor | None = None
    maximum_position_error_m = 0.0
    maximum_angle_error_degrees = 0.0
    maximum_camera_position_difference_m = 0.0
    maximum_camera_angle_difference_degrees = 0.0
    witness_position_error_m = 0.0
    witness_angle_error_degrees = 0.0
    door_open_calls = 0
    door_close_calls = 0

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
        world.set_weather(weather_parameter(str(environment["weather"])))
        first_states = source_rows[0]["actors"]
        actor_manifests: list[dict[str, Any]] = []
        for fallback_index, actor_id in enumerate(sorted(source_roster)):
            source_state = first_states[actor_id]
            actor, manifest = spawn_physics_off_actor(
                world,
                actor_id=actor_id,
                type_id=str(source_state["type_id"]),
                kind=str(source_state["kind"]),
                initial_transform=carla_transform(source_state["transform"]),
                fallback_height_index=fallback_index,
            )
            if str(actor.type_id) != str(source_state["type_id"]):
                raise RuntimeError(f"exact replay blueprint mismatch for {actor_id}")
            actors[actor_id] = actor
            owned.append(actor)
            actor_manifests.append(
                {
                    **manifest,
                    "source_carla_actor_id": int(source_state["carla_actor_id"]),
                    "behavior_profile": str(source_roster[actor_id]["behavior_profile"]),
                }
            )

        wearer_spec = capture["wearer"]
        wearer_xy = carla.Location(
            x=float(wearer_spec["world_x_m"]),
            y=float(wearer_spec["world_y_m"]),
            z=0.0,
        )
        wearer_z = road_surface_z(
            world.get_map(),
            np.asarray([wearer_xy.x, wearer_xy.y], dtype=np.float64),
            float(wearer_spec["surface_offset_m"]),
        )
        wearer_transform = carla.Transform(
            carla.Location(x=wearer_xy.x, y=wearer_xy.y, z=wearer_z),
            carla.Rotation(yaw=float(wearer_spec["yaw_degrees"])),
        )
        wearer, wearer_manifest = spawn_physics_off_actor(
            world,
            actor_id="fixed_synthetic_observer",
            type_id=str(wearer_spec["blueprint"]),
            kind="pedestrian",
            initial_transform=wearer_transform,
            fallback_height_index=len(actors) + 1,
        )
        owned.append(wearer)
        wearer_manifest["observer_mode"] = str(wearer_spec["observer_mode"])

        relative = relative_sensor_transform(capture["wearable_relative_transform"])
        witness_expected = carla_transform(bundle["actor_manifest"]["camera"]["transform"])
        for sensor_name in SENSOR_ORDER:
            blueprint = sensor_blueprint(
                world,
                SENSOR_TYPES[sensor_name],
                width,
                height,
                fov,
                fixed_delta,
            )
            if sensor_name == "witness":
                sensor = world.spawn_actor(blueprint, witness_expected)
            else:
                sensor = world.spawn_actor(
                    blueprint,
                    relative,
                    attach_to=wearer,
                    attachment_type=carla.AttachmentType.Rigid,
                )
            sensor.listen(queues[sensor_name].put)
            sensors[sensor_name] = sensor
            owned.append(sensor)

        warmup_counts = {name: 0 for name in SENSOR_ORDER}
        warmup_deadline = time.monotonic() + 120.0
        while min(warmup_counts.values()) < 3:
            for name in SENSOR_ORDER:
                if warmup_counts[name] >= 3:
                    continue
                try:
                    queues[name].get(timeout=0.25)
                    warmup_counts[name] += 1
                except queue.Empty:
                    if time.monotonic() >= warmup_deadline:
                        raise TimeoutError(f"{name} camera warmup timed out")

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
        for values in queues.values():
            while True:
                try:
                    values.get_nowait()
                except queue.Empty:
                    break
        phase_aligned = False
        for _ in range(8):
            frame = world.tick(30.0)
            try:
                for name in SENSOR_ORDER:
                    await_frame(queues[name], frame, name, timeout_seconds=8.0)
                phase_aligned = True
                break
            except TimeoutError:
                continue
        if not phase_aligned:
            raise TimeoutError("could not phase-align all four N2 sensors")

        model_root = output_root / "model"
        evaluator_root = output_root / "evaluator"
        episode_id = str(episode["episode_id"])
        model_episode = model_root / "episodes" / episode_id
        evaluator_episode = evaluator_root / "episodes" / episode_id
        modality_roots = {
            "wearable": model_episode / "rgb",
            "depth": model_episode / "depth",
            "instance": evaluator_episode / "instance",
            "witness": evaluator_episode / "witness",
        }
        for path in modality_roots.values():
            path.mkdir(parents=True, exist_ok=False)
        modality_paths: dict[str, dict[int, Path]] = {
            name: {} for name in SENSOR_ORDER
        }
        payload_inventory: list[dict[str, Any]] = []
        model_rows: list[dict[str, Any]] = []
        evaluator_rows: list[dict[str, Any]] = []
        alignment_rows: list[dict[str, Any]] = []
        replay_event_receipts: list[dict[str, Any]] = []
        frozen_door = door_event(bundle["events"])
        door_actor = actors[str(frozen_door["primary_actor_id"])]
        door_plan = next(
            value
            for value in bundle["plan"]["tail_events"]
            if value["event_id"] == frozen_door["event_id"]
        )
        door_side = str(door_plan["intent_effect"]["door_side"])
        door_selector = carla.VehicleDoor.FL if door_side == "left" else carla.VehicleDoor.FR
        door_is_open = False

        for source_row in source_rows:
            sample_index = int(source_row["sample_index"])
            logical_time_s = float(source_row["time_s"])
            should_open = (
                float(frozen_door["applied_time_s"]) - 1e-9
                <= logical_time_s
                < float(frozen_door["ended_time_s"]) - 1e-9
            )
            if should_open and not door_is_open:
                door_actor.open_door(door_selector)
                door_is_open = True
                door_open_calls += 1
                replay_event_receipts.append(
                    {
                        "event_id": str(frozen_door["event_id"]),
                        "operation": "open_door",
                        "sample_index": sample_index,
                        "time_s": logical_time_s,
                        "source_event_receipt_sha256": protocol["source"]["files"][
                            "event_receipts.json"
                        ],
                    }
                )
            elif not should_open and door_is_open:
                door_actor.close_door(door_selector)
                door_is_open = False
                door_close_calls += 1
                replay_event_receipts.append(
                    {
                        "event_id": str(frozen_door["event_id"]),
                        "operation": "close_door",
                        "sample_index": sample_index,
                        "time_s": logical_time_s,
                        "source_event_receipt_sha256": protocol["source"]["files"][
                            "event_receipts.json"
                        ],
                    }
                )

            labels = ["fixed_synthetic_observer", *sorted(actors)]
            commands = [carla.command.ApplyTransform(wearer.id, wearer_transform)]
            commands.extend(
                carla.command.ApplyTransform(
                    actors[actor_id].id,
                    carla_transform(source_row["actors"][actor_id]["transform"]),
                )
                for actor_id in labels[1:]
            )
            previous_frame = int(world.get_snapshot().frame)
            responses = client.apply_batch_sync(commands, do_tick=True)
            failures = [
                {"actor": label, "error": str(response.error)}
                for label, response in zip(labels, responses, strict=True)
                if response.has_error()
            ]
            if failures:
                raise RuntimeError(f"N2 trace pose batch failed: {json.dumps(failures)}")
            snapshot = world.get_snapshot()
            replay_frame = int(snapshot.frame)
            if replay_frame != previous_frame + 1:
                raise RuntimeError("N2 replay advanced more than one CARLA frame")
            images = {
                name: await_frame(queues[name], replay_frame, name)
                for name in SENSOR_ORDER
            }

            actual_transforms: dict[str, carla.Transform] = {}
            actor_states: dict[str, dict[str, Any]] = {}
            collision_polygons: dict[str, list[list[float]]] = {}
            for actor_id in sorted(actors):
                actor = actors[actor_id]
                actor_snapshot = snapshot.find(actor.id)
                if actor_snapshot is None:
                    raise RuntimeError(f"snapshot omitted N2 actor {actor_id}")
                actual = actor_snapshot.get_transform()
                expected = carla_transform(source_row["actors"][actor_id]["transform"])
                residual = transform_residual(actual, expected)
                maximum_position_error_m = max(
                    maximum_position_error_m, float(residual["position_error_m"])
                )
                maximum_angle_error_degrees = max(
                    maximum_angle_error_degrees,
                    float(residual["angle_error_degrees"]),
                )
                if (
                    float(residual["position_error_m"]) > POSITION_TOLERANCE_M
                    or float(residual["angle_error_degrees"])
                    > ANGLE_TOLERANCE_DEGREES
                ):
                    raise RuntimeError(
                        f"N2 actor pose differs from frozen trace: {actor_id}/{residual}"
                    )
                actual_transforms[actor_id] = actual
                source_state = source_row["actors"][actor_id]
                actor_states[actor_id] = {
                    "actor_id": actor_id,
                    "kind": str(source_state["kind"]),
                    "actual_blueprint": str(actor.type_id),
                    "replay_carla_actor_id": int(actor.id),
                    "source_carla_actor_id": int(source_state["carla_actor_id"]),
                    "source_transform": source_state["transform"],
                    "replay_transform": transform_dict(actual),
                    "replay_pose_residual": residual,
                    "source_velocity": source_state["velocity"],
                    "source_acceleration": source_state["acceleration"],
                    "source_angular_velocity": source_state["angular_velocity"],
                    "source_control": source_state["control"],
                    "bounding_box": bbox_dict(actor, actual),
                }
                collision_polygons[actor_id] = polygon_from_bbox(actor, actual)

            wearer_snapshot = snapshot.find(wearer.id)
            if wearer_snapshot is None:
                raise RuntimeError("snapshot omitted fixed synthetic observer")
            actual_wearer = wearer_snapshot.get_transform()
            wearer_residual = transform_residual(actual_wearer, wearer_transform)
            if float(wearer_residual["position_error_m"]) > POSITION_TOLERANCE_M:
                raise RuntimeError("fixed synthetic observer pose drifted")

            attached_images = [images[name] for name in ("wearable", "depth", "instance")]
            reference_camera = attached_images[0].transform
            for image in attached_images[1:]:
                position_difference, angle_difference = transform_max_difference(
                    reference_camera, image.transform
                )
                maximum_camera_position_difference_m = max(
                    maximum_camera_position_difference_m, position_difference
                )
                maximum_camera_angle_difference_degrees = max(
                    maximum_camera_angle_difference_degrees, angle_difference
                )
            witness_position_error_m, witness_angle_error_degrees = transform_max_difference(
                images["witness"].transform, witness_expected
            )

            payloads: dict[str, dict[str, Any]] = {}
            for sensor_name in SENSOR_ORDER:
                path = modality_roots[sensor_name] / f"{sample_index:06d}.png"
                payload = save_image(images[sensor_name], path)
                payload.update(
                    {
                        "sensor": sensor_name,
                        "episode_id": episode_id,
                        "sample_index": sample_index,
                        "relative_path": safe_relative(path, output_root),
                    }
                )
                payload_inventory.append(payload)
                payloads[sensor_name] = payload
                modality_paths[sensor_name][sample_index] = path

            visibility = batched_instance_metrics(images["instance"], actors)
            contact, minimum_distance, responsible = contact_union(
                (float(actual_wearer.location.x), float(actual_wearer.location.y)),
                collision_polygons,
                wearer_radius_m=float(wearer_spec["body_radius_m"]),
            )
            sensor_world_frames = {
                name: int(images[name].frame) for name in SENSOR_ORDER
            }
            alignment_rows.append(
                {
                    "sample_index": sample_index,
                    "time_s": logical_time_s,
                    "source_world_frame": int(source_row["world_frame"]),
                    "replay_world_frame": replay_frame,
                    "sensor_world_frames": sensor_world_frames,
                }
            )
            model_rows.append(
                {
                    "schema_version": "dtr-c2-model-observation-v2",
                    "episode_id": episode_id,
                    "sample_index": sample_index,
                    "world_frame": replay_frame,
                    "time_s": logical_time_s,
                    "timestamp_s": logical_time_s,
                    "wearable_rgb": {
                        "path": safe_relative(modality_paths["wearable"][sample_index], model_root),
                        "bytes": int(payloads["wearable"]["bytes"]),
                        "sha256": str(payloads["wearable"]["sha256"]),
                        "width": width,
                        "height": height,
                        "source_world_frame": replay_frame,
                    },
                    "metric_depth": {
                        "path": safe_relative(modality_paths["depth"][sample_index], model_root),
                        "bytes": int(payloads["depth"]["bytes"]),
                        "sha256": str(payloads["depth"]["sha256"]),
                        "width": width,
                        "height": height,
                        "codec": capture["camera_calibration"]["depth_codec"],
                        "source_world_frame": replay_frame,
                    },
                    "camera": {
                        "world_transform": transform_dict(reference_camera),
                        "rigid_extrinsic": capture["wearable_relative_transform"],
                        "width": width,
                        "height": height,
                        "fov_degrees": fov,
                        "K": camera_intrinsics(width, height, fov),
                    },
                    "wearer_pose_current": transform_dict(actual_wearer),
                    "navigation": {
                        "navigation_session_id": str(episode["navigation_session_id"]),
                        "issued_plan": {
                            "authority": "NO_PLAN",
                            "path": None,
                            "receipt_sha256": None,
                        },
                    },
                    "frame_alignment": {
                        "authority": "PENDING_SEAL",
                        "reference_modality": "wearable_rgb",
                        "receipt_path": "four_modal_alignment_receipt.json",
                        "receipt_sha256": None,
                        "source_trace_sha256": protocol["source"]["files"][
                            "behavior_trace.jsonl"
                        ],
                        "source_behavior_world_frame": int(source_row["world_frame"]),
                        "same_world_frame_modality_offset": 0,
                    },
                }
            )
            evaluator_rows.append(
                {
                    "schema_version": "dtr-c2-evaluator-frame-v1",
                    "episode_id": episode_id,
                    "layout_id": str(episode["layout_id"]),
                    "sample_index": sample_index,
                    "time_s": logical_time_s,
                    "source_behavior_world_frame": int(source_row["world_frame"]),
                    "replay_world_frame": replay_frame,
                    "instance": {
                        "path": safe_relative(
                            modality_paths["instance"][sample_index], evaluator_root
                        ),
                        "bytes": int(payloads["instance"]["bytes"]),
                        "sha256": str(payloads["instance"]["sha256"]),
                        "source_world_frame": replay_frame,
                    },
                    "witness": {
                        "path": safe_relative(
                            modality_paths["witness"][sample_index], evaluator_root
                        ),
                        "bytes": int(payloads["witness"]["bytes"]),
                        "sha256": str(payloads["witness"]["sha256"]),
                        "source_world_frame": replay_frame,
                    },
                    "camera_transform": transform_dict(images["instance"].transform),
                    "wearer_transform": transform_dict(actual_wearer),
                    "actors": actor_states,
                    "instance_visibility": visibility,
                    "truth": {
                        "source_behavior_trace_sha256": protocol["source"]["files"][
                            "behavior_trace.jsonl"
                        ],
                        "current_contact": bool(contact),
                        "minimum_distance_m": float(minimum_distance),
                        "responsible_assets": responsible,
                        "collision_polygons_xy": collision_polygons,
                        "active_tail_event_ids": active_tail_event_ids(
                            bundle["events"], logical_time_s
                        ),
                    },
                }
            )

        horizon_s = float(episode["future_contact_horizon_seconds"])
        for index, row in enumerate(evaluator_rows):
            future = [
                value
                for value in evaluator_rows[index:]
                if float(value["time_s"]) - float(row["time_s"]) <= horizon_s + 1e-9
                and bool(value["truth"]["current_contact"])
            ]
            row["truth"]["future_contact_within_horizon"] = bool(future)
            row["truth"]["realized_time_to_contact_seconds"] = (
                float(future[0]["time_s"]) - float(row["time_s"]) if future else None
            )

        alignment_receipt = build_alignment_receipt(
            str(bundle["receipt"]["receipt_sha256"]), alignment_rows
        )
        alignment_path = model_root / "four_modal_alignment_receipt.json"
        write_json_atomic(alignment_path, alignment_receipt)
        for row in model_rows:
            row["frame_alignment"].update(
                {
                    "authority": alignment_receipt["authority"],
                    "receipt_sha256": alignment_receipt["receipt_sha256"],
                }
            )
            validate_model_record(row)

        observations_path = model_episode / "observations.jsonl"
        evaluator_frames_path = evaluator_episode / "frames.jsonl"
        write_jsonl(observations_path, model_rows)
        write_jsonl(evaluator_frames_path, evaluator_rows)
        replay_event_path = evaluator_root / "replayed_discrete_events.json"
        write_json_atomic(
            replay_event_path,
            {
                "schema_version": "dtr-carla-n2-replayed-discrete-events-v1",
                "source_event_receipts_sha256": protocol["source"]["files"][
                    "event_receipts.json"
                ],
                "receipts": replay_event_receipts,
            },
        )
        payload_inventory_path = output_root / "payload_inventory.json"
        write_json_atomic(payload_inventory_path, payload_inventory)
        actor_manifest_path = evaluator_root / "actor_manifest.json"
        write_json_atomic(
            actor_manifest_path,
            {
                "schema_version": "dtr-carla-n2-replay-actor-manifest-v1",
                "actors": actor_manifests,
                "wearer": wearer_manifest,
            },
        )

        calibration_path = model_root / "camera_calibration.json"
        write_json_atomic(
            calibration_path,
            {
                "schema_version": "dtr-c2-camera-calibration-v1",
                "resolution": {"width": width, "height": height},
                "fov_degrees": fov,
                "K": camera_intrinsics(width, height, fov),
                "sensor_tick_seconds": fixed_delta,
                "wearable_rigid_extrinsic": capture["wearable_relative_transform"],
                "depth_codec": capture["camera_calibration"]["depth_codec"],
            },
        )
        model_contract_path = model_root / "model_contract.json"
        write_json_atomic(
            model_contract_path,
            {
                **protocol["model_contract"],
                "current_actors_enabled": False,
                "fixed_synthetic_observer": True,
                "source_alignment": {
                    "authority": alignment_receipt["authority"],
                    "receipt_path": safe_relative(alignment_path, model_root),
                    "receipt_sha256": alignment_receipt["receipt_sha256"],
                    "source_bundle_receipt_sha256": bundle["receipt"]["receipt_sha256"],
                    "source_trace_sha256": protocol["source"]["files"][
                        "behavior_trace.jsonl"
                    ],
                },
            },
        )
        model_episode_manifest_path = model_episode / "manifest.json"
        write_json_atomic(
            model_episode_manifest_path,
            {
                "schema_version": "dtr-c2-model-episode-manifest-v2",
                "episode_id": episode_id,
                "frames": len(model_rows),
                "observations_sha256": sha256_file(observations_path),
                "rgb_payloads": len(model_rows),
                "depth_payloads": len(model_rows),
                "navigation_session_id": str(episode["navigation_session_id"]),
                "issued_plan": {"authority": "NO_PLAN"},
                "four_modal_alignment_receipt_sha256": alignment_receipt[
                    "receipt_sha256"
                ],
            },
        )
        model_root_manifest_path = model_root / "manifest.json"
        write_json_atomic(
            model_root_manifest_path,
            {
                "schema_version": "dtr-c2-model-root-manifest-v2",
                "experiment_id": EXPERIMENT_ID,
                "camera_calibration_sha256": sha256_file(calibration_path),
                "model_contract_sha256": sha256_file(model_contract_path),
                "alignment_receipt_sha256": sha256_file(alignment_path),
                "episode_manifest_sha256": sha256_file(model_episode_manifest_path),
            },
        )
        evaluator_manifest_path = evaluator_root / "manifest.json"
        write_json_atomic(
            evaluator_manifest_path,
            {
                "schema_version": "dtr-carla-n2-evaluator-root-manifest-v1",
                "experiment_id": EXPERIMENT_ID,
                "episode_id": episode_id,
                "frames": len(evaluator_rows),
                "frames_sha256": sha256_file(evaluator_frames_path),
                "actor_manifest_sha256": sha256_file(actor_manifest_path),
                "replayed_discrete_events_sha256": sha256_file(replay_event_path),
                "source_bundle_receipt_sha256": bundle["receipt"]["receipt_sha256"],
            },
        )

        event_samples = sorted(
            {
                int(round(float(value["applied_time_s"]) / fixed_delta))
                for value in bundle["events"]["tail_events"]
            }
        )
        contact_sheet_path = evaluator_root / "contact_sheet.png"
        write_contact_sheet(contact_sheet_path, modality_paths, event_samples)

        model_truth_failures = scan_model_tree(model_root)
        model_manifest = seal_tree(model_root)
        sealed_model_manifest_path = output_root / "sealed_model_manifest.json"
        write_json_atomic(sealed_model_manifest_path, model_manifest)
        evidence_files: list[dict[str, Any]] = []
        for directory in (output_root / "inputs", model_root, evaluator_root):
            for path in sorted(value for value in directory.rglob("*") if value.is_file()):
                evidence_files.append(
                    {
                        "path": safe_relative(path, output_root),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        for path in (
            frozen_protocol,
            source_receipt_path,
            payload_inventory_path,
            sealed_model_manifest_path,
        ):
            evidence_files.append(
                {
                    "path": safe_relative(path, output_root),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        evidence_files.sort(key=lambda value: str(value["path"]))
        sealed_evidence_manifest_path = output_root / "sealed_evidence_manifest.json"
        write_json_atomic(sealed_evidence_manifest_path, evidence_files)

        expected_frames = int(protocol["source"]["expected_trace_frames"])
        expected_payloads = expected_frames * len(SENSOR_ORDER)
        checks = {
            "frozen_n1_source_bundle_verified": bundle["receipt"]["authority"]
            == "FROZEN_N1_SOURCE_BUNDLE_VERIFIED",
            "source_trace_frame_and_actor_denominators_exact": len(source_rows)
            == expected_frames
            and len(source_roster) == int(protocol["source"]["expected_actor_count"]),
            "physics_and_collisions_disabled_for_all_replay_actors": all(
                bool(value["simulate_physics_disabled"])
                and bool(value["collisions_disabled"])
                for value in actor_manifests
            ),
            "all_four_modalities_share_each_replay_world_frame": alignment_receipt[
                "frames"
            ]
            == expected_frames,
            "all_four_modal_payloads_complete": len(payload_inventory)
            == expected_payloads,
            "all_formal_payloads_are_1280x720": all(
                int(value["width"]) == width and int(value["height"]) == height
                for value in payload_inventory
            ),
            "all_replayed_actor_poses_match_frozen_trace": maximum_position_error_m
            <= POSITION_TOLERANCE_M
            and maximum_angle_error_degrees <= ANGLE_TOLERANCE_DEGREES,
            "wearable_depth_instance_camera_transforms_identical": (
                maximum_camera_position_difference_m <= POSITION_TOLERANCE_M
                and maximum_camera_angle_difference_degrees
                <= ANGLE_TOLERANCE_DEGREES
            ),
            "witness_transform_matches_source_manifest": witness_position_error_m
            <= POSITION_TOLERANCE_M
            and witness_angle_error_degrees <= ANGLE_TOLERANCE_DEGREES,
            "door_open_discrete_state_replayed_from_frozen_receipt": door_open_calls == 1
            and door_close_calls == 1,
            "c2_model_evaluator_frame_denominators_exact": len(model_rows)
            == expected_frames
            and len(evaluator_rows) == expected_frames,
            "model_root_contains_no_actor_or_evaluator_truth": not model_truth_failures,
            "contact_sheet_materialized": contact_sheet_path.is_file()
            and contact_sheet_path.stat().st_size > 0,
            "sealed_model_and_evidence_manifests_nonempty": bool(model_manifest)
            and bool(evidence_files),
        }
        result = {
            "schema_version": "dtr-carla-n2-frozen-trace-c2-replay-result-v1",
            "experiment_id": EXPERIMENT_ID,
            "status": (
                "DTR_CARLA_N2_FROZEN_TRACE_C2_REPLAY_COMPLETE"
                if all(checks.values())
                else "DTR_CARLA_N2_FROZEN_TRACE_C2_REPLAY_NOT_EVALUABLE"
            ),
            "checks": checks,
            "claim_boundary": protocol["claim_boundary"],
            "protocol_sha256": sha256_file(protocol_path),
            "capture_script_sha256": sha256_file(Path(__file__).resolve()),
            "helper_module_sha256": sha256_file(
                Path(__file__).with_name("dtr_carla_n2_frozen_trace_replay.py")
            ),
            "source_bundle_receipt_sha256": bundle["receipt"]["receipt_sha256"],
            "source_trace_sha256": protocol["source"]["files"]["behavior_trace.jsonl"],
            "source_event_receipts_sha256": protocol["source"]["files"][
                "event_receipts.json"
            ],
            "alignment_receipt_sha256": alignment_receipt["receipt_sha256"],
            "alignment_receipt_file_sha256": sha256_file(alignment_path),
            "carla_client_version": client.get_client_version(),
            "carla_server_version": client.get_server_version(),
            "map": world.get_map().name,
            "fixed_delta_seconds": fixed_delta,
            "frames": expected_frames,
            "source_actor_count": len(source_roster),
            "sensor_order": list(SENSOR_ORDER),
            "total_sensor_payloads": len(payload_inventory),
            "model_observation_frames": len(model_rows),
            "evaluator_frames": len(evaluator_rows),
            "maximum_actor_position_error_m": maximum_position_error_m,
            "maximum_actor_angle_error_degrees": maximum_angle_error_degrees,
            "maximum_attached_camera_position_difference_m": (
                maximum_camera_position_difference_m
            ),
            "maximum_attached_camera_angle_difference_degrees": (
                maximum_camera_angle_difference_degrees
            ),
            "door_replay_calls": {
                "open": door_open_calls,
                "close": door_close_calls,
            },
            "model_truth_failures": model_truth_failures,
            "payload_inventory_sha256": sha256_file(payload_inventory_path),
            "sealed_model_manifest_sha256": sha256_file(sealed_model_manifest_path),
            "sealed_evidence_manifest_sha256": sha256_file(
                sealed_evidence_manifest_path
            ),
            "contact_sheet": {
                "path": str(contact_sheet_path),
                "bytes": contact_sheet_path.stat().st_size,
                "sha256": sha256_file(contact_sheet_path),
                "event_sample_indices": event_samples,
            },
            "model_root": str(model_root),
            "evaluator_root": str(evaluator_root),
        }
        write_json_atomic(output_root / "result.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if all(checks.values()) else 2
    finally:
        for sensor in sensors.values():
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
            except Exception as error:
                print(f"WARNING N2 actor cleanup failed: {error}", file=sys.stderr)
        try:
            world.set_weather(original_weather)
            apply_settings(world, original_settings)
        except Exception as error:
            print(f"WARNING N2 world restore failed: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
