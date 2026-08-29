"""Capture one DTR-CARLA-C0 modality with one long-lived CARLA camera.

Run this file only with the external CARLA 0.9.16 client Python.  It owns no
BlindAssist model code; it materializes raw observable payloads plus evaluator-
only state for later truth-separated observation arms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import queue
import sys
import time
from pathlib import Path
from typing import Any

import carla
import numpy as np


SENSOR_TYPES = {
    "rgb": "sensor.camera.rgb",
    "depth": "sensor.camera.depth",
    "flow": "sensor.camera.optical_flow",
    "instance": "sensor.camera.instance_segmentation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sensor", choices=tuple(SENSOR_TYPES), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    return parser.parse_args()


def load_carla_utilities(carla_root: Path) -> Any:
    source = carla_root / "experiments" / "c0-crossing-v1" / "run_c0.py"
    if not source.is_file():
        raise FileNotFoundError(f"CARLA capture utilities are unavailable: {source}")
    spec = importlib.util.spec_from_file_location("dtr_carla_c0_runtime", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CARLA capture utilities: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def segments(trajectory: dict[str, Any]) -> list[dict[str, float]]:
    values = sorted(trajectory["segments"], key=lambda item: float(item["start_s"]))
    if not values or abs(float(values[0]["start_s"])) > 1e-9:
        raise ValueError("every trajectory must start at t=0")
    return values


def trajectory_position(trajectory: dict[str, Any], time_s: float) -> np.ndarray:
    position = np.asarray(
        [trajectory["start_forward_m"], trajectory["start_right_m"]],
        dtype=np.float64,
    )
    values = segments(trajectory)
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
        velocity = np.asarray(
            [segment["velocity_forward_mps"], segment["velocity_right_mps"]],
            dtype=np.float64,
        )
        position += velocity * (end_s - start_s)
        if end_s >= time_s:
            break
    return position


def trajectory_velocity(trajectory: dict[str, Any], time_s: float) -> np.ndarray:
    selected = segments(trajectory)[0]
    for segment in segments(trajectory):
        if float(segment["start_s"]) <= time_s + 1e-9:
            selected = segment
        else:
            break
    return np.asarray(
        [selected["velocity_forward_mps"], selected["velocity_right_mps"]],
        dtype=np.float64,
    )


def scheduled_camera_yaw(scenario: dict[str, Any], time_s: float) -> float:
    selected = 0.0
    for value in sorted(
        scenario.get("camera_yaw_offsets", [{"start_s": 0.0, "yaw_degrees": 0.0}]),
        key=lambda item: float(item["start_s"]),
    ):
        if float(value["start_s"]) <= time_s + 1e-9:
            selected = float(value["yaw_degrees"])
        else:
            break
    return selected


def world_xy(
    local: np.ndarray,
    center_xy: np.ndarray,
    forward_xy: np.ndarray,
    right_xy: np.ndarray,
) -> np.ndarray:
    return center_xy + forward_xy * float(local[0]) + right_xy * float(local[1])


def world_velocity(
    local: np.ndarray, forward_xy: np.ndarray, right_xy: np.ndarray
) -> np.ndarray:
    return forward_xy * float(local[0]) + right_xy * float(local[1])


def yaw_degrees(velocity_xy: np.ndarray, fallback: float) -> float:
    if float(np.linalg.norm(velocity_xy)) <= 1e-9:
        return fallback
    return math.degrees(math.atan2(float(velocity_xy[1]), float(velocity_xy[0])))


def vector_dict(value: np.ndarray) -> dict[str, float]:
    return {"x": float(value[0]), "y": float(value[1]), "z": 0.0}


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
        "actor_id": int(actor_id),
        "pixels": pixels,
        "visible": pixels >= 16,
        "bbox_uv": (
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if pixels
            else None
        ),
    }


def spawn_occluder(
    world: carla.World,
    blueprint_id: str,
    hidden_transform: carla.Transform,
) -> carla.Actor:
    blueprint = world.get_blueprint_library().find(blueprint_id)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "dtr_c0_occluder")
    actor = world.try_spawn_actor(blueprint, hidden_transform)
    if actor is None:
        raise RuntimeError("failed to spawn the C0 occluder")
    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass
    return actor


def scenario_pose(
    scenario: dict[str, Any],
    time_s: float,
    *,
    center_xy: np.ndarray,
    forward_xy: np.ndarray,
    right_xy: np.ndarray,
    origin_z: float,
    camera_base_yaw_degrees: float,
) -> dict[str, Any]:
    fallback_yaw = math.degrees(math.atan2(float(forward_xy[1]), float(forward_xy[0])))
    ego_local = trajectory_position(scenario["ego"], time_s)
    target_local = trajectory_position(scenario["target"], time_s)
    ego_velocity_local = trajectory_velocity(scenario["ego"], time_s)
    target_velocity_local = trajectory_velocity(scenario["target"], time_s)
    ego_xy = world_xy(ego_local, center_xy, forward_xy, right_xy)
    target_xy = world_xy(target_local, center_xy, forward_xy, right_xy)
    ego_velocity_xy = world_velocity(ego_velocity_local, forward_xy, right_xy)
    target_velocity_xy = world_velocity(target_velocity_local, forward_xy, right_xy)
    body_yaw = yaw_degrees(ego_velocity_xy, fallback_yaw)
    target_yaw = yaw_degrees(target_velocity_xy, fallback_yaw)
    head_offset = scheduled_camera_yaw(scenario, time_s)
    render_yaw = body_yaw + head_offset
    return {
        "ego_transform": carla.Transform(
            carla.Location(x=float(ego_xy[0]), y=float(ego_xy[1]), z=origin_z),
            carla.Rotation(yaw=render_yaw),
        ),
        "target_transform": carla.Transform(
            carla.Location(x=float(target_xy[0]), y=float(target_xy[1]), z=origin_z),
            carla.Rotation(yaw=target_yaw),
        ),
        "ego_xy": ego_xy,
        "target_xy": target_xy,
        "ego_velocity_xy": ego_velocity_xy,
        "target_velocity_xy": target_velocity_xy,
        "body_yaw_degrees": body_yaw,
        "sensor_yaw_degrees": render_yaw + camera_base_yaw_degrees,
        "target_yaw_degrees": target_yaw,
        "head_yaw_offset_degrees": head_offset,
    }


def main() -> int:
    args = parse_args()
    carla_root = args.carla_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    output_root = args.output_root.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != "DTR_CARLA_C0_CAUSAL_BENCHMARK_CANARY_V1":
        raise ValueError("unexpected DTR-CARLA-C0 protocol identity")
    if args.sensor not in protocol["capture"]["sensor_order"]:
        raise ValueError(f"sensor is not frozen by the protocol: {args.sensor}")
    sensor_root = output_root / args.sensor
    if sensor_root.exists():
        raise FileExistsError(f"refusing to overwrite modality capture: {sensor_root}")
    sensor_root.mkdir(parents=True)

    utility = load_carla_utilities(carla_root)
    client = utility.connect(args.host, args.port)
    if client.get_client_version() != "0.9.16" or client.get_server_version() != "0.9.16":
        raise RuntimeError("DTR-CARLA-C0 requires CARLA 0.9.16 client and server")
    world = client.get_world()
    environment = protocol["environment"]
    if world.get_map().name != environment["map"]:
        raise RuntimeError(f"unexpected CARLA map: {world.get_map().name}")
    shared = [
        actor
        for actor in world.get_actors()
        if actor.type_id.startswith(("vehicle.", "walker.", "sensor."))
    ]
    if shared:
        raise RuntimeError("refusing a shared CARLA world")

    original_settings = utility.copy_settings(world.get_settings())
    original_weather = world.get_weather()
    actors: list[carla.Actor] = []
    sensor: carla.Sensor | None = None
    sensor_queue: queue.Queue[carla.SensorData] = queue.Queue()
    center_xy = np.asarray(environment["anchor"]["center_xy_m"], dtype=np.float64)
    forward_xy = np.asarray(environment["anchor"]["forward_xy"], dtype=np.float64)
    right_xy = np.asarray(environment["anchor"]["right_xy"], dtype=np.float64)
    forward_xy /= np.linalg.norm(forward_xy)
    right_xy /= np.linalg.norm(right_xy)
    origin_z = float(environment["walker_origin_z_m"])
    fixed_delta = float(environment["sample_seconds"])
    simulation_delta = float(environment["simulation_tick_seconds"])
    duration_s = float(environment["duration_seconds"])
    frame_count = int(round(duration_s / fixed_delta)) + 1
    camera = protocol["camera"]
    width, height = (int(value) for value in camera["resolution"])
    camera_relative = camera["relative_transform"]
    camera_base_yaw = float(camera_relative["yaw_degrees"])
    hidden_occluder = carla.Transform(
        carla.Location(x=float(center_xy[0]), y=float(center_xy[1]), z=-20.0)
    )

    try:
        utility.apply_settings(
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
        initial = scenario_pose(
            first,
            0.0,
            center_xy=center_xy,
            forward_xy=forward_xy,
            right_xy=right_xy,
            origin_z=origin_z,
            camera_base_yaw_degrees=camera_base_yaw,
        )
        blueprints = protocol["capture"]["actor_blueprints"]
        ego = utility.spawn_walker(world, blueprints["ego"], "dtr_c0_ego", initial["ego_transform"])
        target = utility.spawn_walker(
            world, blueprints["target"], "dtr_c0_target", initial["target_transform"]
        )
        ego.set_simulate_physics(False)
        target.set_simulate_physics(False)
        ego.apply_control(carla.WalkerControl(speed=0.0))
        target.apply_control(carla.WalkerControl(speed=0.0))
        occluder = spawn_occluder(world, blueprints["occluder"], hidden_occluder)
        actors.extend([ego, target, occluder])

        relative_transform = carla.Transform(
            carla.Location(
                x=float(camera_relative["x_m"]),
                y=float(camera_relative["y_m"]),
                z=float(camera_relative["z_m"]),
            ),
            carla.Rotation(
                pitch=float(camera_relative["pitch_degrees"]),
                yaw=camera_base_yaw,
                roll=float(camera_relative["roll_degrees"]),
            ),
        )
        sensor_blueprint = utility.sensor_blueprint(
            world,
            SENSOR_TYPES[args.sensor],
            width,
            height,
            float(camera["fov_degrees"]),
            fixed_delta,
        )
        sensor = world.spawn_actor(
            sensor_blueprint,
            relative_transform,
            attach_to=ego,
            attachment_type=carla.AttachmentType.Rigid,
        )
        sensor.listen(sensor_queue.put)
        actors.append(sensor)

        warmup = 0
        deadline = time.monotonic() + 45.0
        while warmup < 5:
            try:
                sensor_queue.get(timeout=0.25)
                warmup += 1
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{args.sensor} camera warmup timed out")

        utility.apply_settings(
            world,
            {
                "synchronous_mode": True,
                "fixed_delta_seconds": simulation_delta,
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
                utility.await_frame(sensor_queue, frame, args.sensor, timeout_seconds=2.0)
                phase_aligned = True
                break
            except TimeoutError:
                continue
        if not phase_aligned:
            raise TimeoutError(f"could not phase-align the {args.sensor} camera")

        def apply_scene_pose(scenario: dict[str, Any], pose: dict[str, Any]) -> None:
            ego.set_transform(pose["ego_transform"])
            target.set_transform(pose["target_transform"])
            if scenario.get("occluder"):
                value = scenario["occluder"]
                occluder_xy = world_xy(
                    np.asarray([value["forward_m"], value["right_m"]]),
                    center_xy,
                    forward_xy,
                    right_xy,
                )
                waypoint = world.get_map().get_waypoint(
                    carla.Location(
                        x=float(occluder_xy[0]), y=float(occluder_xy[1]), z=0.0
                    ),
                    project_to_road=True,
                    lane_type=carla.LaneType.Any,
                )
                if waypoint is None:
                    raise RuntimeError("occluder has no surface waypoint")
                occluder.set_transform(
                    carla.Transform(
                        carla.Location(
                            x=float(occluder_xy[0]),
                            y=float(occluder_xy[1]),
                            z=float(waypoint.transform.location.z)
                            + float(value["surface_offset_m"]),
                        ),
                        carla.Rotation(yaw=float(value["world_yaw_degrees"])),
                    )
                )
            else:
                occluder.set_transform(hidden_occluder)
            ego.apply_control(carla.WalkerControl(speed=0.0))
            target.apply_control(carla.WalkerControl(speed=0.0))

        summaries: list[dict[str, Any]] = []
        contracts_by_family = {
            str(value["family"]): value for value in protocol["twin_contracts"]
        }
        for scenario in protocol["scenarios"]:
            scenario_id = str(scenario["id"])
            episode_dir = sensor_root / scenario_id
            episode_dir.mkdir()
            payload_dir = episode_dir / (
                "flow_visual" if args.sensor == "flow" else f"{args.sensor}_raw"
            )
            payload_dir.mkdir()
            records: list[dict[str, Any]] = []
            flow_frames: list[np.ndarray] = []
            reset_pose = scenario_pose(
                scenario,
                0.0,
                center_xy=center_xy,
                forward_xy=forward_xy,
                right_xy=right_xy,
                origin_z=origin_z,
                camera_base_yaw_degrees=camera_base_yaw,
            )
            apply_scene_pose(scenario, reset_pose)
            # Two stationary, aligned sensor frames clear teleport/reset motion.
            # This is essential for flow and harmless for the other modalities.
            for _ in range(2):
                world.tick(30.0)
                apply_scene_pose(scenario, reset_pose)
                reset_frame = world.tick(30.0)
                utility.await_frame(sensor_queue, reset_frame, args.sensor)
            for sample_index in range(frame_count):
                time_s = sample_index * fixed_delta
                pose = scenario_pose(
                    scenario,
                    time_s,
                    center_xy=center_xy,
                    forward_xy=forward_xy,
                    right_xy=right_xy,
                    origin_z=origin_z,
                    camera_base_yaw_degrees=camera_base_yaw,
                )
                apply_scene_pose(scenario, pose)
                world.tick(30.0)
                apply_scene_pose(scenario, pose)
                world_frame = world.tick(30.0)
                image = utility.await_frame(sensor_queue, world_frame, args.sensor)
                if len(image.raw_data) <= 0:
                    raise RuntimeError(f"empty {args.sensor} payload at {scenario_id}/{sample_index}")
                snapshot = world.get_snapshot()
                ego_state, _, _ = utility.actor_state(ego, snapshot)
                target_state, target_transform, _ = utility.actor_state(target, snapshot)
                ego_state["command_velocity"] = vector_dict(pose["ego_velocity_xy"])
                ego_state["body_yaw_degrees"] = float(pose["body_yaw_degrees"])
                ego_state["sensor_yaw_degrees"] = float(pose["sensor_yaw_degrees"])
                ego_state["head_yaw_offset_degrees"] = float(pose["head_yaw_offset_degrees"])
                target_state["command_velocity"] = vector_dict(pose["target_velocity_xy"])
                target_state["command_yaw_degrees"] = float(pose["target_yaw_degrees"])
                target_polygon = utility.polygon_from_bbox(target, target_transform)
                route_distance = utility.point_polygon_distance(
                    pose["ego_xy"], target_polygon
                )
                current_contact = route_distance <= float(
                    protocol["route_contract"]["route_body_radius_m"]
                ) + 1e-9
                occluder_state = None
                occluder_route_distance = None
                occluder_current_contact = False
                if scenario.get("occluder"):
                    occluder_state, occluder_transform, _ = utility.actor_state(
                        occluder, snapshot
                    )
                    occluder_polygon = utility.polygon_from_bbox(
                        occluder, occluder_transform
                    )
                    occluder_route_distance = utility.point_polygon_distance(
                        pose["ego_xy"], occluder_polygon
                    )
                    occluder_current_contact = occluder_route_distance <= float(
                        protocol["route_contract"]["route_body_radius_m"]
                    ) + 1e-9
                file_name = f"{sample_index:06d}.png"
                payload_path = payload_dir / file_name
                if args.sensor == "flow":
                    flow = np.frombuffer(image.raw_data, dtype=np.float32).reshape(
                        (height, width, 2)
                    )
                    flow_frames.append(flow.copy())
                    utility.save_flow_visualization(image, payload_path)
                else:
                    image.save_to_disk(str(payload_path))
                observation = None
                if args.sensor == "instance":
                    observation = {
                        "target_instance": instance_metrics(image, target.id),
                        "occluder_instance": (
                            instance_metrics(image, occluder.id)
                            if scenario.get("occluder")
                            else None
                        ),
                    }
                records.append(
                    {
                        "sample_index": sample_index,
                        "time_s": time_s,
                        "world_frame": int(world_frame),
                        "sensor": args.sensor,
                        "sensor_path": str(payload_path.relative_to(episode_dir)).replace("\\", "/"),
                        "sensor_payload_bytes": len(image.raw_data),
                        "camera_transform": utility.transform_dict(image.transform),
                        "ego": ego_state,
                        "target": target_state,
                        "occluder": occluder_state,
                        "observation": observation,
                        "truth": {
                            "target_obb_polygon_xy": target_polygon,
                            "route_body_to_target_obb_distance_m": float(route_distance),
                            "current_contact": bool(current_contact),
                            "occluder_route_body_distance_m": (
                                float(occluder_route_distance)
                                if occluder_route_distance is not None
                                else None
                            ),
                            "occluder_current_contact": bool(
                                occluder_current_contact
                            ),
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
                    and value["truth"]["current_contact"]
                ]
                record["truth"]["future_contact_within_horizon"] = bool(future_contacts)
                record["truth"]["realized_time_to_contact_seconds"] = (
                    float(future_contacts[0]["time_s"]) - current_time
                    if future_contacts
                    else None
                )
            flow_path = None
            if flow_frames:
                flow_path = episode_dir / "flow_raw_float32.npz"
                np.savez_compressed(
                    flow_path,
                    sample_indices=np.arange(frame_count, dtype=np.int32),
                    flow_xy=np.stack(flow_frames, axis=0),
                )
            frames_path = episode_dir / "frames.jsonl"
            write_jsonl(frames_path, records)
            contact_times = [
                float(value["time_s"])
                for value in records
                if value["truth"]["current_contact"]
            ]
            summary = {
                "scenario_id": scenario_id,
                "family": scenario["family"],
                "twin": scenario["twin"],
                "sensor": args.sensor,
                "frames": frame_count,
                "expected_critical": bool(scenario["expected_critical"]),
                "observed_critical": bool(contact_times),
                "first_contact_time_s": contact_times[0] if contact_times else None,
                "last_contact_time_s": contact_times[-1] if contact_times else None,
                "minimum_route_distance_m": min(
                    float(value["truth"]["route_body_to_target_obb_distance_m"])
                    for value in records
                ),
                "target_visible_frames": (
                    sum(
                        bool(value["observation"]["target_instance"]["visible"])
                        for value in records
                    )
                    if args.sensor == "instance"
                    else None
                ),
                "visibility_window_visible_frames": (
                    sum(
                        bool(value["observation"]["target_instance"]["visible"])
                        for value in records
                        if float(value["time_s"])
                        >= float(
                            contracts_by_family[scenario["family"]][
                                "visibility_gate"
                            ]["window_start_s"]
                        )
                        - 1e-9
                        and float(value["time_s"])
                        < float(
                            contracts_by_family[scenario["family"]][
                                "visibility_gate"
                            ]["window_end_s"]
                        )
                        - 1e-9
                    )
                    if args.sensor == "instance"
                    and scenario["family"] == "same_scene_different_visibility"
                    else None
                ),
                "occluder_route_contact_frames": sum(
                    bool(value["truth"]["occluder_current_contact"])
                    for value in records
                ),
            }
            summary_path = episode_dir / "summary.json"
            write_json(summary_path, summary)
            manifest = {
                "scenario_id": scenario_id,
                "sensor": args.sensor,
                "frames": frame_count,
                "frames_path": str(frames_path),
                "frames_sha256": sha256_file(frames_path),
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "payload_count": len(list(payload_dir.glob("*.png"))),
                "flow_path": str(flow_path) if flow_path else None,
                "flow_sha256": sha256_file(flow_path) if flow_path else None,
            }
            write_json(episode_dir / "manifest.json", manifest)
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)

        checks = {
            "all_scenarios_captured": len(summaries) == len(protocol["scenarios"]),
            "all_frame_counts_match": all(value["frames"] == frame_count for value in summaries),
            "all_expected_critical_match": all(
                value["expected_critical"] == value["observed_critical"]
                for value in summaries
            ),
            "all_physical_occluders_off_route": all(
                int(value["occluder_route_contact_frames"]) == 0
                for value in summaries
            ),
        }
        if args.sensor == "instance":
            by_id = {value["scenario_id"]: value for value in summaries}
            visibility_gate = contracts_by_family[
                "same_scene_different_visibility"
            ]["visibility_gate"]
            visibility_drop = (
                int(by_id["visibility_forward"]["visibility_window_visible_frames"])
                - int(
                    by_id["visibility_head_away"][
                        "visibility_window_visible_frames"
                    ]
                )
            )
            checks["head_yaw_changes_visibility"] = (
                visibility_drop
                >= int(visibility_gate["minimum_visible_frame_drop"])
            )
            occlusion_gate = contracts_by_family[
                "same_target_visible_then_occluded"
            ]["visibility_gate"]
            checks["physical_occluder_changes_visibility"] = (
                int(by_id["occlusion_blocked"]["target_visible_frames"])
                <= int(by_id["occlusion_clear"]["target_visible_frames"])
                - int(occlusion_gate["minimum_visible_frame_drop"])
            )
        result = {
            "schema_version": "dtr-carla-c0-modality-capture-v1",
            "experiment_id": protocol["experiment_id"],
            "status": (
                "DTR_CARLA_C0_MODALITY_CAPTURE_COMPLETE"
                if all(checks.values())
                else "DTR_CARLA_C0_MODALITY_CAPTURE_NOT_EVALUABLE"
            ),
            "sensor": args.sensor,
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "capture_script_sha256": sha256_file(Path(__file__).resolve()),
            "checks": checks,
            "episodes": summaries,
        }
        write_json(sensor_root / "result.json", result)
        return 0 if all(checks.values()) else 2
    finally:
        if sensor is not None:
            try:
                if sensor.is_listening:
                    sensor.stop()
            except Exception:
                pass
        if actors:
            try:
                client.apply_batch_sync(
                    [carla.command.DestroyActor(actor.id) for actor in actors],
                    do_tick=bool(world.get_settings().synchronous_mode),
                )
            except Exception as exc:
                print(f"WARNING actor cleanup failed: {exc}", file=sys.stderr)
        try:
            utility.apply_settings(world, original_settings)
            world.set_weather(original_weather)
        except Exception as exc:
            print(f"WARNING world restore failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
