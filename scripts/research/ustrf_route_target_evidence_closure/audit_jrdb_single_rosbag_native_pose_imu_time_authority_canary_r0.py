#!/usr/bin/env python3
"""Audit native pose, IMU, LiDAR and time authority in one frozen JRDB ROS1 bag."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

STAGE = "JRDB_SINGLE_ROSBAG_NATIVE_POSE_IMU_TIME_AUTHORITY_CANARY_R0"
CONFIG_SCHEMA = "blindassist_ustrf_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0_config"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_frame(value: str) -> str:
    return str(value).strip().lstrip("/")


def norm_topic(value: str) -> str:
    return str(value).strip().lstrip("/")


def stamp_ns(stamp: Any) -> int:
    seconds = getattr(stamp, "sec", getattr(stamp, "secs", None))
    nanoseconds = getattr(stamp, "nanosec", getattr(stamp, "nsecs", None))
    require(isinstance(seconds, int) and isinstance(nanoseconds, int), "unsupported_header_stamp")
    return seconds * 1_000_000_000 + nanoseconds


def vector_signature(values: list[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 9) for value in values)


def nearest_delta_seconds(samples_ns: list[int], targets_seconds: list[float]) -> dict[str, float | None]:
    samples = sorted(samples_ns)
    if not samples or not targets_seconds:
        return {"median": None, "maximum": None}
    deltas: list[float] = []
    import bisect
    for target_seconds in targets_seconds:
        target = round(target_seconds * 1_000_000_000)
        index = bisect.bisect_left(samples, target)
        choices = [samples[pos] for pos in (index - 1, index) if 0 <= pos < len(samples)]
        deltas.append(min(abs(value - target) for value in choices) / 1_000_000_000)
    return {"median": statistics.median(deltas), "maximum": max(deltas)}


def summarize(samples: list[dict[str, Any]], window: tuple[float, float], max_gap: float, tolerance: float) -> dict[str, Any]:
    headers = [int(sample["header_ns"]) for sample in samples]
    bags = [int(sample["bag_ns"]) for sample in samples]
    positive_gaps = [(b - a) / 1_000_000_000 for a, b in zip(headers, headers[1:]) if b > a]
    backward = sum(b < a for a, b in zip(headers, headers[1:]))
    duplicates = sum(b == a for a, b in zip(headers, headers[1:]))
    first = min(headers) if headers else None
    last = max(headers) if headers else None
    window_start_ns, window_end_ns = (round(value * 1_000_000_000) for value in window)
    in_window = sum(window_start_ns - round(tolerance * 1e9) <= value <= window_end_ns + round(tolerance * 1e9) for value in headers)
    return {
        "sample_count": len(samples),
        "header_first_ns": first,
        "header_last_ns": last,
        "bag_first_ns": min(bags) if bags else None,
        "bag_last_ns": max(bags) if bags else None,
        "frame_ids": sorted({sample["frame_id"] for sample in samples}),
        "child_frame_ids": sorted({sample.get("child_frame_id", "") for sample in samples if sample.get("child_frame_id")}),
        "unique_measurement_signatures": len({sample["signature"] for sample in samples}),
        "backward_header_steps": backward,
        "duplicate_header_steps": duplicates,
        "median_gap_seconds": statistics.median(positive_gaps) if positive_gaps else None,
        "maximum_gap_seconds": max(positive_gaps) if positive_gaps else None,
        "maximum_allowed_gap_seconds": max_gap,
        "maximum_abs_bag_minus_header_seconds": max((abs(a - b) / 1e9 for a, b in zip(bags, headers)), default=None),
        "covers_external_window": bool(first is not None and first <= window_start_ns + round(tolerance * 1e9) and last >= window_end_ns - round(tolerance * 1e9)),
        "samples_in_external_window": in_window,
    }


def audit(repo: Path, config_path: Path, bag_path: Path, acquisition_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["schema"] == CONFIG_SCHEMA and config["stage"] == STAGE, "config_identity")
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    require(acquisition["status"] == "ACQUIRED" and acquisition["stage"] == STAGE, "acquisition_identity")
    require(bag_path.is_file(), "bag_missing")
    bag_sha = sha256_file(bag_path)
    require(bag_sha == acquisition["bag"]["sha256"], "bag_sha_drift")
    require(bag_path.stat().st_size == acquisition["bag"]["bytes"], "bag_size_drift")
    timestamp_binding = config["local_inputs"]["timestamps"]
    timestamp_path = repo / timestamp_binding["path"]
    require(sha256_file(timestamp_path) == timestamp_binding["sha256"], "timestamp_sha_drift")
    sequence = config["canary"]["sequence"]
    first_frame = int(config["canary"]["window_first_frame"])
    last_frame = int(config["canary"]["window_last_frame"])
    with zipfile.ZipFile(timestamp_path) as bundle:
        image_rows = json.loads(bundle.read(f"timestamps/{sequence}/frames_img.json"))["data"][first_frame:last_frame + 1]
        point_rows = json.loads(bundle.read(f"timestamps/{sequence}/frames_pc.json"))["data"][first_frame:last_frame + 1]
    image_times = [float(row["timestamp"]) for row in image_rows]
    upper_times = [float(next(item for item in row["pointclouds"] if item["name"] == "upper_velodyne")["timestamp"]) for row in point_rows]
    lower_times = [float(next(item for item in row["pointclouds"] if item["name"] == "lower_velodyne")["timestamp"]) for row in point_rows]
    external_window = (min(image_times + upper_times + lower_times), max(image_times + upper_times + lower_times))

    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inventory: list[dict[str, Any]] = []
    role_cfg = config["roles"]
    with Reader(bag_path) as reader:
        for connection in reader.connections:
            inventory.append({
                "topic": connection.topic,
                "message_type": connection.msgtype,
                "message_count": connection.msgcount,
            })
        fixed_topics = {
            norm_topic(role_cfg["dynamic_tf"]["topic"]),
            norm_topic(role_cfg["upper_lidar"]["topic"]),
            norm_topic(role_cfg["lower_lidar"]["topic"]),
        }
        selected = [
            connection for connection in reader.connections
            if norm_topic(connection.topic) in fixed_topics
            or connection.msgtype in {
                role_cfg["odometry_fallback"]["message_type"],
                role_cfg["imu"]["message_type"],
            }
        ]
        for connection in selected:
            if connection.msgtype not in typestore.fielddefs:
                typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for connection, bag_ns, raw in reader.messages(connections=selected):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            if norm_topic(connection.topic) == norm_topic(role_cfg["dynamic_tf"]["topic"]):
                for transform in message.transforms:
                    if (
                        norm_frame(transform.header.frame_id) == role_cfg["dynamic_tf"]["parent_frame"]
                        and norm_frame(transform.child_frame_id) == role_cfg["dynamic_tf"]["child_frame"]
                    ):
                        value = transform.transform
                        samples["dynamic_tf"].append({
                            "bag_ns": bag_ns,
                            "header_ns": stamp_ns(transform.header.stamp),
                            "frame_id": norm_frame(transform.header.frame_id),
                            "child_frame_id": norm_frame(transform.child_frame_id),
                            "signature": vector_signature([
                                value.translation.x, value.translation.y, value.translation.z,
                                value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w,
                            ]),
                        })
            elif connection.msgtype == role_cfg["odometry_fallback"]["message_type"]:
                if norm_frame(message.header.frame_id) == role_cfg["odometry_fallback"]["parent_frame"] and norm_frame(message.child_frame_id) == role_cfg["odometry_fallback"]["child_frame"]:
                    pose = message.pose.pose
                    samples[f"odometry_fallback::{norm_topic(connection.topic)}"].append({
                        "bag_ns": bag_ns, "header_ns": stamp_ns(message.header.stamp),
                        "frame_id": norm_frame(message.header.frame_id),
                        "child_frame_id": norm_frame(message.child_frame_id),
                        "signature": vector_signature([
                            pose.position.x, pose.position.y, pose.position.z,
                            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
                        ]),
                    })
            elif connection.msgtype == role_cfg["imu"]["message_type"]:
                samples[f"imu::{norm_topic(connection.topic)}"].append({
                    "bag_ns": bag_ns, "header_ns": stamp_ns(message.header.stamp),
                    "frame_id": norm_frame(message.header.frame_id),
                    "signature": vector_signature([
                        message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z,
                        message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z,
                    ]),
                })
            else:
                for role in ("upper_lidar", "lower_lidar"):
                    if norm_topic(connection.topic) == norm_topic(role_cfg[role]["topic"]) and connection.msgtype == role_cfg[role]["message_type"]:
                        samples[role].append({
                            "bag_ns": bag_ns, "header_ns": stamp_ns(message.header.stamp),
                            "frame_id": norm_frame(message.header.frame_id),
                            "signature": (int(message.width), int(message.height), int(message.row_step)),
                        })
        recording = {
            "start_time_ns": reader.start_time,
            "end_time_ns": reader.end_time,
            "duration_seconds": reader.duration / 1e9,
            "message_count": reader.message_count,
        }

    tolerance = float(config["clock_gate"]["coverage_tolerance_seconds"])
    summaries = {
        role: summarize(samples[role], external_window, float(role_cfg[role]["max_gap_seconds"]), tolerance)
        for role in ("dynamic_tf", "upper_lidar", "lower_lidar")
    }
    odometry_candidates = {
        key.split("::", 1)[1]: summarize(values, external_window, float(role_cfg["odometry_fallback"]["max_gap_seconds"]), tolerance)
        for key, values in sorted(samples.items()) if key.startswith("odometry_fallback::")
    }
    imu_candidates = {
        key.split("::", 1)[1]: summarize(values, external_window, float(role_cfg["imu"]["max_gap_seconds"]), tolerance)
        for key, values in sorted(samples.items()) if key.startswith("imu::")
    }
    summaries["upper_lidar"]["nearest_external_delta_seconds"] = nearest_delta_seconds([x["header_ns"] for x in samples["upper_lidar"]], upper_times)
    summaries["lower_lidar"]["nearest_external_delta_seconds"] = nearest_delta_seconds([x["header_ns"] for x in samples["lower_lidar"]], lower_times)
    minimum = int(config["clock_gate"]["minimum_samples_per_role_in_window"])
    maximum_clock_delta = float(config["clock_gate"]["maximum_abs_bag_minus_header_seconds"])

    def row_ready(row: dict[str, Any]) -> bool:
        return bool(
            row["sample_count"] >= minimum
            and row["header_first_ns"] and row["covers_external_window"]
            and row["samples_in_external_window"] >= minimum
            and row["backward_header_steps"] == 0
            and row["maximum_gap_seconds"] is not None
            and row["maximum_gap_seconds"] <= row["maximum_allowed_gap_seconds"]
            and row["maximum_abs_bag_minus_header_seconds"] is not None
            and row["maximum_abs_bag_minus_header_seconds"] <= maximum_clock_delta
        )

    ready_odometry = sorted(
        topic for topic, row in odometry_candidates.items()
        if row_ready(row) and row["unique_measurement_signatures"] >= 2
    )
    tf_ready = row_ready(summaries["dynamic_tf"]) and summaries["dynamic_tf"]["unique_measurement_signatures"] >= 2
    if tf_ready:
        dynamic_pose_role = "tf:odom->base_link"
        pose_ready = True
    elif ready_odometry:
        dynamic_pose_role = f"odometry:{ready_odometry[0]}"
        pose_ready = True
    else:
        dynamic_pose_role = "none"
        pose_ready = False
    ready_imu = sorted(
        topic for topic, row in imu_candidates.items()
        if row_ready(row) and row["unique_measurement_signatures"] >= 2
    )
    selected_imu_topic = ready_imu[0] if ready_imu else None
    imu_ready = selected_imu_topic is not None
    lidar_ready: dict[str, bool] = {}
    for role in ("upper_lidar", "lower_lidar"):
        nearest = summaries[role]["nearest_external_delta_seconds"]["maximum"]
        lidar_ready[role] = bool(row_ready(summaries[role]) and nearest is not None and nearest <= float(role_cfg[role]["maximum_nearest_external_delta_seconds"]))
    clock_frame_ready = pose_ready and imu_ready and all(lidar_ready.values())
    if not pose_ready:
        terminal = "NATIVE_POSE_AUTHORITY_ABSENT"
    elif not imu_ready:
        terminal = "NATIVE_IMU_TIME_AUTHORITY_ABSENT"
    elif not clock_frame_ready:
        terminal = "NATIVE_CLOCK_FRAME_CHAIN_NOT_CLOSED"
    else:
        terminal = "NATIVE_POSE_IMU_TIME_AUTHORITY_PRESENT"
    return {
        "schema": "blindassist_ustrf_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0",
        "stage": STAGE,
        "status": "AUDIT_COMPLETE",
        "terminal_state": terminal,
        "config_sha256": sha256_file(config_path),
        "acquisition_sha256": sha256_file(acquisition_path),
        "bag": {"path": bag_path.as_posix(), "bytes": bag_path.stat().st_size, "sha256": bag_sha},
        "recording": recording,
        "external_window": {
            "frames": len(image_rows),
            "first_seconds": external_window[0],
            "last_seconds": external_window[1],
            "image_count": len(image_times),
            "upper_pointcloud_count": len(upper_times),
            "lower_pointcloud_count": len(lower_times),
        },
        "topic_inventory": sorted(inventory, key=lambda row: (row["topic"], row["message_type"])),
        "roles": summaries,
        "odometry_candidates": odometry_candidates,
        "imu_candidates": imu_candidates,
        "gates": {
            "dynamic_pose_role": dynamic_pose_role,
            "selected_imu_topic": selected_imu_topic,
            "measured_dynamic_pose": pose_ready,
            "native_imu_time": imu_ready,
            "upper_lidar_time": lidar_ready["upper_lidar"],
            "lower_lidar_time": lidar_ready["lower_lidar"],
            "clock_frame_chain": clock_frame_ready,
        },
        "authority": {
            "p1b_complete": terminal == "NATIVE_POSE_IMU_TIME_AUTHORITY_PRESENT",
            "p2_separate_goal_may_be_frozen": terminal == "NATIVE_POSE_IMU_TIME_AUTHORITY_PRESENT",
            "p2_executed": False,
            "relative_person_motion_computed": False,
            "intended_route_truth": False,
            "route_event_safety_authority": False,
            "android_human_production_authority": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo.resolve(), args.config.resolve(), args.bag.resolve(), args.acquisition.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"terminal_state": result["terminal_state"], "gates": result["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
