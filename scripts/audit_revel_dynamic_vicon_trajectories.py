#!/usr/bin/env python3
"""CUDA audit of REveL Dynamic Vicon helmet and sensor trajectories.

REveL records two helmet markers and the event/LiDAR sensor suite in a common
Vicon world frame.  This script decodes those source transforms, rejects the
dataset-documented origin fallback, synchronizes each helmet to the sensor,
and computes source-native relative 3D motion on CUDA.  It does not turn a
helmet marker into a user body capsule, a camera calibration into a wearable
mount, or relative range into an assistive-event label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SENSOR_TOPIC = "/vicon/event_lidar/event_lidar"
PERSON_TOPICS = ("/vicon/helmet_green/helmet_green", "/vicon/helmet_yellow/helmet_yellow")
ORIGIN_EPSILON_M = 1e-9
SYNC_MAX_DELTA_MS = 20.0
MAX_CONSECUTIVE_INTERVAL_S = 0.1
MIN_CONTINUOUS_INTERVAL_S = 0.005
MAX_CONTINUOUS_INTERVAL_S = 0.05
MAX_CONTINUOUS_WORLD_SPEED_MPS = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_topic(reader: Any, typestore: Any, topic: str) -> dict[str, Any]:
    info = reader.topics.get(topic)
    if info is None or len(info.connections) != 1:
        raise ValueError(f"expected exactly one connection for {topic}")
    connection = info.connections[0]
    timestamps: list[int] = []; positions: list[tuple[float, float, float]] = []; quaternions: list[tuple[float, float, float, float]] = []
    frame_ids: set[str] = set(); child_frame_ids: set[str] = set()
    for _, timestamp, rawdata in reader.messages(connections=[connection]):
        message = typestore.deserialize_ros1(rawdata, connection.msgtype)
        transform = message.transform
        timestamps.append(timestamp)
        positions.append((transform.translation.x, transform.translation.y, transform.translation.z))
        quaternions.append((transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w))
        frame_ids.add(message.header.frame_id); child_frame_ids.add(message.child_frame_id)
    return {
        "topic": topic, "msgtype": connection.msgtype, "timestamps_ns": np.asarray(timestamps, dtype=np.int64),
        "positions": np.asarray(positions, dtype=np.float64), "quaternions": np.asarray(quaternions, dtype=np.float64),
        "frame_ids": sorted(frame_ids), "child_frame_ids": sorted(child_frame_ids),
    }


def _nearest_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    right = np.searchsorted(reference, query, side="left")
    left = np.clip(right - 1, 0, len(reference) - 1); right = np.clip(right, 0, len(reference) - 1)
    return np.where(np.abs(reference[left] - query) <= np.abs(reference[right] - query), left, right)


def _rotation_matrix(quaternion: Any) -> Any:
    """Return world-from-local matrix for x,y,z,w quaternions on CUDA."""
    import torch

    q = quaternion / torch.linalg.vector_norm(quaternion, dim=1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(dim=1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), dim=1).reshape(-1, 3, 3)


def _stats(values: Any) -> dict[str, float]:
    import torch

    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": int(values.numel()), "min": float(values.min().item()), "median": float(values.median().item()),
        "p95": float(torch.quantile(values, .95).item()), "max": float(values.max().item()),
    }


def _topic_metrics(track: dict[str, Any], device: Any) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    position = torch.as_tensor(track["positions"], dtype=torch.float64, device=device)
    quaternion = torch.as_tensor(track["quaternions"], dtype=torch.float64, device=device)
    finite = torch.isfinite(position).all(dim=1) & torch.isfinite(quaternion).all(dim=1)
    origin = torch.linalg.vector_norm(position, dim=1) <= ORIGIN_EPSILON_M
    qnorm = torch.linalg.vector_norm(quaternion, dim=1)
    valid = finite & ~origin & (qnorm > 1e-6)
    timestamps_s = torch.as_tensor(track["timestamps_ns"], dtype=torch.float64, device=device) / 1e9
    dt = timestamps_s[1:] - timestamps_s[:-1]
    raw_pair = valid[1:] & valid[:-1] & (dt > 0) & (dt <= MAX_CONSECUTIVE_INTERVAL_S)
    raw_speed = torch.linalg.vector_norm(position[1:] - position[:-1], dim=1)[raw_pair] / dt[raw_pair]
    interval_pair = valid[1:] & valid[:-1] & (dt >= MIN_CONTINUOUS_INTERVAL_S) & (dt <= MAX_CONTINUOUS_INTERVAL_S)
    interval_speed = torch.linalg.vector_norm(position[1:] - position[:-1], dim=1)[interval_pair] / dt[interval_pair]
    continuous_pair = torch.zeros_like(dt, dtype=torch.bool)
    continuous_pair[interval_pair] = interval_speed <= MAX_CONTINUOUS_WORLD_SPEED_MPS
    return valid, continuous_pair, {
        "source_message_count": int(position.shape[0]), "frame_ids": track["frame_ids"], "child_frame_ids": track["child_frame_ids"],
        "documented_origin_fallback_count": int(origin.sum().item()), "finite_pose_count": int(finite.sum().item()),
        "valid_nonorigin_pose_count": int(valid.sum().item()), "valid_pose_fraction": float(valid.double().mean().item()),
        "quaternion_norm": _stats(qnorm[finite]), "raw_period_s": _stats(dt[dt > 0]), "raw_world_speed_mps": _stats(raw_speed),
        "continuity_filter": {
            "min_interval_s": MIN_CONTINUOUS_INTERVAL_S, "max_interval_s": MAX_CONTINUOUS_INTERVAL_S,
            "max_world_speed_mps": MAX_CONTINUOUS_WORLD_SPEED_MPS,
            "interval_eligible_pair_count": int(interval_pair.sum().item()),
            "speed_spike_rejected_pair_count": int((interval_speed > MAX_CONTINUOUS_WORLD_SPEED_MPS).sum().item()),
            "continuous_pair_count": int(continuous_pair.sum().item()),
            "continuous_world_speed_mps": _stats(torch.linalg.vector_norm(position[1:] - position[:-1], dim=1)[continuous_pair] / dt[continuous_pair]),
        },
    }


def _relative_metrics(person: dict[str, Any], sensor: dict[str, Any], person_valid: Any, person_continuous: Any, sensor_valid: Any, sensor_continuous: Any, device: Any) -> dict[str, Any]:
    import torch

    person_times = person["timestamps_ns"]; sensor_times = sensor["timestamps_ns"]
    nearest = _nearest_indices(person_times, sensor_times)
    sync_delta_ms = np.abs(sensor_times[nearest] - person_times).astype(np.float64) / 1e6
    person_position = torch.as_tensor(person["positions"], dtype=torch.float64, device=device)
    sensor_position = torch.as_tensor(sensor["positions"][nearest], dtype=torch.float64, device=device)
    sensor_quaternion = torch.as_tensor(sensor["quaternions"][nearest], dtype=torch.float64, device=device)
    aligned = person_valid & sensor_valid[torch.as_tensor(nearest, device=device)] & torch.as_tensor(sync_delta_ms <= SYNC_MAX_DELTA_MS, device=device)
    relative_world = person_position - sensor_position
    local = torch.bmm(_rotation_matrix(sensor_quaternion).transpose(1, 2), relative_world.unsqueeze(2)).squeeze(2)
    range_m = torch.linalg.vector_norm(local, dim=1)
    timestamps_s = torch.as_tensor(person_times, dtype=torch.float64, device=device) / 1e9
    dt = timestamps_s[1:] - timestamps_s[:-1]
    sensor_index_delta = nearest[1:] - nearest[:-1]
    sensor_pair_ok = torch.as_tensor(sensor_index_delta == 0, device=device)
    advance = sensor_index_delta == 1
    if advance.any():
        sensor_pair_ok[torch.as_tensor(advance, device=device)] = sensor_continuous[torch.as_tensor(nearest[:-1][advance], dtype=torch.long, device=device)]
    pair_valid = aligned[1:] & aligned[:-1] & person_continuous & sensor_pair_ok & (dt > 0)
    relative_speed = torch.linalg.vector_norm(relative_world[1:] - relative_world[:-1], dim=1)[pair_valid] / dt[pair_valid]
    radial_speed = (range_m[1:] - range_m[:-1])[pair_valid] / dt[pair_valid]
    return {
        "sync_max_delta_ms": SYNC_MAX_DELTA_MS, "synchronized_valid_pose_count": int(aligned.sum().item()),
        "synchronized_valid_pose_fraction": float(aligned.double().mean().item()),
        "timestamp_delta_ms": _stats(torch.as_tensor(sync_delta_ms, dtype=torch.float64, device=device)[aligned]),
        "sensor_local_range_m": _stats(range_m[aligned]), "sensor_relative_world_speed_mps": _stats(relative_speed),
        "continuity_filtered_relative_pair_count": int(pair_valid.sum().item()),
        "sensor_local_radial_speed_mps": _stats(radial_speed),
    }


def audit(dataset_root: Path) -> dict[str, Any]:
    import torch
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for REveL Vicon trajectory audit")
    bag = dataset_root / "dynamic.bag"
    if not bag.is_file():
        raise FileNotFoundError(f"missing bag: {bag}")
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag) as reader:
        sensor = _extract_topic(reader, typestore, SENSOR_TOPIC)
        people = [_extract_topic(reader, typestore, topic) for topic in PERSON_TOPICS]
    device = torch.device("cuda")
    sensor_valid, sensor_continuous, sensor_report = _topic_metrics(sensor, device)
    people_report: dict[str, Any] = {}
    for person in people:
        person_valid, person_continuous, metrics = _topic_metrics(person, device)
        metrics["relative_to_sensor"] = _relative_metrics(person, sensor, person_valid, person_continuous, sensor_valid, sensor_continuous, device)
        people_report[person["topic"].rsplit("/", 1)[-1]] = metrics
    report = {
        "format": "blindassist_revel_dynamic_vicon_trajectory_audit_v1",
        "source": {"bag": bag.name, "bytes": bag.stat().st_size, "sha256": _sha256(bag), "world_frame": sensor_report["frame_ids"]},
        "sensor_suite": sensor_report, "helmet_people": people_report,
        "admission": {
            "external_metric_person_sensor_trajectory_truth_admitted": True,
            "admitted_for": ["offline metric person-to-sensor trajectory", "source-native relative range and range-rate stratification"],
            "not_admitted_for": ["physical assistive TTC", "user body capsule", "body-local safe corridor", "assistive event truth", "on-device safety"],
            "reason": "Vicon tracks helmet and sensor-suite markers in an arbitrary source world; origin fallback is source-documented, and no user body envelope/device calibration/event labels are established here",
        },
        "compute_backend": {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)}, "production_authority": False,
    }
    qa = dataset_root / "qa"; qa.mkdir(exist_ok=True)
    (qa / "revel_dynamic_vicon_trajectory_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset_root)
    print(json.dumps({"device": report["compute_backend"]["device"], "people": list(report["helmet_people"]), "sha256": report["source"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
