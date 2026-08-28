"""Iterate truth-blind, sensor-separated JRDB LiDAR rays in world coordinates.

The selection contract intentionally matches R7: for each frame and each
LiDAR independently, select the latest sweep whose header timestamp is not
later than the frame, and admit it only when its age is at most 0.10 seconds.
Raw clouds are decoded lazily and upper/lower rays are never merged here.

Inputs are limited to a raw C25 bag, frame identifiers with their timestamps,
and JRDB LiDAR calibration.  Labels, rosters, evaluator identities, and future
truth are neither accepted nor read.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Literal

from dtr_r7_occupancy_flow_canary import (
    LIDAR_MAX_AGE_S,
    _ego_to_world,
    _pointcloud_xyz,
    _sensor_to_ego,
    _sweep_pose,
)
from jrdb_rgb_bridge import read_bag_pose_and_rgb, require, stamp_ns


SensorName = Literal["upper", "lower"]
SENSOR_ORDER: tuple[SensorName, ...] = ("upper", "lower")
LIDAR_TOPICS: dict[str, SensorName] = {
    "upper_velodyne/velodyne_points": "upper",
    "lower_velodyne/velodyne_points": "lower",
}


@dataclass(frozen=True)
class RawRaySweep:
    """One independently selected sensor sweep expressed in world coordinates."""

    sensor: SensorName
    sweep_timestamp_ns: int
    sweep_payload_sha256: str
    age_s: float
    world_origin_m: Any
    world_endpoints_m: Any


@dataclass(frozen=True)
class RawRayFrame:
    """The causal upper/lower ray observations selected for one video frame."""

    frame: int
    frame_time_s: float
    frame_timestamp_ns: int
    upper: RawRaySweep | None
    lower: RawRaySweep | None

    def sensor(self, name: SensorName) -> RawRaySweep | None:
        return self.upper if name == "upper" else self.lower


@dataclass(frozen=True)
class _SweepCandidate:
    timestamp_ns: int
    message: Any
    payload_sha256: str


def _frame_rows(
    frames: Sequence[int],
    frame_time_s: Sequence[float] | Mapping[int, float],
) -> list[tuple[int, float, int]]:
    frame_values = [int(value) for value in frames]
    require(bool(frame_values), "raw_ray_frames_empty")
    require(len(set(frame_values)) == len(frame_values), "raw_ray_frames_duplicate")

    if isinstance(frame_time_s, Mapping):
        require(
            all(frame in frame_time_s for frame in frame_values),
            "raw_ray_frame_time_missing",
        )
        pairs = [(frame, float(frame_time_s[frame])) for frame in frame_values]
    else:
        time_values = [float(value) for value in frame_time_s]
        require(len(time_values) == len(frame_values), "raw_ray_frame_time_length")
        pairs = list(zip(frame_values, time_values, strict=True))

    pairs.sort(key=lambda row: row[0])
    require(all(math.isfinite(time_s) for _frame, time_s in pairs), "raw_ray_frame_time_nonfinite")
    return [(frame, time_s, round(time_s * 1e9)) for frame, time_s in pairs]


def _read_lidar_candidates(
    bag_path: Path,
    *,
    first_timestamp_ns: int,
    last_timestamp_ns: int,
) -> dict[SensorName, list[_SweepCandidate]]:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    candidates: dict[SensorName, list[_SweepCandidate]] = {
        "upper": [],
        "lower": [],
    }
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag_path) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic.lstrip("/") in LIDAR_TOPICS
        ]
        require(len(connections) == len(LIDAR_TOPICS), "raw_lidar_topic_missing")
        for connection in connections:
            if connection.msgtype not in typestore.fielddefs:
                typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))

        for connection, _bag_time, raw in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            timestamp_ns = stamp_ns(message.header.stamp)
            if first_timestamp_ns <= timestamp_ns <= last_timestamp_ns:
                sensor = LIDAR_TOPICS[connection.topic.lstrip("/")]
                candidates[sensor].append(
                    _SweepCandidate(
                        timestamp_ns=timestamp_ns,
                        message=message,
                        payload_sha256=hashlib.sha256(bytes(message.data)).hexdigest(),
                    )
                )

    for values in candidates.values():
        values.sort(key=lambda value: value.timestamp_ns)
    return candidates


def _decode_world_rays(
    *,
    sensor: SensorName,
    candidate: _SweepCandidate,
    sensor_to_ego: Any,
    poses: Sequence[dict[str, Any]],
) -> tuple[Any, Any]:
    import numpy as np

    xyz = _pointcloud_xyz(candidate.message)
    homogeneous = np.concatenate(
        [xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1
    ).T
    ego_endpoints = (sensor_to_ego @ homogeneous)[:3].T
    ego_origin = (sensor_to_ego @ np.asarray([0.0, 0.0, 0.0, 1.0]))[:3]
    pose = _sweep_pose(poses, candidate.timestamp_ns)
    world_endpoints = _ego_to_world(ego_endpoints, pose)
    world_origin = _ego_to_world(ego_origin.reshape(1, 3), pose)[0]
    require(world_endpoints.ndim == 2 and world_endpoints.shape[1] == 3, f"raw_ray_endpoint_shape:{sensor}")
    require(world_origin.shape == (3,), f"raw_ray_origin_shape:{sensor}")
    return world_origin, world_endpoints


def iter_raw_ray_frames(
    *,
    bag_path: Path,
    frames: Sequence[int],
    frame_time_s: Sequence[float] | Mapping[int, float],
    calibration_dir: Path,
) -> Iterator[RawRayFrame]:
    """Yield R7-equivalent causal LiDAR selections without merging sensors.

    Frames are emitted in ascending frame-id order, matching R7's sorted
    timestamp-key traversal.  A missing or stale sensor sweep is represented by
    ``None``.  The iterator retains at most the latest decoded world cloud for
    each sensor; consumers control whether yielded arrays remain resident.
    """

    rows = _frame_rows(frames, frame_time_s)
    target_ns = [row[2] for row in rows]
    poses, _rgb_times, _bag_authority = read_bag_pose_and_rgb(bag_path)
    candidates = _read_lidar_candidates(
        bag_path,
        first_timestamp_ns=target_ns[0] - round(LIDAR_MAX_AGE_S * 1e9),
        last_timestamp_ns=target_ns[-1],
    )
    candidate_times = {
        sensor: [value.timestamp_ns for value in values]
        for sensor, values in candidates.items()
    }
    transforms = _sensor_to_ego(calibration_dir)
    decoded: dict[SensorName, tuple[tuple[int, str], Any, Any]] = {}

    for frame, time_s, frame_ns in rows:
        selected: dict[SensorName, RawRaySweep | None] = {
            "upper": None,
            "lower": None,
        }
        for sensor in SENSOR_ORDER:
            index = bisect.bisect_right(candidate_times[sensor], frame_ns) - 1
            if index < 0:
                continue
            candidate = candidates[sensor][index]
            age_s = (frame_ns - candidate.timestamp_ns) / 1e9
            if age_s > LIDAR_MAX_AGE_S + 1e-9:
                continue

            cache_key = (candidate.timestamp_ns, candidate.payload_sha256)
            cached = decoded.get(sensor)
            if cached is None or cached[0] != cache_key:
                world_origin, world_endpoints = _decode_world_rays(
                    sensor=sensor,
                    candidate=candidate,
                    sensor_to_ego=transforms[sensor],
                    poses=poses,
                )
                decoded[sensor] = (cache_key, world_origin, world_endpoints)
            else:
                _key, world_origin, world_endpoints = cached

            selected[sensor] = RawRaySweep(
                sensor=sensor,
                sweep_timestamp_ns=candidate.timestamp_ns,
                sweep_payload_sha256=candidate.payload_sha256,
                age_s=age_s,
                world_origin_m=world_origin,
                world_endpoints_m=world_endpoints,
            )

        yield RawRayFrame(
            frame=frame,
            frame_time_s=time_s,
            frame_timestamp_ns=frame_ns,
            upper=selected["upper"],
            lower=selected["lower"],
        )


__all__ = ["RawRayFrame", "RawRaySweep", "iter_raw_ray_frames"]
