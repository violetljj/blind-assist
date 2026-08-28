"""Run the R7-P causal raw-LiDAR occupancy-flow Development ceiling.

The flow ledger is generated from current and past raw JRDB LiDAR plus causal
ego poses before evaluator labels are opened.  It contains dynamic occupied BEV
cells ``(forward, left, velocity_forward, velocity_left)``.  Temporal matching
uses only voxel-component correspondence; evaluator physical IDs never enter
the correspondence or velocity estimate.

After the ledger is hash sealed, native 3-D labels are used only for
current-frame spatial attribution and scoring on the already opened 143-frame,
three-event, nine-dropout Development canary.  R2 and R6-P are carried forward
from the sealed R6 result without changing their matcher or metrics.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import platform
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dtr_r0 import CausalFrame, DTRConfig, Prediction, Signal, Vec2
from dtr_r1 import RiskEventLifecycle, _first_tube_entry_s
from dtr_r2 import DTRR2Arm, FROZEN_R2_CONFIG
from dtr_r5_dropout_canary import (
    ACTIVE_SIGNALS,
    DROPOUT_DURATIONS_S,
    SegmentCase,
    base_urgent,
    cases_from_tracks,
    dropout_frames,
    metrics_for_run,
    ratio,
    sample_pose,
    sensor_observation,
)
from dtr_r6_metric_occupancy_canary import _pointcloud_xyz
from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    FIRST_FRAME,
    HORIZON_S,
    LAST_FRAME,
    ROUTE_HALF_WIDTH_M,
    SEQUENCE,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
    stamp_ns,
    yaw_from_q,
)
from jrdb_sensor_geometry_bridge import (
    SensorSample,
    load_truth_and_associate,
    read_jsonl,
    write_json,
)


SCHEMA = "blindassist-dtr-r7-causal-occupancy-flow-canary-v1"
LEDGER_SCHEMA = "blindassist-dtr-r7-truth-blind-occupancy-flow-ledger-v1"
CLAIM_CEILING = "CURATED_PUBLIC_REAL_INDUCED_DROPOUT_DEVELOPMENT_CANARY_ONLY"
LIDAR_MAX_AGE_S = 0.10
MINIMUM_CLOSING_SPEED_MPS = 0.05
REQUIRED_RECOVERIES = 7
FALSE_SEGMENT_FACTOR = 1.10


@dataclass(frozen=True)
class FlowConfig:
    history_target_s: float = 0.35
    history_min_s: float = 0.25
    history_max_s: float = 0.45
    voxel_size_m: float = 0.12
    roi_forward_m: tuple[float, float] = (-1.0, 12.0)
    roi_left_m: tuple[float, float] = (-8.0, 8.0)
    roi_height_m: tuple[float, float] = (-0.60, 1.80)
    component_connectivity_cells: int = 1
    minimum_component_cells: int = 3
    maximum_component_cells: int = 240
    maximum_component_extent_m: float = 2.40
    minimum_shape_overlap: float = 0.20
    minimum_size_ratio: float = 0.25
    maximum_size_ratio: float = 4.0
    minimum_dynamic_speed_mps: float = 0.25
    maximum_dynamic_speed_mps: float = 3.50
    association_margin_cells: float = math.sqrt(2.0) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FROZEN_FLOW_CONFIG = FlowConfig()


@dataclass(frozen=True)
class Component:
    keys: frozenset[tuple[int, int]]
    centroid_x_m: float
    centroid_y_m: float
    extent_x_m: float
    extent_y_m: float


@dataclass(frozen=True)
class FlowLedger:
    frames: Any
    offsets: Any
    forward_m: Any
    left_m: Any
    velocity_forward_mps: Any
    velocity_left_mps: Any
    component_id: Any
    manifest: dict[str, Any]

    def frame_cells(self, frame: int) -> tuple[Any, Any, Any, Any, Any]:
        import numpy as np

        index = int(np.searchsorted(self.frames, frame))
        require(index < len(self.frames) and int(self.frames[index]) == frame, f"flow_frame_missing:{frame}")
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return (
            self.forward_m[start:stop],
            self.left_m[start:stop],
            self.velocity_forward_mps[start:stop],
            self.velocity_left_mps[start:stop],
            self.component_id[start:stop],
        )


@dataclass(frozen=True)
class ArmRun:
    predictions: tuple[Prediction, ...]
    flow_available_frames: int = 0
    flow_risk_frames: int = 0
    attributed_flow_cells: int = 0
    flow_entry_s: tuple[float, ...] = ()


def ledger_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(output.stem + ".occupancy-flow.npz"),
        output.with_name(output.stem + ".occupancy-flow.json"),
    )


def atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(partial, **arrays)
    os.replace(partial, path)


def _load_yaml(path: Path) -> dict[str, Any]:
    from ruamel.yaml import YAML

    payload = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"yaml_root_not_mapping:{path}")
    return payload


def _nearest_frame(target_ns: Sequence[int], value_ns: int) -> int | None:
    if not target_ns:
        return None
    position = bisect.bisect_left(target_ns, value_ns)
    candidates = []
    if position < len(target_ns):
        candidates.append(position)
    if position > 0:
        candidates.append(position - 1)
    return min(candidates, key=lambda index: abs(target_ns[index] - value_ns))


def _pose_xy(pose: dict[str, Any]) -> tuple[float, float, float]:
    return float(pose["x_m"]), float(pose["y_m"]), float(pose["yaw_rad"])


def _causal_pose(samples: Sequence[dict[str, Any]], target_ns: int) -> dict[str, Any]:
    times = [int(row["timestamp_ns"]) for row in samples]
    index = bisect.bisect_right(times, target_ns) - 1
    require(index >= 0, f"causal_pose_unavailable:{target_ns}")
    row = samples[index]
    age_s = (target_ns - int(row["timestamp_ns"])) / 1e9
    require(age_s <= 0.10 + 1e-12, f"causal_pose_stale:{age_s}")
    return {
        "x_m": float(row["translation"][0]),
        "y_m": float(row["translation"][1]),
        "yaw_rad": yaw_from_q(row["quaternion_xyzw"]),
        "causal_age_s": age_s,
    }


def _ego_to_world(points: Any, pose: dict[str, Any]) -> Any:
    import numpy as np

    x_m, y_m, yaw = _pose_xy(pose)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    output = np.asarray(points, dtype=np.float64).copy()
    forward = output[:, 0].copy()
    left = output[:, 1].copy()
    output[:, 0] = x_m + cosine * forward - sine * left
    output[:, 1] = y_m + sine * forward + cosine * left
    return output


def _world_to_ego_xy(world_xy: Any, pose: dict[str, Any]) -> Any:
    import numpy as np

    x_m, y_m, yaw = _pose_xy(pose)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    delta = np.asarray(world_xy, dtype=np.float64) - np.asarray([x_m, y_m])
    return np.column_stack(
        [cosine * delta[:, 0] + sine * delta[:, 1], -sine * delta[:, 0] + cosine * delta[:, 1]]
    )


def _rotate_world_velocity_to_ego(world_xy: Any, pose: dict[str, Any]) -> Any:
    import numpy as np

    _x_m, _y_m, yaw = _pose_xy(pose)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    values = np.asarray(world_xy, dtype=np.float64)
    return np.column_stack(
        [cosine * values[:, 0] + sine * values[:, 1], -sine * values[:, 0] + cosine * values[:, 1]]
    )


def _componentize(world_xyz: Any, pose: dict[str, Any], config: FlowConfig) -> list[Component]:
    import numpy as np

    if len(world_xyz) == 0:
        return []
    local_xy = _world_to_ego_xy(world_xyz[:, :2], pose)
    keep = (
        (local_xy[:, 0] >= config.roi_forward_m[0])
        & (local_xy[:, 0] <= config.roi_forward_m[1])
        & (local_xy[:, 1] >= config.roi_left_m[0])
        & (local_xy[:, 1] <= config.roi_left_m[1])
        & (world_xyz[:, 2] >= config.roi_height_m[0])
        & (world_xyz[:, 2] <= config.roi_height_m[1])
    )
    selected = world_xyz[keep, :2]
    if len(selected) == 0:
        return []
    grid = np.floor(selected / config.voxel_size_m).astype(np.int32)
    keys = set(map(tuple, np.unique(grid, axis=0).tolist()))
    components = []
    radius = config.component_connectivity_cells
    neighbor_offsets = [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if dx != 0 or dy != 0
    ]
    while keys:
        start = keys.pop()
        pending = deque([start])
        values = [start]
        while pending:
            current = pending.popleft()
            for dx, dy in neighbor_offsets:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in keys:
                    keys.remove(neighbor)
                    pending.append(neighbor)
                    values.append(neighbor)
        if not config.minimum_component_cells <= len(values) <= config.maximum_component_cells:
            continue
        array = np.asarray(values, dtype=np.float64)
        extent = (array.max(axis=0) - array.min(axis=0) + 1.0) * config.voxel_size_m
        if float(max(extent)) > config.maximum_component_extent_m:
            continue
        centers = (array + 0.5) * config.voxel_size_m
        centroid = centers.mean(axis=0)
        components.append(
            Component(
                keys=frozenset(values),
                centroid_x_m=float(centroid[0]),
                centroid_y_m=float(centroid[1]),
                extent_x_m=float(extent[0]),
                extent_y_m=float(extent[1]),
            )
        )
    return components


def _translated_overlap(previous: Component, current: Component, shift: tuple[int, int]) -> float:
    shifted = {(x + shift[0], y + shift[1]) for x, y in previous.keys}
    overlap = len(shifted & current.keys)
    return overlap / max(1, min(len(previous.keys), len(current.keys)))


def _match_components(
    previous: Sequence[Component],
    current: Sequence[Component],
    span_s: float,
    config: FlowConfig,
) -> list[tuple[int, int, float, float, float]]:
    candidates = []
    for current_index, right in enumerate(current):
        for previous_index, left in enumerate(previous):
            size_ratio = len(right.keys) / len(left.keys)
            if not config.minimum_size_ratio <= size_ratio <= config.maximum_size_ratio:
                continue
            dx = right.centroid_x_m - left.centroid_x_m
            dy = right.centroid_y_m - left.centroid_y_m
            speed = math.hypot(dx, dy) / span_s
            if speed > config.maximum_dynamic_speed_mps:
                continue
            shift = (
                int(round(dx / config.voxel_size_m)),
                int(round(dy / config.voxel_size_m)),
            )
            overlap = _translated_overlap(left, right, shift)
            if overlap + 1e-12 < config.minimum_shape_overlap:
                continue
            size_similarity = min(size_ratio, 1.0 / size_ratio)
            distance_fraction = speed / config.maximum_dynamic_speed_mps
            score = 0.65 * overlap + 0.25 * size_similarity - 0.10 * distance_fraction
            candidates.append((score, current_index, previous_index, dx, dy, overlap))
    matches = []
    used_current: set[int] = set()
    used_previous: set[int] = set()
    for _score, current_index, previous_index, dx, dy, overlap in sorted(candidates, reverse=True):
        if current_index in used_current or previous_index in used_previous:
            continue
        used_current.add(current_index)
        used_previous.add(previous_index)
        speed = math.hypot(dx, dy) / span_s
        if config.minimum_dynamic_speed_mps <= speed <= config.maximum_dynamic_speed_mps:
            matches.append((current_index, previous_index, dx / span_s, dy / span_s, overlap))
    return matches


def _history_index(times_s: Sequence[float], index: int, config: FlowConfig) -> int | None:
    current = times_s[index]
    candidates = [
        prior
        for prior in range(index)
        if config.history_min_s <= current - times_s[prior] <= config.history_max_s
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda prior: abs((current - times_s[prior]) - config.history_target_s))


def _sensor_to_ego(calibration_dir: Path) -> dict[str, Any]:
    import numpy as np

    values = _load_yaml(calibration_dir / "lidars.yaml")["lidar"]
    upper = np.asarray(values["upper2ego"], dtype=np.float64)
    lower_to_upper = np.asarray(values["lower2upper"], dtype=np.float64)
    return {"upper": upper, "lower": upper @ lower_to_upper}


def materialize_flow_ledger(
    *,
    bag_path: Path,
    timestamps_path: Path,
    calibration_dir: Path,
    output_path: Path,
    manifest_path: Path,
    config: FlowConfig = FROZEN_FLOW_CONFIG,
) -> dict[str, Any]:
    import numpy as np
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    timestamps = load_image_timestamps(timestamps_path)
    frames = np.asarray(sorted(timestamps), dtype=np.int32)
    target_ns = [round(timestamps[int(frame)] * 1e9) for frame in frames]
    frame_by_index = {index: int(frame) for index, frame in enumerate(frames)}
    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    frame_poses = {
        int(frame): _causal_pose(poses, target_ns[index])
        for index, frame in enumerate(frames)
    }
    lidar_topics = {
        "upper_velodyne/velodyne_points": "upper",
        "lower_velodyne/velodyne_points": "lower",
    }
    cloud_candidates: dict[str, list[tuple[int, Any, str]]] = {"upper": [], "lower": []}
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag_path) as reader:
        selected_connections = [
            connection
            for connection in reader.connections
            if connection.topic.lstrip("/") in lidar_topics
        ]
        require(len(selected_connections) == len(lidar_topics), "raw_lidar_topic_missing")
        for connection in selected_connections:
            if connection.msgtype not in typestore.fielddefs:
                typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        first = target_ns[0] - round(LIDAR_MAX_AGE_S * 1e9)
        last = target_ns[-1]
        for connection, _bag_time, raw in reader.messages(connections=selected_connections):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            timestamp_ns = stamp_ns(message.header.stamp)
            if first <= timestamp_ns <= last:
                sensor = lidar_topics[connection.topic.lstrip("/")]
                cloud_candidates[sensor].append(
                    (timestamp_ns, message, hashlib.sha256(bytes(message.data)).hexdigest())
                )

    selected_clouds: dict[tuple[int, str], tuple[int, Any, str]] = {}
    lidar_ages = []
    for sensor, values in cloud_candidates.items():
        values.sort(key=lambda item: item[0])
        times = [item[0] for item in values]
        for index, frame_ns in enumerate(target_ns):
            selected_index = bisect.bisect_right(times, frame_ns) - 1
            if selected_index < 0:
                continue
            value = values[selected_index]
            age_s = (frame_ns - value[0]) / 1e9
            if age_s <= LIDAR_MAX_AGE_S + 1e-9:
                selected_clouds[(frame_by_index[index], sensor)] = value
                lidar_ages.append(age_s)

    transforms = _sensor_to_ego(calibration_dir)
    decoded: dict[tuple[str, int], Any] = {}
    world_clouds = []
    z_samples = []
    for frame_index, frame_value in enumerate(frames, start=1):
        frame = int(frame_value)
        parts = []
        for sensor in ("upper", "lower"):
            selected = selected_clouds.get((frame, sensor))
            if selected is None:
                continue
            key = (sensor, selected[0])
            if key not in decoded:
                xyz = _pointcloud_xyz(selected[1])
                homogeneous = np.concatenate(
                    [xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1
                ).T
                ego_xyz = (transforms[sensor] @ homogeneous)[:3].T
                sweep_pose = _causal_pose(poses, selected[0])
                decoded[key] = _ego_to_world(ego_xyz, sweep_pose)
            parts.append(decoded[key])
        world = np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.float64)
        world_clouds.append(world)
        if len(world):
            z_samples.append(world[:: max(1, len(world) // 1000), 2])
        if frame_index % 20 == 0 or frame_index == len(frames):
            print(json.dumps({"r7_cloud_frames": frame_index, "total": len(frames)}), flush=True)

    components = [
        _componentize(world, frame_poses[int(frame)], config)
        for frame, world in zip(frames, world_clouds)
    ]
    times_s = [float(timestamps[int(frame)]) for frame in frames]
    forward_rows = []
    left_rows = []
    velocity_forward_rows = []
    velocity_left_rows = []
    component_rows = []
    offsets = [0]
    frame_counts = {}
    match_counts = {}
    overlap_values = []
    speed_values = []
    history_spans = []
    for index, frame_value in enumerate(frames):
        frame = int(frame_value)
        history = _history_index(times_s, index, config)
        frame_forward = []
        frame_left = []
        frame_vf = []
        frame_vl = []
        frame_component = []
        matches = []
        if history is not None:
            span_s = times_s[index] - times_s[history]
            history_spans.append(span_s)
            matches = _match_components(components[history], components[index], span_s, config)
            pose = frame_poses[frame]
            for local_component_id, (current_index, _previous_index, vx, vy, overlap) in enumerate(matches):
                component = components[index][current_index]
                centers_world = (
                    np.asarray(sorted(component.keys), dtype=np.float64) + 0.5
                ) * config.voxel_size_m
                centers_ego = _world_to_ego_xy(centers_world, pose)
                velocity_ego = _rotate_world_velocity_to_ego(
                    np.asarray([[vx, vy]], dtype=np.float64), pose
                )[0]
                frame_forward.extend(centers_ego[:, 0].tolist())
                frame_left.extend(centers_ego[:, 1].tolist())
                frame_vf.extend([float(velocity_ego[0])] * len(centers_ego))
                frame_vl.extend([float(velocity_ego[1])] * len(centers_ego))
                frame_component.extend([local_component_id] * len(centers_ego))
                overlap_values.append(overlap)
                speed_values.append(math.hypot(vx, vy))
        forward_rows.append(np.asarray(frame_forward, dtype=np.float32))
        left_rows.append(np.asarray(frame_left, dtype=np.float32))
        velocity_forward_rows.append(np.asarray(frame_vf, dtype=np.float32))
        velocity_left_rows.append(np.asarray(frame_vl, dtype=np.float32))
        component_rows.append(np.asarray(frame_component, dtype=np.int32))
        offsets.append(offsets[-1] + len(frame_forward))
        frame_counts[f"{frame:06d}"] = len(frame_forward)
        match_counts[f"{frame:06d}"] = len(matches)

    arrays = {
        "frames": frames,
        "offsets": np.asarray(offsets, dtype=np.int64),
        "forward_m": np.concatenate(forward_rows),
        "left_m": np.concatenate(left_rows),
        "velocity_forward_mps": np.concatenate(velocity_forward_rows),
        "velocity_left_mps": np.concatenate(velocity_left_rows),
        "component_id": np.concatenate(component_rows),
    }
    atomic_npz(output_path, **arrays)
    z_values = np.concatenate(z_samples) if z_samples else np.empty(0)

    def summary(values: Sequence[float]) -> dict[str, float | None]:
        if len(values) == 0:
            return {"minimum": None, "median": None, "maximum": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "minimum": float(array.min()),
            "median": float(np.median(array)),
            "maximum": float(array.max()),
        }

    manifest = {
        "schema_version": LEDGER_SCHEMA,
        "truth_blind": True,
        "sequence": SEQUENCE,
        "frames": {"first": FIRST_FRAME, "last": LAST_FRAME, "count": len(frames)},
        "causal_inputs": "current/past raw upper+lower Velodyne and odom->base_link pose only",
        "forbidden_inputs": [
            "evaluator physical ID",
            "native 2-D or 3-D labels",
            "future LiDAR",
            "future ego pose",
            "RGB detector or tracker output",
        ],
        "temporal_association": "truth-blind ego-compensated BEV voxel-component correspondence",
        "config": config.to_dict(),
        "source": {
            "bag": str(bag_path),
            "bag_sha256": sha256_file(bag_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "calibration_dir": str(calibration_dir),
            "calibration_sha256": sha256_file(calibration_dir / "lidars.yaml"),
            "selected_lidar_payload_sha256": {
                f"{frame:06d}/{sensor}": row[2]
                for (frame, sensor), row in sorted(selected_clouds.items())
            },
            "bag_authority": bag_authority,
        },
        "diagnostics": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "selected_sweeps": len(selected_clouds),
            "lidar_age_s": summary(lidar_ages),
            "history_span_s": summary(history_spans),
            "causal_frame_pose_age_s": summary(
                [float(frame_poses[int(frame)]["causal_age_s"]) for frame in frames]
            ),
            "sampled_raw_z_m": summary(z_values),
            "candidate_components_by_frame": {
                f"{int(frame):06d}": len(value)
                for frame, value in zip(frames, components)
            },
            "dynamic_components_by_frame": match_counts,
            "dynamic_cells_by_frame": frame_counts,
            "shape_overlap": summary(overlap_values),
            "dynamic_speed_mps": summary(speed_values),
            "dynamic_cells_total": int(len(arrays["forward_m"])),
            "frames_with_dynamic_cells": sum(value > 0 for value in frame_counts.values()),
        },
        "ledger": str(output_path),
        "ledger_sha256": sha256_file(output_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_flow_ledger(path: Path, manifest_path: Path) -> FlowLedger:
    import numpy as np

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == LEDGER_SCHEMA, "flow_ledger_manifest_schema")
    require(manifest.get("truth_blind") is True, "flow_ledger_not_truth_blind")
    require(
        manifest.get("config")
        == json.loads(json.dumps(FROZEN_FLOW_CONFIG.to_dict(), sort_keys=True)),
        "flow_ledger_config_drift",
    )
    require(sha256_file(path) == manifest["ledger_sha256"], "flow_ledger_hash_drift")
    values = np.load(path, allow_pickle=False)
    frames = values["frames"]
    require(len(frames) == LAST_FRAME - FIRST_FRAME + 1, "flow_ledger_frame_count")
    require(int(frames[0]) == FIRST_FRAME and int(frames[-1]) == LAST_FRAME, "flow_ledger_frame_range")
    return FlowLedger(
        frames=frames,
        offsets=values["offsets"],
        forward_m=values["forward_m"],
        left_m=values["left_m"],
        velocity_forward_mps=values["velocity_forward_mps"],
        velocity_left_mps=values["velocity_left_mps"],
        component_id=values["component_id"],
        manifest=manifest,
    )


def _entry_s(forward: float, left: float, velocity_forward: float, velocity_left: float) -> float | None:
    return _first_tube_entry_s(
        Vec2(forward, left),
        Vec2(velocity_forward, velocity_left),
        ROUTE_HALF_WIDTH_M,
        HORIZON_S,
        MINIMUM_CLOSING_SPEED_MPS,
    )


def attributed_entries(ledger: FlowLedger, sample: SensorSample) -> tuple[int, list[float]]:
    import numpy as np

    forward, left, velocity_forward, velocity_left, _component = ledger.frame_cells(sample.frame_index)
    if len(forward) == 0:
        return 0, []
    margin = FROZEN_FLOW_CONFIG.association_margin_cells * FROZEN_FLOW_CONFIG.voxel_size_m
    within = np.hypot(forward - sample.forward_m, left - sample.left_m) <= sample.truth_radius_m + margin
    indices = np.nonzero(within)[0]
    entries = []
    for index in indices:
        entry = _entry_s(
            float(forward[index]),
            float(left[index]),
            float(velocity_forward[index]),
            float(velocity_left[index]),
        )
        if entry is not None:
            entries.append(entry)
    return int(len(indices)), entries


def run_flow_arm(case: SegmentCase, dropout: set[int], ledger: FlowLedger) -> ArmRun:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    runner = DTRR2Arm(config)
    origin = case.samples[0].time_s
    base_predictions = []
    for sample in case.samples:
        real = sensor_observation(sample)
        observation = None if sample.frame_index in dropout and real is not None else real
        frame = CausalFrame(
            time_s=sample.time_s - origin,
            ego_pose=sample_pose(sample),
            observations=() if observation is None else (observation,),
            person_detection_count=int(observation is not None),
        )
        base_predictions.append(runner.step(frame))

    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    fused = []
    available = 0
    risk_frames = 0
    cell_count = 0
    entry_values = []
    guard_boundary_s = config.route_horizon_s * FROZEN_R2_CONFIG.imminent_horizon_fraction
    for sample, base in zip(case.samples, base_predictions):
        attributed, entries = attributed_entries(ledger, sample)
        available += int(attributed > 0)
        cell_count += attributed
        entry = min(entries) if entries else None
        if entry is not None:
            entry_values.append(entry)
        flow_risk = entry is not None
        risk_frames += int(flow_risk)
        raw_alert = True if flow_risk else base.raw_alert
        urgent = bool(flow_risk and entry <= guard_boundary_s + 1e-9)
        urgent = urgent or base_urgent(base, guard_boundary_s)
        fused.append(
            Prediction(
                time_s=base.time_s,
                signal=lifecycle.update(base.time_s, raw_alert, urgent=urgent),
                raw_alert=raw_alert,
                reason="raw_lidar_occupancy_flow" if flow_risk else base.reason,
                track_id="occupancy-flow" if flow_risk else base.track_id,
                diagnostic={
                    **base.diagnostic,
                    "flow_attributed_cells": attributed,
                    "flow_entry_s": entry if entry is not None else "none",
                    "flow_temporal_association": "voxel_component_correspondence",
                },
            )
        )
    return ArmRun(
        tuple(fused),
        flow_available_frames=available,
        flow_risk_frames=risk_frames,
        attributed_flow_cells=cell_count,
        flow_entry_s=tuple(entry_values),
    )


def evaluate_original(cases: Sequence[SegmentCase], ledger: FlowLedger) -> dict[str, Any]:
    from jrdb_native_ceiling import ArmAccumulator

    accumulator = ArmAccumulator()
    diagnostics = {"flow_available_frames": 0, "flow_risk_frames": 0, "attributed_flow_cells": 0}
    entries = []
    for case in cases:
        current = run_flow_arm(case, set(), ledger)
        accumulator.merge(metrics_for_run(case, current))
        diagnostics["flow_available_frames"] += current.flow_available_frames
        diagnostics["flow_risk_frames"] += current.flow_risk_frames
        diagnostics["attributed_flow_cells"] += current.attributed_flow_cells
        entries.extend(current.flow_entry_s)
    diagnostics["minimum_flow_entry_s"] = min(entries) if entries else None
    return {**accumulator.to_dict(include_escalation=True), **diagnostics}


def evaluate_stress(cases: Sequence[SegmentCase], ledger: FlowLedger) -> dict[str, Any]:
    from jrdb_native_ceiling import ArmAccumulator

    output = {}
    for duration_s in DROPOUT_DURATIONS_S:
        accumulator = ArmAccumulator()
        counts = {
            "dropout_window_alerted_trials": 0,
            "dropout_window_known_evidence_frames": 0,
            "dropout_window_frames": 0,
            "flow_available_frames": 0,
            "flow_risk_frames": 0,
            "attributed_flow_cells": 0,
        }
        trial_rows = []
        for case in cases:
            for event_index, event in enumerate(case.events):
                dropped = dropout_frames(case.samples, event, duration_s)
                current = run_flow_arm(case, dropped, ledger)
                event_case = SegmentCase(
                    case.label_id,
                    case.segment_index,
                    case.samples,
                    case.truth,
                    case.known,
                    (event,),
                )
                metrics = metrics_for_run(event_case, current)
                accumulator.merge(metrics)
                indices = [
                    index for index, sample in enumerate(case.samples) if sample.frame_index in dropped
                ]
                alerted = any(
                    current.predictions[index].signal in ACTIVE_SIGNALS
                    and current.predictions[index].raw_alert is True
                    for index in indices
                )
                known_evidence = sum(
                    current.predictions[index].raw_alert is not None for index in indices
                )
                counts["dropout_window_alerted_trials"] += int(alerted)
                counts["dropout_window_known_evidence_frames"] += known_evidence
                counts["dropout_window_frames"] += len(indices)
                counts["flow_available_frames"] += current.flow_available_frames
                counts["flow_risk_frames"] += current.flow_risk_frames
                counts["attributed_flow_cells"] += current.attributed_flow_cells
                trial_rows.append(
                    {
                        "label_id": case.label_id,
                        "event_index": event_index,
                        "category": event.category,
                        "event_start_frame": case.samples[event.start_index].frame_index,
                        "contact_frame": case.samples[event.contact_index].frame_index,
                        "dropout_frames": sorted(dropped),
                        "dropout_window_alerted": alerted,
                        "flow_risk_frames": current.flow_risk_frames,
                        "flow_available_frames": current.flow_available_frames,
                        "minimum_flow_entry_s": min(current.flow_entry_s) if current.flow_entry_s else None,
                        "metrics": metrics.to_dict(include_escalation=True),
                    }
                )
        trials = len(trial_rows)
        counts["dropout_window_alert_recall"] = ratio(counts["dropout_window_alerted_trials"], trials)
        counts["dropout_window_known_evidence_rate"] = ratio(
            counts["dropout_window_known_evidence_frames"], counts["dropout_window_frames"]
        )
        counts["recovered_track_only_window_misses"] = counts["dropout_window_alerted_trials"]
        counts["recovery_of_track_only_window_misses"] = ratio(
            counts["recovered_track_only_window_misses"], trials
        )
        output[f"{duration_s:.1f}"] = {
            "duration_s": duration_s,
            "trials": trials,
            "track_only_dropout_window_misses": trials,
            "occupancy_flow": {**accumulator.to_dict(include_escalation=True), **counts},
            "by_trial": trial_rows,
        }
    return output


def _segments(frames: Sequence[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(frames))
    if not ordered:
        return []
    output = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            output.append((start, previous))
            start = frame
        previous = frame
    output.append((start, previous))
    return output


def global_nuisance(cases: Sequence[SegmentCase], ledger: FlowLedger) -> dict[str, Any]:
    risk_frames = []
    cell_counts = {}
    for frame in range(FIRST_FRAME, LAST_FRAME + 1):
        forward, left, vf, vl, _component = ledger.frame_cells(frame)
        risky = 0
        for values in zip(forward, left, vf, vl):
            risky += int(_entry_s(*(float(value) for value in values)) is not None)
        cell_counts[f"{frame:06d}"] = risky
        if risky:
            risk_frames.append(frame)
    positive_frames = set()
    for case in cases:
        for index, is_positive in enumerate(case.truth):
            if is_positive is True:
                positive_frames.add(case.samples[index].frame_index)
    segments = _segments(risk_frames)
    false_segments = [
        segment
        for segment in segments
        if not any(frame in positive_frames for frame in range(segment[0], segment[1] + 1))
    ]
    return {
        "definition": "global truth-blind flow-risk timeline; false if a segment overlaps no known positive target frame",
        "risk_frames": len(risk_frames),
        "risk_frame_rate": ratio(len(risk_frames), LAST_FRAME - FIRST_FRAME + 1),
        "risk_segments": len(segments),
        "false_segments": len(false_segments),
        "segments": [{"first_frame": left, "last_frame": right} for left, right in segments],
        "false_segment_ranges": [
            {"first_frame": left, "last_frame": right} for left, right in false_segments
        ],
        "risky_cells_by_frame": cell_counts,
    }


def gate(
    r6_result: dict[str, Any],
    original_flow: dict[str, Any],
    stress_flow: dict[str, Any],
    nuisance: dict[str, Any],
) -> dict[str, Any]:
    baseline = r6_result["original_cohort"]["track_only"]
    recovered = sum(
        row["occupancy_flow"]["recovered_track_only_window_misses"]
        for row in stress_flow.values()
    )
    total = sum(row["trials"] for row in stress_flow.values())
    false_limit = baseline["false_alert_segments"] * FALSE_SEGMENT_FACTOR
    checks = {
        "all_nine_track_only_misses_preserved": total == 9,
        "recovers_at_least_seven_of_nine": recovered >= REQUIRED_RECOVERIES,
        "critical_event_recall_not_lower": (
            original_flow["critical_event_recall"] is not None
            and baseline["critical_event_recall"] is not None
            and original_flow["critical_event_recall"] >= baseline["critical_event_recall"]
        ),
        "event_f1_not_lower": (
            original_flow["event_detection_f1"] is not None
            and baseline["event_detection_f1"] is not None
            and original_flow["event_detection_f1"] >= baseline["event_detection_f1"]
        ),
        "target_aware_false_segments_within_ten_percent": (
            original_flow["false_alert_segments"] <= false_limit + 1e-9
        ),
        "global_flow_false_segments_not_above_r2": (
            nuisance["false_segments"] <= baseline["false_alert_segments"]
        ),
    }
    passed = all(checks.values())
    return {
        "verdict": (
            "R7_P_CAUSAL_OCCUPANCY_FLOW_DEVELOPMENT_GATE_MET_OPEN_R8"
            if passed
            else "R7_P_CAUSAL_OCCUPANCY_FLOW_DEVELOPMENT_GATE_NOT_MET_NO_R8"
        ),
        "passed": passed,
        "checks": checks,
        "recovered_window_misses": recovered,
        "total_track_only_window_misses": total,
        "dropout_recovery": ratio(recovered, total),
        "false_segment_limit": false_limit,
        "r8_authorized": passed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    r6_result_path = args.r6_result.resolve(strict=True)
    known_tracks_path = args.known_height_tracks.resolve(strict=True)
    labels_path = args.labels_zip.resolve(strict=True)
    timestamps_path = args.timestamps_zip.resolve(strict=True)
    bag_path = args.bag.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    r6_result = json.loads(r6_result_path.read_text(encoding="utf-8"))
    require(
        sha256_file(known_tracks_path) == r6_result["source"]["known_height_tracks_sha256"],
        "known_height_tracks_hash_drift",
    )
    flow_path, flow_manifest_path = ledger_paths(args.output.resolve())
    if args.reuse_flow_ledger and flow_path.exists() and flow_manifest_path.exists():
        ledger = load_flow_ledger(flow_path, flow_manifest_path)
    else:
        materialize_flow_ledger(
            bag_path=bag_path,
            timestamps_path=timestamps_path,
            calibration_dir=calibration_dir,
            output_path=flow_path,
            manifest_path=flow_manifest_path,
        )
        ledger = load_flow_ledger(flow_path, flow_manifest_path)

    # Evaluator labels are opened only after the complete truth-blind flow ledger is sealed.
    sensor_rows = read_jsonl(known_tracks_path)
    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    timestamps = load_image_timestamps(timestamps_path)
    context = {
        frame: {
            "image_time_s": timestamps[frame],
            "pose": interpolate_pose(poses, round(timestamps[frame] * 1e9)),
        }
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    tracks, geometry_quality = load_truth_and_associate(labels_path, sensor_rows, context)
    cases = cases_from_tracks(tracks)
    original_flow = evaluate_original(cases, ledger)
    stress_flow = evaluate_stress(cases, ledger)
    nuisance = global_nuisance(cases, ledger)
    gate_result = gate(r6_result, original_flow, stress_flow, nuisance)
    return {
        "schema_version": SCHEMA,
        "status": "DTR_R7_P_CAUSAL_OCCUPANCY_FLOW_CANARY_COMPLETE",
        "claim_ceiling": CLAIM_CEILING,
        "question": (
            "After detector/track dropout, can current-and-past raw LiDAR alone show dynamic "
            "occupancy whose measured flow enters the frozen wearer route tube within 0-3 s?"
        ),
        "frozen": {
            "r2": FROZEN_R2_CONFIG.to_dict(),
            "route_horizon_s": HORIZON_S,
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "dropout_durations_s": list(DROPOUT_DURATIONS_S),
            "flow": FROZEN_FLOW_CONFIG.to_dict(),
            "development_gate": {
                "minimum_recovered_windows": REQUIRED_RECOVERIES,
                "target_aware_false_segment_factor": FALSE_SEGMENT_FACTOR,
                "critical_event_recall_not_lower": True,
                "event_f1_not_lower": True,
                "global_flow_false_segments_not_above_r2": True,
            },
        },
        "arms": {
            "r2": "frozen detector/track only",
            "r6_static_lidar": "sealed R6-P current/past raw-LiDAR static metric occupancy",
            "r7_p_occupancy_flow": "truth-blind causal raw-LiDAR BEV occupancy flow",
        },
        "flow_source": ledger.manifest,
        "source": {
            "dataset": "JRDB public train split",
            "sequence": SEQUENCE,
            "window": {"first_frame": FIRST_FRAME, "last_frame": LAST_FRAME},
            "r6_result": str(r6_result_path),
            "r6_result_sha256": sha256_file(r6_result_path),
            "known_height_tracks": str(known_tracks_path),
            "known_height_tracks_sha256": sha256_file(known_tracks_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "bag": str(bag_path),
            "bag_sha256": sha256_file(bag_path),
            "bag_authority": bag_authority,
            "evaluable_target_segments": len(cases),
            "critical_events": sum(len(case.events) for case in cases),
        },
        "evaluator_firewall": {
            "flow_generation": "truth-blind and hash sealed before labels are loaded",
            "temporal_association": "voxel-component correspondence only; no evaluator physical ID",
            "current_frame_attribution": (
                "native 3-D target center/radius used only after ledger seal to attribute cells for scoring"
            ),
            "future_truth": "native future contact used only for metrics after ledger seal",
            "geometry_quality": geometry_quality,
        },
        "original_cohort": {
            "r2": r6_result["original_cohort"]["track_only"],
            "r6_static_lidar": r6_result["original_cohort"]["lidar_metric"],
            "r7_p_occupancy_flow": original_flow,
        },
        "stress_by_duration_s": {
            duration: {
                "duration_s": value["duration_s"],
                "trials": value["trials"],
                "r2": r6_result["stress_by_duration_s"][duration]["arms"]["track_only"],
                "r6_static_lidar": r6_result["stress_by_duration_s"][duration]["arms"]["lidar_metric"],
                "r7_p_occupancy_flow": value["occupancy_flow"],
                "r7_by_trial": value["by_trial"],
            }
            for duration, value in stress_flow.items()
        },
        "global_flow_nuisance": nuisance,
        "gate": gate_result,
        "limitations": [
            "One transparently curated 143-frame Development window with three events and nine repeated induced-dropout trials.",
            "The three durations reuse the same three events and are not nine independent natural events.",
            "Voxel-component correspondence is a classical privileged ceiling, not a trained or deployable RGB model.",
            "Evaluator 3-D geometry is used only for current-frame metric attribution after the truth-blind flow ledger is sealed.",
            "Hard occupied cells are not calibrated collision probabilities.",
            "No source-disjoint generalization, Android runtime, user benefit, product reliability, or safety performance is established.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6-result", type=Path, required=True)
    parser.add_argument("--known-height-tracks", type=Path, required=True)
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--timestamps-zip", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-flow-ledger", action="store_true")
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "output_must_be_json")
    result = run(args)
    write_json(args.output.resolve(), result)
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
