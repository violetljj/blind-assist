"""Run the DTR-M1-O causal native-box point-velocity oracle ceiling.

This is a privileged scorer-side ceiling on the sealed JRDB R7-P Development
window.  It does not estimate motion from occupancy cells.  Instead, current
raw LiDAR points inside a native 3-D box receive the causal piecewise-rigid
velocity implied by that box and its latest admissible past instance.  Point
velocities are robustly reduced to the frozen R7 BEV cells, then passed to the
unchanged R7 route-risk, R2 fusion, lifecycle, dropout stress, and evaluator.

The oracle answers only whether a selective point-wise velocity source can make
the frozen downstream computation reachable.  It is label-dependent, is not a
deployable estimator, and can authorize M1-T estimator work but never R8.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import platform
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dtr_m0_r7_error_attribution import (
    CAUSE_ATTRIBUTION,
    PROVENANCE_INHERITED,
    _base_predictions,
    _false_segments,
    _flow_detail,
    _target_velocity,
    classify_provenance,
)
from dtr_r5_dropout_canary import cases_from_tracks
from dtr_r6_metric_occupancy_canary import _pointcloud_xyz
from dtr_r7_occupancy_flow_canary import (
    FROZEN_FLOW_CONFIG,
    FlowLedger,
    _causal_pose,
    _ego_to_world,
    _rotate_world_velocity_to_ego,
    _sensor_to_ego,
    _sweep_pose,
    _world_to_ego_xy,
    atomic_npz,
    evaluate_original,
    evaluate_stress,
    global_nuisance,
    run_flow_arm,
)
from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    BASE_LINK_FROM_LOGICAL_RGB360_X_M,
    BASE_LINK_FROM_LOGICAL_RGB360_Y_M,
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
from jrdb_sensor_geometry_bridge import load_truth_and_associate, read_jsonl, write_json


SCHEMA = "blindassist-dtr-m1-causal-point-velocity-oracle-v1"
LEDGER_SCHEMA = "blindassist-dtr-m1-causal-point-velocity-oracle-ledger-v1"
CLAIM_CEILING = "PRIVILEGED_LABEL_DERIVED_ORACLE_ON_CONSUMED_JRDB_DEVELOPMENT_COHORT"
LIDAR_MAX_AGE_S = 0.10
MAXIMUM_FLOW_INDUCED_FALSE_SEGMENTS = 2
MAXIMUM_SURVIVING_M0_DIAGNOSTIC_SEGMENTS = 2


@dataclass(frozen=True)
class NativeBox:
    frame: int
    time_s: float
    label_id: str
    center_forward_m: float
    center_left_m: float
    center_z_m: float
    length_m: float
    width_m: float
    height_m: float
    yaw_ego_rad: float
    ego_x_m: float
    ego_y_m: float
    ego_yaw_rad: float

    @property
    def center_world_xy(self) -> tuple[float, float]:
        cosine = math.cos(self.ego_yaw_rad)
        sine = math.sin(self.ego_yaw_rad)
        return (
            self.ego_x_m
            + cosine * self.center_forward_m
            - sine * self.center_left_m,
            self.ego_y_m
            + sine * self.center_forward_m
            + cosine * self.center_left_m,
        )

    @property
    def yaw_world_rad(self) -> float:
        return self.ego_yaw_rad + self.yaw_ego_rad


def ledger_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(output.stem + ".point-velocity-oracle.npz"),
        output.with_name(output.stem + ".point-velocity-oracle.json"),
    )


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    import numpy as np

    if not values:
        return {"minimum": None, "median": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def load_native_boxes(
    labels_path: Path,
    timestamps: dict[int, float],
    frame_poses: dict[int, dict[str, Any]],
    *,
    sequence: str = SEQUENCE,
) -> dict[int, list[NativeBox]]:
    with zipfile.ZipFile(labels_path) as bundle:
        values = json.loads(bundle.read(f"labels/labels_3d/{sequence}.json"))["labels"]
    output: dict[int, list[NativeBox]] = {}
    for frame in sorted(timestamps):
        pose = frame_poses[frame]
        boxes = []
        for item in values[f"{frame:06d}.pcd"]:
            if bool(item.get("attributes", {}).get("no_eval", False)):
                continue
            box = item["box"]
            boxes.append(
                NativeBox(
                    frame=frame,
                    time_s=float(timestamps[frame]),
                    label_id=str(item["label_id"]),
                    center_forward_m=(
                        float(box["cx"]) + BASE_LINK_FROM_LOGICAL_RGB360_X_M
                    ),
                    center_left_m=(
                        float(box["cy"]) + BASE_LINK_FROM_LOGICAL_RGB360_Y_M
                    ),
                    center_z_m=float(box["cz"]),
                    length_m=float(box["l"]),
                    width_m=float(box["w"]),
                    height_m=float(box["h"]),
                    yaw_ego_rad=float(box["rot_z"]),
                    ego_x_m=float(pose["x_m"]),
                    ego_y_m=float(pose["y_m"]),
                    ego_yaw_rad=float(pose["yaw_rad"]),
                )
            )
        output[frame] = boxes
    return output


def _box_history(
    boxes_by_frame: dict[int, list[NativeBox]],
) -> dict[tuple[int, str], NativeBox]:
    history: dict[tuple[int, str], NativeBox] = {}
    seen: dict[str, list[NativeBox]] = {}
    for frame in sorted(boxes_by_frame):
        for current in boxes_by_frame.get(frame, ()):
            candidates = [
                previous
                for previous in seen.get(current.label_id, [])
                if FROZEN_FLOW_CONFIG.history_min_s
                <= current.time_s - previous.time_s
                <= FROZEN_FLOW_CONFIG.history_max_s
            ]
            if candidates:
                history[(frame, current.label_id)] = min(
                    candidates,
                    key=lambda previous: abs(
                        (current.time_s - previous.time_s)
                        - FROZEN_FLOW_CONFIG.history_target_s
                    ),
                )
            seen.setdefault(current.label_id, []).append(current)
    return history


def load_world_clouds(
    *,
    bag_path: Path,
    timestamps_path: Path,
    calibration_dir: Path,
    timestamps_override: dict[int, float] | None = None,
) -> tuple[
    Any,
    dict[int, float],
    dict[int, dict[str, Any]],
    list[Any],
    dict[str, Any],
]:
    import numpy as np
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    timestamps = (
        load_image_timestamps(timestamps_path)
        if timestamps_override is None
        else dict(timestamps_override)
    )
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
    candidates: dict[str, list[tuple[int, Any, str]]] = {"upper": [], "lower": []}
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
                candidates[sensor].append(
                    (timestamp_ns, message, hashlib.sha256(bytes(message.data)).hexdigest())
                )

    selected_clouds: dict[tuple[int, str], tuple[int, Any, str]] = {}
    lidar_ages = []
    for sensor, values in candidates.items():
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
    for frame_value in frames:
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
                decoded[key] = _ego_to_world(ego_xyz, _sweep_pose(poses, selected[0]))
            parts.append(decoded[key])
        world_clouds.append(
            np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.float64)
        )

    return (
        frames,
        timestamps,
        frame_poses,
        world_clouds,
        {
            "bag_authority": bag_authority,
            "selected_sweeps": len(selected_clouds),
            "lidar_age_s": _summary(lidar_ages),
            "selected_lidar_payload_sha256": {
                f"{frame:06d}/{sensor}": row[2]
                for (frame, sensor), row in sorted(selected_clouds.items())
            },
        },
    )


def _points_in_box(local_xyz: Any, box: NativeBox) -> Any:
    import numpy as np

    dx = local_xyz[:, 0] - box.center_forward_m
    dy = local_xyz[:, 1] - box.center_left_m
    cosine = math.cos(box.yaw_ego_rad)
    sine = math.sin(box.yaw_ego_rad)
    along = cosine * dx + sine * dy
    across = -sine * dx + cosine * dy
    dz = local_xyz[:, 2] - box.center_z_m
    return (
        (np.abs(along) <= 0.5 * box.length_m + 1e-9)
        & (np.abs(across) <= 0.5 * box.width_m + 1e-9)
        & (np.abs(dz) <= 0.5 * box.height_m + 1e-9)
    )


def _rigid_world_velocity(points_world_xy: Any, current: NativeBox, previous: NativeBox) -> Any:
    import numpy as np

    current_center = np.asarray(current.center_world_xy, dtype=np.float64)
    previous_center = np.asarray(previous.center_world_xy, dtype=np.float64)
    current_yaw = current.yaw_world_rad
    previous_yaw = previous.yaw_world_rad
    delta = np.asarray(points_world_xy, dtype=np.float64) - current_center
    current_cosine = math.cos(current_yaw)
    current_sine = math.sin(current_yaw)
    object_xy = np.column_stack(
        [
            current_cosine * delta[:, 0] + current_sine * delta[:, 1],
            -current_sine * delta[:, 0] + current_cosine * delta[:, 1],
        ]
    )
    previous_cosine = math.cos(previous_yaw)
    previous_sine = math.sin(previous_yaw)
    previous_world = previous_center + np.column_stack(
        [
            previous_cosine * object_xy[:, 0] - previous_sine * object_xy[:, 1],
            previous_sine * object_xy[:, 0] + previous_cosine * object_xy[:, 1],
        ]
    )
    return (np.asarray(points_world_xy, dtype=np.float64) - previous_world) / (
        current.time_s - previous.time_s
    )


def aggregate_frame(
    world_xyz: Any,
    pose: dict[str, Any],
    boxes: Sequence[NativeBox],
    history: dict[tuple[int, str], NativeBox],
    label_indices: dict[str, int],
) -> tuple[dict[str, Any], dict[str, int | float]]:
    import numpy as np

    if len(world_xyz) == 0:
        return {
            "forward": np.empty(0, dtype=np.float32),
            "left": np.empty(0, dtype=np.float32),
            "vf": np.empty(0, dtype=np.float32),
            "vl": np.empty(0, dtype=np.float32),
            "component": np.empty(0, dtype=np.int32),
            "support": np.empty(0, dtype=np.float32),
            "point_count": np.empty(0, dtype=np.int32),
        }, {"eligible_boxes": 0, "assigned_points": 0, "retained_cells": 0}

    local_xy = _world_to_ego_xy(world_xyz[:, :2], pose)
    local_xyz = np.column_stack([local_xy, world_xyz[:, 2]])
    assigned_world = []
    assigned_velocity = []
    assigned_label = []
    eligible_boxes = 0
    for box in boxes:
        previous = history.get((box.frame, box.label_id))
        if previous is None:
            continue
        eligible_boxes += 1
        indices = np.nonzero(_points_in_box(local_xyz, box))[0]
        if len(indices) == 0:
            continue
        points = world_xyz[indices]
        assigned_world.append(points[:, :2])
        assigned_velocity.append(_rigid_world_velocity(points[:, :2], box, previous))
        assigned_label.append(
            np.full(len(indices), label_indices[box.label_id], dtype=np.int32)
        )

    if not assigned_world:
        return {
            "forward": np.empty(0, dtype=np.float32),
            "left": np.empty(0, dtype=np.float32),
            "vf": np.empty(0, dtype=np.float32),
            "vl": np.empty(0, dtype=np.float32),
            "component": np.empty(0, dtype=np.int32),
            "support": np.empty(0, dtype=np.float32),
            "point_count": np.empty(0, dtype=np.int32),
        }, {"eligible_boxes": eligible_boxes, "assigned_points": 0, "retained_cells": 0}

    points_xy = np.concatenate(assigned_world)
    velocity_world = np.concatenate(assigned_velocity)
    labels = np.concatenate(assigned_label)
    grid = np.floor(points_xy / FROZEN_FLOW_CONFIG.voxel_size_m).astype(np.int32)
    keys = np.column_stack([grid, labels])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    centers_world = []
    median_velocity_world = []
    point_counts = []
    components = []
    for group_index, key in enumerate(unique):
        within = inverse == group_index
        velocity = np.median(velocity_world[within], axis=0)
        speed = float(np.hypot(*velocity))
        if not (
            FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps - 1e-12
            <= speed
            <= FROZEN_FLOW_CONFIG.maximum_dynamic_speed_mps + 1e-12
        ):
            continue
        centers_world.append((key[:2].astype(np.float64) + 0.5) * FROZEN_FLOW_CONFIG.voxel_size_m)
        median_velocity_world.append(velocity)
        point_counts.append(int(within.sum()))
        components.append(int(key[2]))

    if not centers_world:
        centers_world_array = np.empty((0, 2), dtype=np.float64)
        velocity_world_array = np.empty((0, 2), dtype=np.float64)
    else:
        centers_world_array = np.asarray(centers_world, dtype=np.float64)
        velocity_world_array = np.asarray(median_velocity_world, dtype=np.float64)
    centers_ego = _world_to_ego_xy(centers_world_array, pose)
    velocity_ego = _rotate_world_velocity_to_ego(velocity_world_array, pose)
    count_array = np.asarray(point_counts, dtype=np.int32)
    return {
        "forward": centers_ego[:, 0].astype(np.float32),
        "left": centers_ego[:, 1].astype(np.float32),
        "vf": velocity_ego[:, 0].astype(np.float32),
        "vl": velocity_ego[:, 1].astype(np.float32),
        "component": np.asarray(components, dtype=np.int32),
        "support": np.ones(len(centers_ego), dtype=np.float32),
        "point_count": count_array,
    }, {
        "eligible_boxes": eligible_boxes,
        "assigned_points": int(len(points_xy)),
        "retained_cells": int(len(centers_ego)),
    }


def materialize_oracle_ledger(
    *,
    bag_path: Path,
    timestamps_path: Path,
    calibration_dir: Path,
    labels_path: Path,
    output_path: Path,
    manifest_path: Path,
    sequence: str = SEQUENCE,
    timestamps_override: dict[int, float] | None = None,
) -> dict[str, Any]:
    import numpy as np

    frames, timestamps, frame_poses, world_clouds, lidar_diagnostics = load_world_clouds(
        bag_path=bag_path,
        timestamps_path=timestamps_path,
        calibration_dir=calibration_dir,
        timestamps_override=timestamps_override,
    )
    boxes_by_frame = load_native_boxes(
        labels_path,
        timestamps,
        frame_poses,
        sequence=sequence,
    )
    history = _box_history(boxes_by_frame)
    label_indices = {
        label: index
        for index, label in enumerate(
            sorted({box.label_id for boxes in boxes_by_frame.values() for box in boxes})
        )
    }
    rows = []
    diagnostics = {}
    offsets = [0]
    for frame_value, world in zip(frames, world_clouds):
        frame = int(frame_value)
        row, frame_diagnostics = aggregate_frame(
            world,
            frame_poses[frame],
            boxes_by_frame[frame],
            history,
            label_indices,
        )
        rows.append(row)
        diagnostics[f"{frame:06d}"] = frame_diagnostics
        offsets.append(offsets[-1] + len(row["forward"]))

    arrays = {
        "frames": frames,
        "frame_time_s": np.asarray(
            [timestamps[int(frame)] for frame in frames], dtype=np.float64
        ),
        "frame_ego_x_m": np.asarray(
            [frame_poses[int(frame)]["x_m"] for frame in frames], dtype=np.float64
        ),
        "frame_ego_y_m": np.asarray(
            [frame_poses[int(frame)]["y_m"] for frame in frames], dtype=np.float64
        ),
        "frame_ego_yaw_rad": np.asarray(
            [frame_poses[int(frame)]["yaw_rad"] for frame in frames], dtype=np.float64
        ),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "forward_m": np.concatenate([row["forward"] for row in rows]),
        "left_m": np.concatenate([row["left"] for row in rows]),
        "velocity_forward_mps": np.concatenate([row["vf"] for row in rows]),
        "velocity_left_mps": np.concatenate([row["vl"] for row in rows]),
        "component_id": np.concatenate([row["component"] for row in rows]),
        "flow_support": np.concatenate([row["support"] for row in rows]),
        "source_point_count": np.concatenate([row["point_count"] for row in rows]),
    }
    atomic_npz(output_path, **arrays)
    history_spans = [
        box.time_s - previous.time_s
        for (frame, label), previous in history.items()
        for box in boxes_by_frame[frame]
        if box.label_id == label
    ]
    manifest = {
        "schema_version": LEDGER_SCHEMA,
        "oracle": True,
        "truth_blind": False,
        "sequence": sequence,
        "frames": {
            "first": int(frames[0]),
            "last": int(frames[-1]),
            "count": len(frames),
        },
        "motion_source": (
            "current raw LiDAR point plus current/native past 3-D box piecewise-rigid velocity"
        ),
        "causality": "latest admissible past native box through current frame; no future box or LiDAR",
        "support_definition": "1.0 for native-box-bound point velocities; no calibrated probability claim",
        "aggregation": "per-native-instance BEV voxel confidence-weighted median; oracle support is constant",
        "frozen_downstream": {
            "r7_flow_config": FROZEN_FLOW_CONFIG.to_dict(),
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "route_horizon_s": HORIZON_S,
        },
        "source": {
            "bag": str(bag_path),
            "bag_sha256": sha256_file(bag_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "calibration_dir": str(calibration_dir),
            "calibration_sha256": sha256_file(calibration_dir / "lidars.yaml"),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
        },
        "diagnostics": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            **lidar_diagnostics,
            "history_span_s": _summary(history_spans),
            "eligible_native_box_histories": len(history),
            "dynamic_cells_total": int(len(arrays["forward_m"])),
            "frames_with_dynamic_cells": sum(len(row["forward"]) > 0 for row in rows),
            "frame_counts": diagnostics,
            "source_point_count": _summary(arrays["source_point_count"].tolist()),
            "flow_support": {"unique": [1.0], "calibrated_probability": False},
        },
        "ledger": str(output_path),
        "ledger_sha256": sha256_file(output_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_oracle_ledger(
    path: Path,
    manifest_path: Path,
    *,
    expected_sequence: str = SEQUENCE,
    expected_frames: Sequence[int] | None = None,
) -> FlowLedger:
    import numpy as np

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == LEDGER_SCHEMA, "oracle_ledger_schema")
    require(manifest.get("oracle") is True, "oracle_flag_missing")
    require(manifest.get("truth_blind") is False, "oracle_truth_boundary_drift")
    require(sha256_file(path) == manifest["ledger_sha256"], "oracle_ledger_hash_drift")
    values = np.load(path, allow_pickle=False)
    frames = values["frames"]
    expected = (
        list(range(FIRST_FRAME, LAST_FRAME + 1))
        if expected_frames is None
        else [int(frame) for frame in expected_frames]
    )
    require(manifest.get("sequence") == expected_sequence, "oracle_sequence")
    require(frames.tolist() == expected, "oracle_frame_range")
    require("flow_support" in values.files, "flow_support_missing")
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


def motion_source_false_delta(cases: Sequence[Any], ledger: FlowLedger) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    rows = []
    for case in cases:
        baseline = _base_predictions(case)
        oracle = run_flow_arm(case, set(), ledger).predictions
        baseline_false = _false_segments(case, baseline)
        for segment in _false_segments(case, oracle):
            flow_only = [
                index
                for index in range(segment.start_index, segment.end_index + 1)
                if oracle[index].raw_alert is True and baseline[index].raw_alert is not True
            ]
            provenance = classify_provenance(segment, baseline_false, flow_only)
            counts[provenance] += 1
            rows.append(
                {
                    "label_id": case.label_id,
                    "target_segment_index": case.segment_index,
                    "first_frame": case.samples[segment.start_index].frame_index,
                    "last_frame": case.samples[segment.end_index].frame_index,
                    "provenance": provenance,
                    "point_velocity_only_frames": len(flow_only),
                }
            )
    induced = sum(value for key, value in counts.items() if key != PROVENANCE_INHERITED)
    return {
        "definition": "M1 false segments with point-velocity-only risk versus frozen R2",
        "provenance_counts": dict(sorted(counts.items())),
        "flow_induced_or_modified_false_segments": induced,
        "rows": rows,
    }


def m0_diagnostic_replay(
    cases: Sequence[Any], ledger: FlowLedger, m0: dict[str, Any]
) -> dict[str, Any]:
    case_by_key = {(case.label_id, case.segment_index): case for case in cases}
    rows = []
    for source in m0["false_segments"]:
        if source["provenance"] == PROVENANCE_INHERITED:
            continue
        case = case_by_key[(source["label_id"], int(source["target_segment_index"]))]
        sample_index = next(
            index
            for index, sample in enumerate(case.samples)
            if sample.frame_index == int(source["diagnostic_frame"])
        )
        detail = _flow_detail(ledger, case.samples[sample_index])
        target_velocity = _target_velocity(case, sample_index)
        velocity_error = None
        if detail["mean_velocity_forward_mps"] is not None and target_velocity is not None:
            velocity_error = math.hypot(
                detail["mean_velocity_forward_mps"] - target_velocity[0],
                detail["mean_velocity_left_mps"] - target_velocity[1],
            )
        rows.append(
            {
                "segment_id": source["segment_id"],
                "m0_primary_cause": source["primary_cause"],
                "diagnostic_frame": source["diagnostic_frame"],
                "m1_point_velocity_risk": detail["risk"],
                "m1_velocity_error_mps": velocity_error,
            }
        )
    attribution_errors = [
        row["m1_velocity_error_mps"]
        for row in rows
        if row["m0_primary_cause"] == CAUSE_ATTRIBUTION
        and row["m1_velocity_error_mps"] is not None
    ]
    return {
        "segments": len(rows),
        "segments_with_surviving_point_velocity_risk": sum(
            bool(row["m1_point_velocity_risk"]) for row in rows
        ),
        "attribution_or_fragmentation_velocity_error_mps": _summary(attribution_errors),
        "support_stratified_velocity_error": (
            "NOT_EVALUABLE_ORACLE_FLOW_SUPPORT_IS_CONSTANT_1"
        ),
        "rows": rows,
    }


def gate(
    r7: dict[str, Any],
    original: dict[str, Any],
    stress: dict[str, Any],
    nuisance: dict[str, Any],
    false_delta: dict[str, Any],
    diagnostic_replay: dict[str, Any],
) -> dict[str, Any]:
    baseline = r7["original_cohort"]["r2"]
    recovered = sum(
        row["occupancy_flow"]["recovered_track_only_window_misses"]
        for row in stress.values()
    )
    total = sum(row["trials"] for row in stress.values())
    checks = {
        "preserves_all_nine_dropout_recoveries": total == 9 and recovered == 9,
        "critical_event_recall_not_lower": (
            original["critical_event_recall"] is not None
            and baseline["critical_event_recall"] is not None
            and original["critical_event_recall"] >= baseline["critical_event_recall"]
        ),
        "event_f1_strictly_higher_than_r7": (
            original["event_detection_f1"] is not None
            and original["event_detection_f1"]
            > r7["original_cohort"]["r7_p_occupancy_flow"]["event_detection_f1"] + 1e-12
        ),
        "at_most_two_point_velocity_induced_false_segments": (
            false_delta["flow_induced_or_modified_false_segments"]
            <= MAXIMUM_FLOW_INDUCED_FALSE_SEGMENTS
        ),
        "at_most_two_m0_diagnostic_segments_survive": (
            diagnostic_replay["segments_with_surviving_point_velocity_risk"]
            <= MAXIMUM_SURVIVING_M0_DIAGNOSTIC_SEGMENTS
        ),
        "global_point_velocity_false_segments_not_above_r2": (
            nuisance["false_segments"] <= baseline["false_alert_segments"]
        ),
    }
    passed = all(checks.values())
    return {
        "verdict": (
            "DTR_M1_O_POINT_VELOCITY_ORACLE_CEILING_MET_OPEN_M1_T"
            if passed
            else "DTR_M1_O_POINT_VELOCITY_ORACLE_CEILING_NOT_MET_CLOSE_SCENE_FLOW_ROUTE"
        ),
        "passed": passed,
        "checks": checks,
        "dropout_recovery": {"recovered": recovered, "total": total},
        "m1_t_authorized": passed,
        "r8_authorized": False,
        "route_conditioned_forecasting_authorized": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    r7_path = args.r7_result.resolve(strict=True)
    m0_path = args.m0_result.resolve(strict=True)
    known_tracks_path = args.known_height_tracks.resolve(strict=True)
    labels_path = args.labels_zip.resolve(strict=True)
    timestamps_path = args.timestamps_zip.resolve(strict=True)
    bag_path = args.bag.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    r7 = json.loads(r7_path.read_text(encoding="utf-8"))
    m0 = json.loads(m0_path.read_text(encoding="utf-8"))
    require(
        r7["gate"]["verdict"]
        == "R7_P_CAUSAL_OCCUPANCY_FLOW_DEVELOPMENT_GATE_NOT_MET_NO_R8",
        "r7_terminal_drift",
    )
    require(
        m0["status"] == "DTR_M0_R7_READ_ONLY_ERROR_ATTRIBUTION_COMPLETE",
        "m0_status_drift",
    )
    require(m0["source"]["r7_result_sha256"] == sha256_file(r7_path), "m0_r7_hash_drift")
    require(
        sha256_file(known_tracks_path) == r7["source"]["known_height_tracks_sha256"],
        "known_height_tracks_hash_drift",
    )
    require(sha256_file(labels_path) == r7["source"]["labels_sha256"], "labels_hash_drift")
    require(
        sha256_file(timestamps_path) == r7["source"]["timestamps_sha256"],
        "timestamps_hash_drift",
    )
    require(sha256_file(bag_path) == r7["source"]["bag_sha256"], "bag_hash_drift")

    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    timestamps = load_image_timestamps(timestamps_path)
    context = {
        frame: {
            "image_time_s": timestamps[frame],
            "pose": interpolate_pose(poses, round(timestamps[frame] * 1e9)),
        }
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    tracks, geometry_quality = load_truth_and_associate(
        labels_path, read_jsonl(known_tracks_path), context
    )
    cases = cases_from_tracks(tracks)
    oracle_path, manifest_path = ledger_paths(args.output.resolve())
    if not (args.reuse_oracle_ledger and oracle_path.exists() and manifest_path.exists()):
        materialize_oracle_ledger(
            bag_path=bag_path,
            timestamps_path=timestamps_path,
            calibration_dir=calibration_dir,
            labels_path=labels_path,
            output_path=oracle_path,
            manifest_path=manifest_path,
        )
    ledger = load_oracle_ledger(oracle_path, manifest_path)
    original = evaluate_original(cases, ledger)
    stress = evaluate_stress(cases, ledger)
    nuisance = global_nuisance(cases, ledger)
    false_delta = motion_source_false_delta(cases, ledger)
    diagnostic_replay = m0_diagnostic_replay(cases, ledger, m0)
    gate_result = gate(r7, original, stress, nuisance, false_delta, diagnostic_replay)
    return {
        "schema_version": SCHEMA,
        "status": "DTR_M1_O_CAUSAL_POINT_VELOCITY_ORACLE_COMPLETE",
        "claim_ceiling": CLAIM_CEILING,
        "question": (
            "Can label-derived, temporally supported point-wise 3-D velocity make the frozen "
            "R7 downstream retain 9/9 dropout recovery while suppressing R7 pseudo-motion?"
        ),
        "frozen": {
            "r7_motion_thresholds_and_voxel": FROZEN_FLOW_CONFIG.to_dict(),
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "route_horizon_s": HORIZON_S,
            "r2_route_risk_lifecycle_and_evaluator": "unchanged from sealed R7-P",
            "maximum_flow_induced_false_segments": MAXIMUM_FLOW_INDUCED_FALSE_SEGMENTS,
            "maximum_surviving_m0_diagnostic_segments": MAXIMUM_SURVIVING_M0_DIAGNOSTIC_SEGMENTS,
        },
        "intervention": {
            "changed": "motion evidence source only",
            "source": "causal native-box piecewise-rigid point velocity on current raw LiDAR",
            "aggregation": "per-instance confidence-weighted median BEV cell velocity",
            "flow_support": "constant oracle support 1.0; not a calibrated probability",
        },
        "source": {
            "dataset": "JRDB public train split",
            "sequence": SEQUENCE,
            "window": {"first_frame": FIRST_FRAME, "last_frame": LAST_FRAME},
            "r7_result": str(r7_path),
            "r7_result_sha256": sha256_file(r7_path),
            "m0_result": str(m0_path),
            "m0_result_sha256": sha256_file(m0_path),
            "known_height_tracks": str(known_tracks_path),
            "known_height_tracks_sha256": sha256_file(known_tracks_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "bag": str(bag_path),
            "bag_sha256": sha256_file(bag_path),
            "bag_authority": bag_authority,
            "calibration_dir": str(calibration_dir),
        },
        "oracle_ledger": ledger.manifest,
        "evaluator_firewall": {
            "oracle_truth_use": (
                "native current/past 3-D boxes and identity generate point velocity before evaluation"
            ),
            "future_truth": "future contact is evaluator-only; no future box enters the oracle source",
            "geometry_quality": geometry_quality,
        },
        "original_cohort": {
            "r2": r7["original_cohort"]["r2"],
            "r7_p_occupancy_flow": r7["original_cohort"]["r7_p_occupancy_flow"],
            "m1_o_point_velocity_oracle": original,
        },
        "stress_by_duration_s": stress,
        "motion_source_false_delta": false_delta,
        "m0_diagnostic_replay": diagnostic_replay,
        "global_point_velocity_nuisance": nuisance,
        "gate": gate_result,
        "limitations": [
            "This is a label-dependent oracle on the same consumed Development cohort, not estimator performance.",
            "JRDB native boxes stand in for native point-flow authority; AV2 is a separate source and cohort.",
            "Oracle flow_support is constant 1.0, so support/error stratification is NOT_EVALUABLE here.",
            "The three durations reuse three events; 9/9 is not nine independent natural events.",
            "R8 and route-conditioned forecasting remain closed regardless of this oracle verdict.",
            "No source-disjoint, Android, product, user-benefit, or safety claim follows.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r7-result", type=Path, required=True)
    parser.add_argument("--m0-result", type=Path, required=True)
    parser.add_argument("--known-height-tracks", type=Path, required=True)
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--timestamps-zip", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-oracle-ledger", action="store_true")
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "output_must_be_json")
    result = run(args)
    write_json(args.output.resolve(), result)
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
