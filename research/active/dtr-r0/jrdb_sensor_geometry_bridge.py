"""Replace privileged current JRDB 3-D centers with causal raw lidar geometry.

This fixed Development bridge reuses the sealed truth-blind RGB detector and
tracker ledger from ``jrdb_rgb_bridge.py``.  For each image frame it selects the
latest upper-Velodyne scan at or before the image timestamp, ego-motion
compensates the scan into the image-time base frame, projects it with the
official JRDB cylindrical-camera calibration, and estimates one metric point
per detector box.  Native labels are opened only after that sensor ledger is
written, for evaluator association and future event truth.

There is deliberately one estimator and no parameter sweep: the median x/y of
the closest 10 percent of projected lidar points inside the detector box, with
at least three supporting points.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence
import zipfile

from dtr_r0 import Arm, CausalFrame, DTRConfig, EgoPose, Observation, run_arm
from jrdb_native_ceiling import ArmAccumulator, future_hits, score_arm, truth_events
from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    BASE_LINK_FROM_LOGICAL_RGB360_X_M,
    BASE_LINK_FROM_LOGICAL_RGB360_Y_M,
    FIRST_FRAME,
    HORIZON_S,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LAST_FRAME,
    MINIMUM_EVALUATOR_IOU,
    PRIMARY_TARGETS,
    ROUTE_HALF_WIDTH_M,
    SCHEMA as RGB_BRIDGE_SCHEMA,
    SEQUENCE,
    associate_frame,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
)


SCHEMA = "dtr-r0-jrdb-causal-lidar-geometry-bridge-v1"
SENSOR_LEDGER_SCHEMA = "dtr-r0-jrdb-causal-lidar-track-geometry-v1"
CLAIM_CEILING = "CURATED_PUBLIC_REAL_RGB_TRACK_PLUS_CAUSAL_RAW_LIDAR_GEOMETRY_ONLY"
UPPER_LIDAR_TOPIC = "upper_velodyne/velodyne_points"
LOWER_LIDAR_TOPIC = "lower_velodyne/velodyne_points"
FOREGROUND_RANGE_QUANTILE = 0.10
MINIMUM_BOX_POINT_SUPPORT = 3
PERSON_RADIUS_M = 0.30


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
    os.replace(partial, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"jsonl_row_not_object:{line_number}")
            rows.append(value)
    return rows


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_calibration(calibration_dir: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("jrdb_sensor_geometry_bridge requires PyYAML") from error

    defaults_path = calibration_dir / "defaults.yaml"
    cameras_path = calibration_dir / "cameras.yaml"
    defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    cameras = yaml.safe_load(cameras_path.read_text(encoding="utf-8"))
    upper = defaults["calibrated"]["lidar_upper_to_rgb"]
    lower = defaults["calibrated"]["lidar_lower_to_rgb"]
    selected = {"sensor_0", "sensor_2", "sensor_4", "sensor_6", "sensor_8"}
    focal_y = []
    center_y = []
    for name, config in cameras["cameras"].items():
        if name not in selected:
            continue
        matrix = [float(value) for value in str(config["K"]).split()]
        require(len(matrix) == 9, f"camera_k_shape:{name}")
        focal_y.append(matrix[4])
        center_y.append(matrix[5])
    require(len(focal_y) == 5, "omni_camera_calibration_missing")
    return {
        "defaults_path": str(defaults_path.resolve()),
        "defaults_sha256": sha256_file(defaults_path),
        "cameras_path": str(cameras_path.resolve()),
        "cameras_sha256": sha256_file(cameras_path),
        "upper_translation_m": [float(value) for value in upper["translation"]],
        "upper_rotation_z_rad": float(upper["rotation"][-1]),
        "lower_translation_m": [float(value) for value in lower["translation"]],
        "lower_rotation_z_rad": float(lower["rotation"][-1]),
        "median_focal_y_px": statistics.median(focal_y),
        "median_center_y_px": statistics.median(center_y),
    }


@dataclass(frozen=True)
class LidarScan:
    time_ns: int
    logical_points: Any


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def decode_xyz(message: Any) -> Any:
    import numpy as np

    require(not bool(message.is_bigendian), "big_endian_pointcloud_not_supported")
    fields = {str(item.name): item for item in message.fields}
    require(all(name in fields for name in ("x", "y", "z")), "pointcloud_xyz_missing")
    require(
        all(int(fields[name].datatype) == 7 for name in ("x", "y", "z")),
        "pointcloud_xyz_not_float32",
    )
    count = int(message.width) * int(message.height)
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [
                int(fields["x"].offset),
                int(fields["y"].offset),
                int(fields["z"].offset),
            ],
            "itemsize": int(message.point_step),
        }
    )
    values = np.frombuffer(message.data, dtype=dtype, count=count)
    points = np.column_stack((values["x"], values["y"], values["z"])).astype(
        np.float64, copy=False
    )
    return points[np.all(np.isfinite(points), axis=1)]


def lidar_to_logical_rgb(
    points: Any, calibration: dict[str, Any], sensor: str = "upper"
) -> Any:
    import numpy as np

    output = np.asarray(points, dtype=np.float64).copy()
    require(sensor in {"upper", "lower"}, f"unsupported_lidar_sensor:{sensor}")
    output[:, :3] -= np.asarray(
        calibration[f"{sensor}_translation_m"], dtype=np.float64
    )
    theta = float(calibration[f"{sensor}_rotation_z_rad"])
    cosine, sine = math.cos(theta), math.sin(theta)
    x = cosine * output[:, 0] - sine * output[:, 1]
    y = sine * output[:, 0] + cosine * output[:, 1]
    output[:, 0], output[:, 1] = x, y
    return output


def read_lidar(
    bag_path: Path,
    calibration: dict[str, Any],
    minimum_ns: int,
    maximum_ns: int,
    *,
    sensor: str,
    topic: str,
    frame_id: str,
) -> tuple[list[LidarScan], dict[str, Any]]:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as error:
        raise RuntimeError("jrdb_sensor_geometry_bridge requires rosbags") from error

    typestore = get_typestore(Stores.ROS1_NOETIC)
    scans: list[LidarScan] = []
    source_messages = 0
    with Reader(bag_path) as reader:
        selected = [
            item
            for item in reader.connections
            if item.topic.lstrip("/") == topic
        ]
        require(len(selected) == 1, f"{sensor}_lidar_topic_missing_or_duplicated")
        connection = selected[0]
        if connection.msgtype not in typestore.fielddefs:
            typestore.register(
                get_types_from_msg(connection.msgdef.data, connection.msgtype)
            )
        for _connection, _bag_time, raw in reader.messages(connections=selected):
            source_messages += 1
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            current_ns = stamp_ns(message.header.stamp)
            if current_ns < minimum_ns or current_ns > maximum_ns:
                continue
            require(
                message.header.frame_id.lstrip("/") == frame_id,
                f"{sensor}_lidar_frame_drift",
            )
            scans.append(
                LidarScan(
                    time_ns=current_ns,
                    logical_points=lidar_to_logical_rgb(
                        decode_xyz(message), calibration, sensor
                    ),
                )
            )
    scans.sort(key=lambda item: item.time_ns)
    require(len(scans) >= 2, f"{sensor}_lidar_window_empty")
    require(
        all(right.time_ns > left.time_ns for left, right in zip(scans, scans[1:])),
        f"{sensor}_lidar_time_not_strict",
    )
    return scans, {
        "bag_topic_messages": source_messages,
        "window_scans": len(scans),
        "window_points": sum(len(item.logical_points) for item in scans),
    }


def read_upper_lidar(
    bag_path: Path,
    calibration: dict[str, Any],
    minimum_ns: int,
    maximum_ns: int,
) -> tuple[list[LidarScan], dict[str, Any]]:
    return read_lidar(
        bag_path,
        calibration,
        minimum_ns,
        maximum_ns,
        sensor="upper",
        topic=UPPER_LIDAR_TOPIC,
        frame_id="upper_velodyne_frame",
    )


def read_lower_lidar(
    bag_path: Path,
    calibration: dict[str, Any],
    minimum_ns: int,
    maximum_ns: int,
) -> tuple[list[LidarScan], dict[str, Any]]:
    return read_lidar(
        bag_path,
        calibration,
        minimum_ns,
        maximum_ns,
        sensor="lower",
        topic=LOWER_LIDAR_TOPIC,
        frame_id="lower_velodyne_frame",
    )


def transform_points_to_image_time(
    logical_points: Any,
    lidar_pose: dict[str, Any],
    image_pose: dict[str, Any],
) -> tuple[Any, Any]:
    """Return image-time base XY and logical XYZ for projection."""
    import numpy as np

    points = np.asarray(logical_points, dtype=np.float64)
    base_x = points[:, 0] + BASE_LINK_FROM_LOGICAL_RGB360_X_M
    base_y = points[:, 1] + BASE_LINK_FROM_LOGICAL_RGB360_Y_M
    lidar_yaw = float(lidar_pose["yaw_rad"])
    world_x = (
        float(lidar_pose["x_m"])
        + base_x * math.cos(lidar_yaw)
        - base_y * math.sin(lidar_yaw)
    )
    world_y = (
        float(lidar_pose["y_m"])
        + base_x * math.sin(lidar_yaw)
        + base_y * math.cos(lidar_yaw)
    )
    delta_x = world_x - float(image_pose["x_m"])
    delta_y = world_y - float(image_pose["y_m"])
    image_yaw = float(image_pose["yaw_rad"])
    current_base_x = delta_x * math.cos(image_yaw) + delta_y * math.sin(image_yaw)
    current_base_y = -delta_x * math.sin(image_yaw) + delta_y * math.cos(image_yaw)
    current_logical = np.column_stack(
        (
            current_base_x - BASE_LINK_FROM_LOGICAL_RGB360_X_M,
            current_base_y - BASE_LINK_FROM_LOGICAL_RGB360_Y_M,
            points[:, 2],
        )
    )
    current_base_xy = np.column_stack((current_base_x, current_base_y))
    return current_base_xy, current_logical


def project_logical_to_stitched(points: Any, calibration: dict[str, Any]) -> Any:
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    ref_x = -values[:, 1]
    ref_y = -values[:, 2]
    ref_z = values[:, 0]
    theta = (np.arctan2(ref_x, ref_z) + math.pi) % (2.0 * math.pi)
    u = (theta / (2.0 * math.pi) * IMAGE_WIDTH) % IMAGE_WIDTH
    safe = np.abs(ref_z) > 1e-6
    v = np.full_like(u, np.nan)
    v[safe] = -float(calibration["median_focal_y_px"]) * (
        ref_y[safe] * np.cos(theta[safe]) / ref_z[safe]
    ) + float(calibration["median_center_y_px"])
    return np.column_stack((u, v))


def estimate_box_geometry(
    bbox: Sequence[float],
    projected: Any,
    base_xy: Any,
) -> dict[str, Any] | None:
    import numpy as np

    x1, y1, x2, y2 = (float(value) for value in bbox)
    pixels = np.asarray(projected, dtype=np.float64)
    points = np.asarray(base_xy, dtype=np.float64)
    planar_range = np.linalg.norm(points, axis=1)
    mask = (
        np.all(np.isfinite(pixels), axis=1)
        & np.all(np.isfinite(points), axis=1)
        & (pixels[:, 0] >= x1)
        & (pixels[:, 0] <= x2)
        & (pixels[:, 1] >= y1)
        & (pixels[:, 1] <= y2)
        & (planar_range > 0.25)
    )
    candidates = points[mask]
    ranges = planar_range[mask]
    support = len(candidates)
    if support < MINIMUM_BOX_POINT_SUPPORT:
        return None
    foreground_count = max(1, int(math.ceil(support * FOREGROUND_RANGE_QUANTILE)))
    closest = candidates[np.argsort(ranges, kind="stable")[:foreground_count]]
    return {
        "forward_m": float(np.median(closest[:, 0])),
        "left_m": float(np.median(closest[:, 1])),
        "box_point_support": support,
        "foreground_point_support": foreground_count,
    }


def materialize_sensor_ledger(
    detector_rows: Sequence[dict[str, Any]],
    timestamps: dict[int, float],
    poses: Sequence[dict[str, Any]],
    scans: Sequence[LidarScan],
    calibration: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in detector_rows:
        by_frame.setdefault(int(row["frame_index"]), []).append(row)
    scan_times = [item.time_ns for item in scans]
    output = []
    lidar_ages = []
    frame_context: dict[int, dict[str, Any]] = {}
    frames_with_geometry = 0
    for frame in range(FIRST_FRAME, LAST_FRAME + 1):
        image_ns = round(timestamps[frame] * 1e9)
        scan_index = bisect_right(scan_times, image_ns) - 1
        require(scan_index >= 0, f"causal_upper_lidar_missing:{frame}")
        scan = scans[scan_index]
        age_s = (image_ns - scan.time_ns) / 1e9
        require(age_s >= 0.0, f"future_lidar_selected:{frame}")
        lidar_pose = interpolate_pose(poses, scan.time_ns)
        image_pose = interpolate_pose(poses, image_ns)
        base_xy, logical = transform_points_to_image_time(
            scan.logical_points, lidar_pose, image_pose
        )
        projected = project_logical_to_stitched(logical, calibration)
        frame_rows = 0
        for detector in by_frame.get(frame, []):
            geometry = estimate_box_geometry(
                detector["bbox_xyxy"], projected, base_xy
            )
            output.append(
                {
                    "schema": SENSOR_LEDGER_SCHEMA,
                    "sequence": SEQUENCE,
                    "frame_index": frame,
                    "image_time_s": timestamps[frame],
                    "track_id": detector["track_id"],
                    "bbox_xyxy": detector["bbox_xyxy"],
                    "confidence": detector["confidence"],
                    "image_sha256": detector["image_sha256"],
                    "upper_lidar_time_s": scan.time_ns / 1e9,
                    "upper_lidar_age_s": age_s,
                    "geometry": geometry,
                }
            )
            frame_rows += int(geometry is not None)
        frames_with_geometry += int(frame_rows > 0)
        lidar_ages.append(age_s)
        frame_context[frame] = {
            "pose": image_pose,
            "image_time_s": timestamps[frame],
        }
    geometry_rows = sum(row["geometry"] is not None for row in output)
    return output, {
        "frames": LAST_FRAME - FIRST_FRAME + 1,
        "frames_with_any_geometry": frames_with_geometry,
        "detector_track_occurrences": len(output),
        "geometry_occurrences": geometry_rows,
        "geometry_coverage": geometry_rows / len(output) if output else None,
        "causal_lidar_age_s": {
            "minimum": min(lidar_ages),
            "median": statistics.median(lidar_ages),
            "maximum": max(lidar_ages),
        },
    }, frame_context


@dataclass(frozen=True)
class SensorSample:
    frame_index: int
    time_s: float
    ego_x_m: float
    ego_y_m: float
    ego_yaw_rad: float
    forward_m: float
    left_m: float
    truth_radius_m: float
    observed_radius_m: float
    detector_track_id: str | None
    observed_forward_m: float | None
    observed_left_m: float | None

    @property
    def distance_m(self) -> float:
        return math.hypot(self.forward_m, self.left_m)

    @property
    def tube_threshold_m(self) -> float:
        return ROUTE_HALF_WIDTH_M + self.truth_radius_m


def load_truth_and_associate(
    labels_path: Path,
    sensor_rows: Sequence[dict[str, Any]],
    frame_context: dict[int, dict[str, Any]],
) -> tuple[dict[str, list[SensorSample]], dict[str, Any]]:
    with zipfile.ZipFile(labels_path) as bundle:
        labels_2d = json.loads(
            bundle.read(f"labels/labels_2d_stitched/{SEQUENCE}.json")
        )["labels"]
        labels_3d = json.loads(
            bundle.read(f"labels/labels_3d/{SEQUENCE}.json")
        )["labels"]
    sensor_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in sensor_rows:
        sensor_by_frame.setdefault(int(row["frame_index"]), []).append(row)
    tracks: dict[str, list[SensorSample]] = {}
    errors_xy = []
    errors_range = []
    matched = 0
    matched_with_geometry = 0
    primary = {
        target: {"native_frames": 0, "detector_matched_frames": 0, "sensor_geometry_frames": 0}
        for target in PRIMARY_TARGETS
    }
    for frame in range(FIRST_FRAME, LAST_FRAME + 1):
        stem = f"{frame:06d}"
        truth_2d = []
        for item in labels_2d[f"{stem}.jpg"]:
            if bool(item.get("attributes", {}).get("no_eval", False)):
                continue
            x, y, width, height = (float(value) for value in item["box"])
            if width > 0.0 and height > 0.0:
                truth_2d.append(
                    {
                        "label_id": str(item["label_id"]),
                        "bbox_xyxy": [x, y, x + width, y + height],
                    }
                )
        source = sensor_by_frame.get(frame, [])
        matches: dict[str, dict[str, Any]] = {}
        for source_index, truth_index, overlap in associate_frame(source, truth_2d):
            label_id = truth_2d[truth_index]["label_id"]
            matches[label_id] = {**source[source_index], "evaluator_iou": overlap}
            matched += 1
            matched_with_geometry += int(source[source_index]["geometry"] is not None)
            if label_id in primary:
                primary[label_id]["detector_matched_frames"] += 1
                primary[label_id]["sensor_geometry_frames"] += int(
                    source[source_index]["geometry"] is not None
                )
        context = frame_context[frame]
        pose = context["pose"]
        for item in labels_3d[f"{stem}.pcd"]:
            if bool(item.get("attributes", {}).get("no_eval", False)):
                continue
            label_id = str(item["label_id"])
            box = item["box"]
            truth_forward = float(box["cx"]) + BASE_LINK_FROM_LOGICAL_RGB360_X_M
            truth_left = float(box["cy"]) + BASE_LINK_FROM_LOGICAL_RGB360_Y_M
            truth_radius = max(
                0.15,
                0.5 * max(float(box.get("w", 0.60)), float(box.get("l", 0.60))),
            )
            match = matches.get(label_id)
            geometry = None if match is None else match["geometry"]
            if label_id in primary:
                primary[label_id]["native_frames"] += 1
            if geometry is not None:
                observed_forward = float(geometry["forward_m"])
                observed_left = float(geometry["left_m"])
                errors_xy.append(
                    math.hypot(
                        observed_forward - truth_forward,
                        observed_left - truth_left,
                    )
                )
                errors_range.append(
                    abs(
                        math.hypot(observed_forward, observed_left)
                        - math.hypot(truth_forward, truth_left)
                    )
                )
            tracks.setdefault(label_id, []).append(
                SensorSample(
                    frame_index=frame,
                    time_s=float(context["image_time_s"]),
                    ego_x_m=float(pose["x_m"]),
                    ego_y_m=float(pose["y_m"]),
                    ego_yaw_rad=float(pose["yaw_rad"]),
                    forward_m=truth_forward,
                    left_m=truth_left,
                    truth_radius_m=truth_radius,
                    observed_radius_m=PERSON_RADIUS_M,
                    detector_track_id=None if geometry is None else str(match["track_id"]),
                    observed_forward_m=(
                        None if geometry is None else float(geometry["forward_m"])
                    ),
                    observed_left_m=(
                        None if geometry is None else float(geometry["left_m"])
                    ),
                )
            )
    return tracks, {
        "detector_native_matches": matched,
        "matches_with_sensor_geometry": matched_with_geometry,
        "matched_geometry_coverage": matched_with_geometry / matched if matched else None,
        "position_error_m": {
            "median": percentile(errors_xy, 0.50),
            "p90": percentile(errors_xy, 0.90),
        },
        "range_error_m": {
            "median": percentile(errors_range, 0.50),
            "p90": percentile(errors_range, 0.90),
        },
        "primary_targets": primary,
    }


def contiguous_segments(samples: Sequence[SensorSample]) -> Iterable[list[SensorSample]]:
    current: list[SensorSample] = []
    for sample in samples:
        if current and (
            sample.frame_index != current[-1].frame_index + 1
            or sample.time_s <= current[-1].time_s
        ):
            yield current
            current = []
        current.append(sample)
    if current:
        yield current


def causal_frames(samples: Sequence[SensorSample]) -> list[CausalFrame]:
    origin = samples[0].time_s
    output = []
    for sample in samples:
        observations = ()
        if (
            sample.detector_track_id is not None
            and sample.observed_forward_m is not None
            and sample.observed_left_m is not None
        ):
            observations = (
                Observation(
                    track_id=sample.detector_track_id,
                    forward_m=sample.observed_forward_m,
                    left_m=sample.observed_left_m,
                    radius_m=sample.observed_radius_m,
                ),
            )
        output.append(
            CausalFrame(
                time_s=sample.time_s - origin,
                ego_pose=EgoPose(
                    sample.ego_x_m,
                    sample.ego_y_m,
                    sample.ego_yaw_rad,
                    sample.ego_yaw_rad,
                ),
                observations=observations,
                person_detection_count=int(bool(observations)),
            )
        )
    return output


def evaluate_tracks(tracks: dict[str, list[SensorSample]]) -> dict[str, Any]:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    pooled = {
        arm: ArmAccumulator()
        for arm in (Arm.B2_RADIAL_TTC, Arm.C_ROUTE_INTERSECTION)
    }
    by_event_target = []
    event_count = 0
    evaluable_segments = 0
    for label_id, values in sorted(tracks.items()):
        for segment_index, samples in enumerate(contiguous_segments(values)):
            if (
                len(samples) < 2
                or samples[-1].time_s - samples[0].time_s
                < config.minimum_track_span_s + HORIZON_S
            ):
                continue
            truth, contacts = future_hits(samples)
            events, known = truth_events(samples, truth, contacts, config.minimum_track_span_s)
            if not any(known):
                continue
            frames = causal_frames(samples)
            arm_rows = {}
            for arm in (Arm.B2_RADIAL_TTC, Arm.C_ROUTE_INTERSECTION):
                predictions = run_arm(frames, arm, config)
                metrics = score_arm(samples, predictions, events, known, truth)
                pooled[arm].merge(metrics)
                arm_rows[arm.value] = metrics.to_dict()
            if events:
                by_event_target.append(
                    {
                        "label_id": label_id,
                        "segment_index": segment_index,
                        "sensor_geometry_frames": sum(
                            sample.observed_forward_m is not None for sample in samples
                        ),
                        "detector_track_ids": sorted(
                            {
                                sample.detector_track_id
                                for sample in samples
                                if sample.detector_track_id is not None
                            }
                        ),
                        "events": [
                            {
                                "category": event.category,
                                "start_frame": samples[event.start_index].frame_index,
                                "contact_frame": samples[event.contact_index].frame_index,
                                "end_frame": samples[event.end_index].frame_index,
                            }
                            for event in events
                        ],
                        "arms": arm_rows,
                    }
                )
            event_count += len(events)
            evaluable_segments += 1
    pooled_rows = {arm.value: metrics.to_dict() for arm, metrics in pooled.items()}
    return {
        "evaluable_target_segments": evaluable_segments,
        "critical_events": event_count,
        "pooled": pooled_rows,
        "by_event_target": by_event_target,
        "interpretation": "numeric_sensor_bridge_observation_no_new_gate",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rgb_result_path = args.rgb_result.resolve(strict=True)
    rgb_tracks_path = args.rgb_tracks.resolve(strict=True)
    labels_path = args.labels_zip.resolve(strict=True)
    timestamps_path = args.timestamps_zip.resolve(strict=True)
    bag_path = args.bag.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    rgb_result = json.loads(rgb_result_path.read_text(encoding="utf-8"))
    require(rgb_result.get("schema_version") == RGB_BRIDGE_SCHEMA, "rgb_result_schema")
    require(
        sha256_file(rgb_tracks_path)
        == rgb_result["truth_blind_detector_tracker"]["ledger_sha256"],
        "rgb_track_ledger_hash_drift",
    )
    detector_rows = read_jsonl(rgb_tracks_path)
    timestamps = load_image_timestamps(timestamps_path)
    calibration = load_calibration(calibration_dir)
    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    image_ns = [round(timestamps[frame] * 1e9) for frame in range(FIRST_FRAME, LAST_FRAME + 1)]
    scans, scan_coverage = read_upper_lidar(
        bag_path,
        calibration,
        min(image_ns) - 200_000_000,
        max(image_ns),
    )
    sensor_rows, sensor_coverage, frame_context = materialize_sensor_ledger(
        detector_rows, timestamps, poses, scans, calibration
    )
    sensor_ledger = args.output.with_name(args.output.stem + ".sensor-tracks.jsonl").resolve()
    write_jsonl(sensor_ledger, sensor_rows)
    sensor_ledger_sha = sha256_file(sensor_ledger)

    # Evaluator-only annotation access begins after the raw sensor ledger is sealed.
    tracks, geometry_quality = load_truth_and_associate(
        labels_path, sensor_rows, frame_context
    )
    evaluation = evaluate_tracks(tracks)
    return {
        "schema_version": SCHEMA,
        "status": "DTR_R0_CAUSAL_RAW_LIDAR_GEOMETRY_OBSERVATION_AVAILABLE",
        "claim_ceiling": CLAIM_CEILING,
        "source": {
            "dataset": "JRDB public train split",
            "sequence": SEQUENCE,
            "window": {"first_frame": FIRST_FRAME, "last_frame": LAST_FRAME},
            "rgb_bridge_result": str(rgb_result_path),
            "rgb_bridge_result_sha256": sha256_file(rgb_result_path),
            "rgb_track_ledger": str(rgb_tracks_path),
            "rgb_track_ledger_sha256": sha256_file(rgb_tracks_path),
            "labels_zip": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps_zip": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "bag": str(bag_path),
            "bag_sha256": sha256_file(bag_path),
            "calibration": calibration,
            "bag_authority": bag_authority,
        },
        "truth_blind_sensor_geometry": {
            "ledger": str(sensor_ledger),
            "ledger_sha256": sensor_ledger_sha,
            "topic": UPPER_LIDAR_TOPIC,
            "temporal_rule": "latest upper lidar header timestamp <= image timestamp",
            "motion_compensation": "lidar-time base -> odom -> image-time base using causal bag TF",
            "projection": "official JRDB logical RGB360 cylindrical projection",
            "estimator": {
                "rule": "median base-frame x/y of closest range quantile inside full detector box",
                "closest_range_quantile": FOREGROUND_RANGE_QUANTILE,
                "minimum_box_point_support": MINIMUM_BOX_POINT_SUPPORT,
                "parameter_sweep": False,
            },
            "scan_coverage": scan_coverage,
            "coverage": sensor_coverage,
        },
        "privileged_evaluator": {
            "association": "current-frame detector bbox to native stitched 2-D label at IoU >= 0.30",
            "future_truth": "future native 3-D centers and body extent",
            "geometry_quality": geometry_quality,
        },
        "evaluation": evaluation,
        "limitations": [
            "This is the same single curated 143-frame Development window as the RGB bridge.",
            "Projected lidar points inside a detector box are not person segmentation and may select foreground clutter or floor.",
            "Evaluator identity plus future center/body-extent truth still use JRDB annotations; current DTR metric observations use raw lidar centers and a fixed 0.30 m person radius.",
            "Only upper Velodyne is used; there is no estimator, threshold, tracker, or route-matcher sweep.",
            "This is offline public-data evidence, not phone/Android runtime, user benefit, natural-distribution, or safety evidence.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-result", type=Path, required=True)
    parser.add_argument("--rgb-tracks", type=Path, required=True)
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--timestamps-zip", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "output_must_be_json")
    result = run(args)
    write_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "sensor_coverage": result["truth_blind_sensor_geometry"]["coverage"],
                "geometry_quality": result["privileged_evaluator"]["geometry_quality"],
                "evaluation": result["evaluation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
