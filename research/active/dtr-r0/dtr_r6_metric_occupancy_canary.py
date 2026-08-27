"""Run the R6 direct-metric residual-occupancy Development canary.

R6 keeps the R5 semantic person mask, evaluator-only component association,
induced 0.2/0.4/0.8 second track dropouts, frozen R2 route matcher, and event
lifecycle.  It changes only the metric occupancy source:

* ``rgb_metric`` runs a frozen zero-shot metric-depth model on the calibrated
  undistorted JRDB perspective cameras and back-projects semantic residual
  pixels into ego-frame metric occupancy;
* ``lidar_metric`` projects current/past-only raw JRDB Velodyne returns into
  the same semantic residual and serves as a privileged metric-source ceiling.

The full per-frame metric point ledger is materialized and hash-sealed before
native 2-D identity or future 3-D contact truth is opened.  Metric points are
occupied surface samples, so route intersection uses the frozen route tube
half-width directly rather than applying the center-based person-radius
dilation a second time.

This is a curated Development stress canary, not source-disjoint, calibrated-
probability, product, user-benefit, or safety evidence.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dtr_r0 import CausalFrame, DTRConfig, EgoPose, Observation, Prediction, Signal, Vec2
from dtr_r1 import RiskEventLifecycle, _first_tube_entry_s
from dtr_r2 import FROZEN_R2_CONFIG, DTRR2Arm
from dtr_r5_dropout_canary import (
    ACTIVE_SIGNALS,
    DROPOUT_DURATIONS_S,
    EGO_HISTORY_S,
    FIRST_FRAME,
    HORIZON_S,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LAST_FRAME,
    MINIMUM_CLOSING_SPEED_MPS,
    ROUTE_HALF_WIDTH_M,
    ArmRun,
    SegmentCase,
    base_urgent,
    cases_from_tracks,
    dropout_frames,
    load_dense_ledger,
    load_truth_boxes,
    metrics_for_run,
    ratio,
    sample_pose,
    sensor_observation,
)
from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    SEQUENCE,
    associate_frame,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
    stamp_ns,
)
from jrdb_sensor_geometry_bridge import (
    SensorSample,
    load_truth_and_associate,
    read_jsonl,
    write_json,
)

SCHEMA = "blindassist-dtr-r6-direct-metric-occupancy-canary-v1"
LEDGER_SCHEMA = "blindassist-dtr-r6-truth-blind-metric-point-ledger-v1"
CLAIM_CEILING = "CURATED_PUBLIC_REAL_INDUCED_DROPOUT_DEVELOPMENT_CANARY_ONLY"
CAMERA_IDS = (6, 8, 0, 2, 4)
RGB_SAMPLE_STRIDE_PX = 4
RGB_MIN_DEPTH_M = 0.10
RGB_MAX_DEPTH_M = 20.0
LIDAR_MAX_AGE_S = 0.10
MODEL_INPUT_SIZE = 518
METRIC_ARMS = ("rgb_metric", "lidar_metric")


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: int
    distorted_k: Any
    distortion: Any
    undistorted_k: Any
    cam_to_ego: Any
    undistort_map_x: Any
    undistort_map_y: Any
    raw_to_stitch: Any


@dataclass(frozen=True)
class PointLedger:
    frames: Any
    offsets: dict[str, Any]
    stitched_y: dict[str, Any]
    stitched_x: dict[str, Any]
    forward_m: dict[str, Any]
    left_m: dict[str, Any]
    manifest: dict[str, Any]

    def frame_points(self, arm: str, frame: int) -> tuple[Any, Any, Any, Any]:
        import numpy as np

        index = int(np.searchsorted(self.frames, frame))
        require(index < len(self.frames) and int(self.frames[index]) == frame, f"ledger_frame_missing:{frame}")
        start = int(self.offsets[arm][index])
        end = int(self.offsets[arm][index + 1])
        return (
            self.stitched_y[arm][start:end],
            self.stitched_x[arm][start:end],
            self.forward_m[arm][start:end],
            self.left_m[arm][start:end],
        )


def ledger_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(output.stem + ".metric-points.npz"),
        output.with_name(output.stem + ".metric-points.json"),
    )


def atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(partial, **arrays)
    os.replace(partial, path)


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"yaml_root_not_mapping:{path}")
    return payload


def load_calibrations(calibration_dir: Path) -> dict[int, CameraCalibration]:
    import cv2
    import numpy as np

    lidar = _yaml(calibration_dir / "lidars.yaml")
    mapping_dir = calibration_dir / "indi2stitch_mappings"
    output = {}
    for camera_id in CAMERA_IDS:
        row = lidar[f"sensor_{camera_id}"]
        distorted_k = np.asarray(row["distorted_img_K"], dtype=np.float64)
        distortion = np.asarray(row["D"], dtype=np.float64)
        undistorted_k = np.asarray(row["undistorted_img_K"], dtype=np.float64)
        cam_to_ego = np.asarray(row["cam2ego"], dtype=np.float64)
        map_x, map_y = cv2.initUndistortRectifyMap(
            distorted_k,
            distortion,
            np.eye(3, dtype=np.float64),
            undistorted_k,
            (752, 480),
            cv2.CV_32FC1,
        )
        mapping_path = mapping_dir / f"indi2stitch_mapping_camera_{camera_id}.npy"
        raw_to_stitch = np.load(mapping_path)
        require(raw_to_stitch.shape == (480, 752, 10), f"mapping_shape:{camera_id}")
        output[camera_id] = CameraCalibration(
            camera_id,
            distorted_k,
            distortion,
            undistorted_k,
            cam_to_ego,
            map_x,
            map_y,
            raw_to_stitch,
        )
    return output


def _load_depth_model(source_dir: Path, checkpoint: Path) -> Any:
    import torch

    metric_source = source_dir / "metric_depth"
    require(metric_source.is_dir(), "metric_depth_source_missing")
    sys.path.insert(0, str(metric_source))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    finally:
        sys.path.pop(0)
    require(torch.cuda.is_available(), "cuda_required_for_metric_depth")
    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        max_depth=RGB_MAX_DEPTH_M,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.cuda().eval()


def _nearest_frame(target_ns: Sequence[int], value_ns: int, *, causal: bool = False) -> int | None:
    if causal:
        index = bisect.bisect_right(target_ns, value_ns) - 1
        return index if index >= 0 else None
    index = bisect.bisect_left(target_ns, value_ns)
    options = [item for item in (index - 1, index) if 0 <= item < len(target_ns)]
    if not options:
        return None
    return min(options, key=lambda item: abs(target_ns[item] - value_ns))


def _mapped_semantic_pixels(
    calibration: CameraCalibration,
    semantic_mask: Any,
    sample_y: Any,
    sample_x: Any,
) -> tuple[Any, Any, Any]:
    """Return sampled undistorted pixels and one semantic stitched location each."""
    import numpy as np

    raw_x = np.rint(calibration.undistort_map_x[sample_y, sample_x]).astype(np.int32)
    raw_y = np.rint(calibration.undistort_map_y[sample_y, sample_x]).astype(np.int32)
    inside = (raw_x >= 0) & (raw_x < 752) & (raw_y >= 0) & (raw_y < 480)
    sample_y = sample_y[inside]
    sample_x = sample_x[inside]
    raw_y = raw_y[inside]
    raw_x = raw_x[inside]
    mapping = calibration.raw_to_stitch[raw_y, raw_x].reshape(-1, 5, 2).astype(np.int32)
    mapped_y = mapping[:, :, 0]
    mapped_x = mapping[:, :, 1]
    valid = (
        (mapped_y >= 0)
        & (mapped_y < IMAGE_HEIGHT)
        & (mapped_x >= 0)
        & (mapped_x < IMAGE_WIDTH)
    )
    semantic = np.zeros(valid.shape, dtype=bool)
    rows, columns = np.nonzero(valid)
    semantic[rows, columns] = semantic_mask[mapped_y[rows, columns], mapped_x[rows, columns]]
    has_semantic = semantic.any(axis=1)
    choice = semantic.argmax(axis=1)
    rows = np.arange(len(mapping))
    return (
        sample_y[has_semantic],
        sample_x[has_semantic],
        np.stack(
            [mapped_y[rows, choice][has_semantic], mapped_x[rows, choice][has_semantic]],
            axis=1,
        ),
    )


def _rgb_metric_points(
    image_bytes: bytes,
    calibration: CameraCalibration,
    semantic_mask: Any,
    model: Any,
) -> tuple[Any, Any, Any, Any]:
    import cv2
    import numpy as np

    raw = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    require(raw is not None and raw.shape == (480, 752, 3), f"raw_camera_shape:{calibration.camera_id}")
    undistorted = cv2.remap(
        raw,
        calibration.undistort_map_x,
        calibration.undistort_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    depth = model.infer_image(undistorted, MODEL_INPUT_SIZE)
    sample_y, sample_x = np.meshgrid(
        np.arange(0, 480, RGB_SAMPLE_STRIDE_PX, dtype=np.int32),
        np.arange(0, 752, RGB_SAMPLE_STRIDE_PX, dtype=np.int32),
        indexing="ij",
    )
    sample_y = sample_y.ravel()
    sample_x = sample_x.ravel()
    sample_y, sample_x, stitched = _mapped_semantic_pixels(
        calibration, semantic_mask, sample_y, sample_x
    )
    values = depth[sample_y, sample_x].astype(np.float64)
    valid = np.isfinite(values) & (values >= RGB_MIN_DEPTH_M) & (values <= RGB_MAX_DEPTH_M)
    sample_y = sample_y[valid]
    sample_x = sample_x[valid]
    stitched = stitched[valid]
    values = values[valid]
    k = calibration.undistorted_k
    camera = np.stack(
        [
            (sample_x.astype(np.float64) - k[0, 2]) / k[0, 0] * values,
            (sample_y.astype(np.float64) - k[1, 2]) / k[1, 1] * values,
            values,
            np.ones_like(values),
        ],
        axis=0,
    )
    ego = calibration.cam_to_ego @ camera
    finite = np.isfinite(ego[0]) & np.isfinite(ego[1])
    return stitched[finite, 0], stitched[finite, 1], ego[0, finite], ego[1, finite]


def _pointcloud_xyz(message: Any) -> Any:
    import numpy as np

    fields = {field.name: int(field.offset) for field in message.fields}
    require(all(name in fields for name in ("x", "y", "z")), "pointcloud_xyz_fields_missing")
    endian = ">" if bool(message.is_bigendian) else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [f"{endian}f4", f"{endian}f4", f"{endian}f4"],
            "offsets": [fields["x"], fields["y"], fields["z"]],
            "itemsize": int(message.point_step),
        }
    )
    values = np.frombuffer(message.data, dtype=dtype, count=int(message.width * message.height))
    xyz = np.stack([values["x"], values["y"], values["z"]], axis=1).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def _lidar_metric_points(
    ego_xyz: Any,
    calibrations: dict[int, CameraCalibration],
    semantic_mask: Any,
) -> tuple[Any, Any, Any, Any]:
    import cv2
    import numpy as np

    stitched_rows = []
    stitched_columns = []
    forward_rows = []
    left_rows = []
    homogeneous = np.concatenate([ego_xyz, np.ones((len(ego_xyz), 1), dtype=np.float64)], axis=1).T
    for camera_id in CAMERA_IDS:
        calibration = calibrations[camera_id]
        ego_to_cam = np.linalg.inv(calibration.cam_to_ego)
        camera = (ego_to_cam @ homogeneous)[:3].T
        in_front = camera[:, 2] > 1e-6
        if not in_front.any():
            continue
        indices = np.nonzero(in_front)[0]
        projected, _ = cv2.projectPoints(
            camera[in_front],
            np.zeros(3),
            np.zeros(3),
            calibration.distorted_k,
            calibration.distortion,
        )
        projected = projected.reshape(-1, 2)
        finite_projection = np.isfinite(projected).all(axis=1)
        projected = projected[finite_projection]
        indices = indices[finite_projection]
        raw = np.rint(projected).astype(np.int32)
        inside = (raw[:, 0] >= 0) & (raw[:, 0] < 752) & (raw[:, 1] >= 0) & (raw[:, 1] < 480)
        raw = raw[inside]
        indices = indices[inside]
        mapping = calibration.raw_to_stitch[raw[:, 1], raw[:, 0]].reshape(-1, 5, 2).astype(np.int32)
        mapped_y = mapping[:, :, 0]
        mapped_x = mapping[:, :, 1]
        valid = (
            (mapped_y >= 0)
            & (mapped_y < IMAGE_HEIGHT)
            & (mapped_x >= 0)
            & (mapped_x < IMAGE_WIDTH)
        )
        semantic = np.zeros(valid.shape, dtype=bool)
        rows, columns = np.nonzero(valid)
        semantic[rows, columns] = semantic_mask[mapped_y[rows, columns], mapped_x[rows, columns]]
        has_semantic = semantic.any(axis=1)
        choice = semantic.argmax(axis=1)
        rows = np.arange(len(mapping))
        selected = indices[has_semantic]
        stitched_rows.append(mapped_y[rows, choice][has_semantic])
        stitched_columns.append(mapped_x[rows, choice][has_semantic])
        forward_rows.append(ego_xyz[selected, 0])
        left_rows.append(ego_xyz[selected, 1])
    if not stitched_rows:
        empty_i = np.empty(0, dtype=np.int16)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_f
    return (
        np.concatenate(stitched_rows),
        np.concatenate(stitched_columns),
        np.concatenate(forward_rows),
        np.concatenate(left_rows),
    )


def materialize_metric_ledger(
    *,
    bag_path: Path,
    timestamps_path: Path,
    dense_ledger_path: Path,
    dense_manifest_path: Path,
    calibration_dir: Path,
    depth_source_dir: Path,
    depth_checkpoint: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    masks, dense_manifest = load_dense_ledger(dense_ledger_path, dense_manifest_path)
    calibrations = load_calibrations(calibration_dir)
    model = _load_depth_model(depth_source_dir, depth_checkpoint)
    timestamps = load_image_timestamps(timestamps_path)
    frames = np.asarray(sorted(timestamps), dtype=np.int32)
    target_ns = [round(timestamps[int(frame)] * 1e9) for frame in frames]
    frame_by_index = {index: int(frame) for index, frame in enumerate(frames)}
    image_topics = {
        f"ros_indigosdk_node/image{camera_id}/compressed": camera_id
        for camera_id in CAMERA_IDS
    }
    lidar_topics = {
        "upper_velodyne/velodyne_points": "upper",
        "lower_velodyne/velodyne_points": "lower",
    }
    image_payloads: dict[tuple[int, int], bytes] = {}
    image_hashes: dict[str, str] = {}
    cloud_candidates: dict[str, list[tuple[int, Any, str]]] = {"upper": [], "lower": []}
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag_path) as reader:
        selected = [
            connection
            for connection in reader.connections
            if connection.topic.lstrip("/") in set(image_topics) | set(lidar_topics)
        ]
        require(len(selected) == len(image_topics) + len(lidar_topics), "raw_camera_or_lidar_topic_missing")
        for connection in selected:
            if connection.msgtype not in typestore.fielddefs:
                typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for connection, _bag_time, raw in reader.messages(connections=selected):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            topic = connection.topic.lstrip("/")
            timestamp_ns = stamp_ns(message.header.stamp)
            if topic in image_topics:
                nearest = _nearest_frame(target_ns, timestamp_ns)
                if nearest is None or abs(target_ns[nearest] - timestamp_ns) > 1_000_000:
                    continue
                frame = frame_by_index[nearest]
                camera_id = image_topics[topic]
                payload = bytes(message.data)
                image_payloads[(frame, camera_id)] = payload
                image_hashes[f"{frame:06d}/image{camera_id}"] = hashlib.sha256(payload).hexdigest()
            else:
                first = target_ns[0] - round(LIDAR_MAX_AGE_S * 1e9)
                last = target_ns[-1]
                if first <= timestamp_ns <= last:
                    payload_hash = hashlib.sha256(bytes(message.data)).hexdigest()
                    cloud_candidates[lidar_topics[topic]].append((timestamp_ns, message, payload_hash))
    require(len(image_payloads) == len(frames) * len(CAMERA_IDS), "perspective_frame_missing")
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

    lidar_yaml = _yaml(calibration_dir / "lidars.yaml")["lidar"]
    upper_to_ego = np.asarray(lidar_yaml["upper2ego"], dtype=np.float64)
    lower_to_upper = np.asarray(lidar_yaml["lower2upper"], dtype=np.float64)
    sensor_to_ego = {"upper": upper_to_ego, "lower": upper_to_ego @ lower_to_upper}
    points: dict[str, dict[str, list[Any]]] = {
        arm: {name: [] for name in ("stitched_y", "stitched_x", "forward_m", "left_m")}
        for arm in METRIC_ARMS
    }
    offsets = {arm: [0] for arm in METRIC_ARMS}
    frame_counts = {arm: {} for arm in METRIC_ARMS}
    torch.cuda.reset_peak_memory_stats()
    for frame_index, frame in enumerate(frames, start=1):
        frame = int(frame)
        rgb_parts = []
        for camera_id in CAMERA_IDS:
            rgb_parts.append(
                _rgb_metric_points(
                    image_payloads[(frame, camera_id)],
                    calibrations[camera_id],
                    masks[frame],
                    model,
                )
            )
        rgb = tuple(np.concatenate([part[index] for part in rgb_parts]) for index in range(4))
        lidar_xyz = []
        for sensor in ("upper", "lower"):
            selected = selected_clouds.get((frame, sensor))
            if selected is None:
                continue
            xyz = _pointcloud_xyz(selected[1])
            homogeneous = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1).T
            lidar_xyz.append((sensor_to_ego[sensor] @ homogeneous)[:3].T)
        lidar = _lidar_metric_points(
            np.concatenate(lidar_xyz) if lidar_xyz else np.empty((0, 3), dtype=np.float64),
            calibrations,
            masks[frame],
        )
        for arm, values in (("rgb_metric", rgb), ("lidar_metric", lidar)):
            converted = (
                values[0].astype(np.int16),
                values[1].astype(np.int16),
                values[2].astype(np.float32),
                values[3].astype(np.float32),
            )
            for name, value in zip(("stitched_y", "stitched_x", "forward_m", "left_m"), converted):
                points[arm][name].append(value)
            offsets[arm].append(offsets[arm][-1] + len(converted[0]))
            frame_counts[arm][f"{frame:06d}"] = len(converted[0])
        if frame_index % 10 == 0 or frame_index == len(frames):
            print(json.dumps({"r6_frames": frame_index, "total": len(frames)}), flush=True)

    arrays: dict[str, Any] = {"frames": frames}
    for arm in METRIC_ARMS:
        arrays[f"{arm}_offsets"] = np.asarray(offsets[arm], dtype=np.int64)
        for name in ("stitched_y", "stitched_x", "forward_m", "left_m"):
            arrays[f"{arm}_{name}"] = np.concatenate(points[arm][name])
    atomic_npz(output_path, **arrays)
    manifest = {
        "schema_version": LEDGER_SCHEMA,
        "truth_blind": True,
        "sequence": SEQUENCE,
        "frames": {"first": FIRST_FRAME, "last": LAST_FRAME, "count": len(frames)},
        "source": {
            "bag": str(bag_path),
            "bag_sha256": sha256_file(bag_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "dense_semantic_ledger": str(dense_ledger_path),
            "dense_semantic_ledger_sha256": sha256_file(dense_ledger_path),
            "dense_semantic_manifest_sha256": sha256_file(dense_manifest_path),
            "depth_source": str(depth_source_dir),
            "depth_checkpoint": str(depth_checkpoint),
            "depth_checkpoint_sha256": sha256_file(depth_checkpoint),
            "calibration_dir": str(calibration_dir),
            "calibration_sha256": {
                name: sha256_file(calibration_dir / name)
                for name in ("cameras.yaml", "lidars.yaml", "defaults.yaml")
            },
            "mapping_sha256": {
                str(camera_id): sha256_file(
                    calibration_dir / "indi2stitch_mappings" / f"indi2stitch_mapping_camera_{camera_id}.npy"
                )
                for camera_id in CAMERA_IDS
            },
            "compressed_perspective_image_sha256": image_hashes,
            "selected_lidar_payload_sha256": {
                f"{frame:06d}/{sensor}": row[2]
                for (frame, sensor), row in sorted(selected_clouds.items())
            },
        },
        "inference": {
            "rgb_model": "Depth Anything V2 vits metric Hypersim",
            "rgb_output": "native metric depth in metres; no scale or shift alignment",
            "rgb_input": "five official perspective cameras, calibrated undistortion before inference",
            "rgb_input_size": MODEL_INPUT_SIZE,
            "rgb_sample_stride_px": RGB_SAMPLE_STRIDE_PX,
            "rgb_valid_depth_m": [RGB_MIN_DEPTH_M, RGB_MAX_DEPTH_M],
            "lidar_input": "latest upper/lower raw Velodyne sweep at or before each image",
            "lidar_max_age_s": LIDAR_MAX_AGE_S,
            "lidar_selected_sweeps": len(selected_clouds),
            "lidar_age_s": {
                "minimum": min(lidar_ages) if lidar_ages else None,
                "median": float(np.median(lidar_ages)) if lidar_ages else None,
                "maximum": max(lidar_ages) if lidar_ages else None,
            },
            "semantic_binding": "R5 sealed ADE20K argmax person mask via official individual-to-stitched mapping",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "point_count_by_frame": frame_counts,
        },
        "ledger": str(output_path),
        "ledger_sha256": sha256_file(output_path),
        "dense_semantic_manifest": dense_manifest,
    }
    write_json(manifest_path, manifest)
    return manifest


def load_point_ledger(path: Path, manifest_path: Path) -> PointLedger:
    import numpy as np

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == LEDGER_SCHEMA, "metric_ledger_manifest_schema")
    require(sha256_file(path) == manifest["ledger_sha256"], "metric_ledger_hash_drift")
    payload = np.load(path)
    frames = payload["frames"]
    require(len(frames) == LAST_FRAME - FIRST_FRAME + 1, "metric_ledger_frame_count")
    return PointLedger(
        frames=frames,
        offsets={arm: payload[f"{arm}_offsets"] for arm in METRIC_ARMS},
        stitched_y={arm: payload[f"{arm}_stitched_y"] for arm in METRIC_ARMS},
        stitched_x={arm: payload[f"{arm}_stitched_x"] for arm in METRIC_ARMS},
        forward_m={arm: payload[f"{arm}_forward_m"] for arm in METRIC_ARMS},
        left_m={arm: payload[f"{arm}_left_m"] for arm in METRIC_ARMS},
        manifest=manifest,
    )


class MetricResidualLookup:
    def __init__(
        self,
        masks: dict[int, Any],
        detector_rows: Sequence[dict[str, Any]],
        truth_boxes: dict[int, list[dict[str, Any]]],
        ledger: PointLedger,
    ) -> None:
        self.masks = masks
        self.truth_boxes = truth_boxes
        self.ledger = ledger
        self.detector_by_frame: dict[int, list[dict[str, Any]]] = {}
        for row in detector_rows:
            self.detector_by_frame.setdefault(int(row["frame_index"]), []).append(row)
        self.cache: dict[tuple[str, int, str | None], dict[str, dict[str, Any]]] = {}

    def by_target(self, arm: str, frame: int, excluded_track_id: str | None) -> dict[str, dict[str, Any]]:
        import cv2
        import numpy as np

        key = (arm, frame, excluded_track_id)
        if key in self.cache:
            return self.cache[key]
        residual = self.masks[frame].copy()
        for row in self.detector_by_frame.get(frame, []):
            if excluded_track_id is not None and str(row["track_id"]) == excluded_track_id:
                continue
            x1, y1, x2, y2 = (float(value) for value in row["bbox_xyxy"])
            left = max(0, min(IMAGE_WIDTH, math.floor(x1)))
            top = max(0, min(IMAGE_HEIGHT, math.floor(y1)))
            right = max(0, min(IMAGE_WIDTH, math.ceil(x2)))
            bottom = max(0, min(IMAGE_HEIGHT, math.ceil(y2)))
            residual[top:bottom, left:right] = False
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            residual.astype(np.uint8), connectivity=8
        )
        stitched_y, stitched_x, forward, left = self.ledger.frame_points(arm, frame)
        point_components = labels[stitched_y, stitched_x]
        candidates = []
        for component in range(1, count):
            selected = point_components == component
            if not selected.any():
                continue
            box_left, box_top, width, height, pixels = (
                int(value) for value in stats[component]
            )
            component_forward = forward[selected].astype(np.float64)
            component_left = left[selected].astype(np.float64)
            distance = np.hypot(component_forward, component_left)
            candidates.append(
                {
                    "bbox_xyxy": [
                        float(box_left),
                        float(box_top),
                        float(box_left + width),
                        float(box_top + height),
                    ],
                    "forward_m": component_forward,
                    "left_m": component_left,
                    "person_pixels": pixels,
                    "metric_point_count": int(selected.sum()),
                    "metric_min_distance_m": float(distance.min()),
                    "metric_median_distance_m": float(np.median(distance)),
                }
            )
        matched = {}
        for source_index, truth_index, overlap in associate_frame(candidates, self.truth_boxes[frame]):
            target = str(self.truth_boxes[frame][truth_index]["label_id"])
            matched[target] = {**candidates[source_index], "evaluator_iou": overlap}
        self.cache[key] = matched
        return matched


def metric_entry_s(samples: Sequence[SensorSample], index: int, residual: dict[str, Any]) -> float | None:
    import numpy as np

    current = samples[index]
    earliest = index
    while earliest > 0 and current.time_s - samples[earliest - 1].time_s <= EGO_HISTORY_S + 1e-9:
        earliest -= 1
    if earliest == index:
        return None
    previous = samples[earliest]
    span_s = current.time_s - previous.time_s
    if span_s <= 0.0:
        return None
    dx = current.ego_x_m - previous.ego_x_m
    dy = current.ego_y_m - previous.ego_y_m
    cosine = math.cos(current.ego_yaw_rad)
    sine = math.sin(current.ego_yaw_rad)
    velocity = Vec2(
        -(dx * cosine + dy * sine) / span_s,
        -(-dx * sine + dy * cosine) / span_s,
    )
    forward = np.asarray(residual["forward_m"])
    left = np.asarray(residual["left_m"])
    order = np.argsort(np.hypot(forward, left), kind="stable")
    entries = []
    for point_index in order:
        entry = _first_tube_entry_s(
            Vec2(float(forward[point_index]), float(left[point_index])),
            velocity,
            ROUTE_HALF_WIDTH_M,
            HORIZON_S,
            MINIMUM_CLOSING_SPEED_MPS,
        )
        if entry is not None:
            entries.append(entry)
            if entry <= 0.0:
                break
    return min(entries) if entries else None


def run_arm(
    case: SegmentCase,
    arm: str,
    dropout: set[int],
    residual_lookup: MetricResidualLookup,
) -> ArmRun:
    require(arm in {"track_only", *METRIC_ARMS}, f"unknown_arm:{arm}")
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    runner = DTRR2Arm(config)
    origin = case.samples[0].time_s
    observations = []
    base_predictions = []
    for sample in case.samples:
        real = sensor_observation(sample)
        dropped = sample.frame_index in dropout and real is not None
        observation = None if dropped else real
        observations.append(observation)
        frame = CausalFrame(
            time_s=sample.time_s - origin,
            ego_pose=sample_pose(sample),
            observations=() if observation is None else (observation,),
            person_detection_count=int(observation is not None),
        )
        base_predictions.append(runner.step(frame))
    if arm == "track_only":
        return ArmRun(tuple(base_predictions))

    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    fused = []
    available = 0
    risk_frames = 0
    distances = []
    evaluator_ious = []
    guard_boundary_s = config.route_horizon_s * FROZEN_R2_CONFIG.imminent_horizon_fraction
    for index, (sample, base, observation) in enumerate(zip(case.samples, base_predictions, observations)):
        residual = None
        entry_s = None
        if observation is None:
            excluded = sample.detector_track_id if sample.frame_index in dropout else None
            residual = residual_lookup.by_target(arm, sample.frame_index, excluded).get(case.label_id)
            available += int(residual is not None)
            if residual is not None:
                distances.append(float(residual["metric_min_distance_m"]))
                evaluator_ious.append(float(residual["evaluator_iou"]))
                entry_s = metric_entry_s(case.samples, index, residual)
        residual_risk = entry_s is not None
        risk_frames += int(residual_risk)
        raw_alert = True if residual_risk else base.raw_alert
        urgent = residual_risk and entry_s <= guard_boundary_s + 1e-9
        urgent = urgent or base_urgent(base, guard_boundary_s)
        fused.append(
            Prediction(
                time_s=base.time_s,
                signal=lifecycle.update(base.time_s, raw_alert, urgent=urgent),
                raw_alert=raw_alert,
                reason=(f"{arm}_occupancy" if residual_risk else base.reason),
                track_id=(f"residual:{case.label_id}" if residual_risk else base.track_id),
                diagnostic={
                    **base.diagnostic,
                    "metric_arm": arm,
                    "residual_available": str(residual is not None).lower(),
                    "residual_entry_s": entry_s if entry_s is not None else "none",
                    "residual_metric_points": int(residual["metric_point_count"]) if residual else 0,
                    "residual_metric_min_distance_m": residual["metric_min_distance_m"] if residual else "none",
                    "residual_metric_median_distance_m": residual["metric_median_distance_m"] if residual else "none",
                    "residual_evaluator_iou": residual["evaluator_iou"] if residual else "none",
                },
            )
        )
    return ArmRun(
        tuple(fused),
        residual_available_frames=available,
        residual_risk_frames=risk_frames,
        residual_distances_m=tuple(distances),
        residual_evaluator_ious=tuple(evaluator_ious),
    )


def original_cohort(cases: Sequence[SegmentCase], lookup: MetricResidualLookup) -> dict[str, Any]:
    from jrdb_native_ceiling import ArmAccumulator

    names = ("track_only", *METRIC_ARMS)
    arms = {name: ArmAccumulator() for name in names}
    diagnostics = {
        name: {"residual_available_frames": 0, "residual_risk_frames": 0}
        for name in names
    }
    for case in cases:
        for name, accumulator in arms.items():
            current = run_arm(case, name, set(), lookup)
            accumulator.merge(metrics_for_run(case, current))
            diagnostics[name]["residual_available_frames"] += current.residual_available_frames
            diagnostics[name]["residual_risk_frames"] += current.residual_risk_frames
    return {
        name: {**accumulator.to_dict(include_escalation=True), **diagnostics[name]}
        for name, accumulator in arms.items()
    }


def stress_trials(cases: Sequence[SegmentCase], lookup: MetricResidualLookup) -> dict[str, Any]:
    from jrdb_native_ceiling import ArmAccumulator

    output = {}
    names = ("track_only", *METRIC_ARMS)
    for duration_s in DROPOUT_DURATIONS_S:
        accumulators = {name: ArmAccumulator() for name in names}
        counts = {
            name: {
                "dropout_window_alerted_trials": 0,
                "dropout_window_known_evidence_frames": 0,
                "dropout_window_frames": 0,
                "residual_available_frames": 0,
                "residual_risk_frames": 0,
            }
            for name in names
        }
        trial_rows = []
        for case in cases:
            for event_index, event in enumerate(case.events):
                dropped = dropout_frames(case.samples, event, duration_s)
                arm_rows = {}
                for name, accumulator in accumulators.items():
                    current = run_arm(case, name, dropped, lookup)
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
                    window_indices = [
                        index for index, sample in enumerate(case.samples) if sample.frame_index in dropped
                    ]
                    alerted = any(
                        current.predictions[index].signal in ACTIVE_SIGNALS
                        and current.predictions[index].raw_alert is True
                        for index in window_indices
                    )
                    known_evidence = sum(
                        current.predictions[index].raw_alert is not None for index in window_indices
                    )
                    counts[name]["dropout_window_alerted_trials"] += int(alerted)
                    counts[name]["dropout_window_known_evidence_frames"] += known_evidence
                    counts[name]["dropout_window_frames"] += len(window_indices)
                    counts[name]["residual_available_frames"] += current.residual_available_frames
                    counts[name]["residual_risk_frames"] += current.residual_risk_frames
                    arm_rows[name] = {
                        "dropout_window_alerted": alerted,
                        "dropout_window_known_evidence_frames": known_evidence,
                        "residual_available_frames": current.residual_available_frames,
                        "residual_risk_frames": current.residual_risk_frames,
                        "residual_min_metric_distance_m": min(current.residual_distances_m) if current.residual_distances_m else None,
                        "residual_max_evaluator_iou": max(current.residual_evaluator_ious) if current.residual_evaluator_ious else None,
                        "metrics": metrics.to_dict(include_escalation=True),
                    }
                trial_rows.append(
                    {
                        "label_id": case.label_id,
                        "segment_index": case.segment_index,
                        "event_index": event_index,
                        "category": event.category,
                        "event_start_frame": case.samples[event.start_index].frame_index,
                        "contact_frame": case.samples[event.contact_index].frame_index,
                        "contact_truth_distance_m": case.samples[event.contact_index].distance_m,
                        "contact_truth_tube_threshold_m": case.samples[event.contact_index].tube_threshold_m,
                        "dropout_frames": sorted(dropped),
                        "arms": arm_rows,
                    }
                )
        trials = len(trial_rows)
        track_misses = sum(not row["arms"]["track_only"]["dropout_window_alerted"] for row in trial_rows)
        for name, values in counts.items():
            values["dropout_window_alert_recall"] = ratio(values["dropout_window_alerted_trials"], trials)
            values["dropout_window_known_evidence_rate"] = ratio(
                values["dropout_window_known_evidence_frames"], values["dropout_window_frames"]
            )
            values["recovered_track_only_window_misses"] = sum(
                not row["arms"]["track_only"]["dropout_window_alerted"]
                and row["arms"][name]["dropout_window_alerted"]
                for row in trial_rows
            )
            values["recovery_of_track_only_window_misses"] = ratio(
                values["recovered_track_only_window_misses"], track_misses
            )
        output[f"{duration_s:.1f}"] = {
            "duration_s": duration_s,
            "trials": trials,
            "track_only_dropout_window_misses": track_misses,
            "arms": {
                name: {**accumulators[name].to_dict(include_escalation=True), **counts[name]}
                for name in names
            },
            "by_trial": trial_rows,
        }
    return output


def gate(original: dict[str, Any], stress: dict[str, Any]) -> dict[str, Any]:
    baseline = original["track_only"]
    total_misses = sum(row["track_only_dropout_window_misses"] for row in stress.values())
    false_limit = baseline["false_alert_segments"] * 1.10
    arms = {}
    for name in METRIC_ARMS:
        recovered = sum(
            row["arms"][name]["recovered_track_only_window_misses"] for row in stress.values()
        )
        checks = {
            "track_only_induced_window_misses_exist": total_misses > 0,
            "recovers_at_least_half": total_misses > 0 and recovered / total_misses >= 0.50,
            "original_one_to_one_event_recall_not_lower": (
                original[name]["event_detection_recall"] is not None
                and baseline["event_detection_recall"] is not None
                and original[name]["event_detection_recall"] >= baseline["event_detection_recall"]
            ),
            "original_false_segments_within_ten_percent": (
                original[name]["false_alert_segments"] <= false_limit + 1e-9
            ),
        }
        arms[name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "recovered_window_misses": recovered,
            "recovery_rate": ratio(recovered, total_misses),
        }
    return {
        "verdict": (
            "R6_RGB_DIRECT_METRIC_DEVELOPMENT_GATE_MET"
            if arms["rgb_metric"]["passed"]
            else "R6_RGB_DIRECT_METRIC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "track_only_induced_window_misses": total_misses,
        "false_segment_limit": false_limit,
        "arms": arms,
        "clear_evidence": (
            "NOT_EVALUABLE_NO_CLEAR_ELIGIBLE_EVENTS"
            if baseline["clear_rate"] is None
            else "EVALUABLE"
        ),
        "note": (
            "The RGB arm is the deployable challenger. The raw-LiDAR arm is a privileged "
            "metric-source ceiling and does not authorize a product dependency."
        ),
    }


def structural_reachability(cases: Sequence[SegmentCase]) -> dict[str, Any]:
    rows = []
    for case in cases:
        for event_index, event in enumerate(case.events):
            contact_time = case.samples[event.contact_index].time_s
            speeds = []
            for index, current in enumerate(case.samples):
                if not (
                    contact_time - max(DROPOUT_DURATIONS_S) - 1e-9
                    <= current.time_s
                    <= contact_time + 1e-9
                ):
                    continue
                earliest = index
                while (
                    earliest > 0
                    and current.time_s - case.samples[earliest - 1].time_s
                    <= EGO_HISTORY_S + 1e-9
                ):
                    earliest -= 1
                if earliest == index:
                    continue
                previous = case.samples[earliest]
                span_s = current.time_s - previous.time_s
                if span_s <= 0.0:
                    continue
                speeds.append(
                    math.hypot(
                        current.ego_x_m - previous.ego_x_m,
                        current.ego_y_m - previous.ego_y_m,
                    )
                    / span_s
                )
            maximum = max(speeds) if speeds else None
            rows.append(
                {
                    "label_id": case.label_id,
                    "event_index": event_index,
                    "contact_frame": case.samples[event.contact_index].frame_index,
                    "maximum_ego_speed_mps_in_0.8s_window": maximum,
                    "minimum_closing_speed_mps": MINIMUM_CLOSING_SPEED_MPS,
                    "static_occupancy_entry_reachable": (
                        maximum is not None
                        and maximum + 1e-9 >= MINIMUM_CLOSING_SPEED_MPS
                    ),
                }
            )
    reachable = sum(bool(row["static_occupancy_entry_reachable"]) for row in rows)
    return {
        "matcher": "R5 static residual occupancy; relative velocity is negative ego velocity only",
        "events": rows,
        "reachable_events": reachable,
        "total_events": len(rows),
        "verdict": (
            "EVALUABLE"
            if reachable == len(rows)
            else "NOT_EVALUABLE_STATIC_OCCUPANCY_MATCHER_UNREACHABLE_ON_EVENT_WINDOW"
        ),
        "interpretation": (
            "A direct metric source cannot isolate the depth hypothesis when the frozen "
            "residual matcher has no admissible closing-velocity source."
        ),
    }


def plot_result(result: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    durations = [float(value) for value in result["stress_by_duration_s"]]
    names = ("track_only", *METRIC_ARMS)
    labels = {
        "track_only": "R2 track-only",
        "rgb_metric": "R6-RGB direct metric",
        "lidar_metric": "R6-P raw LiDAR ceiling",
    }
    colors = {"track_only": "#C23B22", "rgb_metric": "#0F766E", "lidar_metric": "#2563EB"}
    markers = {"track_only": "x", "rgb_metric": "o", "lidar_metric": "^"}
    styles = {"track_only": "--", "rgb_metric": "-", "lidar_metric": "-."}
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for name in names:
        window_recall = [
            result["stress_by_duration_s"][f"{duration:.1f}"]["arms"][name]["dropout_window_alert_recall"]
            for duration in durations
        ]
        event_f1 = [
            result["stress_by_duration_s"][f"{duration:.1f}"]["arms"][name]["event_detection_f1"]
            for duration in durations
        ]
        options = {
            "marker": markers[name],
            "linestyle": styles[name],
            "linewidth": 2.2,
            "markersize": 7,
            "label": labels[name],
            "color": colors[name],
        }
        axes[0].plot(durations, window_recall, **options)
        axes[1].plot(durations, event_f1, **options)
    for axis in axes:
        axis.set_xlabel("Track dropout duration (s)")
        axis.set_xticks(durations)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].set_title("Evidence during induced dropout")
    axes[0].set_ylabel("Dropout-window alert recall")
    axes[1].set_title("Frozen standard event evaluator")
    axes[1].set_ylabel("One-to-one event F1")
    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle("DTR R6 direct metric residual occupancy", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    known_result_path = args.known_height_result.resolve(strict=True)
    known_tracks_path = args.known_height_tracks.resolve(strict=True)
    labels_path = args.labels_zip.resolve(strict=True)
    timestamps_path = args.timestamps_zip.resolve(strict=True)
    bag_path = args.bag.resolve(strict=True)
    dense_ledger_path = args.dense_ledger.resolve(strict=True)
    dense_manifest_path = args.dense_manifest.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    depth_source_dir = args.depth_source.resolve(strict=True)
    depth_checkpoint = args.depth_checkpoint.resolve(strict=True)
    known_result = json.loads(known_result_path.read_text(encoding="utf-8"))
    require(
        sha256_file(known_tracks_path) == known_result["truth_blind_sensor_geometry"]["ledger_sha256"],
        "known_height_ledger_hash_drift",
    )
    point_path, point_manifest_path = ledger_paths(args.output.resolve())
    if args.reuse_metric_ledger and point_path.exists() and point_manifest_path.exists():
        ledger = load_point_ledger(point_path, point_manifest_path)
    else:
        materialize_metric_ledger(
            bag_path=bag_path,
            timestamps_path=timestamps_path,
            dense_ledger_path=dense_ledger_path,
            dense_manifest_path=dense_manifest_path,
            calibration_dir=calibration_dir,
            depth_source_dir=depth_source_dir,
            depth_checkpoint=depth_checkpoint,
            output_path=point_path,
            manifest_path=point_manifest_path,
        )
        ledger = load_point_ledger(point_path, point_manifest_path)

    # Privileged native identity and future contact are opened only after the
    # complete RGB/LiDAR metric ledger above is sealed.
    sensor_rows = read_jsonl(known_tracks_path)
    masks, _dense_manifest = load_dense_ledger(dense_ledger_path, dense_manifest_path)
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
    lookup = MetricResidualLookup(masks, sensor_rows, load_truth_boxes(labels_path), ledger)
    original = original_cohort(cases, lookup)
    stress = stress_trials(cases, lookup)
    reachability = structural_reachability(cases)
    gate_result = gate(original, stress)
    if reachability["reachable_events"] < reachability["total_events"]:
        gate_result["verdict"] = (
            "R6_DIRECT_METRIC_SINGLE_FACTOR_NOT_EVALUABLE_STATIC_OCCUPANCY_MATCHER_UNREACHABLE"
        )
        gate_result["passed"] = False
        gate_result["structural_reachability"] = reachability["verdict"]
    result = {
        "schema_version": SCHEMA,
        "status": "DTR_R6_DIRECT_METRIC_OCCUPANCY_CANARY_COMPLETE",
        "claim_ceiling": CLAIM_CEILING,
        "question": (
            "Does replacing R5 fixed-height geometry with direct per-pixel metric occupancy "
            "restore route-risk evidence during the same detector dropouts?"
        ),
        "frozen": {
            "r2": FROZEN_R2_CONFIG.to_dict(),
            "route_horizon_s": HORIZON_S,
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "lifecycle": "unchanged ONSET/HOLD/ESCALATE/CLEAR; missing remains UNKNOWN",
            "semantic_mask": "R5 sealed ADE20K argmax person mask",
            "dropout_durations_s": list(DROPOUT_DURATIONS_S),
        },
        "intervention": {
            "rgb_metric": "calibrated perspective metric depth -> back-projected occupied surface points",
            "lidar_metric": "current/past-only raw Velodyne -> semantic-supported occupied surface points",
            "route_intersection": (
                "frozen route half-width against occupied surface points; no duplicate person-radius dilation"
            ),
        },
        "metric_point_source": ledger.manifest,
        "source": {
            "dataset": "JRDB public train split",
            "sequence": SEQUENCE,
            "window": {"first_frame": FIRST_FRAME, "last_frame": LAST_FRAME},
            "known_height_result": str(known_result_path),
            "known_height_result_sha256": sha256_file(known_result_path),
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
        "privileged_evaluator": {
            "identity_binding": "metric-supported residual component to native current 2-D box at frozen IoU >= 0.30",
            "future_truth": "native future 3-D contact only after metric ledger seal",
            "geometry_quality": geometry_quality,
            "occupancy_probability_calibration": "NOT_EVALUABLE_HARD_SEMANTIC_AND_POINT_OCCUPANCY",
        },
        "structural_reachability": reachability,
        "original_cohort": original,
        "stress_by_duration_s": stress,
        "gate": gate_result,
        "limitations": [
            "The 143-frame window is transparently curated Development evidence with three events.",
            "Repeated duration trials are stress cases over the same events, not independent natural events.",
            "Depth Anything V2 Hypersim is used zero-shot with no scale/shift alignment; this run does not rank metric-depth backbones.",
            "The raw-LiDAR arm is a privileged source ceiling, not a deployable RGB result.",
            "The frozen R5 residual matcher estimates closing only from ego motion; all three event windows are structurally unreachable at its 0.05 m/s minimum closing speed.",
            "Therefore 0/9 is not evidence that direct metric depth or raw LiDAR occupancy lacks useful geometry; this single-factor cohort cannot isolate that hypothesis.",
            "Evaluator-only current 2-D identity binding and future 3-D contact do not enter metric inference.",
            "Hard semantic masks and point occupancy are not calibrated collision probabilities.",
            "No source-disjoint generalization, Android runtime, user benefit, product reliability, or safety performance is established.",
        ],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-height-result", type=Path, required=True)
    parser.add_argument("--known-height-tracks", type=Path, required=True)
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--timestamps-zip", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--dense-ledger", type=Path, required=True)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--depth-source", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--reuse-metric-ledger", action="store_true")
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "output_must_be_json")
    result = run(args)
    write_json(args.output.resolve(), result)
    plot_result(result, args.plot.resolve())
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
