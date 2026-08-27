"""Run one fixed public-JRDB RGB detector/tracker bridge for DTR-R0.

The detector and causal tracker are run and written to a truth-blind ledger
before JRDB annotations are opened.  Evaluation then uses current-frame 2-D
IoU only to bind a detector track occurrence to the corresponding native 3-D
metric center.  Future native geometry is evaluator-only truth.

This is a curated Development bridge, not an independent benchmark, product
test, or safety result.  It deliberately keeps the DTR route matcher frozen.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import binascii
import concurrent.futures
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import struct
import time
from typing import Any, Iterable, Sequence
import zipfile
import zlib

from dtr_r0 import Arm, CausalFrame, DTRConfig, EgoPose, Observation, run_arm
from jrdb_native_ceiling import (
    ArmAccumulator,
    future_hits,
    score_arm,
    truth_events,
)
from jrdb_range_acquire import get_range, parse_central, sha256_file
from real_observation_adapter import (
    BBox,
    CausalPersonTracker,
    Detection,
)


SCHEMA = "dtr-r0-jrdb-rgb-bridge-v1"
TRACK_SCHEMA = "dtr-r0-jrdb-rgb-tracks-v1"
CLAIM_CEILING = "CURATED_PUBLIC_REAL_RGB_DETECTOR_TRACKER_BRIDGE_ONLY"
SEQUENCE = "packard-poster-session-2019-03-20_1"
FIRST_FRAME = 115
LAST_FRAME = 257
FRAME_COUNT = LAST_FRAME - FIRST_FRAME + 1
PRIMARY_TARGETS = ("pedestrian:34", "pedestrian:35")

IMAGE_ARCHIVE_URL = "https://jrdb.erc.monash.edu/static/downloads/train_images.zip"
IMAGE_ARCHIVE_ETAG = '"52ff877b7-5e0c071fdaa94"'
IMAGE_CENTRAL_OFFSET = 22_257_177_895
IMAGE_CENTRAL_SIZE = 22_471_214
IMAGE_WIDTH = 3760
IMAGE_HEIGHT = 480

# Existing D33 JRDB source-only detector settings are reused without tuning.
TILE_WIDTH = 960
TILE_STARTS = (0, 700, 1400, 2100, 2800)
DETECTOR_CONFIDENCE = 0.10
DETECTOR_NMS_IOU = 0.50
DETECTOR_MAX_DET = 50
INFERENCE_SIZE = 640
MINIMUM_EVALUATOR_IOU = 0.30

HORIZON_S = 3.0
ROUTE_HALF_WIDTH_M = 0.65
MAXIMUM_POSE_BRACKET_S = 0.05
MAXIMUM_POSE_ENDPOINT_S = 0.025
MAXIMUM_EXTERNAL_BAG_RGB_DELTA_S = 0.001

# Shared JRDB platform chain, previously derived from the official toolkit
# calibration and the base_link/static platform chain.  Planar rotation is
# identity; only the x offset affects this 2-D experiment.
BASE_LINK_FROM_LOGICAL_RGB360_X_M = -0.019685
BASE_LINK_FROM_LOGICAL_RGB360_Y_M = 0.0


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
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


def read_range(url: str, start: int, end: int, attempts: int = 3) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            return get_range(url, start, end)
        except Exception as current:  # bounded retry for small public members
            error = current
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    assert error is not None
    raise error


def fetch_member(member: dict[str, Any]) -> bytes:
    offset = int(member["offset"])
    header = read_range(IMAGE_ARCHIVE_URL, offset, offset + 29)
    values = struct.unpack("<4s5H3L2H", header)
    require(values[0] == b"PK\x03\x04", "image_local_header_signature")
    name_length, extra_length = values[-2:]
    tail = read_range(
        IMAGE_ARCHIVE_URL,
        offset + 30,
        offset + 30 + name_length + extra_length - 1,
    )
    require(
        tail[:name_length].decode("utf-8") == member["name"],
        "image_local_name_drift",
    )
    start = offset + 30 + name_length + extra_length
    compressed = read_range(
        IMAGE_ARCHIVE_URL,
        start,
        start + int(member["compressed"]) - 1,
    )
    if int(member["method"]) == 0:
        raw = compressed
    elif int(member["method"]) == 8:
        raw = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported_image_zip_method:{member['method']}")
    require(len(raw) == int(member["uncompressed"]), "image_uncompressed_size_drift")
    require(
        binascii.crc32(raw) & 0xFFFFFFFF == int(member["crc32"]),
        "image_crc_drift",
    )
    return raw


def ensure_images(output_root: Path) -> dict[str, Any]:
    central = parse_central(
        read_range(
            IMAGE_ARCHIVE_URL,
            IMAGE_CENTRAL_OFFSET,
            IMAGE_CENTRAL_OFFSET + IMAGE_CENTRAL_SIZE - 1,
        )
    )
    wanted = {
        f"images/image_stitched/{SEQUENCE}/{frame:06d}.jpg": frame
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    index = {row["name"]: row for row in central if row["name"] in wanted}
    require(set(index) == set(wanted), "fixed_window_image_member_missing")

    def restore(name: str) -> tuple[dict[str, Any], bool]:
        member = index[name]
        frame = wanted[name]
        output = output_root / SEQUENCE / f"{frame:06d}.jpg"
        reused = False
        if output.is_file() and output.stat().st_size == int(member["uncompressed"]):
            payload = output.read_bytes()
            if binascii.crc32(payload) & 0xFFFFFFFF == int(member["crc32"]):
                reused = True
            else:
                payload = fetch_member(member)
        else:
            payload = fetch_member(member)
        if not reused:
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_name(output.name + ".partial")
            partial.write_bytes(payload)
            os.replace(partial, output)
        return (
            {
                "frame_index": frame,
                "member": name,
                "path": str(output.resolve()),
                "bytes": len(payload),
                "crc32": int(member["crc32"]),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            reused,
        )

    results: list[tuple[dict[str, Any], bool]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(restore, name) for name in sorted(wanted)]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            results.append(future.result())
            if completed % 40 == 0 or completed == FRAME_COUNT:
                print(
                    json.dumps({"images": completed, "total": FRAME_COUNT}),
                    flush=True,
                )
    records = sorted((row for row, _ in results), key=lambda row: row["frame_index"])
    return {
        "archive": {
            "url": IMAGE_ARCHIVE_URL,
            "etag": IMAGE_ARCHIVE_ETAG,
            "central_directory_offset": IMAGE_CENTRAL_OFFSET,
            "central_directory_size": IMAGE_CENTRAL_SIZE,
        },
        "records": records,
        "reused_frames": sum(reused for _, reused in results),
        "payload_bytes": sum(row["bytes"] for row in records),
    }


def load_image_timestamps(path: Path) -> dict[int, float]:
    with zipfile.ZipFile(path) as bundle:
        payload = json.loads(bundle.read(f"timestamps/{SEQUENCE}/frames_img.json"))
    result: dict[int, float] = {}
    for row in payload["data"]:
        cameras = [item for item in row["cameras"] if item["name"] == "stitched_image0"]
        if not cameras:
            continue
        frame = int(PurePosixPath(cameras[0]["url"]).stem)
        if FIRST_FRAME <= frame <= LAST_FRAME:
            result[frame] = float(cameras[0]["timestamp"])
    require(len(result) == FRAME_COUNT, "fixed_window_image_timestamp_missing")
    return result


def nms(boxes: Any) -> Any:
    import numpy as np

    values = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    if not len(values):
        return values
    order = np.argsort(-values[:, 4], kind="stable")
    kept: list[int] = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        left = np.maximum(values[current, 0], values[rest, 0])
        top = np.maximum(values[current, 1], values[rest, 1])
        right = np.minimum(values[current, 2], values[rest, 2])
        bottom = np.minimum(values[current, 3], values[rest, 3])
        intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
        area = max(0.0, float(values[current, 2] - values[current, 0])) * max(
            0.0, float(values[current, 3] - values[current, 1])
        )
        rest_area = np.maximum(0.0, values[rest, 2] - values[rest, 0]) * np.maximum(
            0.0, values[rest, 3] - values[rest, 1]
        )
        union = area + rest_area - intersection
        overlap = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[overlap <= DETECTOR_NMS_IOU]
    return values[np.asarray(kept, dtype=np.int64)][:DETECTOR_MAX_DET]


def tiled_detections(predictions: Sequence[Any]) -> list[Detection]:
    import numpy as np

    values = []
    require(len(predictions) == len(TILE_STARTS), "tile_prediction_count_drift")
    for start, prediction in zip(TILE_STARTS, predictions):
        boxes = getattr(prediction, "boxes", None)
        if boxes is None or not len(boxes):
            continue
        coordinates = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy().reshape(-1, 1)
        coordinates[:, 0] += start
        coordinates[:, 2] += start
        coordinates[:, 0] = np.clip(coordinates[:, 0], 0, IMAGE_WIDTH)
        coordinates[:, 2] = np.clip(coordinates[:, 2], 0, IMAGE_WIDTH)
        values.append(np.concatenate((coordinates, confidence), axis=1))
    merged = nms(np.concatenate(values, axis=0) if values else np.empty((0, 5)))
    return [
        Detection(BBox(*(float(item) for item in row[:4])), float(row[4]))
        for row in merged
        if row[2] > row[0] and row[3] > row[1]
    ]


def run_detector_tracker(
    image_records: Sequence[dict[str, Any]],
    timestamps: dict[int, float],
    model_path: Path,
    batch_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    require(batch_frames > 0, "batch_frames_must_be_positive")
    model = YOLO(str(model_path.resolve(strict=True)), task="detect")
    tracker = CausalPersonTracker()
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    rows: list[dict[str, Any]] = []
    raw_detections = 0
    new_tracks = 0
    for batch_start in range(0, len(image_records), batch_frames):
        batch = image_records[batch_start : batch_start + batch_frames]
        crops = []
        for record in batch:
            image = cv2.imread(record["path"], cv2.IMREAD_COLOR)
            require(image is not None, f"image_decode_failed:{record['frame_index']}")
            require(
                image.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH),
                f"image_shape_drift:{record['frame_index']}:{image.shape[:2]}",
            )
            crops.extend(image[:, start : start + TILE_WIDTH] for start in TILE_STARTS)
        predictions = model.predict(
            crops,
            imgsz=INFERENCE_SIZE,
            conf=DETECTOR_CONFIDENCE,
            iou=DETECTOR_NMS_IOU,
            classes=[0],
            max_det=DETECTOR_MAX_DET,
            augment=False,
            device=device,
            batch=len(crops),
            verbose=False,
        )
        require(len(predictions) == len(crops), "detector_batch_length_drift")
        for offset, record in enumerate(batch):
            frame = int(record["frame_index"])
            detections = tiled_detections(
                predictions[
                    offset * len(TILE_STARTS) : (offset + 1) * len(TILE_STARTS)
                ]
            )
            raw_detections += len(detections)
            tracked = tracker.update(detections, time_s=timestamps[frame])
            new_tracks += sum(item.is_new_track for item in tracked)
            for item in tracked:
                rows.append(
                    {
                        "schema": TRACK_SCHEMA,
                        "sequence": SEQUENCE,
                        "frame_index": frame,
                        "time_s": timestamps[frame],
                        "track_id": item.track_id,
                        "bbox_xyxy": item.detection.bbox.to_list(),
                        "confidence": item.detection.confidence,
                        "is_new_track": item.is_new_track,
                        "image_sha256": record["sha256"],
                    }
                )
        completed = min(batch_start + len(batch), len(image_records))
        print(
            json.dumps(
                {
                    "detector_frames": completed,
                    "total": len(image_records),
                    "tracked_occurrences": len(rows),
                }
            ),
            flush=True,
        )
    return rows, {
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "raw_detection_count": raw_detections,
        "tracked_occurrence_count": len(rows),
        "track_count": len({row["track_id"] for row in rows}),
        "new_track_occurrences": new_tracks,
    }


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def q_normalize(value: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(float(item) for item in value)
    magnitude = math.sqrt(sum(item * item for item in values))
    require(magnitude > 0.0 and math.isfinite(magnitude), "invalid_pose_quaternion")
    return tuple(item / magnitude for item in values)  # type: ignore[return-value]


def q_slerp(
    left: Iterable[float], right: Iterable[float], weight: float
) -> tuple[float, float, float, float]:
    a = q_normalize(left)
    b = q_normalize(right)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-item for item in b)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return q_normalize((1.0 - weight) * x + weight * y for x, y in zip(a, b))
    theta = math.acos(dot)
    scale = math.sin(theta)
    return tuple(
        math.sin((1.0 - weight) * theta) / scale * x
        + math.sin(weight * theta) / scale * y
        for x, y in zip(a, b)
    )  # type: ignore[return-value]


def yaw_from_q(value: Iterable[float]) -> float:
    x, y, z, w = q_normalize(value)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def read_bag_pose_and_rgb(
    path: Path,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as error:
        raise RuntimeError("jrdb_rgb_bridge requires rosbags") from error

    typestore = get_typestore(Stores.ROS1_NOETIC)
    poses: list[dict[str, Any]] = []
    rgb_times: list[int] = []
    with Reader(path) as reader:
        topic_names = sorted({item.topic.lstrip("/") for item in reader.connections})
        selected = [
            item
            for item in reader.connections
            if item.topic.lstrip("/")
            in {"tf", "ros_indigosdk_node/stitched_image0/compressed"}
        ]
        require(len(selected) == 2, "bag_pose_or_rgb_topic_missing")
        for connection in selected:
            if connection.msgtype not in typestore.fielddefs:
                typestore.register(
                    get_types_from_msg(connection.msgdef.data, connection.msgtype)
                )
        for connection, _bag_time, raw in reader.messages(connections=selected):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            topic = connection.topic.lstrip("/")
            if topic == "tf":
                for item in message.transforms:
                    if (
                        item.header.frame_id.lstrip("/") == "odom"
                        and item.child_frame_id.lstrip("/") == "base_link"
                    ):
                        transform = item.transform
                        poses.append(
                            {
                                "timestamp_ns": stamp_ns(item.header.stamp),
                                "translation": [
                                    float(transform.translation.x),
                                    float(transform.translation.y),
                                    float(transform.translation.z),
                                ],
                                "quaternion_xyzw": [
                                    float(transform.rotation.x),
                                    float(transform.rotation.y),
                                    float(transform.rotation.z),
                                    float(transform.rotation.w),
                                ],
                            }
                        )
            else:
                rgb_times.append(stamp_ns(message.header.stamp))
    poses.sort(key=lambda row: row["timestamp_ns"])
    rgb_times.sort()
    require(len(poses) >= 2, "bag_pose_samples_missing")
    require(bool(rgb_times), "bag_rgb_samples_missing")
    return poses, rgb_times, {
        "bag_has_tf_static": "tf_static" in topic_names,
        "dynamic_pose_topic": "tf",
        "dynamic_pose_edge": "odom->base_link",
        "stitched_rgb_topic": "ros_indigosdk_node/stitched_image0/compressed",
    }


def interpolate_pose(samples: Sequence[dict[str, Any]], target_ns: int) -> dict[str, Any]:
    times = [int(row["timestamp_ns"]) for row in samples]
    index = bisect_left(times, target_ns)
    if index < len(times) and times[index] == target_ns:
        row = samples[index]
        return {
            "x_m": row["translation"][0],
            "y_m": row["translation"][1],
            "yaw_rad": yaw_from_q(row["quaternion_xyzw"]),
            "bracket_s": 0.0,
            "maximum_endpoint_s": 0.0,
        }
    require(0 < index < len(samples), f"pose_unbracketed:{target_ns}")
    left, right = samples[index - 1], samples[index]
    span = (right["timestamp_ns"] - left["timestamp_ns"]) / 1e9
    left_delta = (target_ns - left["timestamp_ns"]) / 1e9
    right_delta = (right["timestamp_ns"] - target_ns) / 1e9
    require(span <= MAXIMUM_POSE_BRACKET_S + 1e-12, f"pose_bracket:{span}")
    require(
        max(left_delta, right_delta) <= MAXIMUM_POSE_ENDPOINT_S + 1e-12,
        f"pose_endpoint:{max(left_delta, right_delta)}",
    )
    weight = left_delta / span
    translation = [
        (1.0 - weight) * a + weight * b
        for a, b in zip(left["translation"], right["translation"])
    ]
    quaternion = q_slerp(left["quaternion_xyzw"], right["quaternion_xyzw"], weight)
    return {
        "x_m": translation[0],
        "y_m": translation[1],
        "yaw_rad": yaw_from_q(quaternion),
        "bracket_s": span,
        "maximum_endpoint_s": max(left_delta, right_delta),
    }


def nearest_delta_s(values: Sequence[int], target: int) -> float:
    index = bisect_left(values, target)
    candidates = values[max(0, index - 1) : min(len(values), index + 1)]
    require(bool(candidates), "nearest_bag_rgb_missing")
    return min(abs(item - target) for item in candidates) / 1e9


@dataclass(frozen=True)
class BridgeSample:
    frame_index: int
    time_s: float
    ego_x_m: float
    ego_y_m: float
    ego_yaw_rad: float
    forward_m: float
    left_m: float
    radius_m: float
    detector_track_id: str | None = None

    @property
    def target_x_m(self) -> float:
        return self.ego_x_m + self.forward_m * math.cos(self.ego_yaw_rad) - self.left_m * math.sin(self.ego_yaw_rad)

    @property
    def target_y_m(self) -> float:
        return self.ego_y_m + self.forward_m * math.sin(self.ego_yaw_rad) + self.left_m * math.cos(self.ego_yaw_rad)

    @property
    def distance_m(self) -> float:
        return math.hypot(self.forward_m, self.left_m)

    @property
    def tube_threshold_m(self) -> float:
        return ROUTE_HALF_WIDTH_M + self.radius_m


def iou_matrix(source: Sequence[dict[str, Any]], truth: Sequence[dict[str, Any]]) -> Any:
    import numpy as np

    left_boxes = np.asarray([row["bbox_xyxy"] for row in source], dtype=np.float64).reshape(-1, 4)
    right_boxes = np.asarray([row["bbox_xyxy"] for row in truth], dtype=np.float64).reshape(-1, 4)
    if not len(left_boxes) or not len(right_boxes):
        return np.zeros((len(left_boxes), len(right_boxes)), dtype=np.float64)
    left = np.maximum(left_boxes[:, None, 0], right_boxes[None, :, 0])
    top = np.maximum(left_boxes[:, None, 1], right_boxes[None, :, 1])
    right = np.minimum(left_boxes[:, None, 2], right_boxes[None, :, 2])
    bottom = np.minimum(left_boxes[:, None, 3], right_boxes[None, :, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    source_area = np.maximum(0.0, left_boxes[:, 2] - left_boxes[:, 0]) * np.maximum(0.0, left_boxes[:, 3] - left_boxes[:, 1])
    truth_area = np.maximum(0.0, right_boxes[:, 2] - right_boxes[:, 0]) * np.maximum(0.0, right_boxes[:, 3] - right_boxes[:, 1])
    union = source_area[:, None] + truth_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def associate_frame(
    source: Sequence[dict[str, Any]], truth: Sequence[dict[str, Any]]
) -> list[tuple[int, int, float]]:
    if not source or not truth:
        return []
    from scipy.optimize import linear_sum_assignment

    matrix = iou_matrix(source, truth)
    source_indices, truth_indices = linear_sum_assignment(1.0 - matrix)
    return [
        (int(source_index), int(truth_index), float(matrix[source_index, truth_index]))
        for source_index, truth_index in zip(source_indices, truth_indices)
        if float(matrix[source_index, truth_index]) >= MINIMUM_EVALUATOR_IOU
    ]


def load_native_and_associate(
    labels_path: Path,
    timestamps: dict[int, float],
    poses: Sequence[dict[str, Any]],
    bag_rgb_times: Sequence[int],
    detector_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, list[BridgeSample]], dict[str, Any], dict[str, Any]]:
    with zipfile.ZipFile(labels_path) as bundle:
        labels_2d = json.loads(
            bundle.read(f"labels/labels_2d_stitched/{SEQUENCE}.json")
        )["labels"]
        labels_3d = json.loads(
            bundle.read(f"labels/labels_3d/{SEQUENCE}.json")
        )["labels"]
    detector_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in detector_rows:
        detector_by_frame.setdefault(int(row["frame_index"]), []).append(row)

    tracks: dict[str, list[BridgeSample]] = {}
    matches_by_frame_and_id: dict[tuple[int, str], dict[str, Any]] = {}
    match_ious: list[float] = []
    native_2d_rows = 0
    native_3d_rows = 0
    joined_rows = 0
    pose_brackets = []
    pose_endpoints = []
    bag_rgb_deltas = []
    primary = {
        target: {"native_2d_frames": 0, "native_3d_frames": 0, "matched_frames": 0, "detector_track_ids": set()}
        for target in PRIMARY_TARGETS
    }

    for frame in range(FIRST_FRAME, LAST_FRAME + 1):
        stem = f"{frame:06d}"
        time_s = timestamps[frame]
        target_ns = round(time_s * 1e9)
        pose = interpolate_pose(poses, target_ns)
        pose_brackets.append(pose["bracket_s"])
        pose_endpoints.append(pose["maximum_endpoint_s"])
        bag_delta = nearest_delta_s(bag_rgb_times, target_ns)
        require(
            bag_delta <= MAXIMUM_EXTERNAL_BAG_RGB_DELTA_S + 1e-12,
            f"external_bag_rgb_delta:{frame}:{bag_delta}",
        )
        bag_rgb_deltas.append(bag_delta)

        truth_2d = []
        ids_2d: set[str] = set()
        for item in labels_2d[f"{stem}.jpg"]:
            if bool(item.get("attributes", {}).get("no_eval", False)):
                continue
            x, y, width, height = (float(value) for value in item["box"])
            if width <= 0.0 or height <= 0.0:
                continue
            label_id = str(item["label_id"])
            ids_2d.add(label_id)
            truth_2d.append(
                {"label_id": label_id, "bbox_xyxy": [x, y, x + width, y + height]}
            )
            native_2d_rows += 1
            if label_id in primary:
                primary[label_id]["native_2d_frames"] += 1

        source = detector_by_frame.get(frame, [])
        for source_index, truth_index, overlap in associate_frame(source, truth_2d):
            source_row = source[source_index]
            truth_row = truth_2d[truth_index]
            label_id = truth_row["label_id"]
            matches_by_frame_and_id[(frame, label_id)] = {
                "track_id": source_row["track_id"],
                "iou": overlap,
                "confidence": source_row["confidence"],
            }
            match_ious.append(overlap)
            if label_id in primary:
                primary[label_id]["matched_frames"] += 1
                primary[label_id]["detector_track_ids"].add(source_row["track_id"])

        for item in labels_3d[f"{stem}.pcd"]:
            if bool(item.get("attributes", {}).get("no_eval", False)):
                continue
            label_id = str(item["label_id"])
            box = item["box"]
            forward = float(box["cx"]) + BASE_LINK_FROM_LOGICAL_RGB360_X_M
            left = float(box["cy"]) + BASE_LINK_FROM_LOGICAL_RGB360_Y_M
            width = float(box.get("w", 0.60))
            length = float(box.get("l", 0.60))
            if not all(math.isfinite(value) for value in (forward, left, width, length)):
                continue
            native_3d_rows += 1
            joined_rows += int(label_id in ids_2d)
            if label_id in primary:
                primary[label_id]["native_3d_frames"] += 1
            match = matches_by_frame_and_id.get((frame, label_id))
            tracks.setdefault(label_id, []).append(
                BridgeSample(
                    frame_index=frame,
                    time_s=time_s,
                    ego_x_m=float(pose["x_m"]),
                    ego_y_m=float(pose["y_m"]),
                    ego_yaw_rad=float(pose["yaw_rad"]),
                    forward_m=forward,
                    left_m=left,
                    radius_m=max(0.15, 0.5 * max(width, length)),
                    detector_track_id=None if match is None else str(match["track_id"]),
                )
            )

    primary_serializable = {
        target: {
            **values,
            "detector_track_ids": sorted(values["detector_track_ids"]),
            "match_coverage": values["matched_frames"] / values["native_2d_frames"] if values["native_2d_frames"] else None,
        }
        for target, values in primary.items()
    }
    return tracks, {
        "native_2d_rows": native_2d_rows,
        "native_3d_rows": native_3d_rows,
        "joined_2d_3d_rows": joined_rows,
        "detector_native_matches": len(match_ious),
        "match_iou_minimum": min(match_ious) if match_ious else None,
        "match_iou_median": sorted(match_ious)[len(match_ious) // 2] if match_ious else None,
        "primary_targets": primary_serializable,
    }, {
        "pose_samples": len(poses),
        "bag_rgb_samples": len(bag_rgb_times),
        "maximum_pose_bracket_s": max(pose_brackets),
        "maximum_pose_endpoint_s": max(pose_endpoints),
        "maximum_external_bag_rgb_delta_s": max(bag_rgb_deltas),
        "interpolated_window_frames": sum(value > 0.0 for value in pose_brackets),
    }


def contiguous_segments(samples: Sequence[BridgeSample]) -> Iterable[list[BridgeSample]]:
    current: list[BridgeSample] = []
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


def causal_frames(samples: Sequence[BridgeSample]) -> list[CausalFrame]:
    origin = samples[0].time_s
    result = []
    for sample in samples:
        observations = ()
        if sample.detector_track_id is not None:
            observations = (
                Observation(
                    track_id=sample.detector_track_id,
                    forward_m=sample.forward_m,
                    left_m=sample.left_m,
                    radius_m=sample.radius_m,
                ),
            )
        result.append(
            CausalFrame(
                time_s=sample.time_s - origin,
                ego_pose=EgoPose(
                    x_m=sample.ego_x_m,
                    y_m=sample.ego_y_m,
                    body_yaw_rad=sample.ego_yaw_rad,
                    sensor_yaw_rad=sample.ego_yaw_rad,
                ),
                observations=observations,
                person_detection_count=int(bool(observations)),
            )
        )
    return result


def evaluate_tracks(tracks: dict[str, list[BridgeSample]]) -> dict[str, Any]:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    pooled = {
        arm: ArmAccumulator()
        for arm in (Arm.B2_RADIAL_TTC, Arm.C_ROUTE_INTERSECTION)
    }
    by_target = []
    evaluable_segments = 0
    event_count = 0
    for label_id, values in sorted(tracks.items()):
        for segment_index, samples in enumerate(contiguous_segments(values)):
            if (
                len(samples) < 2
                or samples[-1].time_s - samples[0].time_s
                < config.minimum_track_span_s + HORIZON_S
            ):
                continue
            truth, contacts = future_hits(samples)
            events, known = truth_events(
                samples, truth, contacts, config.minimum_track_span_s
            )
            if not any(known):
                continue
            frames = causal_frames(samples)
            arm_rows = {}
            for arm in (Arm.B2_RADIAL_TTC, Arm.C_ROUTE_INTERSECTION):
                predictions = run_arm(frames, arm, config)
                metrics = score_arm(samples, predictions, events, known, truth)
                pooled[arm].merge(metrics)
                arm_rows[arm.value] = {
                    **metrics.to_dict(),
                    "signals": {
                        signal: sum(item.signal.value == signal for item in predictions)
                        for signal in ("ONSET", "HOLD", "CLEAR", "UNKNOWN")
                    },
                }
            by_target.append(
                {
                    "label_id": label_id,
                    "segment_index": segment_index,
                    "first_frame": samples[0].frame_index,
                    "last_frame": samples[-1].frame_index,
                    "sample_frames": len(samples),
                    "detector_matched_frames": sum(
                        sample.detector_track_id is not None for sample in samples
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
            evaluable_segments += 1
            event_count += len(events)
    pooled_rows = {arm.value: value.to_dict() for arm, value in pooled.items()}
    b2 = pooled_rows[Arm.B2_RADIAL_TTC.value]
    challenger = pooled_rows[Arm.C_ROUTE_INTERSECTION.value]
    recall_direction = (
        None
        if b2["critical_event_recall"] is None
        or challenger["critical_event_recall"] is None
        else challenger["critical_event_recall"] >= b2["critical_event_recall"]
    )
    false_direction = challenger["false_alert_segments"] <= b2["false_alert_segments"]
    return {
        "evaluable_target_segments": evaluable_segments,
        "critical_events": event_count,
        "pooled": pooled_rows,
        "directional_read": {
            "critical_recall_non_decrease": recall_direction,
            "false_alert_segments_non_increase": false_direction,
            "supportive": bool(event_count and recall_direction and false_direction),
            "role": "single_curated_window_observation_not_advancement_gate",
        },
        "by_target": by_target,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    labels = args.labels_zip.resolve(strict=True)
    timestamps_path = args.timestamps_zip.resolve(strict=True)
    bag = args.bag.resolve(strict=True)
    model = args.model.resolve(strict=True)
    image_state = ensure_images(args.images_dir.resolve())
    timestamps = load_image_timestamps(timestamps_path)

    # Truth-blind source stage is sealed before any annotation is loaded.
    detector_rows, detector_summary = run_detector_tracker(
        image_state["records"], timestamps, model, args.batch_frames
    )
    tracks_output = args.output.with_name(args.output.stem + ".tracks.jsonl").resolve()
    write_jsonl(tracks_output, detector_rows)
    tracks_sha = sha256_file(tracks_output)

    poses, bag_rgb_times, bag_authority = read_bag_pose_and_rgb(bag)
    native_tracks, association, clock = load_native_and_associate(
        labels, timestamps, poses, bag_rgb_times, detector_rows
    )
    evaluation = evaluate_tracks(native_tracks)
    return {
        "schema_version": SCHEMA,
        "status": "DTR_R0_RGB_BRIDGE_OBSERVATION_AVAILABLE",
        "claim_ceiling": CLAIM_CEILING,
        "source": {
            "dataset": "JRDB public train split",
            "sequence": SEQUENCE,
            "window": {
                "first_frame": FIRST_FRAME,
                "last_frame": LAST_FRAME,
                "frame_count": FRAME_COUNT,
                "selection": (
                    "fixed Development window containing one previously derived lateral "
                    "crossing and one oncoming relative-geometry event"
                ),
            },
            "labels_zip": str(labels),
            "labels_sha256": sha256_file(labels),
            "timestamps_zip": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "bag": str(bag),
            "bag_sha256": sha256_file(bag),
            "static_planar_transform_provenance": {
                "calibration_defaults": str(args.calibration_defaults.resolve(strict=True)),
                "calibration_defaults_sha256": sha256_file(
                    args.calibration_defaults.resolve(strict=True)
                ),
                "sensor_setup_pdf": str(args.sensor_setup_pdf.resolve(strict=True)),
                "sensor_setup_pdf_sha256": sha256_file(
                    args.sensor_setup_pdf.resolve(strict=True)
                ),
                "kinova_movo_urdf": (
                    "https://raw.githubusercontent.com/Kinovarobotics/kinova-movo/"
                    "9f515fa476f6c761829a4b5b19e769f869e1cfce/"
                    "movo_common/movo_description/urdf/movo_components/"
                    "movo_base.urdf.xacro"
                ),
                "base_link_from_logical_rgb360_planar_translation_m": [
                    BASE_LINK_FROM_LOGICAL_RGB360_X_M,
                    BASE_LINK_FROM_LOGICAL_RGB360_Y_M,
                ],
                "planar_rotation": "identity",
            },
            "images": image_state,
        },
        "truth_blind_detector_tracker": {
            "ledger": str(tracks_output),
            "ledger_sha256": tracks_sha,
            "model": str(model),
            "model_sha256": sha256_file(model),
            "configuration": {
                "tile_width": TILE_WIDTH,
                "tile_starts": list(TILE_STARTS),
                "image_size": INFERENCE_SIZE,
                "confidence": DETECTOR_CONFIDENCE,
                "nms_iou": DETECTOR_NMS_IOU,
                "max_detections": DETECTOR_MAX_DET,
                "tracker": "real_observation_adapter.CausalPersonTracker defaults",
            },
            "coverage": detector_summary,
        },
        "privileged_evaluator_bridge": {
            "association": "current-frame Hungarian IoU to native stitched 2-D label",
            "minimum_iou": MINIMUM_EVALUATOR_IOU,
            "metric_position": (
                "matched native 3-D center transformed to planar base_link; this is not "
                "runtime RGB metric-depth evidence"
            ),
            "future_truth": "future native 3-D geometry is evaluator-only",
            "association_coverage": association,
            "clock_and_pose": {**clock, **bag_authority},
        },
        "evaluation": evaluation,
        "limitations": [
            "The 143-frame sequence window is transparently curated Development evidence.",
            "Current metric depth remains privileged native 3-D geometry after RGB detection and tracking.",
            "Evaluation identity binding uses current-frame annotation IoU and is not a runtime identity capability.",
            "The bag has dynamic odom-to-base_link TF but no tf_static; the fixed planar static chain is external official calibration/URDF provenance.",
            "JRDB 3-D boxes may be source-interpolated annotations rather than direct measurements.",
            "This does not establish natural-distribution performance, phone/Android runtime behavior, user benefit, or safety.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--timestamps-zip", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-defaults", type=Path, required=True)
    parser.add_argument("--sensor-setup-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-frames", type=int, default=4)
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "output_must_be_json")
    result = run(args)
    write_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "evaluation": {
                    "critical_events": result["evaluation"]["critical_events"],
                    "pooled": result["evaluation"]["pooled"],
                    "directional_read": result["evaluation"]["directional_read"],
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
