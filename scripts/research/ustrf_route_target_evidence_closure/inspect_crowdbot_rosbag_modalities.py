#!/usr/bin/env python3
"""Inspect forward RGB-D modalities in one CrowdBot ROS1 bag."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


RGB_TOPIC = "/camera_left/color/image_raw"
DEPTH_TOPIC = "/camera_left/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera_left/color/camera_info"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float | None:
    return float(np.percentile(values, quantile)) if values else None


def decode_rgb_message(message: Any) -> np.ndarray:
    if (
        message.encoding not in {"rgb8", "bgr8"}
        or message.width != 640
        or message.height != 480
        or message.step != 1920
    ):
        raise RuntimeError("unexpected RGB encoding or geometry")
    rgb = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height, message.step
    )[:, : message.width * 3]
    rgb = rgb.reshape(message.height, message.width, 3)
    return rgb if message.encoding == "rgb8" else rgb[:, :, ::-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preview-dir", required=True, type=Path)
    args = parser.parse_args()
    typestore = get_typestore(Stores.ROS1_NOETIC)
    topic_counts: Counter[str] = Counter()
    topic_types: dict[str, str] = {}
    topic_first: dict[str, int] = {}
    topic_last: dict[str, int] = {}
    rgb_rows: list[tuple[int, Any]] = []
    depth_rows: list[tuple[int, Any]] = []
    camera_info: Any | None = None
    with Reader(args.bag) as reader:
        for connection, timestamp, raw in reader.messages():
            topic_counts[connection.topic] += 1
            topic_types[connection.topic] = connection.msgtype
            topic_first.setdefault(connection.topic, timestamp)
            topic_last[connection.topic] = timestamp
            if connection.topic not in {RGB_TOPIC, DEPTH_TOPIC, CAMERA_INFO_TOPIC}:
                continue
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            if connection.topic == RGB_TOPIC:
                rgb_rows.append((timestamp, message))
            elif connection.topic == DEPTH_TOPIC:
                depth_rows.append((timestamp, message))
            elif camera_info is None:
                camera_info = message
    if not rgb_rows or not depth_rows or camera_info is None:
        raise RuntimeError("required forward RGB-D topics are incomplete")
    rgb_timestamps = np.asarray([row[0] for row in rgb_rows], dtype=np.int64)
    depth_timestamps = np.asarray([row[0] for row in depth_rows], dtype=np.int64)
    nearest_depth_deltas_ms: list[float] = []
    exact_matches = 0
    depth_by_timestamp = {timestamp: message for timestamp, message in depth_rows}
    for timestamp in rgb_timestamps:
        insertion = int(np.searchsorted(depth_timestamps, timestamp))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(depth_timestamps)]
        delta = min(abs(int(depth_timestamps[index]) - int(timestamp)) for index in candidates)
        nearest_depth_deltas_ms.append(delta / 1_000_000.0)
        if int(timestamp) in depth_by_timestamp:
            exact_matches += 1
    preview_indices = sorted({0, len(rgb_rows) // 2, len(rgb_rows) - 1})
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    previews = []
    for index in preview_indices:
        timestamp, message = rgb_rows[index]
        rgb = decode_rgb_message(message)
        preview_path = args.preview_dir / f"rgb-{index:05d}-{timestamp}.png"
        Image.fromarray(rgb, mode="RGB").save(preview_path)
        depth = depth_by_timestamp.get(timestamp)
        depth_stats = None
        if depth is not None:
            if depth.encoding != "16UC1" or depth.width != 640 or depth.height != 480 or depth.step != 1280:
                raise RuntimeError("unexpected aligned depth encoding or geometry")
            depth_array = np.frombuffer(depth.data, dtype="<u2").reshape(depth.height, depth.width)
            valid = depth_array > 0
            depth_stats = {
                "valid_fraction": float(np.mean(valid)),
                "median_valid_raw_units": float(np.median(depth_array[valid])) if np.any(valid) else None,
            }
        previews.append(
            {
                "rgb_index": index,
                "timestamp_ns": timestamp,
                "path": preview_path.as_posix(),
                "sha256": sha256_file(preview_path),
                "exact_aligned_depth": depth is not None,
                "depth_stats": depth_stats,
            }
        )
    topics = [
        {
            "topic": topic,
            "message_type": topic_types[topic],
            "message_count": count,
            "duration_seconds": (topic_last[topic] - topic_first[topic]) / 1_000_000_000.0,
        }
        for topic, count in sorted(topic_counts.items())
    ]
    payload = {
        "schema": "blindassist_crowdbot_rosbag_modality_probe_r1",
        "authority": "holdout_source_modality_probe_not_truth_not_candidate_score",
        "candidate_outputs_executed": False,
        "bag": {"path": args.bag.as_posix(), "sha256": sha256_file(args.bag), "bytes": args.bag.stat().st_size},
        "required_modalities": {
            "rgb": {"topic": RGB_TOPIC, "count": len(rgb_rows), "encoding": rgb_rows[0][1].encoding, "width": rgb_rows[0][1].width, "height": rgb_rows[0][1].height},
            "materialized_rgb_encoding": "rgb8_png",
            "bgr8_normalization": "channel_reverse_to_rgb_before_preview_or_materialization",
            "aligned_depth": {"topic": DEPTH_TOPIC, "count": len(depth_rows), "encoding": depth_rows[0][1].encoding, "width": depth_rows[0][1].width, "height": depth_rows[0][1].height},
            "camera_info": {"topic": CAMERA_INFO_TOPIC, "width": camera_info.width, "height": camera_info.height, "k": [float(value) for value in camera_info.K]},
        },
        "rgb_depth_sync": {
            "exact_timestamp_match_count": exact_matches,
            "rgb_count": len(rgb_rows),
            "nearest_delta_ms_p50": percentile(nearest_depth_deltas_ms, 50),
            "nearest_delta_ms_p95": percentile(nearest_depth_deltas_ms, 95),
            "nearest_delta_ms_max": max(nearest_depth_deltas_ms),
        },
        "previews": previews,
        "topics": topics,
        "face_defacing_visual_review": "pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rgb": len(rgb_rows), "depth": len(depth_rows), "exact_matches": exact_matches, "p95_ms": payload["rgb_depth_sync"]["nearest_delta_ms_p95"], "preview_count": len(previews)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
