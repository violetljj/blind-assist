#!/usr/bin/env python3
"""Materialize lossless forward RGB-D frames from one verified CrowdBot bag."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


RGB_TOPIC = "/camera_left/color/image_raw"
DEPTH_TOPIC = "/camera_left/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera_left/color/camera_info"
TF_TOPIC = "/tf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_rgb_message(message: Any) -> np.ndarray:
    if (
        message.encoding not in {"rgb8", "bgr8"}
        or message.width != 640
        or message.height != 480
        or message.step != 1920
    ):
        raise RuntimeError("unexpected RGB encoding or geometry")
    array = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height, message.step
    )[:, : message.width * 3]
    array = array.reshape(message.height, message.width, 3)
    return array if message.encoding == "rgb8" else array[:, :, ::-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--bag-receipt", required=True, type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    bundle_path = args.output_dir / "bundle.json"
    frames_path = args.output_dir / "frames.jsonl"
    tf_inventory_path = args.output_dir / "tf-frame-inventory.json"
    if bundle_path.exists() or frames_path.exists():
        raise RuntimeError("sequence bundle already exists; refusing to overwrite")
    receipt_bytes = args.bag_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    bag_sha256 = sha256_file(args.bag)
    if receipt.get("output_sha256") != bag_sha256 or receipt.get("uncompressed_bytes") != args.bag.stat().st_size:
        raise RuntimeError("bag does not match streamed extraction receipt")
    rgb_dir = args.output_dir / "rgb"
    depth_dir = args.output_dir / "aligned_depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    rgb_rows: dict[int, dict[str, Any]] = {}
    rgb_message_count = 0
    rgb_encodings: set[str] = set()
    depth_rows: dict[int, dict[str, Any]] = {}
    camera_info: dict[str, Any] | None = None
    tf_frames: dict[tuple[str, str], dict[str, Any]] = {}
    with Reader(args.bag) as reader:
        selected = [
            connection
            for connection in reader.connections
            if connection.topic in {RGB_TOPIC, DEPTH_TOPIC, CAMERA_INFO_TOPIC, TF_TOPIC}
        ]
        for connection in selected:
            if connection.msgtype not in typestore.fielddefs:
                typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for connection, timestamp, raw in reader.messages(connections=selected):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            if connection.topic == TF_TOPIC:
                for stamped in message.transforms:
                    key = (stamped.header.frame_id, stamped.child_frame_id)
                    transform = stamped.transform
                    sample = {
                        "timestamp_ns": timestamp,
                        "translation_xyz": [
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
                    row = tf_frames.setdefault(
                        key,
                        {
                            "parent_frame": key[0],
                            "child_frame": key[1],
                            "message_count": 0,
                            "first": sample,
                            "last": sample,
                        },
                    )
                    row["message_count"] += 1
                    row["last"] = sample
                continue
            if connection.topic == CAMERA_INFO_TOPIC:
                if camera_info is None:
                    camera_info = {
                        "frame_id": message.header.frame_id,
                        "width": int(message.width),
                        "height": int(message.height),
                        "K": [float(value) for value in message.K],
                        "D": [float(value) for value in message.D],
                        "distortion_model": message.distortion_model,
                    }
                continue
            if connection.topic == RGB_TOPIC:
                rgb_message_count += 1
                rgb_encodings.add(message.encoding)
                array = decode_rgb_message(message)
                path = rgb_dir / f"{timestamp}.png"
                Image.fromarray(array, mode="RGB").save(path, compress_level=1)
                # A few CrowdBot bags contain repeated RGB messages with the
                # same bag timestamp. The timestamp-named file is inherently
                # last-write-wins, so the manifest must use the same policy
                # rather than retaining stale hashes for overwritten files.
                rgb_rows[timestamp] = {"timestamp_ns": timestamp, "path": path, "sha256": sha256_file(path)}
                continue
            if message.encoding != "16UC1" or message.width != 640 or message.height != 480 or message.step != 1280:
                raise RuntimeError("unexpected aligned depth encoding or geometry")
            array = np.frombuffer(message.data, dtype="<u2").reshape(message.height, message.width)
            path = depth_dir / f"{timestamp}.png"
            Image.fromarray(array).save(path, compress_level=1)
            valid = array > 0
            depth_rows[timestamp] = {
                "path": path,
                "sha256": sha256_file(path),
                "valid_fraction": float(np.mean(valid)),
                "median_valid_raw_units": float(np.median(array[valid])) if np.any(valid) else None,
            }
    if not rgb_rows or not depth_rows or camera_info is None or not tf_frames:
        raise RuntimeError("required RGB-D or TF stream missing from sequence")
    frame_records = []
    for frame_number, rgb in enumerate(rgb_rows.values()):
        depth = depth_rows.get(rgb["timestamp_ns"])
        frame_records.append(
            {
                "frame_id": f"{frame_number:06d}",
                "source_id": args.source_id,
                "sequence_id": args.sequence_id,
                "source_capture_timestamp_ns": rgb["timestamp_ns"],
                "rgb_path": rgb["path"].relative_to(args.output_dir).as_posix(),
                "rgb_sha256": rgb["sha256"],
                "exact_aligned_depth": depth is not None,
                "aligned_depth_path": depth["path"].relative_to(args.output_dir).as_posix() if depth else None,
                "aligned_depth_sha256": depth["sha256"] if depth else None,
                "aligned_depth_valid_fraction": depth["valid_fraction"] if depth else None,
                "aligned_depth_median_raw_units": depth["median_valid_raw_units"] if depth else None,
            }
        )
    with frames_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in frame_records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    tf_inventory = {
        "schema": "blindassist_crowdbot_tf_frame_inventory_r1",
        "authority": "camera_extrinsic_and_causal_route_input_support_not_future_route_truth",
        "candidate_outputs_executed": False,
        "source_id": args.source_id,
        "sequence_id": args.sequence_id,
        "frame_pairs": [tf_frames[key] for key in sorted(tf_frames)],
    }
    tf_inventory_path.write_text(json.dumps(tf_inventory, indent=2) + "\n", encoding="utf-8")
    bundle = {
        "schema": "blindassist_crowdbot_rgbd_sequence_bundle_r1",
        "authority": "sealed_holdout_input_not_truth_not_candidate_score",
        "candidate_outputs_executed": False,
        "source_id": args.source_id,
        "sequence_id": args.sequence_id,
        "bag": {"path": args.bag.as_posix(), "sha256": bag_sha256, "bytes": args.bag.stat().st_size},
        "bag_receipt": {"path": args.bag_receipt.as_posix(), "sha256": hashlib.sha256(receipt_bytes).hexdigest()},
        "camera_info": camera_info,
        "rgb_frame_count": len(rgb_rows),
        "rgb_message_count": rgb_message_count,
        "source_rgb_encodings": sorted(rgb_encodings),
        "materialized_rgb_encoding": "rgb8_png",
        "bgr8_normalization": "channel_reverse_to_rgb_before_lossless_png",
        "rgb_duplicate_extra_message_count": rgb_message_count - len(rgb_rows),
        "rgb_duplicate_timestamp_policy": "same_bag_timestamp_last_message_wins_v1",
        "aligned_depth_frame_count": len(depth_rows),
        "exact_rgb_depth_frame_count": sum(row["exact_aligned_depth"] for row in frame_records),
        "frames_path": frames_path.as_posix(),
        "frames_sha256": sha256_file(frames_path),
        "tf_frame_inventory_path": tf_inventory_path.as_posix(),
        "tf_frame_inventory_sha256": sha256_file(tf_inventory_path),
        "tf_frame_pair_count": len(tf_frames),
        "face_defacing_authority": "official_defaced_archive_with_sample_only_visual_confirmation",
        "external_redistribution_authority": False,
    }
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rgb": len(rgb_rows), "depth": len(depth_rows), "exact": bundle["exact_rgb_depth_frame_count"], "frames_sha256": bundle["frames_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
