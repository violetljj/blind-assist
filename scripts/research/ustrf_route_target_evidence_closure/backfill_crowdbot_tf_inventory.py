#!/usr/bin/env python3
"""Backfill a candidate-blind TF frame inventory into an existing verified RGB-D bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


TF_TOPIC = "/tf"
CAMERA_INFO_TOPIC = "/camera_left/color/camera_info"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--bag-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    bundle_path = args.output_dir / "bundle.json"
    frames_path = args.output_dir / "frames.jsonl"
    tf_path = args.output_dir / "tf-frame-inventory.json"
    if not bundle_path.is_file() or not frames_path.is_file():
        raise RuntimeError("verified RGB-D bundle is missing")
    if tf_path.exists():
        raise RuntimeError("TF frame inventory already exists; refusing to overwrite")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    receipt = json.loads(args.bag_receipt.read_text(encoding="utf-8"))
    bag_sha = sha256_file(args.bag)
    if bundle.get("candidate_outputs_executed") is not False:
        raise RuntimeError("bundle is not candidate blind")
    if bundle.get("frames_sha256") != sha256_file(frames_path):
        raise RuntimeError("existing RGB-D frames hash mismatch")
    if receipt.get("output_sha256") != bag_sha or bundle.get("bag", {}).get("sha256") != bag_sha:
        raise RuntimeError("bag, receipt, and existing bundle do not match")
    typestore = get_typestore(Stores.ROS1_NOETIC)
    frame_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    camera_frame_id: str | None = None
    with Reader(args.bag) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in {TF_TOPIC, CAMERA_INFO_TOPIC}
        ]
        if not any(connection.topic == TF_TOPIC for connection in connections):
            raise RuntimeError("bag has no /tf topic")
        for connection in connections:
            if connection.msgtype not in typestore.fielddefs:
                typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for connection, timestamp, raw in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            if connection.topic == CAMERA_INFO_TOPIC:
                if camera_frame_id is None:
                    camera_frame_id = message.header.frame_id
                continue
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
                row = frame_pairs.setdefault(
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
    if camera_frame_id is None:
        raise RuntimeError("bag has no color camera frame ID")
    payload = {
        "schema": "blindassist_crowdbot_tf_frame_inventory_r1",
        "authority": "camera_extrinsic_and_causal_route_input_support_not_future_route_truth",
        "candidate_outputs_executed": False,
        "source_id": bundle["source_id"],
        "sequence_id": bundle["sequence_id"],
        "frame_pairs": [frame_pairs[key] for key in sorted(frame_pairs)],
        "backfill_bag_receipt_sha256": sha256_file(args.bag_receipt),
    }
    write_json_atomic(tf_path, payload)
    bundle.update({
        "tf_frame_inventory_path": tf_path.as_posix(),
        "tf_frame_inventory_sha256": sha256_file(tf_path),
        "tf_frame_pair_count": len(frame_pairs),
    })
    bundle.setdefault("camera_info", {})["frame_id"] = camera_frame_id
    write_json_atomic(bundle_path, bundle)
    print(json.dumps({
        "status": "TF_INVENTORY_BACKFILLED",
        "frame_pair_count": len(frame_pairs),
        "tf_frame_inventory_sha256": bundle["tf_frame_inventory_sha256"],
        "bundle_sha256": sha256_file(bundle_path),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
