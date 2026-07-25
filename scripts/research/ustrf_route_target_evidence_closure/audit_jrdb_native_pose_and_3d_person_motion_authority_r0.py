#!/usr/bin/env python3
"""Bounded JRDB native multisensor metadata and directory audit."""
from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "blindassist_ustrf_jrdb_native_pose_and_3d_person_motion_authority_audit_r0"
CONFIG_SCHEMA = f"{SCHEMA}_config"
STAGE = "JRDB_NATIVE_POSE_AND_3D_PERSON_MOTION_AUTHORITY_AUDIT_R0"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RangeReader:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.bytes_read = 0
        self.requests: list[dict[str, Any]] = []

    def get(self, url: str, start: int, end: int) -> bytes:
        require(0 <= start <= end, "invalid_range")
        requested = end - start + 1
        require(self.bytes_read + requested <= self.budget, "network_budget_exceeded")
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            require(response.status == 206, f"range_status:{response.status}")
            payload = response.read(requested + 1)
        require(len(payload) == requested, "range_length_drift")
        self.bytes_read += requested
        self.requests.append({"url": url, "start": start, "end": end, "bytes": requested})
        return payload


def zip64_values(extra: bytes, compressed: int, uncompressed: int, offset: int) -> tuple[int, int, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        field = extra[cursor + 4 : cursor + 4 + field_size]
        cursor += 4 + field_size
        if field_id != 1:
            continue
        pos = 0
        if uncompressed == 0xFFFFFFFF:
            uncompressed = struct.unpack_from("<Q", field, pos)[0]
            pos += 8
        if compressed == 0xFFFFFFFF:
            compressed = struct.unpack_from("<Q", field, pos)[0]
            pos += 8
        if offset == 0xFFFFFFFF:
            offset = struct.unpack_from("<Q", field, pos)[0]
        break
    return compressed, uncompressed, offset


def parse_central(payload: bytes) -> list[dict[str, Any]]:
    cursor = 0
    members: list[dict[str, Any]] = []
    while cursor < len(payload):
        require(payload[cursor : cursor + 4] == b"PK\x01\x02", f"central_signature:{cursor}")
        values = struct.unpack_from("<4s6H3L5H2L", payload, cursor)
        flags, method, crc32 = values[3], values[4], values[7]
        compressed, uncompressed = values[8], values[9]
        name_len, extra_len, comment_len, offset = values[10], values[11], values[12], values[16]
        start = cursor + 46
        name = payload[start : start + name_len].decode("utf-8")
        extra = payload[start + name_len : start + name_len + extra_len]
        compressed, uncompressed, offset = zip64_values(extra, compressed, uncompressed, offset)
        members.append(
            {
                "name": name,
                "flags": flags,
                "method": method,
                "crc32": crc32,
                "compressed": compressed,
                "uncompressed": uncompressed,
                "offset": offset,
            }
        )
        cursor = start + name_len + extra_len + comment_len
    require(cursor == len(payload), "central_length_drift")
    return members


def fetch_member(reader: RangeReader, url: str, member: dict[str, Any]) -> bytes:
    offset = int(member["offset"])
    header = reader.get(url, offset, offset + 29)
    values = struct.unpack("<4s5H3L2H", header)
    require(values[0] == b"PK\x03\x04", "local_header_signature")
    name_len, extra_len = values[-2], values[-1]
    tail = reader.get(url, offset + 30, offset + 30 + name_len + extra_len - 1)
    require(tail[:name_len].decode("utf-8") == member["name"], "local_name_drift")
    data_start = offset + 30 + name_len + extra_len
    compressed = reader.get(url, data_start, data_start + int(member["compressed"]) - 1)
    if member["method"] == 0:
        raw = compressed
    elif member["method"] == 8:
        raw = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported_method:{member['method']}")
    require(len(raw) == member["uncompressed"], "uncompressed_size_drift")
    require(binascii.crc32(raw) & 0xFFFFFFFF == member["crc32"], "member_crc_drift")
    return raw


def frame_stem(value: str) -> str:
    return PurePosixPath(value).stem


def audit(repo: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["schema"] == CONFIG_SCHEMA and config["stage"] == STAGE, "config_identity")
    for binding in config["local_bindings"].values():
        path = repo / binding["path"]
        require(path.is_file(), f"missing:{path}")
        require(sha256_file(path) == binding["sha256"], f"sha_drift:{path}")

    reader = RangeReader(int(config["resource_gate"]["maximum_network_bytes"]))
    archives: dict[str, list[dict[str, Any]]] = {}
    for name, remote in config["remote_archives"].items():
        start = int(remote["central_directory_offset"])
        size = int(remote["central_directory_size"])
        archives[name] = parse_central(reader.get(remote["url"], start, start + size - 1))

    sequence = config["canary"]["sequence"]
    timestamp_path = repo / config["local_bindings"]["train_timestamps"]["path"]
    with zipfile.ZipFile(timestamp_path) as bundle:
        img = json.loads(bundle.read(f"timestamps/{sequence}/frames_img.json"))
        pc = json.loads(bundle.read(f"timestamps/{sequence}/frames_pc.json"))
        timestamp_sequences = sorted(
            {
                name.split("/")[1]
                for name in bundle.namelist()
                if name.startswith("timestamps/") and name.endswith("/frames_img.json")
            }
        )
    image_timestamp_frames = {
        frame_stem(row["cameras"][0]["url"]): float(row["timestamp"]) for row in img["data"]
    }
    point_timestamp_frames = {
        frame_stem(row["pointclouds"][0]["url"]): float(row["timestamp"]) for row in pc["data"]
    }

    stream_frames: dict[str, set[str]] = {}
    for camera in ("image_0", "image_2", "image_4", "image_6", "image_8", "image_stitched"):
        prefix = f"images/{camera}/{sequence}/"
        stream_frames[camera] = {
            frame_stem(member["name"])
            for member in archives["images"]
            if member["name"].startswith(prefix) and member["name"].endswith(".jpg")
        }
    for lidar in ("upper_velodyne", "lower_velodyne"):
        prefix = f"pointclouds/{lidar}/{sequence}/"
        stream_frames[lidar] = {
            frame_stem(member["name"])
            for member in archives["pointclouds"]
            if member["name"].startswith(prefix) and member["name"].endswith(".pcd")
        }

    label_summaries: dict[str, Any] = {}
    label_frames: dict[str, set[str]] = {}
    for role, member_name in {
        "labels_2d_stitched": f"labels/labels_2d_stitched/{sequence}.json",
        "labels_3d": f"labels/labels_3d/{sequence}.json",
    }.items():
        matches = [member for member in archives["labels"] if member["name"] == member_name]
        require(len(matches) == 1, f"label_member_count:{role}")
        payload = json.loads(fetch_member(reader, config["remote_archives"]["labels"]["url"], matches[0]))
        label_frames[role] = {frame_stem(key) for key in payload["labels"]}
        ids = {obj["label_id"] for objects in payload["labels"].values() for obj in objects}
        label_summaries[role] = {
            "frames": len(payload["labels"]),
            "objects": sum(len(objects) for objects in payload["labels"].values()),
            "unique_track_ids": len(ids),
            "uses_label_id": all("label_id" in obj for objects in payload["labels"].values() for obj in objects),
        }

    expected_window = {
        f"{frame:06d}"
        for frame in range(
            int(config["canary"]["window_first_frame"]),
            int(config["canary"]["window_last_frame"]) + 1,
        )
    }
    all_frame_sets = {
        "timestamp_img": set(image_timestamp_frames),
        "timestamp_pc": set(point_timestamp_frames),
        **stream_frames,
        **label_frames,
    }
    window_complete = {name: expected_window <= frames for name, frames in all_frame_sets.items()}

    rosbag_sequences = {
        PurePosixPath(member["name"]).stem
        for member in archives["rosbags"]
        if member["name"].endswith(".bag")
    }
    calibration_defaults = (repo / config["local_bindings"]["calibration_defaults"]["path"]).read_text(encoding="utf-8")
    calibration_lidars = (repo / config["local_bindings"]["calibration_lidars"]["path"]).read_text(encoding="utf-8")
    pose_consumer = (repo / config["local_bindings"]["third_party_pose_consumer"]["path"]).read_text(encoding="utf-8")
    lidar_consumer = (repo / config["local_bindings"]["third_party_lidar_consumer"]["path"]).read_text(encoding="utf-8")
    static_transform_contract = all(
        token in calibration_defaults + calibration_lidars
        for token in ("base_link", "occam", "upper2ego", "lower2upper")
    )
    third_party_pose_contract = all(
        token in pose_consumer for token in ('"odom"', '"base_link"', '"base_chassis_link"', '"/tf"', '"/tf_static"')
    )
    third_party_lidar_contract = all(
        token in lidar_consumer
        for token in ("/upper_velodyne/velodyne_points", "/lower_velodyne/velodyne_points", "target_frame = 'odom'")
    )

    deltas = [
        image_timestamp_frames[key] - point_timestamp_frames[key]
        for key in sorted(set(image_timestamp_frames) & set(point_timestamp_frames))
    ]
    p1_directory_ready = (
        sequence in rosbag_sequences
        and all(window_complete.values())
        and static_transform_contract
        and third_party_pose_contract
        and third_party_lidar_contract
    )
    pose_imu_payload_ready = (
        config["authority_limits"]["native_pose_topic_payload_audited"]
        and config["authority_limits"]["imu_topic_payload_audited"]
    )
    if p1_directory_ready and pose_imu_payload_ready:
        terminal = "NATIVE_POSE_AND_3D_PERSON_MOTION_AUTHORITY_PRESENT"
    elif p1_directory_ready:
        terminal = "NATIVE_MULTISENSOR_CANARY_ELIGIBLE_POSE_IMU_TOPIC_AUDIT_REQUIRED"
    else:
        terminal = "NATIVE_MULTISENSOR_BINDING_ABSENT"

    return {
        "schema": SCHEMA,
        "stage": STAGE,
        "status": "AUDIT_COMPLETE",
        "terminal_state": terminal,
        "config_sha256": sha256_file(config_path),
        "network": {
            "bytes_read": reader.bytes_read,
            "budget": reader.budget,
            "full_archive_downloaded": False,
            "requests": reader.requests,
        },
        "archive_inventory": {
            name: {"entries": len(members), "sequence_matches": sum(sequence in m["name"] for m in members)}
            for name, members in archives.items()
        },
        "coverage": {
            "timestamp_sequences": len(timestamp_sequences),
            "rosbag_sequences": len(rosbag_sequences),
            "timestamp_sequences_with_rosbag": len(set(timestamp_sequences) & rosbag_sequences),
            "selected_sequence_has_rosbag": sequence in rosbag_sequences,
            "selected_sequence_frames": {name: len(frames) for name, frames in all_frame_sets.items()},
            "selected_sequence_missing_vs_image_timestamp": {
                name: sorted(set(image_timestamp_frames) - frames) for name, frames in all_frame_sets.items()
            },
            "first_120_complete": window_complete,
            "image_minus_pointcloud_timestamp_seconds": {
                "minimum": min(deltas),
                "median": sorted(deltas)[len(deltas) // 2],
                "maximum": max(deltas),
            },
        },
        "person_tracks": label_summaries,
        "contracts": {
            "static_robot_camera_lidar_transform": static_transform_contract,
            "third_party_dynamic_pose_consumer": third_party_pose_contract,
            "third_party_lidar_consumer": third_party_lidar_contract,
            "person_join_key": "sequence + frame stem + label_id",
            "sensor_join_key": "sequence + frame stem + source timestamp",
        },
        "authority": {
            "official_source_claims": config["source_claims"],
            "native_pose_topic_payload_audited": False,
            "imu_topic_payload_audited": False,
            "p2_authorized": terminal == "NATIVE_POSE_AND_3D_PERSON_MOTION_AUTHORITY_PRESENT",
            "route_event_safety_authority": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.repo.resolve(), args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"terminal_state": receipt["terminal_state"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
