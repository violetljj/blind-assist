#!/usr/bin/env python3
"""Materialize and audit the frozen JRDB native multisensor geometry canary."""
from __future__ import annotations

import argparse
import binascii
import bisect
import concurrent.futures
import hashlib
import json
import math
import re
import struct
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

STAGE = "JRDB_SINGLE_SEQUENCE_NATIVE_MULTISENSOR_PERSON_GEOMETRY_CANARY_R0"
CONFIG_SCHEMA = "blindassist_ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0_config"
PACKET_SCHEMA = "blindassist_ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0_observation_packet"
RECEIPT_SCHEMA = "blindassist_ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0_receipt"
MATERIALIZATION_SCHEMA = "blindassist_ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0_materialization"


class GateError(RuntimeError):
    """A named, fail-closed protocol violation."""

    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}:{detail}")
        self.gate = gate
        self.detail = detail


def require(value: bool, gate: str, detail: str) -> None:
    if not value:
        raise GateError(gate, detail)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_canonical(path: Path, value: Any) -> None:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, "packet", f"immutable_output_drift:{path}")
        return
    path.write_bytes(payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def norm(value: str) -> str:
    return value.lstrip("/")


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def frame_stem(value: str) -> str:
    return PurePosixPath(value).stem


def _zip64_values(extra: bytes, compressed: int, uncompressed: int, offset: int) -> tuple[int, int, int]:
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
        require(payload[cursor : cursor + 4] == b"PK\x01\x02", "packet", f"central_signature:{cursor}")
        values = struct.unpack_from("<4s6H3L5H2L", payload, cursor)
        flags, method, crc32 = values[3], values[4], values[7]
        compressed, uncompressed = values[8], values[9]
        name_len, extra_len, comment_len, offset = values[10], values[11], values[12], values[16]
        start = cursor + 46
        name = payload[start : start + name_len].decode("utf-8")
        extra = payload[start + name_len : start + name_len + extra_len]
        compressed, uncompressed, offset = _zip64_values(extra, compressed, uncompressed, offset)
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
    require(cursor == len(payload), "packet", "central_length_drift")
    return members


def get_range(url: str, start: int, end: int, expected_etag: str) -> bytes:
    require(0 <= start <= end, "packet", "invalid_range")
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        require(response.status == 206, "packet", f"range_status:{response.status}")
        content_range = response.headers.get("Content-Range", "")
        require(content_range.startswith(f"bytes {start}-{end}/"), "packet", f"content_range:{content_range}")
        observed_etag = response.headers.get("ETag")
        if observed_etag:
            require(observed_etag == expected_etag, "packet", f"etag_drift:{observed_etag}")
        payload = response.read(end - start + 2)
    require(len(payload) == end - start + 1, "packet", "range_length_drift")
    return payload


def fetch_member(archive: dict[str, Any], member: dict[str, Any]) -> tuple[bytes, int]:
    header = get_range(
        archive["url"],
        int(member["offset"]),
        int(member["offset"]) + 29,
        archive["etag"],
    )
    values = struct.unpack("<4s5H3L2H", header)
    require(values[0] == b"PK\x03\x04", "packet", f"local_header:{member['name']}")
    name_len, extra_len = values[-2], values[-1]
    start = int(member["offset"]) + 30
    end = start + name_len + extra_len + int(member["compressed"]) - 1
    tail = get_range(archive["url"], start, end, archive["etag"])
    require(tail[:name_len].decode("utf-8") == member["name"], "packet", f"local_name:{member['name']}")
    compressed = tail[name_len + extra_len :]
    if member["method"] == 0:
        raw = compressed
    elif member["method"] == 8:
        raw = zlib.decompress(compressed, -15)
    else:
        raise GateError("packet", f"unsupported_zip_method:{member['method']}")
    require(len(raw) == int(member["uncompressed"]), "packet", f"size:{member['name']}")
    require(binascii.crc32(raw) & 0xFFFFFFFF == int(member["crc32"]), "packet", f"crc:{member['name']}")
    return raw, 30 + len(tail)


def selected_members(config: dict[str, Any]) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], int]:
    sequence = config["canary"]["sequence"]
    first = int(config["canary"]["window_first_frame"])
    last = int(config["canary"]["window_last_frame"])
    expected_frames = {f"{idx:06d}" for idx in range(first, last + 1)}
    selected: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    network_bytes = 0
    for role in ("labels", "images", "pointclouds"):
        archive = config["remote_archives"][role]
        start = int(archive["central_directory_offset"])
        size = int(archive["central_directory_size"])
        central = get_range(archive["url"], start, start + size - 1, archive["etag"])
        network_bytes += len(central)
        rows = parse_central(central)
        if role == "labels":
            names = {
                f"labels/labels_2d_stitched/{sequence}.json",
                f"labels/labels_3d/{sequence}.json",
            }
            chosen = [row for row in rows if row["name"] in names]
            require({row["name"] for row in chosen} == names, "label", "label_documents_missing")
        elif role == "images":
            prefix = f"images/image_stitched/{sequence}/"
            chosen = [
                row
                for row in rows
                if row["name"].startswith(prefix) and frame_stem(row["name"]) in expected_frames
            ]
            require({frame_stem(row["name"]) for row in chosen} == expected_frames, "packet", "rgb_frames_missing")
        else:
            chosen = []
            for lidar in ("upper_velodyne", "lower_velodyne"):
                prefix = f"pointclouds/{lidar}/{sequence}/"
                lidar_rows = [
                    row
                    for row in rows
                    if row["name"].startswith(prefix) and frame_stem(row["name"]) in expected_frames
                ]
                require(
                    {frame_stem(row["name"]) for row in lidar_rows} == expected_frames,
                    "pointcloud",
                    f"{lidar}_frames_missing",
                )
                chosen.extend(lidar_rows)
        selected.extend((role, archive, row) for row in chosen)
    require(len(selected) == 362, "packet", f"selected_member_count:{len(selected)}")
    payload_bytes = sum(int(row["uncompressed"]) for _, _, row in selected)
    require(
        payload_bytes <= int(config["resource_gate"]["maximum_payload_bytes"]),
        "packet",
        f"payload_budget:{payload_bytes}",
    )
    projected = network_bytes + sum(int(row["compressed"]) + 4096 for _, _, row in selected)
    require(
        projected <= int(config["resource_gate"]["maximum_network_bytes"]),
        "packet",
        f"network_budget_projection:{projected}",
    )
    return selected, network_bytes


def materialize_payload(repo: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload_root = repo / config["outputs"]["payload_root"]
    payload_root.mkdir(parents=True, exist_ok=True)
    members, central_bytes = selected_members(config)

    def load_one(item: tuple[str, dict[str, Any], dict[str, Any]]) -> tuple[dict[str, Any], int, bool]:
        role, archive, member = item
        path = payload_root / PurePosixPath(member["name"])
        if path.exists():
            raw = path.read_bytes()
            require(len(raw) == int(member["uncompressed"]), "packet", f"cached_size:{member['name']}")
            require(
                binascii.crc32(raw) & 0xFFFFFFFF == int(member["crc32"]),
                "packet",
                f"cached_crc:{member['name']}",
            )
            network = 0
            reused = True
        else:
            raw, network = fetch_member(archive, member)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            reused = False
        return (
            {
                "role": role,
                "archive_url": archive["url"],
                "member": member["name"],
                "path": path.relative_to(repo).as_posix(),
                "bytes": len(raw),
                "crc32": int(member["crc32"]),
                "sha256": sha256_bytes(raw),
            },
            network,
            reused,
        )

    results: list[tuple[dict[str, Any], int, bool]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(config["resource_gate"]["maximum_workers"])) as pool:
        futures = [pool.submit(load_one, item) for item in members]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    files = sorted((row for row, _, _ in results), key=lambda row: row["member"])
    network_bytes = central_bytes + sum(value for _, value, _ in results)
    require(
        network_bytes <= int(config["resource_gate"]["maximum_network_bytes"]),
        "packet",
        f"network_budget_actual:{network_bytes}",
    )
    transport = {
        "central_directory_bytes": central_bytes,
        "network_bytes": network_bytes,
        "payload_bytes": sum(row["bytes"] for row in files),
        "members": len(files),
        "reused_members": sum(1 for _, _, reused in results if reused),
        "full_archive_downloaded": False,
        "second_sequence_accessed": False,
    }
    return files, transport


def jpeg_size(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    require(payload[:2] == b"\xff\xd8", "packet", f"jpeg_magic:{path.name}")
    cursor = 2
    while cursor + 9 <= len(payload):
        if payload[cursor] != 0xFF:
            cursor += 1
            continue
        marker = payload[cursor + 1]
        cursor += 2
        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            break
        require(cursor + 2 <= len(payload), "packet", f"jpeg_segment:{path.name}")
        length = int.from_bytes(payload[cursor : cursor + 2], "big")
        require(length >= 2 and cursor + length <= len(payload), "packet", f"jpeg_length:{path.name}")
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height = int.from_bytes(payload[cursor + 3 : cursor + 5], "big")
            width = int.from_bytes(payload[cursor + 5 : cursor + 7], "big")
            require(width > 0 and height > 0, "packet", f"jpeg_dimensions:{path.name}")
            return width, height
        cursor += length
    raise GateError("packet", f"jpeg_sof_missing:{path.name}")


def pcd_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        lines: list[str] = []
        for _ in range(64):
            line = handle.readline()
            require(line, "pointcloud", f"pcd_header_eof:{path.name}")
            text = line.decode("ascii").strip()
            lines.append(text)
            if text.upper().startswith("DATA "):
                break
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        values[key.upper()] = value.strip()
    fields = values.get("FIELDS", "").split()
    points = int(values.get("POINTS", "0"))
    width = int(values.get("WIDTH", "0"))
    height = int(values.get("HEIGHT", "0"))
    require({"x", "y", "z"}.issubset(fields), "pointcloud", f"pcd_xyz:{path.name}")
    require(points > 0 and width > 0 and height > 0, "pointcloud", f"pcd_empty:{path.name}")
    require(values.get("DATA") in {"ascii", "binary", "binary_compressed"}, "pointcloud", f"pcd_data:{path.name}")
    return {"fields": fields, "points": points, "width": width, "height": height, "data": values["DATA"]}


def q_normalize(q: Iterable[float]) -> list[float]:
    values = [float(x) for x in q]
    magnitude = math.sqrt(sum(x * x for x in values))
    require(magnitude > 0 and math.isfinite(magnitude), "static", "quaternion_invalid")
    return [x / magnitude for x in values]


def q_slerp(q0: Iterable[float], q1: Iterable[float], weight: float) -> list[float]:
    a = q_normalize(q0)
    b = q_normalize(q1)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b = [-x for x in b]
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return q_normalize([(1 - weight) * x + weight * y for x, y in zip(a, b)])
    theta = math.acos(dot)
    scale = math.sin(theta)
    return [
        math.sin((1 - weight) * theta) / scale * x + math.sin(weight * theta) / scale * y
        for x, y in zip(a, b)
    ]


def q_matrix(q: Iterable[float]) -> list[list[float]]:
    x, y, z, w = q_normalize(q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def transform(rotation: list[list[float]], translation: Iterable[float]) -> dict[str, Any]:
    return {"rotation": rotation, "translation": [float(x) for x in translation]}


def transform_from_q(translation: Iterable[float], quaternion: Iterable[float]) -> dict[str, Any]:
    return transform(q_matrix(quaternion), translation)


def mat_vec(matrix: list[list[float]], vector: Iterable[float]) -> list[float]:
    values = list(vector)
    return [sum(matrix[row][col] * values[col] for col in range(3)) for row in range(3)]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3)] for row in range(3)]


def compose(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    rotated = mat_vec(a["rotation"], b["translation"])
    return transform(
        mat_mul(a["rotation"], b["rotation"]),
        [rotated[idx] + a["translation"][idx] for idx in range(3)],
    )


def inverse(value: dict[str, Any]) -> dict[str, Any]:
    rotation = [[value["rotation"][col][row] for col in range(3)] for row in range(3)]
    translated = mat_vec(rotation, [-x for x in value["translation"]])
    return transform(rotation, translated)


def apply_transform(value: dict[str, Any], point: Iterable[float]) -> list[float]:
    rotated = mat_vec(value["rotation"], point)
    return [rotated[idx] + value["translation"][idx] for idx in range(3)]


def matrix_delta(a: list[list[float]], b: list[list[float]]) -> float:
    return max(abs(a[row][col] - b[row][col]) for row in range(3) for col in range(3))


def parse_default_calibration(path: Path, role: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    pattern = rf"(?ms)^\s{{2}}{re.escape(role)}:\s*.*?translation:\s*\[([^\]]+)\].*?rotation:\s*\[([^\]]+)\]"
    match = re.search(pattern, text)
    require(match is not None, "static", f"calibration_role:{role}")
    translation = [float(value.strip()) for value in match.group(1).split(",")]
    rotations = [float(value.strip()) for value in match.group(2).split(",")]
    require(len(translation) == 3 and len(rotations) == 3, "static", f"calibration_shape:{role}")
    theta = rotations[-1]
    rotation = [
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ]
    # Official toolkit implements p_rgb = R * (p_lidar - translation).
    offset = mat_vec(rotation, [-x for x in translation])
    return transform(rotation, offset)


def interpolate(samples: list[dict[str, Any]], target_ns: int, max_bracket: float, max_endpoint: float, gate: str) -> dict[str, Any]:
    times = [row["timestamp_ns"] for row in samples]
    index = bisect.bisect_left(times, target_ns)
    require(0 < index < len(samples), gate, f"unbracketed:{target_ns}")
    left, right = samples[index - 1], samples[index]
    span = (right["timestamp_ns"] - left["timestamp_ns"]) / 1e9
    left_delta = (target_ns - left["timestamp_ns"]) / 1e9
    right_delta = (right["timestamp_ns"] - target_ns) / 1e9
    require(span <= max_bracket + 1e-12, gate, f"bracket:{span}")
    require(max(left_delta, right_delta) <= max_endpoint + 1e-12, gate, f"endpoint:{max(left_delta, right_delta)}")
    require(span > 0 and left_delta >= 0 and right_delta >= 0, gate, "invalid_bracket")
    weight = left_delta / span
    return {
        "left_timestamp_ns": left["timestamp_ns"],
        "right_timestamp_ns": right["timestamp_ns"],
        "bracket_seconds": span,
        "maximum_endpoint_delta_seconds": max(left_delta, right_delta),
        "weight": weight,
        "left": left,
        "right": right,
    }


def nearest_timestamp(samples: list[int], target_ns: int) -> tuple[int, float]:
    index = bisect.bisect_left(samples, target_ns)
    candidates = samples[max(0, index - 1) : min(len(samples), index + 1)]
    require(bool(candidates), "clock", "nearest_sample_absent")
    chosen = min(candidates, key=lambda value: abs(value - target_ns))
    return chosen, abs(chosen - target_ns) / 1e9


def nearest_record(samples: list[dict[str, Any]], target_ns: int) -> tuple[dict[str, Any], float]:
    times = [row["timestamp_ns"] for row in samples]
    chosen_ns, delta = nearest_timestamp(times, target_ns)
    return samples[bisect.bisect_left(times, chosen_ns)], delta


def read_bag_roles(bag_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise GateError("packet", f"rosbags_import:{exc}") from exc

    topics = config["topics"]
    selected_names = {
        norm(topics["stitched_rgb"]["topic"]),
        norm(topics["dynamic_pose"]["topic"]),
        norm(topics["static_transform"]["topic"]),
        norm(topics["imu"]["topic"]),
        norm(topics["upper_lidar"]["topic"]),
        norm(topics["lower_lidar"]["topic"]),
    }
    typestore = get_typestore(Stores.ROS1_NOETIC)
    pose: list[dict[str, Any]] = []
    imu: list[dict[str, Any]] = []
    rgb: list[dict[str, Any]] = []
    lidar: dict[str, list[dict[str, Any]]] = {"upper_lidar": [], "lower_lidar": []}
    static: dict[tuple[str, str], dict[str, Any]] = {}
    with Reader(bag_path) as reader:
        selected = [connection for connection in reader.connections if norm(connection.topic) in selected_names]
        for connection in selected:
            if connection.msgtype not in typestore.fielddefs:
                typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for connection, _bag_ns, raw in reader.messages(connections=selected):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            topic = norm(connection.topic)
            if topic == norm(topics["stitched_rgb"]["topic"]):
                require(norm(message.header.frame_id) == topics["stitched_rgb"]["frame"], "static", "stitched_rgb_frame")
                rgb.append(
                    {
                        "timestamp_ns": stamp_ns(message.header.stamp),
                        "frame": norm(message.header.frame_id),
                        "format": message.format,
                    }
                )
            elif topic == norm(topics["dynamic_pose"]["topic"]):
                for item in message.transforms:
                    if (
                        norm(item.header.frame_id) == topics["dynamic_pose"]["parent_frame"]
                        and norm(item.child_frame_id) == topics["dynamic_pose"]["child_frame"]
                    ):
                        value = item.transform
                        pose.append(
                            {
                                "timestamp_ns": stamp_ns(item.header.stamp),
                                "translation": [value.translation.x, value.translation.y, value.translation.z],
                                "quaternion_xyzw": [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
                            }
                        )
            elif topic == norm(topics["static_transform"]["topic"]):
                for item in message.transforms:
                    value = item.transform
                    static[(norm(item.header.frame_id), norm(item.child_frame_id))] = {
                        "parent": norm(item.header.frame_id),
                        "child": norm(item.child_frame_id),
                        "translation": [value.translation.x, value.translation.y, value.translation.z],
                        "quaternion_xyzw": [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
                    }
            elif topic == norm(topics["imu"]["topic"]):
                require(norm(message.header.frame_id) == topics["imu"]["frame"], "static", "imu_frame")
                imu.append(
                    {
                        "timestamp_ns": stamp_ns(message.header.stamp),
                        "orientation_xyzw": [
                            message.orientation.x,
                            message.orientation.y,
                            message.orientation.z,
                            message.orientation.w,
                        ],
                        "angular_velocity": [
                            message.angular_velocity.x,
                            message.angular_velocity.y,
                            message.angular_velocity.z,
                        ],
                        "linear_acceleration": [
                            message.linear_acceleration.x,
                            message.linear_acceleration.y,
                            message.linear_acceleration.z,
                        ],
                    }
                )
            else:
                for role in ("upper_lidar", "lower_lidar"):
                    if topic == norm(topics[role]["topic"]):
                        require(norm(message.header.frame_id) == topics[role]["frame"], "static", f"{role}_frame")
                        lidar[role].append(
                            {
                                "timestamp_ns": stamp_ns(message.header.stamp),
                                "frame": norm(message.header.frame_id),
                                "width": int(message.width),
                                "height": int(message.height),
                                "point_step": int(message.point_step),
                                "row_step": int(message.row_step),
                                "is_bigendian": bool(message.is_bigendian),
                                "fields": [
                                    {
                                        "name": field.name,
                                        "offset": int(field.offset),
                                        "datatype": int(field.datatype),
                                        "count": int(field.count),
                                    }
                                    for field in message.fields
                                ],
                            }
                        )
    pose.sort(key=lambda row: row["timestamp_ns"])
    imu.sort(key=lambda row: row["timestamp_ns"])
    rgb.sort(key=lambda row: row["timestamp_ns"])
    for values in lidar.values():
        values.sort(key=lambda row: row["timestamp_ns"])
    require(len(pose) >= 2, "interpolation", "pose_samples")
    require(len(imu) >= 2, "interpolation", "imu_samples")
    require(bool(rgb), "clock", "bag_stitched_rgb_samples")
    require(all(lidar.values()), "pointcloud", "bag_lidar_samples")
    return {"pose": pose, "imu": imu, "rgb": rgb, "lidar": lidar, "static": static}


def static_contract(repo: Path, config: dict[str, Any], static: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    required = [tuple(edge) for edge in config["topics"]["static_transform"]["required_edges"]]
    missing = [edge for edge in required if edge not in static]
    require(not missing, "static", f"missing_edges:{missing}")

    def edge(parent: str, child: str) -> dict[str, Any]:
        row = static[(parent, child)]
        return transform_from_q(row["translation"], row["quaternion_xyzw"])

    defaults = repo / config["local_inputs"]["calibration_defaults"]["path"]
    upper_to_rgb = parse_default_calibration(defaults, "lidar_upper_to_rgb")
    lower_to_rgb = parse_default_calibration(defaults, "lidar_lower_to_rgb")
    base_chassis_from_upper = edge("base_chassis_link", "upper_velodyne_frame")
    base_chassis_from_lower = edge("base_chassis_link", "lower_velodyne_frame")
    base_chassis_from_rgb_upper = compose(base_chassis_from_upper, inverse(upper_to_rgb))
    base_chassis_from_rgb_lower = compose(base_chassis_from_lower, inverse(lower_to_rgb))
    translation_delta = max(
        abs(a - b)
        for a, b in zip(
            base_chassis_from_rgb_upper["translation"],
            base_chassis_from_rgb_lower["translation"],
        )
    )
    rotation_delta = matrix_delta(
        base_chassis_from_rgb_upper["rotation"],
        base_chassis_from_rgb_lower["rotation"],
    )
    gates = config["gates"]
    require(
        translation_delta <= float(gates["static_translation_tolerance_meters"]),
        "static",
        f"upper_lower_translation:{translation_delta}",
    )
    require(
        rotation_delta <= float(gates["static_rotation_tolerance"]),
        "static",
        f"upper_lower_rotation:{rotation_delta}",
    )
    occam_translation = static[("base_chassis_link", "occam")]["translation"]
    occam_delta = max(
        abs(a - b) for a, b in zip(base_chassis_from_rgb_upper["translation"], occam_translation)
    )
    require(
        occam_delta <= float(gates["static_translation_tolerance_meters"]),
        "static",
        f"logical_rgb_origin_vs_occam:{occam_delta}",
    )
    base_link_from_rgb = compose(
        edge("base_link", "base_chassis_link"),
        base_chassis_from_rgb_upper,
    )
    return {
        "required_edges": [static[edge_key] for edge_key in required],
        "logical_rgb360_semantics": "official_toolkit_R_times_point_minus_translation",
        "upper_to_logical_rgb360": upper_to_rgb,
        "lower_to_logical_rgb360": lower_to_rgb,
        "base_chassis_from_logical_rgb360": base_chassis_from_rgb_upper,
        "base_link_from_logical_rgb360": base_link_from_rgb,
        "upper_lower_translation_delta_meters": translation_delta,
        "upper_lower_rotation_delta": rotation_delta,
        "logical_rgb_origin_vs_occam_translation_delta_meters": occam_delta,
    }


def input_hash_contract(repo: Path, config: dict[str, Any]) -> None:
    require(config["schema"] == CONFIG_SCHEMA and config["stage"] == STAGE, "packet", "config_identity")
    for role in ("result", "receipt", "validation"):
        binding = config["parent"][role]
        path = repo / binding["path"]
        require(path.is_file() and sha256_file(path) == binding["sha256"], "packet", f"parent_{role}_drift")
    parent_receipt = json.loads((repo / config["parent"]["receipt"]["path"]).read_text(encoding="utf-8"))
    parent_validation = json.loads((repo / config["parent"]["validation"]["path"]).read_text(encoding="utf-8"))
    require(
        parent_receipt["terminal_state"] == config["parent"]["receipt"]["required_terminal"],
        "packet",
        "parent_terminal",
    )
    require(
        parent_validation["status"] == config["parent"]["validation"]["required_status"],
        "packet",
        "parent_validation",
    )
    for role, binding in config["local_inputs"].items():
        path = repo / binding["path"]
        require(path.is_file() and sha256_file(path) == binding["sha256"], "packet", f"local_{role}_drift")
        if "bytes" in binding:
            require(path.stat().st_size == int(binding["bytes"]), "packet", f"local_{role}_size")


def file_index(files: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in files:
        require(row["member"] not in output, "packet", f"duplicate_member:{row['member']}")
        output[row["member"]] = row
    return output


def materialization_binding(materialization: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": materialization["schema"],
        "stage": materialization["stage"],
        "status": materialization["status"],
        "config_sha256": materialization["config_sha256"],
        "transport": materialization["transport"],
        "files": materialization["files"],
        "authority": materialization["authority"],
    }


def label_frame(payload: dict[str, Any], frame: str, suffix: str) -> list[dict[str, Any]]:
    key = f"{frame}.{suffix}"
    require(key in payload["labels"], "label", f"frame_missing:{key}")
    values = payload["labels"][key]
    require(isinstance(values, list), "label", f"frame_type:{key}")
    return values


def build_packet(repo: Path, config_path: Path, materialization: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_hash_contract(repo, config)
    require(materialization["status"] == "MATERIALIZED", "packet", "materialization_not_complete")
    files = materialization["files"]
    index = file_index(files)
    for row in files:
        path = repo / row["path"]
        require(path.is_file(), "packet", f"payload_missing:{row['member']}")
        require(path.stat().st_size == row["bytes"], "packet", f"payload_size:{row['member']}")
        require(sha256_file(path) == row["sha256"], "packet", f"payload_sha:{row['member']}")

    sequence = config["canary"]["sequence"]
    first = int(config["canary"]["window_first_frame"])
    last = int(config["canary"]["window_last_frame"])
    frame_names = [f"{idx:06d}" for idx in range(first, last + 1)]
    timestamp_path = repo / config["local_inputs"]["timestamps"]["path"]
    with zipfile.ZipFile(timestamp_path) as bundle:
        image_doc = json.loads(bundle.read(f"timestamps/{sequence}/frames_img.json"))
        point_doc = json.loads(bundle.read(f"timestamps/{sequence}/frames_pc.json"))
    image_rows = {
        frame_stem(next(camera for camera in row["cameras"] if camera["name"] == "stitched_image0")["url"]): row
        for row in image_doc["data"]
    }
    point_rows = {
        frame_stem(next(point for point in row["pointclouds"] if point["name"] == "upper_velodyne")["url"]): row
        for row in point_doc["data"]
    }
    require(all(frame in image_rows and frame in point_rows for frame in frame_names), "clock", "timestamp_frame_missing")

    labels_2d_name = f"labels/labels_2d_stitched/{sequence}.json"
    labels_3d_name = f"labels/labels_3d/{sequence}.json"
    labels_2d = json.loads((repo / index[labels_2d_name]["path"]).read_text(encoding="utf-8"))
    labels_3d = json.loads((repo / index[labels_3d_name]["path"]).read_text(encoding="utf-8"))
    require("labels" in labels_2d and "labels" in labels_3d, "label", "labels_root")

    bag_roles = read_bag_roles(repo / config["local_inputs"]["bag"]["path"], config)
    static_info = static_contract(repo, config, bag_roles["static"])
    gates = config["gates"]
    base_from_rgb = static_info["base_link_from_logical_rgb360"]
    packet_frames: list[dict[str, Any]] = []
    prior_times: tuple[int, int, int] | None = None
    prior_bag_matches: dict[str, int | None] = {"rgb": None, "upper": None, "lower": None}
    for frame_index, frame in enumerate(frame_names):
        image_member = f"images/image_stitched/{sequence}/{frame}.jpg"
        upper_member = f"pointclouds/upper_velodyne/{sequence}/{frame}.pcd"
        lower_member = f"pointclouds/lower_velodyne/{sequence}/{frame}.pcd"
        require(all(name in index for name in (image_member, upper_member, lower_member)), "pointcloud", f"raw_frame:{frame}")
        image_path = repo / index[image_member]["path"]
        upper_path = repo / index[upper_member]["path"]
        lower_path = repo / index[lower_member]["path"]
        width, height = jpeg_size(image_path)
        require((width, height) == (3760, 480), "packet", f"rgb_geometry:{frame}:{width}x{height}")
        upper_header = pcd_header(upper_path)
        lower_header = pcd_header(lower_path)

        image_row = image_rows[frame]
        point_row = point_rows[frame]
        image_seconds = float(
            next(camera for camera in image_row["cameras"] if camera["name"] == "stitched_image0")["timestamp"]
        )
        upper_seconds = float(
            next(point for point in point_row["pointclouds"] if point["name"] == "upper_velodyne")["timestamp"]
        )
        lower_seconds = float(
            next(point for point in point_row["pointclouds"] if point["name"] == "lower_velodyne")["timestamp"]
        )
        image_ns, upper_ns, lower_ns = (round(value * 1e9) for value in (image_seconds, upper_seconds, lower_seconds))
        if prior_times:
            require(
                image_ns > prior_times[0] and upper_ns > prior_times[1] and lower_ns > prior_times[2],
                "clock",
                f"timestamp_nonmonotonic:{frame}",
            )
        prior_times = (image_ns, upper_ns, lower_ns)
        require(
            max(abs(image_seconds - upper_seconds), abs(image_seconds - lower_seconds))
            <= float(gates["maximum_image_pointcloud_delta_seconds"]),
            "clock",
            f"image_pointcloud_delta:{frame}",
        )
        bag_rgb, bag_rgb_delta = nearest_record(bag_roles["rgb"], image_ns)
        bag_upper, bag_upper_delta = nearest_record(bag_roles["lidar"]["upper_lidar"], upper_ns)
        bag_lower, bag_lower_delta = nearest_record(bag_roles["lidar"]["lower_lidar"], lower_ns)
        bag_rgb_ns = bag_rgb["timestamp_ns"]
        bag_upper_ns = bag_upper["timestamp_ns"]
        bag_lower_ns = bag_lower["timestamp_ns"]
        require(
            bag_rgb_delta <= float(gates["maximum_external_to_bag_rgb_delta_seconds"]),
            "clock",
            f"external_bag_rgb_delta:{frame}",
        )
        require(
            max(bag_upper_delta, bag_lower_delta)
            <= float(gates["maximum_external_to_bag_lidar_delta_seconds"]),
            "clock",
            f"external_bag_lidar_delta:{frame}",
        )
        for role, value in (("rgb", bag_rgb_ns), ("upper", bag_upper_ns), ("lower", bag_lower_ns)):
            prior = prior_bag_matches[role]
            require(prior is None or value > prior, "clock", f"bag_match_reuse_or_nonmonotonic:{role}:{frame}")
            prior_bag_matches[role] = value
        for role, pcd, bag_row in (
            ("upper", upper_header, bag_upper),
            ("lower", lower_header, bag_lower),
        ):
            bag_fields = [field["name"] for field in bag_row["fields"] if field["name"]]
            require(pcd["fields"] == bag_fields, "pointcloud", f"{role}_fields:{frame}")
            require(
                pcd["points"] == bag_row["width"] * bag_row["height"],
                "pointcloud",
                f"{role}_point_count:{frame}",
            )
            require(not bag_row["is_bigendian"], "pointcloud", f"{role}_bigendian:{frame}")
        pose_interp = interpolate(
            bag_roles["pose"],
            upper_ns,
            float(gates["maximum_pose_bracket_seconds"]),
            float(gates["maximum_pose_endpoint_delta_seconds"]),
            "interpolation",
        )
        imu_interp = interpolate(
            bag_roles["imu"],
            upper_ns,
            float(gates["maximum_imu_bracket_seconds"]),
            float(gates["maximum_imu_endpoint_delta_seconds"]),
            "interpolation",
        )
        pose_weight = pose_interp["weight"]
        pose_translation = [
            (1 - pose_weight) * a + pose_weight * b
            for a, b in zip(pose_interp["left"]["translation"], pose_interp["right"]["translation"])
        ]
        pose_quaternion = q_slerp(
            pose_interp["left"]["quaternion_xyzw"],
            pose_interp["right"]["quaternion_xyzw"],
            pose_weight,
        )
        odom_from_base = transform_from_q(pose_translation, pose_quaternion)
        imu_weight = imu_interp["weight"]
        imu_value = {
            key: [
                (1 - imu_weight) * a + imu_weight * b
                for a, b in zip(imu_interp["left"][key], imu_interp["right"][key])
            ]
            for key in ("angular_velocity", "linear_acceleration")
        }
        imu_value["orientation_xyzw"] = q_slerp(
            imu_interp["left"]["orientation_xyzw"],
            imu_interp["right"]["orientation_xyzw"],
            imu_weight,
        )

        frame_2d = label_frame(labels_2d, frame, "jpg")
        frame_3d = label_frame(labels_3d, frame, "pcd")
        ids_2d: dict[str, dict[str, Any]] = {}
        ids_3d: dict[str, dict[str, Any]] = {}
        for item in frame_2d:
            label_id = item.get("label_id")
            require(isinstance(label_id, str) and label_id not in ids_2d, "label", f"2d_duplicate:{frame}:{label_id}")
            ids_2d[label_id] = item
        for item in frame_3d:
            label_id = item.get("label_id")
            require(isinstance(label_id, str) and label_id not in ids_3d, "label", f"3d_duplicate:{frame}:{label_id}")
            ids_3d[label_id] = item
        missing_2d = sorted(set(ids_3d) - set(ids_2d))
        joined: list[dict[str, Any]] = []
        for label_id in sorted(set(ids_3d) & set(ids_2d)):
            item_3d = ids_3d[label_id]
            box = item_3d.get("box", {})
            center_rgb = [float(box[key]) for key in ("cx", "cy", "cz")]
            require(all(math.isfinite(value) for value in center_rgb), "label", f"nonfinite_center:{frame}:{label_id}")
            center_base = apply_transform(base_from_rgb, center_rgb)
            center_odom = apply_transform(odom_from_base, center_base)
            box_2d = [float(value) for value in ids_2d[label_id].get("box", [])]
            require(len(box_2d) == 4 and all(math.isfinite(value) for value in box_2d), "label", f"2d_box:{frame}:{label_id}")
            joined.append(
                {
                    "label_id": label_id,
                    "box_2d_xywh": box_2d,
                    "box_3d": {key: float(box[key]) for key in ("cx", "cy", "cz", "w", "l", "h", "rot_z")},
                    "center_logical_rgb360_m": center_rgb,
                    "center_base_link_m": center_base,
                    "center_odom_m": center_odom,
                    "occlusion_2d": ids_2d[label_id].get("attributes", {}).get("occlusion"),
                    "label_3d_interpolated": item_3d.get("attributes", {}).get("interpolated"),
                }
            )
        packet_frames.append(
            {
                "frame_index": frame_index + first,
                "frame_stem": frame,
                "source": {
                    "image": {**index[image_member], "width": width, "height": height},
                    "upper_pointcloud": {**index[upper_member], **upper_header},
                    "lower_pointcloud": {**index[lower_member], **lower_header},
                },
                "time": {
                    "clock_domain": "unix_epoch_seconds_from_jrdb_external_and_ros_header",
                    "image_timestamp_ns": image_ns,
                    "upper_pointcloud_timestamp_ns": upper_ns,
                    "lower_pointcloud_timestamp_ns": lower_ns,
                    "bag_stitched_rgb_header_timestamp_ns": bag_rgb_ns,
                    "bag_upper_lidar_header_timestamp_ns": bag_upper_ns,
                    "bag_lower_lidar_header_timestamp_ns": bag_lower_ns,
                    "bag_rgb_delta_seconds": bag_rgb_delta,
                    "bag_upper_delta_seconds": bag_upper_delta,
                    "bag_lower_delta_seconds": bag_lower_delta,
                },
                "bag_sensor_metadata": {
                    "stitched_rgb": bag_rgb,
                    "upper_lidar": bag_upper,
                    "lower_lidar": bag_lower,
                },
                "pose": {
                    "parent_frame": "odom",
                    "child_frame": "base_link",
                    "target_timestamp_ns": upper_ns,
                    "left_timestamp_ns": pose_interp["left_timestamp_ns"],
                    "right_timestamp_ns": pose_interp["right_timestamp_ns"],
                    "bracket_seconds": pose_interp["bracket_seconds"],
                    "maximum_endpoint_delta_seconds": pose_interp["maximum_endpoint_delta_seconds"],
                    "weight": pose_weight,
                    "translation": pose_translation,
                    "quaternion_xyzw": pose_quaternion,
                },
                "imu": {
                    "frame": config["topics"]["imu"]["frame"],
                    "target_timestamp_ns": upper_ns,
                    "left_timestamp_ns": imu_interp["left_timestamp_ns"],
                    "right_timestamp_ns": imu_interp["right_timestamp_ns"],
                    "bracket_seconds": imu_interp["bracket_seconds"],
                    "maximum_endpoint_delta_seconds": imu_interp["maximum_endpoint_delta_seconds"],
                    "weight": imu_weight,
                    **imu_value,
                },
                "labels": {
                    "join_key": "sequence + frame_stem + label_id",
                    "labels_2d_count": len(ids_2d),
                    "labels_3d_count": len(ids_3d),
                    "joined_count": len(joined),
                    "labels_2d_only": sorted(set(ids_2d) - set(ids_3d)),
                    "labels_3d_without_2d": missing_2d,
                    "joined": joined,
                },
            }
        )
    return {
        "schema": PACKET_SCHEMA,
        "stage": STAGE,
        "status": "IMMUTABLE_OBSERVATION_PACKET",
        "config_sha256": sha256_file(config_path),
        "parent_receipt_sha256": config["parent"]["receipt"]["sha256"],
        "materialization_binding_sha256": sha256_bytes(canonical_bytes(materialization_binding(materialization))),
        "sequence": sequence,
        "window": {"first_frame": first, "last_frame": last, "frame_count": len(packet_frames)},
        "raw_payload": {
            "member_count": len(files),
            "payload_bytes": sum(row["bytes"] for row in files),
            "files": files,
        },
        "calibration": {
            "defaults_sha256": config["local_inputs"]["calibration_defaults"]["sha256"],
            "lidars_sha256": config["local_inputs"]["calibration_lidars"]["sha256"],
            **static_info,
        },
        "frames": packet_frames,
        "authority": config["authority"],
    }


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "p95": None, "maximum": None}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {"minimum": ordered[0], "median": pick(0.5), "p95": pick(0.95), "maximum": ordered[-1]}


def audit_packet(config: dict[str, Any], packet: dict[str, Any], packet_sha256: str) -> dict[str, Any]:
    require(packet["schema"] == PACKET_SCHEMA and packet["stage"] == STAGE, "packet", "packet_identity")
    frame_count = int(config["canary"]["frame_count"])
    require(len(packet["frames"]) == frame_count, "packet", f"frame_count:{len(packet['frames'])}")
    joined_frames = sum(1 for frame in packet["frames"] if frame["labels"]["joined_count"] > 0)
    joined_observations = sum(frame["labels"]["joined_count"] for frame in packet["frames"])
    two_d_only = sum(len(frame["labels"]["labels_2d_only"]) for frame in packet["frames"])
    three_d_without_two_d = sum(
        len(frame["labels"]["labels_3d_without_2d"]) for frame in packet["frames"]
    )
    interpolated_joined = sum(
        1
        for frame in packet["frames"]
        for item in frame["labels"]["joined"]
        if item["label_3d_interpolated"] is True
    )
    label_join_ready = three_d_without_two_d == 0
    if config["gates"]["require_all_3d_labels_join_2d"] and not label_join_ready:
        return {
            "schema": RECEIPT_SCHEMA,
            "stage": STAGE,
            "status": "FAIL_CLOSED",
            "terminal_state": "FAIL_CLOSED_LABEL_JOIN",
            "config_sha256": packet["config_sha256"],
            "observation_packet_sha256": packet_sha256,
            "sequence": packet["sequence"],
            "window": packet["window"],
            "availability": {
                "joined_person_frames": joined_frames,
                "joined_person_observations": joined_observations,
                "labels_2d_only_observations": two_d_only,
                "labels_3d_without_2d_observations": three_d_without_two_d,
                "joined_3d_labels_with_source_interpolated_true": interpolated_joined,
                "valid_adjacent_motion_pairs": 0,
                "motion_track_count": 0,
            },
            "gates": {
                "packet_frames_complete": len(packet["frames"]) == frame_count,
                "all_3d_labels_joined_2d": False,
                "motion_computation_permitted": False,
            },
            "motion_pairs": [],
            "motion_computation_skipped_reason": "label_join_gate_failed_before_motion",
            "authority": config["authority"],
            "claims": {
                "source_native_person_motion_available": False,
                "robot_relative_geometry_available": False,
                "route_risk_computed": False,
                "event_lifecycle_computed": False,
                "alert_logic_computed": False,
                "human_or_production_authority": False,
            },
        }
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    robot_ranges: list[float] = []
    for frame in packet["frames"]:
        for item in frame["labels"]["joined"]:
            tracks[item["label_id"]].append(
                {
                    "frame_index": frame["frame_index"],
                    "timestamp_ns": frame["time"]["upper_pointcloud_timestamp_ns"],
                    "base": item["center_base_link_m"],
                    "odom": item["center_odom_m"],
                }
            )
            robot_ranges.append(math.sqrt(sum(value * value for value in item["center_base_link_m"])))
    motion_pairs: list[dict[str, Any]] = []
    motion_tracks: set[str] = set()
    max_gap = float(config["gates"]["maximum_person_motion_gap_seconds"])
    for label_id, observations in sorted(tracks.items()):
        observations.sort(key=lambda row: row["frame_index"])
        for left, right in zip(observations, observations[1:]):
            gap = (right["timestamp_ns"] - left["timestamp_ns"]) / 1e9
            if right["frame_index"] != left["frame_index"] + 1 or gap <= 0 or gap > max_gap:
                continue
            odom_velocity = [(b - a) / gap for a, b in zip(left["odom"], right["odom"])]
            relative_velocity = [(b - a) / gap for a, b in zip(left["base"], right["base"])]
            require(
                all(math.isfinite(value) for value in odom_velocity + relative_velocity),
                "packet",
                f"motion_nonfinite:{label_id}",
            )
            motion_tracks.add(label_id)
            motion_pairs.append(
                {
                    "label_id": label_id,
                    "left_frame": left["frame_index"],
                    "right_frame": right["frame_index"],
                    "gap_seconds": gap,
                    "source_native_odom_velocity_mps": odom_velocity,
                    "source_native_odom_speed_mps": math.sqrt(sum(value * value for value in odom_velocity)),
                    "robot_relative_velocity_mps": relative_velocity,
                    "robot_relative_speed_mps": math.sqrt(sum(value * value for value in relative_velocity)),
                }
            )
    availability = {
        "joined_person_frames": joined_frames,
        "joined_person_observations": joined_observations,
        "labels_2d_only_observations": two_d_only,
        "labels_3d_without_2d_observations": three_d_without_two_d,
        "joined_3d_labels_with_source_interpolated_true": interpolated_joined,
        "unique_joined_tracks": len(tracks),
        "valid_adjacent_motion_pairs": len(motion_pairs),
        "motion_track_count": len(motion_tracks),
    }
    gate_results = {
        "packet_frames_complete": len(packet["frames"]) == frame_count,
        "all_3d_labels_joined_2d": all(
            not frame["labels"]["labels_3d_without_2d"] for frame in packet["frames"]
        ),
        "minimum_joined_person_frames": joined_frames >= int(config["gates"]["minimum_joined_person_frames"]),
        "minimum_valid_motion_pairs": len(motion_pairs) >= int(config["gates"]["minimum_valid_motion_pairs"]),
        "minimum_motion_track_count": len(motion_tracks) >= int(config["gates"]["minimum_motion_track_count"]),
    }
    terminal = (
        "SOURCE_NATIVE_PERSON_GEOMETRY_AVAILABLE"
        if all(gate_results.values())
        else "PERSON_GEOMETRY_AVAILABILITY_INSUFFICIENT"
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "stage": STAGE,
        "status": "AUDIT_COMPLETE",
        "terminal_state": terminal,
        "config_sha256": packet["config_sha256"],
        "observation_packet_sha256": packet_sha256,
        "sequence": packet["sequence"],
        "window": packet["window"],
        "availability": availability,
        "gates": gate_results,
        "quality": {
            "image_pointcloud_delta_seconds": quantiles(
                [
                    max(
                        abs(frame["time"]["image_timestamp_ns"] - frame["time"]["upper_pointcloud_timestamp_ns"]),
                        abs(frame["time"]["image_timestamp_ns"] - frame["time"]["lower_pointcloud_timestamp_ns"]),
                    )
                    / 1e9
                    for frame in packet["frames"]
                ]
            ),
            "bag_lidar_delta_seconds": quantiles(
                [
                    max(frame["time"]["bag_upper_delta_seconds"], frame["time"]["bag_lower_delta_seconds"])
                    for frame in packet["frames"]
                ]
            ),
            "bag_rgb_delta_seconds": quantiles(
                [frame["time"]["bag_rgb_delta_seconds"] for frame in packet["frames"]]
            ),
            "pose_bracket_seconds": quantiles([frame["pose"]["bracket_seconds"] for frame in packet["frames"]]),
            "imu_bracket_seconds": quantiles([frame["imu"]["bracket_seconds"] for frame in packet["frames"]]),
            "robot_relative_range_meters": quantiles(robot_ranges),
            "source_native_odom_speed_mps": quantiles(
                [pair["source_native_odom_speed_mps"] for pair in motion_pairs]
            ),
            "robot_relative_speed_mps": quantiles(
                [pair["robot_relative_speed_mps"] for pair in motion_pairs]
            ),
        },
        "motion_pairs": motion_pairs,
        "authority": config["authority"],
        "claims": {
            "source_native_person_motion_available": terminal == "SOURCE_NATIVE_PERSON_GEOMETRY_AVAILABLE",
            "robot_relative_geometry_available": terminal == "SOURCE_NATIVE_PERSON_GEOMETRY_AVAILABLE",
            "route_risk_computed": False,
            "event_lifecycle_computed": False,
            "alert_logic_computed": False,
            "human_or_production_authority": False,
        },
    }


def terminal_for_gate(gate: str) -> str:
    return {
        "clock": "FAIL_CLOSED_CLOCK_BINDING",
        "static": "FAIL_CLOSED_STATIC_FRAME_CHAIN",
        "pointcloud": "FAIL_CLOSED_POINTCLOUD_FRAME_MISSING",
        "label": "FAIL_CLOSED_LABEL_JOIN",
        "interpolation": "FAIL_CLOSED_INTERPOLATION_BOUND",
    }.get(gate, "FAIL_CLOSED_ACQUISITION_OR_PACKET_INCOMPLETE")


def materialize(repo: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_hash_contract(repo, config)
    try:
        files, transport = materialize_payload(repo, config)
        provisional = {
            "schema": MATERIALIZATION_SCHEMA,
            "stage": STAGE,
            "status": "MATERIALIZED",
            "terminal_state": None,
            "config_sha256": sha256_file(config_path),
            "transport": transport,
            "files": files,
            "authority": config["authority"],
        }
        packet = build_packet(repo, config_path, provisional)
        packet_path = repo / config["outputs"]["observation_packet"]
        write_canonical(packet_path, packet)
        provisional["observation_packet"] = {
            "path": packet_path.relative_to(repo).as_posix(),
            "bytes": packet_path.stat().st_size,
            "sha256": sha256_file(packet_path),
        }
        return provisional
    except GateError as error:
        return {
            "schema": MATERIALIZATION_SCHEMA,
            "stage": STAGE,
            "status": "FAIL_CLOSED",
            "terminal_state": terminal_for_gate(error.gate),
            "config_sha256": sha256_file(config_path),
            "failure": {"gate": error.gate, "detail": error.detail},
            "authority": config["authority"],
        }


def audit(repo: Path, config_path: Path, materialization_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    require(materialization["schema"] == MATERIALIZATION_SCHEMA, "packet", "materialization_identity")
    if materialization["status"] != "MATERIALIZED":
        return {
            "schema": RECEIPT_SCHEMA,
            "stage": STAGE,
            "status": "FAIL_CLOSED",
            "terminal_state": materialization["terminal_state"],
            "config_sha256": sha256_file(config_path),
            "materialization_sha256": sha256_file(materialization_path),
            "failure": materialization["failure"],
            "authority": config["authority"],
        }
    packet_path = repo / config["outputs"]["observation_packet"]
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    receipt = audit_packet(config, packet, sha256_file(packet_path))
    receipt["materialization_sha256"] = sha256_file(materialization_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("materialize", "audit"), required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.phase == "materialize":
        result = materialize(repo, config_path)
        output = repo / config["outputs"]["materialization"]
    else:
        materialization_path = repo / config["outputs"]["materialization"]
        result = audit(repo, config_path, materialization_path)
        output = repo / config["outputs"]["receipt"]
    write_canonical(output, result)
    print(json.dumps({"output": output.as_posix(), "status": result["status"], "terminal_state": result.get("terminal_state")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
