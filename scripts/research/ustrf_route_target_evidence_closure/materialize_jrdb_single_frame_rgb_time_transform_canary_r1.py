#!/usr/bin/env python3
"""Materialize one JRDB stitched JPEG via bounded ZIP64 byte ranges."""
from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import struct
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "blindassist_ustrf_jrdb_single_frame_rgb_time_transform_canary_r1"
CONFIG_SCHEMA = f"{SCHEMA}_config"
STAGE = "JRDB_SINGLE_FRAME_RGB_TIME_TRANSFORM_CANARY_R1"
TERMINALS = (
    "FAIL_CLOSED_AUDIT_INCOMPLETE",
    "RANGE_EXTRACTION_RESOURCE_BLOCKED",
    "SAME_FRAME_IDENTITY_OR_TIME_INSUFFICIENT",
    "RGB_TIME_TRANSFORM_CANARY_PRESENT",
)
IMPLEMENTATIONS = {
    "producer": "scripts/research/ustrf_route_target_evidence_closure/materialize_jrdb_single_frame_rgb_time_transform_canary_r1.py",
    "validator": "scripts/research/ustrf_route_target_evidence_closure/validate_jrdb_single_frame_rgb_time_transform_canary_r1.py",
}


class CanaryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_root_not_object:{path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def load_config(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    require(config["schema"] == CONFIG_SCHEMA, "config_schema_drift")
    require(config["stage"] == STAGE, "stage_drift")
    require(config["status"] == "frozen_before_execution", "config_not_frozen")
    require(tuple(config["terminal_states"]) == TERMINALS, "terminal_order_drift")
    for label, binding in config["bindings"].items():
        path = repo / binding["path"]
        require(path.is_file(), f"{label}_missing")
        require(sha256_file(path) == binding["sha256"], f"{label}_sha256_drift")
    digests = config["research_implementation_digests"]
    require(set(digests) == set(IMPLEMENTATIONS), "implementation_digest_keys_drift")
    for label, relative_path in IMPLEMENTATIONS.items():
        require(sha256_file(repo / relative_path) == digests[label], f"{label}_implementation_drift")
    return config


class RangeClient:
    def __init__(self, url: str, budget: int) -> None:
        self.url = url
        self.budget = budget
        self.bytes_read = 0
        self.requests: list[dict[str, Any]] = []

    def get(self, start: int, end: int) -> bytes:
        require(0 <= start <= end, "invalid_range")
        requested = end - start + 1
        require(self.bytes_read + requested <= self.budget, "range_budget_exceeded")
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            require(response.status == 206, f"range_status_not_206:{response.status}")
            expected_content_range = f"bytes {start}-{end}/"
            content_range = response.headers.get("Content-Range", "")
            require(content_range.startswith(expected_content_range), f"content_range_drift:{content_range}")
            payload = response.read(requested + 1)
            require(len(payload) == requested, f"range_length_drift:{len(payload)}:{requested}")
        self.bytes_read += len(payload)
        self.requests.append({"start": start, "end": end, "bytes": len(payload)})
        return payload


def zip64_values(extra: bytes, compressed: int, uncompressed: int, offset: int) -> tuple[int, int, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        field = extra[cursor + 4 : cursor + 4 + field_size]
        cursor += 4 + field_size
        if field_id != 0x0001:
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


def find_member(central: bytes, suffix: str) -> dict[str, Any]:
    cursor = 0
    matches: list[dict[str, Any]] = []
    while cursor < len(central):
        require(central[cursor : cursor + 4] == b"PK\x01\x02", f"central_signature_drift:{cursor}")
        values = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        (
            _signature,
            _made,
            _needed,
            flags,
            compression,
            _mtime,
            _mdate,
            crc32,
            compressed,
            uncompressed,
            name_len,
            extra_len,
            comment_len,
            _disk,
            _internal,
            _external,
            local_offset,
        ) = values
        name_start = cursor + 46
        name = central[name_start : name_start + name_len].decode("utf-8")
        extra = central[name_start + name_len : name_start + name_len + extra_len]
        compressed, uncompressed, local_offset = zip64_values(
            extra, compressed, uncompressed, local_offset
        )
        require(safe_member(name), f"unsafe_member:{name}")
        if name.endswith(suffix):
            matches.append(
                {
                    "name": name,
                    "flags": flags,
                    "compression": compression,
                    "crc32": crc32,
                    "compressed_size": compressed,
                    "uncompressed_size": uncompressed,
                    "local_offset": local_offset,
                }
            )
        cursor = name_start + name_len + extra_len + comment_len
    require(cursor == len(central), "central_directory_length_drift")
    require(len(matches) == 1, f"target_member_count:{len(matches)}")
    return matches[0]


def jpeg_geometry(payload: bytes) -> dict[str, int]:
    require(payload[:2] == b"\xff\xd8", "jpeg_soi_missing")
    cursor = 2
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while cursor + 4 <= len(payload):
        require(payload[cursor] == 0xFF, f"jpeg_marker_drift:{cursor}")
        while cursor < len(payload) and payload[cursor] == 0xFF:
            cursor += 1
        marker = payload[cursor]
        cursor += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        length = struct.unpack_from(">H", payload, cursor)[0]
        require(length >= 2 and cursor + length <= len(payload), "jpeg_segment_length_drift")
        if marker in sof:
            precision, height, width, components = struct.unpack_from(">BHHB", payload, cursor + 2)
            require(width > 0 and height > 0 and components in {1, 3, 4}, "jpeg_geometry_invalid")
            return {
                "width": width,
                "height": height,
                "components": components,
                "precision_bits": precision,
            }
        cursor += length
    raise CanaryError("jpeg_sof_missing")


def timestamp_binding(archive: Path, sequence: str, frame: str, camera: str) -> dict[str, Any]:
    member = f"timestamps/{sequence}/frames_img.json"
    with zipfile.ZipFile(archive) as bundle:
        require(member in bundle.namelist(), "timestamp_member_missing")
        payload = json.loads(bundle.read(member))
    suffix = f"/{sequence}/{camera}/{frame}"
    matches = []
    for row in payload["data"]:
        for item in row.get("cameras", []):
            if item.get("name") == camera and str(item.get("url", "")).endswith(suffix):
                matches.append(
                    {
                        "capture_timestamp": item.get("timestamp"),
                        "row_timestamp": row.get("timestamp"),
                        "source_frame_id": row.get("frame_id"),
                        "source_url": item.get("url"),
                    }
                )
    require(len(matches) == 1, f"timestamp_binding_count:{len(matches)}")
    require(isinstance(matches[0]["capture_timestamp"], (int, float)), "capture_timestamp_not_numeric")
    return matches[0]


def audit(repo: Path, config_path: Path, persist_image: bool) -> dict[str, Any]:
    config = load_config(repo, config_path)
    sequence = config["canary"]["sequence"]
    frame = config["canary"]["frame"]
    camera = config["canary"]["timestamp_camera"]
    suffix = f"images/image_stitched/{sequence}/{frame}"

    labels_path = repo / config["bindings"]["test_labels"]["path"]
    with zipfile.ZipFile(labels_path) as labels:
        label_member = f"labels_2d_stitched/{sequence}.json"
        require(label_member in labels.namelist(), "label_sequence_missing")
        label_payload = json.loads(labels.read(label_member))
    require(frame in label_payload["labels"], "label_frame_missing")

    timestamp = timestamp_binding(
        repo / config["bindings"]["test_timestamps"]["path"], sequence, frame, camera
    )
    calibration_path = repo / config["bindings"]["test_calibration"]["path"]
    with zipfile.ZipFile(calibration_path) as calibration:
        calibration_members = set(calibration.namelist())
    require(
        {"calibration/defaults.yaml", "calibration/cameras.yaml"} <= calibration_members,
        "calibration_members_missing",
    )

    remote = config["remote_archive"]
    client = RangeClient(remote["url"], int(config["resource_gate"]["maximum_network_bytes"]))
    cd_start = int(remote["central_directory_offset"])
    cd_size = int(remote["central_directory_size"])
    central = client.get(cd_start, cd_start + cd_size - 1)
    member = find_member(central, suffix)
    require(member["flags"] & 0x1 == 0, "encrypted_member")
    require(member["compressed_size"] <= int(config["resource_gate"]["maximum_member_bytes"]), "member_over_budget")

    local = client.get(member["local_offset"], member["local_offset"] + 29)
    values = struct.unpack("<4s5H3L2H", local)
    require(values[0] == b"PK\x03\x04", "local_header_signature_drift")
    name_len, extra_len = values[-2], values[-1]
    local_tail = client.get(
        member["local_offset"] + 30,
        member["local_offset"] + 30 + name_len + extra_len - 1,
    )
    local_name = local_tail[:name_len].decode("utf-8")
    require(local_name == member["name"], "local_central_name_mismatch")
    data_start = member["local_offset"] + 30 + name_len + extra_len
    compressed = client.get(data_start, data_start + member["compressed_size"] - 1)
    if member["compression"] == 0:
        jpeg = compressed
    elif member["compression"] == 8:
        jpeg = zlib.decompress(compressed, -15)
    else:
        raise CanaryError(f"unsupported_compression:{member['compression']}")
    require(len(jpeg) == member["uncompressed_size"], "uncompressed_size_drift")
    require(binascii.crc32(jpeg) & 0xFFFFFFFF == member["crc32"], "jpeg_crc_drift")
    geometry = jpeg_geometry(jpeg)

    image_output = repo / config["outputs"]["image"]
    if persist_image:
        atomic_write(image_output, jpeg)
    image_sha = hashlib.sha256(jpeg).hexdigest()
    return {
        "schema": SCHEMA,
        "stage": STAGE,
        "status": "AUDIT_COMPLETE",
        "terminal_state": "RGB_TIME_TRANSFORM_CANARY_PRESENT",
        "process_id": os.getpid(),
        "config_sha256": sha256_file(config_path),
        "canary": {
            "sequence": sequence,
            "frame": frame,
            "label_object_count": len(label_payload["labels"][frame]),
            "timestamp": timestamp,
            "remote_member": member,
            "jpeg": {
                **geometry,
                "bytes": len(jpeg),
                "sha256": image_sha,
                "output": image_output.resolve().relative_to(repo.resolve()).as_posix(),
            },
            "calibration_members": sorted(calibration_members),
        },
        "network": {
            "archive_url": remote["url"],
            "archive_bytes": remote["content_length"],
            "archive_etag": remote["etag"],
            "bytes_read": client.bytes_read,
            "budget_bytes": client.budget,
            "requests": client.requests,
            "full_archive_downloaded": False,
        },
        "claim_boundary": {
            "source_authority_canary_only": True,
            "g1_authorized": False,
            "signal_authorized": False,
            "route_truth_authorized": False,
            "android_authorized": False,
            "human_authorized": False,
            "production_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve()
    result = audit(repo, config_path, persist_image=True)
    output = repo / load_json(config_path)["outputs"]["receipt"]
    atomic_write(output, canonical_bytes(result))
    print(json.dumps({"terminal_state": result["terminal_state"], "process_id": result["process_id"], "network_bytes": result["network"]["bytes_read"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

