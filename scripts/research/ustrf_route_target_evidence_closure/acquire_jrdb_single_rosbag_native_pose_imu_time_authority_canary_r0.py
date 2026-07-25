#!/usr/bin/env python3
"""Range-acquire and stream-decompress one frozen JRDB rosbag ZIP member."""
from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import struct
import urllib.request
import zlib
from pathlib import Path

from audit_jrdb_native_pose_and_3d_person_motion_authority_r0 import RangeReader, parse_central

STAGE = "JRDB_SINGLE_ROSBAG_NATIVE_POSE_IMU_TIME_AUTHORITY_CANARY_R0"
CONFIG_SCHEMA = "blindassist_ustrf_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0_config"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(config_path: Path, output_bag: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["schema"] == CONFIG_SCHEMA and config["stage"] == STAGE, "config_identity")
    remote = config["remote_zip"]
    expected = remote["member"]
    reader = RangeReader(int(config["resource_gate"]["maximum_network_bytes"]))
    start = int(remote["central_directory_offset"])
    central = parse_central(reader.get(remote["url"], start, start + int(remote["central_directory_size"]) - 1))
    matches = [item for item in central if item["name"] == expected["name"]]
    require(len(matches) == 1, "frozen_member_not_unique")
    member = matches[0]
    for observed_key, expected_key in (
        ("flags", "flags"), ("method", "method"), ("crc32", "crc32"),
        ("compressed", "compressed_size"), ("uncompressed", "uncompressed_size"),
        ("offset", "local_header_offset"),
    ):
        require(int(member[observed_key]) == int(expected[expected_key]), f"member_drift:{observed_key}")
    require(member["method"] == 8 and member["flags"] & 1 == 0, "member_not_plain_deflate")
    offset = int(member["offset"])
    header = reader.get(remote["url"], offset, offset + 29)
    values = struct.unpack("<4s5H3L2H", header)
    require(values[0] == b"PK\x03\x04", "local_header_signature")
    name_len, extra_len = values[-2], values[-1]
    tail = reader.get(remote["url"], offset + 30, offset + 30 + name_len + extra_len - 1)
    require(tail[:name_len].decode("utf-8") == member["name"], "local_name_drift")
    data_start = offset + 30 + name_len + extra_len
    requested = int(member["compressed"])
    require(reader.bytes_read + requested <= reader.budget, "network_budget_exceeded")
    request = urllib.request.Request(
        remote["url"],
        headers={"Range": f"bytes={data_start}-{data_start + requested - 1}"},
    )
    output_bag.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_bag.with_suffix(output_bag.suffix + ".partial")
    decompressor = zlib.decompressobj(-15)
    digest = hashlib.sha256()
    crc = 0
    compressed_read = 0
    uncompressed_written = 0
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as sink:
        require(response.status == 206, f"range_status:{response.status}")
        while compressed_read < requested:
            block = response.read(min(4 * 1024 * 1024, requested - compressed_read))
            require(bool(block), "truncated_member_response")
            compressed_read += len(block)
            raw = decompressor.decompress(block)
            if raw:
                sink.write(raw)
                digest.update(raw)
                crc = binascii.crc32(raw, crc)
                uncompressed_written += len(raw)
        raw = decompressor.flush()
        if raw:
            sink.write(raw)
            digest.update(raw)
            crc = binascii.crc32(raw, crc)
            uncompressed_written += len(raw)
    reader.bytes_read += compressed_read
    reader.requests.append({
        "url": remote["url"], "start": data_start,
        "end": data_start + requested - 1, "bytes": compressed_read,
    })
    require(decompressor.eof, "deflate_stream_incomplete")
    require(uncompressed_written == int(member["uncompressed"]), "uncompressed_size_drift")
    require((crc & 0xFFFFFFFF) == int(member["crc32"]), "member_crc_drift")
    require(uncompressed_written <= int(config["resource_gate"]["maximum_local_bag_bytes"]), "local_bag_budget_exceeded")
    temporary.replace(output_bag)
    return {
        "schema": "blindassist_ustrf_jrdb_single_rosbag_acquisition_r0",
        "stage": STAGE,
        "status": "ACQUIRED",
        "config_sha256": sha256_file(config_path),
        "member": member,
        "network": {
            "bytes_read": reader.bytes_read,
            "budget": reader.budget,
            "full_archive_downloaded": False,
            "second_bag_downloaded": False,
            "requests": reader.requests,
        },
        "bag": {
            "path": output_bag.as_posix(),
            "bytes": uncompressed_written,
            "sha256": digest.hexdigest(),
            "crc32": crc & 0xFFFFFFFF,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-bag", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = acquire(args.config.resolve(), args.output_bag.resolve())
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "bag": result["bag"], "network_bytes": result["network"]["bytes_read"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
