"""Reliably range-acquire one hash-bound JRDB rosbag ZIP member.

The JRDB archive host can close a long response before a large deflate member
finishes. This reader keeps the decompressor state and reconnects at the next
compressed byte instead of restarting the full archive member.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
from pathlib import Path
import struct
from typing import Any
import urllib.request
import zlib


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        require(response.status == 206, f"range_status:{response.status}")
        payload = response.read(end - start + 2)
    require(len(payload) == end - start + 1, "range_length_drift")
    return payload


def zip64_values(
    extra: bytes, compressed: int, uncompressed: int, offset: int
) -> tuple[int, int, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        field = extra[cursor + 4 : cursor + 4 + field_size]
        cursor += 4 + field_size
        if field_id != 1:
            continue
        position = 0
        if uncompressed == 0xFFFFFFFF:
            uncompressed = struct.unpack_from("<Q", field, position)[0]
            position += 8
        if compressed == 0xFFFFFFFF:
            compressed = struct.unpack_from("<Q", field, position)[0]
            position += 8
        if offset == 0xFFFFFFFF:
            offset = struct.unpack_from("<Q", field, position)[0]
        break
    return compressed, uncompressed, offset


def parse_central(payload: bytes) -> list[dict[str, Any]]:
    cursor = 0
    members = []
    while cursor < len(payload):
        require(payload[cursor : cursor + 4] == b"PK\x01\x02", "central_signature")
        values = struct.unpack_from("<4s6H3L5H2L", payload, cursor)
        flags, method, crc32 = values[3], values[4], values[7]
        compressed, uncompressed = values[8], values[9]
        name_length, extra_length, comment_length = values[10:13]
        offset = values[16]
        start = cursor + 46
        name = payload[start : start + name_length].decode("utf-8")
        extra = payload[
            start + name_length : start + name_length + extra_length
        ]
        compressed, uncompressed, offset = zip64_values(
            extra, compressed, uncompressed, offset
        )
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
        cursor = start + name_length + extra_length + comment_length
    require(cursor == len(payload), "central_length_drift")
    return members


def consume_member(
    *,
    url: str,
    data_start: int,
    compressed_size: int,
    output_partial: Path,
    maximum_reconnects: int,
) -> tuple[int, int, int, str, list[dict[str, Any]], zlib.Decompress]:
    decompressor = zlib.decompressobj(-15)
    digest = hashlib.sha256()
    crc = 0
    compressed_read = 0
    uncompressed_written = 0
    requests = []
    reconnects = 0
    next_progress = 128 * 1024 * 1024
    with output_partial.open("wb") as sink:
        while compressed_read < compressed_size:
            request_start = data_start + compressed_read
            request_end = data_start + compressed_size - 1
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={request_start}-{request_end}"}
            )
            bytes_this_request = 0
            failure: str | None = None
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    require(response.status == 206, f"range_status:{response.status}")
                    while compressed_read < compressed_size:
                        block = response.read(
                            min(4 * 1024 * 1024, compressed_size - compressed_read)
                        )
                        if not block:
                            failure = "early_eof"
                            break
                        compressed_read += len(block)
                        bytes_this_request += len(block)
                        raw = decompressor.decompress(block)
                        if raw:
                            sink.write(raw)
                            digest.update(raw)
                            crc = binascii.crc32(raw, crc)
                            uncompressed_written += len(raw)
                        if compressed_read >= next_progress:
                            print(
                                f"compressed={compressed_read}/{compressed_size}",
                                flush=True,
                            )
                            next_progress += 128 * 1024 * 1024
            except Exception as error:  # reconnect preserves decompressor state
                failure = f"{type(error).__name__}:{error}"
            requests.append(
                {
                    "start": request_start,
                    "end": request_start + bytes_this_request - 1,
                    "bytes": bytes_this_request,
                    "failure": failure,
                }
            )
            if compressed_read >= compressed_size:
                break
            reconnects += 1
            require(reconnects <= maximum_reconnects, "reconnect_budget_exceeded")
            print(
                f"reconnect={reconnects} next_compressed_byte={compressed_read} "
                f"reason={failure}",
                flush=True,
            )
        raw = decompressor.flush()
        if raw:
            sink.write(raw)
            digest.update(raw)
            crc = binascii.crc32(raw, crc)
            uncompressed_written += len(raw)
    return (
        compressed_read,
        uncompressed_written,
        crc & 0xFFFFFFFF,
        digest.hexdigest(),
        requests,
        decompressor,
    )


def acquire(config_path: Path, output_bag: Path, maximum_reconnects: int) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    remote = config["remote_zip"]
    expected = remote["member"]
    central_start = int(remote["central_directory_offset"])
    central_size = int(remote["central_directory_size"])
    central = parse_central(
        get_range(remote["url"], central_start, central_start + central_size - 1)
    )
    matches = [member for member in central if member["name"] == expected["name"]]
    require(len(matches) == 1, "frozen_member_not_unique")
    member = matches[0]
    for observed, frozen in (
        ("flags", "flags"),
        ("method", "method"),
        ("crc32", "crc32"),
        ("compressed", "compressed_size"),
        ("uncompressed", "uncompressed_size"),
        ("offset", "local_header_offset"),
    ):
        require(int(member[observed]) == int(expected[frozen]), f"member_drift:{observed}")
    require(member["method"] == 8 and member["flags"] & 1 == 0, "member_not_plain_deflate")
    offset = int(member["offset"])
    header = get_range(remote["url"], offset, offset + 29)
    values = struct.unpack("<4s5H3L2H", header)
    require(values[0] == b"PK\x03\x04", "local_header_signature")
    name_length, extra_length = values[-2:]
    tail = get_range(
        remote["url"], offset + 30, offset + 30 + name_length + extra_length - 1
    )
    require(tail[:name_length].decode("utf-8") == member["name"], "local_name_drift")
    data_start = offset + 30 + name_length + extra_length
    output_bag.parent.mkdir(parents=True, exist_ok=True)
    require(not output_bag.exists(), "refusing_to_overwrite_completed_bag")
    partial = output_bag.with_suffix(output_bag.suffix + ".partial")
    compressed_read, written, crc, digest, requests, decompressor = consume_member(
        url=remote["url"],
        data_start=data_start,
        compressed_size=int(member["compressed"]),
        output_partial=partial,
        maximum_reconnects=maximum_reconnects,
    )
    require(compressed_read == int(member["compressed"]), "compressed_size_drift")
    require(decompressor.eof, "deflate_stream_incomplete")
    require(written == int(member["uncompressed"]), "uncompressed_size_drift")
    require(crc == int(member["crc32"]), "member_crc_drift")
    require(
        written <= int(config["resource_gate"]["maximum_local_bag_bytes"]),
        "local_bag_budget_exceeded",
    )
    partial.replace(output_bag)
    return {
        "schema": "dtr-r0-jrdb-range-acquisition-v1",
        "status": "ACQUIRED",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "member": member,
        "requests": requests,
        "bag": {
            "path": str(output_bag.resolve()),
            "bytes": written,
            "sha256": digest,
            "crc32": crc,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-bag", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--maximum-reconnects", type=int, default=8)
    args = parser.parse_args()
    result = acquire(
        args.config.resolve(), args.output_bag.resolve(), args.maximum_reconnects
    )
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "bag": result["bag"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
