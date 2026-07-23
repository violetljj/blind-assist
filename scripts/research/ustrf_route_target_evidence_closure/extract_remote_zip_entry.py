#!/usr/bin/env python3
"""Extract one hash-verifiable ZIP entry using HTTP byte ranges."""

from __future__ import annotations

import argparse
import binascii
import json
import struct
import zlib
from pathlib import Path

from inspect_remote_zip_inventory import LOCAL_ENTRY_SIGNATURE, range_get


MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    matches = [row for row in inventory["entries"] if row["name"] == args.entry]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one matching entry, got {len(matches)}")
    entry = matches[0]
    compressed_size = int(entry["compressed_size"])
    uncompressed_size = int(entry["uncompressed_size"])
    if compressed_size > MAX_COMPRESSED_BYTES or uncompressed_size > MAX_UNCOMPRESSED_BYTES:
        raise RuntimeError("entry exceeds bounded metadata extraction cap")
    offset = int(entry["local_header_offset"])
    local_header, _ = range_get(inventory["url"], offset, offset + 29)
    if local_header[:4] != LOCAL_ENTRY_SIGNATURE:
        raise RuntimeError("invalid local file header signature")
    (
        _needed,
        flags,
        method,
        _mtime,
        _mdate,
        _crc32,
        _compressed_size,
        _uncompressed_size,
        name_length,
        extra_length,
    ) = struct.unpack_from("<5H3I2H", local_header, 4)
    if flags & 0x1:
        raise RuntimeError("encrypted ZIP entries are unsupported")
    data_start = offset + 30 + name_length + extra_length
    compressed, _ = range_get(inventory["url"], data_start, data_start + compressed_size - 1)
    if method == 0:
        payload = compressed
    elif method == 8:
        payload = zlib.decompress(compressed, wbits=-15)
    else:
        raise RuntimeError(f"unsupported ZIP compression method: {method}")
    if len(payload) != uncompressed_size:
        raise RuntimeError(f"uncompressed size mismatch: expected {uncompressed_size}, got {len(payload)}")
    crc32 = f"{binascii.crc32(payload) & 0xFFFFFFFF:08x}"
    if crc32 != entry["crc32"]:
        raise RuntimeError(f"CRC32 mismatch: expected {entry['crc32']}, got {crc32}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(json.dumps({"entry": args.entry, "bytes": len(payload), "crc32": crc32}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
