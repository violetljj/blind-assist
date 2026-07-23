#!/usr/bin/env python3
"""Read a remote ZIP central directory without downloading archive payloads."""

from __future__ import annotations

import argparse
import json
import struct
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
CENTRAL_ENTRY_SIGNATURE = b"PK\x01\x02"
LOCAL_ENTRY_SIGNATURE = b"PK\x03\x04"
MAX_DIRECTORY_BYTES = 64 * 1024 * 1024


def range_get(url: str, start: int, end: int) -> tuple[bytes, dict[str, str]]:
    last_error: BaseException | None = None
    for attempt in range(9):
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "BlindAssist-USTRF-research/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", None)
                headers = {key.lower(): value for key, value in response.headers.items()}
                if status != 206:
                    raise RuntimeError(
                        f"server did not honor byte range before body read: HTTP {status}"
                    )
                content_range = headers.get("content-range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"unexpected Content-Range before body read: {content_range!r}")
                expected = end - start + 1
                content_length = headers.get("content-length")
                if content_length is not None and int(content_length) != expected:
                    raise RuntimeError(
                        f"range Content-Length mismatch before body read: expected {expected}, got {content_length}"
                    )
                payload = response.read(expected + 1)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == 8:
                raise
            time.sleep(min(2**attempt, 30))
    else:  # pragma: no cover - loop either breaks or raises
        raise RuntimeError("range request retry loop exhausted") from last_error
    expected = end - start + 1
    if len(payload) != expected:
        raise RuntimeError(f"range length mismatch: expected {expected}, got {len(payload)}")
    return payload, headers


def zip64_value(extra: bytes, field_index: int, required_fields: list[bool]) -> int:
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        data = extra[cursor : cursor + size]
        cursor += size
        if header_id != 0x0001:
            continue
        values: list[int] = []
        data_cursor = 0
        for required in required_fields:
            if not required:
                values.append(-1)
                continue
            if data_cursor + 8 > len(data):
                raise RuntimeError("truncated ZIP64 extended information")
            values.append(struct.unpack_from("<Q", data, data_cursor)[0])
            data_cursor += 8
        return values[field_index]
    raise RuntimeError("ZIP64 sentinel present without ZIP64 extended information")


def parse_entries(directory: bytes, expected_count: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(directory):
        if directory[cursor : cursor + 4] != CENTRAL_ENTRY_SIGNATURE:
            raise RuntimeError(f"invalid central directory signature at byte {cursor}")
        if cursor + 46 > len(directory):
            raise RuntimeError("truncated central directory entry")
        values = struct.unpack_from("<6H3I5H2I", directory, cursor + 4)
        (
            _made_by,
            _needed,
            flags,
            method,
            _mtime,
            _mdate,
            crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            _disk_start,
            _internal_attributes,
            _external_attributes,
            local_offset,
        ) = values
        variable_start = cursor + 46
        name_raw = directory[variable_start : variable_start + name_length]
        extra_start = variable_start + name_length
        extra = directory[extra_start : extra_start + extra_length]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_raw.decode(encoding, errors="replace")
        required = [
            uncompressed_size == 0xFFFFFFFF,
            compressed_size == 0xFFFFFFFF,
            local_offset == 0xFFFFFFFF,
        ]
        if required[0]:
            uncompressed_size = zip64_value(extra, 0, required)
        if required[1]:
            compressed_size = zip64_value(extra, 1, required)
        if required[2]:
            local_offset = zip64_value(extra, 2, required)
        entries.append(
            {
                "name": name,
                "compression_method": method,
                "flags": flags,
                "crc32": f"{crc32:08x}",
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_header_offset": local_offset,
                "is_directory": name.endswith("/"),
            }
        )
        cursor = extra_start + extra_length + comment_length
    if len(entries) != expected_count:
        raise RuntimeError(f"entry count mismatch: expected {expected_count}, got {len(entries)}")
    return entries


def inspect(url: str, archive_size: int) -> dict[str, Any]:
    tail_size = min(archive_size, 65_557)
    tail_start = archive_size - tail_size
    tail, tail_headers = range_get(url, tail_start, archive_size - 1)
    eocd_in_tail = tail.rfind(EOCD_SIGNATURE)
    if eocd_in_tail < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    eocd_offset = tail_start + eocd_in_tail
    (
        disk_number,
        directory_disk,
        _entries_on_disk,
        entry_count,
        directory_size,
        directory_offset,
        comment_length,
    ) = struct.unpack_from("<4H2IH", tail, eocd_in_tail + 4)
    if disk_number != 0 or directory_disk != 0:
        raise RuntimeError("multi-disk ZIP archives are not supported")
    zip64 = entry_count == 0xFFFF or directory_size == 0xFFFFFFFF or directory_offset == 0xFFFFFFFF
    extra_range_bytes = 0
    if zip64:
        locator_offset = eocd_offset - 20
        locator, _ = range_get(url, locator_offset, eocd_offset - 1)
        extra_range_bytes += len(locator)
        if locator[:4] != ZIP64_LOCATOR_SIGNATURE:
            raise RuntimeError("ZIP64 locator not found")
        _disk, zip64_offset, _disk_count = struct.unpack_from("<IQI", locator, 4)
        zip64_record, _ = range_get(url, zip64_offset, zip64_offset + 55)
        extra_range_bytes += len(zip64_record)
        if zip64_record[:4] != ZIP64_EOCD_SIGNATURE:
            raise RuntimeError("ZIP64 end-of-central-directory record not found")
        (
            _record_size,
            _made_by,
            _needed,
            zip64_disk,
            zip64_directory_disk,
            _entries_on_disk_64,
            entry_count,
            directory_size,
            directory_offset,
        ) = struct.unpack_from("<Q2H2I4Q", zip64_record, 4)
        if zip64_disk != 0 or zip64_directory_disk != 0:
            raise RuntimeError("multi-disk ZIP64 archives are not supported")
    if directory_size > MAX_DIRECTORY_BYTES:
        raise RuntimeError(f"central directory exceeds safety cap: {directory_size} bytes")
    if directory_offset + directory_size > eocd_offset:
        raise RuntimeError("central directory overlaps end record")
    directory, _ = range_get(url, directory_offset, directory_offset + directory_size - 1)
    entries = parse_entries(directory, entry_count)
    suffixes = Counter()
    top_levels = Counter()
    for entry in entries:
        name = entry["name"].rstrip("/")
        if not name:
            continue
        top_levels[name.split("/", 1)[0]] += 1
        leaf = name.rsplit("/", 1)[-1]
        suffix = Path(leaf).suffix.lower() or "<none>"
        suffixes[suffix] += 1
    return {
        "schema": "blindassist_remote_zip_central_directory_inventory_r1",
        "url": url,
        "archive_size": archive_size,
        "zip64": zip64,
        "entry_count": entry_count,
        "central_directory_offset": directory_offset,
        "central_directory_size": directory_size,
        "archive_payload_bytes_downloaded": 0,
        "metadata_range_bytes_downloaded": tail_size + extra_range_bytes + directory_size,
        "range_supported": tail_headers.get("accept-ranges", "").lower() == "bytes",
        "sequence_content_decoded": False,
        "filename_inventory_decoded": True,
        "suffix_counts": dict(sorted(suffixes.items())),
        "top_level_counts": dict(sorted(top_levels.items())),
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--archive-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = inspect(args.url, args.archive_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("entry_count", "central_directory_size", "metadata_range_bytes_downloaded", "suffix_counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
