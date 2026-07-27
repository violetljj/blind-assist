#!/usr/bin/env python3
"""List or extract selected members from a remote ZIP using HTTP byte ranges.

The archive itself is never downloaded in full. Python's ZIP64-aware ``zipfile``
reader is backed by a seekable HTTP Range reader. Extraction is fail-closed:
member names must be relative, symlinks are rejected, and optional source
length/ETag/MD5 locks can bind the download to an attested object.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import stat
import struct
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath


class HttpRangeReader(io.RawIOBase):
    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url
        self.timeout = timeout
        self.position = 0
        self.requests = 0
        self.transferred = 0
        self.headers = self._range(0, 0)[1]
        content_range = self.headers.get("Content-Range", "")
        if "/" not in content_range:
            raise RuntimeError(f"server did not return Content-Range: {content_range!r}")
        self.length = int(content_range.rsplit("/", 1)[1])

    def _range(self, start: int, end: int) -> tuple[bytes, object]:
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "BlindAssist-source-fetch/1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if response.status != 206:
                raise RuntimeError(f"HTTP Range unsupported: expected 206, got {response.status}")
            body = response.read()
            self.requests += 1
            self.transferred += len(body)
            return body, response.headers

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def readinto(self, buffer: bytearray) -> int:
        if self.position >= self.length or not buffer:
            return 0
        end = min(self.length - 1, self.position + len(buffer) - 1)
        body, _ = self._range(self.position, end)
        count = len(body)
        buffer[:count] = body
        self.position += count
        return count


def source_metadata(reader: HttpRangeReader) -> dict[str, object]:
    headers = reader.headers
    return {
        "url": reader.url,
        "content_length": reader.length,
        "etag": headers.get("ETag", "").strip('"'),
        "generation": headers.get("x-goog-generation", ""),
        "md5_base64": next(
            (item.split("=", 1)[1] for item in headers.get_all("x-goog-hash", []) if item.startswith("md5=")),
            "",
        ),
    }


def verify_source(metadata: dict[str, object], args: argparse.Namespace) -> None:
    expected = {
        "content_length": args.expect_length,
        "etag": args.expect_etag,
        "md5_base64": args.expect_md5_base64,
    }
    for key, value in expected.items():
        if value is not None and metadata[key] != value:
            raise RuntimeError(f"source {key} mismatch: expected {value!r}, got {metadata[key]!r}")


def safe_destination(root: Path, name: str) -> Path:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise RuntimeError(f"unsafe ZIP member path: {name!r}")
    destination = root.joinpath(*member.parts).resolve()
    root_resolved = root.resolve()
    if root_resolved != destination and root_resolved not in destination.parents:
        raise RuntimeError(f"ZIP member escapes output root: {name!r}")
    return destination


def is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


def matches(name: str, includes: list[str], excludes: list[str]) -> bool:
    return (not includes or any(fnmatch.fnmatchcase(name, pattern) for pattern in includes)) and not any(
        fnmatch.fnmatchcase(name, pattern) for pattern in excludes
    )


def extract_coalesced(
    reader: HttpRangeReader,
    archive: zipfile.ZipFile,
    selected: list[zipfile.ZipInfo],
    output_root: Path,
    maximum_gap_bytes: int,
) -> list[dict[str, object]]:
    """Extract selected members with a few contiguous HTTP range requests."""
    if maximum_gap_bytes < 0:
        raise ValueError("coalesce gap must be non-negative")
    all_files = sorted(
        (info for info in archive.infolist() if not info.is_dir()),
        key=lambda item: item.header_offset,
    )
    next_offset: dict[int, int] = {}
    for index, info in enumerate(all_files):
        next_offset[info.header_offset] = (
            all_files[index + 1].header_offset
            if index + 1 < len(all_files)
            else archive.start_dir
        )
    spans = [
        (info.header_offset, next_offset[info.header_offset], info)
        for info in sorted(
            (item for item in selected if not item.is_dir()),
            key=lambda item: item.header_offset,
        )
    ]
    runs: list[tuple[int, int, list[zipfile.ZipInfo]]] = []
    for start, end, info in spans:
        if runs and start - runs[-1][1] <= maximum_gap_bytes:
            previous_start, _, previous_infos = runs[-1]
            runs[-1] = (previous_start, end, [*previous_infos, info])
        else:
            runs.append((start, end, [info]))

    receipts: list[dict[str, object]] = []
    for run_start, run_end, infos in runs:
        payload, _ = reader._range(run_start, run_end - 1)
        for info in infos:
            relative = info.header_offset - run_start
            header = payload[relative : relative + 30]
            if len(header) != 30:
                raise RuntimeError(f"truncated local ZIP header: {info.filename!r}")
            (
                signature,
                _version,
                flags,
                compression,
                _mtime,
                _mdate,
                _crc32,
                _compressed_size,
                _file_size,
                name_length,
                extra_length,
            ) = struct.unpack("<IHHHHHIIIHH", header)
            if signature != 0x04034B50 or flags & 0x1:
                raise RuntimeError(f"unsupported local ZIP header: {info.filename!r}")
            data_start = relative + 30 + name_length + extra_length
            compressed = payload[data_start : data_start + info.compress_size]
            if len(compressed) != info.compress_size:
                raise RuntimeError(f"truncated ZIP member: {info.filename!r}")
            if compression == zipfile.ZIP_STORED:
                data = compressed
            elif compression == zipfile.ZIP_DEFLATED:
                data = zlib.decompress(compressed, -15)
            else:
                raise RuntimeError(
                    f"coalesced extraction compression unsupported: {compression}"
                )
            if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
                raise RuntimeError(f"ZIP member CRC/size mismatch: {info.filename!r}")
            destination = safe_destination(output_root, info.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            receipts.append(
                {
                    "path": info.filename,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
    for info in selected:
        if info.is_dir():
            safe_destination(output_root, info.filename).mkdir(
                parents=True, exist_ok=True
            )
    return [
        {
            "range_start": start,
            "range_end_exclusive": end,
            "member_count": len(infos),
            "bytes": end - start,
        }
        for start, end, infos in runs
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--include", action="append", default=[], help="case-sensitive glob; repeatable")
    parser.add_argument(
        "--include-from",
        type=Path,
        help="UTF-8 file with one case-sensitive glob per line; blank and # lines ignored",
    )
    parser.add_argument("--exclude", action="append", default=[], help="case-sensitive glob; repeatable")
    parser.add_argument("--list", action="store_true", help="print matching members as JSON Lines")
    parser.add_argument("--output", type=Path, help="extract matching members under this directory")
    parser.add_argument("--inventory", type=Path, help="write source and selected-member inventory JSON")
    parser.add_argument("--expect-length", type=int)
    parser.add_argument("--expect-etag")
    parser.add_argument("--expect-md5-base64")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--coalesce-gap-bytes",
        type=int,
        help="extract selected members in contiguous range runs; avoids one request per member",
    )
    args = parser.parse_args()
    include_file_sha256 = None
    if args.include_from is not None:
        include_payload = args.include_from.read_bytes()
        include_file_sha256 = hashlib.sha256(include_payload).hexdigest()
        args.include.extend(
            line.strip()
            for line in include_payload.decode("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not args.list and args.output is None and args.inventory is None:
        parser.error("at least one of --list, --output, or --inventory is required")

    reader = HttpRangeReader(args.url, args.timeout)
    metadata = source_metadata(reader)
    verify_source(metadata, args)
    selected: list[zipfile.ZipInfo] = []
    with zipfile.ZipFile(reader) as archive:
        for info in archive.infolist():
            if matches(info.filename, args.include, args.exclude):
                if is_symlink(info):
                    raise RuntimeError(f"symlink member rejected: {info.filename!r}")
                selected.append(info)
                if args.list:
                    print(json.dumps({
                        "path": info.filename,
                        "size": info.file_size,
                        "compressed_size": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                    }, ensure_ascii=False))
        coalesced_runs: list[dict[str, object]] = []
        if args.output is not None and args.coalesce_gap_bytes is not None:
            coalesced_runs = extract_coalesced(
                reader,
                archive,
                selected,
                args.output,
                args.coalesce_gap_bytes,
            )
        elif args.output is not None:
            for info in selected:
                destination = safe_destination(args.output, info.filename)
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                hasher = hashlib.sha256()
                with archive.open(info) as source, destination.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                        hasher.update(chunk)
                print(json.dumps({"extracted": info.filename, "sha256": hasher.hexdigest()}, ensure_ascii=False))

    inventory = {
        "schema": "blindassist.remote_zip_inventory.v1",
        "source": metadata,
        "selection": {
            "include": args.include,
            "include_from": (
                {
                    "path": str(args.include_from),
                    "sha256": include_file_sha256,
                }
                if args.include_from is not None
                else None
            ),
            "exclude": args.exclude,
        },
        "members": [{
            "path": item.filename,
            "size": item.file_size,
            "compressed_size": item.compress_size,
            "crc32": f"{item.CRC:08x}",
        } for item in selected],
        "http_range_requests": reader.requests,
        "http_bytes_transferred": reader.transferred,
        "coalesced_runs": coalesced_runs,
    }
    if args.inventory is not None:
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_members": len(selected),
        "selected_uncompressed_bytes": sum(item.file_size for item in selected),
        "range_requests": reader.requests,
        "bytes_transferred": reader.transferred,
    }, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
