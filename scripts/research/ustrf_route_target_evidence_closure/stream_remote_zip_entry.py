#!/usr/bin/env python3
"""Stream one large ZIP entry from an HTTP-range source to a verified file."""

from __future__ import annotations

import argparse
import binascii
import concurrent.futures
import hashlib
import json
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Callable

from inspect_remote_zip_inventory import LOCAL_ENTRY_SIGNATURE, range_get


CHUNK_BYTES = 4 * 1024 * 1024
PROGRESS_BYTES = 256 * 1024 * 1024
MAX_RANGE_WORKERS = 12
MAX_RANGE_PARTS = 128


def partition_range(start: int, size: int, parts: int) -> list[tuple[int, int]]:
    if size <= 0:
        raise ValueError("range size must be positive")
    if parts < 1 or parts > MAX_RANGE_PARTS:
        raise ValueError(f"range parts must be between 1 and {MAX_RANGE_PARTS}")
    part_count = min(parts, size)
    base, remainder = divmod(size, part_count)
    parts: list[tuple[int, int]] = []
    cursor = start
    for index in range(part_count):
        part_size = base + (1 if index < remainder else 0)
        end = cursor + part_size - 1
        parts.append((cursor, end))
        cursor = end + 1
    if cursor != start + size:
        raise RuntimeError("range partition did not cover the requested bytes")
    return parts


def download_range_part(
    *,
    url: str,
    start: int,
    end: int,
    output: Path,
    progress: Callable[[int], None],
    request_timeout_seconds: int,
) -> dict[str, object]:
    expected_size = end - start + 1
    if expected_size <= 0:
        raise ValueError("range part must contain at least one byte")
    if output.exists() and output.stat().st_size == expected_size:
        return {
            "path": output,
            "range": [start, end],
            "bytes": expected_size,
            "sha256": sha256_file(output),
            "reused": True,
        }
    if output.exists() and output.stat().st_size > expected_size:
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = output.stat().st_size if output.exists() else 0
    retry_count = 0
    mode = "ab" if output.exists() else "xb"
    with output.open(mode) as stream:
        while downloaded < expected_size:
            request_start = start + downloaded
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={request_start}-{end}",
                    "User-Agent": "BlindAssist-USTRF-research/1.0",
                },
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=request_timeout_seconds,
                ) as response:
                    status = getattr(response, "status", None)
                    content_range = response.headers.get("Content-Range", "")
                    if status != 206 or not content_range.startswith(f"bytes {request_start}-{end}/"):
                        raise RuntimeError(
                            f"server did not honor part byte range: HTTP {status}, {content_range!r}"
                        )
                    while downloaded < expected_size:
                        chunk = response.read(min(CHUNK_BYTES, expected_size - downloaded))
                        if not chunk:
                            raise EOFError("part range ended before declared size")
                        stream.write(chunk)
                        downloaded += len(chunk)
                        progress(len(chunk))
                retry_count = 0
            except (urllib.error.URLError, TimeoutError, EOFError, OSError):
                retry_count += 1
                if retry_count > 8:
                    raise
                stream.flush()
                time.sleep(min(2 ** (retry_count - 1), 30))
    if output.stat().st_size != expected_size:
        raise RuntimeError(f"part size mismatch: expected {expected_size}, got {output.stat().st_size}")
    return {
        "path": output,
        "range": [start, end],
        "bytes": expected_size,
        "sha256": sha256_file(output),
        "reused": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_cached_parts(
    *,
    parts: list[dict[str, object]],
    method: int,
    output: Path,
    expected_uncompressed_size: int,
) -> tuple[int, str, str]:
    decompressor = zlib.decompressobj(wbits=-15) if method == 8 else None
    crc = 0
    sha256 = hashlib.sha256()
    uncompressed_written = 0
    with output.open("xb") as target:
        for part in parts:
            path = part["path"]
            if not isinstance(path, Path):
                raise TypeError("cached part path is invalid")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                    decoded = decompressor.decompress(chunk) if decompressor else chunk
                    if decoded:
                        target.write(decoded)
                        uncompressed_written += len(decoded)
                        crc = binascii.crc32(decoded, crc)
                        sha256.update(decoded)
                        if uncompressed_written > expected_uncompressed_size:
                            raise RuntimeError("decoded more bytes than declared")
        if decompressor:
            decoded = decompressor.flush()
            if decoded:
                target.write(decoded)
                uncompressed_written += len(decoded)
                crc = binascii.crc32(decoded, crc)
                sha256.update(decoded)
            if not decompressor.eof:
                raise RuntimeError("deflate stream ended before end-of-stream marker")
    return uncompressed_written, f"{crc & 0xFFFFFFFF:08x}", sha256.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--max-compressed-bytes", required=True, type=int)
    parser.add_argument("--max-uncompressed-bytes", required=True, type=int)
    parser.add_argument("--range-workers", type=int)
    parser.add_argument("--range-parts", type=int)
    parser.add_argument("--request-timeout-seconds", type=int)
    parser.add_argument("--compressed-cache-root", type=Path)
    parser.add_argument("--reuse-receipt", action="store_true")
    args = parser.parse_args()
    transport_config_candidates = [
        args.receipt.parent / "transport-acceleration-r7.json",
        args.receipt.parent / "transport-acceleration-r6.json",
        args.receipt.parent / "transport-acceleration-r5.json",
        args.receipt.parent / "transport-acceleration-r4.json",
        args.receipt.parent / "transport-acceleration-r3.json",
        args.receipt.parent / "transport-acceleration-r2.json",
        args.receipt.parent / "transport-acceleration-r1.json",
    ]
    transport_config_path = next(
        (path for path in transport_config_candidates if path.is_file()),
        transport_config_candidates[-1],
    )
    transport_config_sha256 = None
    if args.range_workers is None and transport_config_path.is_file():
        transport_config_bytes = transport_config_path.read_bytes()
        transport_config = json.loads(transport_config_bytes)
        if transport_config.get("schema") != "blindassist_range_transport_acceleration_r1":
            raise RuntimeError("unsupported transport acceleration config schema")
        if transport_config.get("candidate_outputs_executed") is not False:
            raise RuntimeError("transport acceleration config must remain candidate blind")
        args.range_workers = int(transport_config["range_workers"])
        args.range_parts = int(transport_config.get("range_parts", args.range_workers))
        args.request_timeout_seconds = int(
            transport_config.get("request_timeout_seconds", 120)
        )
        args.compressed_cache_root = Path(transport_config["compressed_cache_root"])
        transport_config_sha256 = hashlib.sha256(transport_config_bytes).hexdigest()
    if args.range_workers is None:
        args.range_workers = 1
    if args.range_parts is None:
        args.range_parts = args.range_workers
    if args.request_timeout_seconds is None:
        args.request_timeout_seconds = 120
    if args.range_workers < 1 or args.range_workers > MAX_RANGE_WORKERS:
        raise RuntimeError(f"--range-workers must be between 1 and {MAX_RANGE_WORKERS}")
    if args.range_parts < args.range_workers or args.range_parts > MAX_RANGE_PARTS:
        raise RuntimeError(
            f"--range-parts must be between --range-workers and {MAX_RANGE_PARTS}"
        )
    if args.request_timeout_seconds < 15 or args.request_timeout_seconds > 300:
        raise RuntimeError("--request-timeout-seconds must be between 15 and 300")
    if args.range_workers > 1 and args.compressed_cache_root is None:
        raise RuntimeError("--compressed-cache-root is required when --range-workers is greater than 1")
    inventory_bytes = args.inventory.read_bytes()
    inventory = json.loads(inventory_bytes)
    matches = [row for row in inventory["entries"] if row["name"] == args.entry]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one matching entry, got {len(matches)}")
    entry = matches[0]
    compressed_size = int(entry["compressed_size"])
    uncompressed_size = int(entry["uncompressed_size"])
    if compressed_size > args.max_compressed_bytes:
        raise RuntimeError("entry exceeds explicit compressed-byte cap")
    if uncompressed_size > args.max_uncompressed_bytes:
        raise RuntimeError("entry exceeds explicit uncompressed-byte cap")
    if args.output.exists():
        raise RuntimeError("output already exists; refusing to overwrite")
    existing_receipt = None
    if args.reuse_receipt:
        if not args.receipt.is_file():
            raise RuntimeError("--reuse-receipt requires an existing receipt")
        existing_receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        if (
            existing_receipt.get("inventory_sha256") != hashlib.sha256(inventory_bytes).hexdigest()
            or existing_receipt.get("url") != inventory.get("url")
            or existing_receipt.get("entry") != args.entry
            or existing_receipt.get("compressed_bytes") != compressed_size
            or existing_receipt.get("uncompressed_bytes") != uncompressed_size
        ):
            raise RuntimeError("existing receipt does not bind the requested remote ZIP entry")
    elif args.receipt.exists():
        raise RuntimeError("receipt already exists; use --reuse-receipt only for verified rehydration")
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
    if method not in (0, 8):
        raise RuntimeError(f"unsupported ZIP compression method: {method}")
    data_start = offset + 30 + name_length + extra_length
    data_end = data_start + compressed_size - 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    if partial.exists():
        partial.unlink()
    transport: dict[str, object] = {"mode": "single_stream", "range_workers": 1}
    try:
        if args.range_workers == 1:
            decompressor = zlib.decompressobj(wbits=-15) if method == 8 else None
            crc = 0
            sha256 = hashlib.sha256()
            compressed_read = 0
            uncompressed_written = 0
            next_progress = PROGRESS_BYTES
            with partial.open("xb") as output:
                retry_count = 0
                while compressed_read < compressed_size:
                    request_start = data_start + compressed_read
                    request = urllib.request.Request(
                        inventory["url"],
                        headers={
                            "Range": f"bytes={request_start}-{data_end}",
                            "User-Agent": "BlindAssist-USTRF-research/1.0",
                        },
                    )
                    try:
                        with urllib.request.urlopen(
                            request,
                            timeout=args.request_timeout_seconds,
                        ) as response:
                            status = getattr(response, "status", None)
                            content_range = response.headers.get("Content-Range", "")
                            if status != 206 or not content_range.startswith(f"bytes {request_start}-{data_end}/"):
                                raise RuntimeError(
                                    f"server did not honor entry byte range: HTTP {status}, {content_range!r}"
                                )
                            while compressed_read < compressed_size:
                                chunk = response.read(min(CHUNK_BYTES, compressed_size - compressed_read))
                                if not chunk:
                                    raise EOFError("entry range ended before declared compressed size")
                                compressed_read += len(chunk)
                                decoded = decompressor.decompress(chunk) if decompressor else chunk
                                if decoded:
                                    output.write(decoded)
                                    uncompressed_written += len(decoded)
                                    crc = binascii.crc32(decoded, crc)
                                    sha256.update(decoded)
                                    if uncompressed_written > uncompressed_size:
                                        raise RuntimeError("decoded more bytes than declared")
                                if compressed_read >= next_progress:
                                    print(
                                        json.dumps(
                                            {
                                                "compressed_bytes": compressed_read,
                                                "uncompressed_bytes": uncompressed_written,
                                            }
                                        ),
                                        flush=True,
                                    )
                                    next_progress += PROGRESS_BYTES
                        retry_count = 0
                    except (urllib.error.URLError, TimeoutError, EOFError, OSError):
                        retry_count += 1
                        if retry_count > 8:
                            raise
                        time.sleep(min(2 ** (retry_count - 1), 30))
                if decompressor:
                    decoded = decompressor.flush()
                    if decoded:
                        output.write(decoded)
                        uncompressed_written += len(decoded)
                        crc = binascii.crc32(decoded, crc)
                        sha256.update(decoded)
                    if not decompressor.eof:
                        raise RuntimeError("deflate stream ended before end-of-stream marker")
            crc32 = f"{crc & 0xFFFFFFFF:08x}"
            output_sha256 = sha256.hexdigest()
        else:
            inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
            entry_key = hashlib.sha256(args.entry.encode("utf-8")).hexdigest()[:24]
            cache_dir = args.compressed_cache_root / inventory_sha256[:16] / entry_key
            ranges = partition_range(data_start, compressed_size, args.range_parts)
            progress_lock = threading.Lock()
            compressed_read = 0
            next_progress = PROGRESS_BYTES

            def report_progress(delta: int) -> None:
                nonlocal compressed_read, next_progress
                with progress_lock:
                    compressed_read += delta
                    if compressed_read >= next_progress:
                        print(
                            json.dumps(
                                {
                                    "transport": "parallel_range_prefetch",
                                    "compressed_bytes": compressed_read,
                                    "compressed_total": compressed_size,
                                }
                            ),
                            flush=True,
                        )
                        while compressed_read >= next_progress:
                            next_progress += PROGRESS_BYTES

            part_specs = [
                {
                    "index": index,
                    "start": start,
                    "end": end,
                    "path": cache_dir / f"part-{index:04d}.bin",
                }
                for index, (start, end) in enumerate(ranges)
            ]
            for spec in part_specs:
                path = spec["path"]
                if isinstance(path, Path) and path.exists():
                    existing_size = path.stat().st_size
                    expected_size = int(spec["end"]) - int(spec["start"]) + 1
                    if existing_size <= expected_size:
                        compressed_read += existing_size
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.range_workers) as executor:
                futures = [
                    executor.submit(
                        download_range_part,
                        url=inventory["url"],
                        start=int(spec["start"]),
                        end=int(spec["end"]),
                        output=spec["path"],
                        progress=report_progress,
                        request_timeout_seconds=args.request_timeout_seconds,
                    )
                    for spec in part_specs
                ]
                parts = [future.result() for future in futures]
            parts.sort(key=lambda part: int(part["range"][0]))
            compressed_read = sum(int(part["bytes"]) for part in parts)
            uncompressed_written, crc32, output_sha256 = decode_cached_parts(
                parts=parts,
                method=method,
                output=partial,
                expected_uncompressed_size=uncompressed_size,
            )
            transport = {
                "mode": "parallel_range_prefetch",
                "range_workers": args.range_workers,
                "range_parts": args.range_parts,
                "request_timeout_seconds": args.request_timeout_seconds,
                "compressed_cache_root": args.compressed_cache_root.as_posix(),
                "transport_config_path": (
                    transport_config_path.as_posix() if transport_config_sha256 is not None else None
                ),
                "transport_config_sha256": transport_config_sha256,
                "parts": [
                    {
                        "range": part["range"],
                        "bytes": part["bytes"],
                        "sha256": part["sha256"],
                        "reused": part["reused"],
                    }
                    for part in parts
                ],
            }
        if compressed_read != compressed_size:
            raise RuntimeError(f"compressed size mismatch: expected {compressed_size}, got {compressed_read}")
        if uncompressed_written != uncompressed_size:
            raise RuntimeError(f"uncompressed size mismatch: expected {uncompressed_size}, got {uncompressed_written}")
        if crc32 != entry["crc32"]:
            raise RuntimeError(f"CRC32 mismatch: expected {entry['crc32']}, got {crc32}")
        partial.replace(args.output)
        receipt = {
            "schema": "blindassist_streamed_remote_zip_entry_receipt_r1",
            "inventory_path": args.inventory.as_posix(),
            "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            "url": inventory["url"],
            "entry": args.entry,
            "compressed_range": [data_start, data_end],
            "compressed_bytes": compressed_read,
            "uncompressed_bytes": uncompressed_written,
            "zip_crc32": crc32,
            "output_path": args.output.as_posix(),
            "output_sha256": output_sha256,
            "transport": transport,
        }
        if existing_receipt is not None:
            for key in (
                "inventory_sha256",
                "url",
                "entry",
                "compressed_range",
                "compressed_bytes",
                "uncompressed_bytes",
                "zip_crc32",
                "output_sha256",
            ):
                if existing_receipt.get(key) != receipt.get(key):
                    args.output.unlink(missing_ok=True)
                    raise RuntimeError(f"rehydrated entry differs from existing receipt: {key}")
            print(json.dumps({"status": "REHYDRATED_FROM_BOUND_RECEIPT", **receipt}))
        else:
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(receipt))
        if args.range_workers > 1:
            for part in parts:
                path = part["path"]
                if isinstance(path, Path):
                    path.unlink(missing_ok=True)
            for directory in (cache_dir, cache_dir.parent):
                try:
                    directory.rmdir()
                except OSError:
                    pass
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
