#!/usr/bin/env python3
"""Probe archive indexing and selective extraction without opening SVRF outcomes."""

from __future__ import annotations

import argparse
import binascii
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib


RANGE_PATTERN = re.compile(r"bytes (?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+)")
A2D2_BUCKET = "https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
SPRING_DATAFILE = "https://darus.uni-stuttgart.de/api/access/datafile/{datafile_id}"
USER_AGENT = "BlindAssist-SVRF-O0-Materialization-Preflight/1"
ZIP_TAIL_BYTES = 131_072
MAX_CRC_SAMPLE_BYTES = 4 * 1024 * 1024
STORAGE_SAFETY_RESERVE_BYTES = 64 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class RangeResponse:
    data: bytes
    start: int
    end: int
    total: int
    etag: str | None
    last_modified: str | None
    content_type: str | None


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_range(url: str, start: int, end: int, *, timeout: int = 60) -> RangeResponse:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _validated_range_response(
                url,
                start,
                end,
                response.read(),
                dict(response.headers.items()),
                response.status,
            )
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return _request_range_curl(url, start, end, timeout=timeout)


def _validated_range_response(
    url: str,
    start: int,
    end: int,
    data: bytes,
    headers: dict[str, str],
    status: int,
) -> RangeResponse:
    normalized = {key.lower(): value.strip() for key, value in headers.items()}
    content_range = normalized.get("content-range", "")
    match = RANGE_PATTERN.fullmatch(content_range)
    if status != 206 or match is None:
        raise ValueError(f"range unsupported for {url}: status={status} range={content_range!r}")
    actual_start = int(match.group("start"))
    actual_end = int(match.group("end"))
    total = int(match.group("total"))
    if actual_start != start or actual_end != end or len(data) != end - start + 1:
        raise ValueError(f"range identity drift for {url}")
    return RangeResponse(
        data=data,
        start=actual_start,
        end=actual_end,
        total=total,
        etag=normalized.get("etag"),
        last_modified=normalized.get("last-modified"),
        content_type=normalized.get("content-type"),
    )


def _request_range_curl(url: str, start: int, end: int, *, timeout: int) -> RangeResponse:
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if executable is None:
        raise ValueError("curl fallback is unavailable after urllib transport failure")
    with tempfile.TemporaryDirectory(prefix="svrf-o0-range-") as directory:
        header_path = Path(directory) / "headers.txt"
        body_path = Path(directory) / "body.bin"
        command = [
            executable,
            "-sS",
            "-L",
            "--range",
            f"{start}-{end}",
            "--max-time",
            str(timeout),
            "--user-agent",
            USER_AGENT,
            "--dump-header",
            str(header_path),
            "--output",
            str(body_path),
            url,
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise ValueError(f"curl range fallback failed: {completed.stderr.strip()}")
        blocks = re.split(r"\r?\n\r?\n", header_path.read_text(encoding="iso-8859-1"))
        final = next((block for block in reversed(blocks) if re.search(r"(?im)^content-range:", block)), None)
        if final is None:
            raise ValueError("curl range fallback returned no final Content-Range header")
        lines = final.splitlines()
        status_match = re.match(r"HTTP/\S+\s+(\d+)", lines[0])
        if status_match is None:
            raise ValueError("curl range fallback returned an invalid status line")
        headers = {
            key.strip(): value.strip()
            for line in lines[1:]
            if ":" in line
            for key, value in [line.split(":", 1)]
        }
        return _validated_range_response(
            url,
            start,
            end,
            body_path.read_bytes(),
            headers,
            int(status_match.group(1)),
        )


def parse_tar_header(block: bytes) -> dict[str, Any] | None:
    if len(block) != 512:
        raise ValueError("TAR header must be exactly 512 bytes")
    if block == bytes(512):
        return None
    stored_checksum = int(block[148:156].split(b"\0", 1)[0].strip() or b"0", 8)
    checksum_block = block[:148] + b" " * 8 + block[156:]
    if sum(checksum_block) != stored_checksum:
        raise ValueError("TAR header checksum mismatch")
    name = block[:100].split(b"\0", 1)[0].decode("utf-8", "replace")
    prefix = block[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
    if prefix:
        name = f"{prefix}/{name}"
    size = int(block[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
    return {"name": name, "size": size, "type": block[156:157].decode("ascii", "replace")}


def probe_tar_prefix(url: str, expected_bytes: int, *, member_limit: int = 4) -> dict[str, Any]:
    offset = 0
    members = []
    first_response = None
    pending_long_name = None
    for _ in range(member_limit):
        response = request_range(url, offset, offset + 511)
        first_response = first_response or response
        if response.total != expected_bytes:
            raise ValueError(f"TAR byte-size drift for {url}")
        header = parse_tar_header(response.data)
        if header is None:
            break
        payload_offset = offset + 512
        if header["type"] == "L" and int(header["size"]) <= 4096:
            long_name = request_range(
                url,
                payload_offset,
                payload_offset + int(header["size"]) - 1,
            ).data
            pending_long_name = long_name.split(b"\0", 1)[0].decode("utf-8", "replace")
            header["long_name_target"] = pending_long_name
        elif pending_long_name is not None:
            header["name"] = pending_long_name
            pending_long_name = None
        members.append({"header_offset": offset, **header})
        offset += 512 + ((int(header["size"]) + 511) // 512) * 512
    if first_response is None:
        raise ValueError("TAR probe did not issue a request")
    return {
        "range_supported": True,
        "bytes": expected_bytes,
        "etag": first_response.etag,
        "last_modified": first_response.last_modified,
        "content_type": first_response.content_type,
        "prefix_members": members,
        "central_member_index": False,
    }


def zip_directory_location(tail: bytes, *, total_bytes: int, tail_start: int) -> tuple[int, int, int]:
    eocd_signature = bytes.fromhex("504b0506")
    eocd_offset = tail.rfind(eocd_signature)
    if eocd_offset < 0:
        raise ValueError("ZIP EOCD not found in bounded tail")
    _, _, _, disk_entries, total_entries, directory_bytes, directory_offset, _ = struct.unpack_from(
        "<4s4H2LH", tail, eocd_offset
    )
    if disk_entries != total_entries:
        raise ValueError("multi-disk ZIP is unsupported")
    if total_entries != 0xFFFF and directory_offset != 0xFFFFFFFF:
        return directory_offset, directory_bytes, total_entries
    locator_signature = bytes.fromhex("504b0607")
    locator_offset = tail.rfind(locator_signature, 0, eocd_offset)
    if locator_offset < 0:
        raise ValueError("ZIP64 locator not found")
    _, disk_number, zip64_offset, disk_count = struct.unpack_from("<4sLQL", tail, locator_offset)
    if disk_number != 0 or disk_count != 1:
        raise ValueError("multi-disk ZIP64 is unsupported")
    relative = zip64_offset - tail_start
    if not 0 <= relative <= len(tail) - 56:
        raise ValueError("ZIP64 EOCD lies outside bounded tail")
    values = struct.unpack_from("<4sQ2H2L4Q", tail, relative)
    if values[0] != bytes.fromhex("504b0606") or values[6] != values[7]:
        raise ValueError("ZIP64 directory identity is invalid")
    if zip64_offset >= total_bytes:
        raise ValueError("ZIP64 EOCD offset exceeds object")
    return int(values[9]), int(values[8]), int(values[7])


def _zip64_values(extra: bytes) -> list[int]:
    offset = 0
    while offset + 4 <= len(extra):
        field_id, size = struct.unpack_from("<HH", extra, offset)
        payload = extra[offset + 4 : offset + 4 + size]
        if field_id == 0x0001:
            if len(payload) % 8:
                raise ValueError("malformed ZIP64 extra field")
            return list(struct.unpack("<" + "Q" * (len(payload) // 8), payload))
        offset += 4 + size
    return []


def parse_zip_central_directory(data: bytes, *, expected_entries: int) -> list[dict[str, Any]]:
    members = []
    offset = 0
    signature = bytes.fromhex("504b0102")
    while offset < len(data):
        if data[offset : offset + 4] != signature:
            raise ValueError(f"ZIP central-directory signature mismatch at {offset}")
        values = struct.unpack_from("<4s6H3L5H2L", data, offset)
        method = values[4]
        crc32 = values[7]
        compressed = values[8]
        uncompressed = values[9]
        name_length, extra_length, comment_length = values[10:13]
        local_offset = values[16]
        name_start = offset + 46
        name = data[name_start : name_start + name_length].decode("utf-8", "replace")
        extra = data[name_start + name_length : name_start + name_length + extra_length]
        zip64 = iter(_zip64_values(extra))
        if uncompressed == 0xFFFFFFFF:
            uncompressed = next(zip64)
        if compressed == 0xFFFFFFFF:
            compressed = next(zip64)
        if local_offset == 0xFFFFFFFF:
            local_offset = next(zip64)
        members.append(
            {
                "name": name,
                "compression_method": method,
                "crc32": f"{crc32:08x}",
                "compressed_bytes": int(compressed),
                "uncompressed_bytes": int(uncompressed),
                "local_header_offset": int(local_offset),
            }
        )
        offset = name_start + name_length + extra_length + comment_length
    if len(members) != expected_entries:
        raise ValueError(f"ZIP member-count drift: {len(members)} != {expected_entries}")
    return members


def extract_zip_member(url: str, member: dict[str, Any], expected_total: int) -> dict[str, Any]:
    offset = int(member["local_header_offset"])
    header = request_range(url, offset, offset + 29)
    if header.total != expected_total:
        raise ValueError("ZIP sample object-size drift")
    values = struct.unpack("<4s5H3L2H", header.data)
    if values[0] != bytes.fromhex("504b0304"):
        raise ValueError("ZIP local-header signature mismatch")
    name_length, extra_length = values[-2:]
    payload_start = offset + 30 + name_length + extra_length
    compressed_bytes = int(member["compressed_bytes"])
    if compressed_bytes > MAX_CRC_SAMPLE_BYTES:
        return {"member": member["name"], "status": "SKIPPED_OVER_4_MIB_BOUND"}
    payload = request_range(url, payload_start, payload_start + compressed_bytes - 1).data
    method = int(member["compression_method"])
    if method == 0:
        decoded = payload
    elif method == 8:
        decoded = zlib.decompress(payload, -15)
    else:
        return {"member": member["name"], "status": f"UNSUPPORTED_COMPRESSION_{method}"}
    crc = f"{binascii.crc32(decoded) & 0xFFFFFFFF:08x}"
    if crc != member["crc32"] or len(decoded) != member["uncompressed_bytes"]:
        raise ValueError("ZIP selective member CRC/size mismatch")
    return {
        "member": member["name"],
        "status": "CRC_VALID_WITHOUT_CONTENT_DECODE",
        "compressed_bytes": compressed_bytes,
        "uncompressed_bytes": len(decoded),
        "sha256": hashlib.sha256(decoded).hexdigest(),
    }


def probe_zip(url: str, expected_bytes: int, locked_parents: set[str]) -> dict[str, Any]:
    tail_start = max(0, expected_bytes - ZIP_TAIL_BYTES)
    tail_response = request_range(url, tail_start, expected_bytes - 1)
    if tail_response.total != expected_bytes:
        raise ValueError(f"ZIP byte-size drift for {url}")
    directory_offset, directory_bytes, entries = zip_directory_location(
        tail_response.data,
        total_bytes=expected_bytes,
        tail_start=tail_start,
    )
    directory = request_range(url, directory_offset, directory_offset + directory_bytes - 1)
    members = parse_zip_central_directory(directory.data, expected_entries=entries)
    selected = [
        member
        for member in members
        if any(f"/{parent}/" in f"/{member['name']}" for parent in locked_parents)
    ]
    if not selected:
        raise ValueError("ZIP has no members for locked Spring parents")
    sample = min((member for member in selected if member["compressed_bytes"] > 0), key=lambda item: item["compressed_bytes"])
    return {
        "range_supported": True,
        "bytes": expected_bytes,
        "etag": tail_response.etag,
        "last_modified": tail_response.last_modified,
        "content_type": tail_response.content_type,
        "central_member_index": True,
        "central_directory_bytes": directory_bytes,
        "member_count": entries,
        "locked_parent_member_count": len(selected),
        "locked_parent_compressed_bytes": sum(int(member["compressed_bytes"]) for member in selected),
        "locked_parent_uncompressed_bytes": sum(int(member["uncompressed_bytes"]) for member in selected),
        "locked_parent_counts": {
            parent: sum(f"/{parent}/" in f"/{member['name']}" for member in selected)
            for parent in sorted(locked_parents)
        },
        "selective_crc_sample": extract_zip_member(url, sample, expected_bytes),
    }


def a2d2_bucket_inventory() -> dict[str, Any]:
    url = A2D2_BUCKET + "?" + urllib.parse.urlencode({"list-type": "2", "max-keys": 1000})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ET.fromstring(response.read())
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [item.text or "" for item in root.findall("s3:Contents/s3:Key", namespace)]
    individual = [
        key
        for key in keys
        if re.search(r"\.(png|jpg|jpeg|npz)$", key, re.IGNORECASE)
    ]
    return {
        "object_count": len(keys),
        "listing_truncated": root.findtext("s3:IsTruncated", default="", namespaces=namespace).lower() == "true",
        "individual_sensor_object_count": len(individual),
        "archive_object_count": sum(key.endswith(".tar") for key in keys),
        "inventory_sha256": hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest(),
    }


def run(lock_path: Path, artifact_root: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    sources = {source["source_id"]: source for source in lock["sources"]}
    a2d2 = sources["A2D2_SENSOR_FUSION"]
    spring = sources["SPRING_V2"]
    a2d2_archives = []
    for parent in a2d2["parents"]:
        for modality, archive in parent["archives"].items():
            name = archive["name"]
            a2d2_archives.append(
                {
                    "parent_id": parent["parent_id"],
                    "modality": modality,
                    "archive": name,
                    **probe_tar_prefix(A2D2_BUCKET + name, int(archive["bytes"])),
                }
            )
    locked_parents = {parent["parent_id"] for parent in spring["parents"]}
    spring_archives = []
    for name, archive in spring["archive_bindings"].items():
        url = SPRING_DATAFILE.format(datafile_id=archive["datafile_id"])
        spring_archives.append(
            {
                "archive": name,
                "datafile_id": archive["datafile_id"],
                **probe_zip(url, int(archive["bytes"]), locked_parents),
            }
        )
    a2d2_bytes = sum(item["bytes"] for item in a2d2_archives)
    spring_bytes = sum(item["bytes"] for item in spring_archives)
    logical_root = artifact_root.absolute()
    physical_root = artifact_root.resolve()
    free_bytes = shutil.disk_usage(physical_root).free
    all_bound_bytes = a2d2_bytes + spring_bytes
    inventory = a2d2_bucket_inventory()
    return {
        "schema": "blindassist.svrf_o0.archive_access_capability.v1",
        "status": "PARTIAL_PASS_SPRING_RANDOM_ACCESS_A2D2_STREAM_PLAN_REQUIRED",
        "source_lock_sha256": file_hash(lock_path),
        "outcome_accessed": False,
        "candidate_run_count": 0,
        "archive_member_crc_validation_count": sum(
            item["selective_crc_sample"]["status"] == "CRC_VALID_WITHOUT_CONTENT_DECODE"
            for item in spring_archives
        ),
        "media_tensor_or_label_decode_count": 0,
        "receipt_semantics": "archive members may be decompressed only to verify exact size and CRC; no image, flow, disparity, point-cloud, pose, label or candidate-output semantics are parsed",
        "materialization_storage": {
            "logical_root": str(logical_root),
            "physical_root": str(physical_root),
            "physical_drive": physical_root.drive,
            "free_bytes_at_probe": free_bytes,
            "safety_reserve_bytes": STORAGE_SAFETY_RESERVE_BYTES,
            "all_bound_archives_bytes": all_bound_bytes,
            "all_bound_archives_fit_raw": free_bytes >= all_bound_bytes,
            "all_bound_archives_fit_with_safety_reserve": free_bytes >= all_bound_bytes + STORAGE_SAFETY_RESERVE_BYTES,
            "spring_archives_fit_with_safety_reserve": free_bytes >= spring_bytes + STORAGE_SAFETY_RESERVE_BYTES,
            "largest_single_archive_fits_with_safety_reserve": free_bytes
            >= max(item["bytes"] for item in a2d2_archives + spring_archives) + STORAGE_SAFETY_RESERVE_BYTES,
        },
        "a2d2": {
            "archive_bytes": a2d2_bytes,
            "bucket_inventory": inventory,
            "archives": a2d2_archives,
            "random_member_index_supported": False,
            "finding": "official bucket exposes TAR objects but no individual PNG/NPZ sensor objects and TAR has no central directory; complete timeline enumeration requires a bounded full-stream or remote sparse-header indexing plan",
        },
        "spring": {
            "archive_bytes": spring_bytes,
            "archives": spring_archives,
            "random_member_index_supported": True,
            "finding": "ZIP64 central directories and CRC-valid bounded member extraction are available through HTTP Range",
        },
        "bulk_download_authorized": False,
        "candidate_run_authorized": False,
        "outcome_access_authorized": False,
        "unique_successor": "SVRF_O0_A2D2_ONE_PASS_STREAM_INDEXER_AND_SPRING_RANGE_MANIFEST_PREFLIGHT",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts.local"))
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.lock, args.artifact_root)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["status"])


if __name__ == "__main__":
    main()
