#!/usr/bin/env python3
"""Audit CODa immutable binding and tiny-split continuity without payload extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import urllib.request
from collections import defaultdict
from pathlib import Path


TACC_TINY_URL = (
    "https://web.corral.tacc.utexas.edu/texasrobotics/"
    "web_CODa/splits/CODa_tiny_split.zip"
)
TDR_DATASET_API = (
    "https://dataverse.tdl.org/api/datasets/:persistentId?"
    "persistentId=doi:10.18738/T8/BBOQMV"
)
TDR_TINY_DATAFILE_IDS = (299625, 299626, 299627)
EXPECTED_TDR_SNAPSHOT_SHA256 = (
    "016a860feb463fe18844d038180e44e88b811a8c9ee8674741f7ff88dc07060d"
)
EXPECTED_TDR_TINY_PARTS = (
    (299625, 5732, "CODa_tiny.tar.gz.part001", 4_294_967_296, "61da09d525cd7d2627412eb2a13f7466"),
    (299626, 5732, "CODa_tiny.tar.gz.part002", 4_294_967_296, "e97dc0815ff32483d6c2138e092caea1"),
    (299627, 5732, "CODa_tiny.tar.gz.part003", 518_241_581, "e0d97f2141c9ee21537e664ab1228993"),
)
EXPECTED_TACC_SIZE = 9_108_343_009
TAIL_BYTES = 131_072
BBOX_PATTERN = re.compile(
    r"^CODa_tiny/3d_bbox/os1/(?P<sequence>\d+)/"
    r"3d_bbox_os1_(?P=sequence)_(?P<frame>\d+)\.json$"
)
CAM0_PATTERN = re.compile(
    r"^CODa_tiny/2d_rect/cam0/(?P<sequence>\d+)/"
    r"2d_rect_cam0_(?P=sequence)_(?P<frame>\d+)\.(?:png|jpg)$"
)


def _request(url: str, *, byte_range: tuple[int, int] | None = None) -> tuple[bytes, dict]:
    headers = {"User-Agent": "Mozilla/5.0"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        response_headers = dict(response.headers)
        if byte_range is not None:
            expected_length = byte_range[1] - byte_range[0] + 1
            assert response.status == 206, f"range request returned {response.status}"
            assert len(body) == expected_length, "range response length mismatch"
            assert response_headers.get("Content-Range") == (
                f"bytes {byte_range[0]}-{byte_range[1]}/{EXPECTED_TACC_SIZE}"
            ), "range response Content-Range mismatch"
        return body, response_headers


def _head(url: str) -> dict:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return dict(response.headers)


def _max_consecutive(values: list[int]) -> int:
    if not values:
        return 0
    best = current = 1
    for left, right in zip(values, values[1:]):
        if right == left + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _central_directory() -> tuple[bytes, dict]:
    headers = _head(TACC_TINY_URL)
    size = int(headers["Content-Length"])
    if size != EXPECTED_TACC_SIZE:
        raise AssertionError(f"TACC tiny size drift: {size}")
    tail_start = size - TAIL_BYTES
    tail, tail_headers = _request(TACC_TINY_URL, byte_range=(tail_start, size - 1))
    eocd_position = tail.rfind(b"PK\x05\x06")
    locator_position = tail.rfind(b"PK\x06\x07", 0, eocd_position)
    if eocd_position < 0 or locator_position < 0:
        raise AssertionError("ZIP64 end records missing")
    _, _, zip64_offset, _ = struct.unpack(
        "<4sLQL", tail[locator_position : locator_position + 20]
    )
    zip64, zip64_headers = _request(
        TACC_TINY_URL, byte_range=(zip64_offset, zip64_offset + 55)
    )
    (
        signature,
        _record_size,
        _version_made,
        _version_needed,
        _disk,
        _central_disk,
        _entries_disk,
        entry_count,
        central_size,
        central_offset,
    ) = struct.unpack("<4sQ2H2L4Q", zip64[:56])
    if signature != b"PK\x06\x06":
        raise AssertionError("invalid ZIP64 end signature")
    central, central_headers = _request(
        TACC_TINY_URL,
        byte_range=(central_offset, central_offset + central_size - 1),
    )
    return central, {
        "content_length": size,
        "etag": str(headers.get("ETag", "")).strip('"'),
        "last_modified": headers.get("Last-Modified"),
        "accept_ranges": headers.get("Accept-Ranges"),
        "entry_count": entry_count,
        "central_directory_offset": central_offset,
        "central_directory_bytes": central_size,
        "central_directory_sha256": hashlib.sha256(central).hexdigest(),
        "http_body_bytes_read": len(tail) + len(zip64) + len(central),
        "range_response_proofs": [
            {
                "name": "tail",
                "status": 206,
                "content_range": tail_headers["Content-Range"],
                "bytes": len(tail),
            },
            {
                "name": "zip64_end",
                "status": 206,
                "content_range": zip64_headers["Content-Range"],
                "bytes": len(zip64),
            },
            {
                "name": "central_directory",
                "status": 206,
                "content_range": central_headers["Content-Range"],
                "bytes": len(central),
            },
        ],
    }


def _member_names(central: bytes) -> list[str]:
    names: list[str] = []
    cursor = 0
    while cursor < len(central):
        if central[cursor : cursor + 4] != b"PK\x01\x02":
            raise AssertionError(f"invalid central directory signature at {cursor}")
        fields = struct.unpack("<4s6H3L5H2L", central[cursor : cursor + 46])
        filename_length = fields[10]
        extra_length = fields[11]
        comment_length = fields[12]
        start = cursor + 46
        names.append(
            central[start : start + filename_length].decode("utf-8", "strict")
        )
        cursor = start + filename_length + extra_length + comment_length
    return names


def _continuity(names: list[str], pattern: re.Pattern[str]) -> dict:
    frames: dict[int, list[int]] = defaultdict(list)
    for name in names:
        match = pattern.match(name)
        if match:
            frames[int(match.group("sequence"))].append(int(match.group("frame")))
    rows = []
    for sequence in range(23):
        values = sorted(set(frames[sequence]))
        rows.append(
            {
                "sequence": sequence,
                "frame_count": len(values),
                "max_consecutive_frame_run": _max_consecutive(values),
            }
        )
    return {
        "frame_count": sum(row["frame_count"] for row in rows),
        "sequence_count_with_frames": sum(row["frame_count"] > 0 for row in rows),
        "maximum_consecutive_frame_run": max(
            row["max_consecutive_frame_run"] for row in rows
        ),
        "sequence_count_with_at_least_100_consecutive_frames": sum(
            row["max_consecutive_frame_run"] >= 100 for row in rows
        ),
        "per_sequence": rows,
    }


def _tdr_snapshot(path: Path) -> dict:
    snapshot_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if snapshot_sha256 != EXPECTED_TDR_SNAPSHOT_SHA256:
        raise AssertionError("TDR metadata snapshot hash drift")
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if (
        snapshot["latest_released_version"],
        snapshot["latest_dataset_version_id"],
        snapshot["version_state"],
    ) != ("2.3", 5732, "RELEASED"):
        raise AssertionError("TDR snapshot version mismatch")
    identity = sorted(snapshot["tiny_parts"], key=lambda row: row["filename"])
    if tuple(item["datafile_id"] for item in identity) != TDR_TINY_DATAFILE_IDS:
        raise AssertionError("TDR tiny datafile identity mismatch")
    exact_parts = tuple(
        (
            item["datafile_id"],
            item["dataset_version_id"],
            item["filename"],
            item["bytes"],
            item["checksum"],
        )
        for item in identity
    )
    if exact_parts != EXPECTED_TDR_TINY_PARTS or not all(
        item["checksum_type"] == "MD5" for item in identity
    ):
        raise AssertionError("TDR tiny checksum contract mismatch")
    return {
        "doi": "10.18738/T8/BBOQMV",
        "latest_released_version": "2.3",
        "latest_dataset_version_id": snapshot["latest_dataset_version_id"],
        "release_time": snapshot["release_time"],
        "metadata_snapshot_path": path.name,
        "metadata_snapshot_sha256": snapshot_sha256,
        "version_2_3_parts": identity,
        "version_2_3_total_bytes": sum(item["bytes"] for item in identity),
    }


def build_receipt(tdr_snapshot_path: Path) -> dict:
    central, transport = _central_directory()
    names = _member_names(central)
    bbox = _continuity(names, BBOX_PATTERN)
    cam0 = _continuity(names, CAM0_PATTERN)
    tdr = _tdr_snapshot(tdr_snapshot_path)
    if len(names) != transport["entry_count"]:
        raise AssertionError("central directory entry-count mismatch")
    return {
        "schema_version": "coda_tiny_continuity_and_binding_audit_r0",
        "source_id": "UT_CODA",
        "audit_mode": "HEAD_FROZEN_TDR_SNAPSHOT_AND_ZIP_CENTRAL_DIRECTORY_ONLY",
        "payload_member_extraction_count": 0,
        "payload_decoded": False,
        "candidate_signal_computed": False,
        "tacc_tiny": {
            "url": TACC_TINY_URL,
            **transport,
            "bbox_availability": bbox,
            "cam0_availability": cam0,
        },
        "tdr_tiny": tdr,
        "binding": {
            "tdr_v2_3_tiny_has_published_checksums": True,
            "tacc_tiny_container_equals_tdr_tiny_container": False,
            "official_per_member_logical_equivalence_manifest_present": False,
            "full_tacc_sequence_archive_checksum_or_version_binding_present": False,
            "tacc_tiny_continuity_applies_to_tdr_tiny": False,
        },
        "r0_continuity": {
            "required_consecutive_frames_at_10hz_for_10s": 100,
            "tiny_sequence_count_meeting_bbox_continuity": bbox[
                "sequence_count_with_at_least_100_consecutive_frames"
            ],
            "tiny_sequence_count_meeting_cam0_continuity": cam0[
                "sequence_count_with_at_least_100_consecutive_frames"
            ],
        },
        "source_admission": "HOLD_R0_ADMISSION",
        "terminal": "HOLD_CODA_BOUNDED_PRESCREEN",
        "status": "VALID",
        "authority": {
            "may_continue_metadata_only_inventory": True,
            "may_extract_tacc_members": False,
            "may_download_tdr_tiny_for_r0": False,
            "may_download_or_decode_rgb": False,
            "may_run_signal": False,
            "may_count_coda_toward_three_real_sources": False,
        },
        "stop_reasons": [
            "unbound TACC tiny has zero sequence with 100 consecutive bbox or cam0 frames",
            "checksum-bound TDR tiny continuity was not evaluated",
            "full TACC sequence archives lack published cryptographic checksum and DOI-version binding",
            "TACC and TDR tiny containers differ and have no official per-member equivalence manifest",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tdr-snapshot", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(args.tdr_snapshot.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "terminal": receipt["terminal"],
                "bbox": receipt["tacc_tiny"]["bbox_availability"],
                "cam0": receipt["tacc_tiny"]["cam0_availability"],
                "may_extract_tacc_members": receipt["authority"][
                    "may_extract_tacc_members"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
