#!/usr/bin/env python3
"""Build the locked-parent Spring member manifest from ZIP central directories only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .probe_materialization_capability import (
    SPRING_DATAFILE,
    ZIP_TAIL_BYTES,
    parse_zip_central_directory,
    request_range,
    zip_directory_location,
)


SPRING_MEMBER = re.compile(r"^spring/train/(?P<parent>\d{4})/")


def visibility(archive_name: str) -> tuple[bool, bool]:
    if archive_name == "train_frame_left.zip":
        return True, False
    return False, True


def parent_from_member(name: str) -> str | None:
    match = SPRING_MEMBER.match(name)
    return match.group("parent") if match else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(lock_path: Path, manifest_path: Path, receipt_path: Path) -> dict:
    if manifest_path.exists() or receipt_path.exists():
        raise ValueError("Spring range manifest output already exists; preflight outputs are create-once")
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    source = next(item for item in lock["sources"] if item["source_id"] == "SPRING_V2")
    locked_parents = {parent["parent_id"] for parent in source["parents"]}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".part")
    if temporary.exists():
        raise ValueError("stale Spring range manifest partial exists")
    archive_receipts = []
    total_rows = 0
    total_range_bytes = 0
    with temporary.open("x", encoding="utf-8", newline="\n") as output:
        for archive_name, binding in source["archive_bindings"].items():
            expected_bytes = int(binding["bytes"])
            url = SPRING_DATAFILE.format(datafile_id=binding["datafile_id"])
            tail_start = max(0, expected_bytes - ZIP_TAIL_BYTES)
            tail = request_range(url, tail_start, expected_bytes - 1)
            directory_offset, directory_bytes, entries = zip_directory_location(
                tail.data,
                total_bytes=expected_bytes,
                tail_start=tail_start,
            )
            directory = request_range(url, directory_offset, directory_offset + directory_bytes - 1)
            members = parse_zip_central_directory(directory.data, expected_entries=entries)
            candidate_visible, truth_visible = visibility(archive_name)
            selected = []
            for member in members:
                parent_id = parent_from_member(member["name"])
                if parent_id not in locked_parents:
                    continue
                row = {
                    "schema": "blindassist.svrf_o0.materialized_member.v1",
                    "source_id": "SPRING_V2",
                    "parent_id": parent_id,
                    "source_archive": archive_name,
                    "datafile_id": binding["datafile_id"],
                    "archive_member_path": member["name"],
                    "compression_method": member["compression_method"],
                    "compressed_bytes": member["compressed_bytes"],
                    "uncompressed_bytes": member["uncompressed_bytes"],
                    "crc32": member["crc32"],
                    "local_header_offset": member["local_header_offset"],
                    "candidate_visible": candidate_visible,
                    "truth_visible": truth_visible,
                    "payload_materialized": False,
                    "semantic_content_inspected": False,
                }
                selected.append(row)
            selected.sort(key=lambda item: (item["parent_id"], item["archive_member_path"]))
            for row in selected:
                output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            total_rows += len(selected)
            range_bytes = len(tail.data) + len(directory.data)
            total_range_bytes += range_bytes
            archive_receipts.append(
                {
                    "archive": archive_name,
                    "datafile_id": binding["datafile_id"],
                    "archive_bytes": expected_bytes,
                    "central_directory_offset": directory_offset,
                    "central_directory_bytes": directory_bytes,
                    "all_member_count": entries,
                    "locked_parent_member_count": len(selected),
                    "range_bytes_received": range_bytes,
                    "locked_parent_counts": {
                        parent: sum(row["parent_id"] == parent for row in selected)
                        for parent in sorted(locked_parents)
                    },
                }
            )
    temporary.replace(manifest_path)
    result = {
        "schema": "blindassist.svrf_o0.spring_range_manifest_receipt.v1",
        "status": "SPRING_LOCKED_PARENT_RANGE_MANIFEST_COMPLETE",
        "source_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows": total_rows,
        "range_bytes_received": total_range_bytes,
        "archives": archive_receipts,
        "payload_materialized_count": 0,
        "media_tensor_or_label_decode_count": 0,
        "candidate_run_count": 0,
        "outcome_accessed": False,
        "completed_at_utc": utc_now(),
    }
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.lock, args.manifest, args.receipt)
    print(result["status"])


if __name__ == "__main__":
    main()
