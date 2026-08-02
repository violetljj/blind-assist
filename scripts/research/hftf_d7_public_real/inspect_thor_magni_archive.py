#!/usr/bin/env python3
"""Inspect THOR-MAGNI's public ZIP central directory with bounded HTTP ranges.

This command never downloads the 22+ GB archive payload.  It reads only the
Zenodo record JSON and ZIP central-directory metadata, then records member
names/sizes/checksums as source-intake evidence.  It does not pair video with
motion-capture rows or create D7 event truth.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_jsonl, sha256_file, stable_id, utc_now, write_json, write_jsonl


RECORD_API = "https://zenodo.org/api/records/{record_id}"
USER_AGENT = "blindassist-hftf-d7/1.0"


class RangeReader(io.RawIOBase):
    """Seekable read-only file that fetches only requested HTTP byte ranges."""

    def __init__(self, url: str, size: int, *, max_request_bytes: int, max_total_bytes: int) -> None:
        super().__init__()
        if size <= 0 or max_request_bytes <= 0 or max_total_bytes <= 0:
            raise ContractError("range-reader sizes must be positive")
        self.url = url
        self.size = size
        self.max_request_bytes = max_request_bytes
        self.max_total_bytes = max_total_bytes
        self.position = 0
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if target < 0 or target > self.size:
            raise ValueError(f"seek outside archive: {target}")
        self.position = target
        return target

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        remaining = self.size - self.position
        requested = remaining if size is None or size < 0 else min(size, remaining)
        if requested > self.max_request_bytes:
            raise ContractError(
                f"ZIP metadata range exceeds per-request cap: {requested} > {self.max_request_bytes}"
            )
        if self.bytes_fetched + requested > self.max_total_bytes:
            raise ContractError(
                f"ZIP metadata range budget exceeded: {self.bytes_fetched + requested} > {self.max_total_bytes}"
            )
        start = self.position
        end = start + requested - 1
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 206:
                    raise ContractError(f"Zenodo did not honor HTTP Range: status={response.status}")
                content_range = str(response.headers.get("Content-Range") or "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise ContractError(f"unexpected Content-Range: {content_range}")
                payload = response.read(requested)
        except OSError as exc:
            raise ContractError(f"THOR-MAGNI range request failed: {exc}") from exc
        if len(payload) != requested:
            raise ContractError(f"short HTTP range: {len(payload)} != {requested}")
        self.position += len(payload)
        self.bytes_fetched += len(payload)
        return payload


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load Zenodo metadata: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("Zenodo metadata is not an object")
    return payload


def _member_kind(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".mp4") or lower.endswith(".mkv") or lower.endswith(".avi"):
        return "VIDEO"
    if lower.endswith((".csv", ".tsv", ".json", ".jsonl")):
        return "TABULAR_OR_JSON"
    if lower.endswith((".bag", ".mcap", ".rosbag")):
        return "ROBOT_LOG"
    if lower.endswith((".pcd", ".ply", ".las", ".laz")):
        return "POINT_CLOUD"
    if lower.endswith((".zip", ".tar", ".gz", ".bz2")):
        return "ARCHIVE"
    return "OTHER"


def _select_archive(payload: dict[str, Any]) -> dict[str, Any]:
    files = payload.get("files")
    if not isinstance(files, list):
        raise ContractError("Zenodo record has no file list")
    candidates = [
        item for item in files
        if isinstance(item, dict) and str(item.get("key", "")).lower() == "thor_magni.zip"
    ]
    if len(candidates) != 1:
        raise ContractError(f"expected one THOR_MAGNI.zip file, found {len(candidates)}")
    archive = candidates[0]
    url = str(archive.get("links", {}).get("self") or "")
    size = int(archive.get("size") or 0)
    if not url or size <= 0:
        raise ContractError("THOR-MAGNI archive metadata lacks URL or size")
    return {"key": archive.get("key"), "url": url, "size": size, "checksum": archive.get("checksum")}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_request_bytes <= 0 or args.max_total_bytes <= 0:
        raise ContractError("range budgets must be positive")
    root = Path(args.output_root).resolve()
    receipt_path = root / "receipts" / f"thor_magni_archive_receipt_{args.run_id}.json"
    if receipt_path.exists():
        raise ContractError(f"receipt already exists; refusing overwrite: {receipt_path}")
    record_url = RECORD_API.format(record_id=args.record_id)
    record = _get_json(record_url)
    archive = _select_archive(record)
    reader = RangeReader(
        str(archive["url"]),
        int(archive["size"]),
        max_request_bytes=args.max_request_bytes,
        max_total_bytes=args.max_total_bytes,
    )
    try:
        with zipfile.ZipFile(reader) as archive_file:
            members = archive_file.infolist()
    except (OSError, zipfile.BadZipFile, ContractError) as exc:
        raise ContractError(f"bounded THOR-MAGNI central-directory inspection failed: {exc}") from exc

    manifest_rows = [
        {
            "schema": "hftf_d7_public_real_thor_magni_archive_member_v1",
            "member_id": stable_id("d7thor-magni-member", args.record_id, member.filename),
            "dataset_id": "THOR-MAGNI",
            "record_id": args.record_id,
            "member_name": member.filename,
            "member_kind": _member_kind(member.filename),
            "file_size": int(member.file_size),
            "compressed_size": int(member.compress_size),
            "compression_method": int(member.compress_type),
            "header_offset": int(member.header_offset),
            "crc32": f"{int(member.CRC):08x}",
            "source_revision": str(archive.get("checksum") or "UNKNOWN"),
            "license": "CC-BY-4.0 (Zenodo record metadata; verify member terms before event use)",
            "event_truth_authority": False,
        }
        for member in sorted(members, key=lambda item: item.filename)
    ]
    manifest_path = root / "manifests" / f"thor_magni_archive_manifest_{args.run_id}.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    kind_counts: dict[str, int] = {}
    kind_bytes: dict[str, int] = {}
    for row in manifest_rows:
        kind = str(row["member_kind"])
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        kind_bytes[kind] = kind_bytes.get(kind, 0) + int(row["file_size"])
    receipt = {
        "schema": "hftf_d7_public_real_thor_magni_archive_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "dataset_id": "THOR-MAGNI",
        "record_id": args.record_id,
        "record_url": f"https://zenodo.org/records/{args.record_id}",
        "archive_key": archive["key"],
        "archive_url": archive["url"],
        "archive_size": archive["size"],
        "archive_checksum": archive["checksum"],
        "license": "CC-BY-4.0",
        "access_status": "PUBLIC_ARCHIVE_METADATA_RANGE_INSPECTED",
        "status": "PUBLIC_ARCHIVE_METADATA_RANGE_INSPECTED",
        "archive_content_materialized": False,
        "archive_member_count": len(manifest_rows),
        "member_kind_counts": kind_counts,
        "member_kind_bytes": kind_bytes,
        "range_bytes_fetched": reader.bytes_fetched,
        "range_max_request_bytes": args.max_request_bytes,
        "range_max_total_bytes": args.max_total_bytes,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "event_truth_authority": False,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
        "notes": [
            "Only Zenodo record metadata and ZIP central-directory metadata were read.",
            "No archive member content was downloaded or paired with RGB/pose timestamps.",
            "Archive member names and sizes are source-intake evidence, not D7 event truth.",
        ],
    }
    write_json(receipt_path, receipt)
    source_receipt_path = root / "receipts" / "source_receipts.jsonl"
    if source_receipt_path.is_file():
        rows = load_jsonl(source_receipt_path)
        receipt_sha = sha256_file(receipt_path)
        for row in rows:
            if row.get("dataset_id") == "THOR-MAGNI":
                row.update({
                    "access_status": receipt["access_status"],
                    "retrieved_at_utc": receipt["generated_at_utc"],
                    "source_hash": receipt_sha,
                    "source_hash_kind": "MATERIALIZED_INTAKE_RECEIPT",
                    "local_evidence_paths": [str(manifest_path), str(receipt_path)],
                    "receipt_kind": "zenodo_archive_metadata_range_probe",
                    "event_truth_authority": False,
                })
        write_jsonl(source_receipt_path, rows)
    return {
        key: value for key, value in receipt.items()
        if key not in {"member_kind_bytes"}
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--record-id", default="13865754")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-request-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=512 * 1024 * 1024)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
