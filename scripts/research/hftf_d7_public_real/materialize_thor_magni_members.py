#!/usr/bin/env python3
"""Materialize selected THOR-MAGNI ZIP members with bounded HTTP ranges.

The public Zenodo record exposes a 22+ GB ZIP.  This intake command reads the
ZIP central-directory manifest produced by ``inspect_thor_magni_archive.py``
and fetches only explicitly named members.  It verifies the local ZIP header,
decompresses stored/deflate members, checks size and CRC, and writes a receipt.
Extracted media remain source-intake evidence: this command does not pair
modalities, create candidate windows, or create D7 event truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

from inspect_thor_magni_archive import RangeReader, _get_json, _select_archive
from pipeline import ContractError, load_jsonl, sha256_file, stable_id, utc_now, write_json, write_jsonl


LOCAL_HEADER = struct.Struct("<4s5H3I2H")
LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"
SUPPORTED_COMPRESSION = {0, 8}
USER_AGENT = "blindassist-hftf-d7/1.0"


def _safe_member_path(raw_root: Path, member_name: str) -> Path:
    """Resolve an archive member below raw_root and reject traversal."""

    normalized = member_name.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not normalized or normalized.endswith("/") or not parts:
        raise ContractError(f"archive member is not a regular file: {member_name!r}")
    if PurePosixPath(normalized).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ContractError(f"unsafe archive member path: {member_name!r}")
    destination = (raw_root / Path(*parts)).resolve()
    root_resolved = raw_root.resolve()
    try:
        destination.relative_to(root_resolved)
    except ValueError as exc:
        raise ContractError(f"archive member escapes output root: {member_name!r}") from exc
    return destination


def _parse_local_header(header: bytes, *, member_name: str) -> tuple[int, int, int]:
    if len(header) < LOCAL_HEADER.size:
        raise ContractError(f"short local ZIP header for {member_name!r}")
    signature, _version, _flags, compression, _mtime, _mdate, _crc, _compressed, _uncompressed, name_len, extra_len = LOCAL_HEADER.unpack(
        header[: LOCAL_HEADER.size]
    )
    if signature != LOCAL_HEADER_SIGNATURE:
        raise ContractError(f"invalid local ZIP header for {member_name!r}: {signature!r}")
    return int(compression), int(name_len), int(extra_len)


def _decode_payload(compressed: bytes, *, compression: int, member_name: str) -> bytes:
    if compression == 0:
        return compressed
    if compression == 8:
        try:
            return zlib.decompress(compressed, -15)
        except zlib.error as exc:
            raise ContractError(f"deflate decompression failed for {member_name!r}: {exc}") from exc
    raise ContractError(
        f"unsupported ZIP compression for {member_name!r}: {compression}; "
        f"supported={sorted(SUPPORTED_COMPRESSION)}"
    )


def _validate_payload(payload: bytes, row: dict[str, Any], *, member_name: str) -> str:
    expected_size = int(row.get("file_size") or 0)
    expected_crc = int(str(row.get("crc32") or "0"), 16)
    if len(payload) != expected_size:
        raise ContractError(
            f"uncompressed size mismatch for {member_name!r}: {len(payload)} != {expected_size}"
        )
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ContractError(
            f"CRC mismatch for {member_name!r}: {actual_crc:08x} != {expected_crc:08x}"
        )
    return hashlib.sha256(payload).hexdigest()


def _read_range(reader: RangeReader, offset: int, size: int) -> bytes:
    """Read an exact range while honoring the reader's per-request cap."""

    if offset < 0 or size < 0:
        raise ContractError(f"invalid archive range: offset={offset}, size={size}")
    chunks: list[bytes] = []
    remaining = size
    cursor = offset
    while remaining:
        chunk_size = min(remaining, reader.max_request_bytes)
        reader.seek(cursor)
        chunk = reader.read(chunk_size)
        if len(chunk) != chunk_size:
            raise ContractError(f"short archive range at {cursor}: {len(chunk)} != {chunk_size}")
        chunks.append(chunk)
        cursor += chunk_size
        remaining -= chunk_size
    return b"".join(chunks)


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("member_name") or "")
        if not name or name in result:
            raise ContractError(f"duplicate or empty archive member in manifest: {name!r}")
        required = ("header_offset", "compressed_size", "file_size", "compression_method", "crc32")
        if any(key not in row for key in required):
            raise ContractError(f"archive manifest row lacks extraction metadata: {name!r}")
        result[name] = row
    return result


def _download_member(
    reader: RangeReader,
    row: dict[str, Any],
    *,
    destination: Path,
    max_member_bytes: int,
) -> dict[str, Any]:
    member_name = str(row["member_name"])
    file_size = int(row["file_size"])
    compressed_size = int(row["compressed_size"])
    if file_size < 0 or compressed_size < 0 or file_size > max_member_bytes:
        raise ContractError(
            f"member exceeds extraction cap for {member_name!r}: {file_size} > {max_member_bytes}"
        )
    header_offset = int(row["header_offset"])
    if header_offset < 0:
        raise ContractError(f"negative ZIP header offset for {member_name!r}")
    header = _read_range(reader, header_offset, LOCAL_HEADER.size)
    compression, name_len, extra_len = _parse_local_header(header, member_name=member_name)
    expected_compression = int(row["compression_method"])
    if compression != expected_compression:
        raise ContractError(
            f"compression drift for {member_name!r}: local={compression} central={expected_compression}"
        )
    data_offset = header_offset + LOCAL_HEADER.size + name_len + extra_len
    compressed = _read_range(reader, data_offset, compressed_size)
    payload = _decode_payload(compressed, compression=compression, member_name=member_name)
    sha256 = _validate_payload(payload, row, member_name=member_name)
    if destination.exists():
        raise ContractError(f"refusing to overwrite extracted member: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "schema": "hftf_d7_public_real_thor_magni_member_intake_v1",
        "member_id": str(row.get("member_id") or stable_id("d7thor-magni-member", member_name)),
        "dataset_id": "THOR-MAGNI",
        "member_name": member_name,
        "member_kind": row.get("member_kind"),
        "local_path": str(destination),
        "file_size": file_size,
        "compressed_size": compressed_size,
        "compression_method": compression,
        "crc32": str(row["crc32"]),
        "sha256": sha256,
        "source_revision": row.get("source_revision"),
        "event_truth_authority": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.member:
        raise ContractError("at least one explicit --member is required")
    if args.max_member_bytes <= 0 or args.max_output_bytes <= 0:
        raise ContractError("extraction caps must be positive")
    root = Path(args.output_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = _read_manifest(manifest_path)
    requested = list(dict.fromkeys(args.member))
    missing = [name for name in requested if name not in manifest]
    if missing:
        raise ContractError(f"requested members absent from manifest: {missing}")
    for name in requested:
        if str(manifest[name].get("member_kind")) == "OTHER" and int(manifest[name].get("file_size") or 0) == 0:
            raise ContractError(f"directory-like member cannot be extracted: {name!r}")
    output_dir = root / "raw" / args.output_subdir
    receipt_path = root / "receipts" / f"thor_magni_member_receipt_{args.run_id}.json"
    if receipt_path.exists():
        raise ContractError(f"receipt already exists; refusing overwrite: {receipt_path}")
    archive = _select_archive(_get_json(f"https://zenodo.org/api/records/{args.record_id}"))
    reader = RangeReader(
        str(archive["url"]),
        int(archive["size"]),
        max_request_bytes=args.max_request_bytes,
        max_total_bytes=args.max_total_bytes,
    )
    rows: list[dict[str, Any]] = []
    total_output = 0
    for name in requested:
        source_row = manifest[name]
        size = int(source_row["file_size"])
        if total_output + size > args.max_output_bytes:
            raise ContractError(
                f"selected members exceed output cap: {total_output + size} > {args.max_output_bytes}"
            )
        destination = _safe_member_path(output_dir, name.removeprefix("THOR_MAGNI/"))
        row = _download_member(reader, source_row, destination=destination, max_member_bytes=args.max_member_bytes)
        rows.append(row)
        total_output += size
    extracted_paths = [str(row["local_path"]) for row in rows]
    receipt = {
        "schema": "hftf_d7_public_real_thor_magni_member_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "dataset_id": "THOR-MAGNI",
        "record_id": args.record_id,
        "record_url": f"https://zenodo.org/records/{args.record_id}",
        "archive_key": archive["key"],
        "archive_checksum": archive.get("checksum"),
        "manifest_path": str(manifest_path),
        "member_count": len(rows),
        "total_output_bytes": total_output,
        "range_bytes_fetched": reader.bytes_fetched,
        "range_max_request_bytes": args.max_request_bytes,
        "range_max_total_bytes": args.max_total_bytes,
        "output_root": str(output_dir),
        "members": rows,
        "access_status": "PUBLIC_SELECTED_MEMBERS_MATERIALIZED",
        "status": "PUBLIC_SELECTED_MEMBERS_MATERIALIZED",
        "archive_content_materialized": False,
        "event_truth_authority": False,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
        "notes": [
            "Only explicitly selected ZIP members were fetched by HTTP Range.",
            "ZIP size/CRC and local SHA-256 were verified for every extracted member.",
            "Member extraction is source-intake evidence; no modality pairing or event labels were created.",
        ],
    }
    write_json(receipt_path, receipt)
    source_receipt_path = root / "receipts" / "source_receipts.jsonl"
    if source_receipt_path.is_file():
        source_rows = load_jsonl(source_receipt_path)
        receipt_sha = sha256_file(receipt_path)
        for source_row in source_rows:
            if source_row.get("dataset_id") == "THOR-MAGNI":
                evidence = list(source_row.get("local_evidence_paths") or [])
                evidence.extend([str(manifest_path), *extracted_paths, str(receipt_path)])
                source_row.update({
                    "access_status": receipt["access_status"],
                    "retrieved_at_utc": receipt["generated_at_utc"],
                    "source_hash": receipt_sha,
                    "source_hash_kind": "MATERIALIZED_INTAKE_RECEIPT",
                    "local_evidence_paths": list(dict.fromkeys(evidence)),
                    "receipt_kind": "zenodo_selected_member_range_extract",
                    "event_truth_authority": False,
                })
        write_jsonl(source_receipt_path, source_rows)
    return {
        key: value
        for key, value in receipt.items()
        if key not in {"members"}
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--record-id", default="13865754")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-subdir", default="thor-magni-selected")
    parser.add_argument("--member", action="append", default=[])
    parser.add_argument("--max-member-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-output-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-request-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=512 * 1024 * 1024)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
