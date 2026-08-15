#!/usr/bin/env python3
"""Validate a completed, outcome-blind A2D2 TAR stream index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(lock_path: Path, database_path: Path, receipt_path: Path) -> None:
    source_lock_bytes = lock_path.read_bytes()
    source_lock = json.loads(source_lock_bytes)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("schema") == "blindassist.svrf_o0.a2d2_stream_index_receipt.v1", "stream receipt schema drift")
    require(receipt.get("status") == "A2D2_TAR_STREAM_INDEX_COMPLETE", "stream receipt terminal drift")
    require(receipt.get("source_lock_sha256") == hashlib.sha256(source_lock_bytes).hexdigest(), "stream source-lock drift")
    require(receipt.get("database_sha256") == sha256(database_path), "stream database hash drift")
    require(receipt.get("archive_payload_retained") is False, "stream archive payload was retained")
    require(receipt.get("media_tensor_or_label_decode_count") == 0, "stream content semantics were decoded")
    require(receipt.get("candidate_run_count") == 0 and receipt.get("outcome_accessed") is False, "stream opened candidate outcome")

    source = next(item for item in source_lock["sources"] if item["source_id"] == "A2D2_SENSOR_FUSION")
    bindings = {
        archive["name"]: {"parent_id": parent["parent_id"], "modality": modality, **archive}
        for parent in source["parents"]
        for modality, archive in parent["archives"].items()
    }
    binding = bindings.get(receipt["archive"])
    require(binding is not None, "stream archive is outside the source lock")
    require(receipt["archive_bytes"] == binding["bytes"], "stream archive byte-size drift")
    require(receipt["official_md5_bound_not_recomputed"] == binding["md5"], "stream official MD5 binding drift")
    require(receipt["parent_id"] == binding["parent_id"] and receipt["modality"] == binding["modality"], "stream parent/modality drift")

    uri = database_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        require(metadata.get("status") == "COMPLETE", "stream database is incomplete")
        require(int(metadata["next_header_offset"]) == binding["bytes"], "stream terminal offset drift")
        require(metadata.get("pending_long_name") == "", "stream has unresolved GNU long name")
        rows = list(
            connection.execute(
                """
                SELECT header_offset, member_name, member_type, payload_offset, payload_bytes,
                       padded_payload_bytes, payload_sha256, tar_header_checksum,
                       next_header_offset, long_name_target
                FROM members ORDER BY header_offset
                """
            )
        )
        require(rows, "stream member index is empty")
        require(rows[0][0] == 0, "stream first TAR header is not offset zero")
        for index, row in enumerate(rows):
            header_offset, name, _, payload_offset, payload_bytes, padded, payload_hash, checksum, next_offset, _ = row
            require(name != "", "stream member path is empty")
            require(payload_offset == header_offset + 512, "stream payload offset drift")
            require(padded == ((payload_bytes + 511) // 512) * 512, "stream padded-size drift")
            require(next_offset == payload_offset + padded, "stream next-header offset drift")
            require(len(payload_hash) == 64 and all(character in "0123456789abcdef" for character in payload_hash), "stream payload hash drift")
            require(checksum > 0, "stream TAR checksum is absent")
            if index + 1 < len(rows):
                require(next_offset == rows[index + 1][0], "stream member offsets are not continuous")
        require(len(rows) == receipt["member_count"], "stream receipt member denominator drift")
        require(sum(row[4] for row in rows) == receipt["indexed_payload_bytes"], "stream payload-byte denominator drift")
        require(rows[-1][8] == receipt["last_next_header_offset"], "stream final member offset drift")
        requests = list(
            connection.execute(
                "SELECT start_offset, bytes_received, curl_exit_code, completed, error FROM requests ORDER BY request_id"
            )
        )
        require(len(requests) == receipt["request_count"], "stream request denominator drift")
        require(sum(row[1] for row in requests) == receipt["network_bytes_received"], "stream network-byte counter drift")
        require(sum(row[3] for row in requests) == receipt["completed_request_count"], "stream completed-request counter drift")
        require(any(row[3] == 1 for row in requests), "stream has no completed request")
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    validate(args.lock, args.database, args.receipt)
    print("SVRF_O0_A2D2_STREAM_INDEX_VALID")


if __name__ == "__main__":
    main()
