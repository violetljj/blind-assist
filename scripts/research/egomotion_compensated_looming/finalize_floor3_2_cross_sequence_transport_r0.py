"""Finalize the opaque floor3_2 transport after bytes and MD5 verification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


PROTOCOL_ID = (
    "RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_2_"
    "CROSS_SEQUENCE_GEOMETRY_STRATIFIED_DEVELOPMENT_HOLDOUT_R0"
)
EXPECTED_BYTES = 3_274_014_381
EXPECTED_MD5 = "e1a369f7c13cbb777a90d7e792085afa"
FILE_ID = "4909999130e6752b5e2147a0684b59ac"
URL = f"https://china.scidb.cn/download?fileId={FILE_ID}"


def hashes(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return size, md5.hexdigest(), sha256.hexdigest()


def write_exclusive(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True, type=Path)
    parser.add_argument("--final", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    args = parser.parse_args()
    if args.final.exists() or args.receipt.exists():
        raise FileExistsError("FLOOR3_2_FINAL_OR_RECEIPT_ALREADY_EXISTS")
    size, md5, sha256 = hashes(args.part)
    if size != EXPECTED_BYTES:
        raise ValueError(f"FLOOR3_2_BYTE_COUNT:{size}")
    if md5 != EXPECTED_MD5:
        raise ValueError(f"FLOOR3_2_MD5:{md5}")
    source_lock_sha = hashlib.sha256(args.source_lock.read_bytes()).hexdigest()
    args.part.replace(args.final)
    receipt = {
        "schema_version": "rcle.floor3_2.opaque_transport_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport_only": True,
        "zip_opened": False,
        "archive_members_read": 0,
        "rgb_pixels_read": False,
        "geometry_read": False,
        "file_id": FILE_ID,
        "canonical_url": URL,
        "archive_path": str(args.final.resolve()),
        "archive_bytes": size,
        "archive_md5": md5,
        "archive_sha256": sha256,
        "source_transport_lock_sha256": source_lock_sha,
        "same_identity_resume_and_retry_allowed": True,
        "alternate_source_count": 0
    }
    write_exclusive(args.receipt, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
