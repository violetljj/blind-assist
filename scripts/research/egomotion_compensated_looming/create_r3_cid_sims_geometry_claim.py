"""Create the single exclusive R3 geometry claim after opaque cache verification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROTOCOL_ID = "RCLE_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R3_CID_SIMS"
EXPECTED_BYTES = 2_211_008_069
EXPECTED_MD5 = "585d38855ad7d04817991cdbbb72016b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("R3_JSON_OBJECT_REQUIRED")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--transport-receipt", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--scanner", required=True, type=Path)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--geometry-implementation", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--expected-output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.claim.exists() or args.expected_output_dir.exists():
        raise FileExistsError("R3_CLAIM_OR_OUTPUT_ALREADY_EXISTS")
    receipt = load_object(args.transport_receipt)
    archive_sha = sha256(args.archive)
    if (
        args.archive.stat().st_size != EXPECTED_BYTES
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("archive_bytes") != EXPECTED_BYTES
        or receipt.get("archive_md5") != EXPECTED_MD5
        or receipt.get("archive_sha256") != archive_sha
        or receipt.get("zip_opened") is not False
        or receipt.get("archive_members_read") != 0
    ):
        raise ValueError("R3_VERIFIED_CACHE_REQUIRED")
    bindings = {
        "archive": archive_sha,
        "transport_receipt": sha256(args.transport_receipt),
        "source_lock": sha256(args.source_lock),
        "contract": sha256(args.contract),
        "scanner": sha256(args.scanner),
        "validator": sha256(args.validator),
        "geometry_implementation": sha256(args.geometry_implementation),
    }
    claim = {
        "schema_version": "rcle.r3.geometry_claim.v1",
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_count": 1,
        "candidate_id": "CID_SIMS_V6_FLOOR3_1",
        "official_run_id": "floor3_1",
        "archive_bytes": EXPECTED_BYTES,
        "archive_md5": EXPECTED_MD5,
        "archive_sha256": archive_sha,
        "zip_opened_before_claim": False,
        "archive_members_read_before_claim": 0,
        "rgb_pixels_read_before_claim": False,
        "geometry_read_before_claim": False,
        "alternate_source_count": 0,
        "validator_path": str(args.validator.resolve()),
        "bindings": bindings,
    }
    args.claim.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(args.claim), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(claim, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(claim, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
