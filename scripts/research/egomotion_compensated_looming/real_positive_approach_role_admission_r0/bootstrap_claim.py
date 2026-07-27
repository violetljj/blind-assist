"""Pre-access exclusive claim bootstrap with no project imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_ID = "RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R0"
SOURCE_DESCRIPTOR_SHA256 = (
    "7f5170061170bb1fa4ac78fc1af8a172bb7a690720776c6295f4aaf683509a8e"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_claim(contract: Path, claim: Path) -> dict[str, object]:
    contract_sha256 = sha256_file(contract)
    payload = {
        "schema_version": "rcle.real_positive_approach_role_claim.v1",
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "exclusive_create": True,
        "contract_path_lexical": contract.as_posix(),
        "contract_sha256": contract_sha256,
        "source_descriptor_sha256": SOURCE_DESCRIPTOR_SHA256,
        "candidate_id": "EVIMO2_V2_FLEA3_SANITY_LL",
        "source_access_started_before_claim": False,
        "algorithm_outcome_access_started": False,
        "replacement_source_count": 0,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    claim.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(claim),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "nt":
            parent_descriptor = os.open(os.fspath(claim.parent), os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
    except BaseException:
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    args = parser.parse_args()
    payload = create_claim(args.contract, args.claim)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
