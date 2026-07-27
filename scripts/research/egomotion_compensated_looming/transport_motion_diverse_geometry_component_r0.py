"""Exact-identity transport for one rank-one geometry component.

The tool is intentionally geometry-only.  It refuses RGB roles, Floor3_3,
candidate replacement, and manifests that lack exact bytes plus a checksum.
Invoke it through ``scripts/run_research_tool.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import urllib.request


SCHEMA = "rcle.motion_diverse.geometry_component_transport.v1"
ALLOWED_COMPONENT_ROLES = {
    "depth",
    "pose",
    "trajectory",
    "calibration",
    "geometry",
}


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("TRANSPORT_MANIFEST_SCHEMA")
    if int(value.get("candidate_metadata_rank", -1)) != 1:
        raise ValueError("ONLY_RANK_ONE_CANDIDATE_ALLOWED")
    if value.get("candidate_replacement_allowed") is not False:
        raise ValueError("CANDIDATE_REPLACEMENT_MUST_BE_FALSE")
    if "floor3_3" in str(value.get("candidate_id", "")).lower():
        raise ValueError("FLOOR3_3_FORBIDDEN")
    role = str(value.get("component_role", "")).lower()
    if role not in ALLOWED_COMPONENT_ROLES:
        raise ValueError("GEOMETRY_COMPONENT_ROLE_REQUIRED")
    if not str(value.get("canonical_url", "")).startswith("https://"):
        raise ValueError("CANONICAL_HTTPS_URL_REQUIRED")
    if int(value.get("expected_bytes", 0)) <= 0:
        raise ValueError("EXACT_POSITIVE_BYTES_REQUIRED")
    sha256 = value.get("expected_sha256")
    md5 = value.get("expected_md5")
    if not sha256 and not md5:
        raise ValueError("OFFICIAL_CHECKSUM_REQUIRED")
    if sha256 and len(str(sha256)) != 64:
        raise ValueError("SHA256_FORMAT")
    if md5 and len(str(md5)) != 32:
        raise ValueError("MD5_FORMAT")
    return value


def hashes(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(block)
            md5.update(block)
            sha256.update(block)
    return size, md5.hexdigest(), sha256.hexdigest()


def write_exclusive(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--part", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    # Reserve authority before the first network request or output probe.
    claim = {
        "schema": "rcle.motion_diverse.geometry_transport_claim.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_sha256,
        "candidate_id": manifest["candidate_id"],
        "component_role": manifest["component_role"],
        "same_identity_resume_allowed": True,
        "candidate_replacement_allowed": False,
    }
    write_exclusive(args.claim.resolve(), claim)

    part = args.part.resolve()
    final = args.final.resolve()
    receipt = args.receipt.resolve()
    if final.exists() or receipt.exists():
        raise FileExistsError("FINAL_OR_RECEIPT_ALREADY_EXISTS")
    part.parent.mkdir(parents=True, exist_ok=True)
    existing = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(manifest["canonical_url"], headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        mode = "ab" if existing and response.status == 206 else "wb"
        with part.open(mode) as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())

    size, md5, sha256 = hashes(part)
    if size != int(manifest["expected_bytes"]):
        raise ValueError(f"BYTE_COUNT_MISMATCH:{size}")
    if manifest.get("expected_md5") and md5 != manifest["expected_md5"]:
        raise ValueError(f"MD5_MISMATCH:{md5}")
    if manifest.get("expected_sha256") and sha256 != manifest["expected_sha256"]:
        raise ValueError(f"SHA256_MISMATCH:{sha256}")
    part.replace(final)
    result = {
        "schema": "rcle.motion_diverse.geometry_transport_receipt.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_sha256,
        "candidate_id": manifest["candidate_id"],
        "component_role": manifest["component_role"],
        "canonical_url": manifest["canonical_url"],
        "bytes": size,
        "md5": md5,
        "sha256": sha256,
        "rgb_payload": False,
        "candidate_replacement": False,
        "final_path": str(final),
    }
    write_exclusive(receipt, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
