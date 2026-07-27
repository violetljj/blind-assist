"""Create the exclusive rank-one geometry transport claim before any GET."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scripts.research.egomotion_compensated_looming.motion_diverse_rgbd_geometry_admission_r0.template import (
    validate_execution_contract,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def write_exclusive(path: Path, value: object) -> None:
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
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--burned-receipt", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    contract_path = args.contract.resolve()
    contract = load_object(contract_path)
    validate_execution_contract(contract)
    candidate_lock = repo / contract["candidate"]["candidate_lock_path"]
    if sha256_file(candidate_lock) != contract["candidate"]["candidate_lock_sha256"]:
        raise ValueError("CANDIDATE_LOCK_IDENTITY")
    adapter = repo / contract["transport"]["stable_root_adapter"]
    if sha256_file(adapter) != contract["transport"]["stable_root_adapter_sha256"]:
        raise ValueError("ROOT_ADAPTER_IDENTITY")
    burned_path = args.burned_receipt.resolve()
    if sha256_file(burned_path) != contract["execution"]["burned_fixture_receipt_sha256"]:
        raise ValueError("BURNED_RECEIPT_IDENTITY")
    burned = load_object(burned_path)
    if (
        burned.get("status") != "BURNED_FIXTURE_SMOKE_PASS"
        or burned.get("rgb_accessed") is not False
        or int(burned.get("default_workers", -1)) != 8
    ):
        raise ValueError("BURNED_RECEIPT_SEMANTICS")
    claim = {
        "schema": "rcle.motion_diverse_rgbd.geometry_admission.claim.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": sha256_file(contract_path),
        "candidate_lock_sha256": sha256_file(candidate_lock),
        "candidate_id": contract["candidate"]["candidate_id"],
        "metadata_rank": contract["candidate"]["metadata_rank"],
        "stage": "CENTRAL_DIRECTORY_AND_POSE_ONLY",
        "candidate_replacement_allowed": False,
        "whole_archive_download_allowed": False,
        "rgb_payload_allowed": False,
    }
    write_exclusive(args.claim.resolve(), claim)
    print(json.dumps(claim, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
