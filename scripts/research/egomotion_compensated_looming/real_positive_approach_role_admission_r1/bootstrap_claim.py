"""Exclusive pre-access claim creation for RCLE role-admission R1.

This module deliberately has no project imports and performs no network or
candidate-path access.  The claim is the first executable protocol action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


PROTOCOL_ID = "RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R1"
CANDIDATE_ID = "ETH3D_SLAM_SOFA_3_RGBD"

CONTRACT_SHA256 = "e2a3dfdecfbfb660a6c708e8f1146e7c3652c3192c34fdb19b9f13c47f92dc38"
SOURCE_AUTHORITY_SHA256 = (
    "7fc127f42ab50516d198b36938c396d9a1d3bcbbf219c02a72b991853ed7eccf"
)
BURNED_MANIFEST_SHA256 = (
    "0b54cecc1f3908264f3d4bd06a37b7c27b6f149c05e92e5b3949c0a6ef201593"
)
SOURCE_DESCRIPTOR_SHA256 = (
    "11ac41e221ec6bdc16f12e071a9befdb55a2466e00bc8a78ee7fe67185b04756"
)
REQUIRED_IMPLEMENTATION_PATHS = {
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r1/__init__.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r1/bootstrap_claim.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r1/acquire.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r1/producer.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r1/validator.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r1/formal_runner.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r1/pilot.py",
    "scripts/research/egomotion_compensated_looming/pb_h1_role_proxy/geometry.py",
    "scripts/research/egomotion_compensated_looming/tum_fr2_rpy_geometry_audit/audit.py",
    "scripts/research/egomotion_compensated_looming/tests_real_positive_approach_role_admission_r1/__init__.py",
    "scripts/research/egomotion_compensated_looming/tests_real_positive_approach_role_admission_r1/test_r1.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_defaults() -> dict[str, str]:
    return {
        "contract": CONTRACT_SHA256,
        "source_authority": SOURCE_AUTHORITY_SHA256,
        "burned_manifest": BURNED_MANIFEST_SHA256,
    }


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"R1_EXPECTED_HASH_NOT_FROZEN:{label}")


def create_claim(
    contract: Path,
    source_authority: Path,
    burned_manifest: Path,
    implementation_lock: Path,
    claim: Path,
    *,
    expected_hashes: Mapping[str, str] | None = None,
    claim_created_by_runner_only: bool = False,
    verify_implementation_files: bool = True,
    repo_root: Path | None = None,
) -> dict[str, object]:
    if claim_created_by_runner_only is not True:
        raise ValueError("R1_CLAIM_MUST_BE_CREATED_BY_FORMAL_RUNNER")
    paths = {
        "contract": contract,
        "source_authority": source_authority,
        "burned_manifest": burned_manifest,
        "implementation_lock": implementation_lock,
    }
    if expected_hashes is None:
        raise ValueError("R1_EXPECTED_HASHES_REQUIRED")
    expected = dict(expected_hashes)
    if set(expected) != set(paths):
        raise ValueError("R1_EXPECTED_HASH_KEYS")
    observed = {name: sha256_file(path) for name, path in paths.items()}
    for name in paths:
        _validate_sha256(expected[name], name)
        if observed[name] != expected[name]:
            raise ValueError(f"R1_PREACCESS_HASH_MISMATCH:{name}")

    authority = json.loads(source_authority.read_text(encoding="utf-8"))
    manifest = json.loads(burned_manifest.read_text(encoding="utf-8"))
    contract_value = json.loads(contract.read_text(encoding="utf-8"))
    implementation = json.loads(implementation_lock.read_text(encoding="utf-8"))
    descriptor = authority["identity"]["source_descriptor_canonical_json"]
    descriptor_hash = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
    if (
        descriptor_hash != SOURCE_DESCRIPTOR_SHA256
        or authority["identity"]["source_descriptor_sha256"] != descriptor_hash
    ):
        raise ValueError("R1_SOURCE_DESCRIPTOR_HASH_MISMATCH")
    if authority.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("R1_SOURCE_AUTHORITY_PROTOCOL")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("R1_BURNED_MANIFEST_PROTOCOL")
    if contract_value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("R1_CONTRACT_PROTOCOL")
    if implementation.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("R1_IMPLEMENTATION_LOCK_PROTOCOL")
    if (
        implementation.get("contract_sha256") != observed["contract"]
        or implementation.get("source_authority_sha256")
        != observed["source_authority"]
        or implementation.get("burned_manifest_sha256")
        != observed["burned_manifest"]
        or implementation.get("source_descriptor_sha256")
        != SOURCE_DESCRIPTOR_SHA256
    ):
        raise ValueError("R1_IMPLEMENTATION_LOCK_AUTHORITY_BINDING")
    if verify_implementation_files:
        root = (repo_root or Path.cwd()).resolve()
        rows = implementation.get("files")
        if not isinstance(rows, list):
            raise ValueError("R1_IMPLEMENTATION_LOCK_FILES")
        locked = {
            row.get("path"): row.get("sha256")
            for row in rows
            if isinstance(row, dict)
        }
        if set(locked) != REQUIRED_IMPLEMENTATION_PATHS:
            raise ValueError("R1_IMPLEMENTATION_LOCK_SCOPE")
        for relative, expected_digest in locked.items():
            _validate_sha256(expected_digest, relative)
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError("R1_IMPLEMENTATION_LOCK_PATH") from error
            if not path.is_file() or sha256_file(path) != expected_digest:
                raise ValueError(f"R1_IMPLEMENTATION_FILE_MISMATCH:{relative}")
    if authority.get("candidate_count") != 1:
        raise ValueError("R1_CANDIDATE_COUNT")
    if authority["candidate"].get("candidate_id") != CANDIDATE_ID:
        raise ValueError("R1_CANDIDATE_ID")

    payload: dict[str, object] = {
        "schema_version": "rcle.real_positive_approach_role_claim.v2",
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "exclusive_create": True,
        "claim_created_by_runner_only": True,
        "bindings": {
            name: {
                "path_lexical": paths[name].as_posix(),
                "sha256": observed[name],
            }
            for name in sorted(paths)
        },
        "source_descriptor_sha256": descriptor_hash,
        "official_payload_url": authority["candidate"]["official_payload_url"],
        "source_access_started_before_claim": False,
        "candidate_path_probe_started_before_claim": False,
        "algorithm_outcome_access_started": False,
        "replacement_source_count": 0,
        "request_count_before_claim": 0,
        "payload_bytes_read_before_claim": 0,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    claim.parent.mkdir(parents=True, exist_ok=True)
    descriptor_fd = os.open(
        os.fspath(claim),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor_fd, "wb", closefd=True) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name != "nt":
        parent_fd = os.open(os.fspath(claim.parent), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--source-authority", required=True, type=Path)
    parser.add_argument("--burned-manifest", required=True, type=Path)
    parser.add_argument("--implementation-lock", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument(
        "--expected-implementation-lock-sha256",
        required=True,
    )
    parser.add_argument(
        "--created-by-formal-runner-only",
        required=True,
        action="store_true",
    )
    args = parser.parse_args()
    result = create_claim(
        args.contract,
        args.source_authority,
        args.burned_manifest,
        args.implementation_lock,
        args.claim,
        expected_hashes={
            **_expected_defaults(),
            "implementation_lock": args.expected_implementation_lock_sha256,
        },
        claim_created_by_runner_only=args.created_by_formal_runner_only,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
