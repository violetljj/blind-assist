"""Exclusive pre-access claim creation for CID-SIMS Floor3 R2.

This module has no project imports and performs no network or candidate-path
access.  The formal runner must call ``create_claim`` before any payload probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


PROTOCOL_ID = "RCLE_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS"
CANDIDATE_ID = "CID_SIMS_V6_FLOOR3_1"
SEQUENCE_ID = "floor3_1"
FROZEN_RUN_UNIVERSE = ("floor3_1", "floor3_2", "floor3_3")
SELECTION_RULE = (
    "Choose the lexicographically smallest official run ID from the frozen "
    "floor3_1/floor3_2/floor3_3 universe before payload access."
)
OFFICIAL_FILE_ID = "c595882daafe788a29d687872cc1fc2a"
OFFICIAL_PAYLOAD_URL = (
    "https://china.scidb.cn/download?fileId="
    "c595882daafe788a29d687872cc1fc2a"
)
OFFICIAL_PAYLOAD_BYTES = 2_211_008_069
OFFICIAL_PAYLOAD_MD5 = "585d38855ad7d04817991cdbbb72016b"

CONTRACT_SHA256 = "52ceebbe7727952c0fb963dfc855ae9115bfa3a5b2187649665dac49f4c5f6b4"
SOURCE_AUTHORITY_SHA256 = "49fdf51620aeb5b0c06fe7ce5c8d0944d78768c958153644f9ba64ebb4119659"
BURNED_MANIFEST_SHA256 = "0ce9494307c3a872edc8bb7aa00aa061965165cae27a6a6bf34fb47f28c74a26"
SOURCE_DESCRIPTOR_SHA256 = "54c2f5e207cd94b7ca8e2e6f5e795ee6dd61018b5698ae9ad4f08d48d196dc7f"

REQUIRED_IMPLEMENTATION_PATHS = {
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r2_cid_sims/__init__.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r2_cid_sims/bootstrap_claim.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r2_cid_sims/acquire.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r2_cid_sims/producer.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r2_cid_sims/validator.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r2_cid_sims/formal_runner.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r2_cid_sims/pilot.py",
    "scripts/research/egomotion_compensated_looming/tests_real_positive_approach_role_admission_r2_cid_sims/__init__.py",
    "scripts/research/egomotion_compensated_looming/tests_real_positive_approach_role_admission_r2_cid_sims/test_geometry.py",
    "scripts/research/egomotion_compensated_looming/tests_real_positive_approach_role_admission_r2_cid_sims/test_acquisition_claim.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_frozen_candidate(sequence_ids: Iterable[str]) -> str:
    """Apply the preregistered result-blind rule to the exact frozen universe."""

    values = tuple(sequence_ids)
    if len(values) != len(set(values)) or set(values) != set(FROZEN_RUN_UNIVERSE):
        raise ValueError("R2_FROZEN_RUN_UNIVERSE_MISMATCH")
    selected = min(values)
    if selected != SEQUENCE_ID:
        raise ValueError("R2_SELECTION_RULE_DRIFT")
    return selected


def _expected_defaults() -> dict[str, str]:
    return {
        "contract": CONTRACT_SHA256,
        "source_authority": SOURCE_AUTHORITY_SHA256,
        "burned_manifest": BURNED_MANIFEST_SHA256,
    }


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"R2_EXPECTED_HASH_NOT_FROZEN:{label}")


def _candidate(authority: dict[str, object]) -> dict[str, object]:
    candidate = authority.get("candidate")
    if not isinstance(candidate, dict):
        candidate = authority.get("source_selection")
    if not isinstance(candidate, dict):
        raise ValueError("R2_SOURCE_AUTHORITY_CANDIDATE")
    return candidate


def _candidate_value(candidate: dict[str, object], key: str) -> object:
    if key in candidate:
        return candidate[key]
    aliases = {
        "official_run_id": "sequence_id",
        "canonical_url": "official_payload_url",
        "official_bytes": "bytes",
    }
    alias = aliases.get(key)
    if alias is not None and alias in candidate:
        return candidate[alias]
    archive = candidate.get("archive")
    if isinstance(archive, dict) and key in archive:
        return archive[key]
    raise ValueError(f"R2_SOURCE_AUTHORITY_CANDIDATE_FIELD:{key}")


def _validate_selection_authority(authority: dict[str, object]) -> None:
    selection = authority.get("selection_authority")
    if not isinstance(selection, dict):
        raise ValueError("R2_SELECTION_AUTHORITY")
    frozen = authority.get("frozen_candidate_universe")
    universe = frozen.get("official_run_ids") if isinstance(frozen, dict) else None
    if not isinstance(universe, list):
        universe = selection.get("eligible_sequence_ids")
    if not isinstance(universe, list):
        raise ValueError("R2_SELECTION_UNIVERSE")
    if select_frozen_candidate(str(value) for value in universe) != SEQUENCE_ID:
        raise ValueError("R2_SELECTION_RESULT")
    if selection.get(
        "selected_official_run_id", selection.get("selected")
    ) != SEQUENCE_ID:
        raise ValueError("R2_SELECTION_RESULT")
    if selection.get(
        "deterministic_rule", selection.get("deterministic_selection_rule")
    ) not in {
        "Select the lexicographically smallest exact official_run_id.",
        SELECTION_RULE,
    }:
        raise ValueError("R2_SELECTION_RULE_BINDING")
    outcome_blind = (
        frozen.get("payload_access_used_for_selection") is False
        if isinstance(frozen, dict)
        else selection.get("outcome_blind") is True
    )
    if not outcome_blind:
        raise ValueError("R2_SELECTION_OUTCOME_BLIND")


def create_claim(
    contract: Path,
    source_authority: Path,
    burned_manifest: Path,
    implementation_lock: Path,
    claim: Path,
    *,
    expected_hashes: Mapping[str, str] | None = None,
    expected_source_descriptor_sha256: str = SOURCE_DESCRIPTOR_SHA256,
    claim_created_by_runner_only: bool = False,
    verify_implementation_files: bool = True,
    repo_root: Path | None = None,
) -> dict[str, object]:
    if claim_created_by_runner_only is not True:
        raise ValueError("R2_CLAIM_MUST_BE_CREATED_BY_FORMAL_RUNNER")
    paths = {
        "contract": contract,
        "source_authority": source_authority,
        "burned_manifest": burned_manifest,
        "implementation_lock": implementation_lock,
    }
    if expected_hashes is None:
        raise ValueError("R2_EXPECTED_HASHES_REQUIRED")
    expected = dict(expected_hashes)
    if set(expected) != set(paths):
        raise ValueError("R2_EXPECTED_HASH_KEYS")
    observed = {name: sha256_file(path) for name, path in paths.items()}
    for name in paths:
        _validate_sha256(expected[name], name)
        if observed[name] != expected[name]:
            raise ValueError(f"R2_PREACCESS_HASH_MISMATCH:{name}")
    _validate_sha256(
        expected_source_descriptor_sha256,
        "source_descriptor",
    )

    authority = json.loads(source_authority.read_text(encoding="utf-8"))
    manifest = json.loads(burned_manifest.read_text(encoding="utf-8"))
    contract_value = json.loads(contract.read_text(encoding="utf-8"))
    implementation = json.loads(implementation_lock.read_text(encoding="utf-8"))
    for value, label in (
        (authority, "SOURCE_AUTHORITY"),
        (manifest, "BURNED_MANIFEST"),
        (contract_value, "CONTRACT"),
        (implementation, "IMPLEMENTATION_LOCK"),
    ):
        if not isinstance(value, dict) or value.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"R2_{label}_PROTOCOL")

    identity = authority.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("R2_SOURCE_DESCRIPTOR")
    descriptor = identity.get("source_descriptor_canonical_json")
    if not isinstance(descriptor, str):
        raise ValueError("R2_SOURCE_DESCRIPTOR")
    descriptor_hash = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
    if (
        descriptor_hash != expected_source_descriptor_sha256
        or identity.get("source_descriptor_sha256") != descriptor_hash
    ):
        raise ValueError("R2_SOURCE_DESCRIPTOR_HASH_MISMATCH")

    _validate_selection_authority(authority)
    if authority.get("candidate_count") != 1:
        raise ValueError("R2_CANDIDATE_COUNT")
    candidate = _candidate(authority)
    expected_candidate = {
        "candidate_id": CANDIDATE_ID,
        "official_run_id": SEQUENCE_ID,
        "file_id": OFFICIAL_FILE_ID,
        "canonical_url": OFFICIAL_PAYLOAD_URL,
        "official_bytes": OFFICIAL_PAYLOAD_BYTES,
        "official_md5": OFFICIAL_PAYLOAD_MD5,
    }
    for key, expected_value in expected_candidate.items():
        if _candidate_value(candidate, key) != expected_value:
            raise ValueError(f"R2_CANDIDATE_IDENTITY:{key}")

    if (
        implementation.get("contract_sha256") != observed["contract"]
        or implementation.get("source_authority_sha256")
        != observed["source_authority"]
        or implementation.get("burned_manifest_sha256")
        != observed["burned_manifest"]
        or implementation.get("source_descriptor_sha256") != descriptor_hash
    ):
        raise ValueError("R2_IMPLEMENTATION_LOCK_AUTHORITY_BINDING")

    if verify_implementation_files:
        root = (repo_root or Path.cwd()).resolve()
        rows = implementation.get("files")
        if not isinstance(rows, list):
            raise ValueError("R2_IMPLEMENTATION_LOCK_FILES")
        locked = {
            row.get("path"): row.get("sha256")
            for row in rows
            if isinstance(row, dict)
        }
        if set(locked) != REQUIRED_IMPLEMENTATION_PATHS:
            raise ValueError("R2_IMPLEMENTATION_LOCK_SCOPE")
        for relative, expected_digest in locked.items():
            if not isinstance(relative, str) or not isinstance(
                expected_digest, str
            ):
                raise ValueError("R2_IMPLEMENTATION_LOCK_ROW")
            _validate_sha256(expected_digest, relative)
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError("R2_IMPLEMENTATION_LOCK_PATH") from error
            if not path.is_file() or sha256_file(path) != expected_digest:
                raise ValueError(f"R2_IMPLEMENTATION_FILE_MISMATCH:{relative}")

    payload: dict[str, object] = {
        "schema_version": "rcle.real_positive_approach_role_claim.v3",
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "sequence_id": SEQUENCE_ID,
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
        "selection_rule": SELECTION_RULE,
        "frozen_run_universe": list(FROZEN_RUN_UNIVERSE),
        "official_file_id": OFFICIAL_FILE_ID,
        "official_payload_url": OFFICIAL_PAYLOAD_URL,
        "official_payload_bytes": OFFICIAL_PAYLOAD_BYTES,
        "official_payload_md5": OFFICIAL_PAYLOAD_MD5,
        "source_access_started_before_claim": False,
        "candidate_path_probe_started_before_claim": False,
        "algorithm_outcome_access_started": False,
        "replacement_source_count": 0,
        "request_count_before_claim": 0,
        "payload_bytes_read_before_claim": 0,
        "redirect_authorized": False,
        "retry_authorized": False,
        "alternate_run_authorized": False,
        "fallback_source_authorized": False,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    claim.parent.mkdir(parents=True, exist_ok=True)
    claim_fd = os.open(
        os.fspath(claim),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(claim_fd, "wb", closefd=True) as stream:
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
