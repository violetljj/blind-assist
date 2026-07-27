from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROTOCOL_ID = (
    "RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_POSTHOC_VALIDATOR_R1"
)
EXPECTED_LOCK_PATHS = {
    "docs/research/rcle/RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_POSTHOC_VALIDATOR_R1_CONTRACT_2026-07-27.json",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_development_canary_cid_sims_r0_posthoc_validator_r1/__init__.py",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_development_canary_cid_sims_r0_posthoc_validator_r1/validator.py",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_development_canary_cid_sims_r0_posthoc_validator_r1/run.py",
    "scripts/research/egomotion_compensated_looming/tests_rgb_algorithm_development_canary_cid_sims_r0_posthoc_validator_r1/test_validator.py",
}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def verify_implementation_lock(
    repo_root: Path,
    contract_path: Path,
    implementation_lock_path: Path,
    activation_path: Path,
) -> list[str]:
    errors: list[str] = []
    lock = load_object(implementation_lock_path)
    activation = load_object(activation_path)
    if lock.get("schema_version") != (
        "rcle.rgb_algorithm_development_canary.posthoc_implementation_lock.v1"
    ):
        errors.append("IMPLEMENTATION_LOCK_SCHEMA")
    if lock.get("status") != "LOCKED_BEFORE_POSTHOC_VALIDATION":
        errors.append("IMPLEMENTATION_LOCK_STATUS")
    if lock.get("protocol_id") != PROTOCOL_ID:
        errors.append("IMPLEMENTATION_LOCK_PROTOCOL")
    if activation.get("schema_version") != (
        "rcle.rgb_algorithm_development_canary.posthoc_activation.v1"
    ):
        errors.append("ACTIVATION_SCHEMA")
    if activation.get("status") != "AUTHORIZED_FOR_POSTHOC_VALIDATOR_ONLY":
        errors.append("ACTIVATION_STATUS")
    if activation.get("protocol_id") != PROTOCOL_ID:
        errors.append("ACTIVATION_PROTOCOL")
    lock_sha = digest_file(implementation_lock_path)
    if activation.get("implementation_lock_sha256") != lock_sha:
        errors.append("ACTIVATION_IMPLEMENTATION_LOCK_SHA")
    if activation.get("contract_sha256") != digest_file(contract_path):
        errors.append("ACTIVATION_CONTRACT_SHA")
    if activation.get("maximum_authority") != (
        "POSTHOC_R0_IDENTITY_CACHE_LEDGER_AGGREGATE_AUDIT_ONLY"
    ):
        errors.append("ACTIVATION_AUTHORITY")
    for field in (
        "algorithm_reexecution_authorized",
        "r0_evidence_revalidation_authorized",
        "outcome_blind_claim_authorized",
        "threshold_tuning_authorized",
        "independent_confirmation_authorized",
        "performance_qualification_authorized",
        "product_or_safety_claim_authorized",
        "network_access_authorized",
        "download_authorized",
    ):
        if activation.get(field) is not False:
            errors.append(f"ACTIVATION_FORBIDDEN:{field}")
    lock_files = lock.get("files", [])
    if not isinstance(lock_files, list):
        return sorted(set([*errors, "IMPLEMENTATION_LOCK_FILES"]))
    for item in lock_files:
        path = repo_root / item["path"]
        if not path.is_file() or digest_file(path) != item["sha256"]:
            errors.append(f"IMPLEMENTATION_FILE:{item['path']}")
    path_list = [str(item.get("path", "")) for item in lock_files]
    if (
        len(path_list) != len(EXPECTED_LOCK_PATHS)
        or len(path_list) != len(set(path_list))
        or set(path_list) != EXPECTED_LOCK_PATHS
    ):
        errors.append("IMPLEMENTATION_LOCK_ALLOWLIST")
    if not any(
        item.get("path") == str(contract_path.relative_to(repo_root)).replace("\\", "/")
        for item in lock_files
    ):
        errors.append("IMPLEMENTATION_LOCK_CONTRACT_ABSENT")
    return sorted(set(errors))


def failure_payload(errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "rcle.rgb_algorithm_development_canary.posthoc_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "terminal": "POSTHOC_OUTPUT_AUDIT_INVALID / INVALID",
        "status": "INVALID",
        "errors": errors,
        "algorithm_reexecution_performed": False,
        "r0_evidence_revalidated": False,
        "outcome_blind": False,
        "independent_confirmation": False,
        "performance_qualification": False,
        "threshold_tuned": False,
        "network_request_count": 0,
        "downloaded_bytes": 0,
        "authority": "POSTHOC_R0_IDENTITY_CACHE_LEDGER_AGGREGATE_AUDIT_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--implementation-lock", required=True, type=Path)
    parser.add_argument("--activation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    contract_path = args.contract.resolve()
    lock_path = args.implementation_lock.resolve()
    activation_path = args.activation.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError("POSTHOC_OUTPUT_ALREADY_EXISTS")
    errors = verify_implementation_lock(
        repo_root, contract_path, lock_path, activation_path
    )
    if errors:
        payload = failure_payload(errors)
    else:
        from .validator import validate

        payload = validate(repo_root, contract_path)
    payload["contract_sha256"] = digest_file(contract_path)
    payload["implementation_lock_sha256"] = digest_file(lock_path)
    payload["activation_sha256"] = digest_file(activation_path)
    write_exclusive(
        output_path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
