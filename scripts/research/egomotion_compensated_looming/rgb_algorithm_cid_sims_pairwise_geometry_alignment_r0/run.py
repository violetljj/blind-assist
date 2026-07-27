from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROTOCOL_ID = (
    "RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_PAIRWISE_GEOMETRY_ALIGNMENT_R0"
)
EXPECTED_LOCK_PATHS = {
    "docs/research/rcle/RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_PAIRWISE_GEOMETRY_ALIGNMENT_R0_CONTRACT_2026-07-27.json",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/__init__.py",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/producer.py",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/validator.py",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/run.py",
    "scripts/research/egomotion_compensated_looming/tests_rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/__init__.py",
    "scripts/research/egomotion_compensated_looming/tests_rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/test_alignment.py",
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


def verify_locks(
    repo_root: Path,
    contract_path: Path,
    implementation_lock_path: Path,
    activation_path: Path,
    output_dir: Path,
) -> list[str]:
    errors: list[str] = []
    lock = load_object(implementation_lock_path)
    activation = load_object(activation_path)
    if lock.get("schema_version") != (
        "rcle.rgb_algorithm.pairwise_geometry_alignment.implementation_lock.v1"
    ):
        errors.append("IMPLEMENTATION_LOCK_SCHEMA")
    if lock.get("status") != "LOCKED_BEFORE_FULL_PAIR_GEOMETRY_ACCESS":
        errors.append("IMPLEMENTATION_LOCK_STATUS")
    if lock.get("protocol_id") != PROTOCOL_ID:
        errors.append("IMPLEMENTATION_LOCK_PROTOCOL")
    if activation.get("schema_version") != (
        "rcle.rgb_algorithm.pairwise_geometry_alignment.activation.v1"
    ):
        errors.append("ACTIVATION_SCHEMA")
    if activation.get("status") != "AUTHORIZED_FOR_ONE_PAIRWISE_ALIGNMENT_RUN":
        errors.append("ACTIVATION_STATUS")
    if activation.get("protocol_id") != PROTOCOL_ID:
        errors.append("ACTIVATION_PROTOCOL")
    if activation.get("implementation_lock_sha256") != digest_file(
        implementation_lock_path
    ):
        errors.append("ACTIVATION_IMPLEMENTATION_LOCK_SHA")
    if activation.get("contract_sha256") != digest_file(contract_path):
        errors.append("ACTIVATION_CONTRACT_SHA")
    contract = load_object(contract_path)
    if activation.get("archive_sha256") != contract.get("source", {}).get(
        "archive_sha256"
    ):
        errors.append("ACTIVATION_ARCHIVE_SHA")
    try:
        expected_output = str(output_dir.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        errors.append("OUTPUT_OUTSIDE_REPOSITORY")
    else:
        if activation.get("output_dir") != expected_output:
            errors.append("ACTIVATION_OUTPUT_DIR")
    if activation.get("maximum_authority") != (
        "POSTHOC_REAL_DATA_MECHANISM_ALIGNMENT_ONLY"
    ):
        errors.append("ACTIVATION_AUTHORITY")
    forbidden = (
        "algorithm_reexecution_authorized",
        "threshold_tuning_authorized",
        "outcome_blind_claim_authorized",
        "independent_confirmation_authorized",
        "performance_qualification_authorized",
        "product_or_safety_claim_authorized",
        "network_access_authorized",
        "download_authorized",
    )
    for field in forbidden:
        if activation.get(field) is not False:
            errors.append(f"ACTIVATION_FORBIDDEN:{field}")
    files = lock.get("files")
    if not isinstance(files, list):
        return sorted(set([*errors, "IMPLEMENTATION_LOCK_FILES"]))
    paths = [str(item.get("path", "")) for item in files]
    if (
        len(paths) != len(EXPECTED_LOCK_PATHS)
        or len(paths) != len(set(paths))
        or set(paths) != EXPECTED_LOCK_PATHS
    ):
        errors.append("IMPLEMENTATION_LOCK_ALLOWLIST")
    for item in files:
        path = repo_root / str(item.get("path", ""))
        if not path.is_file() or digest_file(path) != item.get("sha256"):
            errors.append(f"IMPLEMENTATION_FILE:{item.get('path', '')}")
    return sorted(set(errors))


def failure_payload(errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "rcle.rgb_algorithm.pairwise_geometry_alignment.validation.v1",
        "protocol_id": PROTOCOL_ID,
        "terminal": "POSTHOC_PAIRWISE_ALIGNMENT_INVALID / INVALID",
        "status": "INVALID",
        "errors": errors,
        "authority": "POSTHOC_REAL_DATA_MECHANISM_ALIGNMENT_ONLY",
        "algorithm_reexecution_performed": False,
        "threshold_tuned": False,
        "outcome_blind": False,
        "independent_confirmation": False,
        "performance_qualification": False,
        "network_request_count": 0,
        "downloaded_bytes": 0,
        "r0_evidence_status": "INVALID_R0_EVIDENCE / INVALID",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--implementation-lock", required=True, type=Path)
    parser.add_argument("--activation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    contract_path = args.contract.resolve()
    lock_path = args.implementation_lock.resolve()
    activation_path = args.activation.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("OUTPUT_DIRECTORY_ALREADY_EXISTS")
    errors = verify_locks(
        repo_root,
        contract_path,
        lock_path,
        activation_path,
        output_dir,
    )
    if errors:
        output_dir.mkdir(parents=True, exist_ok=False)
        validation = failure_payload(errors)
    else:
        from .producer import run as produce
        from .validator import validate

        produce(repo_root, contract_path, output_dir, args.workers)
        validation = validate(
            repo_root,
            contract_path,
            output_dir,
            lock_path,
            activation_path,
            args.workers,
        )
    validation["contract_sha256"] = digest_file(contract_path)
    validation["implementation_lock_sha256"] = digest_file(lock_path)
    validation["activation_sha256"] = digest_file(activation_path)
    write_exclusive(
        output_dir / "validation.json",
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
    return 0 if validation["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
