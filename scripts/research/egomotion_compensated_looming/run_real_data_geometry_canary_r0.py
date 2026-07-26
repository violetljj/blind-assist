from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from egomotion_compensated_looming.real_data_geometry_canary_r0.producer import (
    canonical_bytes,
    load_json,
    produce_archive,
    sha256_file,
)
from egomotion_compensated_looming.real_data_geometry_canary_r0.validator import (
    read_pair_ledger,
    validate_bound_receipt_files,
    validate_materialized,
)


MODULE = Path(__file__).resolve().parent / "real_data_geometry_canary_r0"
REQUIRED_CONTROL_FILES = {
    "docs/research/rcle/RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_CONTRACT_2026-07-26.json",
    "docs/research/rcle/RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_PREREGISTRATION_2026-07-26.md",
    "docs/research/rcle/RCLE_TUM_FR2_RPY_SOURCE_NATIVE_GEOMETRY_AUDIT_R0_CONTRACT_2026-07-26.json",
    "scripts/research/egomotion_compensated_looming/pb_h1_role_proxy/geometry.py",
    "scripts/research/egomotion_compensated_looming/real_data_geometry_canary_r0/__init__.py",
    "scripts/research/egomotion_compensated_looming/real_data_geometry_canary_r0/README.md",
    "scripts/research/egomotion_compensated_looming/real_data_geometry_canary_r0/runtime_config_r0.json",
    "scripts/research/egomotion_compensated_looming/real_data_geometry_canary_r0/output_schema_r0.json",
    "scripts/research/egomotion_compensated_looming/real_data_geometry_canary_r0/producer.py",
    "scripts/research/egomotion_compensated_looming/real_data_geometry_canary_r0/validator.py",
    "scripts/research/egomotion_compensated_looming/run_real_data_geometry_canary_r0.py",
    "scripts/research/egomotion_compensated_looming/tests_real_data_geometry_canary_r0/__init__.py",
    "scripts/research/egomotion_compensated_looming/tests_real_data_geometry_canary_r0/test_canary.py",
}


def _verify_implementation_lock(
    repo_root: Path,
    implementation_lock: Path,
) -> tuple[dict[str, Any], str]:
    raw = implementation_lock.read_bytes()
    lock = json.loads(raw.decode("utf-8"))
    if (
        lock.get("schema_version")
        != "rcle.real_data_geometry_canary.implementation_lock.v1"
        or lock.get("implementation_review_status") != "PASS"
        or lock.get("formal_execution_authorized") is not False
    ):
        raise ValueError("IMPLEMENTATION_LOCK_NOT_REVIEWED")
    control_files = lock.get("control_files")
    if not isinstance(control_files, list):
        raise ValueError("IMPLEMENTATION_CONTROL_MANIFEST_INVALID")
    control_paths = [
        item.get("path")
        for item in control_files
        if isinstance(item, dict)
    ]
    if (
        len(control_paths) != len(control_files)
        or len(control_paths) != len(set(control_paths))
        or set(control_paths) != REQUIRED_CONTROL_FILES
    ):
        raise ValueError("IMPLEMENTATION_CONTROL_MANIFEST_INVALID")
    for item in control_files:
        path = repo_root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(
                f"IMPLEMENTATION_CONTROL_HASH_MISMATCH:{item['path']}"
            )
    return lock, sha256(raw).hexdigest()


def _verify_activation(
    path: Path,
    implementation_lock_sha256: str,
) -> str:
    if not path.is_file():
        raise ValueError("FORMAL_EXECUTION_ACTIVATION_LOCK_MISSING")
    activation = load_json(path)
    if (
        activation.get("schema_version")
        != "rcle.real_data_geometry_canary.activation_lock.v1"
        or activation.get("protocol_id")
        != "RCLE-PHASE-B-REAL-DATA-GEOMETRY-CANARY-R0"
        or activation.get("implementation_lock_sha256")
        != implementation_lock_sha256
        or activation.get("canonical_execution_authorized") is not True
    ):
        raise ValueError("FORMAL_EXECUTION_ACTIVATION_LOCK_INVALID")
    return sha256(path.read_bytes()).hexdigest()


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _verify_bound_inputs(
    repo_root: Path,
    config: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, str]:
    bindings = contract["prior_evidence_bindings"]
    contract_path = repo_root / config["protocol_contract"]
    paths = {
        "contract_sha256": contract_path,
        "archive_sha256": repo_root / config["source_archive"],
        "source_audit_contract_sha256": (
            repo_root / config["source_audit_contract"]
        ),
        "source_audit_result_sha256": (
            repo_root / config["source_audit_result"]
        ),
        "pb_h1_geometry_sha256": (
            repo_root / config["pb_h1_geometry"]
        ),
    }
    expected = {
        "contract_sha256": sha256_file(contract_path),
        "archive_sha256": bindings["source_archive_sha256"],
        "source_audit_contract_sha256": (
            bindings["source_audit_contract"]["sha256"]
        ),
        "source_audit_result_sha256": (
            bindings["source_audit_result_sha256"]
        ),
        "pb_h1_geometry_sha256": (
            bindings["pb_h1_geometry_implementation_sha256"]
        ),
    }
    actual: dict[str, str] = {}
    for key, path in paths.items():
        if not path.is_file():
            raise ValueError(f"BOUND_INPUT_MISSING:{key}")
        actual[key] = sha256_file(path)
        if actual[key] != expected[key]:
            raise ValueError(f"BOUND_INPUT_HASH_MISMATCH:{key}")
    return actual


def _pair_ledger_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--activation-lock", type=Path, required=True)
    arguments = parser.parse_args()
    repo_root = arguments.repo_root.resolve()
    config_path = MODULE / "runtime_config_r0.json"
    schema_path = MODULE / "output_schema_r0.json"
    implementation_lock_path = MODULE / "IMPLEMENTATION_LOCK_R0.json"
    config = load_json(config_path)
    output_schema = load_json(schema_path)
    contract = load_json(repo_root / config["protocol_contract"])
    _, implementation_lock_sha256 = _verify_implementation_lock(
        repo_root,
        implementation_lock_path,
    )
    activation_sha256 = _verify_activation(
        arguments.activation_lock.resolve(),
        implementation_lock_sha256,
    )
    bindings = _verify_bound_inputs(repo_root, config, contract)
    output = repo_root / config["canonical_output"]
    if output.exists():
        raise ValueError("CANONICAL_OUTPUT_ALREADY_EXISTS")
    claim = repo_root / config["canonical_claim"]
    failure_receipt = repo_root / config["canonical_failure_receipt"]
    if claim.exists() or failure_receipt.exists():
        raise ValueError("FORMAL_EXECUTION_ATTEMPT_ALREADY_CONSUMED")
    output.parent.mkdir(parents=True, exist_ok=True)
    claim_payload = canonical_bytes(
        {
            "schema_version": (
                "rcle.real_data_geometry_canary.run_claim.v1"
            ),
            "protocol_id": contract["protocol_id"],
            "implementation_lock_sha256": implementation_lock_sha256,
            "activation_lock_sha256": activation_sha256,
            "canonical_output": config["canonical_output"],
            "bindings": bindings,
        }
    )
    _exclusive_write(claim, claim_payload)
    archive = repo_root / config["source_archive"]
    parent = output.parent
    temporary = parent / f".formal_run_r0.{uuid4().hex}.tmp"
    try:
        producer_rows, producer_summaries = produce_archive(
            archive,
            contract,
            config,
            output_schema,
        )
        temporary.mkdir()
        pair_bytes = _pair_ledger_bytes(producer_rows)
        summary_bytes = canonical_bytes(producer_summaries)
        (temporary / "pair_ledger.jsonl").write_bytes(pair_bytes)
        (temporary / "window_summary.json").write_bytes(summary_bytes)
        receipt = {
            "schema_version": (
                "rcle.real_data_geometry_canary.receipt.v1"
            ),
            "protocol_id": contract["protocol_id"],
            **bindings,
            "implementation_lock_sha256": implementation_lock_sha256,
            "pair_ledger_sha256": sha256(pair_bytes).hexdigest(),
            "window_summary_sha256": sha256(summary_bytes).hexdigest(),
            "pair_record_count": len(producer_rows),
            "window_count": len(producer_summaries),
        }
        receipt_bytes = canonical_bytes(receipt)
        (temporary / "receipt.json").write_bytes(receipt_bytes)
        materialized_rows = read_pair_ledger(
            temporary / "pair_ledger.jsonl"
        )
        materialized_summaries = json.loads(
            (temporary / "window_summary.json").read_text(
                encoding="utf-8"
            )
        )
        validation = validate_materialized(
            archive,
            contract,
            config,
            output_schema,
            materialized_rows,
            materialized_summaries,
            enforce_frozen_counts=True,
        )
        receipt_errors = validate_bound_receipt_files(
            repo_root,
            config,
            contract,
            receipt,
            pair_bytes,
            summary_bytes,
            output_schema,
            implementation_lock_sha256,
        )
        if receipt_errors:
            validation["gate_pass"] = False
            validation["errors"].extend(receipt_errors)
            validation["first_mismatch"] = (
                validation["first_mismatch"]
                or receipt_errors[0]
            )
            validation["terminal"] = contract["result_model"][
                "nonpass_terminal"
            ]
        (temporary / "validation.json").write_bytes(
            canonical_bytes(validation)
        )
        temporary.replace(output)
    except BaseException as error:
        shutil.rmtree(temporary, ignore_errors=True)
        failure_payload = canonical_bytes(
            {
                "schema_version": (
                    "rcle.real_data_geometry_canary."
                    "failure_receipt.v1"
                ),
                "protocol_id": contract["protocol_id"],
                "run_claim_sha256": sha256(
                    claim_payload
                ).hexdigest(),
                "error_type": type(error).__name__,
                "error": str(error),
                "terminal": contract["result_model"][
                    "invalid_terminal"
                ],
            }
        )
        if not failure_receipt.exists():
            _exclusive_write(failure_receipt, failure_payload)
        raise
    print(
        json.dumps(
            {
                "output": str(output),
                "terminal": validation["terminal"],
                "gate_pass": validation["gate_pass"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
