"""Validate the non-execution R0-audit/R1-repair implementation lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_repair_implementation_lock.v1"
LOCK_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_"
    "CALIBRATION_CONTROL_R0_FAILURE_AUDIT_AND_R1_PROTOCOL_REPAIR_LOCK"
)
SUCCESSOR = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_"
    "CALIBRATION_CONTROL_R1_ONE_SHOT_EXECUTION_LOCK"
)
EXPECTED_IMPLEMENTATION = {
    "R1_EXECUTION_CONTRACT": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/contract.py",
    "R1_CONTROL_FORMAT": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/control_format_r1.py",
    "R1_CONTROL_PRODUCER": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/calibration_control_r1.py",
    "R1_INDEPENDENT_VALIDATOR": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/validate_calibration_control_r1.py",
    "R1_REPAIR_LOCK_VALIDATOR": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/validate_calibration_control_r1_repair_lock.py",
    "R1_SYNTHETIC_TEST": "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_calibration_control_r1.py",
    "R1_EXECUTION_CONTRACT_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_contract_and_evidence.py"
    ),
    "R1_HISTORICAL_LOCK_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_implementation_lock.py"
    ),
    "R1_LEGACY_CONTROL_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_calibration_control.py"
    ),
}
EXPECTED_PREDECESSORS = {
    "CONTROL_FORMAT_AND_RUNTIME_REPAIR_IMPLEMENTATION_LOCK": (
        "docs/research/assistive-geometry/"
        "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
        "CONFIRMATION_CONTROL_FORMAT_AND_RUNTIME_BINDING_REPAIR_IMPLEMENTATION_LOCK_2026-08-12.json"
    ),
    "R0_CONSUMED_CONTROL_TERMINAL": (
        "docs/research/assistive-geometry/"
        "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
        "CONFIRMATION_CALIBRATION_CONTROL_PREFLIGHT_ONE_SHOT_RESULT_2026-08-13.json"
    ),
    "R0_FAILURE_AUDIT": (
        "docs/research/assistive-geometry/"
        "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
        "CONFIRMATION_CALIBRATION_CONTROL_R0_FAILURE_AUDIT_2026-08-13.json"
    ),
    "R1_OFFICIAL_CAMERA_SELECTION_EVIDENCE": (
        "docs/research/assistive-geometry/"
        "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
        "CONFIRMATION_CALIBRATION_CONTROL_R1_OFFICIAL_CAMERA_SELECTION_EVIDENCE_2026-08-13.json"
    ),
    "R1_PROTOCOL_AMENDMENT": (
        "docs/research/assistive-geometry/"
        "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
        "CONFIRMATION_CALIBRATION_CONTROL_R1_PROTOCOL_AMENDMENT_2026-08-13.json"
    ),
}
SUPERSEDED_LEGACY_BINDINGS = {
    "EXECUTION_CONTRACT_V2",
    "EXECUTION_CONTRACT_TEST",
    "REPAIR_IMPLEMENTATION_LOCK_TEST",
    "CONTROL_TEST",
}


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValidationError(code)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _bindings(rows: object, expected: Mapping[str, str], repo_root: Path, code: str) -> None:
    _require(isinstance(rows, list) and len(rows) == len(expected), f"{code}_COUNT")
    found: dict[str, str] = {}
    for row in rows:
        _require(
            isinstance(row, Mapping) and set(row) == {"role", "path", "bytes", "sha256"},
            f"{code}_SCHEMA",
        )
        role = row["role"]
        _require(isinstance(role, str) and role not in found, f"{code}_ROLE")
        expected_path = expected.get(role)
        _require(expected_path is not None and row["path"] == expected_path, f"{code}_PATH")
        path = (repo_root / expected_path).resolve()
        _require(
            path.is_file()
            and type(row["bytes"]) is int
            and path.stat().st_size == row["bytes"]
            and _sha(path) == str(row["sha256"]).upper(),
            f"{code}_HASH",
        )
        found[role] = expected_path
    _require(found == dict(expected), f"{code}_SET")


def _validate_preserved_legacy_runtime(path: Path, repo_root: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    _require(
        document.get("schema")
        == "blindassist.ag.r2.cross_sensor_factor_confirmation_control_repair_implementation_lock.v1"
        and document.get("status") == "IMPLEMENTATION_LOCK_PASS_SYNTHETIC_CONTROL_ONLY_SCIENTIFIC_NOT_RUN",
        "F2_R1_REPAIR_LEGACY_LOCK",
    )
    rows = document.get("implementation_bindings")
    _require(isinstance(rows, list), "F2_R1_REPAIR_LEGACY_BINDINGS")
    observed: set[str] = set()
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == {"role", "path", "bytes", "sha256"}, "F2_R1_REPAIR_LEGACY_ROW")
        role = str(row["role"])
        _require(role not in observed, "F2_R1_REPAIR_LEGACY_ROLE")
        observed.add(role)
        if role in SUPERSEDED_LEGACY_BINDINGS:
            continue
        member = (repo_root / str(row["path"])).resolve()
        _require(
            member.is_file()
            and member.stat().st_size == row["bytes"]
            and _sha(member) == str(row["sha256"]).upper(),
            "F2_R1_REPAIR_LEGACY_HASH",
        )
    _require(
        SUPERSEDED_LEGACY_BINDINGS.issubset(observed)
        and {"EXECUTOR_V2", "MODEL_ONLY_PREDICTOR", "ETH3D_SOURCE_ADAPTER_V2"}.issubset(observed),
        "F2_R1_REPAIR_LEGACY_ROLE_SET",
    )


def validate_lock_file(path: Path, repo_root: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    _require(
        set(document)
        == {
            "schema", "lock_id", "date", "status", "predecessor_bindings", "implementation_bindings",
            "test_receipt", "access_receipt", "execution_authority", "unique_successor", "claim_ceiling",
        },
        "F2_R1_REPAIR_LOCK_KEY_SET",
    )
    _require(document["schema"] == SCHEMA, "F2_R1_REPAIR_LOCK_SCHEMA")
    _require(document["lock_id"] == LOCK_ID, "F2_R1_REPAIR_LOCK_ID")
    _require(document["status"] == "R1_REPAIR_IMPLEMENTATION_LOCK_PASS_SYNTHETIC_ONLY_SCIENTIFIC_NOT_RUN", "F2_R1_REPAIR_LOCK_STATUS")
    _bindings(document["predecessor_bindings"], EXPECTED_PREDECESSORS, repo_root, "F2_R1_REPAIR_PREDECESSOR")
    legacy_path = repo_root / EXPECTED_PREDECESSORS["CONTROL_FORMAT_AND_RUNTIME_REPAIR_IMPLEMENTATION_LOCK"]
    _validate_preserved_legacy_runtime(legacy_path.resolve(), repo_root)
    amendment_path = repo_root / EXPECTED_PREDECESSORS["R1_PROTOCOL_AMENDMENT"]
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    hardening = amendment.get("pre_execution_validator_hardening")
    _require(
        amendment.get("status") == "R1_PROTOCOL_REPAIR_FROZEN_NOT_AUTHORIZED_NOT_RUN"
        and isinstance(hardening, Mapping)
        and hardening.get("detected_before_r1_execution_lock") is True
        and hardening.get("detected_before_r1_evidence_root_creation") is True
        and hardening.get("real_archive_access_during_hardening") is False
        and hardening.get("scientific_contract_changed") is False
        and hardening.get("selection_rule_changed") is False
        and hardening.get("data_identity_changed") is False
        and hardening.get("budget_changed") is False
        and hardening.get("session_model_truth_scoring_or_confirmation_authorized") is False
        and isinstance(amendment.get("execution_authority"), Mapping)
        and all(value is False for value in amendment["execution_authority"].values()),
        "F2_R1_REPAIR_AMENDMENT_SEMANTICS",
    )
    _bindings(document["implementation_bindings"], EXPECTED_IMPLEMENTATION, repo_root, "F2_R1_REPAIR_IMPLEMENTATION")
    receipt = document["test_receipt"]
    _require(
        isinstance(receipt, Mapping)
        and type(receipt.get("focused_test_count")) is int
        and receipt["focused_test_count"] > 0
        and receipt.get("focused_test_failures") == 0
        and receipt.get("synthetic_archives_only") is True
        and receipt.get("real_archive_access") is False,
        "F2_R1_REPAIR_TEST_RECEIPT",
    )
    access = document["access_receipt"]
    _require(
        isinstance(access, Mapping)
        and access.get("sealed_r0_evidence_files_read") == 3
        and all(
            access.get(name) == 0
            for name in (
                "real_archive_file_reads",
                "archive_member_enumerations",
                "archive_member_reads",
                "session_archive_reads",
                "model_or_checkpoint_reads",
                "source_truth_materializations",
                "factor_scoring_runs",
                "confirmation_runs",
                "confirmation_roots_created",
            )
        ),
        "F2_R1_REPAIR_ACCESS_RECEIPT",
    )
    authority = document["execution_authority"]
    _require(isinstance(authority, Mapping) and authority and all(value is False for value in authority.values()), "F2_R1_REPAIR_AUTHORITY")
    _require(
        document["unique_successor"]
        == {
            "lock_id": SUCCESSOR,
            "requires_separate_user_authorization": True,
            "requires_new_hash_bound_lock": True,
            "r1_calibration_archive_only": True,
            "execution_authority": False,
            "created_by_this_lock": False,
        },
        "F2_R1_REPAIR_SUCCESSOR",
    )
    return {
        "valid": True,
        "implementation_binding_count": len(EXPECTED_IMPLEMENTATION),
        "predecessor_binding_count": len(EXPECTED_PREDECESSORS),
        "focused_test_count": receipt["focused_test_count"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = validate_lock_file(args.lock.resolve(), args.repo_root.resolve())
    except Exception as error:  # noqa: BLE001 - standalone validator is fail closed.
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
