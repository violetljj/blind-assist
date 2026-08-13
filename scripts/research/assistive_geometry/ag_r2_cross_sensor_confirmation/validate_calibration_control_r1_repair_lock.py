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
    "R1_CONTROL_FORMAT": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/control_format_r1.py",
    "R1_CONTROL_PRODUCER": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/calibration_control_r1.py",
    "R1_INDEPENDENT_VALIDATOR": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/validate_calibration_control_r1.py",
    "R1_REPAIR_LOCK_VALIDATOR": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/validate_calibration_control_r1_repair_lock.py",
    "R1_SYNTHETIC_TEST": "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_calibration_control_r1.py",
}
EXPECTED_PREDECESSORS = {
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
