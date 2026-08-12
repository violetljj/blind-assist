"""Standalone validator for the AG R2 calibration/runtime repair lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

LOCK_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_control_repair_implementation_lock.v1"
LOCK_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_"
    "CONTROL_FORMAT_AND_RUNTIME_BINDING_REPAIR_IMPLEMENTATION_LOCK"
)
SUCCESSOR_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_"
    "CALIBRATION_CONTROL_PREFLIGHT_ONE_SHOT_EXECUTION_LOCK"
)
EXPECTED_BINDING_PATHS = {
    "PACKAGE_INIT": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/__init__.py",
    "EXECUTION_CONTRACT_V2": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/contract.py",
    "KALIBR_CONTROL_FORMAT": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/control_format.py",
    "ETH3D_SOURCE_ADAPTER_V2": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/eth3d_source.py",
    "CALIBRATION_CONTROL_PREFLIGHT": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/calibration_control.py",
    "CALIBRATION_CONTROL_INDEPENDENT_VALIDATOR": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/validate_calibration_control.py",
    "DEPTHART_SOURCE_MANIFEST_INDEPENDENT_VALIDATOR": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/validate_depthart_source_manifest.py",
    "MODEL_ONLY_PREDICTOR": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/model_only.py",
    "EXECUTOR_V2": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/runner.py",
    "REPAIR_IMPLEMENTATION_LOCK_VALIDATOR": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/validate_repair_implementation_lock.py",
    "CLI": "scripts/research/assistive_geometry/run_ag_r2_cross_sensor_factor_accuracy_confirmation.py",
    "OFFICIAL_CONTROL_EVIDENCE": "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_OFFICIAL_CONTROL_EVIDENCE_2026-08-12.json",
    "DEPTHART_SOURCE_MANIFEST": "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_DEPTHART_SOURCE_MANIFEST_2026-08-12.json",
    "CONTROL_TEST": "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_calibration_control.py",
    "SOURCE_MANIFEST_TEST": "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_depthart_source_manifest.py",
    "EXECUTION_CONTRACT_TEST": "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_contract_and_evidence.py",
    "ETH3D_SOURCE_TEST": "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_eth3d_source.py",
    "REPAIR_IMPLEMENTATION_LOCK_TEST": "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_implementation_lock.py",
}


class ImplementationLockError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImplementationLockError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImplementationLockError(f"JSON read failed: {path}") from error
    _require(isinstance(value, dict), "document must be an object")
    return value


def _binding(binding: Mapping, root: Path, expected_relative: str, role: str) -> Path:
    _require(set(binding) == {"role", "path", "bytes", "sha256"}, f"{role} binding schema")
    _require(binding["role"] == role and binding["path"] == expected_relative, f"{role} binding identity")
    path = (root / expected_relative).resolve()
    _require(path.is_file(), f"{role} missing")
    _require(type(binding["bytes"]) is int and path.stat().st_size == binding["bytes"], f"{role} byte count mismatch")
    _require(_sha(path) == str(binding["sha256"]).upper(), f"{role} SHA mismatch")
    return path


def _validate_source_manifest(path: Path) -> tuple[int, int]:
    manifest = _load(path)
    _require(set(manifest) == {"schema", "source_root", "files"}, "source manifest schema")
    _require(manifest["schema"] == "blindassist.depthart.source_manifest.v1", "source manifest schema")
    source_root = Path(manifest["source_root"])
    _require(source_root.is_absolute() and source_root.is_dir(), "source manifest root")
    files = manifest["files"]
    _require(isinstance(files, list) and len(files) == 29, "source manifest file count")
    observed: list[str] = []
    total = 0
    for row in files:
        _require(isinstance(row, Mapping) and set(row) == {"path", "bytes", "sha256"}, "source manifest row")
        relative = row["path"]
        parsed = PurePosixPath(relative)
        _require(isinstance(relative, str) and parsed.as_posix() == relative and ".." not in parsed.parts, "source manifest path")
        member = source_root.joinpath(*parsed.parts)
        _require(member.is_file() and member.stat().st_size == row["bytes"], "source manifest member bytes")
        _require(_sha(member) == str(row["sha256"]).upper(), "source manifest member SHA")
        observed.append(relative)
        total += row["bytes"]
    _require(observed == sorted(observed) and len(set(observed)) == 29, "source manifest exact ordered set")
    return len(files), total


def validate_lock_document(document: Mapping, root: Path) -> dict:
    _require(
        set(document) == {
            "schema", "lock_id", "date", "research_mode", "status", "predecessor_bindings",
            "implementation_bindings", "runtime_source_manifest_receipt", "test_receipt",
            "access_receipt", "execution_authority", "unique_successor", "claim_ceiling",
        },
        "repair lock key set",
    )
    _require(document["schema"] == LOCK_SCHEMA and document["lock_id"] == LOCK_ID, "repair lock identity")
    _require(document["date"] == "2026-08-12" and document["research_mode"] == "WILD_LAB", "repair lock date or mode")
    _require(document["status"] == "IMPLEMENTATION_LOCK_PASS_SYNTHETIC_CONTROL_ONLY_SCIENTIFIC_NOT_RUN", "repair lock status")
    predecessors = document["predecessor_bindings"]
    _require(isinstance(predecessors, list) and len(predecessors) == 2, "predecessor binding count")
    expected_predecessors = {
        "EXECUTOR_IMPLEMENTATION_LOCK": "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_EXECUTOR_IMPLEMENTATION_LOCK_2026-08-12.json",
        "ACTIVATION_PREFLIGHT_BLOCKER": "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_ONE_SHOT_EXECUTION_ACTIVATION_PREFLIGHT_RESULT_2026-08-12.json",
    }
    for row in predecessors:
        role = row.get("role")
        _require(role in expected_predecessors, "predecessor role")
        path = _binding(row, root, expected_predecessors[role], role)
        value = _load(path)
        if role == "ACTIVATION_PREFLIGHT_BLOCKER":
            _require(value.get("decision", {}).get("execution_lock_created") is False, "blocker execution lock state")
            _require(value.get("decision", {}).get("one_shot_consumed") is False, "blocker one-shot state")
    rows = document["implementation_bindings"]
    _require(isinstance(rows, list) and len(rows) == len(EXPECTED_BINDING_PATHS), "implementation binding count")
    observed_roles: set[str] = set()
    bound: dict[str, Path] = {}
    for row in rows:
        role = row.get("role")
        _require(role in EXPECTED_BINDING_PATHS and role not in observed_roles, "implementation binding role")
        observed_roles.add(role)
        bound[role] = _binding(row, root, EXPECTED_BINDING_PATHS[role], role)
    _require(observed_roles == set(EXPECTED_BINDING_PATHS), "implementation binding role set")
    official = _load(bound["OFFICIAL_CONTROL_EVIDENCE"])
    _require(official.get("schema") == "blindassist.ag.r2.cross_sensor_factor_confirmation_official_control_evidence.v1", "official control schema")
    _require(official.get("selection_boundary", {}).get("guessing_member_or_camera_node_allowed") is False, "official control guess boundary")
    file_count, total_bytes = _validate_source_manifest(bound["DEPTHART_SOURCE_MANIFEST"])
    _require(
        document["runtime_source_manifest_receipt"] == {
            "schema": "blindassist.depthart.source_manifest.v1",
            "file_count": file_count,
            "total_bytes": total_bytes,
            "independent_validation": "PASS",
            "checkpoint_or_model_loaded": False,
        },
        "runtime source manifest receipt",
    )
    tests = document["test_receipt"]
    _require(
        isinstance(tests, Mapping)
        and tests.get("focused_test_count") == 51
        and tests.get("focused_test_failures") == 0
        and tests.get("synthetic_control_only") is True
        and tests.get("real_archive_access") is False,
        "test receipt",
    )
    _require(
        document["access_receipt"] == {
            "depthart_source_files_hashed": 29,
            "depthart_source_bytes_hashed": 160284,
            "checkpoint_reads": 0,
            "real_archive_file_content_reads": 0,
            "archive_member_enumerations": 0,
            "archive_member_reads": 0,
            "model_inference_runs": 0,
            "source_truth_materializations": 0,
            "factor_scoring_runs": 0,
            "confirmation_runs": 0,
            "confirmation_evidence_roots_created": 0,
            "network_requests": 0,
        },
        "access receipt",
    )
    _require(
        document["execution_authority"] == {
            "calibration_archive_access": False,
            "session_archive_access": False,
            "model_inference": False,
            "source_truth_read": False,
            "confirmation_scoring": False,
            "confirmation_evidence_root_creation": False,
            "training": False,
            "reducer": False,
            "network": False,
            "device": False,
            "app": False,
            "product": False,
            "safety": False,
        },
        "execution authority",
    )
    _require(
        document["unique_successor"] == {
            "lock_id": SUCCESSOR_ID,
            "requires_separate_user_authorization": True,
            "requires_new_hash_bound_lock": True,
            "calibration_archive_only": True,
            "execution_authority": False,
            "created_by_this_lock": False,
        },
        "unique successor",
    )
    _require(isinstance(document["claim_ceiling"], str) and "not" in document["claim_ceiling"].lower(), "claim ceiling")
    return {"valid": True, "lock_id": LOCK_ID, "implementation_binding_count": len(rows), "depthart_source_file_count": file_count}


def validate_lock_file(path: Path, root: Path) -> dict:
    return validate_lock_document(_load(path), root.resolve())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_lock_file(args.lock, args.repo_root)
    except ImplementationLockError as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
