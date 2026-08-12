"""Fail-closed validator for the AG R2 executor implementation lock.

This validator reads tracked control files only.  It never opens an ETH3D
archive, loads a checkpoint, creates an evidence root, or executes a model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_executor_implementation_lock.v1"
LOCK_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_EXECUTOR_IMPLEMENTATION_LOCK"
)
PROTOCOL_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_R0"
SUCCESSOR_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_ONE_SHOT_EXECUTION_LOCK"
)
STATUS = "IMPLEMENTATION_LOCK_PASS_SYNTHETIC_ONLY_SCIENTIFIC_NOT_RUN"
CLAIM_CEILING = (
    "This lock freezes a synthetic-tested executor implementation only. It is not archive access, "
    "model inference, a scientific Confirmation result, deployment, product, or safety evidence."
)

EXPECTED_TOP_KEYS = {
    "schema",
    "lock_id",
    "date",
    "research_mode",
    "status",
    "predecessor_bindings",
    "frozen_implementation_contract",
    "implementation_bindings",
    "test_receipt",
    "payload_access_receipt",
    "execution_authority",
    "unique_successor",
    "unresolved_execution_preconditions",
    "claim_ceiling",
}
EXPECTED_PREDECESSORS = {
    "PROTOCOL_LOCK": {
        "path": (
            "docs/research/assistive-geometry/"
            "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
            "CONFIRMATION_LOCK_2026-08-12.json"
        ),
        "bytes": 43919,
        "sha256": "8BA036E617531AE886BAAC8DAD60E5445BF8F0F7A2A073B7F8909750478D709F",
    },
    "DATA_IDENTITY_LOCK": {
        "path": (
            "docs/research/assistive-geometry/"
            "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
            "CONFIRMATION_DATA_IDENTITY_2026-08-12.json"
        ),
        "bytes": 8318,
        "sha256": "E755288202F4E7189538671F5F8C120F9D6EF68EBE80757844BC5272382B345B",
    },
}
EXPECTED_IMPLEMENTATION_PATHS = {
    "PACKAGE_INIT": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/__init__.py",
    "EXECUTION_CONTRACT": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/contract.py",
    "ETH3D_SOURCE_ADAPTER": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/eth3d_source.py",
    "EVIDENCE_WRITER": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/evidence.py",
    "INDEPENDENT_VALIDATOR": (
        "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/independent_validator.py"
    ),
    "FACTOR_METRICS": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/metrics.py",
    "MODEL_ONLY_PREDICTOR": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/model_only.py",
    "EXECUTOR": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/runner.py",
    "SOURCE_GEOMETRY": (
        "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/source_geometry.py"
    ),
    "IMPLEMENTATION_LOCK_VALIDATOR": (
        "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/"
        "validate_implementation_lock.py"
    ),
    "CLI": (
        "scripts/research/assistive_geometry/"
        "run_ag_r2_cross_sensor_factor_accuracy_confirmation.py"
    ),
    "CONTRACT_EVIDENCE_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/"
        "test_contract_and_evidence.py"
    ),
    "ETH3D_SOURCE_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/"
        "test_eth3d_source.py"
    ),
    "INDEPENDENT_VALIDATOR_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/"
        "test_independent_validator.py"
    ),
    "METRICS_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_metrics.py"
    ),
    "MODEL_ONLY_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/test_model_only.py"
    ),
    "RUNNER_FIREWALL_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/"
        "test_runner_firewall.py"
    ),
    "SOURCE_GEOMETRY_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/"
        "test_source_geometry.py"
    ),
    "IMPLEMENTATION_LOCK_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/"
        "test_implementation_lock.py"
    ),
}
EXPECTED_AUTHORITY = {
    "archive_member_access": False,
    "real_archive_enumeration": False,
    "model_inference": False,
    "source_truth_read": False,
    "confirmation_scoring": False,
    "evidence_root_creation": False,
    "training": False,
    "reducer": False,
    "network": False,
    "device": False,
    "app": False,
    "product": False,
    "safety": False,
}
EXPECTED_PAYLOAD_RECEIPT = {
    "real_archive_bytes_read": 0,
    "archive_member_reads": 0,
    "archive_member_enumerations": 0,
    "model_inference_runs": 0,
    "confirmation_runs": 0,
    "evidence_roots_created": 0,
    "network_requests": 0,
    "data_identity_manifest_unchanged": True,
    "data_identity_manifest_sha256": (
        "E755288202F4E7189538671F5F8C120F9D6EF68EBE80757844BC5272382B345B"
    ),
}
EXPECTED_FROZEN_CONTRACT = {
    "protocol_id": PROTOCOL_ID,
    "parent_ids": ["mannequin_5", "motion_1", "plant_scene_2"],
    "calibration_frames_per_parent": 12,
    "score_frames_per_parent": 12,
    "roster_rule": "METADATA_ONLY_SHA256_RANK_LOWEST_12_CALIBRATION_NEXT_12_SCORE",
    "raw_model_inputs": ["RGB_UINT8", "K_FLOAT64"],
    "learned_factor_families": ["depth", "support", "obstacle", "boundary"],
    "source_context_only": [
        "K",
        "gravity_up_camera_xyz",
        "support_normal_camera_xyz",
        "camera_height_m",
        "session_metric_scale",
    ],
    "phase_order": [
        "VERIFY_OPAQUE_ARCHIVE_BINDINGS",
        "CONSUME_EXCLUSIVE_EVIDENCE_ROOT",
        "ENUMERATE_BOUND_ARCHIVE_MEMBERS",
        "FREEZE_12_PLUS_12_ROSTER",
        "SEAL_AND_RELOAD_ALL_RAW_SCORE_RGBK_PREDICTIONS",
        "OPEN_CALIBRATION_ROLE_SOURCE_GEOMETRY",
        "FREEZE_ONE_SESSION_CONTEXT_PER_PARENT",
        "SEAL_AND_RELOAD_ALL_CONDITIONED_FACTORS",
        "OPEN_SCORE_ROLE_SOURCE_TRUTH",
        "FACTOR_ONLY_SCORE",
        "INDEPENDENT_VALIDATION",
    ],
    "unknown_semantics": "UNKNOWN_IS_NAN_NEVER_NEGATIVE",
    "support_uncertainty_target": "SIGNED_POINT_TO_FROZEN_SUPPORT_PLANE_RESIDUAL_METERS",
    "uncertainty_rank_order": "TEN_EQUAL_COUNT_STRATA_SIGMA_FRAME_ID_FLAT_INDEX_AVERAGE_RANKS",
    "gate_ids": [f"G{index:02d}" for index in range(1, 28)],
    "reducer_import_allowed": False,
    "independent_validator_constraints": {
        "imports_producer": False,
        "imports_source_adapter": False,
        "imports_recipe": False,
        "imports_metrics": False,
        "imports_reducer": False,
        "manifest_mutation_fail_closed": True,
    },
}
EXPECTED_TEST_RECEIPT = {
    "focused_test_count": 45,
    "focused_test_failures": 0,
    "synthetic_and_metadata_only": True,
    "real_archive_access": False,
    "commands": [
        (
            "E:/codex-tools/tools/venvs/blindassist-venv-export312/Scripts/python.exe -m pytest "
            "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation -q"
        ),
        (
            "E:/codex-tools/tools/venvs/blindassist-venv-export312/Scripts/python.exe -m ruff check "
            "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation "
            "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation "
            "scripts/research/assistive_geometry/run_ag_r2_cross_sensor_factor_accuracy_confirmation.py"
        ),
        (
            "E:/codex-tools/tools/venvs/blindassist-venv-export312/Scripts/python.exe -m compileall -q "
            "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation "
            "scripts/research/assistive_geometry/run_ag_r2_cross_sensor_factor_accuracy_confirmation.py"
        ),
        "git diff --check",
    ],
}
EXPECTED_EXECUTION_PRECONDITIONS = [
    (
        "The one-shot execution lock must freeze the exact official camera-IMU calibration member, "
        "encoding, camera_from_imu key, and mocap scale/anchor/offset keys; any unrecognized layout "
        "fails closed without guessing."
    ),
    (
        "The one-shot execution lock must freeze an official or independently validated ETH3D IMU "
        "column, axis, and accelerometer specific-force sign convention before gravity is admissible."
    ),
]


class ImplementationLockError(ValueError):
    """Raised when any implementation-lock invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImplementationLockError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _resolve_tracked(repo_root: Path, relative: str) -> Path:
    candidate = (repo_root / relative).resolve(strict=True)
    root = repo_root.resolve(strict=True)
    _require(candidate.is_relative_to(root), f"binding escapes repository: {relative}")
    _require(candidate.is_file(), f"binding is not a file: {relative}")
    return candidate


def _validate_binding_set(
    bindings: Any,
    expected_paths: dict[str, str],
    repo_root: Path,
    *,
    exact_metadata: dict[str, dict[str, Any]] | None = None,
) -> None:
    _require(isinstance(bindings, list), "bindings must be a list")
    by_role: dict[str, Any] = {}
    for binding in bindings:
        _require(isinstance(binding, dict), "binding entry must be an object")
        _require(
            set(binding) == {"role", "path", "bytes", "sha256"},
            "binding keys are not exact",
        )
        role = binding["role"]
        _require(isinstance(role, str) and role not in by_role, "binding role is invalid or duplicated")
        by_role[role] = binding
    _require(set(by_role) == set(expected_paths), "binding role set mismatch")

    for role, expected_path in expected_paths.items():
        binding = by_role[role]
        _require(binding["path"] == expected_path, f"{role} path mismatch")
        path = _resolve_tracked(repo_root, expected_path)
        _require(type(binding["bytes"]) is int, f"{role} bytes must be an integer")
        _require(binding["bytes"] == path.stat().st_size, f"{role} byte count mismatch")
        _require(binding["sha256"] == _sha256(path), f"{role} sha256 mismatch")
        if exact_metadata is not None:
            expected = exact_metadata[role]
            _require(binding["bytes"] == expected["bytes"], f"{role} frozen byte count mismatch")
            _require(binding["sha256"] == expected["sha256"], f"{role} frozen sha256 mismatch")


def validate_lock_document(document: Any, repo_root: Path) -> dict[str, Any]:
    """Validate a parsed implementation-lock document and all tracked bindings."""

    _require(isinstance(document, dict), "lock root must be an object")
    _require(set(document) == EXPECTED_TOP_KEYS, "top-level keys are not exact")
    _require(document["schema"] == SCHEMA, "schema mismatch")
    _require(document["lock_id"] == LOCK_ID, "lock_id mismatch")
    _require(document["date"] == "2026-08-12", "date mismatch")
    _require(document["research_mode"] == "WILD_LAB", "research mode mismatch")
    _require(document["status"] == STATUS, "status mismatch")
    _require(document["claim_ceiling"] == CLAIM_CEILING, "claim ceiling mismatch")

    predecessor_paths = {role: value["path"] for role, value in EXPECTED_PREDECESSORS.items()}
    _validate_binding_set(
        document["predecessor_bindings"],
        predecessor_paths,
        repo_root,
        exact_metadata=EXPECTED_PREDECESSORS,
    )
    _validate_binding_set(
        document["implementation_bindings"], EXPECTED_IMPLEMENTATION_PATHS, repo_root
    )

    _require(
        document["frozen_implementation_contract"] == EXPECTED_FROZEN_CONTRACT,
        "frozen implementation contract mismatch",
    )
    _require(document["test_receipt"] == EXPECTED_TEST_RECEIPT, "test receipt mismatch")

    _require(document["payload_access_receipt"] == EXPECTED_PAYLOAD_RECEIPT, "payload access receipt mismatch")
    _require(document["execution_authority"] == EXPECTED_AUTHORITY, "execution authority mismatch")
    _require(
        document["unique_successor"]
        == {
            "lock_id": SUCCESSOR_ID,
            "requires_separate_user_authorization": True,
            "requires_new_hash_bound_lock": True,
            "execution_authority": False,
            "created_by_this_lock": False,
        },
        "unique successor mismatch",
    )

    _require(
        document["unresolved_execution_preconditions"] == EXPECTED_EXECUTION_PRECONDITIONS,
        "execution preconditions mismatch",
    )

    return {
        "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_implementation_validation.v1",
        "lock_id": LOCK_ID,
        "terminal": STATUS,
        "implementation_binding_count": len(EXPECTED_IMPLEMENTATION_PATHS),
        "execution_authority": False,
        "scientific_confirmation": "NOT_RUN",
    }


def validate_lock_file(lock_path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    """Load and validate an implementation-lock JSON file."""

    path = lock_path.resolve(strict=True)
    root = repo_root.resolve(strict=True) if repo_root is not None else Path(__file__).resolve().parents[4]
    document = json.loads(path.read_text(encoding="utf-8"))
    return validate_lock_document(copy.deepcopy(document), root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(validate_lock_file(args.lock), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
