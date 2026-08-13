"""Independent one-shot replay and offline verifier for R1 calibration control.

The formal CLI first consumes an exclusive validator receipt, then performs
exactly one calibration-archive replay.  A later ``validate`` call verifies
only the sealed local receipt chain and never reopens the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import unicodedata
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
RESULT_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_result.v1"
FAILURE_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_failure.v1"
CONTROL_LOCK_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_lock.v1"
CONTROL_LOCK_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_CALIBRATION_CONTROL_R1_ONE_SHOT_EXECUTION_LOCK"
)
CONTROL_STATUS = "ONE_SHOT_CALIBRATION_CONTROL_R1_AUTHORIZED_NOT_STARTED"
PROTOCOL_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_R0"
EXPECTED_IMU_ROSTOPIC = "/uvc_camera/cam_2/imu"
EXPECTED_NAMESPACE = "/uvc_camera/cam_2"
REPLAY_START_SCHEMA = (
    "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_validator_start.v1"
)
REPLAY_RESULT_SCHEMA = (
    "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_validator_result.v1"
)
REPLAY_FAILURE_SCHEMA = (
    "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_validator_failure.v1"
)
REPLAY_MANIFEST_SCHEMA = (
    "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_validator_manifest.v1"
)
REPLAY_PASS = "CALIBRATION_CONTROL_R1_INDEPENDENT_REPLAY_PASS"
REPLAY_CONFIRMED_FAILURE = "CALIBRATION_CONTROL_R1_INDEPENDENT_REPLAY_CONFIRMED_PRODUCER_FAILURE"
REPLAY_FAIL = "CALIBRATION_CONTROL_R1_INDEPENDENT_REPLAY_FAIL_CLOSED"
R1_REPAIR_LOCK_SCHEMA = (
    "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_repair_implementation_lock.v1"
)
R1_REPAIR_LOCK_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_"
    "CALIBRATION_CONTROL_R0_FAILURE_AUDIT_AND_R1_PROTOCOL_REPAIR_LOCK"
)
R1_REPAIR_LOCK_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_CALIBRATION_CONTROL_R0_FAILURE_AUDIT_AND_R1_PROTOCOL_REPAIR_IMPLEMENTATION_LOCK_2026-08-13.json"
)
DATA_IDENTITY_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_DATA_IDENTITY_2026-08-12.json"
)
R1_OFFICIAL_EVIDENCE_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_CALIBRATION_CONTROL_R1_OFFICIAL_CAMERA_SELECTION_EVIDENCE_2026-08-13.json"
)
R1_AMENDMENT_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_CALIBRATION_CONTROL_R1_PROTOCOL_AMENDMENT_2026-08-13.json"
)
R0_TERMINAL_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_CALIBRATION_CONTROL_PREFLIGHT_ONE_SHOT_RESULT_2026-08-13.json"
)
ARCHIVE_ROOT_PATH = REPO_ROOT / "artifacts.local/downloads/ag-r2-eth3d-cross-sensor-confirmation-r0"
OUTPUT_ROOT_PATH = REPO_ROOT / (
    "artifacts.local/evidence/assistive-geometry/ag-r2-cross-sensor-calibration-control-r1"
)
EXPECTED_BUDGET = {
    "max_members": 256,
    "max_member_uncompressed_bytes": 4194304,
    "max_total_uncompressed_bytes": 67108864,
    "max_compression_ratio": 100.0,
    "max_metadata_bytes": 4194304,
    "max_yaml_candidates": 32,
}
EXPECTED_AUTHORITY = {
    "producer_calibration_archive_hash": True,
    "producer_calibration_archive_member_enumeration": True,
    "producer_calibration_yaml_member_read": True,
    "independent_validator_archive_replay": True,
    "session_rgbd_archive_access": False,
    "session_imu_archive_access": False,
    "model_or_checkpoint_access": False,
    "source_truth_materialization": False,
    "factor_scoring": False,
    "confirmation_root": False,
    "training_or_tuning": False,
    "reducer_or_task_state": False,
    "network": False,
    "device": False,
    "default_app": False,
    "product": False,
    "safety": False,
}
EXPECTED_ONE_SHOT = {
    "exclusive_r1_control_root": True,
    "producer_runs": 1,
    "independent_validator_replays": 1,
    "r0_rerun": False,
    "r0_resume": False,
    "r0_replacement": False,
}
EXPECTED_REPAIR_PREDECESSORS = {
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
EXPECTED_REPAIR_IMPLEMENTATION = {
    "R1_EXECUTION_CONTRACT": "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/contract.py",
    "R1_CONTROL_FORMAT": (
        "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/control_format_r1.py"
    ),
    "R1_CONTROL_PRODUCER": (
        "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/calibration_control_r1.py"
    ),
    "R1_INDEPENDENT_VALIDATOR": (
        "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/validate_calibration_control_r1.py"
    ),
    "R1_REPAIR_LOCK_VALIDATOR": (
        "scripts/research/assistive_geometry/ag_r2_cross_sensor_confirmation/"
        "validate_calibration_control_r1_repair_lock.py"
    ),
    "R1_SYNTHETIC_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/"
        "test_calibration_control_r1.py"
    ),
    "R1_EXECUTION_CONTRACT_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/"
        "test_contract_and_evidence.py"
    ),
    "R1_HISTORICAL_LOCK_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/"
        "test_implementation_lock.py"
    ),
    "R1_LEGACY_CONTROL_TEST": (
        "scripts/research/assistive_geometry/tests_ag_r2_cross_sensor_confirmation/"
        "test_calibration_control.py"
    ),
}
SUPERSEDED_LEGACY_BINDINGS = {
    "EXECUTION_CONTRACT_V2",
    "EXECUTION_CONTRACT_TEST",
    "REPAIR_IMPLEMENTATION_LOCK_TEST",
    "CONTROL_TEST",
}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
NODE = re.compile(r"^(?P<indent> *)(?P<name>[A-Za-z][A-Za-z0-9_]*):\s*(?:#.*)?$")
ROW = re.compile(
    rf"^(?P<indent> *)-\s*\[\s*(?P<a>{NUMBER})\s*,\s*(?P<b>{NUMBER})\s*,\s*"
    rf"(?P<c>{NUMBER})\s*,\s*(?P<d>{NUMBER})\s*\]\s*(?:#.*)?$"
)
TOPIC = re.compile(
    r"^(?P<indent> +)rostopic:\s*(?P<value>(?:\"[^\"\r\n]+\"|'[^'\r\n]+'|[^#\s][^#\r\n]*?))\s*(?:#.*)?$"
)
ROS_TOPIC = re.compile(r"^/[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)+$")
WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class ValidationError(RuntimeError):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


def _require(condition: bool, code: str, message: str = "") -> None:
    if not condition:
        raise ValidationError(code, message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _sha_stream(stream: object) -> str:
    digest = hashlib.sha256()
    while True:
        block = stream.read(1024 * 1024)  # type: ignore[attr-defined]
        if not block:
            break
        digest.update(block)
    return digest.hexdigest().upper()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest().upper()


def _load_json(path: Path, code: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(code, str(error)) from error
    _require(isinstance(value, dict), code)
    return value


def _write_exclusive_json(path: Path, value: object) -> None:
    payload = _canonical_bytes(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _verified_binding(lock: Mapping[str, object], key: str, role: str, expected_path: Path) -> Path:
    row = lock.get(key)
    _require(
        isinstance(row, Mapping)
        and set(row) == {"role", "path", "bytes", "sha256"}
        and row.get("role") == role
        and isinstance(row.get("path"), str)
        and Path(str(row["path"])).is_absolute()
        and Path(str(row["path"])) == expected_path
        and type(row.get("bytes")) is int
        and int(row["bytes"]) > 0
        and isinstance(row.get("sha256"), str)
        and len(str(row["sha256"])) == 64,
        f"F2_R1_VALIDATOR_{key.upper()}_BINDING",
    )
    path = Path(str(row["path"])).resolve()
    _require(
        path == expected_path.resolve()
        and path.is_file()
        and path.stat().st_size == row["bytes"]
        and _sha(path) == str(row["sha256"]).upper(),
        f"F2_R1_VALIDATOR_{key.upper()}_HASH",
    )
    return path


def _verify_repair_rows(rows: object, expected: Mapping[str, str], code: str) -> None:
    _require(isinstance(rows, list) and len(rows) == len(expected), f"{code}_COUNT")
    observed: set[str] = set()
    for row in rows:
        _require(
            isinstance(row, Mapping)
            and set(row) == {"role", "path", "bytes", "sha256"}
            and isinstance(row.get("role"), str)
            and row["role"] in expected
            and row["role"] not in observed
            and row.get("path") == expected[row["role"]],
            f"{code}_ROW",
        )
        observed.add(str(row["role"]))
        path = (REPO_ROOT / str(row["path"])).resolve()
        _require(
            path.is_file()
            and type(row.get("bytes")) is int
            and path.stat().st_size == row["bytes"]
            and _sha(path) == str(row.get("sha256", "")).upper(),
            f"{code}_HASH",
        )
    _require(observed == set(expected), f"{code}_SET")


def _verify_preserved_legacy_runtime(path: Path) -> None:
    legacy = _load_json(path, "F2_R1_VALIDATOR_LEGACY_LOCK_READ")
    _require(
        legacy.get("schema")
        == "blindassist.ag.r2.cross_sensor_factor_confirmation_control_repair_implementation_lock.v1"
        and legacy.get("status") == "IMPLEMENTATION_LOCK_PASS_SYNTHETIC_CONTROL_ONLY_SCIENTIFIC_NOT_RUN",
        "F2_R1_VALIDATOR_LEGACY_LOCK",
    )
    rows = legacy.get("implementation_bindings")
    _require(isinstance(rows, list), "F2_R1_VALIDATOR_LEGACY_BINDINGS")
    observed: set[str] = set()
    for row in rows:
        _require(
            isinstance(row, Mapping) and set(row) == {"role", "path", "bytes", "sha256"},
            "F2_R1_VALIDATOR_LEGACY_ROW",
        )
        role = str(row["role"])
        _require(role not in observed, "F2_R1_VALIDATOR_LEGACY_ROLE")
        observed.add(role)
        if role in SUPERSEDED_LEGACY_BINDINGS:
            continue
        member = (REPO_ROOT / str(row["path"])).resolve()
        _require(
            member.is_file()
            and member.stat().st_size == row["bytes"]
            and _sha(member) == str(row["sha256"]).upper(),
            "F2_R1_VALIDATOR_LEGACY_HASH",
        )
    _require(
        SUPERSEDED_LEGACY_BINDINGS.issubset(observed)
        and {"EXECUTOR_V2", "MODEL_ONLY_PREDICTOR", "ETH3D_SOURCE_ADAPTER_V2"}.issubset(observed),
        "F2_R1_VALIDATOR_LEGACY_ROLE_SET",
    )


def _verify_repair_lock(path: Path) -> None:
    repair = _load_json(path, "F2_R1_VALIDATOR_REPAIR_LOCK_READ")
    _require(
        set(repair)
        == {
            "schema",
            "lock_id",
            "date",
            "status",
            "predecessor_bindings",
            "implementation_bindings",
            "test_receipt",
            "access_receipt",
            "execution_authority",
            "unique_successor",
            "claim_ceiling",
        }
        and repair.get("schema") == R1_REPAIR_LOCK_SCHEMA
        and repair.get("lock_id") == R1_REPAIR_LOCK_ID
        and repair.get("status") == "R1_REPAIR_IMPLEMENTATION_LOCK_PASS_SYNTHETIC_ONLY_SCIENTIFIC_NOT_RUN",
        "F2_R1_VALIDATOR_REPAIR_LOCK_SEMANTICS",
    )
    _verify_repair_rows(
        repair.get("predecessor_bindings"),
        EXPECTED_REPAIR_PREDECESSORS,
        "F2_R1_VALIDATOR_REPAIR_PREDECESSOR",
    )
    _verify_preserved_legacy_runtime(
        REPO_ROOT
        / EXPECTED_REPAIR_PREDECESSORS["CONTROL_FORMAT_AND_RUNTIME_REPAIR_IMPLEMENTATION_LOCK"]
    )
    _verify_repair_rows(
        repair.get("implementation_bindings"),
        EXPECTED_REPAIR_IMPLEMENTATION,
        "F2_R1_VALIDATOR_REPAIR_IMPLEMENTATION",
    )
    receipt = repair.get("test_receipt")
    _require(
        isinstance(receipt, Mapping)
        and type(receipt.get("focused_test_count")) is int
        and int(receipt["focused_test_count"]) > 0
        and receipt.get("focused_test_failures") == 0
        and receipt.get("synthetic_archives_only") is True
        and receipt.get("real_archive_access") is False,
        "F2_R1_VALIDATOR_REPAIR_TEST_RECEIPT",
    )
    access = repair.get("access_receipt")
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
        "F2_R1_VALIDATOR_REPAIR_ACCESS_RECEIPT",
    )
    authority = repair.get("execution_authority")
    _require(
        isinstance(authority, Mapping) and authority and all(value is False for value in authority.values()),
        "F2_R1_VALIDATOR_REPAIR_AUTHORITY",
    )
    _require(
        repair.get("unique_successor")
        == {
            "lock_id": CONTROL_LOCK_ID,
            "requires_separate_user_authorization": True,
            "requires_new_hash_bound_lock": True,
            "r1_calibration_archive_only": True,
            "execution_authority": False,
            "created_by_this_lock": False,
        },
        "F2_R1_VALIDATOR_REPAIR_SUCCESSOR",
    )


def _expected_selection_contract() -> dict[str, object]:
    return {
        "dataset_target_viewpoint": "ETH3D_PRIMARY_RGB_DEPTH_RIGHT_RGB_CAMERA",
        "official_target_imu_file": "imu.txt",
        "official_target_imu_rostopic": EXPECTED_IMU_ROSTOPIC,
        "expected_camera_sensor_namespace": EXPECTED_NAMESPACE,
        "camera_identity_field": "same_camera_node.rostopic",
        "namespace_operator": "rostopic.rpartition('/')[0]",
        "required_matrix_key": "T_cam_imu",
        "required_matrix_encoding": "KALIBR_CAMCHAIN_YAML_T_CAM_IMU_NESTED_4X4",
        "required_transform_direction": "IMU_TO_CAMERA_T_CAM_IMU",
        "selection_operator": (
            "Across every bounded YAML candidate, retain every valid camera-node T_cam_imu discovery, "
            "then require exactly one discovery whose same-node rostopic namespace equals /uvc_camera/cam_2."
        ),
        "camera_node_key_is_preselected": False,
        "camchain_order_is_selection_evidence": False,
        "first_or_best_candidate_selection": False,
        "zero_or_multiple_target_matches": "FAIL_CLOSED_WITH_EXACT_COUNT_PRESERVED",
    }


def _load_lock(control_lock: Path) -> tuple[dict[str, object], dict[str, object]]:
    lock = _load_json(control_lock, "F2_R1_VALIDATOR_CONTROL_LOCK_READ")
    _require(
        set(lock)
        == {
            "schema",
            "lock_id",
            "protocol_id",
            "status",
            "repair_implementation_lock",
            "data_identity",
            "official_camera_selection_evidence",
            "protocol_amendment",
            "r0_terminal",
            "archive_root",
            "output_root",
            "budget",
            "authority",
            "one_shot",
        }
        and lock.get("schema") == CONTROL_LOCK_SCHEMA
        and lock.get("lock_id") == CONTROL_LOCK_ID
        and lock.get("protocol_id") == PROTOCOL_ID
        and lock.get("status") == CONTROL_STATUS
        and lock.get("budget") == EXPECTED_BUDGET
        and lock.get("authority") == EXPECTED_AUTHORITY
        and lock.get("one_shot") == EXPECTED_ONE_SHOT,
        "F2_R1_VALIDATOR_CONTROL_LOCK",
    )
    repair_path = _verified_binding(
        lock,
        "repair_implementation_lock",
        "R1_REPAIR_IMPLEMENTATION_LOCK",
        R1_REPAIR_LOCK_PATH,
    )
    identity_path = _verified_binding(lock, "data_identity", "DATA_IDENTITY_PRE_R0_SNAPSHOT", DATA_IDENTITY_PATH)
    official_path = _verified_binding(
        lock,
        "official_camera_selection_evidence",
        "R1_OFFICIAL_CAMERA_SELECTION_EVIDENCE",
        R1_OFFICIAL_EVIDENCE_PATH,
    )
    amendment_path = _verified_binding(lock, "protocol_amendment", "R1_PROTOCOL_AMENDMENT", R1_AMENDMENT_PATH)
    r0_path = _verified_binding(lock, "r0_terminal", "R0_CONSUMED_CONTROL_TERMINAL", R0_TERMINAL_PATH)
    _verify_repair_lock(repair_path)
    identity = _load_json(identity_path, "F2_R1_VALIDATOR_IDENTITY_READ")
    archives = identity.get("archives")
    calibration_rows = [
        row
        for row in archives
        if isinstance(row, Mapping) and row.get("kind") == "CAMERA_IMU_CALIBRATION_ARCHIVE"
    ] if isinstance(archives, list) else []
    _require(identity.get("protocol_id") == PROTOCOL_ID and len(calibration_rows) == 1, "F2_R1_VALIDATOR_IDENTITY")
    binding = dict(calibration_rows[0])
    _require(
        set(binding) == {"parent_id", "kind", "url", "bytes", "sha256"}
        and binding.get("parent_id") == "ALL_THREE_SESSIONS"
        and binding.get("kind") == "CAMERA_IMU_CALIBRATION_ARCHIVE"
        and binding.get("url") == "https://www.eth3d.net/data/slam/camera_imu_calib_radtan.zip"
        and type(binding.get("bytes")) is int
        and int(binding["bytes"]) > 0
        and isinstance(binding.get("sha256"), str)
        and len(str(binding["sha256"])) == 64,
        "F2_R1_VALIDATOR_ARCHIVE_BINDING",
    )
    official = _load_json(official_path, "F2_R1_VALIDATOR_OFFICIAL_READ")
    _require(
        official.get("schema")
        == "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_official_camera_selection_evidence.v1"
        and official.get("selection_contract") == _expected_selection_contract(),
        "F2_R1_VALIDATOR_OFFICIAL_EVIDENCE",
    )
    amendment = _load_json(amendment_path, "F2_R1_VALIDATOR_AMENDMENT_READ")
    hardening = amendment.get("pre_execution_validator_hardening")
    _require(
        amendment.get("status") == "R1_PROTOCOL_REPAIR_FROZEN_NOT_AUTHORIZED_NOT_RUN"
        and isinstance(amendment.get("prior_access_disclosure"), Mapping)
        and amendment["prior_access_disclosure"].get("r0_calibration_archive_member_enumerated") is True
        and isinstance(amendment.get("unchanged_scientific_contract"), Mapping)
        and amendment["unchanged_scientific_contract"].get("scientific_status") == "NOT_RUN"
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
        "F2_R1_VALIDATOR_AMENDMENT",
    )
    r0 = _load_json(r0_path, "F2_R1_VALIDATOR_R0_TERMINAL_READ")
    _require(
        r0.get("status") == "CALIBRATION_CONTROL_FAIL_CLOSED_AMBIGUOUS_OR_MISSING_MATRIX_ONE_SHOT_CONSUMED"
        and isinstance(r0.get("control_outcome"), Mapping)
        and r0["control_outcome"].get("one_shot_consumed") is True,
        "F2_R1_VALIDATOR_R0_TERMINAL",
    )
    _require(Path(str(lock["archive_root"])) == ARCHIVE_ROOT_PATH, "F2_R1_VALIDATOR_ARCHIVE_ROOT_PATH")
    _require(Path(str(lock["output_root"])) == OUTPUT_ROOT_PATH, "F2_R1_VALIDATOR_OUTPUT_ROOT_PATH")
    archive_root = Path(str(lock["archive_root"])).resolve(strict=True)
    output_root = Path(str(lock["output_root"])).resolve(strict=False)
    _require(
        archive_root != output_root
        and archive_root not in output_root.parents
        and output_root not in archive_root.parents,
        "F2_R1_VALIDATOR_ROOT_COLLISION",
    )
    return lock, binding


def _producer_chain(root: Path, control_lock: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    lock, binding = _load_lock(control_lock)
    _require(root == Path(str(lock["output_root"])).resolve(), "F2_R1_VALIDATOR_ROOT_BINDING")
    names = sorted(path.name for path in root.iterdir())
    _require(
        names
        in (
            ["manifest.json", "result.json", "start-receipt.json"],
            ["failure.json", "manifest.json", "start-receipt.json"],
            [
                "manifest.json",
                "result.json",
                "start-receipt.json",
                "validator-manifest.json",
                "validator-result.json",
                "validator-start-receipt.json",
            ],
            [
                "failure.json",
                "manifest.json",
                "start-receipt.json",
                "validator-manifest.json",
                "validator-result.json",
                "validator-start-receipt.json",
            ],
            [
                "failure.json",
                "manifest.json",
                "start-receipt.json",
                "validator-failure.json",
                "validator-manifest.json",
                "validator-start-receipt.json",
            ],
            [
                "manifest.json",
                "result.json",
                "start-receipt.json",
                "validator-failure.json",
                "validator-manifest.json",
                "validator-start-receipt.json",
            ],
        ),
        "F2_R1_VALIDATOR_FILE_SET",
    )
    terminal_name = "result.json" if "result.json" in names else "failure.json"
    terminal = _load_json(root / terminal_name, "F2_R1_VALIDATOR_TERMINAL_READ")
    start = _load_json(root / "start-receipt.json", "F2_R1_VALIDATOR_START_READ")
    manifest = _load_json(root / "manifest.json", "F2_R1_VALIDATOR_MANIFEST_READ")
    _require(
        start
        == {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_start.v1",
            "protocol_id": PROTOCOL_ID,
            "control_lock": {"path": str(control_lock), "sha256": _sha(control_lock)},
            "r0_terminal": {
                "path": lock["r0_terminal"]["path"],  # type: ignore[index]
                "sha256": lock["r0_terminal"]["sha256"],  # type: ignore[index]
            },
            "control_root_consumed_at_start": True,
            "archive_bytes_read_before_start": 0,
            "archive_members_enumerated_before_start": 0,
        },
        "F2_R1_VALIDATOR_START_RECEIPT",
    )
    _require(
        set(manifest)
        == {
            "schema",
            "evidence_root_consumed",
            "terminal",
            "file_count_before_manifest",
            "bytes_before_manifest",
            "files",
        }
        and manifest.get("schema") == "blindassist.ag.r2.cross_sensor_factor_confirmation_manifest.v1"
        and manifest.get("evidence_root_consumed") is True
        and manifest.get("terminal") == terminal.get("status")
        and manifest.get("file_count_before_manifest") == 2
        and isinstance(manifest.get("files"), Mapping)
        and set(manifest["files"]) == {terminal_name, "start-receipt.json"},
        "F2_R1_VALIDATOR_MANIFEST",
    )
    expected_bytes = 0
    for name, row in manifest["files"].items():  # type: ignore[union-attr]
        path = root / str(name)
        _require(
            isinstance(row, Mapping)
            and set(row) == {"path", "bytes", "sha256"}
            and row.get("path") == name
            and path.is_file()
            and path.stat().st_size == row.get("bytes")
            and _sha(path) == row.get("sha256"),
            "F2_R1_VALIDATOR_FILE_HASH",
        )
        expected_bytes += int(row["bytes"])
    _require(manifest.get("bytes_before_manifest") == expected_bytes, "F2_R1_VALIDATOR_MANIFEST_BYTES")
    return lock, binding, terminal


def _topic(token: str) -> str:
    value = token.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    _require(ROS_TOPIC.fullmatch(value) is not None, "F2_R1_KALIBR_ROSTOPIC")
    return value


def _controls(raw: bytes) -> list[tuple[str, str | None, list[list[float]]]]:
    _require(type(raw) is bytes and 0 < len(raw) <= 4 * 1024 * 1024, "F2_R1_KALIBR_CONTROL_SIZE")
    _require("\x00" not in raw.decode("latin-1"), "F2_R1_KALIBR_CONTROL_NUL")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError("F2_R1_KALIBR_CONTROL_UTF8", str(error)) from error
    _require("\t" not in text, "F2_R1_KALIBR_CONTROL_TAB")
    lines = text.splitlines()
    active_node: tuple[str, int] | None = None
    topics: dict[str, str] = {}
    matrices: list[tuple[str, list[list[float]]]] = []
    paths: set[tuple[str, str]] = set()
    index = 0
    while index < len(lines):
        node = NODE.fullmatch(lines[index])
        if node and len(node.group("indent")) == 0:
            active_node = (node.group("name"), 0)
            index += 1
            continue
        topic = TOPIC.fullmatch(lines[index])
        if topic:
            _require(active_node is not None, "F2_R1_KALIBR_TOPIC_WITHOUT_CAMERA_NODE")
            _require(len(topic.group("indent")) > active_node[1], "F2_R1_KALIBR_TOPIC_INDENT")
            name = active_node[0]
            _require(name not in topics, "F2_R1_KALIBR_ROSTOPIC_DUPLICATE")
            topics[name] = _topic(topic.group("value"))
            index += 1
            continue
        key = NODE.fullmatch(lines[index])
        if key and key.group("name") == "T_cam_imu":
            _require(active_node is not None, "F2_R1_KALIBR_MATRIX_WITHOUT_CAMERA_NODE")
            key_indent = len(key.group("indent"))
            _require(key_indent > active_node[1], "F2_R1_KALIBR_MATRIX_INDENT")
            path = (active_node[0], "T_cam_imu")
            _require(path not in paths, "F2_R1_KALIBR_MATRIX_PATH_DUPLICATE")
            paths.add(path)
            rows: list[list[float]] = []
            for offset in range(1, 5):
                _require(index + offset < len(lines), "F2_R1_IMU_CALIBRATION_MATRIX")
                row = ROW.fullmatch(lines[index + offset])
                _require(row is not None, "F2_R1_IMU_CALIBRATION_MATRIX")
                _require(len(row.group("indent")) >= key_indent, "F2_R1_KALIBR_MATRIX_INDENT")
                values = [float(row.group(name)) for name in ("a", "b", "c", "d")]
                _require(all(math.isfinite(value) for value in values), "F2_R1_IMU_CALIBRATION_MATRIX_NONFINITE")
                rows.append(values)
            matrix = np.asarray(rows, dtype=np.float64)
            _require(
                np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12),
                "F2_R1_IMU_CALIBRATION_HOMOGENEOUS_ROW",
            )
            rotation = matrix[:3, :3]
            _require(
                np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
                and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-8,
                "F2_R1_IMU_CALIBRATION_ROTATION",
            )
            matrices.append((active_node[0], rows))
            index += 5
            continue
        index += 1
    return [(node, topics.get(node), rows) for node, rows in matrices]


def _normalize_member(name: str) -> str:
    _require(isinstance(name, str) and name != "" and "\x00" not in name, "F2_ZIP_MEMBER_NAME")
    _require("\\" not in name, "F2_ZIP_MEMBER_BACKSLASH")
    _require(unicodedata.normalize("NFC", name) == name, "F2_ZIP_MEMBER_NOT_NFC")
    _require(not name.startswith("/") and WINDOWS_DRIVE.match(name) is None, "F2_ZIP_MEMBER_ABSOLUTE")
    stripped = name.removesuffix("/")
    _require(stripped != "" and not stripped.endswith("/"), "F2_ZIP_MEMBER_NAME")
    parts = stripped.split("/")
    _require(all(part not in {"", ".", ".."} for part in parts), "F2_ZIP_MEMBER_DIRECTORY_ESCAPE")
    parsed = PurePosixPath(stripped)
    _require(not parsed.is_absolute(), "F2_ZIP_MEMBER_ABSOLUTE")
    normalized = parsed.as_posix()
    _require(normalized == stripped, "F2_ZIP_MEMBER_NORMALIZATION")
    return normalized


def _safe_infos(container: zipfile.ZipFile, access: dict[str, int]) -> tuple[list[zipfile.ZipInfo], int]:
    infos = container.infolist()
    access["archive_member_enumerations"] += 1
    _require(len(infos) <= EXPECTED_BUDGET["max_members"], "F2_ZIP_MEMBER_COUNT_BUDGET")
    seen: set[str] = set()
    yaml_infos: list[zipfile.ZipInfo] = []
    total = 0
    for info in infos:
        normalized = _normalize_member(info.orig_filename)
        folded = normalized.casefold()
        _require(folded not in seen, "F2_ZIP_MEMBER_CASEFOLD_DUPLICATE")
        seen.add(folded)
        _require((info.flag_bits & 0x1) == 0, "F2_ZIP_MEMBER_ENCRYPTED")
        _require(
            type(info.file_size) is int
            and type(info.compress_size) is int
            and info.file_size >= 0
            and info.compress_size >= 0,
            "F2_ZIP_MEMBER_SIZE_INVALID",
        )
        mode = (info.external_attr >> 16) & 0xFFFF
        _require(
            stat.S_IFMT(mode) not in {stat.S_IFLNK, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFIFO, stat.S_IFSOCK},
            "F2_ZIP_MEMBER_SPECIAL_FILE",
        )
        if info.is_dir():
            _require(info.file_size == 0, "F2_ZIP_DIRECTORY_NONEMPTY")
        else:
            _require(
                info.file_size <= EXPECTED_BUDGET["max_member_uncompressed_bytes"],
                "F2_ZIP_MEMBER_SIZE_BUDGET",
            )
            total += info.file_size
            _require(total <= EXPECTED_BUDGET["max_total_uncompressed_bytes"], "F2_ZIP_TOTAL_SIZE_BUDGET")
            if info.file_size > 0:
                _require(info.compress_size > 0, "F2_ZIP_COMPRESSION_BOMB")
                _require(
                    info.file_size / info.compress_size <= EXPECTED_BUDGET["max_compression_ratio"],
                    "F2_ZIP_COMPRESSION_BOMB",
                )
            if normalized.casefold().endswith((".yaml", ".yml")):
                yaml_infos.append(info)
    yaml_infos.sort(key=lambda item: _normalize_member(item.orig_filename))
    return yaml_infos, len(infos)


def _initial_observability() -> dict[str, object]:
    return {
        "archive_hash_verified": False,
        "archive_member_count": None,
        "yaml_candidate_count": None,
        "yaml_candidate_names_sha256": None,
        "yaml_members_read": 0,
        "yaml_member_bytes_read": 0,
        "all_yaml_candidates_read": False,
        "matrix_discovery_count": None,
        "target_namespace_match_count": None,
        "matrix_discoveries_sha256": None,
        "member_receipts_sha256": None,
        "first_or_best_selected": False,
    }


def _expected_access(observability: Mapping[str, object]) -> dict[str, object]:
    return {
        "calibration_archive_member_reads": observability["yaml_members_read"],
        "calibration_archive_member_bytes": observability["yaml_member_bytes_read"],
        "session_rgbd_archive_reads": 0,
        "session_imu_archive_reads": 0,
        "model_or_checkpoint_reads": 0,
        "source_truth_materializations": 0,
        "factor_scoring_runs": 0,
        "confirmation_runs": 0,
        "confirmation_root_created": False,
    }


def _archive_path(lock: Mapping[str, object], binding: Mapping[str, object]) -> Path:
    root = Path(str(lock["archive_root"]))
    _require(root.is_absolute(), "F2_ARCHIVE_ROOT_NOT_ABSOLUTE")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise ValidationError("F2_ARCHIVE_ROOT_INVALID", str(error)) from error
    _require(resolved_root.is_dir(), "F2_ARCHIVE_ROOT_NOT_DIRECTORY")
    filename = PurePosixPath(str(binding["url"])).name
    candidate = root / filename
    _require(candidate.parent == root and candidate.name == filename, "F2_ARCHIVE_NOT_DIRECT_CHILD")
    _require(candidate.exists() and candidate.is_file(), "F2_ARCHIVE_FILE_MISSING")
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise ValidationError("F2_ARCHIVE_PATH_STAT", str(error)) from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    _require(not candidate.is_symlink() and not bool(getattr(metadata, "st_file_attributes", 0) & reparse), "F2_ARCHIVE_REPARSE_POINT")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValidationError("F2_ARCHIVE_PATH_INVALID", str(error)) from error
    _require(resolved.parent == resolved_root, "F2_ARCHIVE_DIRECTORY_ESCAPE")
    return resolved


def _replay_archive(
    lock: Mapping[str, object],
    binding: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], str | None, dict[str, int]]:
    observability = _initial_observability()
    member_receipts: list[dict[str, object]] = []
    discoveries: list[dict[str, object]] = []
    access = {"archive_hash_passes": 0, "archive_member_enumerations": 0}
    error_code: str | None = None
    try:
        archive = _archive_path(lock, binding)
        _require(archive.stat().st_size == binding["bytes"], "F2_ARCHIVE_BYTES_MISMATCH")
        _require(_sha(archive) == str(binding["sha256"]).upper(), "F2_ARCHIVE_SHA_MISMATCH")
        access["archive_hash_passes"] += 1
        observability["archive_hash_verified"] = True
        try:
            with archive.open("rb") as raw_file:
                _require(os.fstat(raw_file.fileno()).st_size == binding["bytes"], "F2_ARCHIVE_BYTES_MISMATCH")
                _require(_sha_stream(raw_file) == str(binding["sha256"]).upper(), "F2_ARCHIVE_SHA_MISMATCH")
                access["archive_hash_passes"] += 1
                raw_file.seek(0)
                with zipfile.ZipFile(raw_file, "r") as container:
                    yaml_infos, member_count = _safe_infos(container, access)
                    observability["archive_member_count"] = member_count
                    candidate_names = [_normalize_member(info.orig_filename) for info in yaml_infos]
                    observability["yaml_candidate_count"] = len(candidate_names)
                    observability["yaml_candidate_names_sha256"] = _canonical_sha(candidate_names)
                    _require(
                        0 < len(candidate_names) <= EXPECTED_BUDGET["max_yaml_candidates"],
                        "F2_R1_CONTROL_YAML_CANDIDATE_COUNT",
                    )
                    for info, name in zip(yaml_infos, candidate_names, strict=True):
                        try:
                            with container.open(info, "r") as stream:
                                payload = stream.read(EXPECTED_BUDGET["max_member_uncompressed_bytes"] + 1)
                        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                            raise ValidationError("F2_ZIP_MEMBER_READ_FAILED", str(error)) from error
                        _require(
                            len(payload) == info.file_size
                            and len(payload) <= EXPECTED_BUDGET["max_member_uncompressed_bytes"],
                            "F2_SOURCE_MEMBER_SIZE_DRIFT",
                        )
                        receipt = {
                            "name": name,
                            "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest().upper(),
                        }
                        member_receipts.append(receipt)
                        observability["yaml_members_read"] = len(member_receipts)
                        observability["yaml_member_bytes_read"] = sum(int(row["bytes"]) for row in member_receipts)
                        observability["member_receipts_sha256"] = _canonical_sha(member_receipts)
                        for node, rostopic, matrix in _controls(payload):
                            namespace = rostopic.rpartition("/")[0] if rostopic is not None else None
                            discoveries.append(
                                {
                                    "name": name,
                                    "bytes": len(payload),
                                    "sha256": receipt["sha256"],
                                    "camera_node_key": node,
                                    "rostopic": rostopic,
                                    "rostopic_namespace": namespace,
                                    "matrix_key": "T_cam_imu",
                                    "matrix_sha256": _canonical_sha(matrix),
                                    "encoding": "KALIBR_CAMCHAIN_YAML_T_CAM_IMU_NESTED_4X4",
                                    "transform_direction": "IMU_TO_CAMERA_T_CAM_IMU",
                                }
                            )
                    matches = [row for row in discoveries if row["rostopic_namespace"] == EXPECTED_NAMESPACE]
                    observability["all_yaml_candidates_read"] = True
                    observability["matrix_discovery_count"] = len(discoveries)
                    observability["target_namespace_match_count"] = len(matches)
                    observability["matrix_discoveries_sha256"] = _canonical_sha(discoveries)
                    observability["member_receipts_sha256"] = _canonical_sha(member_receipts)
                    _require(len(matches) == 1, "F2_R1_CALIBRATION_CONTROL_TARGET_CAMERA_AMBIGUOUS_OR_MISSING")
        except ValidationError:
            raise
        except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
            raise ValidationError("F2_ARCHIVE_PREFLIGHT_FAILED", str(error)) from error
    except ValidationError as error:
        error_code = error.code
    return observability, member_receipts, discoveries, error_code, access


def _assert_equivalence(
    terminal: Mapping[str, object],
    binding: Mapping[str, object],
    observability: Mapping[str, object],
    member_receipts: list[dict[str, object]],
    discoveries: list[dict[str, object]],
    replay_error: str | None,
) -> tuple[str, dict[str, object] | None]:
    matches = [row for row in discoveries if row["rostopic_namespace"] == EXPECTED_NAMESPACE]
    _require(terminal.get("access_receipt") == _expected_access(observability), "F2_R1_VALIDATOR_ACCESS_RECEIPT")
    if terminal.get("schema") == RESULT_SCHEMA:
        _require(
            replay_error is None
            and set(terminal)
            == {
                "schema",
                "status",
                "protocol_id",
                "archive",
                "selection_contract",
                "selected_member",
                "inventory",
                "access_receipt",
                "claim_ceiling",
            }
            and terminal.get("status") == "CALIBRATION_CONTROL_R1_PASS_EXACT_MEMBER_AND_TARGET_CAMERA_BOUND"
            and terminal.get("protocol_id") == PROTOCOL_ID
            and terminal.get("archive")
            == {
                "filename": "camera_imu_calib_radtan.zip",
                "bytes": binding["bytes"],
                "sha256": binding["sha256"],
            }
            and terminal.get("selection_contract")
            == {
                "official_target_imu_rostopic": EXPECTED_IMU_ROSTOPIC,
                "expected_camera_sensor_namespace": EXPECTED_NAMESPACE,
                "first_or_best_selected": False,
            }
            and len(matches) == 1
            and terminal.get("selected_member") == matches[0]
            and terminal.get("inventory") == {**observability, "member_receipts": member_receipts},
            "F2_R1_VALIDATOR_PASS_RESULT",
        )
        return REPLAY_PASS, matches[0]
    _require(
        set(terminal)
        == {
            "schema",
            "status",
            "error_code",
            "one_shot_consumed",
            "observability",
            "access_receipt",
            "selection_receipt",
        }
        and terminal.get("schema") == FAILURE_SCHEMA
        and terminal.get("status") == "CALIBRATION_CONTROL_R1_FAIL_CLOSED"
        and isinstance(terminal.get("error_code"), str)
        and replay_error == terminal.get("error_code")
        and terminal.get("one_shot_consumed") is True
        and terminal.get("observability") == observability
        and terminal.get("selection_receipt")
        == {
            "expected_camera_sensor_namespace": EXPECTED_NAMESPACE,
            "selected_member": None,
            "selected_camera_node": None,
            "first_or_best_selected": False,
        },
        "F2_R1_VALIDATOR_FAILURE_RESULT",
    )
    return REPLAY_CONFIRMED_FAILURE, None


def _write_validator_manifest(root: Path, terminal: str) -> None:
    names = sorted(path.name for path in root.iterdir())
    _require("validator-manifest.json" not in names and len(names) == 5, "F2_R1_VALIDATOR_PRE_MANIFEST_FILE_SET")
    files = {
        name: {"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)}
        for name in names
    }
    _write_exclusive_json(
        root / "validator-manifest.json",
        {
            "schema": REPLAY_MANIFEST_SCHEMA,
            "evidence_root_consumed": True,
            "terminal": terminal,
            "file_count_before_validator_manifest": len(files),
            "bytes_before_validator_manifest": sum(int(row["bytes"]) for row in files.values()),
            "files": files,
        },
    )


def execute_one_shot_replay(root: Path, control_lock: Path) -> dict[str, object]:
    """Consume one replay receipt before any archive byte or member access."""

    root = root.resolve()
    control_lock = control_lock.resolve()
    lock, binding, terminal = _producer_chain(root, control_lock)
    initial_names = sorted(path.name for path in root.iterdir())
    _require(
        initial_names
        in (
            ["manifest.json", "result.json", "start-receipt.json"],
            ["failure.json", "manifest.json", "start-receipt.json"],
        ),
        "F2_R1_VALIDATOR_REPLAY_ALREADY_CONSUMED",
    )
    producer_terminal_name = "result.json" if "result.json" in initial_names else "failure.json"
    replay_start = {
        "schema": REPLAY_START_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "control_lock": {"path": str(control_lock), "sha256": _sha(control_lock)},
        "producer_manifest": {"path": str(root / "manifest.json"), "sha256": _sha(root / "manifest.json")},
        "producer_terminal": {
            "path": str(root / producer_terminal_name),
            "sha256": _sha(root / producer_terminal_name),
        },
        "replay_root_consumed_at_start": True,
        "archive_hash_passes_before_start": 0,
        "archive_members_enumerated_before_start": 0,
        "archive_members_read_before_start": 0,
    }
    try:
        _write_exclusive_json(root / "validator-start-receipt.json", replay_start)
    except FileExistsError as error:
        raise ValidationError("F2_R1_VALIDATOR_REPLAY_ALREADY_CONSUMED", str(error)) from error
    observability = _initial_observability()
    receipts: list[dict[str, object]] = []
    discoveries: list[dict[str, object]] = []
    replay_error: str | None = None
    access = {"archive_hash_passes": 0, "archive_member_enumerations": 0}
    try:
        observability, receipts, discoveries, replay_error, access = _replay_archive(lock, binding)
        replay_status, selected = _assert_equivalence(
            terminal,
            binding,
            observability,
            receipts,
            discoveries,
            replay_error,
        )
        result = {
            "schema": REPLAY_RESULT_SCHEMA,
            "status": replay_status,
            "protocol_id": PROTOCOL_ID,
            "control_lock": {"path": str(control_lock), "sha256": _sha(control_lock)},
            "producer_manifest": {"path": str(root / "manifest.json"), "sha256": _sha(root / "manifest.json")},
            "producer_terminal": {
                "path": str(root / producer_terminal_name),
                "sha256": _sha(root / producer_terminal_name),
                "status": terminal["status"],
                "error_code": terminal.get("error_code"),
            },
            "archive_replay_attempts": 1,
            "archive_hash_passes": access["archive_hash_passes"],
            "archive_member_enumerations": access["archive_member_enumerations"],
            "yaml_member_reads": observability["yaml_members_read"],
            "yaml_member_bytes": observability["yaml_member_bytes_read"],
            "matrix_discovery_count": observability["matrix_discovery_count"],
            "target_namespace_match_count": observability["target_namespace_match_count"],
            "selected_member": selected,
            "producer_equivalence": True,
            "forbidden_access": {
                "session_rgbd_archive_reads": 0,
                "session_imu_archive_reads": 0,
                "model_or_checkpoint_reads": 0,
                "source_truth_materializations": 0,
                "factor_scoring_runs": 0,
                "confirmation_runs": 0,
                "confirmation_root_created": False,
            },
            "claim_ceiling": (
                "Independent R1 calibration-control replay only; no session source, model, scientific Confirmation, "
                "product, or safety evidence."
            ),
        }
        _write_exclusive_json(root / "validator-result.json", result)
        _write_validator_manifest(root, replay_status)
        return result
    except Exception as error:
        code = getattr(error, "code", type(error).__name__)
        failure = {
            "schema": REPLAY_FAILURE_SCHEMA,
            "status": REPLAY_FAIL,
            "error_code": code,
            "one_shot_consumed": True,
            "archive_replay_attempts": 1,
            "archive_hash_passes": access["archive_hash_passes"],
            "archive_member_enumerations": access["archive_member_enumerations"],
            "yaml_member_reads": observability["yaml_members_read"],
            "yaml_member_bytes": observability["yaml_member_bytes_read"],
            "forbidden_access": {
                "session_rgbd_archive_reads": 0,
                "session_imu_archive_reads": 0,
                "model_or_checkpoint_reads": 0,
                "source_truth_materializations": 0,
                "factor_scoring_runs": 0,
                "confirmation_runs": 0,
                "confirmation_root_created": False,
            },
        }
        _write_exclusive_json(root / "validator-failure.json", failure)
        _write_validator_manifest(root, REPLAY_FAIL)
        if isinstance(error, ValidationError):
            raise
        raise ValidationError(str(code), str(error)) from error


def _verify_validator_manifest(root: Path) -> tuple[str, dict[str, object]]:
    manifest = _load_json(root / "validator-manifest.json", "F2_R1_VALIDATOR_REPLAY_MANIFEST_READ")
    terminal_name = "validator-result.json" if (root / "validator-result.json").is_file() else "validator-failure.json"
    terminal = _load_json(root / terminal_name, "F2_R1_VALIDATOR_REPLAY_TERMINAL_READ")
    names = sorted(path.name for path in root.iterdir())
    expected_names = sorted(
        {
            "start-receipt.json",
            "manifest.json",
            "validator-start-receipt.json",
            "validator-manifest.json",
            terminal_name,
            "result.json" if (root / "result.json").is_file() else "failure.json",
        }
    )
    _require(names == expected_names, "F2_R1_VALIDATOR_FINAL_FILE_SET")
    expected_files = set(expected_names) - {"validator-manifest.json"}
    _require(
        set(manifest)
        == {
            "schema",
            "evidence_root_consumed",
            "terminal",
            "file_count_before_validator_manifest",
            "bytes_before_validator_manifest",
            "files",
        }
        and manifest.get("schema") == REPLAY_MANIFEST_SCHEMA
        and manifest.get("evidence_root_consumed") is True
        and manifest.get("terminal") == terminal.get("status")
        and manifest.get("file_count_before_validator_manifest") == 5
        and isinstance(manifest.get("files"), Mapping)
        and set(manifest["files"]) == expected_files,
        "F2_R1_VALIDATOR_REPLAY_MANIFEST",
    )
    total = 0
    for name, row in manifest["files"].items():  # type: ignore[union-attr]
        path = root / str(name)
        _require(
            isinstance(row, Mapping)
            and set(row) == {"path", "bytes", "sha256"}
            and row.get("path") == name
            and path.stat().st_size == row.get("bytes")
            and _sha(path) == row.get("sha256"),
            "F2_R1_VALIDATOR_REPLAY_FILE_HASH",
        )
        total += int(row["bytes"])
    _require(manifest.get("bytes_before_validator_manifest") == total, "F2_R1_VALIDATOR_REPLAY_MANIFEST_BYTES")
    return terminal_name, terminal


def validate(root: Path, control_lock: Path) -> dict[str, object]:
    """Verify the finalized local replay chain without reopening the archive."""

    root = root.resolve()
    control_lock = control_lock.resolve()
    _lock, _binding, producer = _producer_chain(root, control_lock)
    terminal_name, replay = _verify_validator_manifest(root)
    producer_name = "result.json" if (root / "result.json").is_file() else "failure.json"
    start = _load_json(root / "validator-start-receipt.json", "F2_R1_VALIDATOR_REPLAY_START_READ")
    _require(
        start
        == {
            "schema": REPLAY_START_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "control_lock": {"path": str(control_lock), "sha256": _sha(control_lock)},
            "producer_manifest": {"path": str(root / "manifest.json"), "sha256": _sha(root / "manifest.json")},
            "producer_terminal": {
                "path": str(root / producer_name),
                "sha256": _sha(root / producer_name),
            },
            "replay_root_consumed_at_start": True,
            "archive_hash_passes_before_start": 0,
            "archive_members_enumerated_before_start": 0,
            "archive_members_read_before_start": 0,
        },
        "F2_R1_VALIDATOR_REPLAY_START_RECEIPT",
    )
    forbidden = replay.get("forbidden_access")
    _require(
        isinstance(forbidden, Mapping)
        and forbidden
        == {
            "session_rgbd_archive_reads": 0,
            "session_imu_archive_reads": 0,
            "model_or_checkpoint_reads": 0,
            "source_truth_materializations": 0,
            "factor_scoring_runs": 0,
            "confirmation_runs": 0,
            "confirmation_root_created": False,
        },
        "F2_R1_VALIDATOR_REPLAY_FORBIDDEN_ACCESS",
    )
    if terminal_name == "validator-result.json":
        producer_passed = (
            producer.get("status") == "CALIBRATION_CONTROL_R1_PASS_EXACT_MEMBER_AND_TARGET_CAMERA_BOUND"
        )
        expected_status = REPLAY_PASS if producer_passed else REPLAY_CONFIRMED_FAILURE
        producer_observability = producer.get("inventory") if producer_passed else producer.get("observability")
        _require(isinstance(producer_observability, Mapping), "F2_R1_VALIDATOR_REPLAY_PRODUCER_OBSERVABILITY")
        _require(
            set(replay)
            == {
                "schema",
                "status",
                "protocol_id",
                "control_lock",
                "producer_manifest",
                "producer_terminal",
                "archive_replay_attempts",
                "archive_hash_passes",
                "archive_member_enumerations",
                "yaml_member_reads",
                "yaml_member_bytes",
                "matrix_discovery_count",
                "target_namespace_match_count",
                "selected_member",
                "producer_equivalence",
                "forbidden_access",
                "claim_ceiling",
            }
            and replay.get("schema") == REPLAY_RESULT_SCHEMA
            and replay.get("status") == expected_status
            and replay.get("protocol_id") == PROTOCOL_ID
            and replay.get("control_lock") == {"path": str(control_lock), "sha256": _sha(control_lock)}
            and replay.get("producer_manifest")
            == {"path": str(root / "manifest.json"), "sha256": _sha(root / "manifest.json")}
            and replay.get("producer_terminal")
            == {
                "path": str(root / producer_name),
                "sha256": _sha(root / producer_name),
                "status": producer["status"],
                "error_code": producer.get("error_code"),
            }
            and replay.get("archive_replay_attempts") == 1
            and type(replay.get("archive_hash_passes")) is int
            and 0 <= int(replay["archive_hash_passes"]) <= 2
            and type(replay.get("archive_member_enumerations")) is int
            and 0 <= int(replay["archive_member_enumerations"]) <= 1
            and replay.get("yaml_member_reads") == producer_observability.get("yaml_members_read")
            and replay.get("yaml_member_bytes") == producer_observability.get("yaml_member_bytes_read")
            and replay.get("matrix_discovery_count") == producer_observability.get("matrix_discovery_count")
            and replay.get("target_namespace_match_count")
            == producer_observability.get("target_namespace_match_count")
            and replay.get("producer_equivalence") is True
            and replay.get("selected_member") == producer.get("selected_member")
            and replay.get("claim_ceiling")
            == (
                "Independent R1 calibration-control replay only; no session source, model, scientific Confirmation, "
                "product, or safety evidence."
            )
            and (not producer_passed or replay.get("archive_hash_passes") == 2)
            and (not producer_passed or replay.get("archive_member_enumerations") == 1),
            "F2_R1_VALIDATOR_REPLAY_RESULT",
        )
    else:
        _require(
            set(replay)
            == {
                "schema",
                "status",
                "error_code",
                "one_shot_consumed",
                "archive_replay_attempts",
                "archive_hash_passes",
                "archive_member_enumerations",
                "yaml_member_reads",
                "yaml_member_bytes",
                "forbidden_access",
            }
            and replay.get("schema") == REPLAY_FAILURE_SCHEMA
            and replay.get("status") == REPLAY_FAIL
            and isinstance(replay.get("error_code"), str)
            and replay.get("one_shot_consumed") is True
            and replay.get("archive_replay_attempts") == 1
            and type(replay.get("archive_hash_passes")) is int
            and 0 <= int(replay["archive_hash_passes"]) <= 2
            and type(replay.get("archive_member_enumerations")) is int
            and 0 <= int(replay["archive_member_enumerations"]) <= 1
            and type(replay.get("yaml_member_reads")) is int
            and int(replay["yaml_member_reads"]) >= 0
            and type(replay.get("yaml_member_bytes")) is int
            and int(replay["yaml_member_bytes"]) >= 0,
            "F2_R1_VALIDATOR_REPLAY_FAILURE",
        )
    return {
        "valid": True,
        "producer_terminal": producer["status"],
        "replay_terminal": replay["status"],
        "selected_member": replay.get("selected_member"),
        "archive_replay_attempts": replay["archive_replay_attempts"],
        "yaml_member_reads": replay["yaml_member_reads"],
        "matrix_discovery_count": replay.get("matrix_discovery_count"),
        "target_namespace_match_count": replay.get("target_namespace_match_count"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-lock", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate(args.root, args.control_lock) if args.verify_only else execute_one_shot_replay(
            args.root,
            args.control_lock,
        )
    except Exception as error:  # noqa: BLE001 - independent CLI is fail closed.
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
