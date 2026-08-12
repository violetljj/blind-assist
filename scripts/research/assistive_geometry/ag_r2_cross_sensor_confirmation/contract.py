"""Static contract and authority validation for the AG R2 F2 executor."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import EXECUTION_LOCK_ID, IMPLEMENTATION_LOCK_ID, PROTOCOL_ID

REPO_ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_LOCK_2026-08-12.json"
)
DATA_IDENTITY_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_DATA_IDENTITY_2026-08-12.json"
)
IMPLEMENTATION_LOCK_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_EXECUTOR_IMPLEMENTATION_LOCK_2026-08-12.json"
)
PROTOCOL_SHA256 = "8BA036E617531AE886BAAC8DAD60E5445BF8F0F7A2A073B7F8909750478D709F"
DATA_IDENTITY_SHA256 = "E755288202F4E7189538671F5F8C120F9D6EF68EBE80757844BC5272382B345B"
EXECUTION_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_execution_lock.v1"


class ContractError(RuntimeError):
    """Fail-closed contract violation with a stable error code."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


def require(condition: bool, code: str, message: str = "") -> None:
    if not condition:
        raise ContractError(code, message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest().upper()


def load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(code, str(error)) from error
    require(isinstance(value, dict), code)
    return value


def _exact_binding(binding: Mapping[str, Any], expected_path: Path, expected_sha: str, code: str) -> None:
    require(set(binding) == {"path", "sha256"}, f"{code}_FIELDS")
    bound_path = (REPO_ROOT / str(binding["path"])).resolve()
    require(bound_path == expected_path.resolve(), f"{code}_PATH")
    require(str(binding["sha256"]).upper() == expected_sha, f"{code}_DECLARED_SHA")
    require(bound_path.is_file() and sha256_file(bound_path) == expected_sha, f"{code}_FILE_SHA")


def load_frozen_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    require(PROTOCOL_PATH.is_file(), "F2_PROTOCOL_MISSING")
    require(DATA_IDENTITY_PATH.is_file(), "F2_DATA_IDENTITY_MISSING")
    require(sha256_file(PROTOCOL_PATH) == PROTOCOL_SHA256, "F2_PROTOCOL_SHA_DRIFT")
    require(sha256_file(DATA_IDENTITY_PATH) == DATA_IDENTITY_SHA256, "F2_DATA_IDENTITY_SHA_DRIFT")
    protocol = load_json(PROTOCOL_PATH, "F2_PROTOCOL_READ_FAILED")
    identity = load_json(DATA_IDENTITY_PATH, "F2_DATA_IDENTITY_READ_FAILED")
    require(protocol.get("protocol_id") == PROTOCOL_ID, "F2_PROTOCOL_ID_DRIFT")
    require(protocol.get("scientific_status") == "NOT_RUN", "F2_PROTOCOL_ALREADY_RUN")
    require(protocol.get("unique_successor") == IMPLEMENTATION_LOCK_ID, "F2_PREDECESSOR_SUCCESSOR_DRIFT")
    authority = protocol.get("execution_authority")
    require(isinstance(authority, Mapping) and authority.get("authorized") is False, "F2_PREDECESSOR_AUTHORITY_DRIFT")
    partition = protocol.get("data_partitions")
    require(isinstance(partition, list) and len(partition) == 1, "F2_DATA_PARTITION_DRIFT")
    _exact_binding(
        {"path": partition[0].get("identity_manifest_ref"), "sha256": partition[0].get("identity_sha256")},
        DATA_IDENTITY_PATH,
        DATA_IDENTITY_SHA256,
        "F2_IDENTITY_BINDING",
    )
    roster = identity.get("deterministic_roster_contract")
    require(
        isinstance(roster, Mapping)
        and roster.get("parent_ids_in_order") == ["plant_scene_2", "motion_1", "mannequin_5"]
        and roster.get("session_geometry_calibration_count_per_parent") == 12
        and roster.get("confirmation_score_count_per_parent") == 12
        and roster.get("calibration_and_score_identity_overlap") == 0
        and roster.get("post_access_reselection") is False,
        "F2_ROSTER_CONTRACT_DRIFT",
    )
    constraints = protocol.get("constraints")
    require(isinstance(constraints, list), "F2_GATE_LIST_INVALID")
    gates = [row for row in constraints if isinstance(row, Mapping) and row.get("class") == "GATE"]
    require([row.get("id") for row in gates] == [f"AG-R2-XSR-G{index:02d}-{suffix}" for index, suffix in enumerate((
        "PARENT-COUNT", "CALIBRATION-ROSTER", "SCORE-ROSTER", "MINIMUM-ELIGIBLE-PAIRS",
        "SESSION-HEIGHT-RANGE", "SESSION-HEIGHT-STABILITY", "SOURCE-DEPTH-COVERAGE",
        "SOURCE-SUPPORT-COVERAGE", "SOURCE-BOUNDARY-COVERAGE", "METRIC-KNOWN-COVERAGE",
        "SUPPORT-KNOWN-COVERAGE", "BOUNDARY-KNOWN-COVERAGE", "DEPTH-COMBINED-MACRO",
        "DEPTH-COMBINED-WORST", "DEPTH-SHAPE-MACRO", "DEPTH-SHAPE-WORST",
        "DEPTH-SCALE-MACRO", "DEPTH-SCALE-WORST", "SUPPORT-BRIER-MACRO",
        "SUPPORT-BRIER-WORST", "OBSTACLE-BRIER-MACRO", "OBSTACLE-BRIER-WORST",
        "BOUNDARY-ANGLE-MACRO", "BOUNDARY-ANGLE-WORST", "ONE-SIGMA-CALIBRATION",
        "TWO-SIGMA-CALIBRATION", "UNCERTAINTY-ORDERING"), start=1)], "F2_GATE_ID_DRIFT")
    return protocol, identity


def validate_execution_lock(
    path: Path,
    *,
    implementation_lock_path: Path = IMPLEMENTATION_LOCK_PATH,
) -> dict[str, Any]:
    """Validate a future external one-shot lock before any archive member access."""

    lock = load_json(path.resolve(), "F2_EXECUTION_LOCK_READ_FAILED")
    expected_keys = {
        "schema", "lock_id", "protocol_id", "status", "implementation_lock",
        "archive_root", "output_root", "runtime_bindings", "source_contract",
        "runtime", "authority", "one_shot",
    }
    require(set(lock) == expected_keys, "F2_EXECUTION_LOCK_KEY_SET_DRIFT")
    require(lock.get("schema") == EXECUTION_SCHEMA, "F2_EXECUTION_LOCK_SCHEMA_DRIFT")
    require(lock.get("lock_id") == EXECUTION_LOCK_ID, "F2_EXECUTION_LOCK_ID_DRIFT")
    require(lock.get("protocol_id") == PROTOCOL_ID, "F2_EXECUTION_PROTOCOL_DRIFT")
    require(lock.get("status") == "ONE_SHOT_EXECUTION_AUTHORIZED_NOT_STARTED", "F2_EXECUTION_STATUS_DRIFT")
    implementation = lock.get("implementation_lock")
    require(isinstance(implementation, Mapping), "F2_EXECUTION_IMPL_BINDING_INVALID")
    require(implementation_lock_path.is_file(), "F2_IMPLEMENTATION_LOCK_MISSING")
    _exact_binding(implementation, implementation_lock_path, sha256_file(implementation_lock_path), "F2_EXECUTION_IMPL_BINDING")
    authority = lock.get("authority")
    expected_authority = {
        "archive_member_access": True,
        "model_inference": True,
        "source_truth_materialization": True,
        "factor_scoring": True,
        "training_or_tuning": False,
        "reducer_or_task_state": False,
        "network": False,
        "device": False,
        "default_app": False,
        "product": False,
        "safety": False,
    }
    require(authority == expected_authority, "F2_EXECUTION_AUTHORITY_DRIFT")
    one_shot = lock.get("one_shot")
    require(one_shot == {"exclusive_root": True, "rerun": False, "resume": False, "replacement": False}, "F2_EXECUTION_ONE_SHOT_DRIFT")
    for name in ("archive_root", "output_root"):
        require(isinstance(lock.get(name), str) and Path(lock[name]).is_absolute(), f"F2_EXECUTION_{name.upper()}_INVALID")
    archive_root = Path(str(lock["archive_root"])).resolve()
    output_root = Path(str(lock["output_root"])).resolve()
    require(archive_root != output_root and archive_root not in output_root.parents and output_root not in archive_root.parents, "F2_EXECUTION_ROOT_COLLISION")
    runtime = lock.get("runtime")
    require(
        runtime == {
            "device": "cuda:0",
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "network_requests": 0,
            "training_steps": 0,
            "reducer_calls": 0,
        },
        "F2_EXECUTION_RUNTIME_DRIFT",
    )
    source = lock.get("source_contract")
    require(isinstance(source, Mapping) and set(source) == {"archive_budget", "calibration_binding"}, "F2_EXECUTION_SOURCE_CONTRACT_SCHEMA")
    budget = source["archive_budget"]
    require(
        budget == {
            "max_members": 100000,
            "max_member_uncompressed_bytes": 1073741824,
            "max_total_uncompressed_bytes": 34359738368,
            "max_compression_ratio": 200.0,
            "max_metadata_bytes": 16777216,
        },
        "F2_EXECUTION_ARCHIVE_BUDGET_DRIFT",
    )
    calibration = source["calibration_binding"]
    expected_calibration_keys = {
        "member", "camera_from_imu_key", "mocap_time_scale_key",
        "mocap_time_anchor_seconds_key", "mocap_time_offset_seconds_key",
        "camera_timestamp_to_seconds", "imu_timestamp_to_seconds", "imu_clock_domain",
        "groundtruth_timestamp_unit",
        "maximum_pose_bracket_seconds", "imu_half_window_seconds", "minimum_imu_samples",
    }
    require(isinstance(calibration, Mapping) and set(calibration) == expected_calibration_keys, "F2_EXECUTION_CALIBRATION_BINDING_SCHEMA")
    for name in (
        "member", "camera_from_imu_key", "mocap_time_scale_key",
        "mocap_time_anchor_seconds_key", "mocap_time_offset_seconds_key",
        "maximum_pose_bracket_seconds", "imu_half_window_seconds",
    ):
        require(isinstance(calibration[name], str) and calibration[name] != "", f"F2_EXECUTION_CALIBRATION_{name.upper()}_INVALID")
    require(
        calibration["camera_timestamp_to_seconds"] == "INTEGER_NANOSECONDS_TIMES_1E_MINUS_9"
        and calibration["imu_timestamp_to_seconds"] == "INTEGER_NANOSECONDS_TIMES_1E_MINUS_9"
        and calibration["imu_clock_domain"] == "CAMERA_CLOCK_NO_MOCAP_TRANSFORM"
        and calibration["groundtruth_timestamp_unit"] == "SECONDS",
        "F2_EXECUTION_CALIBRATION_TIME_DOMAIN_DRIFT",
    )
    require(type(calibration["minimum_imu_samples"]) is int and calibration["minimum_imu_samples"] >= 5, "F2_EXECUTION_CALIBRATION_MINIMUM_IMU_SAMPLES_INVALID")
    bindings = lock.get("runtime_bindings")
    require(isinstance(bindings, list) and bindings, "F2_EXECUTION_RUNTIME_BINDINGS_INVALID")
    roles: set[str] = set()
    for row in bindings:
        require(isinstance(row, Mapping) and set(row) == {"role", "path", "bytes", "sha256"}, "F2_EXECUTION_RUNTIME_BINDING_ROW_INVALID")
        role = str(row["role"])
        require(role not in roles, "F2_EXECUTION_RUNTIME_BINDING_ROLE_DUPLICATE")
        roles.add(role)
        require(isinstance(row["path"], str) and Path(row["path"]).is_absolute(), "F2_EXECUTION_RUNTIME_BINDING_PATH_INVALID")
        require(type(row["bytes"]) is int and row["bytes"] >= 0, "F2_EXECUTION_RUNTIME_BINDING_BYTES_INVALID")
        require(isinstance(row["sha256"], str) and len(row["sha256"]) == 64, "F2_EXECUTION_RUNTIME_BINDING_SHA_INVALID")
        bound = Path(row["path"]).resolve()
        require(
            bound.is_file() and bound.stat().st_size == row["bytes"]
            and sha256_file(bound) == str(row["sha256"]).upper(),
            "F2_EXECUTION_RUNTIME_BINDING_FILE_DRIFT",
        )
    required_roles = {
        "DEPTHART_SOURCE_MANIFEST", "DEPTHART_EXTENSION", "DEPTHART_CHECKPOINT",
        "FACTOR_BASELINE_RESULT", "FACTOR_STUDENT_RESULT", "FACTOR_STUDENT_CHECKPOINT",
        "METRIC_STUDENT_RESULT", "METRIC_STUDENT_CHECKPOINT", "METRIC_SCALE_BANK_RESULT",
        "METRIC_SCALE_BANK", "FROZEN_HYBRID_RECIPE_RESULT",
    }
    require(roles == required_roles, "F2_EXECUTION_RUNTIME_BINDING_ROLE_DRIFT")
    return lock


def binding_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["role"]): row for row in rows}
