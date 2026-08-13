from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation import (
    EXECUTION_LOCK_ID,
    PROTOCOL_ID,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.calibration_control_r1 import (
    CONTROL_AUTHORITY,
    CONTROL_BUDGET,
    CONTROL_LOCK_ID,
    CONTROL_LOCK_SCHEMA,
    CONTROL_STATUS,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.contract import (
    CALIBRATION_CONTROL_R1_RESULT_SCHEMA,
    DATA_IDENTITY_PATH,
    EXECUTION_SCHEMA,
    OFFICIAL_CONTROL_EVIDENCE_PATH,
    OFFICIAL_CONTROL_EVIDENCE_SCHEMA,
    ContractError,
    expected_official_control_contract,
    load_frozen_contracts,
    validate_execution_lock,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.evidence import (
    EvidenceWriter,
)

ROLES = {
    "DEPTHART_SOURCE_MANIFEST", "DEPTHART_EXTENSION", "DEPTHART_CHECKPOINT",
    "FACTOR_BASELINE_RESULT", "FACTOR_STUDENT_RESULT", "FACTOR_STUDENT_CHECKPOINT",
    "METRIC_STUDENT_RESULT", "METRIC_STUDENT_CHECKPOINT", "METRIC_SCALE_BANK_RESULT",
    "METRIC_SCALE_BANK", "FROZEN_HYBRID_RECIPE_RESULT",
}
REPO_ROOT = Path(__file__).resolve().parents[4]
DEPTHART_SOURCE_MANIFEST = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_DEPTHART_SOURCE_MANIFEST_2026-08-12.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _binding(role: str, path: Path) -> dict:
    return {"role": role, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _execution_lock(tmp_path: Path) -> tuple[dict, Path]:
    implementation = tmp_path / "implementation.json"
    implementation.write_text('{"fixture":true}\n', encoding="utf-8")
    official = OFFICIAL_CONTROL_EVIDENCE_PATH
    assert json.loads(official.read_text(encoding="utf-8"))["schema"] == OFFICIAL_CONTROL_EVIDENCE_SCHEMA
    assert expected_official_control_contract() == json.loads(official.read_text(encoding="utf-8"))["binding_contract"]
    control_root = tmp_path / "calibration-control-root"
    control_root.mkdir()
    control_lock_path = tmp_path / "calibration-control-lock.json"
    repair = tmp_path / "repair-r1.json"
    repair.write_text("{}\n", encoding="utf-8")
    official_r1 = tmp_path / "official-r1.json"
    official_r1.write_text("{}\n", encoding="utf-8")
    amendment = tmp_path / "amendment-r1.json"
    amendment.write_text("{}\n", encoding="utf-8")
    r0_terminal = tmp_path / "r0-terminal.json"
    r0_terminal.write_text("{}\n", encoding="utf-8")
    control_lock = {
        "schema": CONTROL_LOCK_SCHEMA,
        "lock_id": CONTROL_LOCK_ID,
        "protocol_id": PROTOCOL_ID,
        "status": CONTROL_STATUS,
        "repair_implementation_lock": _binding("R1_REPAIR_IMPLEMENTATION_LOCK", repair),
        "data_identity": _binding("DATA_IDENTITY_PRE_R0_SNAPSHOT", DATA_IDENTITY_PATH),
        "official_camera_selection_evidence": _binding("R1_OFFICIAL_CAMERA_SELECTION_EVIDENCE", official_r1),
        "protocol_amendment": _binding("R1_PROTOCOL_AMENDMENT", amendment),
        "r0_terminal": _binding("R0_CONSUMED_CONTROL_TERMINAL", r0_terminal),
        "archive_root": str((tmp_path / "archives").resolve()),
        "output_root": str(control_root.resolve()),
        "budget": CONTROL_BUDGET,
        "authority": CONTROL_AUTHORITY,
        "one_shot": {
            "exclusive_r1_control_root": True,
            "producer_runs": 1,
            "independent_validator_replays": 1,
            "r0_rerun": False,
            "r0_resume": False,
            "r0_replacement": False,
        },
    }
    control_lock_path.write_text(json.dumps(control_lock), encoding="utf-8")
    start = control_root / "start-receipt.json"
    start.write_text(
        json.dumps(
            {
                "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_start.v1",
                "protocol_id": PROTOCOL_ID,
                "control_lock": {"path": str(control_lock_path.resolve()), "sha256": _sha(control_lock_path)},
                "r0_terminal": {"path": str(r0_terminal.resolve()), "sha256": _sha(r0_terminal)},
                "control_root_consumed_at_start": True,
                "archive_bytes_read_before_start": 0,
                "archive_members_enumerated_before_start": 0,
            }
        ),
        encoding="utf-8",
    )
    control = control_root / "result.json"
    control.write_text(
        json.dumps(
            {
                "schema": CALIBRATION_CONTROL_R1_RESULT_SCHEMA,
                "status": "CALIBRATION_CONTROL_R1_PASS_EXACT_MEMBER_AND_TARGET_CAMERA_BOUND",
                "protocol_id": PROTOCOL_ID,
                "archive": {
                    "filename": "camera_imu_calib_radtan.zip",
                    "bytes": 3645288,
                    "sha256": "2588354EFC6BC2E407B0E1EDE4264714F4426D21A0785A595E3A5C78A9DA6437",
                },
                "selected_member": {
                    "name": "calibration/camchain-imucam.yaml",
                    "camera_node_key": "cam0",
                    "rostopic": "/uvc_camera/cam_2/image_raw",
                    "rostopic_namespace": "/uvc_camera/cam_2",
                    "matrix_key": "T_cam_imu",
                    "encoding": "KALIBR_CAMCHAIN_YAML_T_CAM_IMU_NESTED_4X4",
                    "transform_direction": "IMU_TO_CAMERA_T_CAM_IMU",
                },
                "selection_contract": {
                    "official_target_imu_rostopic": "/uvc_camera/cam_2/imu",
                    "expected_camera_sensor_namespace": "/uvc_camera/cam_2",
                    "first_or_best_selected": False,
                },
                "access_receipt": {
                    "session_rgbd_archive_reads": 0,
                    "session_imu_archive_reads": 0,
                    "model_or_checkpoint_reads": 0,
                    "source_truth_materializations": 0,
                    "factor_scoring_runs": 0,
                    "confirmation_runs": 0,
                    "confirmation_root_created": False,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = control_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_manifest.v1",
                "evidence_root_consumed": True,
                "terminal": "CALIBRATION_CONTROL_R1_PASS_EXACT_MEMBER_AND_TARGET_CAMERA_BOUND",
                "files": {
                    "result.json": {"path": "result.json", "bytes": control.stat().st_size, "sha256": _sha(control)},
                    "start-receipt.json": {"path": "start-receipt.json", "bytes": start.stat().st_size, "sha256": _sha(start)},
                },
            }
        ),
        encoding="utf-8",
    )
    replay_start = control_root / "validator-start-receipt.json"
    replay_start.write_text(
        json.dumps(
            {
                "schema": (
                    "blindassist.ag.r2.cross_sensor_factor_confirmation_"
                    "calibration_control_r1_validator_start.v1"
                ),
                "protocol_id": PROTOCOL_ID,
                "control_lock": {"path": str(control_lock_path.resolve()), "sha256": _sha(control_lock_path)},
                "producer_manifest": {"path": str(manifest.resolve()), "sha256": _sha(manifest)},
                "producer_terminal": {"path": str(control.resolve()), "sha256": _sha(control)},
                "replay_root_consumed_at_start": True,
                "archive_hash_passes_before_start": 0,
                "archive_members_enumerated_before_start": 0,
                "archive_members_read_before_start": 0,
            }
        ),
        encoding="utf-8",
    )
    replay_result = control_root / "validator-result.json"
    replay_result.write_text(
        json.dumps(
            {
                "schema": (
                    "blindassist.ag.r2.cross_sensor_factor_confirmation_"
                    "calibration_control_r1_validator_result.v1"
                ),
                "status": "CALIBRATION_CONTROL_R1_INDEPENDENT_REPLAY_PASS",
                "protocol_id": PROTOCOL_ID,
                "control_lock": {"path": str(control_lock_path.resolve()), "sha256": _sha(control_lock_path)},
                "producer_manifest": {"path": str(manifest.resolve()), "sha256": _sha(manifest)},
                "producer_terminal": {
                    "path": str(control.resolve()),
                    "sha256": _sha(control),
                    "status": "CALIBRATION_CONTROL_R1_PASS_EXACT_MEMBER_AND_TARGET_CAMERA_BOUND",
                    "error_code": None,
                },
                "archive_replay_attempts": 1,
                "producer_equivalence": True,
                "selected_member": json.loads(control.read_text())["selected_member"],
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
        ),
        encoding="utf-8",
    )
    replay_files = {
        path.name: {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in (manifest, control, start, replay_result, replay_start)
    }
    replay_manifest = control_root / "validator-manifest.json"
    replay_manifest.write_text(
        json.dumps(
            {
                "schema": (
                    "blindassist.ag.r2.cross_sensor_factor_confirmation_"
                    "calibration_control_r1_validator_manifest.v1"
                ),
                "evidence_root_consumed": True,
                "terminal": "CALIBRATION_CONTROL_R1_INDEPENDENT_REPLAY_PASS",
                "file_count_before_validator_manifest": 5,
                "bytes_before_validator_manifest": sum(row["bytes"] for row in replay_files.values()),
                "files": replay_files,
            }
        ),
        encoding="utf-8",
    )
    bindings = []
    for index, role in enumerate(sorted(ROLES)):
        path = DEPTHART_SOURCE_MANIFEST if role == "DEPTHART_SOURCE_MANIFEST" else tmp_path / f"binding-{index:02d}.bin"
        if role != "DEPTHART_SOURCE_MANIFEST":
            path.write_bytes(role.encode("utf-8"))
        bindings.append({"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)})
    lock = {
        "schema": EXECUTION_SCHEMA,
        "lock_id": EXECUTION_LOCK_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "ONE_SHOT_EXECUTION_AUTHORIZED_NOT_STARTED",
        "implementation_lock": {"path": str(implementation), "sha256": _sha(implementation)},
        "archive_root": str(tmp_path / "archives"),
        "output_root": str(tmp_path / "output"),
        "runtime_bindings": bindings,
        "source_contract": {
            "archive_budget": {
                "max_members": 100000,
                "max_member_uncompressed_bytes": 1073741824,
                "max_total_uncompressed_bytes": 34359738368,
                "max_compression_ratio": 200.0,
                "max_metadata_bytes": 16777216,
            },
            "calibration_binding": {
                "member": "calibration/camchain-imucam.yaml",
                "camera_node_key": "cam0",
                "camera_from_imu_key": "T_cam_imu",
                "calibration_encoding": "KALIBR_CAMCHAIN_YAML_T_CAM_IMU_NESTED_4X4",
                "camera_from_imu_transform_direction": "IMU_TO_CAMERA_T_CAM_IMU",
                "mocap_time_scale_key": "mocap_timescaling_camera",
                "mocap_time_anchor_seconds_key": "timescaling_anchor",
                "mocap_time_offset_seconds_key": "mocap_timeoffset_camera",
                "camera_timestamp_to_seconds": "INTEGER_NANOSECONDS_TIMES_1E_MINUS_9",
                "imu_timestamp_to_seconds": "INTEGER_NANOSECONDS_TIMES_1E_MINUS_9",
                "imu_clock_domain": "CAMERA_CLOCK_NO_MOCAP_TRANSFORM",
                "groundtruth_timestamp_unit": "SECONDS",
                "imu_delimiter_and_column_order": "WHITESPACE_TIMESTAMP_NS_GYRO_XYZ_LINEAR_ACCELERATION_XYZ",
                "imu_axis_and_frame_mapping": "IMU_SENSOR_FRAME_ROTATED_BY_T_CAM_IMU_TO_RIGHT_RGB_DEPTH_CAMERA_FRAME",
                "accelerometer_specific_force_sign": "STATIONARY_SPECIFIC_FORCE_POINTS_UP_OPPOSITE_GRAVITY",
                "maximum_pose_bracket_seconds": "0.10",
                "imu_half_window_seconds": "0.05",
                "minimum_imu_samples": 5,
            },
            "control_evidence_bindings": [
                _binding("OFFICIAL_FORMAT_AND_IMU_CONVENTION", official),
                _binding("CALIBRATION_CONTROL_LOCK", control_lock_path),
                _binding("CALIBRATION_CONTROL_START_RECEIPT", start),
                _binding("CALIBRATION_ARCHIVE_CONTROL_RESULT", control),
                _binding("CALIBRATION_CONTROL_MANIFEST", manifest),
                _binding("CALIBRATION_CONTROL_REPLAY_START_RECEIPT", replay_start),
                _binding("CALIBRATION_CONTROL_REPLAY_RESULT", replay_result),
                _binding("CALIBRATION_CONTROL_REPLAY_MANIFEST", replay_manifest),
            ],
        },
        "runtime": {
            "device": "cuda:0",
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "network_requests": 0,
            "training_steps": 0,
            "reducer_calls": 0,
        },
        "authority": {
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
        },
        "one_shot": {"exclusive_root": True, "rerun": False, "resume": False, "replacement": False},
    }
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return lock, implementation


def _write_lock(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_predecessor_hashes_and_authority_remain_frozen() -> None:
    protocol, identity = load_frozen_contracts()
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["execution_authority"]["authorized"] is False
    assert identity["payload_access_receipt"]["zip_members_enumerated"] is False
    assert identity["payload_access_receipt"]["source_outcome_access_started"] is False


def test_future_execution_lock_exact_schema_and_mutations(tmp_path: Path) -> None:
    lock, implementation = _execution_lock(tmp_path)
    path = tmp_path / "execution.json"
    def control_validator(control_path: Path) -> dict:
        return json.loads(control_path.read_text())

    def replay_validator(_root: Path, _control_path: Path) -> dict:
        return {
            "valid": True,
            "producer_terminal": "CALIBRATION_CONTROL_R1_PASS_EXACT_MEMBER_AND_TARGET_CAMERA_BOUND",
            "replay_terminal": "CALIBRATION_CONTROL_R1_INDEPENDENT_REPLAY_PASS",
            "archive_replay_attempts": 1,
        }
    assert (
        validate_execution_lock(
            path,
            implementation_lock_path=implementation,
            control_lock_validator=control_validator,
            control_replay_validator=replay_validator,
        )["lock_id"]
        == EXECUTION_LOCK_ID
    )
    for name, mutate, code in (
        ("authority", lambda value: value["authority"].__setitem__("training_or_tuning", True), "F2_EXECUTION_AUTHORITY_DRIFT"),
        ("time", lambda value: value["source_contract"]["calibration_binding"].__setitem__("imu_clock_domain", "MOCAP"), "F2_EXECUTION_CALIBRATION_OFFICIAL_CONTRACT_DRIFT"),
        ("encoding", lambda value: value["source_contract"]["calibration_binding"].__setitem__("calibration_encoding", "INLINE"), "F2_EXECUTION_CALIBRATION_OFFICIAL_CONTRACT_DRIFT"),
        ("binding", lambda value: value["runtime_bindings"][0].__setitem__("sha256", "0" * 64), "F2_EXECUTION_RUNTIME_BINDING_FILE_DRIFT"),
        ("one-shot", lambda value: value["one_shot"].__setitem__("rerun", True), "F2_EXECUTION_ONE_SHOT_DRIFT"),
    ):
        with pytest.raises(ContractError, match=code):
            changed = deepcopy(lock)
            mutate(changed)
            _write_lock(path, changed)
            validate_execution_lock(
                path,
                implementation_lock_path=implementation,
                control_lock_validator=control_validator,
                control_replay_validator=replay_validator,
            )
    _write_lock(path, lock)
    with pytest.raises(ContractError, match="F2_CALIBRATION_CONTROL_REPLAY_INDEPENDENT_VALIDATION"):
        validate_execution_lock(
            path,
            implementation_lock_path=implementation,
            control_lock_validator=control_validator,
            control_replay_validator=lambda _root, _control: {"valid": False},
        )


def test_evidence_root_is_exclusive_atomic_and_nonoverwritable(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    writer = EvidenceWriter(root, {"schema": "fixture-start"})
    writer.write_json("record.json", {"value": 1})
    writer.write_npz("array.npz", {"value": np.asarray([1.0], dtype=np.float64)})
    with pytest.raises(ContractError, match="F2_EVIDENCE_OVERWRITE"):
        writer.write_json("record.json", {"value": 2})
    manifest = writer.finalize("FIXTURE_COMPLETE")
    assert manifest["evidence_root_consumed"] is True
    assert not list(root.rglob("*.partial"))
    with pytest.raises(FileExistsError):
        EvidenceWriter(root, {"schema": "second"})
