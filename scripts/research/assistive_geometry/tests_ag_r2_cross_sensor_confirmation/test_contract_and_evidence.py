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
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.contract import (
    EXECUTION_SCHEMA,
    ContractError,
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _execution_lock(tmp_path: Path) -> tuple[dict, Path]:
    implementation = tmp_path / "implementation.json"
    implementation.write_text('{"fixture":true}\n', encoding="utf-8")
    bindings = []
    for index, role in enumerate(sorted(ROLES)):
        path = tmp_path / f"binding-{index:02d}.bin"
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
                "member": "calibration/camera_imu.txt",
                "camera_from_imu_key": "camera_from_imu",
                "mocap_time_scale_key": "scale",
                "mocap_time_anchor_seconds_key": "anchor",
                "mocap_time_offset_seconds_key": "offset",
                "camera_timestamp_to_seconds": "INTEGER_NANOSECONDS_TIMES_1E_MINUS_9",
                "imu_timestamp_to_seconds": "INTEGER_NANOSECONDS_TIMES_1E_MINUS_9",
                "imu_clock_domain": "CAMERA_CLOCK_NO_MOCAP_TRANSFORM",
                "groundtruth_timestamp_unit": "SECONDS",
                "maximum_pose_bracket_seconds": "0.10",
                "imu_half_window_seconds": "0.05",
                "minimum_imu_samples": 5,
            },
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
    assert validate_execution_lock(path, implementation_lock_path=implementation)["lock_id"] == EXECUTION_LOCK_ID
    for name, mutate, code in (
        ("authority", lambda value: value["authority"].__setitem__("training_or_tuning", True), "F2_EXECUTION_AUTHORITY_DRIFT"),
        ("time", lambda value: value["source_contract"]["calibration_binding"].__setitem__("imu_clock_domain", "MOCAP"), "F2_EXECUTION_CALIBRATION_TIME_DOMAIN_DRIFT"),
        ("binding", lambda value: value["runtime_bindings"][0].__setitem__("sha256", "0" * 64), "F2_EXECUTION_RUNTIME_BINDING_FILE_DRIFT"),
        ("one-shot", lambda value: value["one_shot"].__setitem__("rerun", True), "F2_EXECUTION_ONE_SHOT_DRIFT"),
    ):
        with pytest.raises(ContractError, match=code):
            changed = deepcopy(lock)
            mutate(changed)
            _write_lock(path, changed)
            validate_execution_lock(path, implementation_lock_path=implementation)


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
