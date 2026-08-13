"""Separately lockable R1 ETH3D calibration-archive control.

R1 preserves R0 and selects by the officially identified right-RGB sensor
namespace carried by each Kalibr camera node's ``rostopic``.  Import is pure;
formal use requires a future hash-bound R1 execution lock and a fresh root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import PROTOCOL_ID
from .contract import (
    DATA_IDENTITY_PATH,
    ContractError,
    canonical_sha256,
    load_json,
    require,
    sha256_file,
    verified_absolute_binding,
)
from .control_format_r1 import discover_kalibr_camera_controls, rostopic_namespace
from .eth3d_source import (
    ArchiveBinding,
    ArchiveBudget,
    SourcePhase,
    preflight_archive,
    verify_archive_binding,
)
from .evidence import EvidenceWriter

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTROL_LOCK_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_lock.v1"
CONTROL_LOCK_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_CALIBRATION_CONTROL_R1_ONE_SHOT_EXECUTION_LOCK"
)
CONTROL_STATUS = "ONE_SHOT_CALIBRATION_CONTROL_R1_AUTHORIZED_NOT_STARTED"
RESULT_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_result.v1"
FAILURE_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_failure.v1"
EXPECTED_IMU_ROSTOPIC = "/uvc_camera/cam_2/imu"
EXPECTED_CAMERA_NAMESPACE = "/uvc_camera/cam_2"
R1_REPAIR_LOCK_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_CALIBRATION_CONTROL_R0_FAILURE_AUDIT_AND_R1_PROTOCOL_REPAIR_IMPLEMENTATION_LOCK_2026-08-13.json"
)
R1_AMENDMENT_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_CALIBRATION_CONTROL_R1_PROTOCOL_AMENDMENT_2026-08-13.json"
)
R1_OFFICIAL_EVIDENCE_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_CALIBRATION_CONTROL_R1_OFFICIAL_CAMERA_SELECTION_EVIDENCE_2026-08-13.json"
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
CONTROL_BUDGET = {
    "max_members": 256,
    "max_member_uncompressed_bytes": 4194304,
    "max_total_uncompressed_bytes": 67108864,
    "max_compression_ratio": 100.0,
    "max_metadata_bytes": 4194304,
    "max_yaml_candidates": 32,
}
CONTROL_AUTHORITY = {
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


def _small_binding(lock: Mapping[str, Any], name: str, role: str, expected_path: Path) -> Path:
    row = lock.get(name)
    require(isinstance(row, Mapping) and row.get("role") == role, f"F2_R1_CONTROL_{name.upper()}_BINDING")
    path = verified_absolute_binding(row, f"F2_R1_CONTROL_{name.upper()}_BINDING")
    require(path == expected_path.resolve(), f"F2_R1_CONTROL_{name.upper()}_PATH_DRIFT")
    return path


def _expected_selection_contract() -> dict[str, Any]:
    return {
        "dataset_target_viewpoint": "ETH3D_PRIMARY_RGB_DEPTH_RIGHT_RGB_CAMERA",
        "official_target_imu_file": "imu.txt",
        "official_target_imu_rostopic": EXPECTED_IMU_ROSTOPIC,
        "expected_camera_sensor_namespace": EXPECTED_CAMERA_NAMESPACE,
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


def validate_control_lock(path: Path) -> dict[str, Any]:
    lock = load_json(path.resolve(), "F2_R1_CONTROL_LOCK_READ")
    require(
        set(lock) == {
            "schema", "lock_id", "protocol_id", "status", "repair_implementation_lock",
            "data_identity", "official_camera_selection_evidence", "protocol_amendment",
            "r0_terminal", "archive_root", "output_root", "budget", "authority", "one_shot",
        },
        "F2_R1_CONTROL_LOCK_KEY_SET",
    )
    require(lock["schema"] == CONTROL_LOCK_SCHEMA, "F2_R1_CONTROL_LOCK_SCHEMA")
    require(lock["lock_id"] == CONTROL_LOCK_ID, "F2_R1_CONTROL_LOCK_ID")
    require(lock["protocol_id"] == PROTOCOL_ID, "F2_R1_CONTROL_PROTOCOL_ID")
    require(lock["status"] == CONTROL_STATUS, "F2_R1_CONTROL_STATUS")
    implementation = _small_binding(
        lock,
        "repair_implementation_lock",
        "R1_REPAIR_IMPLEMENTATION_LOCK",
        R1_REPAIR_LOCK_PATH,
    )
    identity_path = _small_binding(lock, "data_identity", "DATA_IDENTITY_PRE_R0_SNAPSHOT", DATA_IDENTITY_PATH)
    official_path = _small_binding(
        lock,
        "official_camera_selection_evidence",
        "R1_OFFICIAL_CAMERA_SELECTION_EVIDENCE",
        R1_OFFICIAL_EVIDENCE_PATH,
    )
    amendment_path = _small_binding(
        lock,
        "protocol_amendment",
        "R1_PROTOCOL_AMENDMENT",
        R1_AMENDMENT_PATH,
    )
    r0_terminal_path = _small_binding(lock, "r0_terminal", "R0_CONSUMED_CONTROL_TERMINAL", R0_TERMINAL_PATH)
    try:
        from .validate_calibration_control_r1_repair_lock import validate_lock_file

        validate_lock_file(implementation, REPO_ROOT)
    except Exception as error:
        raise ContractError("F2_R1_CONTROL_REPAIR_IMPLEMENTATION_LOCK_INVALID", str(error)) from error
    identity = load_json(identity_path, "F2_R1_CONTROL_DATA_IDENTITY_READ")
    calibration_rows = [
        row
        for row in identity.get("archives", [])
        if isinstance(row, Mapping) and row.get("kind") == "CAMERA_IMU_CALIBRATION_ARCHIVE"
    ]
    require(identity.get("protocol_id") == PROTOCOL_ID and len(calibration_rows) == 1, "F2_R1_CONTROL_DATA_IDENTITY_DRIFT")
    ArchiveBinding.from_manifest_row(calibration_rows[0])
    official = load_json(official_path, "F2_R1_CONTROL_OFFICIAL_EVIDENCE_READ")
    require(
        official.get("schema")
        == "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_official_camera_selection_evidence.v1"
        and official.get("selection_contract") == _expected_selection_contract(),
        "F2_R1_CONTROL_OFFICIAL_EVIDENCE_DRIFT",
    )
    amendment = load_json(amendment_path, "F2_R1_CONTROL_AMENDMENT_READ")
    require(
        amendment.get("status") == "R1_PROTOCOL_REPAIR_FROZEN_NOT_AUTHORIZED_NOT_RUN"
        and amendment.get("prior_access_disclosure", {}).get("r0_calibration_archive_member_enumerated") is True
        and amendment.get("unchanged_scientific_contract", {}).get("scientific_status") == "NOT_RUN",
        "F2_R1_CONTROL_AMENDMENT_DRIFT",
    )
    r0_terminal = load_json(r0_terminal_path, "F2_R1_CONTROL_R0_TERMINAL_READ")
    require(
        r0_terminal.get("status")
        == "CALIBRATION_CONTROL_FAIL_CLOSED_AMBIGUOUS_OR_MISSING_MATRIX_ONE_SHOT_CONSUMED"
        and r0_terminal.get("control_outcome", {}).get("one_shot_consumed") is True,
        "F2_R1_CONTROL_R0_TERMINAL_DRIFT",
    )
    require(lock["budget"] == CONTROL_BUDGET, "F2_R1_CONTROL_BUDGET_DRIFT")
    require(lock["authority"] == CONTROL_AUTHORITY, "F2_R1_CONTROL_AUTHORITY_DRIFT")
    require(
        lock["one_shot"]
        == {
            "exclusive_r1_control_root": True,
            "producer_runs": 1,
            "independent_validator_replays": 1,
            "r0_rerun": False,
            "r0_resume": False,
            "r0_replacement": False,
        },
        "F2_R1_CONTROL_ONE_SHOT_DRIFT",
    )
    for key in ("archive_root", "output_root"):
        require(isinstance(lock[key], str) and Path(lock[key]).is_absolute(), f"F2_R1_CONTROL_{key.upper()}")
    archive_root = Path(lock["archive_root"])
    output_root = Path(lock["output_root"])
    require(archive_root == ARCHIVE_ROOT_PATH, "F2_R1_CONTROL_ARCHIVE_ROOT_PATH_DRIFT")
    require(output_root == OUTPUT_ROOT_PATH, "F2_R1_CONTROL_OUTPUT_ROOT_PATH_DRIFT")
    resolved_archive_root = archive_root.resolve(strict=True)
    resolved_output_root = output_root.resolve(strict=False)
    require(
        output_root.name == "ag-r2-cross-sensor-calibration-control-r1"
        and output_root.parent == OUTPUT_ROOT_PATH.parent
        and resolved_archive_root != resolved_output_root
        and resolved_archive_root not in resolved_output_root.parents
        and resolved_output_root not in resolved_archive_root.parents,
        "F2_R1_CONTROL_ROOT_COLLISION",
    )
    return lock


def _initial_observability() -> dict[str, Any]:
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


def _zero_access_receipt(observability: Mapping[str, Any]) -> dict[str, Any]:
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


def execute_control_preflight(lock_path: Path) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    lock = validate_control_lock(lock_path)
    identity = load_json(Path(lock["data_identity"]["path"]), "F2_R1_CONTROL_DATA_IDENTITY_READ")
    archive_row = next(row for row in identity["archives"] if row["kind"] == "CAMERA_IMU_CALIBRATION_ARCHIVE")
    binding = ArchiveBinding.from_manifest_row(archive_row)
    writer = EvidenceWriter(
        Path(lock["output_root"]),
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_start.v1",
            "protocol_id": PROTOCOL_ID,
            "control_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
            "r0_terminal": {
                "path": lock["r0_terminal"]["path"],
                "sha256": lock["r0_terminal"]["sha256"],
            },
            "control_root_consumed_at_start": True,
            "archive_bytes_read_before_start": 0,
            "archive_members_enumerated_before_start": 0,
        },
    )
    observability = _initial_observability()
    member_receipts: list[dict[str, Any]] = []
    discoveries: list[dict[str, Any]] = []
    try:
        verified = verify_archive_binding(Path(lock["archive_root"]), binding)
        observability["archive_hash_verified"] = True
        budget = ArchiveBudget(**{key: value for key, value in CONTROL_BUDGET.items() if key != "max_yaml_candidates"})
        with preflight_archive(verified, budget=budget) as archive:
            observability["archive_member_count"] = len(archive.members)
            candidates = sorted(name for name in archive.file_names if name.casefold().endswith((".yaml", ".yml")))
            observability["yaml_candidate_count"] = len(candidates)
            observability["yaml_candidate_names_sha256"] = canonical_sha256(candidates)
            require(0 < len(candidates) <= CONTROL_BUDGET["max_yaml_candidates"], "F2_R1_CONTROL_YAML_CANDIDATE_COUNT")
            for name in candidates:
                payload = archive.read_member_bytes(
                    name,
                    phase=SourcePhase.CALIBRATION_CONTROL,
                    purpose="DISCOVER_KALIBR_CAMERA_IMU_CONTROL",
                    max_bytes=CONTROL_BUDGET["max_member_uncompressed_bytes"],
                )
                receipt = {
                    "name": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest().upper(),
                }
                member_receipts.append(receipt)
                observability["yaml_members_read"] = len(member_receipts)
                observability["yaml_member_bytes_read"] = sum(int(row["bytes"]) for row in member_receipts)
                observability["member_receipts_sha256"] = canonical_sha256(member_receipts)
                for item in discover_kalibr_camera_controls(payload):
                    namespace = rostopic_namespace(item.rostopic) if item.rostopic is not None else None
                    discoveries.append(
                        {
                            "name": name,
                            "bytes": len(payload),
                            "sha256": receipt["sha256"],
                            "camera_node_key": item.camera_node_key,
                            "rostopic": item.rostopic,
                            "rostopic_namespace": namespace,
                            "matrix_key": item.matrix_key,
                            "matrix_sha256": canonical_sha256(item.matrix.tolist()),
                            "encoding": "KALIBR_CAMCHAIN_YAML_T_CAM_IMU_NESTED_4X4",
                            "transform_direction": "IMU_TO_CAMERA_T_CAM_IMU",
                        }
                    )
            matches = [row for row in discoveries if row["rostopic_namespace"] == EXPECTED_CAMERA_NAMESPACE]
            observability["all_yaml_candidates_read"] = True
            observability["matrix_discovery_count"] = len(discoveries)
            observability["target_namespace_match_count"] = len(matches)
            observability["matrix_discoveries_sha256"] = canonical_sha256(discoveries)
            observability["member_receipts_sha256"] = canonical_sha256(member_receipts)
            require(len(matches) == 1, "F2_R1_CALIBRATION_CONTROL_TARGET_CAMERA_AMBIGUOUS_OR_MISSING")
            selected = matches[0]
            result = {
                "schema": RESULT_SCHEMA,
                "status": "CALIBRATION_CONTROL_R1_PASS_EXACT_MEMBER_AND_TARGET_CAMERA_BOUND",
                "protocol_id": PROTOCOL_ID,
                "archive": {"filename": binding.filename, "bytes": binding.bytes, "sha256": binding.sha256},
                "selection_contract": {
                    "official_target_imu_rostopic": EXPECTED_IMU_ROSTOPIC,
                    "expected_camera_sensor_namespace": EXPECTED_CAMERA_NAMESPACE,
                    "first_or_best_selected": False,
                },
                "selected_member": selected,
                "inventory": {**observability, "member_receipts": member_receipts},
                "access_receipt": _zero_access_receipt(observability),
                "claim_ceiling": "Exact R1 calibration-control identity only; no session source, model, scientific Confirmation, product, or safety evidence.",
            }
            writer.write_json("result.json", result)
            writer.finalize(result["status"])
            return result
    except Exception as error:
        if "manifest.json" not in writer.files:
            failure = {
                "schema": FAILURE_SCHEMA,
                "status": "CALIBRATION_CONTROL_R1_FAIL_CLOSED",
                "error_code": getattr(error, "code", type(error).__name__),
                "one_shot_consumed": True,
                "observability": observability,
                "access_receipt": _zero_access_receipt(observability),
                "selection_receipt": {
                    "expected_camera_sensor_namespace": EXPECTED_CAMERA_NAMESPACE,
                    "selected_member": None,
                    "selected_camera_node": None,
                    "first_or_best_selected": False,
                },
            }
            writer.write_json("failure.json", failure)
            writer.finalize("CALIBRATION_CONTROL_R1_FAIL_CLOSED")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute_control_preflight(args.control_lock)
    except Exception as error:  # noqa: BLE001 - CLI reports the fail-closed code.
        print(json.dumps({"passed": False, "error_code": getattr(error, "code", type(error).__name__)}, sort_keys=True))
        return 1
    print(json.dumps({"passed": True, "status": result["status"], "selected_member": result["selected_member"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
