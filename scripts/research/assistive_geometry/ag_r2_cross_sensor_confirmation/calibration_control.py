"""Separately lockable, calibration-archive-only ETH3D control preflight.

Importing this module performs no I/O.  Formal use consumes a dedicated control
root before hashing, enumerating, or reading the camera-IMU calibration archive.
It has no path or API for session RGB-D/IMU archives, checkpoints, or models.
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
    CALIBRATION_CONTROL_RESULT_SCHEMA,
    CALIBRATION_ENCODING,
    CAMERA_FROM_IMU_DIRECTION,
    DATA_IDENTITY_PATH,
    IMPLEMENTATION_LOCK_PATH,
    OFFICIAL_CONTROL_EVIDENCE_SCHEMA,
    ContractError,
    canonical_sha256,
    expected_official_control_contract,
    load_json,
    require,
    sha256_file,
    verified_absolute_binding,
)
from .control_format import discover_kalibr_matrices
from .eth3d_source import (
    ArchiveBinding,
    ArchiveBudget,
    SourcePhase,
    preflight_archive,
    verify_archive_binding,
)
from .evidence import EvidenceWriter

CONTROL_LOCK_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_lock.v1"
CONTROL_LOCK_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_CALIBRATION_CONTROL_PREFLIGHT_ONE_SHOT_EXECUTION_LOCK"
)
CONTROL_STATUS = "ONE_SHOT_CALIBRATION_CONTROL_AUTHORIZED_NOT_STARTED"
OFFICIAL_CONTROL_EVIDENCE_PATH = Path(__file__).resolve().parents[4] / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_OFFICIAL_CONTROL_EVIDENCE_2026-08-12.json"
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
    "calibration_archive_hash": True,
    "calibration_archive_member_enumeration": True,
    "calibration_yaml_member_read": True,
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
    require(isinstance(row, Mapping) and row.get("role") == role, f"F2_CONTROL_{name.upper()}_BINDING")
    path = verified_absolute_binding(row, f"F2_CONTROL_{name.upper()}_BINDING")
    require(path == expected_path.resolve(), f"F2_CONTROL_{name.upper()}_PATH_DRIFT")
    return path


def validate_control_lock(path: Path) -> dict[str, Any]:
    lock = load_json(path.resolve(), "F2_CONTROL_LOCK_READ")
    require(
        set(lock) == {
            "schema", "lock_id", "protocol_id", "status", "implementation_lock",
            "data_identity", "official_control_evidence", "archive_root", "output_root",
            "budget", "authority", "one_shot",
        },
        "F2_CONTROL_LOCK_KEY_SET",
    )
    require(lock["schema"] == CONTROL_LOCK_SCHEMA, "F2_CONTROL_LOCK_SCHEMA")
    require(lock["lock_id"] == CONTROL_LOCK_ID, "F2_CONTROL_LOCK_ID")
    require(lock["protocol_id"] == PROTOCOL_ID, "F2_CONTROL_PROTOCOL_ID")
    require(lock["status"] == CONTROL_STATUS, "F2_CONTROL_STATUS")
    implementation_path = _small_binding(
        lock, "implementation_lock", "REPAIR_IMPLEMENTATION_LOCK", IMPLEMENTATION_LOCK_PATH
    )
    identity_path = _small_binding(lock, "data_identity", "DATA_IDENTITY", DATA_IDENTITY_PATH)
    official_path = _small_binding(
        lock,
        "official_control_evidence",
        "OFFICIAL_FORMAT_AND_IMU_CONVENTION",
        OFFICIAL_CONTROL_EVIDENCE_PATH,
    )
    try:
        from .validate_repair_implementation_lock import validate_lock_file

        validate_lock_file(implementation_path, Path(__file__).resolve().parents[4])
    except Exception as error:
        raise ContractError("F2_CONTROL_REPAIR_IMPLEMENTATION_LOCK_INVALID", str(error)) from error
    identity = load_json(identity_path, "F2_CONTROL_DATA_IDENTITY_READ")
    calibration_rows = [
        row for row in identity.get("archives", [])
        if isinstance(row, Mapping) and row.get("kind") == "CAMERA_IMU_CALIBRATION_ARCHIVE"
    ]
    require(
        identity.get("protocol_id") == PROTOCOL_ID
        and identity.get("payload_access_receipt", {}).get("calibration_payload_opened") is False
        and len(calibration_rows) == 1,
        "F2_CONTROL_DATA_IDENTITY_DRIFT",
    )
    ArchiveBinding.from_manifest_row(calibration_rows[0])
    official = load_json(official_path, "F2_CONTROL_OFFICIAL_EVIDENCE_READ")
    require(
        official.get("schema") == OFFICIAL_CONTROL_EVIDENCE_SCHEMA
        and official.get("binding_contract") == expected_official_control_contract(),
        "F2_CONTROL_OFFICIAL_EVIDENCE_DRIFT",
    )
    require(lock["budget"] == CONTROL_BUDGET, "F2_CONTROL_BUDGET_DRIFT")
    require(lock["authority"] == CONTROL_AUTHORITY, "F2_CONTROL_AUTHORITY_DRIFT")
    require(
        lock["one_shot"] == {"exclusive_control_root": True, "rerun": False, "resume": False, "replacement": False},
        "F2_CONTROL_ONE_SHOT_DRIFT",
    )
    for key in ("archive_root", "output_root"):
        require(isinstance(lock[key], str) and Path(lock[key]).is_absolute(), f"F2_CONTROL_{key.upper()}")
    archive_root = Path(lock["archive_root"])
    output_root = Path(lock["output_root"])
    require(
        archive_root != output_root
        and archive_root not in output_root.parents
        and output_root not in archive_root.parents,
        "F2_CONTROL_ROOT_COLLISION",
    )
    return lock


def execute_control_preflight(lock_path: Path) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    lock = validate_control_lock(lock_path)
    identity_path = Path(lock["data_identity"]["path"])
    identity = load_json(identity_path, "F2_CONTROL_DATA_IDENTITY_READ")
    archive_row = next(row for row in identity["archives"] if row["kind"] == "CAMERA_IMU_CALIBRATION_ARCHIVE")
    binding = ArchiveBinding.from_manifest_row(archive_row)
    writer = EvidenceWriter(
        Path(lock["output_root"]),
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_start.v1",
            "protocol_id": PROTOCOL_ID,
            "control_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
            "control_root_consumed_at_start": True,
            "archive_bytes_read_before_start": 0,
            "archive_members_enumerated_before_start": 0,
        },
    )
    read_events = []
    try:
        verified = verify_archive_binding(Path(lock["archive_root"]), binding)
        budget = ArchiveBudget(**{key: value for key, value in CONTROL_BUDGET.items() if key != "max_yaml_candidates"})
        with preflight_archive(verified, budget=budget, observer=read_events.append) as archive:
            candidates = sorted(
                name for name in archive.file_names if name.casefold().endswith((".yaml", ".yml"))
            )
            require(0 < len(candidates) <= CONTROL_BUDGET["max_yaml_candidates"], "F2_CONTROL_YAML_CANDIDATE_COUNT")
            discoveries: list[dict[str, Any]] = []
            member_receipts: list[dict[str, Any]] = []
            for name in candidates:
                payload = archive.read_member_bytes(
                    name,
                    phase=SourcePhase.CALIBRATION_CONTROL,
                    purpose="DISCOVER_KALIBR_CAMERA_IMU_CONTROL",
                    max_bytes=CONTROL_BUDGET["max_member_uncompressed_bytes"],
                )
                member_receipts.append({"name": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest().upper()})
                try:
                    matrices = discover_kalibr_matrices(payload)
                except ContractError as error:
                    if error.code == "F2_IMU_CALIBRATION_KEY_AMBIGUOUS_OR_MISSING":
                        continue
                    raise
                for item in matrices:
                    discoveries.append(
                        {
                            "name": name,
                            "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest().upper(),
                            "camera_node_key": item.camera_node_key,
                            "matrix_key": item.matrix_key,
                            "matrix_sha256": canonical_sha256(item.matrix.tolist()),
                            "encoding": CALIBRATION_ENCODING,
                            "transform_direction": CAMERA_FROM_IMU_DIRECTION,
                        }
                    )
            require(len(discoveries) == 1, "F2_CALIBRATION_CONTROL_AMBIGUOUS_OR_MISSING_MATRIX")
            selected = discoveries[0]
            result = {
                "schema": CALIBRATION_CONTROL_RESULT_SCHEMA,
                "status": "CALIBRATION_CONTROL_PASS_EXACT_MEMBER_BOUND",
                "protocol_id": PROTOCOL_ID,
                "archive": {
                    "filename": binding.filename,
                    "bytes": binding.bytes,
                    "sha256": binding.sha256,
                },
                "selected_member": selected,
                "inventory": {
                    "member_count": len(archive.members),
                    "yaml_candidate_count": len(candidates),
                    "yaml_candidate_names_sha256": canonical_sha256(candidates),
                    "member_receipts": member_receipts,
                },
                "access_receipt": {
                    "calibration_archive_member_reads": len(read_events),
                    "calibration_archive_member_bytes": sum(event.bytes for event in read_events),
                    "session_rgbd_archive_reads": 0,
                    "session_imu_archive_reads": 0,
                    "model_or_checkpoint_reads": 0,
                    "source_truth_materializations": 0,
                    "factor_scoring_runs": 0,
                    "confirmation_runs": 0,
                    "confirmation_root_created": False,
                },
                "claim_ceiling": "Exact calibration-control member identity only; no session source, model, scientific Confirmation, product, or safety evidence.",
            }
            writer.write_json("result.json", result)
            writer.finalize(result["status"])
            return result
    except Exception as error:
        if "manifest.json" not in writer.files:
            writer.write_json(
                "failure.json",
                {
                    "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_failure.v1",
                    "status": "CALIBRATION_CONTROL_FAIL_CLOSED",
                    "error_code": getattr(error, "code", type(error).__name__),
                    "one_shot_consumed": True,
                },
            )
            writer.finalize("CALIBRATION_CONTROL_FAIL_CLOSED")
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
