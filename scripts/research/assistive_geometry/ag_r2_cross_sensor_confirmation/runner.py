"""Hash-bound one-shot executor for the frozen AG R2 F2 confirmation.

Importing this module has no side effects.  Formal work requires an external
one-shot execution lock; this implementation lock does not create one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from . import PROTOCOL_ID
from .contract import (
    PROTOCOL_PATH,
    binding_map,
    canonical_sha256,
    load_frozen_contracts,
    require,
    sha256_file,
    validate_execution_lock,
)
from .eth3d_source import (
    ArchiveBinding,
    ArchiveBudget,
    CalibrationMemberBinding,
    Eth3dParentSource,
    ReadEvent,
    SourcePhase,
    preflight_archive,
    verify_archive_binding,
)
from .evidence import EvidenceWriter
from .metrics import PARENT_IDS, score_or_not_evaluable
from .model_only import ModelPaths, RGBKFactorPredictor, condition_parent_predictions
from .source_geometry import (
    derive_session_context,
    materialize_score_truth,
    source_parent_summary,
)

RESULT_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_result.v1"
PHASE_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_phase_completion.v1"
INVALID_TERMINAL = "INVALID_AND_CLOSE_EVIDENCE_VERSION_ONLY"
PREDICTION_ARRAY_KEYS = {
    "parent_id", "frame_id", "source_hw", "output_hw", "intrinsics",
    "depth_m", "depth_log_sigma", "depth_known", "support_probability",
    "support_residual_sigma_m", "support_known", "obstacle_probability",
    "boundary_distance_px", "boundary_sigma_px", "evidence_known",
}
TRUTH_ARRAY_KEYS = {
    "parent_id", "frame_id", "fx", "fy", "camera_height_m", "depth_m",
    "depth_known", "support_probability", "support_signed_residual_m",
    "support_known", "obstacle_probability", "boundary_distance_px", "evidence_known",
}


class PhaseFirewall:
    """Enforce roster -> raw -> calibration -> conditioned -> truth reads."""

    _TRANSITIONS: ClassVar[dict[str, str]] = {
        "ROSTER": "RAW_SCORE_PREDICTION",
        "RAW_SCORE_PREDICTION": "CALIBRATION_SOURCE",
        "CALIBRATION_SOURCE": "CONDITIONED_SEALED",
        "CONDITIONED_SEALED": "SCORE_SOURCE",
        "SCORE_SOURCE": "COMPLETE",
    }
    _ALLOWED: ClassVar[dict[str, SourcePhase | None]] = {
        "ROSTER": SourcePhase.ROSTER_METADATA,
        "RAW_SCORE_PREDICTION": SourcePhase.RAW_SCORE_PREDICTION,
        "CALIBRATION_SOURCE": SourcePhase.CALIBRATION_SOURCE,
        "CONDITIONED_SEALED": None,
        "SCORE_SOURCE": SourcePhase.SCORE_SOURCE,
        "COMPLETE": None,
    }

    def __init__(self) -> None:
        self.stage = "ROSTER"
        self.events: list[dict[str, Any]] = []

    def observe(self, event: ReadEvent) -> None:
        allowed = self._ALLOWED[self.stage]
        require(allowed is not None and event.phase is allowed, "F2_FIREWALL_SOURCE_PHASE_VIOLATION")
        self.events.append(
            {
                "event_index": len(self.events),
                "firewall_stage": self.stage,
                "parent_id": event.parent_id,
                "archive_kind": event.archive_kind,
                "source_phase": event.phase.value,
                "purpose": event.purpose,
                "member": event.member,
                "bytes": event.bytes,
            }
        )

    def advance(self, expected: str) -> None:
        require(self.stage == expected, "F2_FIREWALL_STAGE_DRIFT")
        self.stage = self._TRANSITIONS[expected]


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = canonical_sha256(result)
    return result


def _write_sealed(writer: EvidenceWriter, relative: str, value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sealed = _sealed(value)
    receipt = writer.write_json(relative, sealed)
    return sealed, receipt


def _predecessor(path: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {"path": path, "sha256": receipt["sha256"]}


def _record_receipt(parent_id: str, frame_id: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {"parent_id": parent_id, "frame_id": frame_id, **dict(receipt)}


def _prediction_arrays(parent_id: str, frame_id: str, value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result = {
        "parent_id": np.asarray(parent_id),
        "frame_id": np.asarray(frame_id),
        "source_hw": np.asarray(value["source_hw"], dtype=np.int64),
        "output_hw": np.asarray(value["output_hw"], dtype=np.int64),
        "intrinsics": np.asarray(value["intrinsics"], dtype=np.float64),
    }
    for name in PREDICTION_ARRAY_KEYS - set(result):
        result[name] = np.asarray(value[name])
    _validate_prediction_arrays(result)
    return result


def _truth_arrays(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    truth = value["truth"]
    result = {
        "parent_id": np.asarray(value["parent_id"]),
        "frame_id": np.asarray(value["frame_id"]),
        "fx": np.asarray(value["fx"], dtype=np.float64),
        "fy": np.asarray(value["fy"], dtype=np.float64),
        "camera_height_m": np.asarray(value["camera_height_m"], dtype=np.float64),
        **{name: np.asarray(item) for name, item in truth.items()},
    }
    require(set(result) == TRUTH_ARRAY_KEYS, "F2_TRUTH_ARRAY_KEY_SET")
    return result


def _validate_known(value: np.ndarray, known: np.ndarray, code: str, *, positive: bool = False) -> None:
    require(value.dtype == np.dtype("float64") and value.shape == known.shape, f"{code}_SCHEMA")
    require(bool(np.all(np.isfinite(value[known]))) and bool(np.all(np.isnan(value[~known]))), f"{code}_UNKNOWN")
    if positive:
        require(bool(np.all(value[known] > 0.0)), f"{code}_POSITIVE")


def _validate_prediction_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    require(set(arrays) == PREDICTION_ARRAY_KEYS, "F2_PREDICTION_ARRAY_KEY_SET")
    require(arrays["parent_id"].ndim == arrays["frame_id"].ndim == 0, "F2_PREDICTION_IDENTITY_RANK")
    require(arrays["parent_id"].dtype.kind == arrays["frame_id"].dtype.kind == "U", "F2_PREDICTION_IDENTITY_DTYPE")
    output_hw = arrays["output_hw"]
    require(output_hw.dtype == np.dtype("int64") and output_hw.shape == (2,), "F2_PREDICTION_OUTPUT_HW")
    shape = tuple(int(item) for item in output_hw)
    require(shape[0] > 0 and shape[1] > 0, "F2_PREDICTION_OUTPUT_HW_VALUE")
    require(arrays["source_hw"].dtype == np.dtype("int64") and arrays["source_hw"].shape == (2,), "F2_PREDICTION_SOURCE_HW")
    require(arrays["intrinsics"].dtype == np.dtype("float64") and arrays["intrinsics"].shape == (3, 3), "F2_PREDICTION_K")
    masks = {}
    for name in ("depth_known", "support_known", "evidence_known"):
        value = arrays[name]
        require(value.dtype == np.dtype("bool") and value.shape == shape, f"F2_PREDICTION_{name.upper()}")
        masks[name] = value
    _validate_known(arrays["depth_m"], masks["depth_known"], "F2_PREDICTION_DEPTH", positive=True)
    _validate_known(arrays["depth_log_sigma"], masks["depth_known"], "F2_PREDICTION_DEPTH_SIGMA", positive=True)
    _validate_known(arrays["support_probability"], masks["support_known"], "F2_PREDICTION_SUPPORT")
    _validate_known(arrays["support_residual_sigma_m"], masks["support_known"], "F2_PREDICTION_SUPPORT_SIGMA", positive=True)
    _validate_known(arrays["obstacle_probability"], masks["evidence_known"], "F2_PREDICTION_OBSTACLE")
    _validate_known(arrays["boundary_distance_px"], masks["evidence_known"], "F2_PREDICTION_BOUNDARY")
    _validate_known(arrays["boundary_sigma_px"], masks["evidence_known"], "F2_PREDICTION_BOUNDARY_SIGMA", positive=True)
    for name in ("support_probability", "obstacle_probability"):
        mask = masks["support_known" if name.startswith("support") else "evidence_known"]
        require(bool(np.all((arrays[name][mask] >= 0.0) & (arrays[name][mask] <= 1.0))), f"F2_PREDICTION_{name.upper()}_RANGE")
    require(bool(np.all((arrays["boundary_distance_px"][masks["evidence_known"]] >= 0.0) & (arrays["boundary_distance_px"][masks["evidence_known"]] <= 32.0))), "F2_PREDICTION_BOUNDARY_RANGE")


def _reload_records(
    writer: EvidenceWriter,
    records: Sequence[Mapping[str, Any]],
    expected_keys: set[str],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    result: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for receipt in records:
        require(set(receipt) == {"parent_id", "frame_id", "path", "bytes", "sha256"}, "F2_RECORD_RECEIPT_SCHEMA")
        path = (writer.root / str(receipt["path"])).resolve()
        require(writer.root in path.parents and path.is_file(), "F2_RECORD_PATH")
        require(path.stat().st_size == receipt["bytes"] and sha256_file(path) == receipt["sha256"], "F2_RECORD_FILE_DRIFT")
        with np.load(path, allow_pickle=False) as payload:
            require(set(payload.files) == expected_keys, "F2_RECORD_KEY_SET")
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        key = (str(arrays["parent_id"].item()), str(arrays["frame_id"].item()))
        require(key == (receipt["parent_id"], receipt["frame_id"]) and key not in result, "F2_RECORD_IDENTITY")
        if expected_keys == PREDICTION_ARRAY_KEYS:
            _validate_prediction_arrays(arrays)
        result[key] = arrays
    require(len(result) == 36, "F2_RECORD_COUNT")
    return result


def _source_frame(
    adapter: Eth3dParentSource,
    frame_id: str,
    phase: SourcePhase,
    imu_archive: Any,
    calibration_archive: Any,
    calibration_binding: CalibrationMemberBinding,
) -> dict[str, Any]:
    depth = adapter.read_source_arrays(frame_id, phase=phase)
    geometry = adapter.read_pose_and_gravity(
        frame_id,
        phase=phase,
        imu_archive=imu_archive,
        calibration_archive=calibration_archive,
        calibration_binding=calibration_binding,
    )
    require(depth.parent_id == geometry.parent_id and depth.frame_id == geometry.frame_id and depth.role == geometry.role, "F2_SOURCE_MODALITY_IDENTITY")
    value = np.asarray(depth.depth_m_hw, dtype=np.float64)
    known = np.asarray(depth.depth_known_hw, dtype=bool)
    value[~known] = np.nan
    return {
        "parent_id": depth.parent_id,
        "frame_id": depth.frame_id,
        "depth_m": value,
        "depth_known": known,
        "intrinsics": np.asarray(depth.K, dtype=np.float64),
        "camera_to_world": np.asarray(geometry.camera_to_world, dtype=np.float64),
        "gravity_up_camera_xyz": np.asarray(geometry.gravity_up_camera_xyz, dtype=np.float64),
    }


def _metric_frames(
    predictions: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    truths: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    require(set(predictions) == set(truths), "F2_SCORE_IDENTITY_SET")
    rows = []
    prediction_names = {
        "depth_m", "depth_log_sigma", "depth_known", "support_probability",
        "support_residual_sigma_m", "support_known", "obstacle_probability",
        "boundary_distance_px", "boundary_sigma_px", "evidence_known",
    }
    truth_names = {
        "depth_m", "depth_known", "support_probability", "support_signed_residual_m",
        "support_known", "obstacle_probability", "boundary_distance_px", "evidence_known",
    }
    for key in sorted(predictions):
        prediction = predictions[key]
        truth = truths[key]
        rows.append(
            {
                "parent_id": key[0],
                "frame_id": key[1],
                "fx": float(truth["fx"].item()),
                "fy": float(truth["fy"].item()),
                "prediction": {name: prediction[name] for name in prediction_names},
                "truth": {name: truth[name] for name in truth_names},
            }
        )
    return rows


def _scientific_source_error(code: str) -> bool:
    if any(token in code for token in ("PREDICTION", "MODEL", "FIREWALL", "EVIDENCE", "EXECUTION_LOCK")):
        return False
    tokens = (
        "ZIP_", "ROSTER_", "ASSOCIATED_", "CALIBRATION_", "GROUNDTRUTH_",
        "IMU_", "MOCAP_", "POSE_", "SOURCE_", "SESSION_", "SUPPORT_",
    )
    return any(token in code for token in tokens) and "FIREWALL" not in code


def _close_failure(writer: EvidenceWriter, firewall: PhaseFirewall, error: Exception) -> dict[str, Any]:
    code = str(getattr(error, "code", type(error).__name__))
    terminal = "NOT_EVALUABLE" if _scientific_source_error(code) else INVALID_TERMINAL
    failure, failure_receipt = _write_sealed(
        writer,
        "failure.json",
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_failure.v1",
            "terminal": terminal,
            "reason_code": code,
            "message": str(error),
            "firewall_stage": firewall.stage,
            "access_events": firewall.events,
        },
    )
    result = _sealed(
        {
            "schema": RESULT_SCHEMA,
            "terminal": terminal,
            "execution_valid": terminal == "NOT_EVALUABLE",
            "failure_sha256": failure["content_sha256"],
            "failure_file_sha256": failure_receipt["sha256"],
            "summary_sha256": None,
            "model_inference_calls": sum(event["purpose"] == "RAW_SCORE_RGB" for event in firewall.events),
            "training_steps": 0,
            "reducer_calls": 0,
            "network_requests": 0,
        }
    )
    writer.write_json("result.json", result)
    writer.finalize(terminal)
    return result


def execute(execution_lock_path: Path) -> dict[str, Any]:
    """Consume one external lock and run once; never resume or replace."""

    protocol, identity = load_frozen_contracts()
    execution_lock_path = execution_lock_path.resolve()
    lock = validate_execution_lock(execution_lock_path)
    bindings = binding_map(lock["runtime_bindings"])
    model_paths = ModelPaths.from_bindings(bindings)
    archive_root = Path(lock["archive_root"]).resolve()
    output_root = Path(lock["output_root"]).resolve()
    require(archive_root.is_dir() and not output_root.exists(), "F2_EXECUTION_ROOT_STATE")
    archive_bindings = [ArchiveBinding.from_manifest_row(row) for row in identity["archives"]]
    verified = {binding.kind + ":" + binding.parent_id: verify_archive_binding(archive_root, binding) for binding in archive_bindings}
    firewall = PhaseFirewall()
    writer = EvidenceWriter(
        output_root,
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_start_receipt.v1",
            "protocol_id": PROTOCOL_ID,
            "execution_lock": {"path": str(execution_lock_path), "sha256": sha256_file(execution_lock_path)},
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "output_root_consumed_at_start": True,
            "archive_members_enumerated_before_start": False,
            "model_inference_before_start": False,
        },
    )
    try:
        budget = ArchiveBudget(**lock["source_contract"]["archive_budget"])
        calibration_row = lock["source_contract"]["calibration_binding"]
        calibration_binding = CalibrationMemberBinding(
            member=calibration_row["member"],
            camera_from_imu_key=calibration_row["camera_from_imu_key"],
            mocap_time_scale_key=calibration_row["mocap_time_scale_key"],
            mocap_time_anchor_seconds_key=calibration_row["mocap_time_anchor_seconds_key"],
            mocap_time_offset_seconds_key=calibration_row["mocap_time_offset_seconds_key"],
            camera_timestamp_to_seconds=calibration_row["camera_timestamp_to_seconds"],
            imu_timestamp_to_seconds=calibration_row["imu_timestamp_to_seconds"],
            imu_clock_domain=calibration_row["imu_clock_domain"],
            groundtruth_timestamp_unit=calibration_row["groundtruth_timestamp_unit"],
            maximum_pose_bracket_seconds=Decimal(calibration_row["maximum_pose_bracket_seconds"]),
            imu_half_window_seconds=Decimal(calibration_row["imu_half_window_seconds"]),
            minimum_imu_samples=calibration_row["minimum_imu_samples"],
        )
        with ExitStack() as stack:
            archives = {
                key: stack.enter_context(preflight_archive(value, budget=budget, observer=firewall.observe))
                for key, value in verified.items()
            }
            adapters = {
                parent: Eth3dParentSource(archives[f"RGBD_TRAINING_ARCHIVE:{parent}"], parent_id=parent)
                for parent in PARENT_IDS
            }
            _roster, roster_receipt = _write_sealed(
                writer,
                "roster.json",
                {
                    "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_roster.v1",
                    "phase": "ROSTER_SEALED",
                    "parents": [adapters[parent].roster.as_dict() for parent in PARENT_IDS],
                    "parent_count": 3,
                    "calibration_count": 36,
                    "score_count": 36,
                },
            )
            firewall.advance("ROSTER")

            predictor = RGBKFactorPredictor(model_paths, lock["runtime"]["device"])
            raw_receipts = []
            for parent in PARENT_IDS:
                adapter = adapters[parent]
                for frame in adapter.roster.score:
                    source = adapter.read_prediction_input(frame.frame_id, phase=SourcePhase.RAW_SCORE_PREDICTION)
                    raw = predictor.predict(np.asarray(source.rgb_hwc_u8), np.asarray(source.K, dtype=np.float64))
                    arrays = _prediction_arrays(parent, frame.frame_id, raw)
                    receipt = writer.write_npz(f"phase-a/raw/{parent}/{frame.frame_id}.npz", arrays)
                    raw_receipts.append(_record_receipt(parent, frame.frame_id, receipt))
            _raw_completion, raw_completion_receipt = _write_sealed(
                writer,
                "phase-a/raw-prediction-completion.json",
                {
                    "schema": PHASE_SCHEMA,
                    "phase": "RAW_RGBK_PREDICTIONS_SEALED",
                    "record_count": len(raw_receipts),
                    "records": raw_receipts,
                    "predecessor": _predecessor("roster.json", roster_receipt),
                },
            )
            raw_records = _reload_records(writer, raw_receipts, PREDICTION_ARRAY_KEYS)
            firewall.advance("RAW_SCORE_PREDICTION")

            contexts: dict[str, dict[str, Any]] = {}
            identities: dict[str, dict[str, Any]] = {}
            context_rows = []
            for parent in PARENT_IDS:
                adapter = adapters[parent]
                calibration_frames = [
                    _source_frame(
                        adapter,
                        frame.frame_id,
                        SourcePhase.CALIBRATION_SOURCE,
                        archives[f"IMU_ARCHIVE:{parent}"],
                        archives["CAMERA_IMU_CALIBRATION_ARCHIVE:ALL_THREE_SESSIONS"],
                        calibration_binding,
                    )
                    for frame in adapter.roster.calibration
                ]
                context, identity_receipt = derive_session_context(parent, calibration_frames)
                contexts[parent] = context
                identities[parent] = identity_receipt
                context_rows.append(identity_receipt)
            _context_completion, context_completion_receipt = _write_sealed(
                writer,
                "phase-b/session-contexts.json",
                {
                    "schema": PHASE_SCHEMA,
                    "phase": "CALIBRATION_SOURCE_CONTEXT_SEALED",
                    "parent_count": len(context_rows),
                    "contexts": context_rows,
                    "predecessor": _predecessor("phase-a/raw-prediction-completion.json", raw_completion_receipt),
                },
            )
            firewall.advance("CALIBRATION_SOURCE")

            conditioned_receipts = []
            conditioning_rows = []
            for parent in PARENT_IDS:
                ordered = [raw_records[(parent, frame.frame_id)] for frame in adapters[parent].roster.score]
                conditioned, conditioning = condition_parent_predictions(ordered, contexts[parent])
                conditioning_rows.append(conditioning)
                for value in conditioned:
                    frame_id = str(np.asarray(value["frame_id"]).item())
                    arrays = _prediction_arrays(parent, frame_id, value)
                    receipt = writer.write_npz(f"phase-c/conditioned/{parent}/{frame_id}.npz", arrays)
                    conditioned_receipts.append(_record_receipt(parent, frame_id, receipt))
            _conditioned_completion, conditioned_completion_receipt = _write_sealed(
                writer,
                "phase-c/conditioned-factor-completion.json",
                {
                    "schema": PHASE_SCHEMA,
                    "phase": "CONDITIONED_FACTORS_SEALED_BEFORE_SCORE_TRUTH",
                    "record_count": len(conditioned_receipts),
                    "records": conditioned_receipts,
                    "conditioning": conditioning_rows,
                    "predecessor": _predecessor("phase-b/session-contexts.json", context_completion_receipt),
                },
            )
            conditioned_records = _reload_records(writer, conditioned_receipts, PREDICTION_ARRAY_KEYS)
            firewall.advance("CONDITIONED_SEALED")

            truth_receipts = []
            truth_values = []
            source_parent_rows = []
            for parent in PARENT_IDS:
                adapter = adapters[parent]
                parent_truths = []
                for frame in adapter.roster.score:
                    prediction = conditioned_records[(parent, frame.frame_id)]
                    source_frame = _source_frame(
                        adapter,
                        frame.frame_id,
                        SourcePhase.SCORE_SOURCE,
                        archives[f"IMU_ARCHIVE:{parent}"],
                        archives["CAMERA_IMU_CALIBRATION_ARCHIVE:ALL_THREE_SESSIONS"],
                        calibration_binding,
                    )
                    truth = materialize_score_truth(
                        source_frame,
                        identities[parent],
                        tuple(int(item) for item in prediction["output_hw"]),
                    )
                    parent_truths.append(truth)
                    truth_values.append(truth)
                    receipt = writer.write_npz(f"phase-d/truth/{parent}/{frame.frame_id}.npz", _truth_arrays(truth))
                    truth_receipts.append(_record_receipt(parent, frame.frame_id, receipt))
                source_parent_rows.append(
                    source_parent_summary(parent, adapter.roster.eligible_count, contexts[parent], parent_truths)
                )
            _truth_completion, truth_completion_receipt = _write_sealed(
                writer,
                "phase-d/truth-completion.json",
                {
                    "schema": PHASE_SCHEMA,
                    "phase": "SCORE_SOURCE_TRUTH_SEALED",
                    "record_count": len(truth_receipts),
                    "records": truth_receipts,
                    "predecessor": _predecessor("phase-c/conditioned-factor-completion.json", conditioned_completion_receipt),
                },
            )
            truth_records = _reload_records(writer, truth_receipts, TRUTH_ARRAY_KEYS)
            firewall.advance("SCORE_SOURCE")
            source_summary, _source_summary_receipt = _write_sealed(
                writer,
                "source-summary.json",
                {
                    "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_source_summary.v1",
                    "phase": "SOURCE_SUMMARY_SEALED",
                    "parents": source_parent_rows,
                    "predecessor": _predecessor("phase-d/truth-completion.json", truth_completion_receipt),
                },
            )
            scientific = score_or_not_evaluable(
                protocol,
                {"parents": source_parent_rows},
                _metric_frames(conditioned_records, truth_records),
            )
            writer.write_json("access-events.json", _sealed({"schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_access_events.v1", "events": firewall.events}))
            writer.write_json("summary.json", scientific)
            result = _sealed(
                {
                    "schema": RESULT_SCHEMA,
                    "terminal": scientific["terminal"],
                    "execution_valid": True,
                    "summary_sha256": scientific["content_sha256"],
                    "source_summary_sha256": source_summary["content_sha256"],
                    "model_inference_calls": 36,
                    "source_score_frame_count": 36,
                    "training_steps": 0,
                    "reducer_calls": 0,
                    "network_requests": 0,
                    "device_calls_beyond_cuda_model": 0,
                }
            )
            writer.write_json("result.json", result)
            writer.finalize(scientific["terminal"])
        validation = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.independent_validator",
                "--root",
                str(writer.root),
                "--protocol",
                str(PROTOCOL_PATH),
            ],
            cwd=str(PROTOCOL_PATH.parents[3]),
            text=True,
            capture_output=True,
            check=False,
        )
        require(validation.returncode == 0, "F2_INDEPENDENT_VALIDATION_FAILED", validation.stdout + validation.stderr)
        return result
    except Exception as error:
        if "manifest.json" not in writer.files:
            _close_failure(writer, firewall, error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:  # noqa: BLE001 - CLI must serialize every fail-closed terminal.
        print(json.dumps({"passed": False, "error_code": getattr(error, "code", type(error).__name__), "message": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"passed": True, "terminal": result["terminal"], "content_sha256": result["content_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
