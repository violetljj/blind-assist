#!/usr/bin/env python3
"""Validate hash-bound per-frame USTRF U0 prediction evidence.

This module does not score a model.  It proves that every submitted U0 alert
summary is derived from an ordered, truth-bound shared-decision-kernel trace
and that the implementation, artifact, thresholds, and execution receipt are
real local files with matching hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPORT_SCHEMA = "blindassist_ustrf_sc_u0_prediction_evidence_admission_v2"


class ContractError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ContractError(f"{where} must be a lowercase SHA-256")
    return value


def _require_text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where} must be a non-empty string")
    return value


def _local(root: Path, relative: Any, *, where: str) -> Path:
    text = _require_text(relative, where=where)
    path = (root / text).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"{where} escapes prediction root") from error
    if not path.is_file():
        raise ContractError(f"{where} is not a local file")
    return path


def _load_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {where}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{where} must contain a JSON object")
    return value


def _bound_file(root: Path, relative: Any, expected_sha: Any, *, where: str) -> Path:
    expected = _require_sha(expected_sha, where=f"{where}.sha256")
    path = _local(root, relative, where=f"{where}.path")
    if _sha256(path) != expected:
        raise ContractError(f"{where} SHA-256 mismatch")
    return path


def _truth_ledgers(
    manifest: Mapping[str, Any], *, truth_root: Path,
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    by_episode: dict[str, list[dict[str, Any]]] = {}
    canonical_rows: list[dict[str, Any]] = []
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ContractError("truth manifest episodes must be a list")
    for truth in sorted(episodes, key=lambda row: str(row.get("episode_id"))):
        episode_id = _require_text(truth.get("episode_id"), where="truth episode_id")
        ledger_path = _bound_file(
            truth_root,
            truth.get("capture_frame_ledger_path"),
            truth.get("capture_frame_ledger_sha256"),
            where=f"truth/{episode_id}/capture_frame_ledger",
        )
        ledger = _load_json(ledger_path, where=f"truth/{episode_id}/capture_frame_ledger")
        frames = ledger.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ContractError(f"truth/{episode_id} frame ledger must be non-empty")
        normalized: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ContractError(f"truth/{episode_id}/frames[{index}] must be an object")
            expected = {
                "frame_id": frame.get("frame_id"),
                "frame_index": frame.get("frame_index"),
                "capture_timestamp_ns": frame.get("capture_timestamp_ns"),
                "video_pts_ms": frame.get("video_pts_ms"),
                "frame_payload_sha256": frame.get("frame_payload_sha256"),
            }
            _require_text(expected["frame_id"], where=f"truth/{episode_id}/frames[{index}].frame_id")
            _require_sha(expected["frame_payload_sha256"], where=f"truth/{episode_id}/frames[{index}].frame_payload_sha256")
            if expected["frame_index"] != index:
                raise ContractError(f"truth/{episode_id} frame indices are not contiguous")
            for key in ("capture_timestamp_ns", "video_pts_ms"):
                if not isinstance(expected[key], int) or isinstance(expected[key], bool) or expected[key] < 0:
                    raise ContractError(f"truth/{episode_id}/frames[{index}].{key} must be non-negative integer")
            normalized.append(expected)
        by_episode[episode_id] = normalized
        canonical_rows.append({
            "episode_id": episode_id,
            "capture_frame_ledger_sha256": truth.get("capture_frame_ledger_sha256"),
            "frames": normalized,
        })
    return _canonical_sha256(canonical_rows), by_episode


def _validate_decision_cadence(frames: list[dict[str, Any]], evidence: Mapping[str, Any], *, where: str) -> None:
    cadence = evidence.get("decision_cadence")
    expected = {
        "schema": "blindassist_ustrf_sc_u0_decision_cadence_v1",
        "canonical_step_ms": 500,
        "frame_selection_policy": "ledger_exact_grid_from_zero_v1",
        "missing_grid_frame_policy": "fail_episode_v1",
        "interpolation_policy": "forbidden",
        "episode_reset_policy": "reset_before_first_frame_v1",
        "route_sample_policy": "latest_valid_generated_at_or_before_frame_v1",
    }
    if cadence != expected:
        raise ContractError("U0 decision cadence contract is not the frozen 500ms exact-grid policy")
    for index, frame in enumerate(frames):
        if frame.get("video_pts_ms") != index * cadence["canonical_step_ms"]:
            raise ContractError(f"{where}/frames[{index}] is outside the canonical decision grid")
        if index and frame.get("capture_timestamp_ns", 0) <= frames[index - 1].get("capture_timestamp_ns", 0):
            raise ContractError(f"{where}/frames[{index}] capture timestamp is not strictly increasing")


def _validate_decision(
    decision: Any,
    *,
    contract: Mapping[str, Any],
    adapter_id: str,
    event_identity_policy: str,
    where: str,
) -> bool:
    if not isinstance(decision, dict):
        raise ContractError(f"{where}.decision must be an object")
    allowed_risks = set(contract.get("allowed_risk_levels", []))
    if decision.get("raw_risk_level") not in allowed_risks or decision.get("stable_risk_level") not in allowed_risks:
        raise ContractError(f"{where}.decision risk level is outside the contract")
    event_id = decision.get("event_id")
    if event_id is not None and (not isinstance(event_id, str) or not event_id.strip()):
        raise ContractError(f"{where}.decision.event_id must be null or a non-empty string")
    if decision.get("event_state") not in set(contract.get("allowed_event_states", [])):
        raise ContractError(f"{where}.decision.event_state is outside the contract")
    if (event_id is None) != (decision.get("event_state") is None):
        raise ContractError(f"{where}.decision event_id/event_state nullability mismatch")
    receipt = decision.get("feedback_receipt")
    if not isinstance(receipt, dict):
        raise ContractError(f"{where}.decision.feedback_receipt must be an object")
    outcome = receipt.get("outcome")
    if outcome not in set(contract.get("allowed_feedback_outcomes", [])):
        raise ContractError(f"{where}.feedback outcome is outside the contract")
    if not isinstance(receipt.get("delivered"), bool):
        raise ContractError(f"{where}.feedback delivered must be boolean")
    delivered = receipt["delivered"]
    if delivered != (outcome == "TRIGGERED"):
        raise ContractError(f"{where}.feedback delivered/outcome semantics mismatch")
    reason_mapping = contract.get("kernel_feedback_reason_to_outcome")
    kernel_reason = receipt.get("kernel_feedback_reason")
    if not isinstance(reason_mapping, dict) or kernel_reason not in reason_mapping:
        raise ContractError(f"{where}.feedback kernel reason is outside the frozen mapping")
    if reason_mapping[kernel_reason] != outcome:
        raise ContractError(f"{where}.feedback kernel reason/outcome mapping mismatch")
    if delivered and decision.get("stable_risk_level") == "NONE":
        raise ContractError(f"{where}.triggered feedback requires stable risk")
    if event_identity_policy == "kernel_native_required_v1" and delivered and event_id is None:
        raise ContractError(f"{where}.triggered feedback requires kernel-native event identity")
    if event_identity_policy not in {"kernel_native_required_v1", "kernel_native_optional_v1"}:
        raise ContractError(f"{where}.event identity policy is unknown")
    if receipt.get("adapter_id") != contract.get("feedback_adapter_id"):
        raise ContractError(f"{where}.feedback adapter mismatch")
    if receipt.get("kernel_contract_id") != contract.get("shared_decision_kernel_contract_id"):
        raise ContractError(f"{where}.feedback kernel contract mismatch")
    if decision.get("candidate_adapter_id") != adapter_id:
        raise ContractError(f"{where}.decision candidate adapter mismatch")
    return delivered


def _validate_android_backend_receipt(
    output: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    request_sha256: str,
    threshold_config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    where: str,
) -> None:
    receipt_contract = evidence.get("android_backend_receipt_contract")
    if not isinstance(receipt_contract, dict):
        raise ContractError("U0 evidence contract lacks android_backend_receipt_contract")
    required_backend = receipt_contract.get("required_for_backend_id")
    if request.get("kernel_execution_backend_id") != required_backend:
        if output.get("android_backend_receipt") is not None:
            raise ContractError(f"{where}.android_backend_receipt is forbidden for a non-Android backend")
        return
    receipt = output.get("android_backend_receipt")
    if not isinstance(receipt, dict):
        raise ContractError(f"{where}.android_backend_receipt is required")
    expected = {
        "schema": receipt_contract.get("schema"),
        "backend_id": required_backend,
        "adapter_output_origin": receipt_contract.get("adapter_output_origin"),
        "instrumentation_component": receipt_contract.get("instrumentation_component"),
        "target_package": receipt_contract.get("target_package"),
        "shared_decision_kernel_contract_id": request.get("shared_decision_kernel_contract_id"),
        "request_sha256": request_sha256,
        "sanitized_inference_manifest_sha256": request.get("sanitized_inference_manifest_sha256"),
        "input_video_sha256": request.get("input_video_sha256"),
        "source_capture_frame_ledger_sha256": request.get("source_capture_frame_ledger_sha256"),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"{where}.android_backend_receipt.{key} mismatch")
    threshold_bindings = receipt_contract.get("threshold_config_receipt_bindings")
    if not isinstance(threshold_bindings, list) or not threshold_bindings:
        raise ContractError("android backend receipt threshold bindings are invalid")
    for key in threshold_bindings:
        if not isinstance(key, str) or key not in threshold_config or receipt.get(key) != threshold_config.get(key):
            raise ContractError(f"{where}.android_backend_receipt.{key} is not threshold-bound")
    for key in (
        "build_fingerprint_sha256", "target_apk_sha256", "test_apk_sha256",
        "model_asset_sha256", "labels_asset_sha256", "device_adapter_implementation_sha256",
        "host_adapter_implementation_sha256",
    ):
        _require_sha(receipt.get(key), where=f"{where}.android_backend_receipt.{key}")
    for key in ("device_model", "target_version_name", "instrumentation_test_class"):
        _require_text(receipt.get(key), where=f"{where}.android_backend_receipt.{key}")
    for key in ("target_version_code", "device_api_level", "input_size"):
        value = receipt.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ContractError(f"{where}.android_backend_receipt.{key} must be a non-negative integer")
    frames = request.get("frames")
    decoded = receipt.get("decoded_frames")
    if (
        not isinstance(frames, list)
        or not isinstance(decoded, list)
        or len(decoded) != len(frames)
        or receipt.get("requested_frame_count") != len(frames)
        or receipt.get("decoded_frame_count") != len(frames)
    ):
        raise ContractError(f"{where}.android_backend_receipt decoded frame inventory mismatch")
    maximum_error = receipt.get("maximum_decode_pts_error_us")
    if not isinstance(maximum_error, int) or isinstance(maximum_error, bool) or maximum_error < 0:
        raise ContractError(f"{where}.android_backend_receipt maximum decode PTS error is invalid")
    for index, (row, frame) in enumerate(zip(decoded, frames)):
        frame_where = f"{where}.android_backend_receipt.decoded_frames[{index}]"
        if not isinstance(row, dict):
            raise ContractError(f"{frame_where} must be an object")
        requested_pts_us = frame.get("video_pts_ms") * 1_000
        if (
            row.get("frame_id") != frame.get("frame_id")
            or row.get("video_pts_ms") != frame.get("video_pts_ms")
            or row.get("requested_pts_us") != requested_pts_us
        ):
            raise ContractError(f"{frame_where} frame identity mismatch")
        selected_pts_us = row.get("selected_source_pts_us")
        pts_error_us = row.get("pts_error_us")
        if (
            not isinstance(selected_pts_us, int)
            or isinstance(selected_pts_us, bool)
            or selected_pts_us < 0
            or not isinstance(pts_error_us, int)
            or isinstance(pts_error_us, bool)
            or pts_error_us != abs(selected_pts_us - requested_pts_us)
            or pts_error_us > maximum_error
        ):
            raise ContractError(f"{frame_where} PTS binding is invalid")
        _require_sha(row.get("decoded_rgba8888_sha256"), where=f"{frame_where}.decoded_rgba8888_sha256")
        _require_sha(row.get("encoded_sample_sha256"), where=f"{frame_where}.encoded_sample_sha256")
        for key in ("width", "height"):
            value = row.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ContractError(f"{frame_where}.{key} must be positive integer")
        for key in ("decode_duration_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "total_detect_ms"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ContractError(f"{frame_where}.{key} must be non-negative number")
        detections = row.get("detection_count")
        if not isinstance(detections, int) or isinstance(detections, bool) or detections < 0:
            raise ContractError(f"{frame_where}.detection_count must be non-negative integer")


def _validate_bbox_route_conditioning_receipt(
    output: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    threshold_config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    where: str,
) -> None:
    contract = evidence.get("route_conditioning_receipt_contract")
    if not isinstance(contract, dict):
        raise ContractError("U0 evidence contract lacks route_conditioning_receipt_contract")
    required_adapter = contract.get("required_for_candidate_adapter_id")
    required_backend = contract.get("required_for_backend_id")
    receipt = output.get("route_conditioning_receipt")
    if request.get("candidate_adapter_id") != required_adapter or request.get("kernel_execution_backend_id") != required_backend:
        if request.get("route_input_policy") == "no_route_input_v1" and receipt is not None:
            raise ContractError(f"{where}.route_conditioning_receipt is forbidden for no-route adapter")
        return
    if not isinstance(receipt, dict):
        raise ContractError(f"{where}.route_conditioning_receipt is required")
    expected = {
        "schema": contract.get("schema"),
        "arm_id": request.get("arm_id"),
        "candidate_adapter_id": required_adapter,
        "route_input_policy": request.get("route_input_policy"),
        "route_input_sha256": request.get("adapter_route_input_sha256"),
        "route_episode_id": request.get("episode_id"),
        "route_gate_contract_id": contract.get("route_gate_contract_id"),
        "unknown_route_policy": contract.get("unknown_route_policy"),
        "route_sample_policy": request.get("decision_cadence", {}).get("route_sample_policy"),
        "future_inputs_used": False,
        "risk_model_inferred_route": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"{where}.route_conditioning_receipt.{key} mismatch")
    for key in ("route_parent_source_id", "route_provider_type", "route_provider_id", "projection_receipt_id"):
        _require_text(receipt.get(key), where=f"{where}.route_conditioning_receipt.{key}")
    bindings = contract.get("threshold_config_receipt_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ContractError("route conditioning receipt threshold bindings are invalid")
    for key in bindings:
        if not isinstance(key, str) or key not in threshold_config or receipt.get(key) != threshold_config.get(key):
            raise ContractError(f"{where}.route_conditioning_receipt.{key} is not threshold-bound")
    _require_sha(
        receipt.get("route_gate_implementation_sha256"),
        where=f"{where}.route_conditioning_receipt.route_gate_implementation_sha256",
    )
    request_frames = request.get("frames")
    route_frames = receipt.get("frames")
    backend_frames = output.get("android_backend_receipt", {}).get("decoded_frames")
    if (
        not isinstance(request_frames, list)
        or not isinstance(route_frames, list)
        or not isinstance(backend_frames, list)
        or len(route_frames) != len(request_frames)
        or len(backend_frames) != len(request_frames)
        or receipt.get("frame_count") != len(request_frames)
    ):
        raise ContractError(f"{where}.route_conditioning_receipt frame inventory mismatch")
    allowed_reasons = {
        "ROUTE_USABLE", "NO_CAUSAL_ROUTE_SAMPLE", "ROUTE_MARKED_INVALID", "ROUTE_STALE",
        "ROUTE_TOO_OLD", "ROUTE_LOW_CONFIDENCE", "ROUTE_WAYPOINT_CONTRACT_INVALID",
    }
    for index, (row, request_frame, backend_frame) in enumerate(zip(route_frames, request_frames, backend_frames)):
        frame_where = f"{where}.route_conditioning_receipt.frames[{index}]"
        if not isinstance(row, dict) or not isinstance(backend_frame, dict):
            raise ContractError(f"{frame_where} must be an object")
        frame_ms = request_frame.get("video_pts_ms")
        if row.get("frame_id") != request_frame.get("frame_id") or row.get("frame_timestamp_ms") != frame_ms:
            raise ContractError(f"{frame_where} identity mismatch")
        route_usable = row.get("route_usable")
        reason = row.get("gate_reason")
        if not isinstance(route_usable, bool) or reason not in allowed_reasons or route_usable != (reason == "ROUTE_USABLE"):
            raise ContractError(f"{frame_where} route usability/reason mismatch")
        sample_timestamp = row.get("selected_sample_timestamp_ms")
        valid_until = row.get("selected_valid_until_timestamp_ms")
        confidence = row.get("selected_route_confidence")
        if sample_timestamp is not None and (
            not isinstance(sample_timestamp, int) or isinstance(sample_timestamp, bool) or sample_timestamp > frame_ms
        ):
            raise ContractError(f"{frame_where} selected a future route sample")
        if route_usable and (
            not isinstance(valid_until, int)
            or isinstance(valid_until, bool)
            or valid_until < frame_ms
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or confidence < threshold_config.get("minimum_route_confidence")
        ):
            raise ContractError(f"{frame_where} usable route validity/confidence mismatch")
        input_count = row.get("input_detection_count")
        retained_count = row.get("retained_detection_count")
        kernel_count = backend_frame.get("kernel_input_detection_count")
        detector_count = backend_frame.get("detection_count")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (
            input_count, retained_count, kernel_count, detector_count
        )):
            raise ContractError(f"{frame_where} detection counts are invalid")
        expected_input_count = detector_count if route_usable else 0
        if input_count != expected_input_count or retained_count != kernel_count or retained_count > input_count:
            raise ContractError(f"{frame_where} kernel input count is not route-gate-bound")
        detections = row.get("detections")
        if not isinstance(detections, list) or len(detections) != input_count:
            raise ContractError(f"{frame_where} detection inventory mismatch")
        kept_count = 0
        for detection_index, detection in enumerate(detections):
            detection_where = f"{frame_where}.detections[{detection_index}]"
            if not isinstance(detection, dict) or detection.get("detection_index") != detection_index:
                raise ContractError(f"{detection_where} identity mismatch")
            distance = detection.get("minimum_route_distance_px")
            corridor = detection.get("corridor_half_width_px")
            kept = detection.get("kept")
            if (
                not isinstance(distance, (int, float)) or isinstance(distance, bool) or distance < 0
                or not isinstance(corridor, (int, float)) or isinstance(corridor, bool) or corridor < 0
                or not isinstance(kept, bool) or kept != (distance <= corridor)
            ):
                raise ContractError(f"{detection_where} gate arithmetic mismatch")
            kept_count += int(kept)
        if kept_count != retained_count:
            raise ContractError(f"{frame_where} retained detection count mismatch")


def _validate_dense_risk_evidence_receipt(
    output: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    threshold_config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    where: str,
) -> None:
    contract = evidence.get("dense_risk_evidence_receipt_contract")
    if not isinstance(contract, dict):
        raise ContractError("U0 evidence contract lacks dense_risk_evidence_receipt_contract")
    adapter_ids = contract.get("required_for_candidate_adapter_ids")
    if not isinstance(adapter_ids, list) or not adapter_ids or any(not isinstance(value, str) for value in adapter_ids):
        raise ContractError("dense risk evidence adapter inventory is invalid")
    required = (
        request.get("candidate_adapter_id") in adapter_ids
        and request.get("kernel_execution_backend_id") == contract.get("required_for_backend_id")
    )
    receipt = output.get("dense_risk_evidence_receipt")
    if not required:
        if receipt is not None:
            raise ContractError(f"{where}.dense_risk_evidence_receipt is forbidden")
        return
    if not isinstance(receipt, dict):
        raise ContractError(f"{where}.dense_risk_evidence_receipt is required")
    expected = {
        "schema": contract.get("schema"),
        "arm_id": request.get("arm_id"),
        "candidate_adapter_id": request.get("candidate_adapter_id"),
        "route_input_policy": request.get("route_input_policy"),
        "route_input_sha256": request.get("adapter_route_input_sha256"),
        "episode_id": request.get("episode_id"),
        "fold_artifact_sha256": request.get("fold_artifact_sha256"),
        "fold_training_input_manifest_sha256": request.get("fold_training_input_manifest_sha256"),
        "fold_training_receipt_sha256": request.get("fold_training_receipt_sha256"),
        "dense_field_contract_id": contract.get("dense_field_contract_id"),
        "normalization_contract_id": contract.get("normalization_contract_id"),
        "shared_risk_evidence_input_contract_id": contract.get("shared_risk_evidence_input_contract_id"),
        "teacher_output_role": contract.get("teacher_output_role"),
        "blind_accessed": False,
        "future_inputs_used": False,
        "human_event_truth_used": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"{where}.dense_risk_evidence_receipt.{key} mismatch")
    bindings = contract.get("threshold_config_receipt_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ContractError("dense risk evidence threshold bindings are invalid")
    for key in bindings:
        if not isinstance(key, str) or key not in threshold_config or receipt.get(key) != threshold_config.get(key):
            raise ContractError(f"{where}.dense_risk_evidence_receipt.{key} is not threshold-bound")
    low = threshold_config.get("low_threshold")
    medium = threshold_config.get("medium_threshold")
    high = threshold_config.get("high_threshold")
    peak_weight = threshold_config.get("local_peak_weight")
    maximum_unknown = threshold_config.get("maximum_route_unknown_fraction")
    if (
        any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in (low, medium, high, peak_weight, maximum_unknown))
        or not (0.0 <= low <= medium <= high <= 1.0)
        or not (0.0 <= peak_weight <= 1.0)
        or not (0.0 <= maximum_unknown <= 1.0)
    ):
        raise ContractError("dense risk evidence thresholds are invalid")
    provenance = receipt.get("teacher_provenance")
    if not isinstance(provenance, dict):
        raise ContractError(f"{where}.dense_risk_evidence_receipt.teacher_provenance is required")
    for key in ("model_name", "model_version", "license_identifier", "inference_runtime"):
        _require_text(provenance.get(key), where=f"{where}.dense_risk_evidence_receipt.teacher_provenance.{key}")
    for key in ("model_weights_sha256", "inference_implementation_sha256"):
        _require_sha(provenance.get(key), where=f"{where}.dense_risk_evidence_receipt.teacher_provenance.{key}")
    request_frames = request.get("frames")
    output_frames = output.get("frames")
    receipt_frames = receipt.get("frames")
    if (
        not isinstance(request_frames, list)
        or not isinstance(output_frames, list)
        or not isinstance(receipt_frames, list)
        or len(receipt_frames) != len(request_frames)
        or len(output_frames) != len(request_frames)
        or receipt.get("frame_count") != len(request_frames)
    ):
        raise ContractError(f"{where}.dense_risk_evidence_receipt frame inventory mismatch")
    for index, (row, request_frame, output_frame) in enumerate(zip(receipt_frames, request_frames, output_frames)):
        frame_where = f"{where}.dense_risk_evidence_receipt.frames[{index}]"
        if not isinstance(row, dict) or not isinstance(output_frame, dict):
            raise ContractError(f"{frame_where} must be an object")
        frame_ms = request_frame.get("video_pts_ms")
        if (
            row.get("frame_id") != request_frame.get("frame_id")
            or row.get("frame_timestamp_ms") != frame_ms
            or row.get("observed_at_ms") != frame_ms
            or row.get("source_frame_payload_sha256") != request_frame.get("frame_payload_sha256")
        ):
            raise ContractError(f"{frame_where} frame/time identity mismatch")
        valid_until = row.get("valid_until_ms")
        evidence_count = row.get("risk_evidence_count")
        field_cell_count = row.get("field_cell_count")
        if (
            not isinstance(valid_until, int) or isinstance(valid_until, bool) or valid_until < frame_ms
            or not isinstance(evidence_count, int) or isinstance(evidence_count, bool) or evidence_count <= 0
            or not isinstance(field_cell_count, int) or isinstance(field_cell_count, bool) or field_cell_count < evidence_count
        ):
            raise ContractError(f"{frame_where} validity/evidence inventory is invalid")
        _require_sha(row.get("source_field_sha256"), where=f"{frame_where}.source_field_sha256")
        route_id = _require_text(row.get("route_intent_id"), where=f"{frame_where}.route_intent_id")
        if row.get("event_key") != f"{request.get('episode_id')}:{route_id}":
            raise ContractError(f"{frame_where} event key mismatch")
        sources = row.get("risk_sources")
        if (
            not isinstance(sources, list)
            or not sources
            or any(not isinstance(source, str) or not source.strip() for source in sources)
            or sources != sorted(set(sources))
        ):
            raise ContractError(f"{frame_where} risk source inventory is invalid")
        average = row.get("route_intrusion_score")
        peak = row.get("maximum_route_cell_risk")
        unknown = row.get("route_unknown_fraction")
        normalized = row.get("normalized_risk_score")
        if (
            any(not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0 for value in (average, peak, unknown, normalized))
            or unknown > maximum_unknown
        ):
            raise ContractError(f"{frame_where} dense score/unknown value is invalid")
        expected_score = max(average, peak * peak_weight)
        if abs(normalized - expected_score) > 1e-6:
            raise ContractError(f"{frame_where} normalized risk score arithmetic mismatch")
        expected_level = "HIGH" if normalized >= high else "MEDIUM" if normalized >= medium else "LOW" if normalized >= low else "NONE"
        expected_direction = "NONE" if expected_level == "NONE" else "CENTER"
        decision = output_frame.get("decision")
        if (
            row.get("raw_risk_level") != expected_level
            or row.get("risk_direction") != expected_direction
            or not isinstance(decision, dict)
            or decision.get("raw_risk_level") != expected_level
        ):
            raise ContractError(f"{frame_where} normalized risk decision mismatch")


def _normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _required_arm_inventory(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = contract.get("required_arms")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ContractError("U0 required_arms must be an object list")
    result = {row.get("arm_id"): row for row in rows}
    if len(result) != len(rows) or None in result:
        raise ContractError("U0 required_arms identities must be unique")
    return result


def _validate_runner_and_registry(
    contract: Mapping[str, Any],
    predictions: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    prediction_root: Path,
    required_arms: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, dict[str, dict[str, Any]]]:
    runner_path = _bound_file(
        prediction_root,
        predictions.get("runner_implementation_path"),
        predictions.get("runner_implementation_file_sha256"),
        where="runner_implementation",
    )
    runner_sha = predictions["runner_implementation_sha256"]
    if runner_sha != evidence.get("runner_implementation_sha256") or _normalized_text_sha256(runner_path) != runner_sha:
        raise ContractError("runner implementation differs from the frozen contract")
    kernel_path = _bound_file(
        prediction_root,
        predictions.get("shared_decision_kernel_implementation_path"),
        predictions.get("shared_decision_kernel_implementation_file_sha256"),
        where="shared_decision_kernel_implementation",
    )
    kernel_sha = predictions["shared_decision_kernel_implementation_sha256"]
    if kernel_sha != evidence.get("shared_decision_kernel_implementation_sha256") or _normalized_text_sha256(kernel_path) != kernel_sha:
        raise ContractError("shared decision kernel implementation differs from the frozen contract")
    expected_dependencies = evidence.get("shared_decision_kernel_dependency_sha256")
    dependency_rows = predictions.get("shared_decision_kernel_dependencies")
    if not isinstance(expected_dependencies, dict) or not isinstance(dependency_rows, list):
        raise ContractError("shared decision kernel dependency inventory is missing")
    if (
        any(not isinstance(row, dict) for row in dependency_rows)
        or len({row.get("name") for row in dependency_rows}) != len(dependency_rows)
        or {row.get("name") for row in dependency_rows} != set(expected_dependencies)
    ):
        raise ContractError("shared decision kernel dependency inventory mismatch")
    for row in dependency_rows:
        name = row["name"]
        path = _bound_file(
            prediction_root,
            row.get("path"),
            row.get("file_sha256"),
            where=f"shared_decision_kernel_dependencies/{name}",
        )
        if row.get("normalized_text_sha256") != expected_dependencies[name] or _normalized_text_sha256(path) != expected_dependencies[name]:
            raise ContractError(f"shared decision kernel dependency differs: {name}")
    if predictions.get("kernel_execution_backend_id") != evidence.get("kernel_execution_backend_id"):
        raise ContractError("prediction kernel execution backend mismatch")
    registry_path = _bound_file(
        prediction_root,
        predictions.get("adapter_registry_path"),
        predictions.get("adapter_registry_sha256"),
        where="adapter_registry",
    )
    registry_sha = predictions["adapter_registry_sha256"]
    registry = _load_json(registry_path, where="adapter_registry")
    if registry.get("schema") != evidence.get("adapter_registry_schema") or registry.get("contract_id") != contract.get("contract_id"):
        raise ContractError("adapter registry identity mismatch")
    if registry.get("synthetic_fixture") is not predictions.get("synthetic_fixture"):
        raise ContractError("adapter registry synthetic_fixture mismatch")
    for key in ("blind_accessed", "future_inputs_used", "production_model_replacement_authorized"):
        if registry.get(key) is not False:
            raise ContractError(f"adapter registry must declare {key}=false")
    if registry.get("kernel_execution_backend_id") != evidence.get("kernel_execution_backend_id"):
        raise ContractError("adapter registry kernel backend mismatch")
    if registry.get("shared_decision_kernel_implementation_sha256") != evidence.get("shared_decision_kernel_implementation_sha256"):
        raise ContractError("adapter registry kernel implementation mismatch")
    if registry.get("shared_decision_kernel_dependency_sha256") != expected_dependencies:
        raise ContractError("adapter registry kernel dependency inventory mismatch")
    rows = registry.get("arms")
    if (
        not isinstance(rows, list)
        or len(rows) != len(required_arms)
        or any(not isinstance(row, dict) for row in rows)
        or len({row.get("arm_id") for row in rows}) != len(rows)
        or {row.get("arm_id") for row in rows} != set(required_arms)
    ):
        raise ContractError("adapter registry must contain every preregistered arm exactly once")
    by_arm = {row["arm_id"]: row for row in rows}
    for arm_id, row in by_arm.items():
        required = required_arms[arm_id]
        for key in ("candidate_adapter_id", "fit_policy", "event_identity_policy", "route_input_policy"):
            if row.get(key) != required.get(key):
                raise ContractError(f"adapter registry {arm_id}.{key} differs from preregistration")
        if row.get("runtime_id") not in evidence.get("allowed_adapter_runtime_ids", []):
            raise ContractError(f"adapter registry {arm_id}.runtime_id is not allowed")
    return runner_sha, registry_sha, by_arm


def _expected_fold_inventory(
    *,
    contract: Mapping[str, Any],
    truth_manifest: Mapping[str, Any],
    truth_manifest_sha256: str,
    arm_id: str,
    arm_contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    prediction_root: Path,
    evidence: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    inventory_path = _bound_file(
        prediction_root,
        arm.get("artifact_path"),
        arm.get("artifact_sha256"),
        where=f"{arm_id}/fold_artifact_inventory",
    )
    inventory = _load_json(inventory_path, where=f"{arm_id}/fold_artifact_inventory")
    expected_header = {
        "schema": evidence.get("fold_artifact_inventory_schema"),
        "contract_id": contract.get("contract_id"),
        "arm_id": arm_id,
        "candidate_adapter_id": arm_contract.get("candidate_adapter_id"),
        "fit_policy": arm_contract.get("fit_policy"),
        "truth_manifest_sha256": truth_manifest_sha256,
    }
    for key, expected in expected_header.items():
        if inventory.get(key) != expected:
            raise ContractError(f"{arm_id}/fold_artifact_inventory.{key} mismatch")
    sessions = sorted({row.get("session_id") for row in truth_manifest.get("episodes", [])})
    if any(not isinstance(session_id, str) or not session_id for session_id in sessions):
        raise ContractError("truth sessions must be non-empty strings")
    folds = inventory.get("folds")
    if (
        not isinstance(folds, list)
        or len(folds) != len(sessions)
        or any(not isinstance(row, dict) for row in folds)
        or len({row.get("held_out_session_id") for row in folds}) != len(folds)
        or {row.get("held_out_session_id") for row in folds} != set(sessions)
    ):
        raise ContractError(f"{arm_id} fold artifact inventory must cover every session exactly once")
    episodes_by_session: dict[str, list[str]] = {
        session_id: sorted(
            row["episode_id"] for row in truth_manifest["episodes"] if row.get("session_id") == session_id
        )
        for session_id in sessions
    }
    by_session: dict[str, dict[str, Any]] = {}
    for fold in folds:
        held_out = fold["held_out_session_id"]
        artifact_path = _bound_file(
            prediction_root,
            fold.get("artifact_path"),
            fold.get("artifact_sha256"),
            where=f"{arm_id}/{held_out}/artifact",
        )
        manifest_path = _bound_file(
            prediction_root,
            fold.get("training_input_manifest_path"),
            fold.get("training_input_manifest_sha256"),
            where=f"{arm_id}/{held_out}/training_input_manifest",
        )
        receipt_path = _bound_file(
            prediction_root,
            fold.get("training_receipt_path"),
            fold.get("training_receipt_sha256"),
            where=f"{arm_id}/{held_out}/training_receipt",
        )
        training_manifest = _load_json(manifest_path, where=f"{arm_id}/{held_out}/training_input_manifest")
        fit_policy = arm_contract.get("fit_policy")
        expected_training_sessions = [] if fit_policy == "fixed_no_fit_v1" else [value for value in sessions if value != held_out]
        expected_training_episodes = [] if fit_policy == "fixed_no_fit_v1" else sorted(
            episode_id
            for session_id in expected_training_sessions
            for episode_id in episodes_by_session[session_id]
        )
        expected_training = {
            "schema": evidence.get("fold_training_input_manifest_schema"),
            "contract_id": contract.get("contract_id"),
            "arm_id": arm_id,
            "candidate_adapter_id": arm_contract.get("candidate_adapter_id"),
            "fit_policy": fit_policy,
            "held_out_session_id": held_out,
            "truth_manifest_sha256": truth_manifest_sha256,
            "training_session_ids": expected_training_sessions,
            "training_episode_ids": expected_training_episodes,
            "held_out_inputs_used": False,
            "blind_accessed": False,
            "future_inputs_used": False,
        }
        if training_manifest != expected_training:
            raise ContractError(f"{arm_id}/{held_out} training input manifest is not the exact LOSO inventory")
        training_receipt = _load_json(receipt_path, where=f"{arm_id}/{held_out}/training_receipt")
        expected_receipt = {
            "schema": evidence.get("fold_training_receipt_schema"),
            "contract_id": contract.get("contract_id"),
            "arm_id": arm_id,
            "candidate_adapter_id": arm_contract.get("candidate_adapter_id"),
            "fit_policy": fit_policy,
            "held_out_session_id": held_out,
            "training_input_manifest_sha256": fold.get("training_input_manifest_sha256"),
            "artifact_sha256": fold.get("artifact_sha256"),
            "fit_executed": fit_policy == "leave_one_session_out_fit_v1",
            "held_out_inputs_used": False,
            "blind_accessed": False,
            "future_inputs_used": False,
            "provenance_completed": True,
            "failure_count": 0,
        }
        if training_receipt != expected_receipt:
            raise ContractError(f"{arm_id}/{held_out} training receipt mismatch")
        by_session[held_out] = {
            **fold,
            "artifact_path_resolved": artifact_path,
        }
    return by_session


def _shuffled_route_sources(truth_manifest: Mapping[str, Any]) -> dict[str, str]:
    by_session: dict[str, list[str]] = {}
    for row in truth_manifest.get("episodes", []):
        by_session.setdefault(row["session_id"], []).append(row["episode_id"])
    result: dict[str, str] = {}
    for session_id, episode_ids in by_session.items():
        ordered = sorted(episode_ids)
        if len(ordered) < 2:
            raise ContractError(f"shuffled route control requires at least two episodes in session {session_id}")
        for index, episode_id in enumerate(ordered):
            result[episode_id] = ordered[(index + 1) % len(ordered)]
    return result


def _validate_inference_manifest(
    *,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    arm_contract: Mapping[str, Any],
    episode: Mapping[str, Any],
    truth_by_id: Mapping[str, Mapping[str, Any]],
    expected_frames: list[dict[str, Any]],
    shuffled_sources: Mapping[str, str],
    summary: Mapping[str, Any],
    prediction_root: Path,
) -> tuple[dict[str, Any], str]:
    episode_id = episode["episode_id"]
    path = _bound_file(
        prediction_root,
        summary.get("sanitized_inference_manifest_path"),
        summary.get("sanitized_inference_manifest_sha256"),
        where=f"{arm_contract['arm_id']}/{episode_id}/sanitized_inference_manifest",
    )
    manifest = _load_json(path, where=f"{arm_contract['arm_id']}/{episode_id}/sanitized_inference_manifest")
    allowed_keys = {
        "schema", "contract_id", "arm_id", "episode_id", "input_video_path", "input_video_sha256",
        "capture_frame_ledger_path", "capture_frame_ledger_sha256", "truth_route_intent_sha256",
        "route_input_policy", "adapter_route_input_path", "adapter_route_input_sha256",
        "adapter_route_source_episode_id", "decision_cadence", "frames", "blind_accessed",
        "future_inputs_used", "review_fields_present", "adjudication_fields_present", "event_label_fields_present",
    }
    if set(manifest) != allowed_keys:
        raise ContractError(f"{arm_contract['arm_id']}/{episode_id} inference manifest key inventory is not sanitized")
    expected_header = {
        "schema": evidence.get("sanitized_inference_manifest_schema"),
        "contract_id": contract.get("contract_id"),
        "arm_id": arm_contract.get("arm_id"),
        "episode_id": episode_id,
        "input_video_sha256": episode.get("video_sha256"),
        "capture_frame_ledger_sha256": episode.get("capture_frame_ledger_sha256"),
        "truth_route_intent_sha256": episode.get("route_intent_sha256"),
        "route_input_policy": arm_contract.get("route_input_policy"),
        "decision_cadence": evidence.get("decision_cadence"),
        "frames": expected_frames,
        "blind_accessed": False,
        "future_inputs_used": False,
        "review_fields_present": False,
        "adjudication_fields_present": False,
        "event_label_fields_present": False,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ContractError(f"{arm_contract['arm_id']}/{episode_id} inference manifest {key} mismatch")
    video_path = _bound_file(
        prediction_root,
        manifest.get("input_video_path"),
        manifest.get("input_video_sha256"),
        where=f"{arm_contract['arm_id']}/{episode_id}/inference_video",
    )
    ledger_path = _bound_file(
        prediction_root,
        manifest.get("capture_frame_ledger_path"),
        manifest.get("capture_frame_ledger_sha256"),
        where=f"{arm_contract['arm_id']}/{episode_id}/inference_ledger",
    )
    del video_path, ledger_path
    policy = arm_contract.get("route_input_policy")
    route_path = manifest.get("adapter_route_input_path")
    route_sha = manifest.get("adapter_route_input_sha256")
    route_source = manifest.get("adapter_route_source_episode_id")
    if policy == "no_route_input_v1":
        if any(value is not None for value in (route_path, route_sha, route_source)):
            raise ContractError(f"{arm_contract['arm_id']}/{episode_id} baseline must receive no route input")
    elif policy == "episode_explicit_causal_route_v1":
        _bound_file(prediction_root, route_path, route_sha, where=f"{arm_contract['arm_id']}/{episode_id}/route_input")
        if route_sha != episode.get("route_intent_sha256") or route_source != episode_id:
            raise ContractError(f"{arm_contract['arm_id']}/{episode_id} explicit route input differs from truth-bound route")
    elif policy == "within_heldout_session_sorted_episode_cyclic_shift_one_v1":
        source_id = shuffled_sources[episode_id]
        source = truth_by_id[source_id]
        _bound_file(prediction_root, route_path, route_sha, where=f"{arm_contract['arm_id']}/{episode_id}/route_input")
        if route_sha != source.get("route_intent_sha256") or route_source != source_id or source_id == episode_id:
            raise ContractError(f"{arm_contract['arm_id']}/{episode_id} shuffled route input is not the frozen derangement")
    elif policy == "uniform_full_frame_equal_weight_v1":
        uniform_path = _bound_file(prediction_root, route_path, route_sha, where=f"{arm_contract['arm_id']}/{episode_id}/route_input")
        uniform = _load_json(uniform_path, where=f"{arm_contract['arm_id']}/{episode_id}/uniform_route")
        expected_uniform = {
            "schema": "blindassist_ustrf_sc_u0_uniform_route_control_v1",
            "contract_id": contract.get("contract_id"),
            "episode_id": episode_id,
            "field_definition": "full_frame_equal_weight",
            "constant_weight": 1.0,
            "uses_episode_route": False,
            "uses_labels": False,
            "future_inputs_used": False,
        }
        if uniform != expected_uniform or route_source is not None:
            raise ContractError(f"{arm_contract['arm_id']}/{episode_id} uniform route control mismatch")
    else:
        raise ContractError(f"{arm_contract['arm_id']}/{episode_id} route input policy is unknown")
    return manifest, str(summary["sanitized_inference_manifest_sha256"])


def validate_bundle(
    contract: Mapping[str, Any],
    truth_manifest: Mapping[str, Any],
    predictions: Mapping[str, Any],
    *,
    truth_root: Path,
    prediction_root: Path,
) -> dict[str, Any]:
    evidence = contract.get("prediction_evidence_contract")
    if not isinstance(evidence, dict):
        raise ContractError("U0 contract lacks prediction_evidence_contract")
    if evidence.get("schema") != "blindassist_ustrf_sc_u0_prediction_evidence_contract_v2":
        raise ContractError("unexpected prediction evidence contract schema")
    if evidence.get("adapter_input_policy") != "sanitized_inference_manifest_only_v1":
        raise ContractError("U0 adapter input policy must remain sanitized and label-free")
    if contract.get("prediction_schema") != "blindassist_ustrf_sc_u0_six_arm_predictions_v2":
        raise ContractError("unexpected U0 prediction bundle schema contract")
    expected_ledger_sha, truth_frames = _truth_ledgers(truth_manifest, truth_root=truth_root)
    for episode_id, frames in truth_frames.items():
        _validate_decision_cadence(frames, evidence, where=f"truth/{episode_id}")
    truth_by_id = {row["episode_id"]: row for row in truth_manifest["episodes"]}
    if len(truth_by_id) != len(truth_manifest["episodes"]):
        raise ContractError("truth episode identities must be unique")
    required_arms = _required_arm_inventory(contract)
    arms = predictions.get("arms")
    if (
        not isinstance(arms, list)
        or len(arms) != len(required_arms)
        or any(not isinstance(arm, dict) for arm in arms)
        or len({arm.get("arm_id") for arm in arms}) != len(arms)
        or {arm.get("arm_id") for arm in arms} != set(required_arms)
    ):
        raise ContractError("prediction arms must contain every preregistered arm exactly once")
    runner_sha, registry_sha, registry_by_arm = _validate_runner_and_registry(
        contract,
        predictions,
        evidence=evidence,
        prediction_root=prediction_root,
        required_arms=required_arms,
    )
    shuffled_sources = _shuffled_route_sources(truth_manifest)
    truth_manifest_sha = _require_sha(predictions.get("truth_manifest_sha256"), where="truth_manifest_sha256")

    trace_count = 0
    frame_count = 0
    derived_alert_count = 0
    arm_reports: dict[str, Any] = {}
    for arm in arms:
        arm_id = _require_text(arm.get("arm_id"), where="prediction arm_id")
        arm_contract = required_arms[arm_id]
        adapter_id = arm_contract.get("candidate_adapter_id")
        if arm.get("candidate_adapter_id") != adapter_id or not isinstance(adapter_id, str):
            raise ContractError(f"{arm_id}.candidate_adapter_id differs from the preregistered adapter")
        for key in ("fit_policy", "event_identity_policy", "route_input_policy"):
            if arm.get(key) != arm_contract.get(key):
                raise ContractError(f"{arm_id}.{key} differs from preregistration")
        if arm.get("shared_decision_kernel_contract_id") != evidence.get("shared_decision_kernel_contract_id"):
            raise ContractError(f"{arm_id} bypasses the shared decision kernel")
        if arm.get("shared_decision_kernel_implementation_sha256") != evidence.get("shared_decision_kernel_implementation_sha256"):
            raise ContractError(f"{arm_id} shared decision kernel implementation mismatch")
        if arm.get("kernel_execution_backend_id") != evidence.get("kernel_execution_backend_id"):
            raise ContractError(f"{arm_id} kernel execution backend mismatch")
        if arm.get("adapter_runtime_id") not in evidence.get("allowed_adapter_runtime_ids", []):
            raise ContractError(f"{arm_id} adapter runtime is not allowed")
        if arm.get("ordered_frame_ledger_sha256") != expected_ledger_sha:
            raise ContractError(f"{arm_id}.ordered_frame_ledger_sha256 is not recomputed truth-frame evidence")
        bound_arm_files: dict[str, Path] = {}
        for stem, sha_key in (("implementation", "implementation_sha256"), ("threshold_config", "threshold_config_sha256")):
            bound_arm_files[stem] = _bound_file(
                prediction_root,
                arm.get(f"{stem}_path"),
                arm.get(sha_key),
                where=f"{arm_id}/{stem}",
            )
        threshold_config = _load_json(bound_arm_files["threshold_config"], where=f"{arm_id}/threshold_config")
        registry_row = registry_by_arm[arm_id]
        if (
            registry_row.get("implementation_sha256") != arm.get("implementation_sha256")
            or registry_row.get("threshold_config_sha256") != arm.get("threshold_config_sha256")
        ):
            raise ContractError(f"{arm_id} registry/arm input hash mismatch")
        folds = _expected_fold_inventory(
            contract=contract,
            truth_manifest=truth_manifest,
            truth_manifest_sha256=truth_manifest_sha,
            arm_id=arm_id,
            arm_contract=arm_contract,
            arm=arm,
            prediction_root=prediction_root,
            evidence=evidence,
        )
        registry_folds = registry_row.get("folds")
        expected_registry_folds = sorted(
            ({
                "held_out_session_id": session_id,
                "artifact_sha256": row["artifact_sha256"],
                "training_input_manifest_sha256": row["training_input_manifest_sha256"],
                "training_receipt_sha256": row["training_receipt_sha256"],
            } for session_id, row in folds.items()),
            key=lambda value: value["held_out_session_id"],
        )
        if registry_folds != expected_registry_folds:
            raise ContractError(f"{arm_id} registry/fold artifact inventory mismatch")

        receipt_path = _bound_file(
            prediction_root,
            arm.get("execution_receipt_path"),
            arm.get("execution_receipt_sha256"),
            where=f"{arm_id}/execution_receipt",
        )
        receipt = _load_json(receipt_path, where=f"{arm_id}/execution_receipt")
        expected_receipt = {
            "schema": evidence.get("execution_receipt_schema"),
            "contract_id": contract.get("contract_id"),
            "arm_id": arm_id,
            "candidate_adapter_id": adapter_id,
            "adapter_runtime_id": arm.get("adapter_runtime_id"),
            "runner_implementation_sha256": runner_sha,
            "adapter_registry_sha256": registry_sha,
            "shared_decision_kernel_contract_id": evidence.get("shared_decision_kernel_contract_id"),
            "shared_decision_kernel_implementation_sha256": evidence.get("shared_decision_kernel_implementation_sha256"),
            "kernel_execution_backend_id": evidence.get("kernel_execution_backend_id"),
            "decision_profile_id": evidence.get("decision_profile_id"),
            "feedback_adapter_id": evidence.get("feedback_adapter_id"),
            "decision_cadence": evidence.get("decision_cadence"),
            "fit_policy": arm_contract.get("fit_policy"),
            "event_identity_policy": arm_contract.get("event_identity_policy"),
            "route_input_policy": arm_contract.get("route_input_policy"),
            "implementation_sha256": arm.get("implementation_sha256"),
            "artifact_sha256": arm.get("artifact_sha256"),
            "threshold_config_sha256": arm.get("threshold_config_sha256"),
            "ordered_frame_ledger_sha256": expected_ledger_sha,
            "synthetic_fixture": predictions.get("synthetic_fixture"),
            "blind_accessed": False,
            "future_inputs_used": False,
            "execution_completed": True,
            "failure_count": 0,
        }
        for key, expected in expected_receipt.items():
            if receipt.get(key) != expected:
                raise ContractError(f"{arm_id}/execution_receipt.{key} mismatch")

        episode_rows = arm.get("episodes")
        if (
            not isinstance(episode_rows, list)
            or len(episode_rows) != len(truth_by_id)
            or any(not isinstance(row, dict) for row in episode_rows)
            or len({row.get("episode_id") for row in episode_rows}) != len(episode_rows)
            or {row.get("episode_id") for row in episode_rows} != set(truth_by_id)
        ):
            raise ContractError(f"{arm_id}.episodes must contain every truth episode exactly once")
        trace_sha_by_episode: dict[str, str] = {}
        request_sha_by_episode: dict[str, str] = {}
        output_sha_by_episode: dict[str, str] = {}
        inference_sha_by_episode: dict[str, str] = {}
        fold_artifact_sha_by_episode: dict[str, str] = {}
        fold_training_receipt_sha_by_episode: dict[str, str] = {}
        exit_code_by_episode: dict[str, int] = {}
        arm_frame_count = 0
        arm_alert_count = 0
        for summary in episode_rows:
            episode_id = _require_text(summary.get("episode_id"), where=f"{arm_id}.episode_id")
            truth = truth_by_id[episode_id]
            held_out_session = truth.get("session_id")
            fold = folds.get(held_out_session)
            if fold is None or summary.get("fold_held_out_session_id") != held_out_session:
                raise ContractError(f"{arm_id}/{episode_id} fold identity mismatch")
            if summary.get("fold_artifact_sha256") != fold.get("artifact_sha256"):
                raise ContractError(f"{arm_id}/{episode_id} fold artifact mismatch")
            if summary.get("fold_training_receipt_sha256") != fold.get("training_receipt_sha256"):
                raise ContractError(f"{arm_id}/{episode_id} fold training receipt mismatch")
            inference_manifest, inference_sha = _validate_inference_manifest(
                contract=contract,
                evidence=evidence,
                arm_contract=arm_contract,
                episode=truth,
                truth_by_id=truth_by_id,
                expected_frames=truth_frames[episode_id],
                shuffled_sources=shuffled_sources,
                summary=summary,
                prediction_root=prediction_root,
            )
            request_path = _bound_file(
                prediction_root,
                summary.get("adapter_request_path"),
                summary.get("adapter_request_sha256"),
                where=f"{arm_id}/{episode_id}/adapter_request",
            )
            output_path = _bound_file(
                prediction_root,
                summary.get("adapter_output_path"),
                summary.get("adapter_output_sha256"),
                where=f"{arm_id}/{episode_id}/adapter_output",
            )
            for stream in ("stdout", "stderr"):
                _bound_file(
                    prediction_root,
                    summary.get(f"adapter_{stream}_path"),
                    summary.get(f"adapter_{stream}_sha256"),
                    where=f"{arm_id}/{episode_id}/adapter_{stream}",
                )
            if summary.get("adapter_exit_code") != 0:
                raise ContractError(f"{arm_id}/{episode_id} adapter did not exit successfully")
            duration_ms = summary.get("adapter_duration_ms")
            if not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms < 0:
                raise ContractError(f"{arm_id}/{episode_id} adapter duration is invalid")
            request = _load_json(request_path, where=f"{arm_id}/{episode_id}/adapter_request")
            request_expected = {
                "schema": evidence.get("adapter_request_schema"),
                "contract_id": contract.get("contract_id"),
                "arm_id": arm_id,
                "candidate_adapter_id": adapter_id,
                "adapter_runtime_id": arm.get("adapter_runtime_id"),
                "shared_decision_kernel_contract_id": evidence.get("shared_decision_kernel_contract_id"),
                "shared_decision_kernel_implementation_sha256": evidence.get("shared_decision_kernel_implementation_sha256"),
                "kernel_execution_backend_id": evidence.get("kernel_execution_backend_id"),
                "decision_profile_id": evidence.get("decision_profile_id"),
                "feedback_adapter_id": evidence.get("feedback_adapter_id"),
                "kernel_trace_order": evidence.get("kernel_trace_order"),
                "decision_cadence": evidence.get("decision_cadence"),
                "fit_policy": arm_contract.get("fit_policy"),
                "event_identity_policy": arm_contract.get("event_identity_policy"),
                "route_input_policy": arm_contract.get("route_input_policy"),
                "implementation_sha256": arm.get("implementation_sha256"),
                "artifact_inventory_sha256": arm.get("artifact_sha256"),
                "threshold_config_sha256": arm.get("threshold_config_sha256"),
                "fold_held_out_session_id": held_out_session,
                "fold_artifact_sha256": fold.get("artifact_sha256"),
                "fold_training_input_manifest_sha256": fold.get("training_input_manifest_sha256"),
                "fold_training_receipt_sha256": fold.get("training_receipt_sha256"),
                "episode_id": episode_id,
                "sanitized_inference_manifest_sha256": inference_sha,
                "input_video_sha256": truth.get("video_sha256"),
                "truth_route_intent_sha256": truth.get("route_intent_sha256"),
                "adapter_route_input_sha256": inference_manifest.get("adapter_route_input_sha256"),
                "adapter_route_source_episode_id": inference_manifest.get("adapter_route_source_episode_id"),
                "source_capture_frame_ledger_sha256": truth.get("capture_frame_ledger_sha256"),
                "synthetic_fixture": predictions.get("synthetic_fixture"),
                "blind_accessed": False,
                "future_inputs_used": False,
                "production_model_replacement_authorized": False,
                "frames": truth_frames[episode_id],
            }
            for key, expected in request_expected.items():
                if request.get(key) != expected:
                    raise ContractError(f"{arm_id}/{episode_id}/adapter_request.{key} mismatch")
            raw_output = _load_json(output_path, where=f"{arm_id}/{episode_id}/adapter_output")
            output_expected = {
                **{key: request_expected[key] for key in request_expected if key != "schema" and key != "frames"},
                "schema": evidence.get("adapter_output_schema"),
                "kernel_trace_order": evidence.get("kernel_trace_order"),
                "execution_completed": True,
                "failure_count": 0,
                "abstained": summary.get("abstained"),
            }
            for key, expected in output_expected.items():
                if raw_output.get(key) != expected:
                    raise ContractError(f"{arm_id}/{episode_id}/adapter_output.{key} mismatch")
            _validate_android_backend_receipt(
                raw_output,
                request=request,
                request_sha256=summary.get("adapter_request_sha256"),
                threshold_config=threshold_config,
                evidence=evidence,
                where=f"{arm_id}/{episode_id}/adapter_output",
            )
            _validate_bbox_route_conditioning_receipt(
                raw_output,
                request=request,
                threshold_config=threshold_config,
                evidence=evidence,
                where=f"{arm_id}/{episode_id}/adapter_output",
            )
            _validate_dense_risk_evidence_receipt(
                raw_output,
                request=request,
                threshold_config=threshold_config,
                evidence=evidence,
                where=f"{arm_id}/{episode_id}/adapter_output",
            )
            trace_path = _bound_file(
                prediction_root,
                summary.get("prediction_trace_path"),
                summary.get("prediction_trace_sha256"),
                where=f"{arm_id}/{episode_id}/prediction_trace",
            )
            trace = _load_json(trace_path, where=f"{arm_id}/{episode_id}/prediction_trace")
            expected_trace = {
                "schema": evidence.get("episode_trace_schema"),
                "contract_id": contract.get("contract_id"),
                "arm_id": arm_id,
                "episode_id": episode_id,
                "candidate_adapter_id": adapter_id,
                "adapter_runtime_id": arm.get("adapter_runtime_id"),
                "runner_implementation_sha256": runner_sha,
                "adapter_registry_sha256": registry_sha,
                "adapter_request_sha256": summary.get("adapter_request_sha256"),
                "adapter_output_sha256": summary.get("adapter_output_sha256"),
                "shared_decision_kernel_contract_id": evidence.get("shared_decision_kernel_contract_id"),
                "shared_decision_kernel_implementation_sha256": evidence.get("shared_decision_kernel_implementation_sha256"),
                "kernel_execution_backend_id": evidence.get("kernel_execution_backend_id"),
                "decision_profile_id": evidence.get("decision_profile_id"),
                "feedback_adapter_id": evidence.get("feedback_adapter_id"),
                "decision_cadence": evidence.get("decision_cadence"),
                "fit_policy": arm_contract.get("fit_policy"),
                "event_identity_policy": arm_contract.get("event_identity_policy"),
                "route_input_policy": arm_contract.get("route_input_policy"),
                "fold_held_out_session_id": held_out_session,
                "fold_artifact_sha256": fold.get("artifact_sha256"),
                "fold_training_receipt_sha256": fold.get("training_receipt_sha256"),
                "sanitized_inference_manifest_sha256": inference_sha,
                "input_video_sha256": truth.get("video_sha256"),
                "truth_route_intent_sha256": truth.get("route_intent_sha256"),
                "adapter_route_input_sha256": inference_manifest.get("adapter_route_input_sha256"),
                "adapter_route_source_episode_id": inference_manifest.get("adapter_route_source_episode_id"),
                "source_capture_frame_ledger_sha256": truth.get("capture_frame_ledger_sha256"),
                "abstained": summary.get("abstained"),
            }
            for key, expected in expected_trace.items():
                if trace.get(key) != expected:
                    raise ContractError(f"{arm_id}/{episode_id}/trace.{key} mismatch")
            if trace.get("kernel_trace_order") != evidence.get("kernel_trace_order"):
                raise ContractError(f"{arm_id}/{episode_id} kernel trace order mismatch")
            frames = trace.get("frames")
            expected_frames = truth_frames[episode_id]
            if not isinstance(frames, list) or frames != raw_output.get("frames") or len(frames) != len(expected_frames):
                raise ContractError(f"{arm_id}/{episode_id} trace must contain every truth frame exactly once")
            alerts: list[int] = []
            for index, (actual, expected_frame) in enumerate(zip(frames, expected_frames)):
                where = f"{arm_id}/{episode_id}/frames[{index}]"
                if not isinstance(actual, dict):
                    raise ContractError(f"{where} must be an object")
                for key, expected in expected_frame.items():
                    if actual.get(key) != expected:
                        raise ContractError(f"{where}.{key} differs from truth ledger")
                delivered = _validate_decision(
                    actual.get("decision"),
                    contract=evidence,
                    adapter_id=adapter_id,
                    event_identity_policy=str(arm_contract.get("event_identity_policy")),
                    where=where,
                )
                if delivered:
                    alerts.append(expected_frame["video_pts_ms"])
            if trace.get("abstained") is True and alerts:
                raise ContractError(f"{arm_id}/{episode_id} cannot deliver feedback while abstained")
            if summary.get("alert_timestamps_ms") != alerts:
                raise ContractError(f"{arm_id}/{episode_id} alert summary differs from trace-derived feedback receipts")
            expected_frame_ids_sha = _canonical_sha256([row["frame_id"] for row in expected_frames])
            if summary.get("frame_ids_sha256") != expected_frame_ids_sha:
                raise ContractError(f"{arm_id}/{episode_id}.frame_ids_sha256 mismatch")
            trace_sha_by_episode[episode_id] = summary["prediction_trace_sha256"]
            request_sha_by_episode[episode_id] = summary["adapter_request_sha256"]
            output_sha_by_episode[episode_id] = summary["adapter_output_sha256"]
            inference_sha_by_episode[episode_id] = inference_sha
            fold_artifact_sha_by_episode[episode_id] = summary["fold_artifact_sha256"]
            fold_training_receipt_sha_by_episode[episode_id] = summary["fold_training_receipt_sha256"]
            exit_code_by_episode[episode_id] = summary["adapter_exit_code"]
            trace_count += 1
            frame_count += len(frames)
            derived_alert_count += len(alerts)
            arm_frame_count += len(frames)
            arm_alert_count += len(alerts)
        if receipt.get("prediction_trace_sha256_by_episode") != trace_sha_by_episode:
            raise ContractError(f"{arm_id}/execution_receipt trace hash inventory mismatch")
        for key, expected in (
            ("adapter_request_sha256_by_episode", request_sha_by_episode),
            ("adapter_output_sha256_by_episode", output_sha_by_episode),
            ("sanitized_inference_manifest_sha256_by_episode", inference_sha_by_episode),
            ("fold_artifact_sha256_by_episode", fold_artifact_sha_by_episode),
            ("fold_training_receipt_sha256_by_episode", fold_training_receipt_sha_by_episode),
            ("adapter_exit_code_by_episode", exit_code_by_episode),
        ):
            if receipt.get(key) != expected:
                raise ContractError(f"{arm_id}/execution_receipt {key} mismatch")
        durations = receipt.get("adapter_duration_ms_by_episode")
        if not isinstance(durations, dict) or set(durations) != set(truth_by_id) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in durations.values()
        ):
            raise ContractError(f"{arm_id}/execution_receipt duration inventory mismatch")
        arm_reports[arm_id] = {
            "candidate_adapter_id": adapter_id,
            "episode_trace_count": len(episode_rows),
            "frame_trace_count": arm_frame_count,
            "trace_derived_alert_count": arm_alert_count,
            "execution_receipt_sha256": arm.get("execution_receipt_sha256"),
        }
    return {
        "schema": REPORT_SCHEMA,
        "ordered_frame_ledger_sha256": expected_ledger_sha,
        "arm_count": len(arm_reports),
        "episode_trace_count": trace_count,
        "frame_trace_count": frame_count,
        "trace_derived_alert_count": derived_alert_count,
        "all_alerts_trace_derived": True,
        "arms": arm_reports,
        "u0_authority_granted": False,
        "training_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
