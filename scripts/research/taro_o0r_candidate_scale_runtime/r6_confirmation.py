"""Two-stage untouched-parent confirmation mechanics for TARO R6.

Phase A owns only RGB, AppleDepth, confidence, intrinsics, trajectory, and the
sealed DepthART candidate.  Registered FARO depth is accepted only by the
Phase-B truth factory after the Phase-A completion seal has been reloaded.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale
from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_support as direct
from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import r6_factor_split as split
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


ANALYSIS_ROLE = split.UNTOUCHED_CONFIRMATION
POLICY_ID = split.POLICY_ID
ROSTER = (
    ("467175", "47333514"),
    ("467312", "45261569"),
    ("435329", "42899445"),
    ("423306", "42897745"),
    ("466652", "45261100"),
    ("469650", "47333562"),
    ("470439", "47115427"),
    ("469830", "47334055"),
)
EXPECTED_PARENT_FRAME_COUNTS = (16, 14, 8, 13, 11, 24, 5, 29)
EXPECTED_FRAME_COUNT = 120
EXPECTED_QUERY_COUNT = EXPECTED_FRAME_COUNT * 9

SOURCE_SCHEMA = "blindassist.taro.o0r.r6_untouched_phase_a_source.v1"
CANDIDATE_INPUT_SCHEMA = "blindassist.taro.o0r.r6_untouched_candidate_input.v1"
INFERENCE_SCHEMA = "blindassist.taro.o0r.r6_untouched_depthart_inference.v1"
CANDIDATE_FRAME_SCHEMA = "blindassist.taro.o0r.r6_untouched_candidate_frame.v1"
CANDIDATE_COMPLETION_SCHEMA = "blindassist.taro.o0r.r6_untouched_candidate_completion.v1"
SOURCE_SCALE_SCHEMA = "blindassist.taro.o0r.r6_untouched_source_scale.v1"
SOURCE_DECISION_SCHEMA = "blindassist.taro.o0r.r6_untouched_source_decision.v1"
PHASE_A_COMPLETION_SCHEMA = "blindassist.taro.o0r.r6_untouched_phase_a_completion.v1"
TRUTH_BINDING_SCHEMA = "blindassist.taro.o0r.r6_untouched_faro_truth_binding.v1"
TRUTH_SCORING_SCHEMA = "blindassist.taro.o0r.r6_untouched_truth_scoring.v1"

PHASE_A_ASSET_ROLES = ("color", "lowres_depth", "confidence", "intrinsics", "trajectory")
FORBIDDEN_PHASE_A_READS = ("FARO", "QUERY_TRUTH", "TASK_METRIC", "PRIOR_OUTCOME")
_SHA256 = re.compile(r"^[0-9A-F]{64}$")


class R6ConfirmationError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R6ConfirmationError(code, message, **context)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(dict(value))
    require("content_sha256" not in record, "R6_CONFIRMATION_SEAL_COLLISION", "caller supplied content hash")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R6_CONFIRMATION_RECORD_INVALID", "sealed R6 record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(isinstance(observed, str) and bool(_SHA256.fullmatch(observed)) and adapter.canonical_sha256(record) == observed, "R6_CONFIRMATION_SEAL_MISMATCH", "R6 record seal drift", schema=schema)
    record["content_sha256"] = observed
    require(record.get("schema") == schema, "R6_CONFIRMATION_SCHEMA_DRIFT", "R6 record schema drift", expected=schema)
    return record


def _hash(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(_SHA256.fullmatch(value)), "R6_CONFIRMATION_HASH_INVALID", "R6 hash binding is malformed", field=field)
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _immutable(value: Any, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(contiguous.shape)


def expected_parent_frame_counts() -> dict[str, int]:
    return {parent: count for (parent, _), count in zip(ROSTER, EXPECTED_PARENT_FRAME_COUNTS)}


def validate_expected_keys(expected_keys: Sequence[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    keys = [(str(parent), str(video), str(token)) for parent, video, token in expected_keys]
    require(len(keys) == EXPECTED_FRAME_COUNT and len(keys) == len(set(keys)), "R6_COHORT_KEY_SEQUENCE_INVALID", "R6 frame sequence is not exact")
    counts = Counter((parent, video) for parent, video, _ in keys)
    require(list(counts) == list(ROSTER) and tuple(counts[row] for row in ROSTER) == EXPECTED_PARENT_FRAME_COUNTS, "R6_COHORT_KEY_SEQUENCE_INVALID", "R6 roster/count sequence drift")
    for parent, video, token in keys:
        require((parent, video) in ROSTER, "R6_COHORT_IDENTITY_INVALID", "R6 frame is outside the frozen untouched roster")
        adapter.decimal_timestamp_ns(token)
    return keys


def build_phase_a_source_receipt(
    *,
    parent_id: str,
    video_id: str,
    timestamp_token: str,
    lowres_intrinsics: Mapping[str, Any],
    trajectory_rows: Sequence[dict[str, Any]],
    container_bindings: Mapping[str, Any],
    asset_bindings: Mapping[str, Any],
    decoded_payload_hashes: Mapping[str, str],
) -> dict[str, Any]:
    require((parent_id, video_id) in ROSTER, "R6_SOURCE_IDENTITY_INVALID", "source is outside the frozen untouched roster")
    require(set(asset_bindings) == set(PHASE_A_ASSET_ROLES) and set(decoded_payload_hashes) == set(PHASE_A_ASSET_ROLES), "R6_SOURCE_ROLE_SET_DRIFT", "Phase-A source roles drift")
    require(set(container_bindings) == {"upsampling", "intrinsics", "trajectory"}, "R6_SOURCE_CONTAINER_SET_DRIFT", "R6 source container roles drift")
    low = copy.deepcopy(dict(lowres_intrinsics))
    transform, pose = adapter.interpolate_camera_to_world_exact(trajectory_rows, timestamp_token)
    high = adapter.scale_lowres_intrinsics(low)
    gravity = adapter._normalize_vector(transform[2, :3], "GRAVITY_INVALID")
    for role, binding in asset_bindings.items():
        require(isinstance(binding, dict) and set(binding) == {"container_sha256", "member_path", "bytes", "sha256", "crc32"}, "R6_SOURCE_ASSET_BINDING_INVALID", "Phase-A asset binding fields drift", role=role)
        _hash(binding["container_sha256"], f"{role}.container_sha256")
        _hash(binding["sha256"], f"{role}.sha256")
        _hash(decoded_payload_hashes[role], f"{role}.decoded_sha256")
    record = _seal(
        {
            "schema": SOURCE_SCHEMA,
            "analysis_role": ANALYSIS_ROLE,
            "parent_id": parent_id,
            "video_id": video_id,
            "timestamp_token": timestamp_token,
            "physical_frame_id": f"{video_id}:{timestamp_token}",
            "sensor_timestamp_ns": pose["frame_timestamp_ns"],
            "pose_bracket": {
                "left_timestamp_ns": pose["left_timestamp_ns"],
                "right_timestamp_ns": pose["right_timestamp_ns"],
                "fraction": pose["fraction"],
                "bracketing_gap_ns": pose["bracketing_gap_ns"],
            },
            "max_source_timestamp_ns": pose["max_source_timestamp_ns"],
            "lowres_intrinsics": low,
            "intrinsics_highres": high,
            "camera_to_world_4x4": transform.tolist(),
            "gravity_up_camera_xyz": gravity.tolist(),
            "container_bindings": copy.deepcopy(dict(container_bindings)),
            "asset_bindings": copy.deepcopy(dict(asset_bindings)),
            "decoded_payload_hashes": copy.deepcopy(dict(decoded_payload_hashes)),
            "phase_a_allowed_payloads": list(PHASE_A_ASSET_ROLES),
            "highres_faro_member_bound": False,
            "truth_payload_read": False,
            "task_metric_read": False,
        }
    )
    return validate_phase_a_source_receipt(record)


def validate_phase_a_source_receipt(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, SOURCE_SCHEMA)
    expected = {
        "schema", "analysis_role", "parent_id", "video_id", "timestamp_token", "physical_frame_id",
        "sensor_timestamp_ns", "pose_bracket", "max_source_timestamp_ns", "lowres_intrinsics",
        "intrinsics_highres", "camera_to_world_4x4", "gravity_up_camera_xyz", "container_bindings",
        "asset_bindings", "decoded_payload_hashes", "phase_a_allowed_payloads", "highres_faro_member_bound",
        "truth_payload_read", "task_metric_read", "content_sha256",
    }
    require(set(record) == expected and record["analysis_role"] == ANALYSIS_ROLE, "R6_SOURCE_KEY_SET_DRIFT", "R6 Phase-A source fields drift")
    require((record["parent_id"], record["video_id"]) in ROSTER and record["physical_frame_id"] == f"{record['video_id']}:{record['timestamp_token']}", "R6_SOURCE_IDENTITY_INVALID", "R6 Phase-A source identity drift")
    require(record["phase_a_allowed_payloads"] == list(PHASE_A_ASSET_ROLES) and set(record["asset_bindings"]) == set(PHASE_A_ASSET_ROLES) and set(record["decoded_payload_hashes"]) == set(PHASE_A_ASSET_ROLES), "R6_SOURCE_ROLE_SET_DRIFT", "R6 Phase-A payload roles drift")
    require(record["highres_faro_member_bound"] is False and record["truth_payload_read"] is False and record["task_metric_read"] is False, "R6_PHASE_A_TRUTH_FIREWALL_BREACH", "R6 Phase-A source crossed truth firewall")
    adapter.decimal_timestamp_ns(record["timestamp_token"])
    adapter._intrinsics_matrix(record["intrinsics_highres"]["matrix_3x3"])
    adapter._normalize_vector(record["gravity_up_camera_xyz"], "GRAVITY_INVALID")
    for role in PHASE_A_ASSET_ROLES:
        _hash(record["asset_bindings"][role]["sha256"], f"{role}.sha256")
        _hash(record["decoded_payload_hashes"][role], f"{role}.decoded_sha256")
    return record


def validate_bound_phase_a_payload(source: Mapping[str, Any], role: str, value: Any) -> None:
    receipt = validate_phase_a_source_receipt(dict(source))
    require(role in PHASE_A_ASSET_ROLES, "R6_PHASE_A_PAYLOAD_ROLE_INVALID", "payload role is not Phase-A allowed", role=role)
    require(adapter.canonical_sha256(value) == receipt["decoded_payload_hashes"][role], "R6_PHASE_A_PAYLOAD_HASH_DRIFT", "decoded Phase-A payload differs from receipt", role=role)


def build_candidate_input(source: Mapping[str, Any], color_rgb_u8: np.ndarray) -> dict[str, Any]:
    receipt = validate_phase_a_source_receipt(dict(source))
    color = np.asarray(color_rgb_u8)
    require(color.shape == (*adapter.HIGHRES_SHAPE_HW, 3) and color.dtype == np.uint8, "R6_COLOR_INVALID", "R6 RGB must be uint8 1440x1920x3")
    validate_bound_phase_a_payload(receipt, "color", color)
    return validate_candidate_input(
        _seal(
            {
                "schema": CANDIDATE_INPUT_SCHEMA,
                "analysis_role": ANALYSIS_ROLE,
                "parent_id": receipt["parent_id"],
                "video_id": receipt["video_id"],
                "timestamp_token": receipt["timestamp_token"],
                "physical_frame_id": receipt["physical_frame_id"],
                "phase_a_source_receipt_sha256": receipt["content_sha256"],
                "color_decoded_sha256": adapter.canonical_sha256(color),
                "intrinsics_highres": receipt["intrinsics_highres"],
                "intrinsics_highres_sha256": adapter.canonical_sha256(receipt["intrinsics_highres"]),
                "allowed_model_inputs": ["REGISTERED_RGB", "BOUND_EFFECTIVE_K"],
                "truth_payload_read": False,
                "prior_outcome_read": False,
            }
        )
    )


def validate_candidate_input(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, CANDIDATE_INPUT_SCHEMA)
    expected = {"schema", "analysis_role", "parent_id", "video_id", "timestamp_token", "physical_frame_id", "phase_a_source_receipt_sha256", "color_decoded_sha256", "intrinsics_highres", "intrinsics_highres_sha256", "allowed_model_inputs", "truth_payload_read", "prior_outcome_read", "content_sha256"}
    require(set(record) == expected and record["analysis_role"] == ANALYSIS_ROLE and (record["parent_id"], record["video_id"]) in ROSTER, "R6_CANDIDATE_INPUT_DRIFT", "R6 candidate input fields/role drift")
    require(record["physical_frame_id"] == f"{record['video_id']}:{record['timestamp_token']}" and record["allowed_model_inputs"] == ["REGISTERED_RGB", "BOUND_EFFECTIVE_K"], "R6_CANDIDATE_INPUT_DRIFT", "R6 candidate identity/input policy drift")
    require(record["truth_payload_read"] is False and record["prior_outcome_read"] is False, "R6_CANDIDATE_TRUTH_LEAK", "R6 candidate input crossed truth firewall")
    for field in ("phase_a_source_receipt_sha256", "color_decoded_sha256", "intrinsics_highres_sha256"):
        _hash(record[field], field)
    require(adapter.canonical_sha256(record["intrinsics_highres"]) == record["intrinsics_highres_sha256"], "R6_CANDIDATE_K_DRIFT", "R6 candidate K binding drift")
    return record


def build_inference_receipt(
    candidate_input: Mapping[str, Any],
    color_rgb_u8: np.ndarray,
    input_tensor_nchw: np.ndarray,
    resized_intrinsics_n33: np.ndarray,
    native_depth_m: np.ndarray,
    highres_depth_m: np.ndarray,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = validate_candidate_input(dict(candidate_input))
    color = np.asarray(color_rgb_u8)
    tensor = np.asarray(input_tensor_nchw)
    resized_k = np.asarray(resized_intrinsics_n33)
    native = np.asarray(native_depth_m)
    highres = np.asarray(highres_depth_m)
    require(adapter.canonical_sha256(color) == candidate["color_decoded_sha256"], "R6_INFERENCE_COLOR_DRIFT", "R6 inference RGB differs from receipt")
    require(tensor.shape == (1, 3, *depthart_runner.NATIVE_SHAPE_HW) and tensor.dtype == np.float32 and bool(np.all(np.isfinite(tensor))), "R6_INFERENCE_INPUT_INVALID", "R6 preprocessed tensor is invalid")
    require(resized_k.shape == (1, 3, 3) and resized_k.dtype == np.float32 and bool(np.all(np.isfinite(resized_k))), "R6_INFERENCE_INPUT_INVALID", "R6 resized K is invalid")
    require(native.shape == depthart_runner.NATIVE_SHAPE_HW and native.dtype == np.float32 and bool(np.all(np.isfinite(native))), "R6_NATIVE_DEPTH_INVALID", "R6 native depth is invalid")
    require(highres.shape == adapter.HIGHRES_SHAPE_HW and highres.dtype == np.float32 and adapter.canonical_sha256(depthart_runner.upsample_native_depth(native)) == adapter.canonical_sha256(highres), "R6_POSTPROCESS_DRIFT", "R6 highres output differs from frozen resize")
    return validate_inference_receipt(
        _seal(
            {
                "schema": INFERENCE_SCHEMA,
                "analysis_role": ANALYSIS_ROLE,
                "model_id": adapter.BASELINE_MODEL_ID,
                "checkpoint_sha256": adapter.BASELINE_CHECKPOINT_SHA256,
                "preprocess_id": depthart_runner.PREPROCESS_ID,
                "postprocess_id": depthart_runner.POSTPROCESS_ID,
                "candidate_input_sha256": candidate["content_sha256"],
                "parent_id": candidate["parent_id"],
                "video_id": candidate["video_id"],
                "timestamp_token": candidate["timestamp_token"],
                "physical_frame_id": candidate["physical_frame_id"],
                "input_tensor_sha256": adapter.canonical_sha256(tensor),
                "resized_intrinsics_sha256": adapter.canonical_sha256(resized_k),
                "native_depth_sha256": adapter.canonical_sha256(native),
                "highres_depth_sha256": adapter.canonical_sha256(highres),
                "runtime_identity": copy.deepcopy(dict(runtime_identity)),
                "truth_payload_read": False,
                "prior_outcome_read": False,
            }
        )
    )


def validate_inference_receipt(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, INFERENCE_SCHEMA)
    expected = {"schema", "analysis_role", "model_id", "checkpoint_sha256", "preprocess_id", "postprocess_id", "candidate_input_sha256", "parent_id", "video_id", "timestamp_token", "physical_frame_id", "input_tensor_sha256", "resized_intrinsics_sha256", "native_depth_sha256", "highres_depth_sha256", "runtime_identity", "truth_payload_read", "prior_outcome_read", "content_sha256"}
    require(set(record) == expected and record["analysis_role"] == ANALYSIS_ROLE and (record["parent_id"], record["video_id"]) in ROSTER, "R6_INFERENCE_RECEIPT_DRIFT", "R6 inference fields/role drift")
    require(record["model_id"] == adapter.BASELINE_MODEL_ID and record["checkpoint_sha256"] == adapter.BASELINE_CHECKPOINT_SHA256 and record["preprocess_id"] == depthart_runner.PREPROCESS_ID and record["postprocess_id"] == depthart_runner.POSTPROCESS_ID, "R6_INFERENCE_IDENTITY_DRIFT", "R6 model/transform identity drift")
    require(record["truth_payload_read"] is False and record["prior_outcome_read"] is False, "R6_INFERENCE_TRUTH_LEAK", "R6 inference crossed truth firewall")
    for field in ("checkpoint_sha256", "candidate_input_sha256", "input_tensor_sha256", "resized_intrinsics_sha256", "native_depth_sha256", "highres_depth_sha256"):
        _hash(record[field], field)
    require(isinstance(record["runtime_identity"], dict) and bool(record["runtime_identity"]), "R6_INFERENCE_RUNTIME_INVALID", "R6 runtime identity is absent")
    return record


def infer_candidate(model: Any, candidate_input: Mapping[str, Any], color_rgb_u8: np.ndarray, runtime_identity: Mapping[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    import torch

    candidate = validate_candidate_input(dict(candidate_input))
    matrix = np.asarray(candidate["intrinsics_highres"]["matrix_3x3"], dtype=np.float32)
    tensor, resized_k = depthart_runner.preprocess_depthart_input(np.asarray(color_rgb_u8), matrix)
    with torch.inference_mode():
        prediction = model(torch.from_numpy(tensor).to(device), torch.from_numpy(resized_k).to(device))
    native_batch = prediction.detach().float().cpu().numpy()
    require(native_batch.shape == (1, *depthart_runner.NATIVE_SHAPE_HW), "R6_NATIVE_DEPTH_INVALID", "official DepthART output must be 1x448x608")
    native = np.ascontiguousarray(native_batch[0], dtype=np.float32)
    highres = depthart_runner.upsample_native_depth(native)
    return {"native_depth_m": native, "highres_depth_m": highres, "inference_receipt": build_inference_receipt(candidate, color_rgb_u8, tensor, resized_k, native, highres, runtime_identity)}


def build_candidate_frame(candidate_input: Mapping[str, Any], inference_receipt: Mapping[str, Any], native_blob: Mapping[str, Any]) -> dict[str, Any]:
    candidate = validate_candidate_input(dict(candidate_input))
    inference = validate_inference_receipt(dict(inference_receipt))
    require(inference["candidate_input_sha256"] == candidate["content_sha256"], "R6_CANDIDATE_LINEAGE_DRIFT", "R6 inference does not bind candidate input")
    blob = copy.deepcopy(dict(native_blob))
    require(set(blob) == {"path", "bytes", "sha256", "array_sha256", "shape_hw", "dtype", "encoding"} and blob["array_sha256"] == inference["native_depth_sha256"], "R6_CANDIDATE_BLOB_DRIFT", "R6 candidate blob binding drift")
    return validate_candidate_frame(
        _seal(
            {
                "schema": CANDIDATE_FRAME_SCHEMA,
                "analysis_role": ANALYSIS_ROLE,
                "parent_id": candidate["parent_id"],
                "video_id": candidate["video_id"],
                "timestamp_token": candidate["timestamp_token"],
                "physical_frame_id": candidate["physical_frame_id"],
                "candidate_input": candidate,
                "inference_receipt": inference,
                "native_depth_blob": blob,
                "truth_payload_read": False,
            }
        )
    )


def validate_candidate_frame(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, CANDIDATE_FRAME_SCHEMA)
    expected = {"schema", "analysis_role", "parent_id", "video_id", "timestamp_token", "physical_frame_id", "candidate_input", "inference_receipt", "native_depth_blob", "truth_payload_read", "content_sha256"}
    require(set(record) == expected and record["analysis_role"] == ANALYSIS_ROLE and record["truth_payload_read"] is False, "R6_CANDIDATE_FRAME_DRIFT", "R6 candidate frame fields/firewall drift")
    candidate = validate_candidate_input(record["candidate_input"])
    inference = validate_inference_receipt(record["inference_receipt"])
    identity = (record["parent_id"], record["video_id"], record["timestamp_token"], record["physical_frame_id"])
    require(identity == (candidate["parent_id"], candidate["video_id"], candidate["timestamp_token"], candidate["physical_frame_id"]) == (inference["parent_id"], inference["video_id"], inference["timestamp_token"], inference["physical_frame_id"]), "R6_CANDIDATE_LINEAGE_DRIFT", "R6 candidate identity lineage drift")
    blob = record["native_depth_blob"]
    require(isinstance(blob, dict) and set(blob) == {"path", "bytes", "sha256", "array_sha256", "shape_hw", "dtype", "encoding"} and blob["array_sha256"] == inference["native_depth_sha256"] and blob["shape_hw"] == list(depthart_runner.NATIVE_SHAPE_HW) and blob["dtype"] == "float32" and blob["encoding"] == "DETERMINISTIC_GZIP_NPY_MTIME_0", "R6_CANDIDATE_BLOB_DRIFT", "R6 candidate blob fields drift")
    _hash(blob["sha256"], "native_blob.sha256")
    return record


def build_candidate_completion(records: Sequence[Mapping[str, Any]], expected_keys: Sequence[tuple[str, str, str]]) -> dict[str, Any]:
    rows = [validate_candidate_frame(dict(row)) for row in records]
    keys = [(row["parent_id"], row["video_id"], row["timestamp_token"]) for row in rows]
    require(keys == validate_expected_keys(expected_keys), "R6_CANDIDATE_COMPLETION_SEQUENCE_DRIFT", "R6 candidates do not exactly cover the frozen sequence")
    return validate_candidate_completion(
        _seal(
            {
                "schema": CANDIDATE_COMPLETION_SCHEMA,
                "analysis_role": ANALYSIS_ROLE,
                "candidate_frame_count": len(rows),
                "candidate_key_sequence_sha256": adapter.canonical_sha256([list(row) for row in keys]),
                "candidate_record_hash_sequence_sha256": adapter.canonical_sha256([row["content_sha256"] for row in rows]),
                "parent_frame_counts": expected_parent_frame_counts(),
                "all_candidates_sealed_before_source_decisions": True,
                "faro_reads": 0,
                "task_metric_reads": 0,
                "prior_outcome_reads": 0,
            }
        )
    )


def validate_candidate_completion(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, CANDIDATE_COMPLETION_SCHEMA)
    expected = {"schema", "analysis_role", "candidate_frame_count", "candidate_key_sequence_sha256", "candidate_record_hash_sequence_sha256", "parent_frame_counts", "all_candidates_sealed_before_source_decisions", "faro_reads", "task_metric_reads", "prior_outcome_reads", "content_sha256"}
    require(set(record) == expected and record["analysis_role"] == ANALYSIS_ROLE and record["candidate_frame_count"] == EXPECTED_FRAME_COUNT and record["parent_frame_counts"] == expected_parent_frame_counts(), "R6_CANDIDATE_COMPLETION_DRIFT", "R6 candidate completion cohort drift")
    require(record["all_candidates_sealed_before_source_decisions"] is True and record["faro_reads"] == record["task_metric_reads"] == record["prior_outcome_reads"] == 0, "R6_CANDIDATE_COMPLETION_FIREWALL_BREACH", "R6 candidate completion crossed truth firewall")
    _hash(record["candidate_key_sequence_sha256"], "candidate_key_sequence_sha256")
    _hash(record["candidate_record_hash_sequence_sha256"], "candidate_record_hash_sequence_sha256")
    return record


def _build_scale(source: dict[str, Any], candidate: dict[str, Any], highres: np.ndarray, apple: np.ndarray, confidence: np.ndarray) -> dict[str, Any]:
    estimate = apple_scale.estimate_source_metric_scale(apple, confidence, apple_scale.sample_candidate_at_apple_centers(highres))
    return validate_source_scale(
        _seal(
            {
                "schema": SOURCE_SCALE_SCHEMA,
                "parent_id": source["parent_id"],
                "video_id": source["video_id"],
                "timestamp_token": source["timestamp_token"],
                "physical_frame_id": source["physical_frame_id"],
                "phase_a_source_receipt_sha256": source["content_sha256"],
                "candidate_frame_sha256": candidate["content_sha256"],
                "candidate_highres_depth_sha256": adapter.canonical_sha256(highres),
                "apple_depth_sha256": adapter.canonical_sha256(apple),
                "confidence_sha256": adapter.canonical_sha256(confidence),
                "estimator_id": apple_scale.ESTIMATOR_ID,
                **estimate,
                "faro_payload_read": False,
                "task_metric_read": False,
            }
        )
    )


def validate_source_scale(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, SOURCE_SCALE_SCHEMA)
    expected = {"schema", "parent_id", "video_id", "timestamp_token", "physical_frame_id", "phase_a_source_receipt_sha256", "candidate_frame_sha256", "candidate_highres_depth_sha256", "apple_depth_sha256", "confidence_sha256", "estimator_id", "evaluable", "reason_codes", "valid_pair_count", "selected_pixel_ids_sha256", "log_metric_scale", "metric_scale", "faro_payload_read", "task_metric_read", "content_sha256"}
    require(set(record) == expected and (record["parent_id"], record["video_id"]) in ROSTER and record["estimator_id"] == apple_scale.ESTIMATOR_ID, "R6_SOURCE_SCALE_DRIFT", "R6 source scale fields/identity drift")
    for field in ("phase_a_source_receipt_sha256", "candidate_frame_sha256", "candidate_highres_depth_sha256", "apple_depth_sha256", "confidence_sha256", "selected_pixel_ids_sha256"):
        _hash(record[field], field)
    require(isinstance(record["evaluable"], bool) and isinstance(record["valid_pair_count"], int) and isinstance(record["reason_codes"], list), "R6_SOURCE_SCALE_DRIFT", "R6 source scale metadata drift")
    if record["evaluable"]:
        require(record["valid_pair_count"] >= apple_scale.MINIMUM_PAIR_COUNT and record["reason_codes"] == [] and _finite(record["metric_scale"]) and float(record["metric_scale"]) > 0.0 and _finite(record["log_metric_scale"]), "R6_SOURCE_SCALE_DRIFT", "evaluable R6 source scale metrics drift")
    else:
        require(record["valid_pair_count"] < apple_scale.MINIMUM_PAIR_COUNT and record["reason_codes"] == ["APPLE_SCALE_COMMON_SUPPORT_INSUFFICIENT"] and record["metric_scale"] is None and record["log_metric_scale"] is None, "R6_SOURCE_SCALE_DRIFT", "unevaluable R6 source scale drift")
    require(record["faro_payload_read"] is False and record["task_metric_read"] is False, "R6_SOURCE_SCALE_TRUTH_LEAK", "R6 source scale crossed truth firewall")
    return record


def build_source_decision(source_value: Mapping[str, Any], candidate_value: Mapping[str, Any], native_depth_m: np.ndarray, apple_depth_mm: np.ndarray, confidence: np.ndarray) -> dict[str, Any]:
    source = validate_phase_a_source_receipt(dict(source_value))
    candidate = validate_candidate_frame(dict(candidate_value))
    native = np.ascontiguousarray(native_depth_m, dtype=np.float32)
    inference = candidate["inference_receipt"]
    require(native.shape == depthart_runner.NATIVE_SHAPE_HW and adapter.canonical_sha256(native) == inference["native_depth_sha256"], "R6_NATIVE_DEPTH_DRIFT", "R6 native candidate differs from sealed receipt")
    highres = depthart_runner.upsample_native_depth(native)
    require(adapter.canonical_sha256(highres) == inference["highres_depth_sha256"], "R6_HIGHRES_DEPTH_DRIFT", "R6 replayed highres candidate drift")
    identity = (source["parent_id"], source["video_id"], source["timestamp_token"], source["physical_frame_id"])
    require(identity == (candidate["parent_id"], candidate["video_id"], candidate["timestamp_token"], candidate["physical_frame_id"]) and candidate["candidate_input"]["phase_a_source_receipt_sha256"] == source["content_sha256"], "R6_SOURCE_CANDIDATE_LINEAGE_DRIFT", "R6 source/candidate lineage drift")
    apple = np.asarray(apple_depth_mm)
    conf = np.asarray(confidence)
    require(apple.shape == adapter.APPLE_SHAPE_HW and apple.dtype == np.uint16 and conf.shape == adapter.APPLE_SHAPE_HW and conf.dtype == np.uint8 and bool(np.all(conf <= 2)), "R6_APPLE_INPUT_INVALID", "R6 AppleDepth/confidence shape or dtype drift")
    validate_bound_phase_a_payload(source, "lowres_depth", apple)
    validate_bound_phase_a_payload(source, "confidence", conf)
    scale = _build_scale(source, candidate, highres, apple, conf)
    anchored_hash: str | None = None
    plane_block: dict[str, Any] | None = None
    failure_code: str | None = None
    if scale["evaluable"]:
        anchored = np.ascontiguousarray(highres.astype(np.float64) * float(scale["metric_scale"]), dtype=np.float64)
        anchored_hash = adapter.canonical_sha256(anchored)
        apple_m = apple.astype(np.float64) / 1000.0
        lower, upper = apple_scale.DEPTH_RANGE_M
        support_mask = (conf == 2) & (apple_m >= lower) & (apple_m <= upper)
        support_ids = np.flatnonzero(support_mask).astype(np.int64)
        low = source["lowres_intrinsics"]
        low_k = adapter._intrinsics_matrix([[low["fx"], 0.0, low["cx"]], [0.0, low["fy"], low["cy"]], [0.0, 0.0, 1.0]], adapter.APPLE_SHAPE_HW)
        gravity = adapter._normalize_vector(source["gravity_up_camera_xyz"], "GRAVITY_INVALID")
        try:
            points, pixels = adapter._unproject(apple_m, support_mask, low_k, 1)
            plane = adapter._fit_support_plane(points, gravity)
            camera_height = float(plane["camera_height_m"])
            require(direct.CAMERA_HEIGHT_RANGE_M[0] <= camera_height <= direct.CAMERA_HEIGHT_RANGE_M[1], "DIRECT_APPLE_SUPPORT_HEIGHT_IMPLAUSIBLE", "Apple support height leaves physical range")
        except (adapter.AdapterError, R6ConfirmationError) as error:
            failure_code = str(getattr(error, "code", type(error).__name__))
        else:
            plane_block = {
                "normal_camera_xyz": np.asarray(plane["normal_camera_xyz"], dtype=np.float64).tolist(),
                "camera_height_m": camera_height,
                "support_count": int(plane["support_count"]),
                "support_fraction": float(plane["support_fraction"]),
                "slope_degrees": float(plane["slope_degrees"]),
                "median_residual_m": float(plane["median_residual_m"]),
                "apple_support_point_count": len(support_ids),
                "apple_support_pixel_ids_sha256": adapter.canonical_sha256(support_ids),
                "sampled_pixels_sha256": adapter.canonical_sha256(pixels),
                "sampled_points_sha256": adapter.canonical_sha256(points),
                "candidate_depth_used_for_support_mask": False,
            }
    else:
        failure_code = str(scale["reason_codes"][0])
    available = plane_block is not None
    return validate_source_decision(
        _seal(
            {
                "schema": SOURCE_DECISION_SCHEMA,
                "analysis_role": ANALYSIS_ROLE,
                "policy_id": POLICY_ID,
                "parent_id": source["parent_id"],
                "video_id": source["video_id"],
                "timestamp_token": source["timestamp_token"],
                "physical_frame_id": source["physical_frame_id"],
                "phase_a_source_receipt_sha256": source["content_sha256"],
                "candidate_frame_sha256": candidate["content_sha256"],
                "candidate_highres_depth_sha256": adapter.canonical_sha256(highres),
                "anchored_candidate_depth_sha256": anchored_hash,
                "intrinsics_highres_sha256": adapter.canonical_sha256(adapter._intrinsics_matrix(source["intrinsics_highres"]["matrix_3x3"])),
                "gravity_up_camera_xyz_sha256": adapter.canonical_sha256(adapter._normalize_vector(source["gravity_up_camera_xyz"], "GRAVITY_INVALID")),
                "scale_record": scale,
                "source_support_available": available,
                "source_failure_code": failure_code,
                "direct_support_plane": plane_block,
                "selected_branch": "DIRECT_APPLE_SUPPORT" if available else "R1_BASELINE",
                "selection_fields_read": ["source_support_available"],
                "outcome_dependent_reselection_allowed": False,
                "faro_payload_read": False,
                "task_metric_read": False,
                "prior_outcome_read": False,
            }
        )
    )


def validate_source_decision(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, SOURCE_DECISION_SCHEMA)
    expected = {"schema", "analysis_role", "policy_id", "parent_id", "video_id", "timestamp_token", "physical_frame_id", "phase_a_source_receipt_sha256", "candidate_frame_sha256", "candidate_highres_depth_sha256", "anchored_candidate_depth_sha256", "intrinsics_highres_sha256", "gravity_up_camera_xyz_sha256", "scale_record", "source_support_available", "source_failure_code", "direct_support_plane", "selected_branch", "selection_fields_read", "outcome_dependent_reselection_allowed", "faro_payload_read", "task_metric_read", "prior_outcome_read", "content_sha256"}
    require(set(record) == expected and record["analysis_role"] == ANALYSIS_ROLE and record["policy_id"] == POLICY_ID and (record["parent_id"], record["video_id"]) in ROSTER, "R6_SOURCE_DECISION_DRIFT", "R6 source decision fields/identity drift")
    scale = validate_source_scale(record["scale_record"])
    for field in ("phase_a_source_receipt_sha256", "candidate_frame_sha256", "candidate_highres_depth_sha256", "intrinsics_highres_sha256", "gravity_up_camera_xyz_sha256"):
        _hash(record[field], field)
    available = record["source_support_available"]
    require(isinstance(available, bool) and record["selected_branch"] == ("DIRECT_APPLE_SUPPORT" if available else "R1_BASELINE") and record["selection_fields_read"] == ["source_support_available"] and record["outcome_dependent_reselection_allowed"] is False, "R6_SOURCE_DECISION_POLICY_DRIFT", "R6 source decision policy drift")
    require(scale["phase_a_source_receipt_sha256"] == record["phase_a_source_receipt_sha256"] and scale["candidate_frame_sha256"] == record["candidate_frame_sha256"], "R6_SOURCE_SCALE_LINEAGE_DRIFT", "R6 scale lineage drift")
    if available:
        _hash(record["anchored_candidate_depth_sha256"], "anchored_candidate_depth_sha256")
        plane = record["direct_support_plane"]
        require(record["source_failure_code"] is None and isinstance(plane, dict) and plane.get("candidate_depth_used_for_support_mask") is False and isinstance(plane.get("support_count"), int) and plane["support_count"] >= adapter.MINIMUM_SUPPORT_POINTS, "R6_SOURCE_DECISION_PLANE_DRIFT", "R6 direct support plane drift")
    else:
        require(isinstance(record["source_failure_code"], str) and bool(record["source_failure_code"]) and record["direct_support_plane"] is None, "R6_SOURCE_DECISION_FALLBACK_DRIFT", "R6 fallback decision drift")
    require(record["faro_payload_read"] is False and record["task_metric_read"] is False and record["prior_outcome_read"] is False, "R6_SOURCE_DECISION_TRUTH_LEAK", "R6 source decision crossed truth firewall")
    return record


def build_phase_a_completion(candidate_completion_value: Mapping[str, Any], decisions: Sequence[Mapping[str, Any]], expected_keys: Sequence[tuple[str, str, str]], *, read_counts: Mapping[str, int]) -> dict[str, Any]:
    candidate = validate_candidate_completion(dict(candidate_completion_value))
    rows = [validate_source_decision(dict(row)) for row in decisions]
    keys = [(row["parent_id"], row["video_id"], row["timestamp_token"]) for row in rows]
    require(keys == validate_expected_keys(expected_keys), "R6_PHASE_A_DECISION_SEQUENCE_DRIFT", "R6 decisions do not exactly cover candidates")
    require(adapter.canonical_sha256([row["candidate_frame_sha256"] for row in rows]) == candidate["candidate_record_hash_sequence_sha256"], "R6_PHASE_A_CANDIDATE_LINEAGE_DRIFT", "R6 decisions do not bind completed candidates")
    require(all(int(read_counts.get(role, -1)) == 0 for role in FORBIDDEN_PHASE_A_READS), "R6_PHASE_A_READ_FIREWALL_BREACH", "forbidden Phase-A payload was read", read_counts=dict(read_counts))
    return validate_phase_a_completion(
        _seal(
            {
                "schema": PHASE_A_COMPLETION_SCHEMA,
                "analysis_role": ANALYSIS_ROLE,
                "policy_id": POLICY_ID,
                "candidate_completion_sha256": candidate["content_sha256"],
                "physical_frame_count": len(rows),
                "decision_key_sequence_sha256": adapter.canonical_sha256([list(row) for row in keys]),
                "decision_hash_sequence_sha256": adapter.canonical_sha256([row["content_sha256"] for row in rows]),
                "direct_selected_frame_count": sum(row["source_support_available"] for row in rows),
                "baseline_fallback_frame_count": sum(not row["source_support_available"] for row in rows),
                "read_counts": {key: int(read_counts.get(key, 0)) for key in sorted(set(read_counts) | set(FORBIDDEN_PHASE_A_READS))},
                "forbidden_zero_read_roles": list(FORBIDDEN_PHASE_A_READS),
                "all_candidates_before_decisions": True,
                "all_decisions_before_faro": True,
            }
        )
    )


def validate_phase_a_completion(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, PHASE_A_COMPLETION_SCHEMA)
    expected = {"schema", "analysis_role", "policy_id", "candidate_completion_sha256", "physical_frame_count", "decision_key_sequence_sha256", "decision_hash_sequence_sha256", "direct_selected_frame_count", "baseline_fallback_frame_count", "read_counts", "forbidden_zero_read_roles", "all_candidates_before_decisions", "all_decisions_before_faro", "content_sha256"}
    require(set(record) == expected and record["analysis_role"] == ANALYSIS_ROLE and record["policy_id"] == POLICY_ID and record["physical_frame_count"] == EXPECTED_FRAME_COUNT, "R6_PHASE_A_COMPLETION_DRIFT", "R6 Phase-A completion fields/count drift")
    require(record["direct_selected_frame_count"] + record["baseline_fallback_frame_count"] == EXPECTED_FRAME_COUNT and record["forbidden_zero_read_roles"] == list(FORBIDDEN_PHASE_A_READS) and all(record["read_counts"].get(role) == 0 for role in FORBIDDEN_PHASE_A_READS), "R6_PHASE_A_COMPLETION_DRIFT", "R6 Phase-A completion counts/firewall drift")
    require(record["all_candidates_before_decisions"] is True and record["all_decisions_before_faro"] is True, "R6_PHASE_A_ORDER_DRIFT", "R6 Phase-A order drift")
    for field in ("candidate_completion_sha256", "decision_key_sequence_sha256", "decision_hash_sequence_sha256"):
        _hash(record[field], field)
    return record


@dataclass(frozen=True)
class R6FaroGeometry:
    parent_id: str
    video_id: str
    physical_frame_id: str
    phase_a_source_receipt_sha256: str
    source_decision_sha256: str
    phase_a_completion_sha256: str
    truth_binding_sha256: str
    highres_depth_array_sha256: str
    max_source_timestamp_ns: int
    intrinsics: np.ndarray
    depth_m: np.ndarray
    valid_depth: np.ndarray
    points_camera_xyz: np.ndarray
    pixels_uv: np.ndarray
    support_normal_camera_xyz: np.ndarray
    camera_height_m: float
    support_count: int
    support_fraction: float
    support_slope_degrees: float
    support_median_residual_m: float
    content_sha256: str


def _geometry_payload(value: R6FaroGeometry) -> dict[str, Any]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__ if field != "content_sha256"}


def validate_faro_geometry(value: Any) -> R6FaroGeometry:
    require(isinstance(value, R6FaroGeometry), "R6_FARO_GEOMETRY_INVALID", "R6 FARO geometry must come from controlled factory")
    geometry = value
    require((geometry.parent_id, geometry.video_id) in ROSTER and geometry.physical_frame_id.startswith(f"{geometry.video_id}:"), "R6_FARO_GEOMETRY_INVALID", "R6 FARO identity drift")
    for field in ("intrinsics", "depth_m", "valid_depth", "points_camera_xyz", "pixels_uv", "support_normal_camera_xyz"):
        require(not np.asarray(getattr(geometry, field)).flags.writeable, "R6_FARO_GEOMETRY_MUTABLE", "R6 FARO arrays must be immutable", field=field)
    for field in ("phase_a_source_receipt_sha256", "source_decision_sha256", "phase_a_completion_sha256", "truth_binding_sha256", "highres_depth_array_sha256"):
        _hash(getattr(geometry, field), field)
    require(adapter.canonical_sha256(_geometry_payload(geometry)) == geometry.content_sha256, "R6_FARO_GEOMETRY_HASH_DRIFT", "R6 FARO geometry hash drift")
    return geometry


def build_truth_binding(source_value: Mapping[str, Any], phase_a_completion_value: Mapping[str, Any], *, member_binding: Mapping[str, Any], highres_depth_mm: np.ndarray) -> dict[str, Any]:
    source = validate_phase_a_source_receipt(dict(source_value))
    phase_a = validate_phase_a_completion(dict(phase_a_completion_value))
    raw = np.asarray(highres_depth_mm)
    require(raw.shape == adapter.HIGHRES_SHAPE_HW and raw.dtype == np.uint16, "R6_FARO_DEPTH_INVALID", "R6 FARO must be uint16 1440x1920")
    member = copy.deepcopy(dict(member_binding))
    require(set(member) == {"container_sha256", "member_path", "bytes", "sha256", "crc32"}, "R6_FARO_MEMBER_BINDING_DRIFT", "R6 FARO member binding fields drift")
    _hash(member["container_sha256"], "faro.container_sha256")
    _hash(member["sha256"], "faro.sha256")
    require(member["container_sha256"] == source["container_bindings"]["upsampling"]["sha256"], "R6_FARO_CONTAINER_LINEAGE_DRIFT", "R6 FARO member does not share Phase-A source container")
    return validate_truth_binding(
        _seal(
            {
                "schema": TRUTH_BINDING_SCHEMA,
                "analysis_role": ANALYSIS_ROLE,
                "parent_id": source["parent_id"],
                "video_id": source["video_id"],
                "timestamp_token": source["timestamp_token"],
                "physical_frame_id": source["physical_frame_id"],
                "phase_a_source_receipt_sha256": source["content_sha256"],
                "phase_a_completion_sha256": phase_a["content_sha256"],
                "faro_member_binding": member,
                "highres_depth_array_sha256": adapter.canonical_sha256(raw),
                "first_faro_read_after_phase_a_reload": True,
                "used_for_scoring_only": True,
            }
        )
    )


def validate_truth_binding(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, TRUTH_BINDING_SCHEMA)
    expected = {"schema", "analysis_role", "parent_id", "video_id", "timestamp_token", "physical_frame_id", "phase_a_source_receipt_sha256", "phase_a_completion_sha256", "faro_member_binding", "highres_depth_array_sha256", "first_faro_read_after_phase_a_reload", "used_for_scoring_only", "content_sha256"}
    require(set(record) == expected and record["analysis_role"] == ANALYSIS_ROLE and (record["parent_id"], record["video_id"]) in ROSTER, "R6_TRUTH_BINDING_DRIFT", "R6 truth binding fields/identity drift")
    require(record["first_faro_read_after_phase_a_reload"] is True and record["used_for_scoring_only"] is True, "R6_TRUTH_FIREWALL_DRIFT", "R6 truth binding phase order drift")
    for field in ("phase_a_source_receipt_sha256", "phase_a_completion_sha256", "highres_depth_array_sha256"):
        _hash(record[field], field)
    return record


def derive_faro_geometry(highres_depth_mm: np.ndarray, source_value: Mapping[str, Any], decision_value: Mapping[str, Any], phase_a_completion_value: Mapping[str, Any], truth_binding_value: Mapping[str, Any]) -> R6FaroGeometry:
    source = validate_phase_a_source_receipt(dict(source_value))
    decision = validate_source_decision(dict(decision_value))
    phase_a = validate_phase_a_completion(dict(phase_a_completion_value))
    truth = validate_truth_binding(dict(truth_binding_value))
    raw = np.asarray(highres_depth_mm)
    require(raw.shape == adapter.HIGHRES_SHAPE_HW and raw.dtype == np.uint16 and adapter.canonical_sha256(raw) == truth["highres_depth_array_sha256"], "R6_FARO_DEPTH_INVALID", "R6 FARO array differs from truth binding")
    require(source["content_sha256"] == decision["phase_a_source_receipt_sha256"] == truth["phase_a_source_receipt_sha256"] and phase_a["content_sha256"] == truth["phase_a_completion_sha256"] and source["physical_frame_id"] == decision["physical_frame_id"] == truth["physical_frame_id"], "R6_FARO_LINEAGE_DRIFT", "R6 FARO inputs do not share Phase-A lineage")
    matrix = adapter._intrinsics_matrix(source["intrinsics_highres"]["matrix_3x3"])
    gravity = adapter._normalize_vector(source["gravity_up_camera_xyz"], "GRAVITY_INVALID")
    depth_m = raw.astype(np.float64) / 1000.0
    valid = np.isfinite(depth_m) & (depth_m >= adapter.DEPTH_RANGE_M[0]) & (depth_m <= adapter.DEPTH_RANGE_M[1])
    sampled, _ = adapter._unproject(depth_m, valid, matrix, adapter.SUPPORT_POINT_STRIDE)
    try:
        plane = adapter._fit_support_plane(sampled, gravity)
    except adapter.AdapterError as error:
        raise R6ConfirmationError(error.code, str(error), **error.context) from error
    points, pixels = adapter._unproject(depth_m, valid, matrix, 1)
    values = {
        "parent_id": source["parent_id"], "video_id": source["video_id"], "physical_frame_id": source["physical_frame_id"],
        "phase_a_source_receipt_sha256": source["content_sha256"], "source_decision_sha256": decision["content_sha256"],
        "phase_a_completion_sha256": phase_a["content_sha256"], "truth_binding_sha256": truth["content_sha256"],
        "highres_depth_array_sha256": adapter.canonical_sha256(raw), "max_source_timestamp_ns": int(source["max_source_timestamp_ns"]),
        "intrinsics": _immutable(matrix, np.float64), "depth_m": _immutable(depth_m, np.float64), "valid_depth": _immutable(valid, np.bool_),
        "points_camera_xyz": _immutable(points, np.float64), "pixels_uv": _immutable(pixels, np.int32),
        "support_normal_camera_xyz": _immutable(plane["normal_camera_xyz"], np.float64), "camera_height_m": float(plane["camera_height_m"]),
        "support_count": int(plane["support_count"]), "support_fraction": float(plane["support_fraction"]),
        "support_slope_degrees": float(plane["slope_degrees"]), "support_median_residual_m": float(plane["median_residual_m"]),
    }
    provisional = R6FaroGeometry(**values, content_sha256="0" * 64)
    return validate_faro_geometry(R6FaroGeometry(**values, content_sha256=adapter.canonical_sha256(_geometry_payload(provisional))))


def _prepared_and_plane(source: dict[str, Any], candidate: dict[str, Any], native_depth_m: np.ndarray, decision: dict[str, Any]) -> tuple[source_factor.PreparedSourceCandidate, direct.DirectAppleSupportPlane | None]:
    native = np.ascontiguousarray(native_depth_m, dtype=np.float32)
    require(native.shape == depthart_runner.NATIVE_SHAPE_HW and adapter.canonical_sha256(native) == candidate["inference_receipt"]["native_depth_sha256"], "R6_QUERY_NATIVE_DEPTH_DRIFT", "R6 query native candidate drift")
    raw = depthart_runner.upsample_native_depth(native)
    raw_hash = adapter.canonical_sha256(raw)
    scale = decision["scale_record"]
    anchored = np.ascontiguousarray(raw.astype(np.float64) * float(scale["metric_scale"]), dtype=np.float64) if scale["evaluable"] else np.ascontiguousarray(raw, dtype=np.float64)
    anchored_hash = adapter.canonical_sha256(anchored)
    prepared = source_factor.PreparedSourceCandidate(
        parent_id=source["parent_id"], physical_frame_id=source["physical_frame_id"], raw_depth_m=_immutable(raw, raw.dtype), anchored_depth_m=_immutable(anchored, np.float64),
        raw_depth_sha256=raw_hash, anchored_depth_sha256=anchored_hash, metric_scale=float(scale["metric_scale"]) if scale["evaluable"] else 1.0,
        source_scale_record_sha256=scale["content_sha256"], candidate_binding_sha256=candidate["content_sha256"], apple_source_receipt_sha256=source["content_sha256"], reliability={},
    )
    if not decision["source_support_available"]:
        return prepared, None
    block = decision["direct_support_plane"]
    plane = direct.DirectAppleSupportPlane(
        parent_id=source["parent_id"], physical_frame_id=source["physical_frame_id"], direct_source_receipt_sha256=source["content_sha256"],
        source_scale_record_sha256=scale["content_sha256"], candidate_binding_sha256=candidate["content_sha256"], anchored_depth_array_sha256=anchored_hash,
        intrinsics_highres_sha256=adapter.canonical_sha256(adapter._intrinsics_matrix(source["intrinsics_highres"]["matrix_3x3"])), gravity_up_camera_xyz_sha256=decision["gravity_up_camera_xyz_sha256"],
        normal_camera_xyz=_immutable(block["normal_camera_xyz"], np.float64), camera_height_m=float(block["camera_height_m"]), support_count=int(block["support_count"]),
        support_fraction=float(block["support_fraction"]), slope_degrees=float(block["slope_degrees"]), median_residual_m=float(block["median_residual_m"]),
        record=decision, content_sha256=decision["content_sha256"],
    )
    return prepared, plane


def _query_truth_base(geometry: R6FaroGeometry, query: dict[str, Any]) -> source_factor.QueryTruthBase:
    # R5's inner factory is roster-neutral and consumes only the controlled
    # geometry/query values.  Its public validator is deliberately bypassed.
    return r5._build_query_truth_base_validated(geometry, query)


def _truth_scoring_record(geometry: R6FaroGeometry, query: dict[str, Any], baseline: Mapping[str, Any], selected: Mapping[str, Any], *, support_unobservable_code: str | None = None) -> dict[str, Any]:
    return _seal(
        {
            "schema": TRUTH_SCORING_SCHEMA,
            "analysis_role": ANALYSIS_ROLE,
            "parent_id": geometry.parent_id,
            "physical_frame_id": geometry.physical_frame_id,
            "query_id": query["query_id"],
            "query_receipt_sha256": query["content_sha256"],
            "faro_geometry_sha256": geometry.content_sha256,
            "truth_binding_sha256": geometry.truth_binding_sha256,
            "baseline_mode_sha256": adapter.canonical_sha256(baseline),
            "selected_support_boundary_mode_sha256": adapter.canonical_sha256(selected),
            "support_unobservable_code": support_unobservable_code,
            "faro_used_for_scoring_only": True,
            "branch_reselection_after_truth": False,
        }
    )


def evaluate_frame(source_value: Mapping[str, Any], candidate_value: Mapping[str, Any], native_depth_m: np.ndarray, decision_value: Mapping[str, Any], geometry_value: R6FaroGeometry) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    source = validate_phase_a_source_receipt(dict(source_value))
    candidate = validate_candidate_frame(dict(candidate_value))
    decision = validate_source_decision(dict(decision_value))
    geometry = validate_faro_geometry(geometry_value)
    require(source["content_sha256"] == geometry.phase_a_source_receipt_sha256 and decision["content_sha256"] == geometry.source_decision_sha256 and candidate["content_sha256"] == decision["candidate_frame_sha256"], "R6_QUERY_LINEAGE_DRIFT", "R6 query inputs do not share lineage")
    queries = [adapter._validate_query_receipt(row) for row in adapter._build_geometry_query_receipts(physical_frame_id=geometry.physical_frame_id, source_frame_receipt_sha256=source["content_sha256"], max_source_timestamp_ns=geometry.max_source_timestamp_ns, support_normal_camera_xyz=geometry.support_normal_camera_xyz, camera_height_m=geometry.camera_height_m)]
    require([row["grid_index"] for row in queries] == list(range(9)), "R6_QUERY_GRID_DRIFT", "R6 query grid drift")
    prepared, plane = _prepared_and_plane(source, candidate, native_depth_m, decision)
    output: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for query in queries:
        try:
            base = _query_truth_base(geometry, query)
            baseline = r5._extract_baseline(prepared, source["intrinsics_highres"]["matrix_3x3"], source["gravity_up_camera_xyz"], base)
            selected = r5._extract_direct(prepared, plane, source["intrinsics_highres"]["matrix_3x3"], source["gravity_up_camera_xyz"], base, decision["source_failure_code"]) if decision["source_support_available"] else baseline
        except (R6ConfirmationError, adapter.AdapterError, source_factor.SourceFactorError) as error:
            code = str(getattr(error, "code", type(error).__name__))
            baseline = source_factor._failed_mode(prepared.raw_depth_sha256, code)
            selected = source_factor._failed_mode(prepared.anchored_depth_sha256, code) if decision["source_support_available"] else baseline
        baseline = r5._validate_mode_result(baseline, "r6.baseline")
        selected = r5._validate_mode_result(selected, "r6.selected_support_boundary")
        truth = _truth_scoring_record(geometry, query, baseline, selected)
        components = split.build_factor_components(
            analysis_role=ANALYSIS_ROLE, parent_id=source["parent_id"], physical_frame_id=source["physical_frame_id"], query_id=query["query_id"], grid_index=query["grid_index"],
            source_frame_receipt_sha256=source["content_sha256"], candidate_frame_record_sha256=candidate["content_sha256"], r6_phase_a_policy_seal_sha256=decision["content_sha256"],
            query_receipt_sha256=query["content_sha256"], truth_scoring_record_sha256=truth["content_sha256"], source_support_available=decision["source_support_available"],
            phase_a_selected_branch=decision["selected_branch"], baseline=baseline, selected_support_boundary=selected,
        )
        output.append((truth, components, split.build_composite_query(components)))
    return output


def evaluate_unobservable_faro_frame(
    source_value: Mapping[str, Any],
    candidate_value: Mapping[str, Any],
    native_depth_m: np.ndarray,
    decision_value: Mapping[str, Any],
    phase_a_completion_value: Mapping[str, Any],
    truth_binding_value: Mapping[str, Any],
    highres_depth_mm: np.ndarray,
    failure_code: str,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Retain nine UNKNOWN factor slots for frozen support-unobservable FARO failures."""

    require(failure_code in adapter._SUPPORT_UNOBSERVABLE_CODES, "R6_FARO_UNKNOWN_CODE_INVALID", "only support-unobservable FARO failures may become UNKNOWN slots")
    source = validate_phase_a_source_receipt(dict(source_value))
    candidate = validate_candidate_frame(dict(candidate_value))
    decision = validate_source_decision(dict(decision_value))
    phase_a = validate_phase_a_completion(dict(phase_a_completion_value))
    truth = validate_truth_binding(dict(truth_binding_value))
    raw = np.asarray(highres_depth_mm)
    require(raw.shape == adapter.HIGHRES_SHAPE_HW and raw.dtype == np.uint16 and adapter.canonical_sha256(raw) == truth["highres_depth_array_sha256"], "R6_FARO_DEPTH_INVALID", "unobservable FARO differs from truth binding")
    require(source["content_sha256"] == decision["phase_a_source_receipt_sha256"] == truth["phase_a_source_receipt_sha256"] and phase_a["content_sha256"] == truth["phase_a_completion_sha256"], "R6_FARO_LINEAGE_DRIFT", "unobservable FARO inputs do not share lineage")
    matrix = adapter._intrinsics_matrix(source["intrinsics_highres"]["matrix_3x3"])
    depth_m = raw.astype(np.float64) / 1000.0
    valid = np.isfinite(depth_m) & (depth_m >= adapter.DEPTH_RANGE_M[0]) & (depth_m <= adapter.DEPTH_RANGE_M[1])
    if decision["source_support_available"]:
        normal = adapter._normalize_vector(decision["direct_support_plane"]["normal_camera_xyz"], "GRAVITY_INVALID")
        camera_height = float(decision["direct_support_plane"]["camera_height_m"])
    else:
        normal = adapter._normalize_vector(source["gravity_up_camera_xyz"], "GRAVITY_INVALID")
        camera_height = 1.20
    values = {
        "parent_id": source["parent_id"], "video_id": source["video_id"], "physical_frame_id": source["physical_frame_id"],
        "phase_a_source_receipt_sha256": source["content_sha256"], "source_decision_sha256": decision["content_sha256"],
        "phase_a_completion_sha256": phase_a["content_sha256"], "truth_binding_sha256": truth["content_sha256"],
        "highres_depth_array_sha256": adapter.canonical_sha256(raw), "max_source_timestamp_ns": int(source["max_source_timestamp_ns"]),
        "intrinsics": _immutable(matrix, np.float64), "depth_m": _immutable(depth_m, np.float64), "valid_depth": _immutable(valid, np.bool_),
        "points_camera_xyz": _immutable(np.empty((0, 3)), np.float64), "pixels_uv": _immutable(np.empty((0, 2)), np.int32),
        "support_normal_camera_xyz": _immutable(normal, np.float64), "camera_height_m": camera_height,
        "support_count": 0, "support_fraction": 0.0, "support_slope_degrees": 0.0, "support_median_residual_m": 0.0,
    }
    provisional = R6FaroGeometry(**values, content_sha256="0" * 64)
    geometry = validate_faro_geometry(R6FaroGeometry(**values, content_sha256=adapter.canonical_sha256(_geometry_payload(provisional))))
    queries = [adapter._validate_query_receipt(row) for row in adapter._build_geometry_query_receipts(physical_frame_id=geometry.physical_frame_id, source_frame_receipt_sha256=source["content_sha256"], max_source_timestamp_ns=geometry.max_source_timestamp_ns, support_normal_camera_xyz=normal, camera_height_m=camera_height)]
    prepared, _ = _prepared_and_plane(source, candidate, native_depth_m, decision)
    baseline = r5._validate_mode_result(source_factor._failed_mode(prepared.raw_depth_sha256, failure_code), "r6.baseline")
    selected = r5._validate_mode_result(source_factor._failed_mode(prepared.anchored_depth_sha256, failure_code), "r6.selected_support_boundary") if decision["source_support_available"] else baseline
    output: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for query in queries:
        scoring = _truth_scoring_record(geometry, query, baseline, selected, support_unobservable_code=failure_code)
        components = split.build_factor_components(
            analysis_role=ANALYSIS_ROLE, parent_id=source["parent_id"], physical_frame_id=source["physical_frame_id"], query_id=query["query_id"], grid_index=query["grid_index"],
            source_frame_receipt_sha256=source["content_sha256"], candidate_frame_record_sha256=candidate["content_sha256"], r6_phase_a_policy_seal_sha256=decision["content_sha256"],
            query_receipt_sha256=query["content_sha256"], truth_scoring_record_sha256=scoring["content_sha256"], source_support_available=decision["source_support_available"],
            phase_a_selected_branch=decision["selected_branch"], baseline=baseline, selected_support_boundary=selected,
        )
        output.append((scoring, components, split.build_composite_query(components)))
    return output


def summarize(records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    pairs = [(components, composite) for _, components, composite in records]
    return split.summarize_factor_split_pairs(pairs, analysis_role=ANALYSIS_ROLE, expected_parent_frame_counts=expected_parent_frame_counts())


__all__ = [
    "ANALYSIS_ROLE", "EXPECTED_FRAME_COUNT", "EXPECTED_PARENT_FRAME_COUNTS", "EXPECTED_QUERY_COUNT", "PHASE_A_ASSET_ROLES", "POLICY_ID", "ROSTER",
    "R6ConfirmationError", "R6FaroGeometry", "build_candidate_completion", "build_candidate_frame", "build_candidate_input", "build_inference_receipt",
    "build_phase_a_completion", "build_phase_a_source_receipt", "build_source_decision", "build_truth_binding", "derive_faro_geometry", "evaluate_frame", "evaluate_unobservable_faro_frame",
    "expected_parent_frame_counts", "infer_candidate", "summarize", "validate_candidate_completion", "validate_candidate_frame", "validate_candidate_input",
    "validate_expected_keys", "validate_faro_geometry", "validate_inference_receipt", "validate_phase_a_completion", "validate_phase_a_source_receipt",
    "validate_source_decision", "validate_source_scale", "validate_truth_binding",
]
