"""Parent-disjoint R5 confirmation mechanics for the frozen R4A policy.

This module introduces an independent R5 role.  It does not relax any of the
older ADAPTER_FIT or O0R_EVAL_CANDIDATE public APIs.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale
from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_support as r3
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


R5_ROLE = "R5_TASK_METRIC_CONFIRMATION"
POLICY_ID = "DIRECT_WHEN_SOURCE_SUPPORT_AVAILABLE_ELSE_R1_BASELINE_V1"
R5_ROSTER = (
    ("470974", "47332075"),
    ("469216", "47332946"),
    ("423614", "42898071"),
    ("467370", "47333776"),
    ("469460", "47333043"),
    ("438794", "44358241"),
    ("467346", "47333876"),
    ("472473", "47204786"),
)
EXPECTED_PARENT_FRAME_COUNTS = (25, 16, 11, 24, 23, 26, 43, 43)
EXPECTED_FRAME_COUNT = sum(EXPECTED_PARENT_FRAME_COUNTS)
EXPECTED_QUERY_COUNT = EXPECTED_FRAME_COUNT * 9
EXPECTED_IDENTITY_SEQUENCE_SHA256 = "52CFCC0CC37ED9DF2B7B3A5C99A617661062E600EB75B5790FC96225D7765B6F"

CANDIDATE_INPUT_SCHEMA = "blindassist.taro.o0r.r5_candidate_input.v1"
INFERENCE_RECEIPT_SCHEMA = "blindassist.taro.o0r.r5_depthart_inference_receipt.v1"
CANDIDATE_FRAME_SCHEMA = "blindassist.taro.o0r.r5_candidate_frame.v1"
CANDIDATE_COMPLETION_SCHEMA = "blindassist.taro.o0r.r5_candidate_phase_completion.v1"
SOURCE_DECISION_SCHEMA = "blindassist.taro.o0r.r5_source_decision.v1"
PHASE_A_COMPLETION_SCHEMA = "blindassist.taro.o0r.r5_phase_a_completion.v1"
QUERY_RECORD_SCHEMA = "blindassist.taro.o0r.r5_query_record.v1"
SUMMARY_SCHEMA = "blindassist.taro.o0r.r5_confirmation_summary.v1"
FARO_UNOBSERVABLE_SCHEMA = "blindassist.taro.o0r.r5_faro_unobservable.v1"
QUERY_UNOBSERVABLE_SLOT_SCHEMA = "blindassist.taro.o0r.r5_query_unobservable_slot.v1"

CLAIM_CEILING = {
    "scope": "EXACT_EIGHT_FORMER_ADAPTER_FIT_PARENTS_211_FRAMES_ARKITSCENES_TRAIN_LANDSCAPE",
    "use": "PARENT_DISJOINT_TASK_METRIC_CONFIRMATION_OF_FROZEN_ZERO_PARAMETER_R4A_POLICY",
    "former_uncertainty_fit_parents": True,
    "untouched_external_dataset_validation": False,
    "excluded_claims": ["FORMAL_O0R_PASS", "RGB_ONLY_CAPABILITY", "DEPLOYMENT", "PRODUCT", "SAFETY"],
}

_SHA256 = re.compile(r"^[0-9A-F]{64}$")


class R5ConfirmationError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R5ConfirmationError(code, message, **context)


def _canonical(value: Any) -> Any:
    return copy.deepcopy(value)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    require("content_sha256" not in result, "R5_SEAL_COLLISION", "caller supplied a content hash")
    result["content_sha256"] = adapter.canonical_sha256(result)
    return result


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R5_RECORD_INVALID", "sealed value must be an object", schema=schema)
    result = copy.deepcopy(value)
    observed = result.pop("content_sha256", None)
    require(isinstance(observed, str) and bool(_SHA256.fullmatch(observed)) and adapter.canonical_sha256(result) == observed, "R5_SEAL_MISMATCH", "record content hash drift", schema=schema)
    result["content_sha256"] = observed
    require(result.get("schema") == schema, "R5_SCHEMA_DRIFT", "record schema drift", expected=schema)
    return result


def _hash(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(_SHA256.fullmatch(value)), "R5_HASH_INVALID", "SHA-256 binding is malformed", field=field)
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _immutable(value: Any, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(contiguous.shape)
    require(not result.flags.writeable, "R5_IMMUTABLE_ARRAY_FAILURE", "array backing is mutable")
    return result


def _validate_expected_keys(expected_keys: Sequence[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    keys = [(str(parent), str(video), str(token)) for parent, video, token in expected_keys]
    require(len(keys) == EXPECTED_FRAME_COUNT and len(keys) == len(set(keys)), "R5_EXPECTED_KEY_SEQUENCE_INVALID", "R5 expected frame sequence is not exact")
    counts = Counter((parent, video) for parent, video, _ in keys)
    require(
        list(counts) == list(R5_ROSTER)
        and tuple(counts[identity] for identity in R5_ROSTER) == EXPECTED_PARENT_FRAME_COUNTS,
        "R5_EXPECTED_KEY_SEQUENCE_INVALID",
        "R5 expected frame sequence roster/count drift",
    )
    identities = []
    for parent, video, token in keys:
        adapter.decimal_timestamp_ns(token)
        identities.append(
            {
                "visit_id": parent,
                "video_id": video,
                "timestamp_token": token,
                "physical_frame_id": f"{video}:{token}",
            }
        )
    require(adapter.canonical_sha256(identities) == EXPECTED_IDENTITY_SEQUENCE_SHA256, "R5_EXPECTED_KEY_SEQUENCE_HASH_DRIFT", "R5 expected frame identity sequence differs from the amendment")
    return keys


def validate_r5_source_receipt(value: Any) -> dict[str, Any]:
    try:
        source = adapter._validate_base_receipt(dict(value))
    except (TypeError, adapter.AdapterError) as error:
        if isinstance(error, adapter.AdapterError):
            raise R5ConfirmationError(error.code, str(error), **error.context) from error
        raise R5ConfirmationError("R5_SOURCE_RECEIPT_INVALID", "source receipt is malformed") from error
    identity = (str(source["parent_id"]), str(source["session_id"]))
    require(source["source_role"] == "ADAPTER_FIT" and identity in R5_ROSTER, "R5_SOURCE_ROLE_INVALID", "R5 source must be one of the exact former ADAPTER_FIT identities", identity=identity)
    return source


def build_candidate_input(source_frame_receipt: Mapping[str, Any], color_rgb_u8: np.ndarray) -> dict[str, Any]:
    source = validate_r5_source_receipt(source_frame_receipt)
    color = np.asarray(color_rgb_u8)
    require(color.shape == (*adapter.HIGHRES_SHAPE_HW, 3) and color.dtype == np.uint8, "R5_COLOR_INVALID", "R5 RGB must be uint8 1440x1920x3")
    try:
        adapter._validate_bound_decoded_payload(source, "color", color)
    except adapter.AdapterError as error:
        raise R5ConfirmationError(error.code, str(error), **error.context) from error
    token = source["sensor_timestamp"]["decimal_token"]
    receipt = _seal(
        {
            "schema": CANDIDATE_INPUT_SCHEMA,
            "r5_role": R5_ROLE,
            "predecessor_source_role": "ADAPTER_FIT",
            "parent_id": source["parent_id"],
            "video_id": source["session_id"],
            "timestamp_token": token,
            "physical_frame_id": source["physical_frame_id"],
            "source_frame_receipt_sha256": source["content_sha256"],
            "color_member_sha256": source["asset_bindings"]["color"]["sha256"],
            "color_decoded_content_sha256": adapter.canonical_sha256(color),
            "intrinsics_highres": source["intrinsics_highres"],
            "effective_intrinsics_sha256": adapter.canonical_sha256(source["intrinsics_highres"]),
            "allowed_model_inputs": ["REGISTERED_RGB", "BOUND_EFFECTIVE_K"],
            "truth_payload_read": False,
            "faro_payload_read": False,
            "prior_eval_outcome_read": False,
        }
    )
    return validate_candidate_input(receipt)


def validate_candidate_input(value: Any) -> dict[str, Any]:
    receipt = _validate_seal(value, CANDIDATE_INPUT_SCHEMA)
    expected = {
        "schema", "r5_role", "predecessor_source_role", "parent_id", "video_id", "timestamp_token",
        "physical_frame_id", "source_frame_receipt_sha256", "color_member_sha256",
        "color_decoded_content_sha256", "intrinsics_highres", "effective_intrinsics_sha256",
        "allowed_model_inputs", "truth_payload_read", "faro_payload_read", "prior_eval_outcome_read",
        "content_sha256",
    }
    require(set(receipt) == expected, "R5_CANDIDATE_INPUT_KEY_SET", "candidate input fields drift")
    identity = (str(receipt["parent_id"]), str(receipt["video_id"]))
    require(receipt["r5_role"] == R5_ROLE and receipt["predecessor_source_role"] == "ADAPTER_FIT" and identity in R5_ROSTER, "R5_CANDIDATE_ROLE_INVALID", "candidate input role/roster drift")
    require(receipt["physical_frame_id"] == f"{receipt['video_id']}:{receipt['timestamp_token']}", "R5_CANDIDATE_IDENTITY_INVALID", "candidate identity drift")
    adapter.decimal_timestamp_ns(receipt["timestamp_token"])
    for field in ("source_frame_receipt_sha256", "color_member_sha256", "color_decoded_content_sha256", "effective_intrinsics_sha256"):
        _hash(receipt[field], field)
    matrix = adapter._intrinsics_matrix(receipt["intrinsics_highres"]["matrix_3x3"])
    require(adapter.canonical_sha256(receipt["intrinsics_highres"]) == receipt["effective_intrinsics_sha256"], "R5_CANDIDATE_K_DRIFT", "candidate K hash drift")
    require(matrix.shape == (3, 3), "R5_CANDIDATE_K_DRIFT", "candidate K shape drift")
    require(receipt["allowed_model_inputs"] == ["REGISTERED_RGB", "BOUND_EFFECTIVE_K"] and receipt["truth_payload_read"] is False and receipt["faro_payload_read"] is False and receipt["prior_eval_outcome_read"] is False, "R5_CANDIDATE_TRUTH_LEAK", "candidate input crossed the Phase-A firewall")
    return receipt


def build_inference_receipt(
    candidate_input_receipt: Mapping[str, Any],
    color_rgb_u8: np.ndarray,
    input_tensor_nchw: np.ndarray,
    resized_intrinsics_n33: np.ndarray,
    native_depth_m: np.ndarray,
    highres_depth_m: np.ndarray,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = validate_candidate_input(candidate_input_receipt)
    color = np.asarray(color_rgb_u8)
    tensor = np.asarray(input_tensor_nchw)
    resized_k = np.asarray(resized_intrinsics_n33)
    native = np.asarray(native_depth_m)
    highres = np.asarray(highres_depth_m)
    require(adapter.canonical_sha256(color) == candidate["color_decoded_content_sha256"], "R5_INFERENCE_COLOR_DRIFT", "inference RGB differs from candidate receipt")
    require(tensor.shape == (1, 3, *depthart_runner.NATIVE_SHAPE_HW) and tensor.dtype == np.float32 and bool(np.all(np.isfinite(tensor))), "R5_INFERENCE_INPUT_INVALID", "preprocessed tensor is invalid")
    require(resized_k.shape == (1, 3, 3) and resized_k.dtype == np.float32 and bool(np.all(np.isfinite(resized_k))), "R5_INFERENCE_INPUT_INVALID", "resized K is invalid")
    require(native.shape == depthart_runner.NATIVE_SHAPE_HW and native.dtype == np.float32 and bool(np.all(np.isfinite(native))), "R5_NATIVE_DEPTH_INVALID", "native output is invalid")
    require(highres.shape == adapter.HIGHRES_SHAPE_HW and highres.dtype == np.float32 and bool(np.all(np.isfinite(highres))), "R5_HIGHRES_DEPTH_INVALID", "registered output is invalid")
    replay = depthart_runner.upsample_native_depth(native)
    require(adapter.canonical_sha256(replay) == adapter.canonical_sha256(highres), "R5_POSTPROCESS_DRIFT", "highres output is not the frozen native-depth resize")
    receipt = _seal(
        {
            "schema": INFERENCE_RECEIPT_SCHEMA,
            "r5_role": R5_ROLE,
            "model_id": adapter.BASELINE_MODEL_ID,
            "checkpoint_sha256": adapter.BASELINE_CHECKPOINT_SHA256,
            "preprocess_id": depthart_runner.PREPROCESS_ID,
            "postprocess_id": depthart_runner.POSTPROCESS_ID,
            "candidate_input_receipt_sha256": candidate["content_sha256"],
            "parent_id": candidate["parent_id"],
            "video_id": candidate["video_id"],
            "timestamp_token": candidate["timestamp_token"],
            "physical_frame_id": candidate["physical_frame_id"],
            "input_color_decoded_content_sha256": candidate["color_decoded_content_sha256"],
            "effective_intrinsics_sha256": candidate["effective_intrinsics_sha256"],
            "input_tensor_sha256": adapter.canonical_sha256(tensor),
            "resized_intrinsics_sha256": adapter.canonical_sha256(resized_k),
            "native_output_array_sha256": adapter.canonical_sha256(native),
            "highres_output_array_sha256": adapter.canonical_sha256(highres),
            "runtime_identity": dict(runtime_identity),
            "truth_alignment_used": False,
            "truth_payload_read": False,
            "faro_payload_read": False,
            "prior_eval_outcome_read": False,
        }
    )
    return validate_inference_receipt(receipt)


def validate_inference_receipt(value: Any) -> dict[str, Any]:
    receipt = _validate_seal(value, INFERENCE_RECEIPT_SCHEMA)
    expected = {
        "schema", "r5_role", "model_id", "checkpoint_sha256", "preprocess_id", "postprocess_id",
        "candidate_input_receipt_sha256", "parent_id", "video_id", "timestamp_token",
        "physical_frame_id", "input_color_decoded_content_sha256", "effective_intrinsics_sha256",
        "input_tensor_sha256", "resized_intrinsics_sha256", "native_output_array_sha256",
        "highres_output_array_sha256", "runtime_identity", "truth_alignment_used", "truth_payload_read",
        "faro_payload_read", "prior_eval_outcome_read", "content_sha256",
    }
    require(set(receipt) == expected, "R5_INFERENCE_KEY_SET", "inference receipt fields drift")
    require(receipt["r5_role"] == R5_ROLE and (receipt["parent_id"], receipt["video_id"]) in R5_ROSTER, "R5_INFERENCE_ROLE_INVALID", "inference role/roster drift")
    require(receipt["model_id"] == adapter.BASELINE_MODEL_ID and receipt["checkpoint_sha256"] == adapter.BASELINE_CHECKPOINT_SHA256, "R5_INFERENCE_MODEL_DRIFT", "model identity drift")
    require(receipt["preprocess_id"] == depthart_runner.PREPROCESS_ID and receipt["postprocess_id"] == depthart_runner.POSTPROCESS_ID, "R5_INFERENCE_TRANSFORM_DRIFT", "transform identity drift")
    require(receipt["physical_frame_id"] == f"{receipt['video_id']}:{receipt['timestamp_token']}", "R5_INFERENCE_IDENTITY_INVALID", "inference identity drift")
    for field in ("checkpoint_sha256", "candidate_input_receipt_sha256", "input_color_decoded_content_sha256", "effective_intrinsics_sha256", "input_tensor_sha256", "resized_intrinsics_sha256", "native_output_array_sha256", "highres_output_array_sha256"):
        _hash(receipt[field], field)
    require(isinstance(receipt["runtime_identity"], dict) and bool(receipt["runtime_identity"]), "R5_INFERENCE_RUNTIME_INVALID", "runtime identity is missing")
    require(receipt["truth_alignment_used"] is False and receipt["truth_payload_read"] is False and receipt["faro_payload_read"] is False and receipt["prior_eval_outcome_read"] is False, "R5_INFERENCE_TRUTH_LEAK", "inference crossed the Phase-A firewall")
    return receipt


def infer_candidate(
    model: Any,
    candidate_input_receipt: Mapping[str, Any],
    color_rgb_u8: np.ndarray,
    runtime_identity: Mapping[str, Any],
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    import torch

    candidate = validate_candidate_input(candidate_input_receipt)
    matrix = np.asarray(candidate["intrinsics_highres"]["matrix_3x3"], dtype=np.float32)
    tensor, resized_k = depthart_runner.preprocess_depthart_input(np.asarray(color_rgb_u8), matrix)
    with torch.inference_mode():
        prediction = model(torch.from_numpy(tensor).to(device), torch.from_numpy(resized_k).to(device))
    native = prediction.detach().float().cpu().numpy()
    require(native.shape == (1, *depthart_runner.NATIVE_SHAPE_HW), "R5_NATIVE_DEPTH_INVALID", "official output must be 1x448x608", actual=list(native.shape))
    native = np.ascontiguousarray(native[0], dtype=np.float32)
    highres = depthart_runner.upsample_native_depth(native)
    receipt = build_inference_receipt(candidate, color_rgb_u8, tensor, resized_k, native, highres, runtime_identity)
    return {"native_depth_m": native, "highres_depth_m": highres, "inference_receipt": receipt}


def build_candidate_frame_record(candidate_input_receipt: Mapping[str, Any], inference_receipt: Mapping[str, Any], native_blob: Mapping[str, Any]) -> dict[str, Any]:
    candidate = validate_candidate_input(candidate_input_receipt)
    inference = validate_inference_receipt(inference_receipt)
    require(inference["candidate_input_receipt_sha256"] == candidate["content_sha256"], "R5_CANDIDATE_LINEAGE_DRIFT", "inference does not bind candidate input")
    blob = dict(native_blob)
    expected_blob = {"path", "bytes", "sha256", "array_sha256", "shape_hw", "dtype", "encoding"}
    require(set(blob) == expected_blob and blob["array_sha256"] == inference["native_output_array_sha256"], "R5_NATIVE_BLOB_BINDING_INVALID", "native blob binding drift")
    record = _seal(
        {
            "schema": CANDIDATE_FRAME_SCHEMA,
            "r5_role": R5_ROLE,
            "parent_id": candidate["parent_id"],
            "video_id": candidate["video_id"],
            "timestamp_token": candidate["timestamp_token"],
            "physical_frame_id": candidate["physical_frame_id"],
            "candidate_input_receipt": candidate,
            "inference_receipt": inference,
            "native_depth_blob": blob,
            "phase_a_truth_read": False,
        }
    )
    return validate_candidate_frame_record(record)


def validate_candidate_frame_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, CANDIDATE_FRAME_SCHEMA)
    expected = {"schema", "r5_role", "parent_id", "video_id", "timestamp_token", "physical_frame_id", "candidate_input_receipt", "inference_receipt", "native_depth_blob", "phase_a_truth_read", "content_sha256"}
    require(set(record) == expected and record["r5_role"] == R5_ROLE, "R5_CANDIDATE_FRAME_KEY_SET", "candidate frame fields drift")
    candidate = validate_candidate_input(record["candidate_input_receipt"])
    inference = validate_inference_receipt(record["inference_receipt"])
    identity = (record["parent_id"], record["video_id"], record["timestamp_token"], record["physical_frame_id"])
    require(identity == (candidate["parent_id"], candidate["video_id"], candidate["timestamp_token"], candidate["physical_frame_id"]) == (inference["parent_id"], inference["video_id"], inference["timestamp_token"], inference["physical_frame_id"]), "R5_CANDIDATE_FRAME_IDENTITY_DRIFT", "candidate frame lineage identity drift")
    blob = record["native_depth_blob"]
    require(isinstance(blob, dict) and set(blob) == {"path", "bytes", "sha256", "array_sha256", "shape_hw", "dtype", "encoding"}, "R5_NATIVE_BLOB_BINDING_INVALID", "native blob binding fields drift")
    require(blob["array_sha256"] == inference["native_output_array_sha256"] and blob["shape_hw"] == list(depthart_runner.NATIVE_SHAPE_HW) and blob["dtype"] == "float32" and blob["encoding"] == "DETERMINISTIC_GZIP_NPY_MTIME_0", "R5_NATIVE_BLOB_BINDING_INVALID", "native blob array binding drift")
    _hash(blob["sha256"], "native_depth_blob.sha256")
    require(record["phase_a_truth_read"] is False, "R5_CANDIDATE_FRAME_TRUTH_LEAK", "candidate frame crossed truth firewall")
    return record


def build_candidate_phase_completion(records: Sequence[Mapping[str, Any]], expected_keys: Sequence[tuple[str, str, str]]) -> dict[str, Any]:
    rows = [validate_candidate_frame_record(dict(row)) for row in records]
    keys = [(row["parent_id"], row["video_id"], row["timestamp_token"]) for row in rows]
    frozen_keys = _validate_expected_keys(expected_keys)
    require(keys == frozen_keys and len(keys) == len(set(keys)), "R5_CANDIDATE_PHASE_INCOMPLETE", "candidate records do not exactly cover the frozen sequence")
    counts = Counter(row["parent_id"] for row in rows)
    return validate_candidate_phase_completion(
        _seal(
            {
                "schema": CANDIDATE_COMPLETION_SCHEMA,
                "r5_role": R5_ROLE,
                "candidate_frame_count": len(rows),
                "candidate_key_sequence_sha256": adapter.canonical_sha256([list(key) for key in keys]),
                "candidate_record_hash_sequence_sha256": adapter.canonical_sha256([row["content_sha256"] for row in rows]),
                "parent_frame_counts": dict(sorted(counts.items())),
                "all_candidates_sealed_before_source_decisions": True,
                "faro_reads": 0,
                "task_metric_reads": 0,
                "prior_eval_outcome_reads": 0,
            }
        )
    )


def validate_candidate_phase_completion(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, CANDIDATE_COMPLETION_SCHEMA)
    expected = {"schema", "r5_role", "candidate_frame_count", "candidate_key_sequence_sha256", "candidate_record_hash_sequence_sha256", "parent_frame_counts", "all_candidates_sealed_before_source_decisions", "faro_reads", "task_metric_reads", "prior_eval_outcome_reads", "content_sha256"}
    require(set(record) == expected and record["r5_role"] == R5_ROLE, "R5_CANDIDATE_COMPLETION_KEY_SET", "candidate completion fields drift")
    require(record["candidate_frame_count"] == EXPECTED_FRAME_COUNT, "R5_CANDIDATE_COMPLETION_INVALID", "candidate completion count is not the frozen 211")
    _hash(record["candidate_key_sequence_sha256"], "candidate_key_sequence_sha256")
    _hash(record["candidate_record_hash_sequence_sha256"], "candidate_record_hash_sequence_sha256")
    require(record["all_candidates_sealed_before_source_decisions"] is True and record["faro_reads"] == record["task_metric_reads"] == record["prior_eval_outcome_reads"] == 0, "R5_CANDIDATE_COMPLETION_FIREWALL_BREACH", "candidate completion crossed the truth firewall")
    expected_counts = {parent: count for (parent, _), count in zip(R5_ROSTER, EXPECTED_PARENT_FRAME_COUNTS)}
    require(record["parent_frame_counts"] == expected_counts, "R5_CANDIDATE_COMPLETION_PARENT_COUNT_DRIFT", "candidate parent counts differ from the amendment")
    return record


def _scale_record(source: Mapping[str, Any], candidate: Mapping[str, Any], highres: np.ndarray, apple: np.ndarray, confidence: np.ndarray) -> dict[str, Any]:
    samples = apple_scale.sample_candidate_at_apple_centers(highres)
    estimate = apple_scale.estimate_source_metric_scale(apple, confidence, samples)
    return validate_source_scale_record(_seal(
        {
            "schema": "blindassist.taro.o0r.r5_source_scale.v1",
            "parent_id": source["parent_id"],
            "video_id": source["session_id"],
            "timestamp_token": source["sensor_timestamp"]["decimal_token"],
            "physical_frame_id": source["physical_frame_id"],
            "source_frame_receipt_sha256": source["content_sha256"],
            "candidate_frame_record_sha256": candidate["content_sha256"],
            "candidate_highres_depth_array_sha256": adapter.canonical_sha256(highres),
            "apple_depth_array_sha256": adapter.canonical_sha256(apple),
            "confidence_array_sha256": adapter.canonical_sha256(confidence),
            "estimator_id": apple_scale.ESTIMATOR_ID,
            **estimate,
            "faro_payload_read": False,
            "task_metric_read": False,
        }
    ))


def validate_source_scale_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, "blindassist.taro.o0r.r5_source_scale.v1")
    expected = {
        "schema", "parent_id", "video_id", "timestamp_token", "physical_frame_id",
        "source_frame_receipt_sha256", "candidate_frame_record_sha256",
        "candidate_highres_depth_array_sha256", "apple_depth_array_sha256",
        "confidence_array_sha256", "estimator_id", "evaluable", "reason_codes",
        "valid_pair_count", "selected_pixel_ids_sha256", "log_metric_scale",
        "metric_scale", "faro_payload_read", "task_metric_read", "content_sha256",
    }
    require(set(record) == expected, "R5_SOURCE_SCALE_KEY_SET", "R5 source scale fields drift")
    require(
        (record["parent_id"], record["video_id"]) in R5_ROSTER
        and record["physical_frame_id"] == f"{record['video_id']}:{record['timestamp_token']}",
        "R5_SOURCE_SCALE_IDENTITY_DRIFT",
        "R5 source scale identity drift",
    )
    adapter.decimal_timestamp_ns(record["timestamp_token"])
    for field in (
        "source_frame_receipt_sha256", "candidate_frame_record_sha256",
        "candidate_highres_depth_array_sha256", "apple_depth_array_sha256",
        "confidence_array_sha256", "selected_pixel_ids_sha256",
    ):
        _hash(record[field], field)
    require(record["estimator_id"] == apple_scale.ESTIMATOR_ID, "R5_SOURCE_SCALE_METHOD_DRIFT", "R5 source scale estimator drift")
    require(isinstance(record["valid_pair_count"], int) and record["valid_pair_count"] >= 0 and isinstance(record["reason_codes"], list), "R5_SOURCE_SCALE_INVALID", "R5 source scale support metadata drift")
    require(isinstance(record["evaluable"], bool), "R5_SOURCE_SCALE_INVALID", "R5 source scale evaluability is not Boolean")
    if record["evaluable"]:
        require(
            record["valid_pair_count"] >= apple_scale.MINIMUM_PAIR_COUNT
            and record["reason_codes"] == []
            and _finite(record["log_metric_scale"])
            and _finite(record["metric_scale"])
            and float(record["metric_scale"]) > 0.0,
            "R5_SOURCE_SCALE_INVALID",
            "evaluable R5 source scale metrics drift",
        )
        require(abs(math.exp(float(record["log_metric_scale"])) - float(record["metric_scale"])) <= 1e-10, "R5_SOURCE_SCALE_INVALID", "R5 metric/log scale mismatch")
    else:
        require(
            record["valid_pair_count"] < apple_scale.MINIMUM_PAIR_COUNT
            and record["reason_codes"] == ["APPLE_SCALE_COMMON_SUPPORT_INSUFFICIENT"]
            and record["log_metric_scale"] is None
            and record["metric_scale"] is None,
            "R5_SOURCE_SCALE_INVALID",
            "unevaluable R5 source scale carries metrics or wrong reason",
        )
    require(record["faro_payload_read"] is False and record["task_metric_read"] is False, "R5_SOURCE_SCALE_TRUTH_LEAK", "R5 source scale crossed Phase-A firewall")
    return record


def build_source_decision(
    source_frame_receipt: Mapping[str, Any],
    candidate_frame_record: Mapping[str, Any],
    native_depth_m: np.ndarray,
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, Any]:
    source = validate_r5_source_receipt(source_frame_receipt)
    candidate = validate_candidate_frame_record(candidate_frame_record)
    native = np.ascontiguousarray(native_depth_m, dtype=np.float32)
    inference = candidate["inference_receipt"]
    require(native.shape == depthart_runner.NATIVE_SHAPE_HW and adapter.canonical_sha256(native) == inference["native_output_array_sha256"], "R5_NATIVE_DEPTH_DRIFT", "native candidate differs from its sealed receipt")
    highres = depthart_runner.upsample_native_depth(native)
    require(adapter.canonical_sha256(highres) == inference["highres_output_array_sha256"], "R5_HIGHRES_DEPTH_DRIFT", "replayed candidate differs from its inference receipt")
    identity = (source["parent_id"], source["session_id"], source["sensor_timestamp"]["decimal_token"], source["physical_frame_id"])
    require(identity == (candidate["parent_id"], candidate["video_id"], candidate["timestamp_token"], candidate["physical_frame_id"]), "R5_SOURCE_CANDIDATE_IDENTITY_DRIFT", "source and candidate identities differ")
    require(candidate["candidate_input_receipt"]["source_frame_receipt_sha256"] == source["content_sha256"], "R5_SOURCE_CANDIDATE_LINEAGE_DRIFT", "candidate does not bind source receipt")
    apple = np.asarray(apple_depth_mm)
    conf = np.asarray(confidence)
    require(apple.shape == adapter.APPLE_SHAPE_HW and apple.dtype == np.uint16, "R5_APPLE_DEPTH_INVALID", "AppleDepth must be uint16 192x256")
    require(conf.shape == adapter.APPLE_SHAPE_HW and conf.dtype == np.uint8 and bool(np.all(conf <= 2)), "R5_CONFIDENCE_INVALID", "confidence must be uint8 0..2")
    try:
        adapter._validate_bound_decoded_payload(source, "lowres_depth", apple)
        adapter._validate_bound_decoded_payload(source, "confidence", conf)
    except adapter.AdapterError as error:
        raise R5ConfirmationError(error.code, str(error), **error.context) from error
    scale = _scale_record(source, candidate, highres, apple, conf)
    plane_block: dict[str, Any] | None = None
    failure_code: str | None = None
    anchored_hash: str | None = None
    if scale["evaluable"]:
        anchored = np.ascontiguousarray(highres.astype(np.float64) * float(scale["metric_scale"]), dtype=np.float64)
        anchored_hash = adapter.canonical_sha256(anchored)
        apple_m = apple.astype(np.float64) / 1000.0
        lower, upper = apple_scale.DEPTH_RANGE_M
        support_mask = (conf == 2) & (apple_m >= lower) & (apple_m <= upper)
        support_ids = np.flatnonzero(support_mask).astype(np.int64)
        low = source["lowres_intrinsics_source"]
        low_k = adapter._intrinsics_matrix(
            [[low["fx"], 0.0, low["cx"]], [0.0, low["fy"], low["cy"]], [0.0, 0.0, 1.0]],
            adapter.APPLE_SHAPE_HW,
        )
        gravity = adapter._normalize_vector(source["gravity_up_camera_xyz"], "GRAVITY_INVALID")
        try:
            points, pixels = adapter._unproject(apple_m, support_mask, low_k, 1)
            plane = adapter._fit_support_plane(points, gravity)
            camera_height = float(plane["camera_height_m"])
            require(r3.CAMERA_HEIGHT_RANGE_M[0] <= camera_height <= r3.CAMERA_HEIGHT_RANGE_M[1], "DIRECT_APPLE_SUPPORT_HEIGHT_IMPLAUSIBLE", "Apple support height leaves frozen physical range", camera_height_m=camera_height)
        except (adapter.AdapterError, R5ConfirmationError) as error:
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
    selected_branch = "DIRECT_APPLE_SUPPORT" if available else "R1_BASELINE"
    decision = _seal(
        {
            "schema": SOURCE_DECISION_SCHEMA,
            "r5_role": R5_ROLE,
            "policy_id": POLICY_ID,
            "parent_id": source["parent_id"],
            "video_id": source["session_id"],
            "timestamp_token": source["sensor_timestamp"]["decimal_token"],
            "physical_frame_id": source["physical_frame_id"],
            "source_frame_receipt_sha256": source["content_sha256"],
            "candidate_frame_record_sha256": candidate["content_sha256"],
            "candidate_highres_depth_array_sha256": adapter.canonical_sha256(highres),
            "anchored_candidate_depth_array_sha256": anchored_hash,
            "intrinsics_highres_sha256": adapter.canonical_sha256(
                adapter._intrinsics_matrix(source["intrinsics_highres"]["matrix_3x3"])
            ),
            "gravity_up_camera_xyz_sha256": adapter.canonical_sha256(adapter._normalize_vector(source["gravity_up_camera_xyz"], "GRAVITY_INVALID")),
            "scale_record": scale,
            "source_support_available": available,
            "source_failure_code": failure_code,
            "direct_support_plane": plane_block,
            "selected_branch": selected_branch,
            "selection_fields_read": ["source_support_available"],
            "outcome_dependent_reselection_allowed": False,
            "faro_payload_read": False,
            "task_metric_read": False,
            "prior_eval_outcome_read": False,
        }
    )
    return validate_source_decision(decision)


def validate_source_decision(value: Any) -> dict[str, Any]:
    decision = _validate_seal(value, SOURCE_DECISION_SCHEMA)
    expected = {
        "schema", "r5_role", "policy_id", "parent_id", "video_id", "timestamp_token", "physical_frame_id",
        "source_frame_receipt_sha256", "candidate_frame_record_sha256", "candidate_highres_depth_array_sha256",
        "anchored_candidate_depth_array_sha256", "intrinsics_highres_sha256", "gravity_up_camera_xyz_sha256",
        "scale_record", "source_support_available", "source_failure_code", "direct_support_plane",
        "selected_branch", "selection_fields_read", "outcome_dependent_reselection_allowed", "faro_payload_read",
        "task_metric_read", "prior_eval_outcome_read", "content_sha256",
    }
    require(set(decision) == expected and decision["r5_role"] == R5_ROLE and decision["policy_id"] == POLICY_ID, "R5_SOURCE_DECISION_KEY_SET", "source decision fields/method drift")
    require((decision["parent_id"], decision["video_id"]) in R5_ROSTER and decision["physical_frame_id"] == f"{decision['video_id']}:{decision['timestamp_token']}", "R5_SOURCE_DECISION_IDENTITY_DRIFT", "source decision identity drift")
    for field in ("source_frame_receipt_sha256", "candidate_frame_record_sha256", "candidate_highres_depth_array_sha256", "intrinsics_highres_sha256", "gravity_up_camera_xyz_sha256"):
        _hash(decision[field], field)
    scale = validate_source_scale_record(decision["scale_record"])
    require(
        (scale["parent_id"], scale["video_id"], scale["timestamp_token"], scale["physical_frame_id"])
        == (decision["parent_id"], decision["video_id"], decision["timestamp_token"], decision["physical_frame_id"])
        and scale["source_frame_receipt_sha256"] == decision["source_frame_receipt_sha256"]
        and scale["candidate_frame_record_sha256"] == decision["candidate_frame_record_sha256"]
        and scale["candidate_highres_depth_array_sha256"] == decision["candidate_highres_depth_array_sha256"],
        "R5_SOURCE_SCALE_LINEAGE_DRIFT",
        "source scale lineage drift",
    )
    available = decision["source_support_available"]
    require(isinstance(available, bool) and decision["selected_branch"] == ("DIRECT_APPLE_SUPPORT" if available else "R1_BASELINE"), "R5_SOURCE_DECISION_POLICY_DRIFT", "branch does not rederive from source availability")
    require(decision["selection_fields_read"] == ["source_support_available"] and decision["outcome_dependent_reselection_allowed"] is False, "R5_SOURCE_DECISION_SELECTION_DRIFT", "decision used non-source selection fields")
    if available:
        _hash(decision["anchored_candidate_depth_array_sha256"], "anchored_candidate_depth_array_sha256")
        require(decision["source_failure_code"] is None and isinstance(decision["direct_support_plane"], dict), "R5_SOURCE_DECISION_PLANE_DRIFT", "available source decision lacks plane")
        plane = decision["direct_support_plane"]
        plane_keys = {
            "normal_camera_xyz", "camera_height_m", "support_count", "support_fraction",
            "slope_degrees", "median_residual_m", "apple_support_point_count",
            "apple_support_pixel_ids_sha256", "sampled_pixels_sha256", "sampled_points_sha256",
            "candidate_depth_used_for_support_mask",
        }
        require(set(plane) == plane_keys, "R5_SOURCE_DECISION_PLANE_DRIFT", "direct Apple plane fields drift")
        normal = np.asarray(plane.get("normal_camera_xyz"), dtype=np.float64)
        require(normal.shape == (3,) and bool(np.all(np.isfinite(normal))) and abs(float(np.linalg.norm(normal)) - 1.0) <= 1e-9, "R5_SOURCE_DECISION_PLANE_DRIFT", "plane normal is invalid")
        require(
            _finite(plane.get("camera_height_m"))
            and r3.CAMERA_HEIGHT_RANGE_M[0] <= float(plane["camera_height_m"]) <= r3.CAMERA_HEIGHT_RANGE_M[1]
            and isinstance(plane.get("support_count"), int)
            and plane["support_count"] >= adapter.MINIMUM_SUPPORT_POINTS
            and isinstance(plane.get("apple_support_point_count"), int)
            and plane["apple_support_point_count"] >= adapter.MINIMUM_SUPPORT_POINTS,
            "R5_SOURCE_DECISION_PLANE_DRIFT",
            "direct Apple plane support metrics drift",
        )
        for field in ("apple_support_pixel_ids_sha256", "sampled_pixels_sha256", "sampled_points_sha256"):
            _hash(plane[field], field)
        require(plane.get("candidate_depth_used_for_support_mask") is False, "R5_SOURCE_DECISION_SUPPORT_LEAK", "candidate depth influenced Apple support mask")
    else:
        require(decision["anchored_candidate_depth_array_sha256"] is None or scale["evaluable"] is True, "R5_SOURCE_DECISION_SCALE_DRIFT", "unevaluable scale carries anchored depth")
        require(isinstance(decision["source_failure_code"], str) and bool(decision["source_failure_code"]) and decision["direct_support_plane"] is None, "R5_SOURCE_DECISION_FAILURE_DRIFT", "fallback decision lacks exact failure")
    require(decision["faro_payload_read"] is False and decision["task_metric_read"] is False and decision["prior_eval_outcome_read"] is False, "R5_SOURCE_DECISION_TRUTH_LEAK", "source decision crossed Phase-A firewall")
    return decision


def build_phase_a_completion(
    candidate_completion: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    expected_keys: Sequence[tuple[str, str, str]],
    *,
    read_counts: Mapping[str, int],
) -> dict[str, Any]:
    candidate_phase = validate_candidate_phase_completion(candidate_completion)
    rows = [validate_source_decision(dict(row)) for row in decisions]
    keys = [(row["parent_id"], row["video_id"], row["timestamp_token"]) for row in rows]
    frozen_keys = _validate_expected_keys(expected_keys)
    require(keys == frozen_keys and len(keys) == len(set(keys)), "R5_PHASE_A_DECISION_SEQUENCE_DRIFT", "source decisions do not exactly cover candidate sequence")
    require(
        adapter.canonical_sha256([list(key) for key in keys]) == candidate_phase["candidate_key_sequence_sha256"]
        and adapter.canonical_sha256([row["candidate_frame_record_sha256"] for row in rows]) == candidate_phase["candidate_record_hash_sequence_sha256"],
        "R5_PHASE_A_CANDIDATE_SEQUENCE_DRIFT",
        "source decisions do not bind the completed candidate sequence",
    )
    zero_roles = ("FARO", "QUERY_TRUTH", "COMPACT_TRUTH", "TASK_METRIC", "PRIOR_EVAL_OUTCOME")
    require(all(int(read_counts.get(role, -1)) == 0 for role in zero_roles), "R5_PHASE_A_READ_FIREWALL_BREACH", "forbidden Phase-A payload was read", read_counts=dict(read_counts))
    record = _seal(
        {
            "schema": PHASE_A_COMPLETION_SCHEMA,
            "r5_role": R5_ROLE,
            "policy_id": POLICY_ID,
            "candidate_phase_completion_sha256": candidate_phase["content_sha256"],
            "physical_frame_count": len(rows),
            "source_decision_key_sequence_sha256": adapter.canonical_sha256([list(key) for key in keys]),
            "source_decision_hash_sequence_sha256": adapter.canonical_sha256([row["content_sha256"] for row in rows]),
            "direct_selected_frame_count": sum(row["selected_branch"] == "DIRECT_APPLE_SUPPORT" for row in rows),
            "baseline_fallback_frame_count": sum(row["selected_branch"] == "R1_BASELINE" for row in rows),
            "read_counts": {role: int(read_counts.get(role, 0)) for role in sorted(set(read_counts) | set(zero_roles))},
            "forbidden_zero_read_roles": list(zero_roles),
            "all_candidates_before_source_decisions": True,
            "all_source_decisions_before_phase_b": True,
        }
    )
    return validate_phase_a_completion(record)


def validate_phase_a_completion(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, PHASE_A_COMPLETION_SCHEMA)
    expected = {"schema", "r5_role", "policy_id", "candidate_phase_completion_sha256", "physical_frame_count", "source_decision_key_sequence_sha256", "source_decision_hash_sequence_sha256", "direct_selected_frame_count", "baseline_fallback_frame_count", "read_counts", "forbidden_zero_read_roles", "all_candidates_before_source_decisions", "all_source_decisions_before_phase_b", "content_sha256"}
    require(set(record) == expected and record["r5_role"] == R5_ROLE and record["policy_id"] == POLICY_ID, "R5_PHASE_A_COMPLETION_KEY_SET", "Phase-A completion fields drift")
    for field in ("candidate_phase_completion_sha256", "source_decision_key_sequence_sha256", "source_decision_hash_sequence_sha256"):
        _hash(record[field], field)
    zero_roles = ["FARO", "QUERY_TRUTH", "COMPACT_TRUTH", "TASK_METRIC", "PRIOR_EVAL_OUTCOME"]
    require(record["forbidden_zero_read_roles"] == zero_roles and all(record["read_counts"].get(role) == 0 for role in zero_roles), "R5_PHASE_A_READ_FIREWALL_BREACH", "Phase-A forbidden read count drift")
    require(record["direct_selected_frame_count"] + record["baseline_fallback_frame_count"] == record["physical_frame_count"], "R5_PHASE_A_COUNT_DRIFT", "Phase-A branch counts do not cover frames")
    require(record["physical_frame_count"] == EXPECTED_FRAME_COUNT, "R5_PHASE_A_COUNT_DRIFT", "Phase-A does not contain the frozen 211 frames")
    require(record["all_candidates_before_source_decisions"] is True and record["all_source_decisions_before_phase_b"] is True, "R5_PHASE_A_ORDER_DRIFT", "Phase-A order guarantee drift")
    return record


@dataclass(frozen=True)
class R5FaroGeometry:
    parent_id: str
    video_id: str
    physical_frame_id: str
    source_frame_receipt_sha256: str
    source_decision_sha256: str
    phase_a_completion_sha256: str
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


def _geometry_payload(geometry: R5FaroGeometry) -> dict[str, Any]:
    return {field: getattr(geometry, field) for field in geometry.__dataclass_fields__ if field != "content_sha256"}


def validate_faro_geometry(value: Any) -> R5FaroGeometry:
    require(isinstance(value, R5FaroGeometry), "R5_FARO_GEOMETRY_TYPE_INVALID", "R5 FARO geometry must come from controlled extractor")
    geometry = value
    for field in ("intrinsics", "depth_m", "valid_depth", "points_camera_xyz", "pixels_uv", "support_normal_camera_xyz"):
        require(not np.asarray(getattr(geometry, field)).flags.writeable, "R5_FARO_GEOMETRY_MUTABLE", "R5 FARO array is mutable", field=field)
    require(
        (geometry.parent_id, geometry.video_id) in R5_ROSTER
        and geometry.physical_frame_id.startswith(f"{geometry.video_id}:")
        and isinstance(geometry.max_source_timestamp_ns, int)
        and geometry.max_source_timestamp_ns >= 0,
        "R5_FARO_GEOMETRY_IDENTITY_DRIFT",
        "R5 FARO geometry identity drift",
    )
    for field in ("source_frame_receipt_sha256", "source_decision_sha256", "phase_a_completion_sha256", "highres_depth_array_sha256"):
        _hash(getattr(geometry, field), field)
    require(geometry.intrinsics.shape == (3, 3) and geometry.intrinsics.dtype == np.float64, "R5_FARO_GEOMETRY_ARRAY_DRIFT", "R5 FARO intrinsics drift")
    require(geometry.depth_m.shape == adapter.HIGHRES_SHAPE_HW and geometry.depth_m.dtype == np.float64, "R5_FARO_GEOMETRY_ARRAY_DRIFT", "R5 FARO depth shape/dtype drift")
    require(geometry.valid_depth.shape == adapter.HIGHRES_SHAPE_HW and geometry.valid_depth.dtype == np.bool_, "R5_FARO_GEOMETRY_ARRAY_DRIFT", "R5 FARO valid mask drift")
    require(geometry.points_camera_xyz.ndim == 2 and geometry.points_camera_xyz.shape[1] == 3 and geometry.points_camera_xyz.dtype == np.float64, "R5_FARO_GEOMETRY_ARRAY_DRIFT", "R5 FARO points drift")
    require(geometry.pixels_uv.shape == (len(geometry.points_camera_xyz), 2) and geometry.pixels_uv.dtype == np.int32, "R5_FARO_GEOMETRY_ARRAY_DRIFT", "R5 FARO pixels drift")
    require(geometry.support_normal_camera_xyz.shape == (3,) and geometry.support_normal_camera_xyz.dtype == np.float64 and abs(float(np.linalg.norm(geometry.support_normal_camera_xyz)) - 1.0) <= 1e-9, "R5_FARO_GEOMETRY_VALUE_DRIFT", "R5 FARO support normal drift")
    require(_finite(geometry.camera_height_m) and geometry.camera_height_m > 0.0 and isinstance(geometry.support_count, int) and geometry.support_count >= adapter.MINIMUM_SUPPORT_POINTS, "R5_FARO_GEOMETRY_VALUE_DRIFT", "R5 FARO support metrics drift")
    require(adapter.canonical_sha256(_geometry_payload(geometry)) == geometry.content_sha256, "R5_FARO_GEOMETRY_HASH_DRIFT", "R5 FARO geometry hash drift")
    return geometry


def derive_faro_geometry(
    highres_depth_mm: np.ndarray,
    source_frame_receipt: Mapping[str, Any],
    source_decision: Mapping[str, Any],
    phase_a_completion: Mapping[str, Any],
) -> R5FaroGeometry:
    source = validate_r5_source_receipt(source_frame_receipt)
    decision = validate_source_decision(source_decision)
    phase_a = validate_phase_a_completion(phase_a_completion)
    require(
        source["content_sha256"] == decision["source_frame_receipt_sha256"]
        and source["physical_frame_id"] == decision["physical_frame_id"],
        "R5_FARO_PHASE_A_LINEAGE_DRIFT",
        "R5 FARO source does not bind its Phase-A decision",
    )
    raw = np.asarray(highres_depth_mm)
    require(raw.shape == adapter.HIGHRES_SHAPE_HW and raw.dtype == np.uint16, "R5_FARO_DEPTH_INVALID", "FARO depth must be uint16 1440x1920")
    try:
        adapter._validate_bound_decoded_payload(source, "highres_depth", raw)
    except adapter.AdapterError as error:
        raise R5ConfirmationError(error.code, str(error), **error.context) from error
    matrix = adapter._intrinsics_matrix(source["intrinsics_highres"]["matrix_3x3"])
    gravity = adapter._normalize_vector(source["gravity_up_camera_xyz"], "GRAVITY_INVALID")
    depth_m = raw.astype(np.float64) / 1000.0
    valid = np.isfinite(depth_m) & (depth_m >= adapter.DEPTH_RANGE_M[0]) & (depth_m <= adapter.DEPTH_RANGE_M[1])
    sampled, _ = adapter._unproject(depth_m, valid, matrix, adapter.SUPPORT_POINT_STRIDE)
    try:
        plane = adapter._fit_support_plane(sampled, gravity)
    except adapter.AdapterError as error:
        raise R5ConfirmationError(error.code, str(error), **error.context) from error
    points, pixels = adapter._unproject(depth_m, valid, matrix, 1)
    values = {
        "parent_id": source["parent_id"],
        "video_id": source["session_id"],
        "physical_frame_id": source["physical_frame_id"],
        "source_frame_receipt_sha256": source["content_sha256"],
        "source_decision_sha256": decision["content_sha256"],
        "phase_a_completion_sha256": phase_a["content_sha256"],
        "highres_depth_array_sha256": adapter.canonical_sha256(raw),
        "max_source_timestamp_ns": int(source["max_source_timestamp_ns"]),
        "intrinsics": _immutable(matrix, np.float64),
        "depth_m": _immutable(depth_m, np.float64),
        "valid_depth": _immutable(valid, np.bool_),
        "points_camera_xyz": _immutable(points, np.float64),
        "pixels_uv": _immutable(pixels, np.int32),
        "support_normal_camera_xyz": _immutable(plane["normal_camera_xyz"], np.float64),
        "camera_height_m": float(plane["camera_height_m"]),
        "support_count": int(plane["support_count"]),
        "support_fraction": float(plane["support_fraction"]),
        "support_slope_degrees": float(plane["slope_degrees"]),
        "support_median_residual_m": float(plane["median_residual_m"]),
    }
    provisional = R5FaroGeometry(**values, content_sha256="0" * 64)
    return validate_faro_geometry(R5FaroGeometry(**values, content_sha256=adapter.canonical_sha256(_geometry_payload(provisional))))


def build_query_receipts(source_frame_receipt: Mapping[str, Any], geometry: R5FaroGeometry) -> list[dict[str, Any]]:
    source = validate_r5_source_receipt(source_frame_receipt)
    faro = validate_faro_geometry(geometry)
    require(faro.physical_frame_id == source["physical_frame_id"] and faro.source_frame_receipt_sha256 == source["content_sha256"], "R5_QUERY_GEOMETRY_BINDING_DRIFT", "R5 query geometry does not bind source")
    return adapter._build_geometry_query_receipts(
        physical_frame_id=faro.physical_frame_id,
        source_frame_receipt_sha256=faro.source_frame_receipt_sha256,
        max_source_timestamp_ns=faro.max_source_timestamp_ns,
        support_normal_camera_xyz=faro.support_normal_camera_xyz,
        camera_height_m=faro.camera_height_m,
    )


def _local_valid_fraction(geometry: R5FaroGeometry, receipt: dict[str, Any]) -> float:
    origin, _, lateral, heading = adapter._query_receipt_vectors(receipt)
    query_up = adapter._normalize_vector(receipt["virtual_query_frame"]["gravity_up_camera_xyz"], "QUERY_FRAME_INVALID")
    path_side = adapter._normalize_vector(np.cross(heading, query_up), "QUERY_FRAME_INVALID")
    path_origin = origin + float(receipt["path_lateral_offset_m"]) * lateral
    matrix = geometry.intrinsics
    valid_count = 0
    total_count = 0
    half_h, half_w = adapter.BOUNDARY_NEIGHBORHOOD_HW[0] // 2, adapter.BOUNDARY_NEIGHBORHOOD_HW[1] // 2
    for forward_m in np.linspace(adapter.MINIMUM_FORWARD_M, adapter.HORIZON_M, 10):
        for lateral_m in (-adapter.CAPSULE_RADIUS_M, 0.0, adapter.CAPSULE_RADIUS_M):
            point = path_origin + forward_m * heading + lateral_m * path_side
            if point[2] <= 1e-9:
                total_count += adapter.BOUNDARY_NEIGHBORHOOD_HW[0] * adapter.BOUNDARY_NEIGHBORHOOD_HW[1]
                continue
            column = int(round(float(matrix[0, 0] * point[0] / point[2] + matrix[0, 2])))
            row = int(round(float(matrix[1, 1] * point[1] / point[2] + matrix[1, 2])))
            for dr in range(-half_h, half_h + 1):
                for dc in range(-half_w, half_w + 1):
                    rr, cc = row + dr, column + dc
                    total_count += 1
                    if 0 <= rr < geometry.valid_depth.shape[0] and 0 <= cc < geometry.valid_depth.shape[1] and bool(geometry.valid_depth[rr, cc]):
                        valid_count += 1
    return valid_count / float(total_count) if total_count else 0.0


def build_query_truth_base(geometry: R5FaroGeometry, query_receipt: Mapping[str, Any]) -> source_factor.QueryTruthBase:
    faro = validate_faro_geometry(geometry)
    query = adapter._validate_query_receipt(dict(query_receipt))
    require(query["physical_frame_id"] == faro.physical_frame_id and query["source_frame_receipt_sha256"] == faro.source_frame_receipt_sha256, "R5_QUERY_IDENTITY_DRIFT", "query and R5 FARO geometry differ")
    return _build_query_truth_base_validated(faro, query)


def _build_query_truth_base_validated(faro: R5FaroGeometry, query: dict[str, Any]) -> source_factor.QueryTruthBase:
    origin, _, lateral, heading = adapter._query_receipt_vectors(query)
    query_up = adapter._normalize_vector(query["virtual_query_frame"]["gravity_up_camera_xyz"], "QUERY_FRAME_INVALID")
    path_side = adapter._normalize_vector(np.cross(heading, query_up), "QUERY_FRAME_INVALID")
    path_origin = origin + float(query["path_lateral_offset_m"]) * lateral
    points = faro.points_camera_xyz
    rel = points - path_origin
    along = rel @ heading
    across = rel @ path_side
    height = points @ faro.support_normal_camera_xyz + faro.camera_height_m
    keep = (along >= adapter.MINIMUM_FORWARD_M - 0.20) & (along <= adapter.HORIZON_M + 0.20) & (np.abs(across) <= adapter.HORIZON_M + adapter.CAPSULE_RADIUS_M) & (height >= -0.20) & (height <= 2.20)
    selected_points = np.ascontiguousarray(points[keep], dtype=np.float64)
    selected_pixels = np.ascontiguousarray(faro.pixels_uv[keep], dtype=np.int32)
    require(len(selected_points) > 0, "R5_QUERY_LOCAL_SURFACE_EMPTY", "R5 query has no FARO local surface")
    truth = adapter._query_support_and_boundary(selected_points, selected_pixels, faro.support_normal_camera_xyz, faro.camera_height_m, query)
    return source_factor.QueryTruthBase(
        physical_frame_id=faro.physical_frame_id,
        query_id=query["query_id"],
        query_receipt=query,
        common_point_ids_uv=_immutable(selected_pixels, np.int32),
        common_point_ids_sha256=adapter.canonical_sha256(selected_pixels),
        local_valid_fraction=_local_valid_fraction(faro, query),
        truth_normal_camera_xyz=_immutable(faro.support_normal_camera_xyz, np.float64),
        truth_camera_height_m=float(faro.camera_height_m),
        truth_query_support_points=int(truth["query_support_points"]),
        truth_observed_forward_m=None if truth["observed_forward_shape_m"] is None else float(truth["observed_forward_shape_m"]),
        truth_boundary_point_ids_uv=_immutable(truth["boundary_point_ids_uv"], np.int32),
        truth_boundary_points_camera_xyz=_immutable(truth["boundary_points_shape_camera_xyz"], np.float64),
    )


def _prepared_and_plane(candidate: Mapping[str, Any], native_depth_m: np.ndarray, decision: Mapping[str, Any]) -> tuple[source_factor.PreparedSourceCandidate, r3.DirectAppleSupportPlane | None]:
    candidate_record = validate_candidate_frame_record(candidate)
    choice = validate_source_decision(decision)
    native = np.ascontiguousarray(native_depth_m, dtype=np.float32)
    require(
        native.shape == depthart_runner.NATIVE_SHAPE_HW
        and adapter.canonical_sha256(native) == candidate_record["inference_receipt"]["native_output_array_sha256"],
        "R5_QUERY_NATIVE_DEPTH_DRIFT",
        "query candidate native depth differs from its sealed inference",
    )
    raw = depthart_runner.upsample_native_depth(native)
    raw_hash = adapter.canonical_sha256(raw)
    require(
        raw_hash == choice["candidate_highres_depth_array_sha256"]
        == candidate_record["inference_receipt"]["highres_output_array_sha256"],
        "R5_QUERY_CANDIDATE_DRIFT",
        "query candidate differs from Phase A",
    )
    scale = choice["scale_record"]
    if scale["evaluable"]:
        anchored = np.ascontiguousarray(raw.astype(np.float64) * float(scale["metric_scale"]), dtype=np.float64)
    else:
        anchored = np.ascontiguousarray(raw, dtype=np.float64)
    anchored_hash = adapter.canonical_sha256(anchored)
    if choice["anchored_candidate_depth_array_sha256"] is not None:
        require(anchored_hash == choice["anchored_candidate_depth_array_sha256"], "R5_QUERY_ANCHORED_DEPTH_DRIFT", "anchored query candidate differs from Phase A")
    prepared = source_factor.PreparedSourceCandidate(
        parent_id=choice["parent_id"], physical_frame_id=choice["physical_frame_id"],
        raw_depth_m=_immutable(raw, raw.dtype), anchored_depth_m=_immutable(anchored, np.float64),
        raw_depth_sha256=raw_hash, anchored_depth_sha256=anchored_hash,
        metric_scale=float(scale["metric_scale"]) if scale["evaluable"] else 1.0,
        source_scale_record_sha256=scale["content_sha256"],
        candidate_binding_sha256=candidate_record["content_sha256"],
        apple_source_receipt_sha256=choice["source_frame_receipt_sha256"],
        reliability={},
    )
    if not choice["source_support_available"]:
        return prepared, None
    block = choice["direct_support_plane"]
    matrix_value = candidate_record["candidate_input_receipt"]["intrinsics_highres"]["matrix_3x3"]
    matrix = adapter._intrinsics_matrix(matrix_value)
    legacy_list_hash = adapter.canonical_sha256(matrix_value)
    normalized_array_hash = adapter.canonical_sha256(matrix)
    require(
        choice["intrinsics_highres_sha256"] in {legacy_list_hash, normalized_array_hash},
        "R5_QUERY_CAMERA_BINDING_DRIFT",
        "Phase-A intrinsics binding matches neither the legacy list nor normalized array representation",
    )
    plane = r3.DirectAppleSupportPlane(
        parent_id=choice["parent_id"], physical_frame_id=choice["physical_frame_id"],
        direct_source_receipt_sha256=choice["source_frame_receipt_sha256"],
        source_scale_record_sha256=scale["content_sha256"],
        candidate_binding_sha256=candidate_record["content_sha256"],
        anchored_depth_array_sha256=anchored_hash,
        # The consumed R5 Phase-A decisions sealed the numerically identical K
        # as a JSON list, while the extractor hashes normalized float64 arrays.
        # Accept the sealed legacy hash above, then bridge to the extractor's
        # representation without changing any K value or branch decision.
        intrinsics_highres_sha256=normalized_array_hash,
        gravity_up_camera_xyz_sha256=choice["gravity_up_camera_xyz_sha256"],
        normal_camera_xyz=_immutable(block["normal_camera_xyz"], np.float64),
        camera_height_m=float(block["camera_height_m"]), support_count=int(block["support_count"]),
        support_fraction=float(block["support_fraction"]), slope_degrees=float(block["slope_degrees"]),
        median_residual_m=float(block["median_residual_m"]), record=choice, content_sha256=choice["content_sha256"],
    )
    return prepared, plane


def _extract_baseline(prepared: source_factor.PreparedSourceCandidate, matrix: Any, gravity: Any, base: source_factor.QueryTruthBase) -> dict[str, Any]:
    try:
        extraction = source_factor._extract_query(prepared.raw_depth_m, prepared.raw_depth_sha256, adapter._intrinsics_matrix(matrix), adapter._normalize_vector(gravity, "GRAVITY_INVALID"), base)
        return source_factor._mode_result(base, extraction)
    except source_factor.SourceFactorError as error:
        return source_factor._failed_mode(prepared.raw_depth_sha256, error.code)


def _extract_direct(prepared: source_factor.PreparedSourceCandidate, plane: r3.DirectAppleSupportPlane | None, matrix: Any, gravity: Any, base: source_factor.QueryTruthBase, failure_code: str | None) -> dict[str, Any]:
    if plane is None:
        return source_factor._failed_mode(prepared.anchored_depth_sha256, str(failure_code))
    try:
        return source_factor._mode_result(base, r3._posthoc_extraction(prepared, matrix, gravity, base, plane))
    except r3.DirectAppleSupportError as error:
        return source_factor._failed_mode(prepared.anchored_depth_sha256, error.code)


def _validate_mode_result(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R5_QUERY_MODE_INVALID", "query factor mode must be an object", field=field)
    mode = copy.deepcopy(value)
    require(
        set(mode) == {"extraction_evaluable", "reason_codes", "depth_array_sha256", "valid_common_point_count", "support", "boundary", "query_point_clearance"},
        "R5_QUERY_MODE_INVALID",
        "query factor mode fields drift",
        field=field,
    )
    _hash(mode["depth_array_sha256"], f"{field}.depth_array_sha256")
    require(isinstance(mode["extraction_evaluable"], bool) and isinstance(mode["reason_codes"], list), "R5_QUERY_MODE_INVALID", "query extraction metadata drift", field=field)
    support = mode["support"]
    boundary = mode["boundary"]
    query = mode["query_point_clearance"]
    require(isinstance(support, dict) and set(support) == {"evaluable", "reason_codes", "normal_angular_error_rad", "height_abs_error_m", "camera_height_m", "support_point_count", "support_fraction", "slope_degrees"}, "R5_QUERY_MODE_INVALID", "support result fields drift", field=field)
    require(isinstance(boundary, dict) and set(boundary) == {"evaluable", "reason_codes", "truth_point_count", "candidate_point_count", "point_id_intersection_count", "point_id_union_count", "point_id_jaccard", "xyz_median_error_m", "local_valid_fraction"}, "R5_QUERY_MODE_INVALID", "boundary result fields drift", field=field)
    require(isinstance(query, dict) and set(query) == {"evaluable", "reason_codes", "value_m", "truth_value_m", "abs_error_m", "query_support_points", "observed_forward_m", "local_valid_fraction"}, "R5_QUERY_MODE_INVALID", "query result fields drift", field=field)
    for name, block in (("support", support), ("boundary", boundary), ("query", query)):
        require(isinstance(block["evaluable"], bool) and isinstance(block["reason_codes"], list), "R5_QUERY_MODE_INVALID", "factor evaluability metadata drift", field=f"{field}.{name}")
        require((block["evaluable"] and block["reason_codes"] == []) or (not block["evaluable"] and bool(block["reason_codes"])), "R5_QUERY_MODE_INVALID", "factor evaluability/reasons disagree", field=f"{field}.{name}")
    if mode["extraction_evaluable"]:
        require(mode["reason_codes"] == [] and isinstance(mode["valid_common_point_count"], int) and mode["valid_common_point_count"] >= adapter.MINIMUM_SUPPORT_POINTS and support["evaluable"], "R5_QUERY_MODE_INVALID", "evaluable extraction metadata drift", field=field)
        for metric in ("normal_angular_error_rad", "height_abs_error_m", "camera_height_m", "support_fraction", "slope_degrees"):
            require(_finite(support[metric]), "R5_QUERY_MODE_INVALID", "support metric is not finite", field=f"{field}.{metric}")
        require(isinstance(support["support_point_count"], int) and support["support_point_count"] >= adapter.MINIMUM_SUPPORT_POINTS, "R5_QUERY_MODE_INVALID", "support point count drift", field=field)
    else:
        require(bool(mode["reason_codes"]) and mode["valid_common_point_count"] is None and not support["evaluable"] and not boundary["evaluable"] and not query["evaluable"], "R5_QUERY_MODE_INVALID", "failed extraction carries evaluable factor", field=field)
    for metric in ("point_id_jaccard", "xyz_median_error_m", "local_valid_fraction"):
        require(boundary[metric] is None or _finite(boundary[metric]), "R5_QUERY_MODE_INVALID", "boundary metric is not finite", field=f"{field}.{metric}")
    for metric in ("value_m", "truth_value_m", "abs_error_m", "observed_forward_m", "local_valid_fraction"):
        require(query[metric] is None or _finite(query[metric]), "R5_QUERY_MODE_INVALID", "query metric is not finite", field=f"{field}.{metric}")
    return mode


def _effects(baseline: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "extraction_recovered_vs_baseline": not baseline["extraction_evaluable"] and selected["extraction_evaluable"],
        "extraction_lost_vs_baseline": baseline["extraction_evaluable"] and not selected["extraction_evaluable"],
        "height_error_reduction_vs_baseline_m": source_factor._difference(baseline, selected, "support", "height_abs_error_m"),
        "normal_error_reduction_vs_baseline_rad": source_factor._difference(baseline, selected, "support", "normal_angular_error_rad"),
        "boundary_evaluability_recovered_vs_baseline": not baseline["boundary"]["evaluable"] and selected["boundary"]["evaluable"],
        "boundary_evaluability_lost_vs_baseline": baseline["boundary"]["evaluable"] and not selected["boundary"]["evaluable"],
        "query_knownness_recovered_vs_baseline": not baseline["query_point_clearance"]["evaluable"] and selected["query_point_clearance"]["evaluable"],
        "query_knownness_lost_vs_baseline": baseline["query_point_clearance"]["evaluable"] and not selected["query_point_clearance"]["evaluable"],
    }


def evaluate_query(
    source_frame_receipt: Mapping[str, Any],
    candidate_frame_record: Mapping[str, Any],
    native_depth_m: np.ndarray,
    source_decision: Mapping[str, Any],
    geometry: R5FaroGeometry,
    query_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    source = validate_r5_source_receipt(source_frame_receipt)
    candidate = validate_candidate_frame_record(candidate_frame_record)
    decision = validate_source_decision(source_decision)
    faro = validate_faro_geometry(geometry)
    query = adapter._validate_query_receipt(dict(query_receipt))
    return _evaluate_query_validated(source, candidate, native_depth_m, decision, faro, query)


def _evaluate_query_validated(
    source: dict[str, Any],
    candidate: dict[str, Any],
    native_depth_m: np.ndarray,
    decision: dict[str, Any],
    faro: R5FaroGeometry,
    query: dict[str, Any],
    *,
    prepared_and_plane: tuple[source_factor.PreparedSourceCandidate, r3.DirectAppleSupportPlane | None] | None = None,
) -> dict[str, Any]:
    require(
        source["content_sha256"] == decision["source_frame_receipt_sha256"] == faro.source_frame_receipt_sha256 == query["source_frame_receipt_sha256"]
        and candidate["content_sha256"] == decision["candidate_frame_record_sha256"]
        and faro.source_decision_sha256 == decision["content_sha256"]
        and source["physical_frame_id"] == decision["physical_frame_id"] == faro.physical_frame_id == query["physical_frame_id"],
        "R5_QUERY_LINEAGE_DRIFT",
        "query inputs do not share Phase-A lineage",
    )
    prepared, plane = prepared_and_plane or _prepared_and_plane(candidate, native_depth_m, decision)
    matrix = source["intrinsics_highres"]["matrix_3x3"]
    gravity = source["gravity_up_camera_xyz"]
    try:
        base = _build_query_truth_base_validated(faro, query)
    except (R5ConfirmationError, adapter.AdapterError) as error:
        code = str(getattr(error, "code", type(error).__name__))
        baseline = source_factor._failed_mode(prepared.raw_depth_sha256, code)
        direct = source_factor._failed_mode(prepared.anchored_depth_sha256, code)
        common_point_ids_sha256 = adapter.canonical_sha256(np.empty((0, 2), dtype=np.int32))
    else:
        baseline = _extract_baseline(prepared, matrix, gravity, base)
        direct = _extract_direct(prepared, plane, matrix, gravity, base, decision["source_failure_code"])
        common_point_ids_sha256 = base.common_point_ids_sha256
    selected = direct if decision["selected_branch"] == "DIRECT_APPLE_SUPPORT" else baseline
    effects = _effects(baseline, selected)
    record = _seal(
        {
            "schema": QUERY_RECORD_SCHEMA,
            "r5_role": R5_ROLE,
            "policy_id": POLICY_ID,
            "parent_id": decision["parent_id"],
            "physical_frame_id": decision["physical_frame_id"],
            "query_id": query["query_id"],
            "grid_index": query["grid_index"],
            "source_frame_receipt_sha256": source["content_sha256"],
            "candidate_frame_record_sha256": candidate["content_sha256"],
            "source_decision_sha256": decision["content_sha256"],
            "phase_a_completion_sha256": faro.phase_a_completion_sha256,
            "faro_geometry_sha256": faro.content_sha256,
            "query_receipt_sha256": query["content_sha256"],
            "common_point_ids_sha256": common_point_ids_sha256,
            "phase_a_selected_branch": decision["selected_branch"],
            "source_support_available": decision["source_support_available"],
            "baseline": baseline,
            "direct_apple_support": direct,
            "selected_hybrid": selected,
            "effects": effects,
            "branch_reselection_after_truth": False,
            "faro_used_for_scoring_only": True,
        }
    )
    return validate_query_record(record, source_decision=decision)


def evaluate_frame(
    source_frame_receipt: Mapping[str, Any],
    candidate_frame_record: Mapping[str, Any],
    native_depth_m: np.ndarray,
    source_decision: Mapping[str, Any],
    geometry: R5FaroGeometry,
) -> list[dict[str, Any]]:
    """Evaluate exactly nine queries while hashing the large FARO geometry once."""

    source = validate_r5_source_receipt(source_frame_receipt)
    candidate = validate_candidate_frame_record(candidate_frame_record)
    decision = validate_source_decision(source_decision)
    faro = validate_faro_geometry(geometry)
    require(faro.physical_frame_id == source["physical_frame_id"] and faro.source_frame_receipt_sha256 == source["content_sha256"], "R5_QUERY_GEOMETRY_BINDING_DRIFT", "R5 query geometry does not bind source")
    queries = [
        adapter._validate_query_receipt(row)
        for row in adapter._build_geometry_query_receipts(
            physical_frame_id=faro.physical_frame_id,
            source_frame_receipt_sha256=faro.source_frame_receipt_sha256,
            max_source_timestamp_ns=faro.max_source_timestamp_ns,
            support_normal_camera_xyz=faro.support_normal_camera_xyz,
            camera_height_m=faro.camera_height_m,
        )
    ]
    require([row["grid_index"] for row in queries] == list(range(9)), "R5_QUERY_GRID_DRIFT", "R5 frame query grid drift")
    prepared = _prepared_and_plane(candidate, native_depth_m, decision)
    return [
        _evaluate_query_validated(source, candidate, native_depth_m, decision, faro, query, prepared_and_plane=prepared)
        for query in queries
    ]


def evaluate_unobservable_faro_frame(
    source_frame_receipt: Mapping[str, Any],
    candidate_frame_record: Mapping[str, Any],
    native_depth_m: np.ndarray,
    source_decision: Mapping[str, Any],
    phase_a_completion: Mapping[str, Any],
    highres_depth_mm: np.ndarray,
    failure_code: str,
) -> list[dict[str, Any]]:
    """Retain nine UNKNOWN slots when FARO cannot identify a support plane.

    This is deliberately narrower than a generic exception fallback.  The
    bound FARO payload must reproduce the exact support-unobservable error;
    integrity, lineage, decode, and all other geometry failures still abort.
    """

    source = validate_r5_source_receipt(source_frame_receipt)
    candidate = validate_candidate_frame_record(candidate_frame_record)
    decision = validate_source_decision(source_decision)
    phase_a = validate_phase_a_completion(phase_a_completion)
    require(
        isinstance(failure_code, str) and failure_code in adapter._SUPPORT_UNOBSERVABLE_CODES,
        "R5_FARO_UNKNOWN_CODE_INVALID",
        "only frozen support-unobservable FARO failures can become UNKNOWN query slots",
        failure_code=failure_code,
    )
    try:
        derive_faro_geometry(highres_depth_mm, source, decision, phase_a)
    except R5ConfirmationError as error:
        require(
            error.code == failure_code,
            "R5_FARO_UNKNOWN_REPRODUCTION_DRIFT",
            "bound FARO payload did not reproduce the claimed support-unobservable failure",
            expected=failure_code,
            observed=error.code,
        )
    else:
        raise R5ConfirmationError(
            "R5_FARO_UNKNOWN_REPRODUCTION_DRIFT",
            "bound FARO payload unexpectedly produced evaluable support geometry",
            expected=failure_code,
        )
    require(
        source["content_sha256"] == decision["source_frame_receipt_sha256"]
        and candidate["content_sha256"] == decision["candidate_frame_record_sha256"]
        and source["physical_frame_id"] == decision["physical_frame_id"],
        "R5_QUERY_LINEAGE_DRIFT",
        "unobservable FARO query inputs do not share Phase-A lineage",
    )
    raw = np.asarray(highres_depth_mm)
    faro_observation = _seal(
        {
            "schema": FARO_UNOBSERVABLE_SCHEMA,
            "parent_id": source["parent_id"],
            "physical_frame_id": source["physical_frame_id"],
            "source_frame_receipt_sha256": source["content_sha256"],
            "source_decision_sha256": decision["content_sha256"],
            "phase_a_completion_sha256": phase_a["content_sha256"],
            "highres_depth_array_sha256": adapter.canonical_sha256(raw),
            "geometry_evaluable": False,
            "reason_codes": [failure_code],
            "retained_as_unknown": True,
        }
    )
    prepared, _ = _prepared_and_plane(candidate, native_depth_m, decision)
    empty_ids_sha256 = adapter.canonical_sha256(np.empty((0, 2), dtype=np.int32))
    records: list[dict[str, Any]] = []
    for grid_index in range(9):
        lateral = adapter.PATH_LATERAL_OFFSETS_M[grid_index // 3]
        yaw = adapter.PATH_YAW_DEGREES[grid_index % 3]
        path_id = f"lat_{adapter._signed_token(lateral, 2)}_yaw_{adapter._signed_token(yaw, 1)}"
        query_id = f"{source['physical_frame_id']}:{path_id}"
        query_slot = _seal(
            {
                "schema": QUERY_UNOBSERVABLE_SLOT_SCHEMA,
                "source_frame_receipt_sha256": source["content_sha256"],
                "physical_frame_id": source["physical_frame_id"],
                "query_id": query_id,
                "path_id": path_id,
                "grid_index": grid_index,
                "grid_order": "LATERAL_MAJOR_THEN_YAW_ASCENDING",
                "faro_unobservable_sha256": faro_observation["content_sha256"],
                "reason_codes": [failure_code],
                "query_geometry_instantiation_absent": True,
            }
        )
        baseline = source_factor._failed_mode(prepared.raw_depth_sha256, failure_code)
        direct = source_factor._failed_mode(prepared.anchored_depth_sha256, failure_code)
        selected = direct if decision["selected_branch"] == "DIRECT_APPLE_SUPPORT" else baseline
        record = _seal(
            {
                "schema": QUERY_RECORD_SCHEMA,
                "r5_role": R5_ROLE,
                "policy_id": POLICY_ID,
                "parent_id": decision["parent_id"],
                "physical_frame_id": decision["physical_frame_id"],
                "query_id": query_id,
                "grid_index": grid_index,
                "source_frame_receipt_sha256": source["content_sha256"],
                "candidate_frame_record_sha256": candidate["content_sha256"],
                "source_decision_sha256": decision["content_sha256"],
                "phase_a_completion_sha256": phase_a["content_sha256"],
                # Legacy field name retained for schema compatibility.  The
                # hash binds the explicit non-geometry FARO observation above.
                "faro_geometry_sha256": faro_observation["content_sha256"],
                "query_receipt_sha256": query_slot["content_sha256"],
                "common_point_ids_sha256": empty_ids_sha256,
                "phase_a_selected_branch": decision["selected_branch"],
                "source_support_available": decision["source_support_available"],
                "baseline": baseline,
                "direct_apple_support": direct,
                "selected_hybrid": selected,
                "effects": _effects(baseline, selected),
                "branch_reselection_after_truth": False,
                "faro_used_for_scoring_only": True,
            }
        )
        records.append(validate_query_record(record, source_decision=decision))
    require([row["grid_index"] for row in records] == list(range(9)), "R5_QUERY_GRID_DRIFT", "unobservable FARO frame did not retain nine query slots")
    return records


def validate_query_record(value: Any, *, source_decision: Mapping[str, Any] | None = None) -> dict[str, Any]:
    record = _validate_seal(value, QUERY_RECORD_SCHEMA)
    expected = {"schema", "r5_role", "policy_id", "parent_id", "physical_frame_id", "query_id", "grid_index", "source_frame_receipt_sha256", "candidate_frame_record_sha256", "source_decision_sha256", "phase_a_completion_sha256", "faro_geometry_sha256", "query_receipt_sha256", "common_point_ids_sha256", "phase_a_selected_branch", "source_support_available", "baseline", "direct_apple_support", "selected_hybrid", "effects", "branch_reselection_after_truth", "faro_used_for_scoring_only", "content_sha256"}
    require(set(record) == expected and record["r5_role"] == R5_ROLE and record["policy_id"] == POLICY_ID, "R5_QUERY_RECORD_KEY_SET", "query record fields/method drift")
    require(
        isinstance(record["physical_frame_id"], str)
        and (record["parent_id"], record["physical_frame_id"].split(":", 1)[0]) in R5_ROSTER
        and isinstance(record["query_id"], str)
        and record["query_id"].startswith(f"{record['physical_frame_id']}:")
        and isinstance(record["grid_index"], int)
        and 0 <= record["grid_index"] < 9
        and isinstance(record["source_support_available"], bool),
        "R5_QUERY_RECORD_IDENTITY_DRIFT",
        "query record identity/grid drift",
    )
    for field in ("source_frame_receipt_sha256", "candidate_frame_record_sha256", "source_decision_sha256", "phase_a_completion_sha256", "faro_geometry_sha256", "query_receipt_sha256", "common_point_ids_sha256"):
        _hash(record[field], field)
    expected_branch = "DIRECT_APPLE_SUPPORT" if record["source_support_available"] else "R1_BASELINE"
    require(record["phase_a_selected_branch"] == expected_branch, "R5_QUERY_BRANCH_DRIFT", "query branch differs from source-only policy")
    baseline = _validate_mode_result(record["baseline"], "baseline")
    direct = _validate_mode_result(record["direct_apple_support"], "direct_apple_support")
    selected_mode = _validate_mode_result(record["selected_hybrid"], "selected_hybrid")
    expected_selected = direct if expected_branch == "DIRECT_APPLE_SUPPORT" else baseline
    require(adapter.canonical_sha256(selected_mode) == adapter.canonical_sha256(expected_selected), "R5_QUERY_OUTCOME_RESELECTION", "selected hybrid differs from Phase-A branch")
    require(record["effects"] == _effects(baseline, selected_mode), "R5_QUERY_EFFECT_DRIFT", "query effects do not rederive")
    require(record["branch_reselection_after_truth"] is False and record["faro_used_for_scoring_only"] is True, "R5_QUERY_FIREWALL_DRIFT", "query record claims outcome reselection")
    if source_decision is not None:
        decision = validate_source_decision(source_decision)
        require(record["source_decision_sha256"] == decision["content_sha256"] and record["phase_a_selected_branch"] == decision["selected_branch"], "R5_QUERY_DECISION_BINDING_DRIFT", "query does not bind external Phase-A decision")
    return record


def _parent_macro(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    by_frame: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row["effects"].get(field)
        if _finite(value):
            by_frame[(row["parent_id"], row["physical_frame_id"])].append(float(value))
    by_parent: dict[str, list[float]] = defaultdict(list)
    for (parent, _), values in by_frame.items():
        by_parent[parent].append(float(np.median(np.asarray(values, dtype=np.float64))))
    parent_values = []
    for parent in sorted({row["parent_id"] for row in rows}):
        values = by_parent.get(parent, [])
        parent_values.append({
            "parent_id": parent,
            "paired_frame_count": len(values),
            "median_frame_effect": float(np.median(np.asarray(values, dtype=np.float64))) if values else None,
        })
    usable = [row["median_frame_effect"] for row in parent_values if row["median_frame_effect"] is not None]
    return {"parents_with_metric": len(usable), "median_of_parent_medians": float(np.median(np.asarray(usable, dtype=np.float64))) if usable else None, "parent_values": parent_values}


def summarize(records: Sequence[Mapping[str, Any]], *, expected_parents: int = len(R5_ROSTER), expected_frames: int = EXPECTED_FRAME_COUNT, expected_queries: int = EXPECTED_QUERY_COUNT) -> dict[str, Any]:
    rows = [validate_query_record(dict(row)) for row in records]
    keys = {(row["physical_frame_id"], row["query_id"]) for row in rows}
    frames = {row["physical_frame_id"] for row in rows}
    parents = {row["parent_id"] for row in rows}
    require(len(rows) == expected_queries and len(keys) == expected_queries and len(frames) == expected_frames and len(parents) == expected_parents, "R5_SUMMARY_COHORT_DRIFT", "query cohort cardinality drift")
    require(parents == {parent for parent, _ in R5_ROSTER} and len({row["phase_a_completion_sha256"] for row in rows}) == 1, "R5_SUMMARY_LINEAGE_DRIFT", "summary mixes roster or Phase-A completions")
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[row["physical_frame_id"]].append(row)
    require(all(sorted(item["grid_index"] for item in frame_rows) == list(range(9)) for frame_rows in by_frame.values()), "R5_SUMMARY_QUERY_GRID_DRIFT", "every frame must retain exactly nine ordered query slots")
    observed_parent_counts = Counter(row["parent_id"] for row in rows)
    require(
        all(observed_parent_counts[parent] == count * 9 for (parent, _), count in zip(R5_ROSTER, EXPECTED_PARENT_FRAME_COUNTS)),
        "R5_SUMMARY_PARENT_COUNT_DRIFT",
        "summary parent/query counts differ from the frozen cohort",
    )
    height = _parent_macro(rows, "height_error_reduction_vs_baseline_m")
    normal = _parent_macro(rows, "normal_error_reduction_vs_baseline_rad")
    height_by_parent = {row["parent_id"]: row["median_frame_effect"] for row in height["parent_values"]}
    normal_by_parent = {row["parent_id"]: row["median_frame_effect"] for row in normal["parent_values"]}
    jointly_positive = sum(_finite(height_by_parent[parent]) and float(height_by_parent[parent]) > 0.0 and _finite(normal_by_parent[parent]) and float(normal_by_parent[parent]) > 0.0 for parent in parents)
    baseline_extraction = sum(bool(row["baseline"]["extraction_evaluable"]) for row in rows)
    hybrid_extraction = sum(bool(row["selected_hybrid"]["extraction_evaluable"]) for row in rows)
    baseline_known = sum(bool(row["baseline"]["query_point_clearance"]["evaluable"]) for row in rows)
    hybrid_known = sum(bool(row["selected_hybrid"]["query_point_clearance"]["evaluable"]) for row in rows)
    denominator_defined = height["parents_with_metric"] == normal["parents_with_metric"] == expected_parents
    gates = [
        {"id": "EXACT_COHORT_AND_LINEAGE", "passed": True},
        {"id": "PHASE_FIREWALL", "passed": True},
        {"id": "PARENT_METRIC_DENOMINATORS", "passed": denominator_defined},
        {"id": "HEIGHT_PARENT_MACRO_POSITIVE", "passed": denominator_defined and float(height["median_of_parent_medians"]) > 0.0},
        {"id": "NORMAL_PARENT_MACRO_POSITIVE", "passed": denominator_defined and float(normal["median_of_parent_medians"]) > 0.0},
        {"id": "ALL_PARENTS_JOINTLY_POSITIVE", "passed": jointly_positive == expected_parents},
        {"id": "EXTRACTION_COVERAGE_NO_REGRET", "passed": hybrid_extraction >= baseline_extraction},
        {"id": "QUERY_KNOWN_COVERAGE_NO_REGRET", "passed": hybrid_known >= baseline_known},
    ]
    if not denominator_defined:
        terminal = "TARO_O0R_DIRECT_APPLE_HYBRID_R5_NOT_EVALUABLE"
    elif all(gate["passed"] for gate in gates):
        terminal = "TARO_O0R_DIRECT_APPLE_HYBRID_R5_TASK_METRIC_CONFIRMATION_PASS"
    else:
        terminal = "TARO_O0R_DIRECT_APPLE_HYBRID_R5_TASK_METRIC_CONFIRMATION_FAIL"
    return _seal(
        {
            "schema": SUMMARY_SCHEMA,
            "claim_ceiling": CLAIM_CEILING,
            "policy_id": POLICY_ID,
            "parent_count": expected_parents,
            "physical_frame_count": expected_frames,
            "query_record_count": expected_queries,
            "direct_selected_frame_count": len({row["physical_frame_id"] for row in rows if row["phase_a_selected_branch"] == "DIRECT_APPLE_SUPPORT"}),
            "baseline_fallback_frame_count": len({row["physical_frame_id"] for row in rows if row["phase_a_selected_branch"] == "R1_BASELINE"}),
            "baseline_extraction_evaluable_query_count": baseline_extraction,
            "hybrid_extraction_evaluable_query_count": hybrid_extraction,
            "baseline_query_known_count": baseline_known,
            "hybrid_query_known_count": hybrid_known,
            "height_error_reduction_vs_baseline_parent_macro_m": height,
            "normal_error_reduction_vs_baseline_parent_macro_rad": normal,
            "parents_jointly_positive_height_and_normal": jointly_positive,
            "effects_counts": {key: sum(bool(row["effects"][key]) for row in rows) for key in ("extraction_recovered_vs_baseline", "extraction_lost_vs_baseline", "boundary_evaluability_recovered_vs_baseline", "boundary_evaluability_lost_vs_baseline", "query_knownness_recovered_vs_baseline", "query_knownness_lost_vs_baseline")},
            "gates": gates,
            "terminal": terminal,
            "training_steps": 0,
            "threshold_count": 0,
        }
    )


__all__ = [
    "CANDIDATE_COMPLETION_SCHEMA", "CANDIDATE_FRAME_SCHEMA", "CANDIDATE_INPUT_SCHEMA",
    "CLAIM_CEILING", "EXPECTED_FRAME_COUNT", "EXPECTED_PARENT_FRAME_COUNTS", "EXPECTED_QUERY_COUNT",
    "FARO_UNOBSERVABLE_SCHEMA", "INFERENCE_RECEIPT_SCHEMA", "PHASE_A_COMPLETION_SCHEMA",
    "POLICY_ID", "QUERY_RECORD_SCHEMA", "R5ConfirmationError", "R5FaroGeometry", "R5_ROLE", "R5_ROSTER",
    "SOURCE_DECISION_SCHEMA", "SUMMARY_SCHEMA", "build_candidate_input", "build_candidate_frame_record",
    "build_candidate_phase_completion", "build_inference_receipt", "build_phase_a_completion",
    "build_query_receipts", "build_query_truth_base", "build_source_decision", "derive_faro_geometry",
    "evaluate_frame", "evaluate_query", "evaluate_unobservable_faro_frame", "infer_candidate", "summarize", "validate_candidate_frame_record",
    "validate_candidate_input", "validate_candidate_phase_completion", "validate_faro_geometry",
    "validate_inference_receipt", "validate_phase_a_completion", "validate_query_record",
    "validate_r5_source_receipt", "validate_source_decision", "validate_source_scale_record",
]
