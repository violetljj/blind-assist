#!/usr/bin/env python3
"""Truth-blind AppleDepth metric anchor for sealed TARO candidates.

The estimator in this module has no FARO/truth input.  A caller must seal all
source-scale records before joining them to an oracle in a separate phase.
"""

from __future__ import annotations

import copy
import io
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase import validate_candidate_frame_record
from scripts.research.taro_o0r_factor_headroom_runtime.depthart_runner import (
    NATIVE_SHAPE_HW,
    upsample_native_depth,
    validate_depthart_inference_receipt,
)
from scripts.research.taro_o0r_factor_headroom_runtime.factor_canary import validate_factor_canary_record
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


APPLE_SCALE_SOURCE_RECEIPT_SCHEMA = "blindassist.taro.o0r.apple_scale_source_receipt.v1"
CANDIDATE_REPLAY_BINDING_SCHEMA = "blindassist.taro.o0r.candidate_scale_replay_binding.v1"
SOURCE_SCALE_RECORD_SCHEMA = "blindassist.taro.o0r.apple_metric_scale_record.v1"
ORACLE_COMPARISON_SCHEMA = "blindassist.taro.o0r.apple_scale_oracle_comparison.v1"
SOURCE_SCALE_SUMMARY_SCHEMA = "blindassist.taro.o0r.apple_scale_canary_summary.v1"

ESTIMATOR_ID = "MEDIAN_LOG_APPLEDEPTH_OVER_CANDIDATE_AT_REGISTERED_APPLE_CENTERS_CONFIDENCE_EQ_2_V1"
REGISTRATION_ID = "APPLE_CENTER_TO_HIGHRES_ROUND_HALF_TO_EVEN_SCALE_7P5_V1"
MEDIAN_RULE = "NUMPY_FLOAT64_MEDIAN_SORTED_MIDDLE_OR_MEAN_OF_TWO_MIDDLE_V1"
SOURCE_ROLES = ("lowres_depth", "confidence")
DEPTH_RANGE_M = tuple(float(value) for value in adapter.DEPTH_RANGE_M)
MINIMUM_PAIR_COUNT = int(adapter.MINIMUM_SUPPORT_POINTS)
CLAIM_CEILING = {
    "scope": "LOCKED_ARKITSCENES_TRAIN_LANDSCAPE_ONLY",
    "use": "POST_HOC_DESCRIPTIVE_TRUTH_BLIND_SCALE_CANARY",
    "threshold_or_pass_fail_decision": False,
    "excluded_claims": ["FORMAL_O0R_PASS", "FINAL_TASK_EFFECTIVENESS", "DEPLOYMENT", "PRODUCT", "SAFETY"],
}

_SHA256 = re.compile(r"^[0-9A-F]{64}$")


class AppleScaleError(RuntimeError):
    """Stable fail-closed error for the source-only scale canary."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def _require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise AppleScaleError(code, message, **context)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    _require("content_sha256" not in result, "SEAL_FIELD_COLLISION", "payload already contains a seal")
    result["content_sha256"] = adapter.canonical_sha256(result)
    return result


def _validate_seal(value: Any, schema: str, code: str) -> dict[str, Any]:
    _require(isinstance(value, dict), code, "sealed value must be an object")
    result = copy.deepcopy(value)
    observed = result.pop("content_sha256", None)
    _require(isinstance(observed, str) and bool(_SHA256.fullmatch(observed)) and adapter.canonical_sha256(result) == observed, code, "sealed value hash drift")
    result["content_sha256"] = observed
    _require(result.get("schema") == schema, code, "sealed value schema drift")
    return result


def _hash(value: Any, *, field: str) -> str:
    _require(isinstance(value, str) and bool(_SHA256.fullmatch(value)), "HASH_INVALID", "SHA-256 binding is malformed", field=field)
    return value


def sample_candidate_at_apple_centers(candidate_highres_depth_m: np.ndarray) -> np.ndarray:
    """Sample a registered 1440x1920 raster at frozen 192x256 Apple centers."""

    depth = np.asarray(candidate_highres_depth_m)
    _require(depth.shape == adapter.HIGHRES_SHAPE_HW and depth.dtype.kind == "f" and bool(np.all(np.isfinite(depth))), "CANDIDATE_DEPTH_INVALID", "candidate depth must be finite 1440x1920 floating-point metres")
    rows, columns = np.mgrid[0 : adapter.APPLE_SHAPE_HW[0], 0 : adapter.APPLE_SHAPE_HW[1]]
    scale_x, scale_y = adapter.LOWRES_TO_HIGHRES_SCALE_XY
    x = np.rint((columns + 0.5) * scale_x - 0.5).astype(np.int64)
    y = np.rint((rows + 0.5) * scale_y - 0.5).astype(np.int64)
    _require(int(x.min()) >= 0 and int(x.max()) < adapter.HIGHRES_SHAPE_HW[1] and int(y.min()) >= 0 and int(y.max()) < adapter.HIGHRES_SHAPE_HW[0], "REGISTRATION_INVALID", "Apple-center mapping leaves the registered raster")
    return np.ascontiguousarray(depth[y, x], dtype=np.float64)


def _decode_png(payload: bytes, role: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            value = np.asarray(image).copy()
    except Exception as error:
        raise AppleScaleError("SOURCE_PNG_INVALID", "source-only PNG cannot be decoded", role=role) from error
    if role == "lowres_depth":
        _require(value.shape == adapter.APPLE_SHAPE_HW and value.dtype == np.uint16, "APPLE_DEPTH_INVALID", "AppleDepth must be uint16 192x256 millimetres")
    else:
        _require(role == "confidence" and value.shape == adapter.APPLE_SHAPE_HW and value.dtype == np.uint8 and bool(np.all(value <= 2)), "CONFIDENCE_INVALID", "confidence must be uint8 0..2 at 192x256")
    return np.ascontiguousarray(value)


def decode_apple_scale_source(
    archive_path: Path,
    container_receipt: Mapping[str, Any],
    *,
    parent_id: str,
    video_id: str,
    timestamp_token: str,
    physical_frame_id: str,
    frame_plan_sha256: str,
    candidate_phase_completion_sha256: str,
) -> dict[str, Any]:
    """Open exactly AppleDepth and confidence; FARO/RGB members stay unopened."""

    adapter.decimal_timestamp_ns(timestamp_token)
    _require(physical_frame_id == f"{video_id}:{timestamp_token}", "SOURCE_IDENTITY_INVALID", "physical-frame identity drift")
    _hash(frame_plan_sha256, field="frame_plan_sha256")
    _hash(candidate_phase_completion_sha256, field="candidate_phase_completion_sha256")
    receipt = dict(container_receipt)
    _require(receipt.get("asset") == "upsampling.zip", "SOURCE_CONTAINER_ROLE_INVALID", "source-only scale requires the bound upsampling archive")
    materializer.verify_bound_container(archive_path, receipt)

    arrays: dict[str, np.ndarray] = {}
    members: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as bundle:
        for role in SOURCE_ROLES:
            member_path = f"{video_id}/{role}/{video_id}_{timestamp_token}.png"
            try:
                info = bundle.getinfo(member_path)
            except KeyError as error:
                raise AppleScaleError("SOURCE_MEMBER_MISSING", "source-only member is absent", role=role, member=member_path) from error
            materializer._validate_zip_info(info)
            _require(not info.is_dir() and info.file_size > 0, "SOURCE_MEMBER_INVALID", "source-only member is not a non-empty file", role=role)
            payload = bundle.read(info)
            _require(len(payload) == info.file_size and materializer.crc32_bytes(payload) == f"{info.CRC:08X}", "SOURCE_MEMBER_HASH_DRIFT", "source-only member bytes/CRC drift", role=role)
            decoded = _decode_png(payload, role)
            arrays[role] = decoded
            members[role] = {
                "role": role,
                "source_member_path": member_path,
                "member_bytes": len(payload),
                "member_sha256": materializer.sha256_bytes(payload),
                "member_crc32": f"{info.CRC:08X}",
                "decoded_dtype": str(decoded.dtype),
                "decoded_shape_hw": list(decoded.shape),
                "decoded_content_sha256": adapter.canonical_sha256(decoded),
            }

    source_receipt = _seal(
        {
            "schema": APPLE_SCALE_SOURCE_RECEIPT_SCHEMA,
            "parent_id": str(parent_id),
            "video_id": str(video_id),
            "timestamp_token": str(timestamp_token),
            "physical_frame_id": str(physical_frame_id),
            "frame_plan_sha256": frame_plan_sha256,
            "candidate_phase_completion_sha256": candidate_phase_completion_sha256,
            "upsampling_container": {
                "relative_path": str(receipt.get("relative_path")),
                "bytes": int(receipt["bytes"]),
                "sha256": str(receipt["sha256"]),
            },
            "members": members,
            "opened_member_roles": list(SOURCE_ROLES),
            "faro_member_opened": False,
            "rgb_member_opened": False,
            "truth_alignment_used": False,
        }
    )
    validate_apple_scale_source_receipt(source_receipt, arrays["lowres_depth"], arrays["confidence"])
    return {"apple_depth_mm": arrays["lowres_depth"], "confidence": arrays["confidence"], "source_receipt": source_receipt}


def validate_apple_scale_source_receipt(value: Any, apple_depth_mm: np.ndarray | None = None, confidence: np.ndarray | None = None) -> dict[str, Any]:
    receipt = _validate_seal(value, APPLE_SCALE_SOURCE_RECEIPT_SCHEMA, "APPLE_SCALE_SOURCE_RECEIPT_INVALID")
    expected = {
        "schema", "parent_id", "video_id", "timestamp_token", "physical_frame_id", "frame_plan_sha256",
        "candidate_phase_completion_sha256", "upsampling_container", "members", "opened_member_roles",
        "faro_member_opened", "rgb_member_opened", "truth_alignment_used", "content_sha256",
    }
    _require(set(receipt) == expected, "APPLE_SCALE_SOURCE_RECEIPT_INVALID", "source receipt field set drift")
    _require(receipt["physical_frame_id"] == f"{receipt['video_id']}:{receipt['timestamp_token']}" and bool(receipt["parent_id"]), "APPLE_SCALE_SOURCE_RECEIPT_INVALID", "source receipt identity drift")
    _hash(receipt["frame_plan_sha256"], field="frame_plan_sha256")
    _hash(receipt["candidate_phase_completion_sha256"], field="candidate_phase_completion_sha256")
    container = receipt["upsampling_container"]
    _require(isinstance(container, dict) and set(container) == {"relative_path", "bytes", "sha256"} and isinstance(container["bytes"], int) and container["bytes"] > 0, "APPLE_SCALE_SOURCE_RECEIPT_INVALID", "container binding drift")
    _hash(container["sha256"], field="upsampling_container.sha256")
    _require(receipt["opened_member_roles"] == list(SOURCE_ROLES) and receipt["faro_member_opened"] is False and receipt["rgb_member_opened"] is False and receipt["truth_alignment_used"] is False, "APPLE_SCALE_FIREWALL_BREACH", "source-only role firewall drift")
    _require(isinstance(receipt["members"], dict) and set(receipt["members"]) == set(SOURCE_ROLES), "APPLE_SCALE_SOURCE_RECEIPT_INVALID", "source-only member role set drift")
    for role in SOURCE_ROLES:
        member = receipt["members"][role]
        _require(isinstance(member, dict) and set(member) == {"role", "source_member_path", "member_bytes", "member_sha256", "member_crc32", "decoded_dtype", "decoded_shape_hw", "decoded_content_sha256"}, "APPLE_SCALE_SOURCE_RECEIPT_INVALID", "member binding field set drift", role=role)
        _require(member["role"] == role and isinstance(member["member_bytes"], int) and member["member_bytes"] > 0 and re.fullmatch(r"[0-9A-F]{8}", str(member["member_crc32"])) is not None, "APPLE_SCALE_SOURCE_RECEIPT_INVALID", "member binding drift", role=role)
        _hash(member["member_sha256"], field=f"{role}.member_sha256")
        _hash(member["decoded_content_sha256"], field=f"{role}.decoded_content_sha256")
    if apple_depth_mm is not None or confidence is not None:
        apple = np.asarray(apple_depth_mm)
        conf = np.asarray(confidence)
        _require(apple.shape == adapter.APPLE_SHAPE_HW and apple.dtype == np.uint16, "APPLE_DEPTH_INVALID", "bound AppleDepth array drift")
        _require(conf.shape == adapter.APPLE_SHAPE_HW and conf.dtype == np.uint8 and bool(np.all(conf <= 2)), "CONFIDENCE_INVALID", "bound confidence array drift")
        _require(adapter.canonical_sha256(apple) == receipt["members"]["lowres_depth"]["decoded_content_sha256"] and adapter.canonical_sha256(conf) == receipt["members"]["confidence"]["decoded_content_sha256"], "SOURCE_DECODED_HASH_DRIFT", "decoded source arrays differ from receipt")
    return receipt


def build_candidate_replay_binding(candidate_frame_record: Mapping[str, Any], native_depth_m: np.ndarray) -> dict[str, Any]:
    frame = validate_candidate_frame_record(dict(candidate_frame_record))
    inference = validate_depthart_inference_receipt(frame["inference_receipt"])
    native = np.ascontiguousarray(np.asarray(native_depth_m, dtype=np.float32))
    _require(native.shape == NATIVE_SHAPE_HW and adapter.canonical_sha256(native) == inference["native_output_array_sha256"], "CANDIDATE_NATIVE_HASH_DRIFT", "sealed native candidate differs from inference receipt")
    highres = upsample_native_depth(native)
    _require(adapter.canonical_sha256(highres) == inference["highres_output_array_sha256"], "CANDIDATE_HIGHRES_HASH_DRIFT", "replayed high-res candidate differs from inference receipt")
    binding = _seal(
        {
            "schema": CANDIDATE_REPLAY_BINDING_SCHEMA,
            "parent_id": inference["parent_id"],
            "video_id": inference["video_id"],
            "timestamp_token": inference["timestamp_token"],
            "physical_frame_id": inference["physical_frame_id"],
            "candidate_frame_record_sha256": frame["content_sha256"],
            "inference_receipt_sha256": inference["content_sha256"],
            "native_depth_array_sha256": inference["native_output_array_sha256"],
            "highres_depth_array_sha256": inference["highres_output_array_sha256"],
            "candidate_truth_payload_read": False,
            "candidate_truth_alignment_used": False,
        }
    )
    return {"candidate_highres_depth_m": highres, "candidate_binding": validate_candidate_replay_binding(binding)}


def validate_candidate_replay_binding(value: Any) -> dict[str, Any]:
    binding = _validate_seal(value, CANDIDATE_REPLAY_BINDING_SCHEMA, "CANDIDATE_REPLAY_BINDING_INVALID")
    expected = {
        "schema", "parent_id", "video_id", "timestamp_token", "physical_frame_id", "candidate_frame_record_sha256",
        "inference_receipt_sha256", "native_depth_array_sha256", "highres_depth_array_sha256",
        "candidate_truth_payload_read", "candidate_truth_alignment_used", "content_sha256",
    }
    _require(set(binding) == expected and binding["physical_frame_id"] == f"{binding['video_id']}:{binding['timestamp_token']}", "CANDIDATE_REPLAY_BINDING_INVALID", "candidate replay binding field/identity drift")
    for field in ("candidate_frame_record_sha256", "inference_receipt_sha256", "native_depth_array_sha256", "highres_depth_array_sha256"):
        _hash(binding[field], field=field)
    _require(binding["candidate_truth_payload_read"] is False and binding["candidate_truth_alignment_used"] is False, "CANDIDATE_REPLAY_TRUTH_FIREWALL_BREACH", "candidate replay crossed the truth firewall")
    return binding


def estimate_source_metric_scale(apple_depth_mm: np.ndarray, confidence: np.ndarray, candidate_samples_m: np.ndarray) -> dict[str, Any]:
    """Pure array estimator used by the sealed record factory and unit tests."""

    apple_raw = np.asarray(apple_depth_mm)
    conf = np.asarray(confidence)
    candidate = np.asarray(candidate_samples_m, dtype=np.float64)
    _require(apple_raw.shape == adapter.APPLE_SHAPE_HW and apple_raw.dtype == np.uint16, "APPLE_DEPTH_INVALID", "AppleDepth must be uint16 192x256")
    _require(conf.shape == adapter.APPLE_SHAPE_HW and conf.dtype == np.uint8 and bool(np.all(conf <= 2)), "CONFIDENCE_INVALID", "confidence must be uint8 0..2 at 192x256")
    _require(candidate.shape == adapter.APPLE_SHAPE_HW and bool(np.all(np.isfinite(candidate))), "CANDIDATE_SAMPLE_INVALID", "candidate samples must be finite 192x256 metres")
    apple_m = apple_raw.astype(np.float64) / 1000.0
    lower, upper = DEPTH_RANGE_M
    valid = (conf == 2) & (apple_m >= lower) & (apple_m <= upper) & (candidate >= lower) & (candidate <= upper)
    pair_count = int(np.sum(valid))
    pixel_ids = np.flatnonzero(valid).astype(np.int64)
    if pair_count < MINIMUM_PAIR_COUNT:
        return {
            "evaluable": False,
            "reason_codes": ["APPLE_SCALE_COMMON_SUPPORT_INSUFFICIENT"],
            "valid_pair_count": pair_count,
            "selected_pixel_ids_sha256": adapter.canonical_sha256(pixel_ids),
            "log_metric_scale": None,
            "metric_scale": None,
        }
    log_ratios = np.log(apple_m[valid] / candidate[valid])
    log_scale = round(float(np.median(np.asarray(log_ratios, dtype=np.float64))), 12)
    metric_scale = round(float(math.exp(log_scale)), 12)
    return {
        "evaluable": True,
        "reason_codes": [],
        "valid_pair_count": pair_count,
        "selected_pixel_ids_sha256": adapter.canonical_sha256(pixel_ids),
        "log_metric_scale": log_scale,
        "metric_scale": metric_scale,
    }


def build_source_scale_record(
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
    candidate_highres_depth_m: np.ndarray,
    source_receipt: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    source = validate_apple_scale_source_receipt(dict(source_receipt), apple_depth_mm, confidence)
    candidate = validate_candidate_replay_binding(dict(candidate_binding))
    identity = (source["parent_id"], source["video_id"], source["timestamp_token"], source["physical_frame_id"])
    _require(identity == (candidate["parent_id"], candidate["video_id"], candidate["timestamp_token"], candidate["physical_frame_id"]), "SOURCE_CANDIDATE_IDENTITY_MISMATCH", "source and sealed candidate identities differ")
    highres = np.asarray(candidate_highres_depth_m)
    _require(adapter.canonical_sha256(highres) == candidate["highres_depth_array_sha256"], "CANDIDATE_HIGHRES_HASH_DRIFT", "candidate high-res array differs from replay binding")
    sampled = sample_candidate_at_apple_centers(highres)
    estimate = estimate_source_metric_scale(np.asarray(apple_depth_mm), np.asarray(confidence), sampled)
    record = _seal(
        {
            "schema": SOURCE_SCALE_RECORD_SCHEMA,
            "claim_ceiling": CLAIM_CEILING,
            "parent_id": source["parent_id"],
            "video_id": source["video_id"],
            "timestamp_token": source["timestamp_token"],
            "physical_frame_id": source["physical_frame_id"],
            "source_receipt_sha256": source["content_sha256"],
            "candidate_binding_sha256": candidate["content_sha256"],
            "candidate_highres_depth_array_sha256": candidate["highres_depth_array_sha256"],
            "apple_depth_array_sha256": source["members"]["lowres_depth"]["decoded_content_sha256"],
            "confidence_array_sha256": source["members"]["confidence"]["decoded_content_sha256"],
            "estimator_id": ESTIMATOR_ID,
            "registration_id": REGISTRATION_ID,
            "median_rule": MEDIAN_RULE,
            "confidence_selection": "CONFIDENCE_EQ_2",
            "depth_range_m": list(DEPTH_RANGE_M),
            "minimum_pair_count": MINIMUM_PAIR_COUNT,
            **estimate,
            "faro_payload_read": False,
            "oracle_scale_read": False,
            "truth_alignment_used": False,
            "computed_before_oracle_join": True,
        }
    )
    return validate_source_scale_record(record)


def validate_source_scale_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, SOURCE_SCALE_RECORD_SCHEMA, "SOURCE_SCALE_RECORD_INVALID")
    expected = {
        "schema", "claim_ceiling", "parent_id", "video_id", "timestamp_token", "physical_frame_id",
        "source_receipt_sha256", "candidate_binding_sha256", "candidate_highres_depth_array_sha256",
        "apple_depth_array_sha256", "confidence_array_sha256", "estimator_id", "registration_id", "median_rule",
        "confidence_selection", "depth_range_m", "minimum_pair_count", "evaluable", "reason_codes",
        "valid_pair_count", "selected_pixel_ids_sha256", "log_metric_scale", "metric_scale", "faro_payload_read",
        "oracle_scale_read", "truth_alignment_used", "computed_before_oracle_join", "content_sha256",
    }
    _require(set(record) == expected and record["claim_ceiling"] == CLAIM_CEILING, "SOURCE_SCALE_RECORD_INVALID", "source scale field set/claim ceiling drift")
    _require(record["physical_frame_id"] == f"{record['video_id']}:{record['timestamp_token']}" and bool(record["parent_id"]), "SOURCE_SCALE_RECORD_INVALID", "source scale identity drift")
    for field in ("source_receipt_sha256", "candidate_binding_sha256", "candidate_highres_depth_array_sha256", "apple_depth_array_sha256", "confidence_array_sha256", "selected_pixel_ids_sha256"):
        _hash(record[field], field=field)
    _require(record["estimator_id"] == ESTIMATOR_ID and record["registration_id"] == REGISTRATION_ID and record["median_rule"] == MEDIAN_RULE and record["confidence_selection"] == "CONFIDENCE_EQ_2", "SOURCE_SCALE_METHOD_DRIFT", "source scale method drift")
    _require(record["depth_range_m"] == list(DEPTH_RANGE_M) and record["minimum_pair_count"] == MINIMUM_PAIR_COUNT, "SOURCE_SCALE_METHOD_DRIFT", "source scale support rule drift")
    _require(isinstance(record["valid_pair_count"], int) and record["valid_pair_count"] >= 0 and isinstance(record["reason_codes"], list), "SOURCE_SCALE_RECORD_INVALID", "source scale support metadata drift")
    if record["evaluable"]:
        _require(record["valid_pair_count"] >= MINIMUM_PAIR_COUNT and record["reason_codes"] == [] and isinstance(record["log_metric_scale"], (int, float)) and isinstance(record["metric_scale"], (int, float)) and record["metric_scale"] > 0.0, "SOURCE_SCALE_RECORD_INVALID", "evaluable source scale metrics drift")
        _require(abs(math.exp(float(record["log_metric_scale"])) - float(record["metric_scale"])) <= 1e-10, "SOURCE_SCALE_RECORD_INVALID", "metric/log scale mismatch")
    else:
        _require(record["valid_pair_count"] < MINIMUM_PAIR_COUNT and bool(record["reason_codes"]) and record["log_metric_scale"] is None and record["metric_scale"] is None, "SOURCE_SCALE_RECORD_INVALID", "unevaluable source scale carries metrics")
    _require(record["faro_payload_read"] is False and record["oracle_scale_read"] is False and record["truth_alignment_used"] is False and record["computed_before_oracle_join"] is True, "SOURCE_SCALE_TRUTH_FIREWALL_BREACH", "source scale record crossed the truth firewall")
    return record


def build_oracle_comparison(source_scale_record: Mapping[str, Any], oracle_canary_record: Mapping[str, Any]) -> dict[str, Any]:
    source = validate_source_scale_record(dict(source_scale_record))
    oracle = validate_factor_canary_record(dict(oracle_canary_record))
    _require(source["parent_id"] == oracle["parent_id"] and source["physical_frame_id"] == oracle["physical_frame_id"], "ORACLE_JOIN_IDENTITY_MISMATCH", "source scale and oracle identities differ")
    _require(source["candidate_highres_depth_array_sha256"] == oracle["candidate_depth_array_sha256"], "ORACLE_JOIN_CANDIDATE_MISMATCH", "source scale and oracle use different sealed candidates")
    oracle_scale = oracle["scale_record"]
    _require(oracle["factors"]["SCALE"]["evaluable"] and isinstance(oracle_scale, dict), "ORACLE_SCALE_NOT_EVALUABLE", "oracle canary scale is unavailable")
    oracle_log = float(oracle_scale["log_metric_scale"])
    if source["evaluable"]:
        source_log = float(source["log_metric_scale"])
        baseline_error = abs(oracle_log)
        source_error = abs(source_log - oracle_log)
        metrics = {
            "evaluable": True,
            "reason_codes": [],
            "oracle_log_metric_scale": oracle_log,
            "source_log_metric_scale": source_log,
            "source_minus_oracle_log_error": source_log - oracle_log,
            "baseline_abs_log_error": baseline_error,
            "source_abs_log_error": source_error,
            "abs_log_error_reduction": baseline_error - source_error,
        }
    else:
        metrics = {
            "evaluable": False,
            "reason_codes": list(source["reason_codes"]),
            "oracle_log_metric_scale": oracle_log,
            "source_log_metric_scale": None,
            "source_minus_oracle_log_error": None,
            "baseline_abs_log_error": abs(oracle_log),
            "source_abs_log_error": None,
            "abs_log_error_reduction": None,
        }
    return _seal(
        {
            "schema": ORACLE_COMPARISON_SCHEMA,
            "claim_ceiling": CLAIM_CEILING,
            "parent_id": source["parent_id"],
            "physical_frame_id": source["physical_frame_id"],
            "query_id": oracle["query_id"],
            "source_scale_record_sha256": source["content_sha256"],
            "oracle_canary_record_sha256": oracle["content_sha256"],
            "candidate_highres_depth_array_sha256": source["candidate_highres_depth_array_sha256"],
            **metrics,
            "aggregation_unit": "QUERY_THEN_FRAME_THEN_PARENT",
            "threshold_or_pass_fail_decision_applied": False,
        }
    )


def validate_oracle_comparison(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, ORACLE_COMPARISON_SCHEMA, "ORACLE_COMPARISON_INVALID")
    expected = {
        "schema", "claim_ceiling", "parent_id", "physical_frame_id", "query_id", "source_scale_record_sha256",
        "oracle_canary_record_sha256", "candidate_highres_depth_array_sha256", "evaluable", "reason_codes",
        "oracle_log_metric_scale", "source_log_metric_scale", "source_minus_oracle_log_error", "baseline_abs_log_error",
        "source_abs_log_error", "abs_log_error_reduction", "aggregation_unit", "threshold_or_pass_fail_decision_applied", "content_sha256",
    }
    _require(set(record) == expected and record["claim_ceiling"] == CLAIM_CEILING and record["aggregation_unit"] == "QUERY_THEN_FRAME_THEN_PARENT" and record["threshold_or_pass_fail_decision_applied"] is False, "ORACLE_COMPARISON_INVALID", "oracle comparison fields/claim drift")
    for field in ("source_scale_record_sha256", "oracle_canary_record_sha256", "candidate_highres_depth_array_sha256"):
        _hash(record[field], field=field)
    _require(isinstance(record["baseline_abs_log_error"], (int, float)) and record["baseline_abs_log_error"] >= 0.0, "ORACLE_COMPARISON_INVALID", "baseline oracle error drift")
    if record["evaluable"]:
        _require(record["reason_codes"] == [] and all(isinstance(record[field], (int, float)) and math.isfinite(float(record[field])) for field in ("oracle_log_metric_scale", "source_log_metric_scale", "source_minus_oracle_log_error", "source_abs_log_error", "abs_log_error_reduction")), "ORACLE_COMPARISON_INVALID", "evaluable oracle comparison drift")
    else:
        _require(bool(record["reason_codes"]) and all(record[field] is None for field in ("source_log_metric_scale", "source_minus_oracle_log_error", "source_abs_log_error", "abs_log_error_reduction")), "ORACLE_COMPARISON_INVALID", "unevaluable oracle comparison carries source metrics")
    return record


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def summarize_source_scale_canary(source_records: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate query -> physical frame -> parent; never treat queries as IID."""

    sources = [validate_source_scale_record(dict(value)) for value in source_records]
    joined = [validate_oracle_comparison(dict(value)) for value in comparisons]
    _require(bool(sources) and len({row["physical_frame_id"] for row in sources}) == len(sources), "SOURCE_SCALE_SUMMARY_INVALID", "source records must be non-empty and frame-unique")
    _require(len({(row["physical_frame_id"], row["query_id"]) for row in joined}) == len(joined), "SOURCE_SCALE_SUMMARY_INVALID", "oracle comparisons are duplicated")
    source_lookup = {row["physical_frame_id"]: row for row in sources}
    _require(all(row["physical_frame_id"] in source_lookup for row in joined), "SOURCE_SCALE_SUMMARY_INVALID", "oracle comparison lacks a source record")

    metric_names = ("baseline_abs_log_error", "source_abs_log_error", "abs_log_error_reduction", "source_minus_oracle_log_error", "source_log_metric_scale", "oracle_log_metric_scale")
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        if row["evaluable"]:
            by_frame[row["physical_frame_id"]].append(row)
    frame_rows: list[dict[str, Any]] = []
    for frame_id in sorted(by_frame):
        rows = by_frame[frame_id]
        frame_rows.append(
            {
                "parent_id": rows[0]["parent_id"],
                "physical_frame_id": frame_id,
                "query_count": len(rows),
                **{name: _median([float(row[name]) for row in rows]) for name in metric_names},
            }
        )
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_parent[row["parent_id"]].append(row)
    parent_rows: list[dict[str, Any]] = []
    for parent_id in sorted(by_parent):
        rows = by_parent[parent_id]
        parent_rows.append(
            {
                "parent_id": parent_id,
                "frame_count": len(rows),
                **{name: _median([float(row[name]) for row in rows]) for name in metric_names},
            }
        )

    reason_counts = Counter(code for row in sources if not row["evaluable"] for code in row["reason_codes"])
    parent_macro = {name: _median([float(row[name]) for row in parent_rows]) for name in metric_names}
    return _seal(
        {
            "schema": SOURCE_SCALE_SUMMARY_SCHEMA,
            "claim_ceiling": CLAIM_CEILING,
            "estimator_id": ESTIMATOR_ID,
            "source_frame_count": len(sources),
            "source_evaluable_frame_count": sum(bool(row["evaluable"]) for row in sources),
            "source_unknown_frame_count": sum(not bool(row["evaluable"]) for row in sources),
            "source_unknown_reason_counts": dict(sorted(reason_counts.items())),
            "oracle_query_count": len(joined),
            "oracle_evaluable_query_count": sum(bool(row["evaluable"]) for row in joined),
            "oracle_paired_frame_count": len(frame_rows),
            "oracle_paired_parent_count": len(parent_rows),
            "aggregation": "MEDIAN_QUERY_WITHIN_FRAME_THEN_MEDIAN_FRAME_WITHIN_PARENT_THEN_MEDIAN_ACROSS_PARENTS",
            "parent_macro_metrics": parent_macro,
            "parents_improved_over_zero_scale": sum(float(row["abs_log_error_reduction"]) > 0.0 for row in parent_rows),
            "frames_improved_over_zero_scale": sum(float(row["abs_log_error_reduction"]) > 0.0 for row in frame_rows),
            "parent_metrics": parent_rows,
            "threshold_or_pass_fail_decision_applied": False,
        }
    )


__all__ = [
    "APPLE_SCALE_SOURCE_RECEIPT_SCHEMA", "AppleScaleError", "CLAIM_CEILING", "DEPTH_RANGE_M", "ESTIMATOR_ID",
    "MINIMUM_PAIR_COUNT", "ORACLE_COMPARISON_SCHEMA", "REGISTRATION_ID", "SOURCE_SCALE_RECORD_SCHEMA",
    "SOURCE_SCALE_SUMMARY_SCHEMA", "build_candidate_replay_binding", "build_oracle_comparison", "build_source_scale_record",
    "decode_apple_scale_source", "estimate_source_metric_scale", "sample_candidate_at_apple_centers",
    "summarize_source_scale_canary", "validate_apple_scale_source_receipt", "validate_candidate_replay_binding",
    "validate_oracle_comparison", "validate_source_scale_record",
]
