#!/usr/bin/env python3
"""Source-only TARO R6 factor-to-interval-reducer integration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale
from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "docs/research/taro/TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_PROTOCOL_LOCK_2026-08-11.json"
PROTOCOL_BYTES = 5500
PROTOCOL_SHA256 = "B7FE9BBC120BB53FA976ED637E8F8BE22D86B1F2A257300BE1F82BDF78D5D056"
RUNTIME_ID = "TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_V1"
REDUCER_VERSION = "TARO_O1R_R6_SOURCE_ONLY_INTERVAL_REDUCER_V1"
QUERY_LOOKUP_SCHEMA = "blindassist.taro.o1r.r6_source_only_query_uncertainty_lookup.v1"
UNCERTAINTY_RECORD_SCHEMA = "blindassist.taro.o1r.r6_query_uncertainty_record.v1"
QUERY_RESULT_SCHEMA = "blindassist.taro.o1r.r6_query_reducer_result.v1"
BUNDLE_RESULT_SCHEMA = "blindassist.taro.o1r.r6_nine_query_reducer_bundle.v1"
UNCERTAINTY_TARGET_FIELDS = {
    "scale_log_abs_residual": "scale_q95_log",
    "support_height_abs_residual_m": "support_height_q95_m",
    "support_normal_abs_residual_rad": "support_normal_q95_rad",
    "boundary_localization_abs_residual_m": "boundary_localization_q95_m",
}
CLAIM_CEILING = "Source-only O1R reducer-integration mechanics; no final effectiveness, deployment, device, product, or safety claim."
_SHA256 = re.compile(r"^[0-9A-F]{64}$")


class ReducerIntegrationError(RuntimeError):
    """Stable fail-closed integration error."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise ReducerIntegrationError(code, message, **context)


def _assert_protocol_binding() -> None:
    require(PROTOCOL_PATH.is_file(), "O1R_PROTOCOL_MISSING", "O1R integration protocol is missing")
    require(PROTOCOL_PATH.stat().st_size == PROTOCOL_BYTES, "O1R_PROTOCOL_BINDING_DRIFT", "O1R protocol byte count drift")
    require(hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest().upper() == PROTOCOL_SHA256, "O1R_PROTOCOL_BINDING_DRIFT", "O1R protocol hash drift")


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "O1R_SEAL_COLLISION", "caller supplied a content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Any, schema: str, code: str) -> dict[str, Any]:
    require(isinstance(value, dict), code, "sealed record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(isinstance(observed, str) and bool(_SHA256.fullmatch(observed)), code, "sealed record hash is malformed")
    require(adapter.canonical_sha256(record) == observed, code, "sealed record hash drift")
    record["content_sha256"] = observed
    require(record.get("schema") == schema, code, "sealed record schema drift")
    return record


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_bound_arrays(
    bundle: Mapping[str, Any],
    candidate_highres_depth_m: np.ndarray,
    confidence: np.ndarray,
    intrinsics_apple_3x3: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate_highres_depth_m)
    conf = np.asarray(confidence)
    matrix = adapter._intrinsics_matrix(intrinsics_apple_3x3, adapter.APPLE_SHAPE_HW)
    require(
        candidate.shape == adapter.HIGHRES_SHAPE_HW
        and candidate.dtype.kind == "f"
        and bool(np.all(np.isfinite(candidate))),
        "O1R_CANDIDATE_INVALID",
        "candidate depth must be finite floating-point 1440x1920 metres",
    )
    require(
        conf.shape == adapter.APPLE_SHAPE_HW
        and conf.dtype == np.uint8
        and bool(np.all(conf <= 2)),
        "O1R_CONFIDENCE_INVALID",
        "confidence must be uint8 0..2 at 192x256",
    )
    candidate64 = np.ascontiguousarray(candidate, dtype=np.float64)
    conf8 = np.ascontiguousarray(conf, dtype=np.uint8)
    bindings = bundle["input_bindings"]
    require(adapter.canonical_sha256(candidate64) == bindings["candidate_highres_depth_sha256"], "O1R_CANDIDATE_BINDING_DRIFT", "candidate depth differs from the R6 binding")
    require(adapter.canonical_sha256(conf8) == bindings["confidence_sha256"], "O1R_CONFIDENCE_BINDING_DRIFT", "confidence differs from the R6 binding")
    require(adapter.canonical_sha256(matrix) == bindings["intrinsics_apple_sha256"], "O1R_INTRINSICS_BINDING_DRIFT", "Apple intrinsics differ from the R6 binding")
    return candidate64, conf8, matrix


def _source_only_lookup(
    bundle: Mapping[str, Any],
    slot: Mapping[str, Any],
    candidate_lowres_m: np.ndarray,
    confidence: np.ndarray,
    matrix: np.ndarray,
) -> dict[str, Any]:
    query = prospective._validate_query(slot["query_receipt"])
    require(query["content_sha256"] == slot["query_receipt"]["content_sha256"], "O1R_QUERY_BINDING_DRIFT", "query receipt seal drift")
    rows, columns = np.mgrid[0 : adapter.APPLE_SHAPE_HW[0], 0 : adapter.APPLE_SHAPE_HW[1]]
    z = np.asarray(candidate_lowres_m, dtype=np.float64)
    points = np.stack(
        (
            (columns.astype(np.float64) - matrix[0, 2]) * z / matrix[0, 0],
            (rows.astype(np.float64) - matrix[1, 2]) * z / matrix[1, 1],
            z,
        ),
        axis=-1,
    )
    origin, _, lateral, heading = adapter._query_receipt_vectors(query)
    up = adapter._normalize_vector(query["virtual_query_frame"]["gravity_up_camera_xyz"], "O1R_QUERY_FRAME_INVALID")
    path_origin = origin + float(query["path_lateral_offset_m"]) * lateral
    rel = points - path_origin
    along = rel @ heading
    perpendicular_vector = rel - along[..., None] * heading
    perpendicular_ground = perpendicular_vector - (perpendicular_vector @ up)[..., None] * up
    perpendicular = np.linalg.norm(perpendicular_ground, axis=-1)
    valid_depth = np.isfinite(z) & (z >= adapter.DEPTH_RANGE_M[0]) & (z <= adapter.DEPTH_RANGE_M[1])
    support = (
        valid_depth
        & (along >= float(query["minimum_forward_m"]))
        & (along <= float(query["horizon_m"]))
        & (perpendicular <= float(query["capsule_radius_m"]))
    )
    support_count = int(np.sum(support))
    support_ids = np.ascontiguousarray(np.column_stack((columns[support], rows[support])), dtype=np.int32)
    if support_count:
        counts = np.bincount(confidence[support].astype(np.int64), minlength=3)
        confidence_value = int(np.flatnonzero(counts == np.max(counts))[-1])
        range_m = float(np.median(z[support]))
    else:
        counts = np.zeros(3, dtype=np.int64)
        confidence_value = None
        range_m = None
    valid = bool(
        support_count >= adapter.MINIMUM_QUERY_SUPPORT_POINTS
        and confidence_value in (0, 1, 2)
        and _finite(range_m)
        and adapter.DEPTH_RANGE_M[0] <= float(range_m) <= adapter.DEPTH_RANGE_M[1]
    )
    return _seal(
        {
            "schema": QUERY_LOOKUP_SCHEMA,
            "physical_frame_id": bundle["physical_frame_id"],
            "source_frame_receipt_sha256": bundle["source_frame_receipt_sha256"],
            "prospective_bundle_sha256": bundle["content_sha256"],
            "prospective_slot_sha256": slot["content_sha256"],
            "query_id": slot["query_id"],
            "query_receipt_sha256": query["content_sha256"],
            "candidate_depth_array_sha256": bundle["input_bindings"]["candidate_highres_depth_sha256"],
            "confidence_array_sha256": bundle["input_bindings"]["confidence_sha256"],
            "intrinsics_apple_sha256": bundle["input_bindings"]["intrinsics_apple_sha256"],
            "support_cell_ids_uv_sha256": adapter.canonical_sha256(support_ids),
            "support_count": support_count,
            "minimum_support_points": adapter.MINIMUM_QUERY_SUPPORT_POINTS,
            "confidence_counts": [int(value) for value in counts],
            "confidence_value": confidence_value,
            "range_m": range_m,
            "range_kind": "R1_BASELINE_CANDIDATE_OPTICAL_AXIS_Z_METERS_MEDIAN",
            "corridor_rule": "MINIMUM_FORWARD_THROUGH_HORIZON_INCLUSIVE_CAPSULE_GROUND_PROJECTION_NO_HEIGHT_GATE",
            "tie_rule": "HIGHEST_CONFIDENCE_INDEX",
            "caller_scalar_allowed": False,
            "valid": valid,
            "reason_codes": [] if valid else ["SOURCE_QUERY_UNCERTAINTY_SUPPORT_INSUFFICIENT"],
        }
    )


def _resolve_uncertainty(model: Any, lookup: Mapping[str, Any]) -> dict[str, Any]:
    if lookup["valid"] is not True:
        return _seal({
            "schema": UNCERTAINTY_RECORD_SCHEMA,
            "valid": False,
            "model_sha256": model.content_sha256,
            "target_resolutions": {},
            "components_m": None,
            "total_m": None,
            "reason_codes": ["SOURCE_QUERY_UNCERTAINTY_SUPPORT_INSUFFICIENT"],
        })
    resolutions: dict[str, Any] = {}
    values: dict[str, float] = {}
    for target, field in UNCERTAINTY_TARGET_FIELDS.items():
        # The public entry validates the factory-bound model once. Reuse the
        # validated resolver here so nine queries do not rehash millions of
        # immutable fit samples 36 additional times.
        resolved = model._resolve_validated(int(lookup["confidence_value"]), float(lookup["range_m"]), target)
        resolutions[target] = dict(resolved)
        if resolved.get("valid") is True and _finite(resolved.get("value")) and float(resolved["value"]) >= 0.0:
            values[field] = float(resolved["value"])
    if len(values) != len(UNCERTAINTY_TARGET_FIELDS):
        return _seal({
            "schema": UNCERTAINTY_RECORD_SCHEMA,
            "valid": False,
            "model_sha256": model.content_sha256,
            "target_resolutions": resolutions,
            "components_m": None,
            "total_m": None,
            "reason_codes": ["UNCERTAINTY_INVALID_QUERY_UNKNOWN"],
        })
    scale_m = adapter.HORIZON_M * (math.exp(values["scale_q95_log"]) - 1.0) if values["scale_q95_log"] <= 50.0 else math.inf
    support_m = values["support_height_q95_m"] + adapter.HORIZON_M * math.tan(min(values["support_normal_q95_rad"], math.radians(45.0)))
    boundary_m = values["boundary_localization_q95_m"]
    total_m = scale_m + support_m + boundary_m
    components = {
        **values,
        "scale_m": scale_m,
        "support_m": support_m,
        "boundary_m": boundary_m,
    }
    valid = all(_finite(value) and float(value) >= 0.0 for value in (*components.values(), total_m))
    return _seal({
        "schema": UNCERTAINTY_RECORD_SCHEMA,
        "valid": valid,
        "model_sha256": model.content_sha256,
        "target_resolutions": resolutions,
        "components_m": components if valid else None,
        "total_m": total_m if valid else None,
        "reason_codes": [] if valid else ["UNCERTAINTY_INVALID_QUERY_UNKNOWN"],
    })


def _knownness(slot: Mapping[str, Any], lookup: Mapping[str, Any] | None) -> dict[str, Any]:
    clearance = slot["factor_blocks"]["QUERY_CLEARANCE"]
    validity = clearance.get("validity", {})
    return {
        "known": bool(clearance.get("evaluable") is True and validity.get("known") is True),
        "query_support_points": validity.get("query_support_points", 0),
        "observed_forward_m": validity.get("observed_forward_m"),
        "local_valid_fraction": validity.get("local_valid_fraction"),
        "uncertainty_support_points": 0 if lookup is None else lookup.get("support_count", 0),
    }


def _unknown_result(
    slot: Mapping[str, Any],
    reason: str,
    *,
    lookup: Mapping[str, Any] | None = None,
    uncertainty: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    query = slot.get("query_receipt")
    return _seal(
        {
            "schema": QUERY_RESULT_SCHEMA,
            "reducer_version": REDUCER_VERSION,
            "physical_frame_id": slot["physical_frame_id"],
            "query_id": slot["query_id"],
            "grid_index": slot["grid_index"],
            "prospective_slot_sha256": slot["content_sha256"],
            "query_receipt_sha256": query.get("content_sha256") if isinstance(query, dict) else None,
            "query_clearance_owner": "R1_BASELINE",
            "value_m": None,
            "uncertainty_m": None,
            "interval_m": {"lower": None, "upper": None},
            "state": "UNKNOWN",
            "knownness": _knownness(slot, lookup),
            "uncertainty_lookup": copy.deepcopy(dict(lookup)) if lookup is not None else None,
            "uncertainty": copy.deepcopy(dict(uncertainty)) if uncertainty is not None else None,
            "reason_codes": [reason],
            "final_state_authorized": True,
            "reducer_executed": True,
        }
    )


def _reduce_slot(slot: Mapping[str, Any], lookup: Mapping[str, Any], uncertainty: Mapping[str, Any]) -> dict[str, Any]:
    blocks = slot["factor_blocks"]
    for name in ("SUPPORT", "BOUNDARY", "QUERY_CLEARANCE"):
        block = blocks[name]
        if block.get("evaluable") is not True or block.get("validity", {}).get("known") is not True:
            reason = str(block.get("reason_codes", ["FACTOR_VALIDITY_INCOMPLETE"])[0])
            return _unknown_result(slot, reason, lookup=lookup, uncertainty=uncertainty)
    support_validity = blocks["SUPPORT"]["validity"]
    boundary_validity = blocks["BOUNDARY"]["validity"]
    support_known = (
        int(support_validity.get("query_support_points", 0)) >= adapter.MINIMUM_QUERY_SUPPORT_POINTS
        and _finite(support_validity.get("observed_forward_m"))
        and float(support_validity["observed_forward_m"]) >= adapter.MINIMUM_QUERY_OBSERVED_FORWARD_M
    )
    boundary_known = (
        _finite(boundary_validity.get("local_valid_fraction"))
        and float(boundary_validity["local_valid_fraction"]) >= adapter.MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION
    )
    if not support_known or not boundary_known:
        return _unknown_result(slot, "QUERY_KNOWNNESS_GATE_FAILED", lookup=lookup, uncertainty=uncertainty)
    clearance = blocks["QUERY_CLEARANCE"]
    require(clearance["owner"] == "R1_BASELINE", "O1R_QUERY_CLEARANCE_OWNER_DRIFT", "query-clearance owner is not R1_BASELINE")
    if lookup.get("valid") is not True:
        return _unknown_result(slot, "SOURCE_QUERY_UNCERTAINTY_SUPPORT_INSUFFICIENT", lookup=lookup, uncertainty=uncertainty)
    if uncertainty.get("valid") is not True:
        return _unknown_result(slot, "UNCERTAINTY_INVALID_QUERY_UNKNOWN", lookup=lookup, uncertainty=uncertainty)
    value_m = clearance.get("value", {}).get("clearance_m")
    uncertainty_m = uncertainty.get("total_m")
    if not _finite(value_m) or not _finite(uncertainty_m) or float(uncertainty_m) < 0.0:
        return _unknown_result(slot, "REDUCER_NUMERIC_INVALID", lookup=lookup, uncertainty=uncertainty)
    value = float(value_m)
    radius = float(uncertainty_m)
    lower, upper = value - radius, value + radius
    if lower > adapter.CLEAR_MARGIN_M:
        state = "CLEAR_OBSERVED"
    elif upper <= adapter.OCCUPIED_MARGIN_M:
        state = "OCCUPIED_OBSERVED"
    else:
        state = "UNKNOWN"
    return _seal(
        {
            "schema": QUERY_RESULT_SCHEMA,
            "reducer_version": REDUCER_VERSION,
            "physical_frame_id": slot["physical_frame_id"],
            "query_id": slot["query_id"],
            "grid_index": slot["grid_index"],
            "prospective_slot_sha256": slot["content_sha256"],
            "query_receipt_sha256": slot["query_receipt"]["content_sha256"],
            "query_clearance_owner": "R1_BASELINE",
            "value_m": value,
            "uncertainty_m": radius,
            "interval_m": {"lower": lower, "upper": upper},
            "state": state,
            "knownness": _knownness(slot, lookup),
            "uncertainty_lookup": copy.deepcopy(dict(lookup)),
            "uncertainty": copy.deepcopy(dict(uncertainty)),
            "reason_codes": [] if state != "UNKNOWN" else ["INTERVAL_STRADDLES_DECISION_MARGINS"],
            "final_state_authorized": True,
            "reducer_executed": True,
        }
    )


def integrate_prospective_factor_bundle(
    *,
    prospective_bundle: dict[str, Any],
    candidate_highres_depth_m: np.ndarray,
    confidence: np.ndarray,
    intrinsics_apple_3x3: Any,
    uncertainty_model: Any,
) -> dict[str, Any]:
    """Attach fit-only uncertainty and execute the sole final-state reducer."""

    _assert_protocol_binding()
    model = adapter._validate_uncertainty_model(uncertainty_model)
    return _integrate_with_validated_model(
        prospective_bundle=prospective_bundle,
        candidate_highres_depth_m=candidate_highres_depth_m,
        confidence=confidence,
        intrinsics_apple_3x3=intrinsics_apple_3x3,
        uncertainty_model=model,
    )


def _integrate_with_validated_model(
    *,
    prospective_bundle: dict[str, Any],
    candidate_highres_depth_m: np.ndarray,
    confidence: np.ndarray,
    intrinsics_apple_3x3: Any,
    uncertainty_model: Any,
) -> dict[str, Any]:
    """Internal batch seam; caller must validate the immutable model once."""

    model = uncertainty_model
    require(
        isinstance(model, adapter._UncertaintyModel)
        and adapter._UNCERTAINTY_FACTORY_FINGERPRINTS.get(model) == model.content_sha256,
        "O1R_UNCERTAINTY_MODEL_NOT_VALIDATED",
        "batch seam requires the already validated factory-bound uncertainty model",
    )
    bundle = prospective.validate_prospective_factor_bundle(
        prospective_bundle,
        candidate_highres_depth_m=np.asarray(candidate_highres_depth_m),
    )
    require(bundle["query_clearance_owner"] == "R1_BASELINE", "O1R_QUERY_CLEARANCE_OWNER_DRIFT", "R6 bundle query-clearance owner drift")
    candidate, conf, matrix = _validate_bound_arrays(bundle, candidate_highres_depth_m, confidence, intrinsics_apple_3x3)
    candidate_lowres = apple_scale.sample_candidate_at_apple_centers(candidate)
    results: list[dict[str, Any]] = []
    for index, slot in enumerate(bundle["query_slots"]):
        require(slot["grid_index"] == index, "O1R_QUERY_ORDER_DRIFT", "R6 query slots are reordered")
        if slot["query_receipt"] is None:
            results.append(_unknown_result(slot, "SOURCE_QUERY_FRAME_UNAVAILABLE"))
            continue
        lookup = _source_only_lookup(bundle, slot, candidate_lowres, conf, matrix)
        uncertainty = _resolve_uncertainty(model, lookup)
        results.append(_reduce_slot(slot, lookup, uncertainty))
    state_counts = {
        state: sum(result["state"] == state for result in results)
        for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
    }
    output = _seal(
        {
            "schema": BUNDLE_RESULT_SCHEMA,
            "protocol_sha256": PROTOCOL_SHA256,
            "runtime_id": RUNTIME_ID,
            "reducer_version": REDUCER_VERSION,
            "prospective_bundle_sha256": bundle["content_sha256"],
            "physical_frame_id": bundle["physical_frame_id"],
            "source_frame_receipt_sha256": bundle["source_frame_receipt_sha256"],
            "candidate_frame_record_sha256": bundle["candidate_frame_record_sha256"],
            "candidate_depth_array_sha256": bundle["input_bindings"]["candidate_highres_depth_sha256"],
            "confidence_array_sha256": bundle["input_bindings"]["confidence_sha256"],
            "uncertainty_model_sha256": model.content_sha256,
            "query_results": results,
            "query_result_count": len(results),
            "state_counts": state_counts,
            "complete_reducer_execution": len(results) == 9 and all(result["reducer_executed"] is True for result in results),
            "final_state_producer": REDUCER_VERSION,
            "r6_bundle_mutated": False,
            "training_steps": 0,
            "network_requests": 0,
            "claim_ceiling": CLAIM_CEILING,
        }
    )
    return validate_reducer_bundle(output)


def validate_reducer_bundle(value: Any) -> dict[str, Any]:
    _assert_protocol_binding()
    bundle = _validate_seal(value, BUNDLE_RESULT_SCHEMA, "O1R_RESULT_SEAL_MISMATCH")
    expected = {
        "schema", "protocol_sha256", "runtime_id", "reducer_version", "prospective_bundle_sha256", "physical_frame_id",
        "source_frame_receipt_sha256", "candidate_frame_record_sha256", "candidate_depth_array_sha256", "confidence_array_sha256",
        "uncertainty_model_sha256", "query_results", "query_result_count", "state_counts", "complete_reducer_execution",
        "final_state_producer", "r6_bundle_mutated", "training_steps", "network_requests", "claim_ceiling", "content_sha256",
    }
    require(set(bundle) == expected, "O1R_RESULT_FIELD_DRIFT", "O1R result fields drift")
    require(bundle["protocol_sha256"] == PROTOCOL_SHA256 and bundle["runtime_id"] == RUNTIME_ID and bundle["reducer_version"] == REDUCER_VERSION and bundle["final_state_producer"] == REDUCER_VERSION, "O1R_RESULT_IDENTITY_DRIFT", "O1R result identity drift")
    for field in ("prospective_bundle_sha256", "source_frame_receipt_sha256", "candidate_frame_record_sha256", "candidate_depth_array_sha256", "confidence_array_sha256", "uncertainty_model_sha256"):
        require(isinstance(bundle[field], str) and bool(_SHA256.fullmatch(bundle[field])), "O1R_RESULT_HASH_INVALID", "O1R result hash is malformed", field=field)
    results = bundle["query_results"]
    require(isinstance(results, list) and len(results) == bundle["query_result_count"] == 9, "O1R_RESULT_CARDINALITY", "O1R result must retain nine queries")
    observed_counts = {state: 0 for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")}
    for index, raw in enumerate(results):
        result = _validate_seal(raw, QUERY_RESULT_SCHEMA, "O1R_QUERY_RESULT_SEAL_MISMATCH")
        result_fields = {
            "schema", "reducer_version", "physical_frame_id", "query_id", "grid_index", "prospective_slot_sha256",
            "query_receipt_sha256", "query_clearance_owner", "value_m", "uncertainty_m", "interval_m", "state",
            "knownness", "uncertainty_lookup", "uncertainty", "reason_codes", "final_state_authorized",
            "reducer_executed", "content_sha256",
        }
        require(set(result) == result_fields, "O1R_QUERY_RESULT_FIELD_DRIFT", "O1R query result fields drift")
        require(result["grid_index"] == index and result["state"] in observed_counts and result["query_clearance_owner"] == "R1_BASELINE", "O1R_QUERY_RESULT_IDENTITY_DRIFT", "O1R query result identity/order drift")
        require(result["reducer_version"] == REDUCER_VERSION and result["final_state_authorized"] is True and result["reducer_executed"] is True, "O1R_FINAL_STATE_AUTHORITY_DRIFT", "O1R query final-state authority drift")
        require(isinstance(result["prospective_slot_sha256"], str) and bool(_SHA256.fullmatch(result["prospective_slot_sha256"])), "O1R_QUERY_RESULT_IDENTITY_DRIFT", "prospective slot hash is malformed")
        lookup = None
        if result["uncertainty_lookup"] is not None:
            lookup = _validate_seal(result["uncertainty_lookup"], QUERY_LOOKUP_SCHEMA, "O1R_QUERY_LOOKUP_SEAL_MISMATCH")
            lookup_fields = {
                "schema", "physical_frame_id", "source_frame_receipt_sha256", "prospective_bundle_sha256", "prospective_slot_sha256",
                "query_id", "query_receipt_sha256", "candidate_depth_array_sha256", "confidence_array_sha256",
                "intrinsics_apple_sha256", "support_cell_ids_uv_sha256", "support_count", "minimum_support_points",
                "confidence_counts", "confidence_value", "range_m", "range_kind", "corridor_rule", "tie_rule",
                "caller_scalar_allowed", "valid", "reason_codes", "content_sha256",
            }
            require(set(lookup) == lookup_fields, "O1R_QUERY_LOOKUP_FIELD_DRIFT", "O1R query lookup fields drift")
            require(lookup["physical_frame_id"] == result["physical_frame_id"] and lookup["query_id"] == result["query_id"] and lookup["prospective_slot_sha256"] == result["prospective_slot_sha256"], "O1R_QUERY_LOOKUP_IDENTITY_DRIFT", "O1R query lookup identity drift")
            require(lookup["candidate_depth_array_sha256"] == bundle["candidate_depth_array_sha256"] and lookup["confidence_array_sha256"] == bundle["confidence_array_sha256"], "O1R_QUERY_LOOKUP_INPUT_DRIFT", "O1R query lookup input binding drift")
        uncertainty = None
        if result["uncertainty"] is not None:
            uncertainty = _validate_seal(result["uncertainty"], UNCERTAINTY_RECORD_SCHEMA, "O1R_UNCERTAINTY_RECORD_SEAL_MISMATCH")
            require(set(uncertainty) == {"schema", "valid", "model_sha256", "target_resolutions", "components_m", "total_m", "reason_codes", "content_sha256"}, "O1R_UNCERTAINTY_RECORD_FIELD_DRIFT", "O1R uncertainty record fields drift")
            require(uncertainty["model_sha256"] == bundle["uncertainty_model_sha256"], "O1R_UNCERTAINTY_MODEL_DRIFT", "O1R uncertainty model binding drift")
        observed_counts[result["state"]] += 1
        if result["value_m"] is None:
            require(result["state"] == "UNKNOWN" and result["uncertainty_m"] is None and result["interval_m"] == {"lower": None, "upper": None} and bool(result["reason_codes"]), "O1R_UNKNOWN_RESULT_DRIFT", "unknown result payload drift")
        else:
            require(_finite(result["value_m"]) and _finite(result["uncertainty_m"]) and float(result["uncertainty_m"]) >= 0.0, "O1R_QUERY_RESULT_NUMERIC_DRIFT", "O1R query result numeric values drift")
            require(lookup is not None and lookup["valid"] is True and uncertainty is not None and uncertainty["valid"] is True and result["knownness"].get("known") is True, "O1R_QUERY_RESULT_PROVENANCE_DRIFT", "numeric O1R result lacks valid lookup/uncertainty provenance")
            require(abs(float(result["uncertainty_m"]) - float(uncertainty["total_m"])) <= 1e-12, "O1R_QUERY_UNCERTAINTY_DRIFT", "O1R result uncertainty differs from its sealed record")
            lower = float(result["value_m"]) - float(result["uncertainty_m"])
            upper = float(result["value_m"]) + float(result["uncertainty_m"])
            require(abs(lower - float(result["interval_m"]["lower"])) <= 1e-12 and abs(upper - float(result["interval_m"]["upper"])) <= 1e-12, "O1R_QUERY_INTERVAL_DRIFT", "O1R query interval is not recomputable")
            expected_state = "CLEAR_OBSERVED" if lower > adapter.CLEAR_MARGIN_M else "OCCUPIED_OBSERVED" if upper <= adapter.OCCUPIED_MARGIN_M else "UNKNOWN"
            require(result["state"] == expected_state, "O1R_QUERY_STATE_DRIFT", "O1R query state is not recomputable from the interval")
            require(result["reason_codes"] == ([] if expected_state != "UNKNOWN" else ["INTERVAL_STRADDLES_DECISION_MARGINS"]), "O1R_QUERY_REASON_DRIFT", "O1R query reason codes drift")
    require(bundle["state_counts"] == observed_counts and bundle["complete_reducer_execution"] is True, "O1R_RESULT_AGGREGATION_DRIFT", "O1R result aggregation drift")
    require(bundle["r6_bundle_mutated"] is False and bundle["training_steps"] == bundle["network_requests"] == 0 and bundle["claim_ceiling"] == CLAIM_CEILING, "O1R_EXECUTION_FIREWALL_DRIFT", "O1R execution firewall drift")
    return bundle


__all__ = [
    "BUNDLE_RESULT_SCHEMA",
    "PROTOCOL_SHA256",
    "QUERY_LOOKUP_SCHEMA",
    "QUERY_RESULT_SCHEMA",
    "UNCERTAINTY_RECORD_SCHEMA",
    "REDUCER_VERSION",
    "RUNTIME_ID",
    "ReducerIntegrationError",
    "integrate_prospective_factor_bundle",
    "validate_reducer_bundle",
]
