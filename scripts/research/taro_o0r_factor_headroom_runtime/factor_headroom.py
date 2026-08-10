#!/usr/bin/env python3
"""Candidate-gauge mechanics for TARO O0R factorial evaluation.

The frozen reducer applies SCALE to every shape-valued geometry block.  FARO
support/boundary values therefore have to be expressed in candidate gauge when
an arm also patches SCALE; otherwise combined arms would scale absolute metric
truth twice.  This module keeps that conversion explicit and hash-bound.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Mapping

import numpy as np

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


CANDIDATE_RELATIVE_SCALE_SCHEMA = adapter.CANDIDATE_RELATIVE_SCALE_SCHEMA
CANDIDATE_GAUGE_ORACLE_ID = adapter.CANDIDATE_GAUGE_ORACLE_ID
CANDIDATE_GAUGE_ORIGIN = adapter.CANDIDATE_GAUGE_ORIGIN
SCALE_VALUE_KIND = adapter.CANDIDATE_SCALE_VALUE_KIND
MINIMUM_SCALE_PAIRS = adapter.MINIMUM_SUPPORT_POINTS


class FactorHeadroomError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FactorHeadroomError(code, message, **context)


def _canonical_copy(value: Any) -> Any:
    return json.loads(adapter.canonical_json_bytes(value).decode("utf-8"))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _canonical_copy(dict(value))
    require("content_sha256" not in payload, "SEAL_INPUT_INVALID", "caller may not supply a content hash")
    payload["content_sha256"] = adapter.canonical_sha256(payload)
    return _canonical_copy(payload)


def _common_ids(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    require(raw.ndim == 2 and raw.shape[1] == 2 and np.issubdtype(raw.dtype, np.integer), "COMMON_SUPPORT_INVALID", "common support must be integer Nx2 pixel ids")
    ids = raw.astype(np.int64, copy=False)
    require(len(ids) >= MINIMUM_SCALE_PAIRS, "COMMON_SUPPORT_INSUFFICIENT", "candidate-relative scale needs at least 256 frozen support points", observed=len(ids))
    require(bool(np.all((ids[:, 0] >= 0) & (ids[:, 0] < adapter.HIGHRES_SHAPE_HW[1]) & (ids[:, 1] >= 0) & (ids[:, 1] < adapter.HIGHRES_SHAPE_HW[0]))), "COMMON_SUPPORT_INVALID", "common support contains out-of-bounds pixels")
    linear = ids[:, 1] * adapter.HIGHRES_SHAPE_HW[1] + ids[:, 0]
    require(len(np.unique(linear)) == len(linear), "COMMON_SUPPORT_DUPLICATE", "common support pixel identities must be unique")
    return ids


def derive_candidate_relative_scale(
    faro_depth_mm: np.ndarray,
    candidate_depth_m: np.ndarray,
    common_point_ids_uv: Any,
    *,
    physical_frame_id: str,
    query_id: str,
    faro_factor_frame_sha256: str,
    candidate_factor_frame_sha256: str,
) -> dict[str, Any]:
    """Compute the frozen robust FARO/candidate log-scale correction."""

    faro = np.asarray(faro_depth_mm)
    candidate = np.asarray(candidate_depth_m)
    require(faro.shape == adapter.HIGHRES_SHAPE_HW and faro.dtype.kind in "iuf" and bool(np.all(np.isfinite(faro))), "FARO_DEPTH_INVALID", "FARO depth must be a finite 1440x1920 millimetre raster")
    require(candidate.shape == adapter.HIGHRES_SHAPE_HW and candidate.dtype.kind in "iuf" and bool(np.all(np.isfinite(candidate))), "CANDIDATE_DEPTH_INVALID", "candidate depth must be a finite 1440x1920 metre raster")
    require(isinstance(physical_frame_id, str) and bool(physical_frame_id) and isinstance(query_id, str) and bool(query_id), "SCALE_IDENTITY_INVALID", "frame/query identity is required")
    for name, value in (("faro_factor_frame_sha256", faro_factor_frame_sha256), ("candidate_factor_frame_sha256", candidate_factor_frame_sha256)):
        require(isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value), "SCALE_IDENTITY_INVALID", "factor frame hash is malformed", field=name)
    ids = _common_ids(common_point_ids_uv)
    u, v = ids[:, 0], ids[:, 1]
    faro_z = faro[v, u].astype(np.float64) / 1000.0
    candidate_z = candidate[v, u].astype(np.float64)
    valid = (
        np.isfinite(faro_z)
        & np.isfinite(candidate_z)
        & (faro_z >= adapter.DEPTH_RANGE_M[0])
        & (faro_z <= adapter.DEPTH_RANGE_M[1])
        & (candidate_z >= adapter.DEPTH_RANGE_M[0])
        & (candidate_z <= adapter.DEPTH_RANGE_M[1])
    )
    pair_count = int(np.sum(valid))
    require(pair_count >= MINIMUM_SCALE_PAIRS, "SCALE_COMMON_SUPPORT_INSUFFICIENT", "too few valid FARO/candidate pairs on frozen common support", observed=pair_count)
    residuals = np.log(faro_z[valid] / candidate_z[valid])
    require(bool(np.all(np.isfinite(residuals))), "SCALE_RESIDUAL_INVALID", "candidate-relative log-scale residual is non-finite")
    correction = float(np.median(residuals))
    require(math.isfinite(correction) and -50.0 <= correction <= 50.0, "SCALE_RANGE_INVALID", "candidate-relative scale correction is outside reducer bounds")
    metric_scale = math.exp(correction)
    require(math.isfinite(metric_scale) and metric_scale > 0.0, "SCALE_RANGE_INVALID", "candidate-relative metric scale is invalid")
    return _seal(
        {
            "schema": CANDIDATE_RELATIVE_SCALE_SCHEMA,
            "physical_frame_id": physical_frame_id,
            "query_id": query_id,
            "faro_factor_frame_sha256": faro_factor_frame_sha256.upper(),
            "candidate_factor_frame_sha256": candidate_factor_frame_sha256.upper(),
            "faro_depth_array_sha256": adapter.canonical_sha256(faro),
            "candidate_depth_array_sha256": adapter.canonical_sha256(candidate),
            "common_support_point_ids_sha256": adapter.canonical_sha256(ids),
            "common_support_point_count": int(len(ids)),
            "valid_pair_count": pair_count,
            "estimator": adapter.CANDIDATE_SCALE_ESTIMATOR,
            "value_kind": SCALE_VALUE_KIND,
            "log_metric_scale": correction,
            "metric_scale": metric_scale,
            "truth_alignment_used_for_candidate_generation": False,
            "computed_only_after_candidate_output_sealed": True,
        }
    )


def oracle_representation_for_arm(arm: str) -> str:
    require(arm in adapter.ARMS, "FACTORIAL_ARM_INVALID", "unknown factorial arm", arm=arm)
    selected = () if arm == "NONE" else tuple(arm.split("_"))
    return "CANDIDATE_GAUGE" if "SCALE" in selected else "ABSOLUTE_METRIC"


def _divide_if_numeric(value: Any, scale: float) -> Any:
    return float(value) / scale if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value)) else value


def reexpress_faro_blocks_in_candidate_gauge(
    faro_blocks: Mapping[str, Any],
    scale_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Express FARO S/P/B in candidate shape units for SCALE-containing arms."""

    require(tuple(faro_blocks) == adapter.FACTOR_NAMES, "FARO_BLOCK_SET_INVALID", "FARO blocks must be ordered SCALE, SUPPORT, BOUNDARY")
    require(scale_record.get("schema") == CANDIDATE_RELATIVE_SCALE_SCHEMA and scale_record.get("value_kind") == SCALE_VALUE_KIND, "SCALE_RECORD_INVALID", "candidate-relative scale record is invalid")
    scale = float(scale_record.get("metric_scale", float("nan")))
    log_scale = float(scale_record.get("log_metric_scale", float("nan")))
    require(math.isfinite(scale) and scale > 0.0 and math.isfinite(log_scale) and abs(math.log(scale) - log_scale) <= 1e-12, "SCALE_RECORD_INVALID", "candidate-relative scale/log-scale mismatch")
    blocks = copy.deepcopy(dict(faro_blocks))
    blocks["SCALE"]["value"] = {"log_metric_scale": log_scale, "value_kind": SCALE_VALUE_KIND}
    blocks["SCALE"]["validity"] = {"valid": True, "model_independent": False}

    support_value = blocks["SUPPORT"]["value"]
    support_value["camera_height_shape_m"] = float(support_value["camera_height_shape_m"]) / scale
    support_validity = blocks["SUPPORT"]["validity"]
    support_validity["observed_forward_shape_m"] = _divide_if_numeric(support_validity.get("observed_forward_shape_m"), scale)
    support_validity["median_residual_m"] = _divide_if_numeric(support_validity.get("median_residual_m"), scale)
    support_uncertainty = blocks["SUPPORT"]["uncertainty"]
    support_uncertainty["height_q95_shape_m"] = _divide_if_numeric(support_uncertainty.get("height_q95_shape_m"), scale)

    boundary_value = blocks["BOUNDARY"]["value"]
    points = np.asarray(boundary_value["boundary_points_shape_camera_xyz"], dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 3 and bool(np.all(np.isfinite(points))), "BOUNDARY_VALUE_INVALID", "FARO boundary points are invalid")
    boundary_value["boundary_points_shape_camera_xyz"] = np.ascontiguousarray(points / scale, dtype=np.float64)
    boundary_uncertainty = blocks["BOUNDARY"]["uncertainty"]
    boundary_uncertainty["localization_q95_shape_m"] = _divide_if_numeric(boundary_uncertainty.get("localization_q95_shape_m"), scale)

    reconstructed_height = scale * float(blocks["SUPPORT"]["value"]["camera_height_shape_m"])
    original_height = float(faro_blocks["SUPPORT"]["value"]["camera_height_shape_m"])
    require(abs(reconstructed_height - original_height) <= 1e-10 * max(1.0, abs(original_height)), "GAUGE_RECONSTRUCTION_MISMATCH", "support height does not reconstruct FARO metric geometry")
    reconstructed_points = scale * np.asarray(blocks["BOUNDARY"]["value"]["boundary_points_shape_camera_xyz"], dtype=np.float64)
    require(bool(np.allclose(reconstructed_points, points, rtol=1e-12, atol=1e-12)), "GAUGE_RECONSTRUCTION_MISMATCH", "boundary points do not reconstruct FARO metric geometry")
    # Keep dense point-id/XYZ arrays as arrays. Canonical JSON represents an
    # ndarray by its receipt, so _canonical_copy would discard reducer inputs.
    return blocks


def build_candidate_gauge_oracle_frame(
    faro_depth_mm: np.ndarray,
    candidate_depth_m: np.ndarray,
    truth_factor_frame: Mapping[str, Any],
    candidate_factor_frame: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the executor-owned FARO oracle used by SCALE-containing arms."""

    truth = adapter.validate_query_factor_frame(dict(truth_factor_frame))
    candidate = adapter.validate_query_factor_frame(dict(candidate_factor_frame))
    require(truth["factor_identity"]["origin"] == "FARO_TRUTH", "FARO_ORACLE_ORIGIN_INVALID", "candidate-gauge oracle requires a genuine FARO truth parent")
    require(candidate["factor_identity"]["origin"] == "CANDIDATE_DEPTH_EXTRACTOR", "CANDIDATE_ORIGIN_INVALID", "candidate-gauge oracle requires a genuine candidate extractor parent")
    for key in ("physical_frame_id", "query_id", "source_frame_receipt_sha256", "query_receipt_sha256", "max_source_timestamp_ns"):
        require(truth[key] == candidate[key], "FACTOR_IDENTITY_MISMATCH", "truth/candidate factor identities differ", field=key)
    require(truth["base_geometry"]["content_sha256"] == candidate["base_geometry"]["content_sha256"], "BASE_GEOMETRY_MISMATCH", "truth/candidate immutable base geometry differs")
    scale = derive_candidate_relative_scale(
        faro_depth_mm,
        candidate_depth_m,
        truth["base_geometry"]["common_point_ids_uv"],
        physical_frame_id=truth["physical_frame_id"],
        query_id=truth["query_id"],
        faro_factor_frame_sha256=truth["content_sha256"],
        candidate_factor_frame_sha256=candidate["content_sha256"],
    )
    blocks = reexpress_faro_blocks_in_candidate_gauge(truth["blocks"], scale)
    frame = copy.deepcopy(truth)
    frame["blocks"] = blocks
    frame["factor_identity"] = {
        "origin": CANDIDATE_GAUGE_ORIGIN,
        "candidate_id": CANDIDATE_GAUGE_ORACLE_ID,
        "truth_only": False,
        "oracle_only": True,
        "uncertainty_complete": all(blocks[name]["uncertainty"]["valid"] for name in adapter.FACTOR_NAMES),
        "faro_truth_factor_frame_sha256": truth["content_sha256"],
        "candidate_factor_frame_sha256": candidate["content_sha256"],
        "candidate_output_receipt": copy.deepcopy(candidate["factor_identity"]["candidate_output_receipt"]),
        "candidate_output_receipt_sha256": candidate["factor_identity"]["candidate_output_receipt_sha256"],
        "candidate_relative_scale_record": scale,
        "candidate_relative_scale_record_sha256": scale["content_sha256"],
        "faro_depth_array_sha256": adapter.canonical_sha256(np.asarray(faro_depth_mm)),
        "faro_geometry_sha256": truth["factor_identity"]["faro_geometry_sha256"],
        "candidate_depth_array_sha256": adapter.canonical_sha256(np.asarray(candidate_depth_m)),
        "representation": "FARO_METRIC_GEOMETRY_EXPRESSED_IN_CANDIDATE_SHAPE_GAUGE",
    }
    frame.pop("content_sha256", None)
    output = adapter._seal(frame)
    adapter.validate_query_factor_frame(output)
    return output, scale


__all__ = [
    "CANDIDATE_GAUGE_ORACLE_ID",
    "CANDIDATE_GAUGE_ORIGIN",
    "CANDIDATE_RELATIVE_SCALE_SCHEMA",
    "FactorHeadroomError",
    "MINIMUM_SCALE_PAIRS",
    "SCALE_VALUE_KIND",
    "build_candidate_gauge_oracle_frame",
    "derive_candidate_relative_scale",
    "oracle_representation_for_arm",
    "reexpress_faro_blocks_in_candidate_gauge",
]
