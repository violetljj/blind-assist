#!/usr/bin/env python3
"""Evaluate the frozen 8-arm by 2-mode TARO factor intervention grid."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase import seal_record
from scripts.research.taro_o0r_factor_headroom_runtime.factor_headroom import (
    FactorHeadroomError,
    build_candidate_gauge_oracle_frame,
    oracle_representation_for_arm,
)
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


FRAME_FACTOR_EVALUATION_SCHEMA = "blindassist.taro.o0r.frame_factor_headroom_evaluation.v1"
CRITICAL_STRATA = (
    "orientation",
    "near_decision_boundary_0p10m",
    "appledepth_confidence",
    "range_band_m",
)


class FactorEvaluationError(RuntimeError):
    """Stable fail-closed factor-evaluation error."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FactorEvaluationError(code, message, **context)


def _range_band(range_m: float) -> str:
    require(math.isfinite(range_m) and 0.0 <= range_m <= 6.0, "STRATUM_RANGE_INVALID", "query range lies outside the frozen source domain")
    if range_m < 1.0:
        return "RANGE_0_TO_1_M"
    if range_m < 2.0:
        return "RANGE_1_TO_2_M"
    if range_m < 3.0:
        return "RANGE_2_TO_3_M"
    return "RANGE_3_TO_6_M"


def frozen_query_strata(truth_result: Mapping[str, Any], uncertainty_lookup: Mapping[str, Any]) -> dict[str, Any]:
    truth_value = truth_result.get("value_m")
    truth_known = truth_result.get("knownness", {}).get("known") is True and isinstance(truth_value, (int, float)) and math.isfinite(float(truth_value))
    near = bool(truth_known and min(abs(float(truth_value)), abs(float(truth_value) - adapter.CLEAR_MARGIN_M)) <= 0.10)
    confidence = uncertainty_lookup.get("confidence_value")
    range_m = float(uncertainty_lookup.get("range_m", float("nan")))
    require(isinstance(confidence, int) and not isinstance(confidence, bool) and confidence in (0, 1, 2), "STRATUM_CONFIDENCE_INVALID", "query confidence stratum is invalid")
    return {
        "orientation": "LANDSCAPE",
        "near_decision_boundary_0p10m": near,
        "appledepth_confidence": int(confidence),
        "range_band_m": _range_band(range_m),
    }


def _statistics_row(
    *,
    parent_id: str,
    truth_result: Mapping[str, Any],
    arm: str,
    mode: str,
    result: Mapping[str, Any],
    strata: Mapping[str, Any],
) -> dict[str, Any]:
    truth_known = truth_result.get("knownness", {}).get("known") is True and truth_result.get("value_m") is not None
    known = result.get("knownness", {}).get("known") is True and result.get("value_m") is not None
    interval = result.get("interval_m", {})
    return {
        "parent_id": parent_id,
        "frame_id": str(truth_result["physical_frame_id"]),
        "query_id": str(truth_result["query_id"]),
        "arm": arm,
        "mode": mode,
        "truth_value_m": float(truth_result["value_m"]) if truth_known else None,
        "truth_state": str(truth_result["state"]),
        "truth_known": bool(truth_known),
        "value_m": float(result["value_m"]) if known else None,
        "interval_lower_m": float(interval["lower"]) if known else None,
        "interval_upper_m": float(interval["upper"]) if known else None,
        "state": str(result["state"]),
        "known": bool(known),
        "strata": dict(strata),
    }


def _unknown_result(query: Mapping[str, Any], reason_code: str) -> dict[str, Any]:
    return {
        "schema": adapter.QUERY_RESULT_SCHEMA,
        "reducer_version": adapter.REDUCER_VERSION,
        "physical_frame_id": query["physical_frame_id"],
        "query_id": query["query_id"],
        "factor_frame_sha256": None,
        "max_source_timestamp_ns": query["max_source_timestamp_ns"],
        "value_m": None,
        "uncertainty_m": None,
        "interval_m": {"lower": None, "upper": None},
        "state": "UNKNOWN",
        "knownness": {"known": False, "support_points": 0, "observed_forward_m": None, "local_valid_fraction": None},
        "reason_codes": [reason_code],
    }


def evaluate_admitted_frame(
    decoded_source_frame: Mapping[str, Any],
    verified_truth: Mapping[str, Any],
    candidate_depth_m: np.ndarray,
    candidate_output_receipt: Mapping[str, Any],
    uncertainty_model: Any,
) -> dict[str, Any]:
    """Run all 144 query-arm-mode reductions and return compact rows/lineage."""

    source = adapter._validate_base_receipt(dict(verified_truth["source_frame_receipt"]))
    require(source["source_role"] == "O0R_EVAL_CANDIDATE", "FACTOR_EVAL_SOURCE_ROLE_INVALID", "factor evaluation requires an eval-candidate source")
    require(verified_truth["truth_query_bundle"]["complete_factor_query_truth"] is True, "FACTOR_EVAL_TRUTH_NOT_ADMITTED", "factor evaluation requires complete 9/9 truth")
    faro = np.asarray(decoded_source_frame["highres_faro_depth_mm"])
    candidate = np.asarray(candidate_depth_m)
    queries = list(verified_truth["query_receipts"])
    truth_frames = list(verified_truth["truth_factor_frames"])
    lookups = list(verified_truth["uncertainty_lookups"])
    truth_results = list(verified_truth["truth_query_bundle"]["results"])
    require(len(queries) == len(truth_frames) == len(lookups) == len(truth_results) == 9, "FACTOR_EVAL_QUERY_CARDINALITY", "factor evaluation requires exactly nine aligned queries")

    rows: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    matrix = np.asarray(source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
    for index, (query, truth_frame, lookup, truth_result) in enumerate(zip(queries, truth_frames, lookups, truth_results, strict=True)):
        require(query["query_id"] == truth_frame["query_id"] == lookup["query_id"] == truth_result["query_id"], "FACTOR_EVAL_QUERY_ALIGNMENT", "factor evaluation query inputs are misaligned", query_index=index)
        strata = frozen_query_strata(truth_result, lookup)
        try:
            candidate_frame = adapter.build_candidate_query_factor_frame(
                candidate,
                matrix,
                source["gravity_up_camera_xyz"],
                source,
                query,
                truth_frame["base_geometry"],
                uncertainty_model,
                dict(candidate_output_receipt),
                confidence_value=lookup["confidence_value"],
                range_m=lookup["range_m"],
            )
            gauge_oracle, scale_record = build_candidate_gauge_oracle_frame(faro, candidate, truth_frame, candidate_frame)
        except (adapter.AdapterError, FactorEvaluationError, FactorHeadroomError) as error:
            code = getattr(error, "code", type(error).__name__)
            failures.append({"query_index": index, "query_id": query["query_id"], "error_code": code})
            for mode in adapter.ORACLE_MODES:
                for arm in adapter.ARMS:
                    unknown = _unknown_result(query, str(code))
                    rows.append(_statistics_row(parent_id=source["parent_id"], truth_result=truth_result, arm=arm, mode=mode, result=unknown, strata=strata))
                    query_records.append({"query_id": query["query_id"], "arm": arm, "mode": mode, "result": unknown, "lineage_available": False})
            continue

        factorial_grid = adapter.reduce_factorial_query_grid(
            candidate_frame,
            truth_frame,
            gauge_oracle,
            query,
        )
        require(len(factorial_grid) == len(adapter.ARMS) * len(adapter.ORACLE_MODES), "FACTOR_EVAL_GRID_CARDINALITY", "batch reducer returned the wrong arm/mode cardinality")
        expected_grid_order = [(arm, mode) for mode in adapter.ORACLE_MODES for arm in adapter.ARMS]
        require(
            [(item.get("arm"), item.get("mode")) if isinstance(item, dict) else (None, None) for item in factorial_grid] == expected_grid_order,
            "FACTOR_EVAL_GRID_ORDER",
            "batch reducer arm/mode order drifted",
        )
        for item in factorial_grid:
            require(
                isinstance(item, dict)
                and set(item)
                == {
                    "arm",
                    "mode",
                    "oracle_representation",
                    "baseline_factor_frame_sha256",
                    "oracle_factor_frame_sha256",
                    "injected_factor_frame_sha256",
                    "result",
                },
                "FACTOR_EVAL_GRID_ROW_INVALID",
                "batch reducer returned a malformed row",
            )
            arm, mode = item["arm"], item["mode"]
            representation = oracle_representation_for_arm(arm)
            expected_oracle_sha256 = gauge_oracle["content_sha256"] if representation == "CANDIDATE_GAUGE" else truth_frame["content_sha256"]
            require(
                mode in adapter.ORACLE_MODES
                and item["oracle_representation"] == representation
                and item["baseline_factor_frame_sha256"] == candidate_frame["content_sha256"]
                and item["oracle_factor_frame_sha256"] == expected_oracle_sha256,
                "FACTOR_EVAL_GRID_ROW_INVALID",
                "batch reducer lineage differs from its validated parents",
            )
            result = item["result"]
            rows.append(_statistics_row(parent_id=source["parent_id"], truth_result=truth_result, arm=arm, mode=mode, result=result, strata=strata))
            query_records.append(
                {
                    "query_id": query["query_id"],
                    "arm": arm,
                    "mode": mode,
                    "oracle_representation": representation,
                    "candidate_factor_frame_sha256": candidate_frame["content_sha256"],
                    "oracle_factor_frame_sha256": item["oracle_factor_frame_sha256"],
                    "injected_factor_frame_sha256": item["injected_factor_frame_sha256"],
                    "candidate_relative_scale_record_sha256": scale_record["content_sha256"] if representation == "CANDIDATE_GAUGE" else None,
                    "result": result,
                    "lineage_available": True,
                }
            )
    require(len(rows) == len(query_records) == 9 * len(adapter.ARMS) * len(adapter.ORACLE_MODES), "FACTOR_EVAL_OUTPUT_CARDINALITY", "factor evaluation did not emit the exact 144 rows")
    compact = seal_record(
        {
            "schema": FRAME_FACTOR_EVALUATION_SCHEMA,
            "parent_id": source["parent_id"],
            "physical_frame_id": source["physical_frame_id"],
            "source_frame_receipt_sha256": source["content_sha256"],
            "candidate_output_receipt_sha256": candidate_output_receipt["content_sha256"],
            "truth_commitment_record_sha256": verified_truth["compact_truth_record_sha256"],
            "query_count": 9,
            "arm_count": len(adapter.ARMS),
            "mode_count": len(adapter.ORACLE_MODES),
            "result_count": len(rows),
            "query_records": query_records,
            "candidate_extractor_failures": failures,
            "dense_factor_arrays_persisted": False,
        }
    )
    return {"statistics_rows": rows, "compact_evaluation": compact}


__all__ = [
    "CRITICAL_STRATA",
    "FRAME_FACTOR_EVALUATION_SCHEMA",
    "FactorEvaluationError",
    "evaluate_admitted_frame",
    "frozen_query_strata",
]
