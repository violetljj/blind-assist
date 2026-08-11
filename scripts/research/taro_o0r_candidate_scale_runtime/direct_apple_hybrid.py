#!/usr/bin/env python3
"""Threshold-free direct-Apple SUPPORT with deterministic baseline fallback.

This module does not learn a selector.  It consumes the sealed R4 comparison
records and applies one source-observable rule: use the direct Apple SUPPORT
branch when its source plane exists; otherwise retain the original R1 baseline
branch.  Evaluation-only extraction status is never a selection input.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_full_cohort as r4
from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_support as r3
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


QUERY_SCHEMA = "blindassist.taro.o0r.direct_apple_hybrid_query.v1"
SUMMARY_SCHEMA = "blindassist.taro.o0r.direct_apple_hybrid_summary.v1"
ANALYSIS_KIND = "POST_HOC_THRESHOLD_FREE_DIRECT_APPLE_WITH_BASELINE_FALLBACK"
POLICY_ID = "DIRECT_WHEN_SOURCE_SUPPORT_AVAILABLE_ELSE_R1_BASELINE_V1"
CLAIM_CEILING = {
    "scope": "SEALED_R4_171_FRAME_1539_QUERY_ARKITSCENES_TRAIN_LANDSCAPE",
    "use": "POST_HOC_DESCRIPTIVE_ZERO_PARAMETER_HYBRID_REPLAY",
    "same_cohort_retrospective_evaluation": True,
    "training": False,
    "threshold_or_pass_fail_decision": False,
    "excluded_claims": ["FRESH_CONFIRMATION", "FORMAL_O0R_PASS", "DEPLOYMENT", "PRODUCT", "SAFETY"],
}


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _canonical(value: Any) -> Any:
    return json.loads(adapter.canonical_json_bytes(value).decode("utf-8"))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = _canonical(dict(value))
    r3._require("content_sha256" not in output, "R4A_SEAL_COLLISION", "hybrid payload already contains a seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _effects(baseline: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "extraction_recovered_vs_baseline": not baseline["extraction_evaluable"] and selected["extraction_evaluable"],
        "extraction_lost_vs_baseline": baseline["extraction_evaluable"] and not selected["extraction_evaluable"],
        "support_no_regret_vs_baseline": r4._support_no_regret(baseline, selected),
        "height_error_reduction_vs_baseline_m": source_factor._difference(
            baseline, selected, "support", "height_abs_error_m"
        ),
        "normal_error_reduction_vs_baseline_rad": source_factor._difference(
            baseline, selected, "support", "normal_angular_error_rad"
        ),
        "boundary_jaccard_increase_vs_baseline": source_factor._difference(
            baseline, selected, "boundary", "point_id_jaccard", higher_is_better=True
        ),
        "boundary_xyz_error_reduction_vs_baseline_m": source_factor._difference(
            baseline, selected, "boundary", "xyz_median_error_m"
        ),
        "query_error_reduction_vs_baseline_m": source_factor._difference(
            baseline, selected, "query_point_clearance", "abs_error_m"
        ),
        "boundary_evaluability_recovered_vs_baseline": not baseline["boundary"]["evaluable"]
        and selected["boundary"]["evaluable"],
        "boundary_evaluability_lost_vs_baseline": baseline["boundary"]["evaluable"]
        and not selected["boundary"]["evaluable"],
        "query_knownness_recovered_vs_baseline": not baseline["query_point_clearance"]["evaluable"]
        and selected["query_point_clearance"]["evaluable"],
        "query_knownness_lost_vs_baseline": baseline["query_point_clearance"]["evaluable"]
        and not selected["query_point_clearance"]["evaluable"],
    }


def build_hybrid_query_record(r4_query_record: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the zero-parameter policy to one externally verified R4 record."""

    source = r4.validate_full_cohort_query_record(dict(r4_query_record))
    baseline = _canonical(source["comparators"]["r1_baseline"])
    direct = _canonical(source["direct_apple_support"])
    source_available = bool(source["source_support_available"])
    use_direct = source_available
    selected = direct if use_direct else baseline
    selected_mode = "DIRECT_APPLE_SUPPORT" if use_direct else "R1_BASELINE_FALLBACK"
    selected_reason = (
        "SOURCE_SUPPORT_AVAILABLE"
        if use_direct
        else "DIRECT_SOURCE_OR_EXTRACTION_UNAVAILABLE"
    )
    return validate_hybrid_query_record(
        _seal(
            {
                "schema": QUERY_SCHEMA,
                "analysis_kind": ANALYSIS_KIND,
                "claim_ceiling": CLAIM_CEILING,
                "policy_id": POLICY_ID,
                "parent_id": source["parent_id"],
                "physical_frame_id": source["physical_frame_id"],
                "query_id": source["query_id"],
                "r4_query_record_sha256": source["content_sha256"],
                "query_receipt_sha256": source["query_receipt_sha256"],
                "candidate_binding_sha256": source["candidate_binding_sha256"],
                "geometry_binding": source["geometry_binding"],
                "selection_inputs": {
                    "source_support_available": source_available,
                },
                "selected_mode": selected_mode,
                "selected_reason": selected_reason,
                "baseline": baseline,
                "selected_result": selected,
                "baseline_result_sha256": adapter.canonical_sha256(baseline),
                "selected_result_sha256": adapter.canonical_sha256(selected),
                "effects": _canonical(_effects(baseline, selected)),
                "selection_metric_fields_read": [],
                "free_parameter_count": 0,
                "training_applied": False,
                "threshold_or_pass_fail_decision_applied": False,
            }
        ),
        source,
    )


def validate_hybrid_query_record(
    value: Any,
    r4_query_record: Mapping[str, Any],
) -> dict[str, Any]:
    record = r3._validate_seal(value, QUERY_SCHEMA)
    expected_keys = {
        "schema", "analysis_kind", "claim_ceiling", "policy_id", "parent_id",
        "physical_frame_id", "query_id", "r4_query_record_sha256", "query_receipt_sha256",
        "candidate_binding_sha256", "geometry_binding", "selection_inputs", "selected_mode",
        "selected_reason", "baseline", "selected_result", "baseline_result_sha256",
        "selected_result_sha256", "effects", "selection_metric_fields_read",
        "free_parameter_count", "training_applied", "threshold_or_pass_fail_decision_applied",
        "content_sha256",
    }
    r3._require(set(record) == expected_keys, "R4A_RECORD_KEY_SET", "hybrid query fields drift")
    r3._require(
        record["analysis_kind"] == ANALYSIS_KIND
        and record["claim_ceiling"] == CLAIM_CEILING
        and record["policy_id"] == POLICY_ID,
        "R4A_POLICY_DRIFT",
        "hybrid policy or claim drift",
    )
    for field in (
        "r4_query_record_sha256", "query_receipt_sha256", "candidate_binding_sha256",
        "baseline_result_sha256", "selected_result_sha256",
    ):
        r3._hash(record[field], field=field)
    inputs = record["selection_inputs"]
    r3._require(
        isinstance(inputs, dict)
        and set(inputs) == {"source_support_available"}
        and all(isinstance(inputs[key], bool) for key in inputs),
        "R4A_SELECTION_INPUT_INVALID",
        "hybrid selection inputs drift",
    )
    use_direct = inputs["source_support_available"]
    expected_mode = "DIRECT_APPLE_SUPPORT" if use_direct else "R1_BASELINE_FALLBACK"
    expected_reason = (
        "SOURCE_SUPPORT_AVAILABLE"
        if use_direct
        else "DIRECT_SOURCE_OR_EXTRACTION_UNAVAILABLE"
    )
    r3._require(
        record["selected_mode"] == expected_mode and record["selected_reason"] == expected_reason,
        "R4A_SELECTION_DRIFT",
        "hybrid selection does not follow the fixed fallback rule",
    )
    baseline = _canonical(record["baseline"])
    selected = _canonical(record["selected_result"])
    r3._require(
        adapter.canonical_sha256(baseline) == record["baseline_result_sha256"]
        and adapter.canonical_sha256(selected) == record["selected_result_sha256"],
        "R4A_RESULT_BINDING_DRIFT",
        "hybrid mode payload hash drift",
    )
    r3._require(
        record["effects"] == _canonical(_effects(baseline, selected)),
        "R4A_EFFECT_DRIFT",
        "hybrid effects do not rederive",
    )
    r3._require(
        record["selection_metric_fields_read"] == []
        and record["free_parameter_count"] == 0
        and record["training_applied"] is False
        and record["threshold_or_pass_fail_decision_applied"] is False,
        "R4A_OUTCOME_SELECTION_FORBIDDEN",
        "hybrid replay used training, metrics, or a threshold for selection",
    )
    source = r4.validate_full_cohort_query_record(dict(r4_query_record))
    direct = _canonical(source["direct_apple_support"])
    expected_inputs = {
        "source_support_available": bool(source["source_support_available"]),
    }
    expected_selected = direct if expected_inputs["source_support_available"] else _canonical(source["comparators"]["r1_baseline"])
    r3._require(
        record["r4_query_record_sha256"] == source["content_sha256"]
        and (record["parent_id"], record["physical_frame_id"], record["query_id"])
        == (source["parent_id"], source["physical_frame_id"], source["query_id"])
        and record["query_receipt_sha256"] == source["query_receipt_sha256"]
        and record["candidate_binding_sha256"] == source["candidate_binding_sha256"]
        and record["geometry_binding"] == source["geometry_binding"]
        and inputs == expected_inputs
        and baseline == _canonical(source["comparators"]["r1_baseline"])
        and selected == expected_selected,
        "R4A_R4_BINDING_DRIFT",
        "hybrid record differs from its bound R4 query",
    )
    return record


def _parent_macro(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    by_frame: dict[tuple[str, str], list[float]] = defaultdict(list)
    all_frames: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        all_frames[row["parent_id"]].add(row["physical_frame_id"])
        value = row["effects"].get(field)
        if _finite(value):
            by_frame[(row["parent_id"], row["physical_frame_id"])].append(float(value))
    by_parent: dict[str, list[float]] = defaultdict(list)
    for (parent, _), values in by_frame.items():
        by_parent[parent].append(float(np.median(np.asarray(values, dtype=np.float64))))
    parent_values = []
    for parent in sorted(all_frames):
        values = by_parent.get(parent, [])
        median = float(np.median(np.asarray(values, dtype=np.float64))) if values else None
        parent_values.append(
            {
                "parent_id": parent,
                "physical_frame_count": len(all_frames[parent]),
                "paired_frame_count": len(values),
                "median_frame_effect": median,
            }
        )
    usable = [row["median_frame_effect"] for row in parent_values if row["median_frame_effect"] is not None]
    return {
        "parent_values": parent_values,
        "parents_with_metric": len(usable),
        "parents_improved": sum(float(value) > 0.0 for value in usable),
        "parents_neutral": sum(float(value) == 0.0 for value in usable),
        "parents_worsened": sum(float(value) < 0.0 for value in usable),
        "median_of_parent_medians": float(np.median(np.asarray(usable, dtype=np.float64))) if usable else None,
    }


def _coverage(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[row["parent_id"]].append(row)
    values = []
    for parent in sorted(by_parent):
        items = by_parent[parent]
        count = sum(bool(row[field]["extraction_evaluable"]) for row in items)
        values.append(
            {
                "parent_id": parent,
                "query_count": len(items),
                "evaluable_query_count": count,
                "query_coverage": count / float(len(items)),
            }
        )
    return {
        "parent_values": values,
        "median_across_parents": float(
            np.median(np.asarray([item["query_coverage"] for item in values], dtype=np.float64))
        ),
    }


def summarize_hybrid(
    records: Sequence[Mapping[str, Any]],
    r4_query_records: Sequence[Mapping[str, Any]],
    *,
    expected_query_count: int = 1539,
    expected_frame_count: int = 171,
    expected_parent_count: int = 16,
) -> dict[str, Any]:
    r3._require(
        isinstance(records, Sequence)
        and isinstance(r4_query_records, Sequence)
        and len(records) == len(r4_query_records),
        "R4A_R4_SEQUENCE_DRIFT",
        "hybrid and R4 query sequences differ in length",
    )
    rows = [
        validate_hybrid_query_record(dict(row), dict(source))
        for row, source in zip(records, r4_query_records)
    ]
    r3._require(
        len(rows) == expected_query_count
        and len({(row["physical_frame_id"], row["query_id"]) for row in rows}) == expected_query_count
        and len({row["physical_frame_id"] for row in rows}) == expected_frame_count
        and len({row["parent_id"] for row in rows}) == expected_parent_count
        and len({row["r4_query_record_sha256"] for row in rows}) == expected_query_count,
        "R4A_COHORT_DRIFT",
        "hybrid query cohort drift",
    )
    effects = [row["effects"] for row in rows]
    height = _parent_macro(rows, "height_error_reduction_vs_baseline_m")
    normal = _parent_macro(rows, "normal_error_reduction_vs_baseline_rad")
    height_by_parent = {row["parent_id"]: row["median_frame_effect"] for row in height["parent_values"]}
    normal_by_parent = {row["parent_id"]: row["median_frame_effect"] for row in normal["parent_values"]}
    joint = sum(
        height_by_parent[parent] is not None
        and normal_by_parent[parent] is not None
        and float(height_by_parent[parent]) > 0.0
        and float(normal_by_parent[parent]) > 0.0
        for parent in sorted(height_by_parent)
    )
    return _seal(
        {
            "schema": SUMMARY_SCHEMA,
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "policy_id": POLICY_ID,
            "physical_frame_count": expected_frame_count,
            "query_record_count": expected_query_count,
            "parent_count": expected_parent_count,
            "direct_selected_query_count": sum(row["selected_mode"] == "DIRECT_APPLE_SUPPORT" for row in rows),
            "baseline_fallback_query_count": sum(row["selected_mode"] == "R1_BASELINE_FALLBACK" for row in rows),
            "baseline_extraction_evaluable_query_count": sum(row["baseline"]["extraction_evaluable"] for row in rows),
            "hybrid_extraction_evaluable_query_count": sum(row["selected_result"]["extraction_evaluable"] for row in rows),
            "fallback_saved_evaluable_query_count": sum(
                row["selected_mode"] == "R1_BASELINE_FALLBACK" and row["baseline"]["extraction_evaluable"]
                for row in rows
            ),
            "hybrid_boundary_evaluable_query_count": sum(row["selected_result"]["boundary"]["evaluable"] for row in rows),
            "hybrid_query_known_count": sum(row["selected_result"]["query_point_clearance"]["evaluable"] for row in rows),
            "support_no_regret_vs_baseline_query_count": sum(effect["support_no_regret_vs_baseline"] for effect in effects),
            "height_improved_vs_baseline_query_count": sum(
                _finite(effect["height_error_reduction_vs_baseline_m"])
                and float(effect["height_error_reduction_vs_baseline_m"]) > 0.0
                for effect in effects
            ),
            "normal_improved_vs_baseline_query_count": sum(
                _finite(effect["normal_error_reduction_vs_baseline_rad"])
                and float(effect["normal_error_reduction_vs_baseline_rad"]) > 0.0
                for effect in effects
            ),
            "effects_counts": {
                key: sum(bool(effect[key]) for effect in effects)
                for key in (
                    "extraction_recovered_vs_baseline", "extraction_lost_vs_baseline",
                    "boundary_evaluability_recovered_vs_baseline", "boundary_evaluability_lost_vs_baseline",
                    "query_knownness_recovered_vs_baseline", "query_knownness_lost_vs_baseline",
                )
            },
            "height_error_reduction_vs_baseline_parent_macro_m": height,
            "normal_error_reduction_vs_baseline_parent_macro_rad": normal,
            "parents_jointly_positive_height_and_normal": joint,
            "support_coverage_parent_macro": {
                "r1_baseline": _coverage(rows, "baseline"),
                "hybrid": _coverage(rows, "selected_result"),
            },
            "selection_metric_fields_read": [],
            "free_parameter_count": 0,
            "training_applied": False,
            "threshold_or_pass_fail_decision_applied": False,
        }
    )


__all__ = [
    "ANALYSIS_KIND", "CLAIM_CEILING", "POLICY_ID", "build_hybrid_query_record",
    "summarize_hybrid", "validate_hybrid_query_record",
]
