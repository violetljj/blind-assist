#!/usr/bin/env python3
"""Full-cohort descriptive replay for direct AppleDepth SUPPORT."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_support as r3
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


PLANE_SCHEMA = "blindassist.taro.o0r.direct_apple_support_full_cohort_plane.v1"
QUERY_SCHEMA = "blindassist.taro.o0r.direct_apple_support_full_cohort_query.v1"
FAILURE_SCHEMA = "blindassist.taro.o0r.direct_apple_support_full_cohort_source_failure.v1"
SUMMARY_SCHEMA = "blindassist.taro.o0r.direct_apple_support_full_cohort_summary.v1"
ANALYSIS_KIND = "POST_HOC_DIRECT_APPLE_SUPPORT_FULL_COHORT_CANARY"
METHOD_ID = "R3_DIRECT_APPLE_SUPPORT_FIXED_METHOD_FULL_171_FRAME_REPLAY_V1"
CLAIM_CEILING = {
    "scope": "ALL_171_EXISTING_O0R_EVAL_TRUTH_FRAMES_LOCKED_ARKITSCENES_TRAIN_LANDSCAPE",
    "use": "POST_HOC_DESCRIPTIVE_DIRECT_APPLE_SUPPORT_FULL_COHORT_MAP",
    "retrospective_cohort": True,
    "threshold_or_pass_fail_decision": False,
    "excluded_claims": ["RGB_ONLY_CAPABILITY", "FORMAL_O0R_PASS", "DEPLOYMENT", "PRODUCT", "SAFETY"],
}


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _canonical(value: Any) -> Any:
    return json.loads(adapter.canonical_json_bytes(value).decode("utf-8"))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = _canonical(dict(value))
    r3._require("content_sha256" not in output, "R4_SEAL_COLLISION", "payload already contains a seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    return r3._validate_seal(value, schema)


def derive_full_cohort_plane(
    prepared: source_factor.PreparedSourceCandidate,
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
    direct_source_receipt: Mapping[str, Any],
    source_scale_record: Mapping[str, Any],
) -> r3.DirectAppleSupportPlane:
    """Apply the exact R3 source method under a full-cohort claim ceiling."""

    legacy = r3.derive_direct_apple_support_plane(
        prepared,
        apple_depth_mm,
        confidence,
        direct_source_receipt,
        source_scale_record,
    )
    payload = copy.deepcopy(legacy.record)
    payload.pop("content_sha256")
    payload["schema"] = PLANE_SCHEMA
    payload["analysis_kind"] = ANALYSIS_KIND
    payload["claim_ceiling"] = CLAIM_CEILING
    payload["method_id"] = METHOD_ID
    payload["fixed_r3_method_id"] = r3.METHOD_ID
    payload["cohort_selection_used_for_plane"] = False
    return load_full_cohort_plane(_seal(payload))


def validate_full_cohort_plane_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, PLANE_SCHEMA)
    r3._require(
        record.get("analysis_kind") == ANALYSIS_KIND
        and record.get("claim_ceiling") == CLAIM_CEILING
        and record.get("method_id") == METHOD_ID
        and record.get("fixed_r3_method_id") == r3.METHOD_ID
        and record.get("cohort_selection_used_for_plane") is False,
        "R4_METHOD_DRIFT",
        "R4 plane method/claim drift",
    )
    legacy = copy.deepcopy(record)
    legacy.pop("content_sha256")
    legacy.pop("fixed_r3_method_id")
    legacy.pop("cohort_selection_used_for_plane")
    legacy["schema"] = r3.PLANE_SCHEMA
    legacy["analysis_kind"] = r3.ANALYSIS_KIND
    legacy["claim_ceiling"] = r3.CLAIM_CEILING
    legacy["method_id"] = r3.METHOD_ID
    r3.validate_direct_apple_support_plane_record(r3._seal(legacy))
    return record


def load_full_cohort_plane(value: Any) -> r3.DirectAppleSupportPlane:
    record = validate_full_cohort_plane_record(value)
    support = record["apple_support"]
    return r3.DirectAppleSupportPlane(
        parent_id=str(record["parent_id"]),
        physical_frame_id=str(record["physical_frame_id"]),
        direct_source_receipt_sha256=str(record["direct_source_receipt_sha256"]),
        source_scale_record_sha256=str(record["source_scale_record_sha256"]),
        candidate_binding_sha256=str(record["candidate_binding_sha256"]),
        anchored_depth_array_sha256=str(record["anchored_candidate_depth_array_sha256"]),
        intrinsics_highres_sha256=str(record["intrinsics_highres_sha256"]),
        gravity_up_camera_xyz_sha256=str(record["gravity_up_camera_xyz_sha256"]),
        normal_camera_xyz=r3._immutable(support["normal_camera_xyz"], np.float64),
        camera_height_m=float(support["camera_height_m"]),
        support_count=int(support["support_count"]),
        support_fraction=float(support["support_fraction"]),
        slope_degrees=float(support["slope_degrees"]),
        median_residual_m=float(support["median_residual_m"]),
        record=record,
        content_sha256=str(record["content_sha256"]),
    )


def build_source_failure_record(
    parent_id: str,
    direct_source_receipt: Mapping[str, Any],
    prepared: source_factor.PreparedSourceCandidate,
    error: Exception,
) -> dict[str, Any]:
    source = r3.validate_direct_apple_source_receipt(dict(direct_source_receipt))
    r3._require(parent_id == source["parent_id"] == prepared.parent_id, "R4_IDENTITY_MISMATCH", "failure identity drift")
    return validate_source_failure_record(
        _seal(
            {
                "schema": FAILURE_SCHEMA,
                "analysis_kind": ANALYSIS_KIND,
                "claim_ceiling": CLAIM_CEILING,
                "method_id": METHOD_ID,
                "parent_id": parent_id,
                "physical_frame_id": source["physical_frame_id"],
                "direct_source_receipt_sha256": source["content_sha256"],
                "source_scale_record_sha256": prepared.source_scale_record_sha256,
                "candidate_binding_sha256": prepared.candidate_binding_sha256,
                "error_code": str(getattr(error, "code", type(error).__name__)),
                "message": str(error),
                "compact_truth_read": False,
                "faro_payload_read": False,
                "query_receipt_read": False,
                "computed_before_truth_join": True,
                "unknown_preserved": True,
            }
        )
    )


def validate_source_failure_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, FAILURE_SCHEMA)
    r3._require(
        record.get("analysis_kind") == ANALYSIS_KIND
        and record.get("claim_ceiling") == CLAIM_CEILING
        and record.get("method_id") == METHOD_ID,
        "R4_METHOD_DRIFT",
        "R4 failure method/claim drift",
    )
    r3._require(
        record.get("compact_truth_read") is False
        and record.get("faro_payload_read") is False
        and record.get("query_receipt_read") is False
        and record.get("computed_before_truth_join") is True
        and record.get("unknown_preserved") is True,
        "R4_SOURCE_FIREWALL_BREACH",
        "R4 source failure crossed truth firewall",
    )
    for field in ("direct_source_receipt_sha256", "source_scale_record_sha256", "candidate_binding_sha256"):
        r3._hash(record.get(field), field=field)
    r3._require(isinstance(record.get("error_code"), str) and bool(record["error_code"]), "R4_FAILURE_INVALID", "failure code missing")
    return record


def _support_no_regret(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    return bool(
        reference.get("support", {}).get("evaluable")
        and candidate.get("support", {}).get("evaluable")
        and float(candidate["support"]["height_abs_error_m"]) <= float(reference["support"]["height_abs_error_m"])
        and float(candidate["support"]["normal_angular_error_rad"]) <= float(reference["support"]["normal_angular_error_rad"])
    )


def _effects(baseline: Mapping[str, Any], anchored: Mapping[str, Any], direct: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "extraction_recovered_vs_baseline": not baseline["extraction_evaluable"] and direct["extraction_evaluable"],
        "extraction_lost_vs_baseline": baseline["extraction_evaluable"] and not direct["extraction_evaluable"],
        "extraction_recovered_vs_source_anchored": not anchored["extraction_evaluable"] and direct["extraction_evaluable"],
        "extraction_lost_vs_source_anchored": anchored["extraction_evaluable"] and not direct["extraction_evaluable"],
        "support_no_regret_vs_baseline": _support_no_regret(baseline, direct),
        "support_no_regret_vs_source_anchored": _support_no_regret(anchored, direct),
        "height_error_reduction_vs_baseline_m": source_factor._difference(baseline, direct, "support", "height_abs_error_m"),
        "normal_error_reduction_vs_baseline_rad": source_factor._difference(baseline, direct, "support", "normal_angular_error_rad"),
        "height_error_reduction_vs_source_anchored_m": source_factor._difference(anchored, direct, "support", "height_abs_error_m"),
        "normal_error_reduction_vs_source_anchored_rad": source_factor._difference(anchored, direct, "support", "normal_angular_error_rad"),
        "boundary_jaccard_increase_vs_baseline": source_factor._difference(baseline, direct, "boundary", "point_id_jaccard", higher_is_better=True),
        "boundary_xyz_error_reduction_vs_baseline_m": source_factor._difference(baseline, direct, "boundary", "xyz_median_error_m"),
        "query_error_reduction_vs_baseline_m": source_factor._difference(baseline, direct, "query_point_clearance", "abs_error_m"),
        "boundary_evaluability_recovered_vs_baseline": not baseline["boundary"]["evaluable"] and direct["boundary"]["evaluable"],
        "boundary_evaluability_lost_vs_baseline": baseline["boundary"]["evaluable"] and not direct["boundary"]["evaluable"],
        "query_knownness_recovered_vs_baseline": not baseline["query_point_clearance"]["evaluable"] and direct["query_point_clearance"]["evaluable"],
        "query_knownness_lost_vs_baseline": baseline["query_point_clearance"]["evaluable"] and not direct["query_point_clearance"]["evaluable"],
    }


def evaluate_full_cohort_query(
    prepared: source_factor.PreparedSourceCandidate,
    intrinsics_highres_3x3: Any,
    gravity_up_camera_xyz: Any,
    base: source_factor.QueryTruthBase,
    plane: r3.DirectAppleSupportPlane | None,
    r1_query_record: Mapping[str, Any],
    *,
    current_faro_geometry_sha256: str,
    compact_faro_geometry_sha256: str,
    source_failure_code: str | None = None,
) -> dict[str, Any]:
    r1_record = source_factor.validate_query_record(dict(r1_query_record))
    current_geometry = r3._hash(current_faro_geometry_sha256, field="current_faro_geometry_sha256")
    compact_geometry = r3._hash(compact_faro_geometry_sha256, field="compact_faro_geometry_sha256")
    r3._require(
        r1_record["physical_frame_id"] == base.physical_frame_id
        and r1_record["query_id"] == base.query_id
        and r1_record["candidate_binding_sha256"] == prepared.candidate_binding_sha256
        and r1_record["source_scale_record_sha256"] == prepared.source_scale_record_sha256
        and r1_record["query_receipt_sha256"] == base.query_receipt["content_sha256"]
        and r1_record["current_common_point_ids_sha256"] == base.common_point_ids_sha256,
        "R4_R1_BINDING_INVALID",
        "R1 query lineage differs from the full-cohort replay",
    )
    r3._require(
        current_geometry == r1_record["current_faro_geometry_sha256"]
        and compact_geometry == r1_record["committed_faro_geometry_sha256"]
        and r1_record["runtime_geometry_matches_r3_commitment"] == (current_geometry == compact_geometry),
        "R4_R1_GEOMETRY_DRIFT",
        "R4 and R1 do not use the same current/committed FARO geometry bindings",
    )
    if plane is None:
        r3._require(isinstance(source_failure_code, str) and bool(source_failure_code), "R4_FAILURE_CODE_MISSING", "source failure code missing")
        direct = source_factor._failed_mode(prepared.anchored_depth_sha256, source_failure_code)
    else:
        r3._require(source_failure_code is None, "R4_FAILURE_CODE_UNEXPECTED", "successful source plane carries failure code")
        try:
            direct = source_factor._mode_result(base, r3._posthoc_extraction(prepared, intrinsics_highres_3x3, gravity_up_camera_xyz, base, plane))
        except r3.DirectAppleSupportError as error:
            direct = source_factor._failed_mode(prepared.anchored_depth_sha256, error.code)
    baseline = _canonical(r1_record["baseline"])
    anchored = _canonical(r1_record["source_anchored"])
    direct = _canonical(direct)
    effects = _canonical(_effects(baseline, anchored, direct))
    return validate_full_cohort_query_record(
        _seal(
            {
                "schema": QUERY_SCHEMA,
                "analysis_kind": ANALYSIS_KIND,
                "claim_ceiling": CLAIM_CEILING,
                "method_id": METHOD_ID,
                "parent_id": prepared.parent_id,
                "physical_frame_id": base.physical_frame_id,
                "query_id": base.query_id,
                "query_receipt_sha256": base.query_receipt["content_sha256"],
                "current_common_point_ids_sha256": base.common_point_ids_sha256,
                "r1_query_record_sha256": r1_record["content_sha256"],
                "full_cohort_plane_sha256": plane.content_sha256 if plane is not None else None,
                "candidate_binding_sha256": prepared.candidate_binding_sha256,
                "source_scale_record_sha256": prepared.source_scale_record_sha256,
                "geometry_binding": {
                    "current_faro_geometry_sha256": current_geometry,
                    "compact_faro_geometry_sha256": compact_geometry,
                    "runtime_geometry_matches_compact_commitment": current_geometry == compact_geometry,
                },
                "source_support_available": plane is not None,
                "comparators": {"r1_baseline": baseline, "r1_source_anchored": anchored},
                "direct_apple_support": direct,
                "effects": effects,
                "faro_used_for_source_support": False,
                "faro_used_for_post_hoc_scoring": True,
                "threshold_or_pass_fail_decision_applied": False,
            }
        )
    )


def validate_full_cohort_query_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, QUERY_SCHEMA)
    r3._require(
        record.get("analysis_kind") == ANALYSIS_KIND
        and record.get("claim_ceiling") == CLAIM_CEILING
        and record.get("method_id") == METHOD_ID,
        "R4_METHOD_DRIFT",
        "R4 query method/claim drift",
    )
    r3._require(
        record.get("faro_used_for_source_support") is False
        and record.get("faro_used_for_post_hoc_scoring") is True
        and record.get("threshold_or_pass_fail_decision_applied") is False,
        "R4_CLAIM_DRIFT",
        "R4 query truth boundary drift",
    )
    for field in ("query_receipt_sha256", "current_common_point_ids_sha256", "r1_query_record_sha256", "candidate_binding_sha256", "source_scale_record_sha256"):
        r3._hash(record.get(field), field=field)
    geometry = record.get("geometry_binding", {})
    r3._require(
        set(geometry) == {
            "current_faro_geometry_sha256",
            "compact_faro_geometry_sha256",
            "runtime_geometry_matches_compact_commitment",
        },
        "R4_GEOMETRY_BINDING_INVALID",
        "R4 geometry binding fields drift",
    )
    current_geometry = r3._hash(geometry.get("current_faro_geometry_sha256"), field="geometry_binding.current_faro_geometry_sha256")
    compact_geometry = r3._hash(geometry.get("compact_faro_geometry_sha256"), field="geometry_binding.compact_faro_geometry_sha256")
    r3._require(
        geometry.get("runtime_geometry_matches_compact_commitment") == (current_geometry == compact_geometry),
        "R4_GEOMETRY_BINDING_INVALID",
        "R4 geometry match flag drift",
    )
    if record.get("source_support_available"):
        r3._hash(record.get("full_cohort_plane_sha256"), field="full_cohort_plane_sha256")
    else:
        r3._require(record.get("full_cohort_plane_sha256") is None, "R4_RECORD_INVALID", "failed support carries a plane hash")
    comparators = record.get("comparators", {})
    r3._require(set(comparators) == {"r1_baseline", "r1_source_anchored"} and isinstance(record.get("direct_apple_support"), dict), "R4_RECORD_INVALID", "comparison modes are malformed")
    expected = _canonical(_effects(comparators["r1_baseline"], comparators["r1_source_anchored"], record["direct_apple_support"]))
    r3._require(record.get("effects") == expected, "R4_EFFECT_DRIFT", "stored full-cohort effects do not rederive")
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
    values = [float(np.median(np.asarray(items, dtype=np.float64))) for items in by_parent.values()]
    return {"parents_with_metric": len(values), "median_of_parent_medians": float(np.median(np.asarray(values, dtype=np.float64))) if values else None}


def _coverage(rows: Sequence[dict[str, Any]], mode: str, factor: str) -> dict[str, Any]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[row["parent_id"]].append(row)
    values = []
    for parent in sorted(by_parent):
        items = by_parent[parent]
        count = sum(bool(row[mode][factor]["evaluable"]) for row in items) if mode == "direct_apple_support" else sum(bool(row["comparators"][mode][factor]["evaluable"]) for row in items)
        values.append({"parent_id": parent, "query_count": len(items), "evaluable_query_count": count, "query_coverage": count / float(len(items))})
    return {"parent_values": values, "median_across_parents": float(np.median(np.asarray([item["query_coverage"] for item in values], dtype=np.float64)))}


def summarize_full_cohort(
    records: Sequence[Mapping[str, Any]],
    source_failures: Sequence[Mapping[str, Any]],
    *,
    expected_query_count: int = 1539,
    expected_frame_count: int = 171,
    expected_parent_count: int = 16,
) -> dict[str, Any]:
    rows = [validate_full_cohort_query_record(dict(row)) for row in records]
    failures = [validate_source_failure_record(dict(row)) for row in source_failures]
    r3._require(
        len(rows) == expected_query_count
        and len({(row["physical_frame_id"], row["query_id"]) for row in rows}) == expected_query_count
        and len({row["physical_frame_id"] for row in rows}) == expected_frame_count
        and len({row["parent_id"] for row in rows}) == expected_parent_count,
        "R4_COHORT_DRIFT",
        "full-cohort query cardinality drift",
    )
    unavailable_frames = {row["physical_frame_id"] for row in rows if not row["source_support_available"]}
    failure_frames = {row["physical_frame_id"] for row in failures}
    r3._require(
        len(failure_frames) == len(failures) and failure_frames == unavailable_frames,
        "R4_SOURCE_FAILURE_COHORT_DRIFT",
        "source failures do not exactly cover unavailable full-cohort frames",
    )
    direct_reasons = Counter(code for row in rows if not row["direct_apple_support"]["extraction_evaluable"] for code in row["direct_apple_support"]["reason_codes"])
    effects = [row["effects"] for row in rows]
    return _seal(
        {
            "schema": SUMMARY_SCHEMA,
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "method_id": METHOD_ID,
            "physical_frame_count": expected_frame_count,
            "query_record_count": expected_query_count,
            "parent_count": expected_parent_count,
            "source_support_frame_count": len({row["physical_frame_id"] for row in rows if row["source_support_available"]}),
            "source_failure_frame_count": len(failures),
            "source_failure_reason_counts": dict(sorted(Counter(row["error_code"] for row in failures).items())),
            "direct_extraction_evaluable_query_count": sum(row["direct_apple_support"]["extraction_evaluable"] for row in rows),
            "baseline_extraction_evaluable_query_count": sum(row["comparators"]["r1_baseline"]["extraction_evaluable"] for row in rows),
            "source_anchored_extraction_evaluable_query_count": sum(row["comparators"]["r1_source_anchored"]["extraction_evaluable"] for row in rows),
            "direct_boundary_evaluable_query_count": sum(row["direct_apple_support"]["boundary"]["evaluable"] for row in rows),
            "direct_query_known_count": sum(row["direct_apple_support"]["query_point_clearance"]["evaluable"] for row in rows),
            "direct_unknown_reason_counts": dict(sorted(direct_reasons.items())),
            "effects_counts": {key: sum(bool(effect[key]) for effect in effects) for key in (
                "extraction_recovered_vs_baseline", "extraction_lost_vs_baseline",
                "extraction_recovered_vs_source_anchored", "extraction_lost_vs_source_anchored",
                "support_no_regret_vs_baseline", "support_no_regret_vs_source_anchored",
                "boundary_evaluability_recovered_vs_baseline", "boundary_evaluability_lost_vs_baseline",
                "query_knownness_recovered_vs_baseline", "query_knownness_lost_vs_baseline",
            )},
            "height_improved_vs_baseline_query_count": sum(_finite(effect["height_error_reduction_vs_baseline_m"]) and float(effect["height_error_reduction_vs_baseline_m"]) > 0.0 for effect in effects),
            "normal_improved_vs_baseline_query_count": sum(_finite(effect["normal_error_reduction_vs_baseline_rad"]) and float(effect["normal_error_reduction_vs_baseline_rad"]) > 0.0 for effect in effects),
            "height_error_reduction_vs_baseline_parent_macro_m": _parent_macro(rows, "height_error_reduction_vs_baseline_m"),
            "normal_error_reduction_vs_baseline_parent_macro_rad": _parent_macro(rows, "normal_error_reduction_vs_baseline_rad"),
            "height_error_reduction_vs_source_anchored_parent_macro_m": _parent_macro(rows, "height_error_reduction_vs_source_anchored_m"),
            "normal_error_reduction_vs_source_anchored_parent_macro_rad": _parent_macro(rows, "normal_error_reduction_vs_source_anchored_rad"),
            "support_coverage_parent_macro": {
                "r1_baseline": _coverage(rows, "r1_baseline", "support"),
                "r1_source_anchored": _coverage(rows, "r1_source_anchored", "support"),
                "direct_apple_support": _coverage(rows, "direct_apple_support", "support"),
            },
            "threshold_or_pass_fail_decision_applied": False,
        }
    )


__all__ = [
    "ANALYSIS_KIND", "CLAIM_CEILING", "METHOD_ID", "build_source_failure_record",
    "derive_full_cohort_plane", "evaluate_full_cohort_query", "load_full_cohort_plane",
    "summarize_full_cohort", "validate_full_cohort_plane_record",
    "validate_full_cohort_query_record", "validate_source_failure_record",
]
