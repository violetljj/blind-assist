#!/usr/bin/env python3
"""Non-promotable FARO scorer for sealed TARO prospective factor bundles."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale
from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as runtime
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


QUERY_SCORE_SCHEMA = "blindassist.taro.o0r.r6_formation_replay_query_score.v1"
SUMMARY_SCHEMA = "blindassist.taro.o0r.r6_formation_replay_summary.v1"
METRICS = {
    "support": ("normal_angular_error_rad", "height_abs_error_m"),
    "boundary": ("point_id_jaccard", "xyz_median_error_m"),
    "query_clearance": ("abs_error_m",),
}


class FormationReplayError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FormationReplayError(code, message, **context)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "FORMATION_SCORE_SEAL_COLLISION", "formation scorer record already contains a seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _unknown(code: str) -> dict[str, Any]:
    return {"evaluable": False, "reason_codes": [code]}


def _normal_angle(left: Any, right: Any) -> float:
    a = adapter._normalize_vector(left, "FORMATION_NORMAL_INVALID")
    b = adapter._normalize_vector(right, "FORMATION_NORMAL_INVALID")
    return float(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def _boundary_metrics(
    truth_ids: np.ndarray,
    truth_points: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_points: np.ndarray,
) -> dict[str, Any]:
    truth_linear = np.asarray(truth_ids, dtype=np.int64)[:, 1] * adapter.HIGHRES_SHAPE_HW[1] + np.asarray(truth_ids, dtype=np.int64)[:, 0]
    candidate_linear = np.asarray(candidate_ids, dtype=np.int64)[:, 1] * adapter.HIGHRES_SHAPE_HW[1] + np.asarray(candidate_ids, dtype=np.int64)[:, 0]
    _, truth_index, candidate_index = np.intersect1d(truth_linear, candidate_linear, assume_unique=True, return_indices=True)
    intersection = len(truth_index)
    union = len(truth_linear) + len(candidate_linear) - intersection
    xyz_error = None
    if intersection:
        xyz_error = float(
            np.median(
                np.linalg.norm(
                    np.asarray(truth_points, dtype=np.float64)[truth_index]
                    - np.asarray(candidate_points, dtype=np.float64)[candidate_index],
                    axis=1,
                )
            )
        )
    return {
        "evaluable": True,
        "reason_codes": [],
        "truth_point_count": int(len(truth_linear)),
        "candidate_point_count": int(len(candidate_linear)),
        "point_id_intersection_count": int(intersection),
        "point_id_union_count": int(union),
        "point_id_jaccard": float(intersection / union) if union else 1.0,
        "xyz_median_error_m": xyz_error,
    }


def _mode_score(
    *,
    geometry: Any | None,
    plane: Mapping[str, Any] | None,
    query: Mapping[str, Any],
    matrix: np.ndarray,
    truth_geometry: Any,
    truth_plane: Mapping[str, Any],
    failure_code: str,
) -> dict[str, Any]:
    if geometry is None or plane is None:
        unknown = _unknown(failure_code)
        return {"support": dict(unknown), "boundary": dict(unknown), "query_clearance": dict(unknown)}
    support = {
        "evaluable": True,
        "reason_codes": [],
        "normal_angular_error_rad": _normal_angle(truth_plane["normal_camera_xyz"], plane["normal_camera_xyz"]),
        "height_abs_error_m": abs(float(truth_plane["camera_height_m"]) - float(plane["camera_height_m"])),
        "camera_height_m": float(plane["camera_height_m"]),
    }
    candidate_surface, candidate_pixels = runtime._surface(geometry, plane, query)
    truth_surface, truth_pixels = runtime._surface(truth_geometry, truth_plane, query)
    candidate_local = runtime._local_valid_fraction(geometry, matrix, query)
    truth_local = runtime._local_valid_fraction(truth_geometry, matrix, query)
    candidate_query = adapter._query_support_and_boundary(
        candidate_surface,
        candidate_pixels,
        plane["normal_camera_xyz"],
        float(plane["camera_height_m"]),
        query,
    )
    truth_query = adapter._query_support_and_boundary(
        truth_surface,
        truth_pixels,
        truth_plane["normal_camera_xyz"],
        float(truth_plane["camera_height_m"]),
        query,
    )
    if truth_local < adapter.MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION:
        boundary = _unknown("FORMATION_TRUTH_BOUNDARY_LOCAL_VALID_FRACTION_INSUFFICIENT")
    elif candidate_local < adapter.MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION:
        boundary = _unknown("SOURCE_BOUNDARY_LOCAL_VALID_FRACTION_INSUFFICIENT")
    else:
        boundary = _boundary_metrics(
            truth_query["boundary_point_ids_uv"],
            truth_query["boundary_points_shape_camera_xyz"],
            candidate_query["boundary_point_ids_uv"],
            candidate_query["boundary_points_shape_camera_xyz"],
        )
        boundary["truth_local_valid_fraction"] = float(truth_local)
        boundary["candidate_local_valid_fraction"] = float(candidate_local)
    candidate_clearance = source_factor._point_clearance(
        np.asarray(plane["normal_camera_xyz"], dtype=np.float64),
        float(plane["camera_height_m"]),
        candidate_query["boundary_points_shape_camera_xyz"],
        int(candidate_query["query_support_points"]),
        candidate_query["observed_forward_shape_m"],
        candidate_local,
        query,
    )
    truth_clearance = source_factor._point_clearance(
        np.asarray(truth_plane["normal_camera_xyz"], dtype=np.float64),
        float(truth_plane["camera_height_m"]),
        truth_query["boundary_points_shape_camera_xyz"],
        int(truth_query["query_support_points"]),
        truth_query["observed_forward_shape_m"],
        truth_local,
        query,
    )
    if not truth_clearance["evaluable"]:
        clearance = _unknown("FORMATION_TRUTH_QUERY_CLEARANCE_NOT_EVALUABLE")
    elif not candidate_clearance["evaluable"]:
        clearance = _unknown(str(candidate_clearance["reason_codes"][0]))
    else:
        clearance = {
            "evaluable": True,
            "reason_codes": [],
            "value_m": float(candidate_clearance["value_m"]),
            "truth_value_m": float(truth_clearance["value_m"]),
            "abs_error_m": abs(float(candidate_clearance["value_m"]) - float(truth_clearance["value_m"])),
        }
    return {"support": support, "boundary": boundary, "query_clearance": clearance}


def _effect(baseline: Mapping[str, Any], prospective: Mapping[str, Any], factor: str, metric: str, *, higher_is_better: bool = False) -> float | None:
    left = baseline[factor].get(metric)
    right = prospective[factor].get(metric)
    if baseline[factor].get("evaluable") is not True or prospective[factor].get("evaluable") is not True or left is None or right is None:
        return None
    return float(right) - float(left) if higher_is_better else float(left) - float(right)


def score_frame(
    *,
    source_role: str,
    source_frame_receipt: Mapping[str, Any],
    candidate_highres_depth_m: np.ndarray,
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
    prospective_bundle: Mapping[str, Any],
    highres_faro_depth_mm: np.ndarray,
) -> list[dict[str, Any]]:
    source = adapter._validate_base_receipt(dict(source_frame_receipt))
    bundle = runtime.validate_prospective_factor_bundle(dict(prospective_bundle), candidate_highres_depth_m=candidate_highres_depth_m)
    require(source["source_role"] == source_role and source["content_sha256"] == bundle["source_frame_receipt_sha256"], "FORMATION_SCORE_SOURCE_DRIFT", "formation bundle/source binding drift")
    matrix = adapter._intrinsics_matrix(source["intrinsics_highres"]["matrix_3x3"])
    raw = np.ascontiguousarray(candidate_highres_depth_m, dtype=np.float64)
    raw_hash = adapter.canonical_sha256(raw)
    require(raw_hash == bundle["input_bindings"]["candidate_highres_depth_sha256"], "FORMATION_SCORE_CANDIDATE_DRIFT", "formation candidate depth differs from bundle")
    apple = np.asarray(apple_depth_mm)
    conf = np.asarray(confidence)
    require(
        apple.shape == adapter.APPLE_SHAPE_HW
        and apple.dtype == np.uint16
        and conf.shape == adapter.APPLE_SHAPE_HW
        and conf.dtype == np.uint8
        and adapter.canonical_sha256(apple) == bundle["input_bindings"]["apple_depth_sha256"]
        and adapter.canonical_sha256(conf) == bundle["input_bindings"]["confidence_sha256"],
        "FORMATION_SCORE_SOURCE_INPUT_DRIFT",
        "formation AppleDepth/confidence differs from Phase-A binding",
    )
    low = source["lowres_intrinsics_source"]
    low_matrix = np.asarray([[low["fx"], 0.0, low["cx"]], [0.0, low["fy"], low["cy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
    gravity = np.asarray(source["gravity_up_camera_xyz"], dtype=np.float64)
    replay_scale = apple_scale.estimate_source_metric_scale(apple, conf, apple_scale.sample_candidate_at_apple_centers(raw))
    replay_baseline = runtime._fit_depth_plane(raw, matrix, gravity)
    replay_direct = runtime._fit_direct_plane(apple, conf, low_matrix, gravity) if replay_scale["evaluable"] else runtime._failed_plane(str(replay_scale["reason_codes"][0]))
    require(
        adapter.canonical_sha256(replay_scale) == adapter.canonical_sha256(bundle["source_scale"])
        and adapter.canonical_sha256(replay_baseline) == adapter.canonical_sha256(bundle["baseline_support"])
        and adapter.canonical_sha256(replay_direct) == adapter.canonical_sha256(bundle["direct_support"]),
        "FORMATION_PHASE_A_SOURCE_STATE_REPLAY_DRIFT",
        "Phase-A source scale/support commitments do not replay from bound inputs",
    )
    baseline_plane = replay_baseline if replay_baseline["evaluable"] else None
    baseline_geometry = runtime._build_geometry(raw, raw_hash, matrix) if baseline_plane is not None else None
    if bundle["selected_support_boundary_owner"] == "DIRECT_APPLE_SUPPORT":
        require(replay_scale["evaluable"] and replay_direct["evaluable"], "FORMATION_PHASE_A_OWNER_REPLAY_DRIFT", "sealed DIRECT owner is unavailable on source replay")
        selected_depth = np.ascontiguousarray(raw * float(replay_scale["metric_scale"]), dtype=np.float64)
        selected_plane = replay_direct
        selected_geometry = runtime._build_geometry(selected_depth, adapter.canonical_sha256(selected_depth), matrix)
    else:
        require(not (replay_scale["evaluable"] and replay_direct["evaluable"]), "FORMATION_PHASE_A_OWNER_REPLAY_DRIFT", "sealed R1 owner differs from source replay")
        selected_plane = baseline_plane
        selected_geometry = baseline_geometry
    faro = np.asarray(highres_faro_depth_mm)
    require(faro.shape == adapter.HIGHRES_SHAPE_HW and faro.dtype == np.uint16, "FORMATION_SCORE_FARO_INVALID", "formation FARO must be uint16 1440x1920")
    faro_m = np.ascontiguousarray(faro.astype(np.float64) / 1000.0, dtype=np.float64)
    truth_plane = runtime._fit_depth_plane(faro_m, matrix, gravity)
    truth_geometry = runtime._build_geometry(faro_m, adapter.canonical_sha256(faro_m), matrix) if truth_plane["evaluable"] else None
    records: list[dict[str, Any]] = []
    for slot in bundle["query_slots"]:
        query = slot["query_receipt"]
        if query is not None:
            replay_blocks = runtime._query_blocks(
                dict(query), matrix, bundle["selected_support_boundary_owner"], selected_geometry, selected_plane,
                baseline_geometry, baseline_plane, "SOURCE_SELECTED_SUPPORT_UNAVAILABLE", "SOURCE_BASELINE_SUPPORT_UNAVAILABLE", raw_hash,
            )
            require(adapter.canonical_sha256(replay_blocks) == adapter.canonical_sha256(slot["factor_blocks"]), "FORMATION_PHASE_A_FACTOR_REPLAY_DRIFT", "sealed prospective factor blocks do not replay")
        if query is None:
            prospective = baseline = {"support": _unknown("SOURCE_QUERY_FRAME_UNAVAILABLE"), "boundary": _unknown("SOURCE_QUERY_FRAME_UNAVAILABLE"), "query_clearance": _unknown("SOURCE_QUERY_FRAME_UNAVAILABLE")}
            truth_status = _unknown("SOURCE_QUERY_FRAME_UNAVAILABLE")
        elif truth_geometry is None:
            code = str(truth_plane["reason_codes"][0])
            prospective = baseline = {"support": _unknown(code), "boundary": _unknown(code), "query_clearance": _unknown(code)}
            truth_status = _unknown(code)
        else:
            prospective = _mode_score(
                geometry=selected_geometry,
                plane=selected_plane,
                query=query,
                matrix=matrix,
                truth_geometry=truth_geometry,
                truth_plane=truth_plane,
                failure_code="SOURCE_SELECTED_SUPPORT_UNAVAILABLE",
            )
            baseline = _mode_score(
                geometry=baseline_geometry,
                plane=baseline_plane,
                query=query,
                matrix=matrix,
                truth_geometry=truth_geometry,
                truth_plane=truth_plane,
                failure_code="SOURCE_BASELINE_SUPPORT_UNAVAILABLE",
            )
            truth_status = {"evaluable": True, "reason_codes": [], "support_plane_sha256": adapter.canonical_sha256(truth_plane)}
        effects = {
            "support_normal_error_reduction_rad": _effect(baseline, prospective, "support", "normal_angular_error_rad"),
            "support_height_error_reduction_m": _effect(baseline, prospective, "support", "height_abs_error_m"),
            "boundary_jaccard_increase": _effect(baseline, prospective, "boundary", "point_id_jaccard", higher_is_better=True),
            "boundary_xyz_error_reduction_m": _effect(baseline, prospective, "boundary", "xyz_median_error_m"),
            "query_clearance_error_reduction_m": _effect(baseline, prospective, "query_clearance", "abs_error_m"),
        }
        records.append(
            _seal(
                {
                    "schema": QUERY_SCORE_SCHEMA,
                    "source_role": source_role,
                    "parent_id": bundle["parent_id"],
                    "video_id": bundle["video_id"],
                    "timestamp_token": bundle["timestamp_token"],
                    "physical_frame_id": bundle["physical_frame_id"],
                    "grid_index": slot["grid_index"],
                    "query_id": slot["query_id"],
                    "query_receipt_sha256": None if query is None else query["content_sha256"],
                    "prospective_bundle_sha256": bundle["content_sha256"],
                    "selected_support_boundary_owner": bundle["selected_support_boundary_owner"],
                    "truth_status": truth_status,
                    "prospective": prospective,
                    "baseline": baseline,
                    "effects": effects,
                    "reselection_after_faro": False,
                    "promotion_authorized": False,
                }
            )
        )
    require(len(records) == 9 and [row["grid_index"] for row in records] == list(range(9)), "FORMATION_QUERY_COUNT_DRIFT", "formation scorer must retain nine query slots")
    return records


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def _aggregate_metric(records: Sequence[Mapping[str, Any]], mode: str, factor: str, metric: str) -> dict[str, Any]:
    by_frame: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        block = row[mode][factor]
        value = block.get(metric)
        if block.get("evaluable") is True and value is not None:
            by_frame[(str(row["parent_id"]), str(row["physical_frame_id"]))].append(float(value))
    frame_values = {key: _median(values) for key, values in by_frame.items()}
    by_parent: defaultdict[str, list[float]] = defaultdict(list)
    for (parent_id, _), value in frame_values.items():
        if value is not None:
            by_parent[parent_id].append(value)
    parent_values = {parent: _median(values) for parent, values in sorted(by_parent.items())}
    return {
        "query_evaluable_count": sum(len(values) for values in by_frame.values()),
        "frame_evaluable_count": len(frame_values),
        "parent_evaluable_count": len(parent_values),
        "parent_medians": parent_values,
        "parent_macro_median": _median([value for value in parent_values.values() if value is not None]),
    }


def _aggregate_effect(records: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    by_frame: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        value = row["effects"].get(metric)
        if value is not None:
            by_frame[(str(row["parent_id"]), str(row["physical_frame_id"]))].append(float(value))
    by_parent: defaultdict[str, list[float]] = defaultdict(list)
    for (parent_id, _), values in by_frame.items():
        value = _median(values)
        if value is not None:
            by_parent[parent_id].append(value)
    parent_values = {parent: _median(values) for parent, values in sorted(by_parent.items())}
    return {
        "query_paired_count": sum(len(values) for values in by_frame.values()),
        "frame_paired_count": len(by_frame),
        "parent_paired_count": len(parent_values),
        "parent_effects": parent_values,
        "parent_macro_median": _median([value for value in parent_values.values() if value is not None]),
        "positive_parent_count": sum(value is not None and value > 0.0 for value in parent_values.values()),
        "negative_parent_count": sum(value is not None and value < 0.0 for value in parent_values.values()),
        "zero_parent_count": sum(value == 0.0 for value in parent_values.values()),
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(row)) for row in records]
    require(len(rows) == 4050, "FORMATION_SUMMARY_QUERY_COUNT_DRIFT", "formation summary requires exactly 4050 query slots")
    parents = sorted({str(row["parent_id"]) for row in rows})
    frames = {(str(row["parent_id"]), str(row["physical_frame_id"])) for row in rows}
    require(len(parents) == 24 and len(frames) == 450, "FORMATION_SUMMARY_COHORT_DRIFT", "formation summary cohort differs from 24/450")
    modes: dict[str, Any] = {}
    for mode in ("prospective", "baseline"):
        modes[mode] = {
            factor: {metric: _aggregate_metric(rows, mode, factor, metric) for metric in metrics}
            for factor, metrics in METRICS.items()
        }
    effects = {
        metric: _aggregate_effect(rows, metric)
        for metric in (
            "support_normal_error_reduction_rad",
            "support_height_error_reduction_m",
            "boundary_jaccard_increase",
            "boundary_xyz_error_reduction_m",
            "query_clearance_error_reduction_m",
        )
    }
    reason_counts = Counter(
        reason
        for row in rows
        for mode in ("prospective", "baseline")
        for factor in METRICS
        if row[mode][factor].get("evaluable") is not True
        for reason in row[mode][factor].get("reason_codes", [])
    )
    role_summaries = {
        role: {
            "parent_count": len({row["parent_id"] for row in rows if row["source_role"] == role}),
            "frame_count": len({row["physical_frame_id"] for row in rows if row["source_role"] == role}),
            "query_slot_count": sum(row["source_role"] == role for row in rows),
        }
        for role in ("ADAPTER_FIT", "O0R_EVAL_CANDIDATE")
    }
    return _seal(
        {
            "schema": SUMMARY_SCHEMA,
            "terminal": "TARO_O0R_R6_FORMATION_REPLAY_COMPLETE_NON_PROMOTABLE",
            "execution_valid": True,
            "scientific_pass_fail_assigned": False,
            "parent_count": len(parents),
            "frame_count": len(frames),
            "query_slot_count": len(rows),
            "role_summaries": role_summaries,
            "modes": modes,
            "paired_effects": effects,
            "unknown_reason_counts": dict(sorted(reason_counts.items())),
            "aggregation_order": "QUERY_TO_FRAME_MEDIAN_THEN_FRAME_TO_PARENT_MEDIAN_THEN_MEDIAN_ACROSS_FIXED_PARENTS",
            "unknown_is_negative": False,
            "promotion_authorized": False,
            "claim_ceiling": "Post-hoc non-promotable WILD_LAB formation replay only.",
        }
    )


__all__ = ["FormationReplayError", "QUERY_SCORE_SCHEMA", "SUMMARY_SCHEMA", "score_frame", "summarize"]
