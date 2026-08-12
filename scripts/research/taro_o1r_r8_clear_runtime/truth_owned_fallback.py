#!/usr/bin/env python3
"""Dense FARO truth labels with a truth-owned fallback query frame."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_phase_a as shared_phase_a
from scripts.research.taro_o1r_r8_clear_runtime import ray_space_clear as ray_v1


SCHEMA = "blindassist.taro.o1r.r8_dense_truth_owned_fallback_label_frame.v1"
LABELER_ID = "TARO_R8_DENSE_FARO_SOURCE_QUERY_OR_TRUTH_OWNED_FALLBACK_V1"
SOURCE_QUERY_OWNER = "SOURCE_QUERY_FRAME"
FALLBACK_QUERY_OWNER = "FARO_TRUTH_OWNED_FALLBACK_QUERY_FRAME"
MINIMUM_CLEAR_QUERY_COUNT = 50
MINIMUM_CLEAR_PARENT_COUNT = 4


class TruthOwnedFallbackError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise TruthOwnedFallbackError(code, message)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R8_FALLBACK_SEAL_COLLISION", "fallback caller supplied a seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(record.get("schema") == SCHEMA and isinstance(observed, str) and adapter.canonical_sha256(record) == observed, "R8_FALLBACK_SEAL_DRIFT", "fallback seal/schema drift")
    record["content_sha256"] = observed
    return record


def _matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    require(matrix.shape == (3, 3) and bool(np.all(np.isfinite(matrix))) and matrix[0, 0] > 0 and matrix[1, 1] > 0, "R8_FALLBACK_INTRINSICS", "fallback intrinsics invalid")
    return matrix


def build_fallback_queries(physical_frame_id: str, plane: Mapping[str, Any]) -> list[dict[str, Any]]:
    queries = []
    for row in ray_v1.build_truth_queries(physical_frame_id, plane):
        queries.append(
            {
                "grid_index": row["grid_index"],
                "query_id": row["query_id"],
                "path_lateral_offset_m": row["path_lateral_offset_m"],
                "path_yaw_degrees": row["path_yaw_degrees"],
                "minimum_forward_m": adapter.MINIMUM_FORWARD_M,
                "horizon_m": adapter.HORIZON_M,
                "capsule_radius_m": adapter.CAPSULE_RADIUS_M,
                "virtual_query_frame": {
                    "kind": "FARO_TRUTH_OWNED_SUPPORT_PLANE_V1",
                    "origin_camera_xyz": row["origin_camera_xyz"],
                    "forward_camera_xyz": row["forward_camera_xyz"],
                    "lateral_camera_xyz": row["lateral_camera_xyz"],
                    "gravity_up_camera_xyz": row["gravity_up_camera_xyz"],
                    "path_heading_camera_xyz": row["path_heading_camera_xyz"],
                },
            }
        )
    return queries


def select_query_frame(feature: Mapping[str, Any], fallback_query: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    if feature.get("query_receipt") is not None:
        return feature["query_receipt"], SOURCE_QUERY_OWNER
    require(feature.get("grid_index") == fallback_query.get("grid_index") and feature.get("query_id") == fallback_query.get("query_id"), "R8_FALLBACK_QUERY_ALIGNMENT", "fallback query identity drift")
    return fallback_query, FALLBACK_QUERY_OWNER


def _unknown_label(reason_codes: Sequence[str], owner: str) -> dict[str, Any]:
    return {
        "state": "UNKNOWN",
        "obstacle_pixel_count": 0,
        "minimum_truth_obstacle_pixels": r7_canary.MINIMUM_TRUTH_OBSTACLE_PIXELS,
        "query_support_points": 0,
        "observed_forward_m": None,
        "local_valid_fraction": 0.0,
        "reason_codes": list(reason_codes),
        "query_frame_owner": owner,
    }


def build_label_frame(source_frame_record: Mapping[str, Any], faro_depth_mm: np.ndarray, intrinsics_highres_3x3: Any, gravity_up_camera_xyz: Any) -> dict[str, Any]:
    source = r7_canary.validate_source_frame_record(dict(source_frame_record))
    faro = np.asarray(faro_depth_mm)
    require(faro.shape == adapter.HIGHRES_SHAPE_HW and faro.dtype == np.uint16, "R8_FALLBACK_FARO", "fallback FARO must be uint16 1440x1920")
    matrix = _matrix(intrinsics_highres_3x3)
    gravity = adapter._normalize_vector(gravity_up_camera_xyz, "R8_FALLBACK_GRAVITY")
    require(adapter.canonical_sha256(matrix) == source["input_bindings"]["intrinsics_highres_sha256"] and adapter.canonical_sha256(gravity) == source["input_bindings"]["gravity_up_camera_xyz_sha256"], "R8_FALLBACK_SOURCE_LINEAGE", "fallback source intrinsics/gravity drift")
    depth_m = np.ascontiguousarray(faro.astype(np.float64) / 1000.0, dtype=np.float64)
    plane = prospective._fit_depth_plane(depth_m, matrix, gravity)
    geometry = prospective._build_geometry(depth_m, adapter.canonical_sha256(depth_m), matrix) if plane["evaluable"] else None
    fallback_queries = build_fallback_queries(source["physical_frame_id"], plane) if plane["evaluable"] else [None] * 9
    labels = []
    for feature, fallback_query in zip(source["query_features"], fallback_queries, strict=True):
        owner = SOURCE_QUERY_OWNER if feature["query_receipt"] is not None else FALLBACK_QUERY_OWNER
        if geometry is None:
            label = _unknown_label(plane["reason_codes"], owner)
        else:
            query, owner = select_query_frame(feature, fallback_query)
            label = {**r7_canary._truth_query_label(geometry, plane, matrix, query), "query_frame_owner": owner}
        labels.append({"grid_index": feature["grid_index"], "query_id": feature["query_id"], **label})
    return validate_label_frame(
        _seal(
            {
                "schema": SCHEMA,
                "labeler_id": LABELER_ID,
                "parent_id": source["parent_id"],
                "video_id": source["video_id"],
                "timestamp_token": source["timestamp_token"],
                "physical_frame_id": source["physical_frame_id"],
                "source_frame_record_sha256": source["content_sha256"],
                "highres_faro_depth_sha256": adapter.canonical_sha256(faro),
                "truth_plane": plane,
                "query_labels": labels,
                "source_query_used_when_present": True,
                "faro_truth_owned_query_used_only_when_source_query_absent": True,
                "dense_faro_label_runtime": r7_canary.REDUCER_ID,
                "unknown_is_negative": False,
            }
        ),
        source,
    )


def validate_label_frame(value: Mapping[str, Any], source_frame_record: Mapping[str, Any]) -> dict[str, Any]:
    source = r7_canary.validate_source_frame_record(dict(source_frame_record))
    record = _validate_seal(value)
    require(record["labeler_id"] == LABELER_ID and record["source_frame_record_sha256"] == source["content_sha256"] and record["physical_frame_id"] == source["physical_frame_id"], "R8_FALLBACK_LABEL_LINEAGE", "fallback label lineage drift")
    require(record["source_query_used_when_present"] is True and record["faro_truth_owned_query_used_only_when_source_query_absent"] is True and record["dense_faro_label_runtime"] == r7_canary.REDUCER_ID and record["unknown_is_negative"] is False, "R8_FALLBACK_FIREWALL", "fallback truth ownership drift")
    labels = record["query_labels"]
    require(len(labels) == 9 and [row["grid_index"] for row in labels] == list(range(9)), "R8_FALLBACK_LABEL_COUNT", "fallback label count/order drift")
    for feature, label in zip(source["query_features"], labels, strict=True):
        expected_owner = SOURCE_QUERY_OWNER if feature["query_receipt"] is not None else FALLBACK_QUERY_OWNER
        require(feature["query_id"] == label["query_id"] and label["state"] in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"} and label["query_frame_owner"] == expected_owner, "R8_FALLBACK_LABEL_ALIGNMENT", "fallback label alignment/owner drift")
    return record


def summarize(sources: Sequence[Mapping[str, Any]], old_labels: Sequence[Mapping[str, Any]], new_labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(sources) == len(old_labels) == len(new_labels) and len(sources) > 0, "R8_FALLBACK_SUMMARY_COUNT", "fallback summary frame count drift")
    counts: Counter[str] = Counter()
    clear_parents = set()
    old_definite_count = old_definite_retained = 0
    fallback_slot_count = fallback_clear = fallback_occupied = fallback_unknown = 0
    positive_on_fallback_clear = changed_without_fallback = 0
    per_parent: dict[str, Counter[str]] = defaultdict(Counter)
    for source, old, new in zip(sources, old_labels, new_labels, strict=True):
        new = validate_label_frame(new, source)
        for feature, old_label, new_label in zip(source["query_features"], old["query_labels"], new["query_labels"], strict=True):
            state = new_label["state"]
            counts[state] += 1
            per_parent[source["parent_id"]][state] += 1
            if state == "CLEAR_OBSERVED":
                clear_parents.add(source["parent_id"])
            if old_label["state"] != "UNKNOWN":
                old_definite_count += 1
                old_definite_retained += state == old_label["state"]
            if feature["query_receipt"] is None:
                fallback_slot_count += 1
                fallback_clear += state == "CLEAR_OBSERVED"
                fallback_occupied += state == "OCCUPIED_OBSERVED"
                fallback_unknown += state == "UNKNOWN"
                if state == "CLEAR_OBSERVED":
                    positive_on_fallback_clear += shared_phase_a._positive_state(feature) == "OCCUPIED_OBSERVED"
            elif state != old_label["state"]:
                changed_without_fallback += 1
    coverage_pass = counts["CLEAR_OBSERVED"] >= MINIMUM_CLEAR_QUERY_COUNT and len(clear_parents) >= MINIMUM_CLEAR_PARENT_COUNT
    guardrails_pass = old_definite_retained == old_definite_count and changed_without_fallback == 0 and positive_on_fallback_clear == 0
    return {
        "label_state_counts": {state: int(counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
        "parents_with_clear": len(clear_parents),
        "fallback_slot_count": fallback_slot_count,
        "fallback_label_state_counts": {"CLEAR_OBSERVED": fallback_clear, "OCCUPIED_OBSERVED": fallback_occupied, "UNKNOWN": fallback_unknown},
        "old_definite_count": old_definite_count,
        "old_definite_retained_exactly": old_definite_retained,
        "nonfallback_labels_changed": changed_without_fallback,
        "positive_occupancy_predictions_on_fallback_clear": positive_on_fallback_clear,
        "coverage_gate": {"minimum_clear_queries": MINIMUM_CLEAR_QUERY_COUNT, "minimum_clear_parents": MINIMUM_CLEAR_PARENT_COUNT, "passed": coverage_pass},
        "guardrails_passed": guardrails_pass,
        "passed": bool(coverage_pass and guardrails_pass),
        "per_parent": {parent: {state: int(row[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")} for parent, row in per_parent.items()},
        "unknown_is_negative": False,
        "claim_ceiling": "Post-hoc dense FARO truth-owned fallback interface canary on consumed R8 selected frames; no effectiveness or route promotion.",
    }


__all__ = ["FALLBACK_QUERY_OWNER", "LABELER_ID", "SOURCE_QUERY_OWNER", "TruthOwnedFallbackError", "build_fallback_queries", "build_label_frame", "select_query_frame", "summarize", "validate_label_frame"]
