#!/usr/bin/env python3
"""Truth-owned FARO ray-space labels for the fixed TARO 3x3 query grid."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_phase_a as shared_phase_a


SCHEMA = "blindassist.taro.o1r.r8_faro_ray_space_label_frame.v1"
LABELER_ID = "TARO_R8_TRUTH_OWNED_FARO_RAY_SPACE_3X3_V1"
FORWARD_SLICES_M = (0.5, 1.0, 1.5, 2.0)
LATERAL_ANCHORS_M = (-0.25, 0.0, 0.25)
HEIGHT_ANCHORS_M = (0.25, 0.75, 1.25)
PATCH_RADIUS_PX = 2
DEPTH_ENDPOINT_TOLERANCE_M = 0.05
MINIMUM_PROJECTED_ANCHORS_PER_SLICE = 2
MINIMUM_VALID_ANCHOR_FRACTION_PER_SLICE = adapter.MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION
MINIMUM_BLOCKED_ANCHORS_FOR_OCCUPIED = 2
MINIMUM_CLEAR_QUERY_COUNT = 50
MINIMUM_CLEAR_PARENT_COUNT = 4


class RaySpaceClearError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise RaySpaceClearError(code, message)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R8_RAY_SEAL_COLLISION", "ray-space caller supplied a seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(record.get("schema") == SCHEMA and isinstance(observed, str) and adapter.canonical_sha256(record) == observed, "R8_RAY_SEAL_DRIFT", "ray-space seal/schema drift")
    record["content_sha256"] = observed
    return record


def _matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    require(matrix.shape == (3, 3) and bool(np.all(np.isfinite(matrix))) and matrix[0, 0] > 0 and matrix[1, 1] > 0, "R8_RAY_INTRINSICS", "ray-space intrinsics invalid")
    return matrix


def build_truth_queries(physical_frame_id: str, plane: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(plane.get("evaluable") is True, "R8_RAY_PLANE_UNAVAILABLE", "truth plane unavailable")
    normal = adapter._normalize_vector(plane["normal_camera_xyz"], "R8_RAY_PLANE_NORMAL")
    height = float(plane["camera_height_m"])
    require(0.45 <= height <= 2.2, "R8_RAY_CAMERA_HEIGHT", "truth camera height invalid")
    optical = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    forward = adapter._normalize_vector(optical - float(np.dot(optical, normal)) * normal, "R8_RAY_FORWARD")
    if float(np.dot(forward, optical)) < 0.0:
        forward = -forward
    lateral = adapter._normalize_vector(np.cross(forward, normal), "R8_RAY_LATERAL")
    origin = -height * normal
    queries = []
    for grid_index, (offset, yaw) in enumerate((offset, yaw) for offset in adapter.PATH_LATERAL_OFFSETS_M for yaw in adapter.PATH_YAW_DEGREES):
        radians = math.radians(float(yaw))
        heading = adapter._normalize_vector(math.cos(radians) * forward + math.sin(radians) * lateral, "R8_RAY_HEADING")
        queries.append({"grid_index": grid_index, "path_lateral_offset_m": float(offset), "path_yaw_degrees": float(yaw), "origin_camera_xyz": origin.tolist(), "forward_camera_xyz": forward.tolist(), "lateral_camera_xyz": lateral.tolist(), "gravity_up_camera_xyz": normal.tolist(), "path_heading_camera_xyz": heading.tolist(), "query_id": f"{physical_frame_id}:lat_{float(offset):+.2f}_yaw_{float(yaw):+.1f}"})
    require(len(queries) == 9, "R8_RAY_QUERY_COUNT", "truth query count drift")
    return queries


def _patch_depth(depth_m: np.ndarray, row: int, column: int) -> float | None:
    r0, r1 = max(0, row - PATCH_RADIUS_PX), min(depth_m.shape[0], row + PATCH_RADIUS_PX + 1)
    c0, c1 = max(0, column - PATCH_RADIUS_PX), min(depth_m.shape[1], column + PATCH_RADIUS_PX + 1)
    patch = depth_m[r0:r1, c0:c1]
    valid = patch[(patch >= adapter.DEPTH_RANGE_M[0]) & (patch <= adapter.DEPTH_RANGE_M[1])]
    return None if valid.size == 0 else float(np.median(valid))


def ray_query_label(depth_m: np.ndarray, intrinsics_3x3: Any, query: Mapping[str, Any]) -> dict[str, Any]:
    depth = np.asarray(depth_m, dtype=np.float64)
    require(depth.shape == adapter.HIGHRES_SHAPE_HW and bool(np.all(np.isfinite(depth))), "R8_RAY_DEPTH", "ray-space depth invalid")
    matrix = _matrix(intrinsics_3x3)
    origin = np.asarray(query["origin_camera_xyz"], dtype=np.float64)
    lateral = adapter._normalize_vector(query["lateral_camera_xyz"], "R8_RAY_QUERY_LATERAL")
    heading = adapter._normalize_vector(query["path_heading_camera_xyz"], "R8_RAY_QUERY_HEADING")
    up = adapter._normalize_vector(query["gravity_up_camera_xyz"], "R8_RAY_QUERY_UP")
    side = adapter._normalize_vector(np.cross(heading, up), "R8_RAY_QUERY_SIDE")
    path_origin = origin + float(query["path_lateral_offset_m"]) * lateral
    slices = []
    total_projected = total_valid = total_blocked = 0
    for forward_m in FORWARD_SLICES_M:
        projected_count = valid_count = blocked_count = 0
        for lateral_m in LATERAL_ANCHORS_M:
            for height_m in HEIGHT_ANCHORS_M:
                point = path_origin + float(forward_m) * heading + float(lateral_m) * side + float(height_m) * up
                if point[2] <= 1e-9:
                    continue
                column = int(round(float(matrix[0, 0] * point[0] / point[2] + matrix[0, 2])))
                row = int(round(float(matrix[1, 1] * point[1] / point[2] + matrix[1, 2])))
                if not (0 <= row < depth.shape[0] and 0 <= column < depth.shape[1]):
                    continue
                projected_count += 1
                observed = _patch_depth(depth, row, column)
                if observed is None:
                    continue
                valid_count += 1
                blocked_count += observed + DEPTH_ENDPOINT_TOLERANCE_M < float(point[2])
        valid_fraction = float(valid_count / projected_count) if projected_count else 0.0
        slices.append({"forward_m": float(forward_m), "projected_anchor_count": projected_count, "valid_anchor_count": valid_count, "valid_anchor_fraction": valid_fraction, "blocked_anchor_count": blocked_count})
        total_projected += projected_count
        total_valid += valid_count
        total_blocked += blocked_count
    occupied = total_blocked >= MINIMUM_BLOCKED_ANCHORS_FOR_OCCUPIED
    full_coverage = all(
        row["projected_anchor_count"] >= MINIMUM_PROJECTED_ANCHORS_PER_SLICE
        and row["valid_anchor_fraction"] >= MINIMUM_VALID_ANCHOR_FRACTION_PER_SLICE
        for row in slices
    )
    clear = total_blocked == 0 and full_coverage
    state = "OCCUPIED_OBSERVED" if occupied else "CLEAR_OBSERVED" if clear else "UNKNOWN"
    reasons = [] if state != "UNKNOWN" else ["FARO_RAY_SPACE_COVERAGE_INSUFFICIENT" if not full_coverage else "FARO_RAY_SPACE_BLOCKAGE_AMBIGUOUS"]
    return {"state": state, "projected_anchor_count": total_projected, "valid_anchor_count": total_valid, "blocked_anchor_count": total_blocked, "minimum_blocked_anchors_for_occupied": MINIMUM_BLOCKED_ANCHORS_FOR_OCCUPIED, "minimum_projected_anchors_per_slice": MINIMUM_PROJECTED_ANCHORS_PER_SLICE, "minimum_valid_anchor_fraction_per_slice": MINIMUM_VALID_ANCHOR_FRACTION_PER_SLICE, "slice_evidence": slices, "reason_codes": reasons}


def build_label_frame(source_frame_record: Mapping[str, Any], faro_depth_mm: np.ndarray, intrinsics_highres_3x3: Any, gravity_up_camera_xyz: Any) -> dict[str, Any]:
    source = r7_canary.validate_source_frame_record(dict(source_frame_record))
    faro = np.asarray(faro_depth_mm)
    require(faro.shape == adapter.HIGHRES_SHAPE_HW and faro.dtype == np.uint16, "R8_RAY_FARO", "ray-space FARO must be uint16 1440x1920")
    matrix = _matrix(intrinsics_highres_3x3)
    gravity = adapter._normalize_vector(gravity_up_camera_xyz, "R8_RAY_GRAVITY")
    require(adapter.canonical_sha256(matrix) == source["input_bindings"]["intrinsics_highres_sha256"] and adapter.canonical_sha256(gravity) == source["input_bindings"]["gravity_up_camera_xyz_sha256"], "R8_RAY_SOURCE_LINEAGE", "ray-space source intrinsics/gravity drift")
    depth_m = np.ascontiguousarray(faro.astype(np.float64) / 1000.0, dtype=np.float64)
    plane = prospective._fit_depth_plane(depth_m, matrix, gravity)
    labels = []
    if plane["evaluable"]:
        truth_queries = build_truth_queries(source["physical_frame_id"], plane)
        for feature, query in zip(source["query_features"], truth_queries, strict=True):
            require(feature["query_id"] == query["query_id"] and feature["grid_index"] == query["grid_index"], "R8_RAY_QUERY_ALIGNMENT", "truth-owned and source query identities drift")
            label = ray_query_label(depth_m, matrix, query)
            labels.append({"grid_index": feature["grid_index"], "query_id": feature["query_id"], **label})
    else:
        for feature in source["query_features"]:
            labels.append({"grid_index": feature["grid_index"], "query_id": feature["query_id"], "state": "UNKNOWN", "projected_anchor_count": 0, "valid_anchor_count": 0, "blocked_anchor_count": 0, "minimum_blocked_anchors_for_occupied": MINIMUM_BLOCKED_ANCHORS_FOR_OCCUPIED, "minimum_projected_anchors_per_slice": MINIMUM_PROJECTED_ANCHORS_PER_SLICE, "minimum_valid_anchor_fraction_per_slice": MINIMUM_VALID_ANCHOR_FRACTION_PER_SLICE, "slice_evidence": [], "reason_codes": list(plane["reason_codes"])})
    return validate_label_frame(_seal({"schema": SCHEMA, "labeler_id": LABELER_ID, "parent_id": source["parent_id"], "video_id": source["video_id"], "timestamp_token": source["timestamp_token"], "physical_frame_id": source["physical_frame_id"], "source_frame_record_sha256": source["content_sha256"], "highres_faro_depth_sha256": adapter.canonical_sha256(faro), "truth_plane": plane, "query_labels": labels, "truth_queries_owned_by_faro": True, "source_query_availability_used_for_truth_query_construction": False, "unknown_is_negative": False}), source)


def validate_label_frame(value: Mapping[str, Any], source_frame_record: Mapping[str, Any]) -> dict[str, Any]:
    source = r7_canary.validate_source_frame_record(dict(source_frame_record))
    record = _validate_seal(value)
    require(record["labeler_id"] == LABELER_ID and record["source_frame_record_sha256"] == source["content_sha256"] and record["physical_frame_id"] == source["physical_frame_id"], "R8_RAY_LABEL_LINEAGE", "ray-space label lineage drift")
    require(record["truth_queries_owned_by_faro"] is True and record["source_query_availability_used_for_truth_query_construction"] is False and record["unknown_is_negative"] is False, "R8_RAY_LABEL_FIREWALL", "ray-space truth ownership drift")
    labels = record["query_labels"]
    require(len(labels) == 9 and [row["grid_index"] for row in labels] == list(range(9)), "R8_RAY_LABEL_COUNT", "ray-space label count/order drift")
    require(all(feature["query_id"] == label["query_id"] and label["state"] in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"} for feature, label in zip(source["query_features"], labels, strict=True)), "R8_RAY_LABEL_ALIGNMENT", "ray-space label alignment drift")
    return record


def summarize(sources: Sequence[Mapping[str, Any]], old_labels: Sequence[Mapping[str, Any]], ray_labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(sources) == len(old_labels) == len(ray_labels) and len(sources) > 0, "R8_RAY_SUMMARY_COUNT", "ray-space summary frame count drift")
    counts: Counter[str] = Counter()
    clear_parents = set()
    old_clear_retained = old_occupied_to_clear = positive_on_ray_clear = 0
    old_clear_count = old_occupied_count = 0
    per_parent: dict[str, Counter[str]] = defaultdict(Counter)
    for source, old, ray in zip(sources, old_labels, ray_labels, strict=True):
        ray = validate_label_frame(ray, source)
        for feature, old_label, new_label in zip(source["query_features"], old["query_labels"], ray["query_labels"], strict=True):
            state = new_label["state"]
            counts[state] += 1
            per_parent[source["parent_id"]][state] += 1
            if state == "CLEAR_OBSERVED":
                clear_parents.add(source["parent_id"])
                positive_on_ray_clear += shared_phase_a._positive_state(feature) == "OCCUPIED_OBSERVED"
            if old_label["state"] == "CLEAR_OBSERVED":
                old_clear_count += 1
                old_clear_retained += state == "CLEAR_OBSERVED"
            if old_label["state"] == "OCCUPIED_OBSERVED":
                old_occupied_count += 1
                old_occupied_to_clear += state == "CLEAR_OBSERVED"
    coverage_pass = counts["CLEAR_OBSERVED"] >= MINIMUM_CLEAR_QUERY_COUNT and len(clear_parents) >= MINIMUM_CLEAR_PARENT_COUNT
    guardrails_pass = old_clear_retained == old_clear_count and old_occupied_to_clear == 0 and positive_on_ray_clear == 0
    return {"label_state_counts": {state: int(counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")}, "parents_with_clear": len(clear_parents), "old_clear_count": old_clear_count, "old_clear_retained": old_clear_retained, "old_occupied_count": old_occupied_count, "old_occupied_reclassified_clear": old_occupied_to_clear, "positive_occupancy_predictions_on_ray_clear": positive_on_ray_clear, "coverage_gate": {"minimum_clear_queries": MINIMUM_CLEAR_QUERY_COUNT, "minimum_clear_parents": MINIMUM_CLEAR_PARENT_COUNT, "passed": coverage_pass}, "guardrails_passed": guardrails_pass, "passed": bool(coverage_pass and guardrails_pass), "per_parent": {parent: {state: int(row[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")} for parent, row in per_parent.items()}, "unknown_is_negative": False, "claim_ceiling": "Post-hoc FARO ray-space truth-interface canary on consumed R8 selected frames; no effectiveness or route promotion."}


__all__ = ["LABELER_ID", "RaySpaceClearError", "build_label_frame", "build_truth_queries", "ray_query_label", "summarize", "validate_label_frame"]
