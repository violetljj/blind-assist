from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from itertools import product
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_reducer_integration_runtime import reducer_integration as r6_reducer


SOURCE_FRAME_SCHEMA = "blindassist.taro.o1r.r7_source_feature_frame.v1"
LABEL_FRAME_SCHEMA = "blindassist.taro.o1r.r7_faro_label_frame.v1"
CANARY_RESULT_SCHEMA = "blindassist.taro.o1r.r7_fit_lopo_canary_result.v1"
REDUCER_ID = "R7_POSITIVE_OCCUPANCY_AND_FAR_CENSORED_CLEAR_REDUCER_V1"

OCCUPIED_PIXELS = (2, 4, 8, 16)
OCCUPIED_HEIGHT_M = (0.08, 0.15, 0.25)
OCCUPIED_MAX_FORWARD_M = (1.0, 1.5, 2.0)
CLEAR_FAR_FRACTION = (0.8, 0.9, 0.95)
CLEAR_FAR_DEPTH_FLOOR_M = (2.5, 3.0, 4.0)
CLEAR_SUPPORT_POINTS = (32, 64, 128)
FAR_SAMPLE_FORWARD_M = tuple(float(value) for value in np.linspace(adapter.MINIMUM_FORWARD_M, adapter.HORIZON_M, 10))
FAR_SAMPLE_LATERAL_M = (-adapter.CAPSULE_RADIUS_M, 0.0, adapter.CAPSULE_RADIUS_M)
FAR_SAMPLE_HEIGHT_M = (0.25, 0.75, 1.25)
FAR_SAMPLE_COUNT = len(FAR_SAMPLE_FORWARD_M) * len(FAR_SAMPLE_LATERAL_M) * len(FAR_SAMPLE_HEIGHT_M)
MINIMUM_FAR_VISIBLE_ANCHORS = 9
MINIMUM_TRUTH_OBSTACLE_PIXELS = 16
WILSON_Z_ONE_SIDED_95 = 1.6448536269514722


class R7CanaryError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R7CanaryError(code, message, **context)


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in output, "R7_SEAL_COLLISION", "caller supplied a content seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R7_RECORD_INVALID", "sealed R7 record must be an object")
    output = copy.deepcopy(value)
    observed = output.pop("content_sha256", None)
    require(output.get("schema") == schema and observed == adapter.canonical_sha256(output), "R7_SEAL_MISMATCH", "R7 record seal drift", schema=schema)
    output["content_sha256"] = observed
    return output


def _matrix(value: Any, field: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    require(matrix.shape == (3, 3) and bool(np.all(np.isfinite(matrix))) and matrix[0, 0] > 0.0 and matrix[1, 1] > 0.0, "R7_INTRINSICS_INVALID", "R7 intrinsics are invalid", field=field)
    return matrix


def _apple_points(apple_depth_mm: np.ndarray, confidence: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    apple = np.asarray(apple_depth_mm)
    conf = np.asarray(confidence)
    require(apple.shape == adapter.APPLE_SHAPE_HW and apple.dtype == np.uint16, "R7_APPLE_DEPTH_INVALID", "AppleDepth must be uint16 192x256")
    require(conf.shape == adapter.APPLE_SHAPE_HW and conf.dtype == np.uint8, "R7_CONFIDENCE_INVALID", "confidence must be uint8 192x256")
    depth = apple.astype(np.float64) / 1000.0
    valid = (conf == 2) & (depth >= adapter.DEPTH_RANGE_M[0]) & (depth <= adapter.DEPTH_RANGE_M[1])
    rows, columns = np.indices(adapter.APPLE_SHAPE_HW, dtype=np.float64)
    points = np.stack(
        (
            (columns - matrix[0, 2]) * depth / matrix[0, 0],
            (rows - matrix[1, 2]) * depth / matrix[1, 1],
            depth,
        ),
        axis=-1,
    )
    return np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(valid, dtype=np.bool_)


def _query_coordinates(points_hw3: np.ndarray, query: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin, _, lateral, heading = adapter._query_receipt_vectors(dict(query))
    up = adapter._normalize_vector(query["virtual_query_frame"]["gravity_up_camera_xyz"], "R7_QUERY_FRAME_INVALID")
    side = adapter._normalize_vector(np.cross(heading, up), "R7_QUERY_FRAME_INVALID")
    path_origin = origin + float(query["path_lateral_offset_m"]) * lateral
    relative = points_hw3 - path_origin[None, None, :]
    along = relative @ heading
    across = relative @ side
    height = relative @ up
    return along, across, height


def _occupied_grid(points_hw3: np.ndarray, valid_hw: np.ndarray, query: Mapping[str, Any]) -> list[list[list[bool]]]:
    along, across, height = _query_coordinates(points_hw3, query)
    mask = (
        valid_hw
        & (along >= adapter.MINIMUM_FORWARD_M)
        & (along <= adapter.HORIZON_M + adapter.GEOMETRY_ENDPOINT_TOLERANCE_M)
        & (np.abs(across) <= adapter.CAPSULE_RADIUS_M)
        & (height >= adapter.OBSTACLE_HEIGHT_RANGE_M[0])
        & (height <= adapter.OBSTACLE_HEIGHT_RANGE_M[1])
    )
    labels, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    components: list[tuple[int, float, float]] = []
    if component_count:
        counts = np.bincount(labels.ravel())
        for component_id in range(1, component_count + 1):
            members = labels == component_id
            components.append((int(counts[component_id]), float(np.max(height[members])), float(np.min(along[members]))))
    return [
        [
            [
                any(count >= pixels and maximum_height >= minimum_height and minimum_forward <= maximum_forward for count, maximum_height, minimum_forward in components)
                for maximum_forward in OCCUPIED_MAX_FORWARD_M
            ]
            for minimum_height in OCCUPIED_HEIGHT_M
        ]
        for pixels in OCCUPIED_PIXELS
    ]


def _far_samples(candidate_metric_depth_m: np.ndarray | None, query: Mapping[str, Any], matrix: np.ndarray) -> list[float | None]:
    if candidate_metric_depth_m is None:
        return [None] * FAR_SAMPLE_COUNT
    depth = np.asarray(candidate_metric_depth_m, dtype=np.float64)
    require(depth.shape == adapter.HIGHRES_SHAPE_HW and bool(np.all(np.isfinite(depth))), "R7_CANDIDATE_DEPTH_INVALID", "candidate metric depth must be finite 1440x1920")
    origin, _, lateral, heading = adapter._query_receipt_vectors(dict(query))
    up = adapter._normalize_vector(query["virtual_query_frame"]["gravity_up_camera_xyz"], "R7_QUERY_FRAME_INVALID")
    side = adapter._normalize_vector(np.cross(heading, up), "R7_QUERY_FRAME_INVALID")
    path_origin = origin + float(query["path_lateral_offset_m"]) * lateral
    samples: list[float | None] = []
    height, width = depth.shape
    for forward_m in FAR_SAMPLE_FORWARD_M:
        for lateral_m in FAR_SAMPLE_LATERAL_M:
            for obstacle_height_m in FAR_SAMPLE_HEIGHT_M:
                point = path_origin + forward_m * heading + lateral_m * side + obstacle_height_m * up
                if point[2] <= 1e-9:
                    samples.append(None)
                    continue
                column = int(round(float(matrix[0, 0] * point[0] / point[2] + matrix[0, 2])))
                row = int(round(float(matrix[1, 1] * point[1] / point[2] + matrix[1, 2])))
                samples.append(float(depth[row, column]) if 0 <= row < height and 0 <= column < width else None)
    require(len(samples) == FAR_SAMPLE_COUNT, "R7_FAR_SAMPLE_CARDINALITY", "far sample cardinality drift")
    return samples


def _far_fractions(samples: Sequence[float | None]) -> list[float]:
    require(len(samples) == FAR_SAMPLE_COUNT, "R7_FAR_SAMPLE_CARDINALITY", "far sample cardinality drift")
    visible = [float(value) for value in samples if value is not None and _finite(value)]
    return [sum(value >= floor for value in visible) / len(visible) if visible else 0.0 for floor in CLEAR_FAR_DEPTH_FLOOR_M]


def build_source_frame_record(
    prospective_bundle: Mapping[str, Any],
    candidate_highres_depth_m: np.ndarray,
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
    intrinsics_apple_3x3: Any,
    intrinsics_highres_3x3: Any,
    r6_reducer_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Build sealed source features. This API intentionally accepts no label-side input."""

    candidate = np.asarray(candidate_highres_depth_m, dtype=np.float64)
    bundle = prospective.validate_prospective_factor_bundle(dict(prospective_bundle), candidate_highres_depth_m=candidate)
    r6 = r6_reducer.validate_reducer_bundle(dict(r6_reducer_bundle))
    require(r6["prospective_bundle_sha256"] == bundle["content_sha256"], "R7_R6_RESULT_LINEAGE_DRIFT", "R6 reducer result is not bound to the R7 prospective bundle")
    apple = np.asarray(apple_depth_mm)
    conf = np.asarray(confidence)
    require(adapter.canonical_sha256(apple) == bundle["input_bindings"]["apple_depth_sha256"], "R7_APPLE_DEPTH_LINEAGE_DRIFT", "AppleDepth differs from the prospective bundle")
    require(adapter.canonical_sha256(conf) == bundle["input_bindings"]["confidence_sha256"], "R7_CONFIDENCE_LINEAGE_DRIFT", "confidence differs from the prospective bundle")
    low_matrix = _matrix(intrinsics_apple_3x3, "intrinsics_apple_3x3")
    high_matrix = _matrix(intrinsics_highres_3x3, "intrinsics_highres_3x3")
    require(adapter.canonical_sha256(low_matrix) == bundle["input_bindings"]["intrinsics_apple_sha256"], "R7_APPLE_INTRINSICS_LINEAGE_DRIFT", "Apple intrinsics differ from the prospective bundle")
    require(adapter.canonical_sha256(high_matrix) == bundle["input_bindings"]["intrinsics_highres_sha256"], "R7_HIGHRES_INTRINSICS_LINEAGE_DRIFT", "highres intrinsics differ from the prospective bundle")
    points, valid = _apple_points(apple, conf, low_matrix)
    source_scale = bundle["source_scale"]
    metric_candidate = None
    if source_scale["evaluable"] is True:
        metric_candidate = np.ascontiguousarray(candidate * float(source_scale["metric_scale"]), dtype=np.float64)
        require(adapter.canonical_sha256(metric_candidate) == bundle["input_bindings"]["anchored_candidate_depth_sha256"], "R7_ANCHORED_CANDIDATE_LINEAGE_DRIFT", "anchored candidate differs from the R6 bundle")
    queries: list[dict[str, Any]] = []
    for index, (slot, r6_query) in enumerate(zip(bundle["query_slots"], r6["query_results"], strict=True)):
        require(slot["grid_index"] == r6_query["grid_index"] == index and slot["query_id"] == r6_query["query_id"], "R7_QUERY_ALIGNMENT_DRIFT", "R6 and prospective query order differ")
        query = slot["query_receipt"]
        if query is None:
            queries.append(
                {
                    "grid_index": index,
                    "query_id": slot["query_id"],
                    "query_receipt": None,
                    "r6_state": r6_query["state"],
                    "occupied_hits": None,
                    "positive_obstacle_veto": None,
                    "far_fractions": None,
                    "far_valid_anchor_count": 0,
                    "observed_support_points": 0,
                    "reason_codes": ["SOURCE_QUERY_FRAME_UNAVAILABLE"],
                }
            )
            continue
        hits = _occupied_grid(points, valid, query)
        samples = _far_samples(metric_candidate, query, high_matrix)
        support = slot["factor_blocks"]["SUPPORT"]
        support_points = int(support["validity"].get("query_support_points", 0)) if support["evaluable"] else 0
        queries.append(
            {
                "grid_index": index,
                "query_id": slot["query_id"],
                "query_receipt": query,
                "r6_state": r6_query["state"],
                "occupied_hits": hits,
                "positive_obstacle_veto": bool(hits[0][0][-1]),
                "far_fractions": _far_fractions(samples),
                "far_valid_anchor_count": sum(value is not None for value in samples),
                "observed_support_points": support_points,
                "reason_codes": [],
            }
        )
    return validate_source_frame_record(
        _seal(
            {
                "schema": SOURCE_FRAME_SCHEMA,
                "reducer_id": REDUCER_ID,
                "parent_id": bundle["parent_id"],
                "video_id": bundle["video_id"],
                "timestamp_token": bundle["timestamp_token"],
                "physical_frame_id": bundle["physical_frame_id"],
                "source_frame_receipt_sha256": bundle["source_frame_receipt_sha256"],
                "candidate_frame_record_sha256": bundle["candidate_frame_record_sha256"],
                "prospective_bundle_sha256": bundle["content_sha256"],
                "r6_reducer_bundle_sha256": r6["content_sha256"],
                "input_bindings": {
                    "candidate_highres_depth_sha256": bundle["input_bindings"]["candidate_highres_depth_sha256"],
                    "anchored_candidate_depth_sha256": bundle["input_bindings"]["anchored_candidate_depth_sha256"],
                    "apple_depth_sha256": bundle["input_bindings"]["apple_depth_sha256"],
                    "confidence_sha256": bundle["input_bindings"]["confidence_sha256"],
                    "intrinsics_apple_sha256": bundle["input_bindings"]["intrinsics_apple_sha256"],
                    "intrinsics_highres_sha256": bundle["input_bindings"]["intrinsics_highres_sha256"],
                    "gravity_up_camera_xyz_sha256": bundle["input_bindings"]["gravity_up_camera_xyz_sha256"],
                },
                "grid": {
                    "occupied_pixels": list(OCCUPIED_PIXELS),
                    "occupied_height_m": list(OCCUPIED_HEIGHT_M),
                    "occupied_max_forward_m": list(OCCUPIED_MAX_FORWARD_M),
                    "clear_far_fraction": list(CLEAR_FAR_FRACTION),
                    "clear_far_depth_floor_m": list(CLEAR_FAR_DEPTH_FLOOR_M),
                    "clear_support_points": list(CLEAR_SUPPORT_POINTS),
                },
                "query_features": queries,
                "source_phase_has_label_input": False,
                "source_payload_roles": ["SEALED_CANDIDATE_DEPTH", "APPLE_DEPTH", "CONFIDENCE", "INTRINSICS", "TRAJECTORY"],
                "training_steps": 0,
                "network_requests": 0,
            }
        )
    )


def validate_source_frame_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, SOURCE_FRAME_SCHEMA)
    require(record["reducer_id"] == REDUCER_ID and record["source_phase_has_label_input"] is False, "R7_SOURCE_FIREWALL_DRIFT", "R7 source-phase identity/firewall drift")
    require(record["training_steps"] == record["network_requests"] == 0, "R7_SOURCE_FIREWALL_DRIFT", "R7 source phase cannot train or use network")
    queries = record["query_features"]
    require(isinstance(queries, list) and len(queries) == 9 and [row["grid_index"] for row in queries] == list(range(9)), "R7_SOURCE_QUERY_CARDINALITY", "R7 source frame must retain nine ordered queries")
    for query in queries:
        require(query["r6_state"] in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}, "R7_SOURCE_R6_STATE_INVALID", "R6 source state is invalid")
        if query["query_receipt"] is None:
            require(query["occupied_hits"] is query["far_fractions"] is query["positive_obstacle_veto"] is None and bool(query["reason_codes"]), "R7_UNAVAILABLE_QUERY_DRIFT", "unavailable query gained source evidence")
        else:
            prospective._validate_query(query["query_receipt"])
            hits = np.asarray(query["occupied_hits"])
            require(hits.shape == (4, 3, 3) and hits.dtype == np.bool_, "R7_OCCUPIED_GRID_DRIFT", "occupied evidence grid drift")
            fractions = query["far_fractions"]
            require(len(fractions) == 3 and all(_finite(item) and 0.0 <= float(item) <= 1.0 for item in fractions), "R7_FAR_FRACTION_INVALID", "far-censored fractions are invalid")
            require(query["positive_obstacle_veto"] == bool(hits[0, 0, -1]), "R7_CLEAR_VETO_DRIFT", "clear veto is not bound to weakest positive evidence")
    return record


def _truth_query_label(geometry: prospective._Geometry, plane: Mapping[str, Any], matrix: np.ndarray, query: Mapping[str, Any]) -> dict[str, Any]:
    points, pixels = prospective._surface(geometry, plane, query)
    local_fraction = prospective._local_valid_fraction(geometry, matrix, query)
    query_geometry = adapter._query_support_and_boundary(points, pixels, plane["normal_camera_xyz"], float(plane["camera_height_m"]), dict(query))
    origin, _, lateral, heading = adapter._query_receipt_vectors(dict(query))
    up = adapter._normalize_vector(query["virtual_query_frame"]["gravity_up_camera_xyz"], "R7_QUERY_FRAME_INVALID")
    path_origin = origin + float(query["path_lateral_offset_m"]) * lateral
    relative = points - path_origin
    along = relative @ heading
    perpendicular_vector = relative - along[:, None] * heading[None, :]
    perpendicular_ground = perpendicular_vector - (perpendicular_vector @ up)[:, None] * up[None, :]
    perpendicular = np.linalg.norm(perpendicular_ground, axis=1)
    normal = adapter._normalize_vector(plane["normal_camera_xyz"], "R7_TRUTH_PLANE_INVALID")
    heights = points @ normal + float(plane["camera_height_m"])
    obstacle = (
        (heights >= adapter.OBSTACLE_HEIGHT_RANGE_M[0])
        & (heights <= adapter.OBSTACLE_HEIGHT_RANGE_M[1])
        & (along >= adapter.MINIMUM_FORWARD_M)
        & (along <= adapter.HORIZON_M + adapter.GEOMETRY_ENDPOINT_TOLERANCE_M)
        & (perpendicular <= adapter.CAPSULE_RADIUS_M)
    )
    obstacle_count = int(np.sum(obstacle))
    support_count = int(query_geometry["query_support_points"])
    observed_forward = query_geometry["observed_forward_shape_m"]
    known_clear = (
        obstacle_count == 0
        and support_count >= adapter.MINIMUM_QUERY_SUPPORT_POINTS
        and observed_forward is not None
        and float(observed_forward) >= adapter.MINIMUM_QUERY_OBSERVED_FORWARD_M
        and local_fraction >= adapter.MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION
    )
    state = "OCCUPIED_OBSERVED" if obstacle_count >= MINIMUM_TRUTH_OBSTACLE_PIXELS else "CLEAR_OBSERVED" if known_clear else "UNKNOWN"
    return {
        "state": state,
        "obstacle_pixel_count": obstacle_count,
        "minimum_truth_obstacle_pixels": MINIMUM_TRUTH_OBSTACLE_PIXELS,
        "query_support_points": support_count,
        "observed_forward_m": None if observed_forward is None else float(observed_forward),
        "local_valid_fraction": float(local_fraction),
        "reason_codes": [] if state != "UNKNOWN" else ["FARO_QUERY_LABEL_NOT_IDENTIFIABLE"],
    }


def build_label_frame_record(
    source_frame_record: Mapping[str, Any],
    highres_faro_depth_mm: np.ndarray,
    intrinsics_highres_3x3: Any,
    gravity_up_camera_xyz: Any,
) -> dict[str, Any]:
    source = validate_source_frame_record(dict(source_frame_record))
    faro = np.asarray(highres_faro_depth_mm)
    require(faro.shape == adapter.HIGHRES_SHAPE_HW and faro.dtype == np.uint16, "R7_FARO_DEPTH_INVALID", "R7 label FARO must be uint16 1440x1920")
    matrix = _matrix(intrinsics_highres_3x3, "intrinsics_highres_3x3")
    gravity = adapter._normalize_vector(gravity_up_camera_xyz, "R7_GRAVITY_INVALID")
    require(adapter.canonical_sha256(matrix) == source["input_bindings"]["intrinsics_highres_sha256"], "R7_LABEL_INTRINSICS_LINEAGE_DRIFT", "R7 label intrinsics drift")
    require(adapter.canonical_sha256(gravity) == source["input_bindings"]["gravity_up_camera_xyz_sha256"], "R7_LABEL_GRAVITY_LINEAGE_DRIFT", "R7 label gravity drift")
    faro_m = np.ascontiguousarray(faro.astype(np.float64) / 1000.0, dtype=np.float64)
    plane = prospective._fit_depth_plane(faro_m, matrix, gravity)
    geometry = prospective._build_geometry(faro_m, adapter.canonical_sha256(faro_m), matrix) if plane["evaluable"] else None
    labels = []
    for query in source["query_features"]:
        if query["query_receipt"] is None:
            label = {"state": "UNKNOWN", "obstacle_pixel_count": 0, "minimum_truth_obstacle_pixels": MINIMUM_TRUTH_OBSTACLE_PIXELS, "query_support_points": 0, "observed_forward_m": None, "local_valid_fraction": 0.0, "reason_codes": ["SOURCE_QUERY_FRAME_UNAVAILABLE"]}
        elif geometry is None:
            label = {"state": "UNKNOWN", "obstacle_pixel_count": 0, "minimum_truth_obstacle_pixels": MINIMUM_TRUTH_OBSTACLE_PIXELS, "query_support_points": 0, "observed_forward_m": None, "local_valid_fraction": 0.0, "reason_codes": list(plane["reason_codes"])}
        else:
            label = _truth_query_label(geometry, plane, matrix, query["query_receipt"])
        labels.append({"grid_index": query["grid_index"], "query_id": query["query_id"], **label})
    return validate_label_frame_record(
        _seal(
            {
                "schema": LABEL_FRAME_SCHEMA,
                "reducer_id": REDUCER_ID,
                "parent_id": source["parent_id"],
                "video_id": source["video_id"],
                "timestamp_token": source["timestamp_token"],
                "physical_frame_id": source["physical_frame_id"],
                "source_frame_record_sha256": source["content_sha256"],
                "highres_faro_depth_sha256": adapter.canonical_sha256(faro),
                "truth_plane": plane,
                "query_labels": labels,
                "source_phase_reselection": False,
                "unknown_is_negative": False,
            }
        ),
        source,
    )


def validate_label_frame_record(value: Any, source_frame_record: Mapping[str, Any]) -> dict[str, Any]:
    source = validate_source_frame_record(dict(source_frame_record))
    record = _validate_seal(value, LABEL_FRAME_SCHEMA)
    require(record["reducer_id"] == REDUCER_ID and record["source_frame_record_sha256"] == source["content_sha256"], "R7_LABEL_SOURCE_LINEAGE_DRIFT", "R7 label is not bound to its sealed source frame")
    require(record["physical_frame_id"] == source["physical_frame_id"] and record["source_phase_reselection"] is False and record["unknown_is_negative"] is False, "R7_LABEL_FIREWALL_DRIFT", "R7 label phase changed source decisions or UNKNOWN semantics")
    labels = record["query_labels"]
    require(len(labels) == 9 and [row["grid_index"] for row in labels] == list(range(9)), "R7_LABEL_QUERY_CARDINALITY", "R7 labels must retain nine ordered queries")
    for feature, label in zip(source["query_features"], labels, strict=True):
        require(feature["query_id"] == label["query_id"] and label["state"] in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}, "R7_LABEL_QUERY_ALIGNMENT_DRIFT", "R7 feature/label query alignment drift")
    return record


def candidate_configs() -> list[tuple[int, int, int, int, int, int]]:
    return list(product(range(4), range(3), range(3), range(3), range(3), range(3)))


def predict_query_state(feature: Mapping[str, Any], config: tuple[int, int, int, int, int, int]) -> str:
    base = str(feature["r6_state"])
    require(base in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}, "R7_BASE_STATE_INVALID", "R7 base state is invalid")
    if base != "UNKNOWN":
        return base
    if feature["query_receipt"] is None:
        return "UNKNOWN"
    occupied_pixel_index, occupied_height_index, occupied_forward_index, clear_fraction_index, clear_floor_index, clear_support_index = config
    occupied = bool(feature["occupied_hits"][occupied_pixel_index][occupied_height_index][occupied_forward_index])
    clear = (
        not bool(feature["positive_obstacle_veto"])
        and int(feature["far_valid_anchor_count"]) >= MINIMUM_FAR_VISIBLE_ANCHORS
        and float(feature["far_fractions"][clear_floor_index]) >= CLEAR_FAR_FRACTION[clear_fraction_index]
        and int(feature["observed_support_points"]) >= CLEAR_SUPPORT_POINTS[clear_support_index]
    )
    return "OCCUPIED_OBSERVED" if occupied else "CLEAR_OBSERVED" if clear else "UNKNOWN"


def _wilson_lower(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    z = WILSON_Z_ONE_SIDED_95
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    return max(0.0, (center - radius) / denominator)


def _metrics(rows: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]], config: tuple[int, int, int, int, int, int]) -> dict[str, Any]:
    predicted = [(parent, predict_query_state(feature, config), label["state"]) for parent, feature, label in rows]
    false_clear = sum(prediction == "CLEAR_OBSERVED" and truth == "OCCUPIED_OBSERVED" for _, prediction, truth in predicted)
    occupied_tp = sum(prediction == truth == "OCCUPIED_OBSERVED" for _, prediction, truth in predicted)
    occupied_fp = sum(prediction == "OCCUPIED_OBSERVED" and truth == "CLEAR_OBSERVED" for _, prediction, truth in predicted)
    occupied_total = occupied_tp + occupied_fp
    definite = sum(prediction != "UNKNOWN" for _, prediction, _ in predicted)
    baseline_definite = sum(feature["r6_state"] != "UNKNOWN" for _, feature, _ in rows)
    truth_definite = sum(truth != "UNKNOWN" for _, _, truth in predicted)
    by_parent: defaultdict[str, list[bool]] = defaultdict(list)
    baseline_by_parent: defaultdict[str, list[bool]] = defaultdict(list)
    truth_by_parent: defaultdict[str, list[bool]] = defaultdict(list)
    for (parent, feature, _), (_, prediction, truth) in zip(rows, predicted, strict=True):
        by_parent[parent].append(prediction != "UNKNOWN")
        baseline_by_parent[parent].append(feature["r6_state"] != "UNKNOWN")
        truth_by_parent[parent].append(truth != "UNKNOWN")
    parent_coverage = {parent: sum(values) / len(values) for parent, values in sorted(by_parent.items())}
    parent_baseline = {parent: sum(baseline_by_parent[parent]) / len(baseline_by_parent[parent]) for parent in sorted(baseline_by_parent)}
    parent_macro_coverage = float(np.median(list(parent_coverage.values()))) if parent_coverage else 0.0
    parent_macro_baseline = float(np.median(list(parent_baseline.values()))) if parent_baseline else 0.0
    return {
        "query_count": len(rows),
        "truth_definite_query_count": truth_definite,
        "evaluable_parent_count": sum(any(values) for values in truth_by_parent.values()),
        "false_clear_count": false_clear,
        "occupied_true_positive_count": occupied_tp,
        "occupied_false_positive_count": occupied_fp,
        "occupied_precision": None if occupied_total == 0 else occupied_tp / occupied_total,
        "occupied_precision_wilson_lower_95": _wilson_lower(occupied_tp, occupied_total),
        "definite_query_count": definite,
        "baseline_definite_query_count": baseline_definite,
        "definite_query_coverage": definite / len(rows) if rows else 0.0,
        "baseline_definite_query_coverage": baseline_definite / len(rows) if rows else 0.0,
        "definite_query_coverage_increase_absolute": (definite - baseline_definite) / len(rows) if rows else 0.0,
        "parent_macro_definite_query_coverage": parent_macro_coverage,
        "parent_macro_baseline_definite_query_coverage": parent_macro_baseline,
        "parent_macro_definite_query_coverage_increase_absolute": parent_macro_coverage - parent_macro_baseline,
        "parent_definite_query_coverage": parent_coverage,
        "parent_baseline_definite_query_coverage": parent_baseline,
    }


def _flatten_join(source_records: Sequence[Mapping[str, Any]], label_records: Sequence[Mapping[str, Any]]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    sources = [validate_source_frame_record(dict(record)) for record in source_records]
    by_source_hash = {record["content_sha256"]: record for record in sources}
    require(len(by_source_hash) == len(sources), "R7_DUPLICATE_SOURCE_FRAME", "R7 source frames are duplicated")
    labels = []
    for raw in label_records:
        require(isinstance(raw, Mapping) and raw.get("source_frame_record_sha256") in by_source_hash, "R7_LABEL_SOURCE_MISSING", "R7 label has no bound source frame")
        labels.append(validate_label_frame_record(dict(raw), by_source_hash[str(raw["source_frame_record_sha256"])]))
    require(len(labels) == len(sources), "R7_SOURCE_LABEL_CARDINALITY", "R7 source/label frame cardinality drift")
    rows = []
    for label in labels:
        source = by_source_hash[label["source_frame_record_sha256"]]
        for feature, truth in zip(source["query_features"], label["query_labels"], strict=True):
            rows.append((source["parent_id"], feature, truth))
    return rows


def run_lopo_canary(source_records: Sequence[Mapping[str, Any]], label_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = _flatten_join(source_records, label_records)
    parents = sorted({parent for parent, _, _ in rows})
    require(len(parents) == 8 and len(rows) == 1899, "R7_FIT_COHORT_DRIFT", "R7 LOPO requires exact 8-parent/1899-query fit cohort")
    configs = candidate_configs()
    selected_by_parent: dict[str, list[int] | None] = {}
    held_predictions: list[tuple[str, Mapping[str, Any], Mapping[str, Any], str]] = []
    train_selection_receipts: dict[str, Any] = {}
    for held_parent in parents:
        train = [(parent, feature, label) for parent, feature, label in rows if parent != held_parent]
        eligible: list[tuple[tuple[float, float, tuple[int, ...]], tuple[int, ...], dict[str, Any]]] = []
        for config in configs:
            metrics = _metrics(train, config)
            if metrics["false_clear_count"] == 0:
                key = (-float(metrics["occupied_precision_wilson_lower_95"]), -float(metrics["definite_query_coverage"]), tuple(config))
                eligible.append((key, config, metrics))
        if not eligible:
            selected_by_parent[held_parent] = None
            train_selection_receipts[held_parent] = {"eligible_config_count": 0, "selected_config": None}
            for parent, feature, label in rows:
                if parent == held_parent:
                    held_predictions.append((parent, feature, label, str(feature["r6_state"])))
            continue
        eligible.sort(key=lambda item: item[0])
        _, selected, train_metrics = eligible[0]
        selected_by_parent[held_parent] = list(selected)
        train_selection_receipts[held_parent] = {"eligible_config_count": len(eligible), "selected_config": list(selected), "selected_train_metrics": train_metrics}
        for parent, feature, label in rows:
            if parent == held_parent:
                held_predictions.append((parent, feature, label, predict_query_state(feature, selected)))

    false_clear = sum(prediction == "CLEAR_OBSERVED" and label["state"] == "OCCUPIED_OBSERVED" for _, _, label, prediction in held_predictions)
    occupied_tp = sum(prediction == label["state"] == "OCCUPIED_OBSERVED" for _, _, label, prediction in held_predictions)
    occupied_fp = sum(prediction == "OCCUPIED_OBSERVED" and label["state"] == "CLEAR_OBSERVED" for _, _, label, prediction in held_predictions)
    occupied_total = occupied_tp + occupied_fp
    definite = sum(prediction != "UNKNOWN" for _, _, _, prediction in held_predictions)
    baseline_definite = sum(feature["r6_state"] != "UNKNOWN" for _, feature, _, _ in held_predictions)
    truth_definite = sum(label["state"] != "UNKNOWN" for _, _, label, _ in held_predictions)
    label_counts = Counter(label["state"] for _, _, label, _ in held_predictions)
    state_counts = Counter(prediction for _, _, _, prediction in held_predictions)
    parent_rows: dict[str, dict[str, Any]] = {}
    for parent in parents:
        subset = [(feature, label, prediction) for observed, feature, label, prediction in held_predictions if observed == parent]
        parent_rows[parent] = {
            "query_count": len(subset),
            "truth_definite_query_count": sum(label["state"] != "UNKNOWN" for _, label, _ in subset),
            "baseline_definite_query_coverage": sum(feature["r6_state"] != "UNKNOWN" for feature, _, _ in subset) / len(subset),
            "r7_definite_query_coverage": sum(prediction != "UNKNOWN" for _, _, prediction in subset) / len(subset),
        }
        parent_rows[parent]["definite_query_coverage_increase_absolute"] = parent_rows[parent]["r7_definite_query_coverage"] - parent_rows[parent]["baseline_definite_query_coverage"]
    parent_macro_increase = float(np.median([row["definite_query_coverage_increase_absolute"] for row in parent_rows.values()]))
    evaluable_parents = sum(row["truth_definite_query_count"] > 0 for row in parent_rows.values())
    precision = None if occupied_total == 0 else occupied_tp / occupied_total
    gates = {
        "minimum_evaluable_parents": evaluable_parents >= 6,
        "false_clear_count_zero": false_clear == 0,
        "minimum_occupied_precision": precision is not None and precision >= 0.9,
        "minimum_definite_query_coverage_increase_absolute": parent_macro_increase >= 0.05,
        "unknown_is_negative": False,
    }
    passed = all(value is True for key, value in gates.items() if key != "unknown_is_negative")
    return _seal(
        {
            "schema": CANARY_RESULT_SCHEMA,
            "terminal": "TARO_O1R_R7_FIT_LOPO_CANARY_PASS" if passed else "TARO_O1R_R7_FIT_LOPO_CANARY_FAIL",
            "execution_valid": True,
            "passed": passed,
            "promotion_authorized": False,
            "parent_count": 8,
            "frame_count": len(source_records),
            "query_count": len(rows),
            "truth_definite_query_count": truth_definite,
            "label_state_counts": {state: int(label_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
            "r7_state_counts": {state: int(state_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
            "baseline_definite_query_count": baseline_definite,
            "r7_definite_query_count": definite,
            "definite_query_coverage_increase_absolute": (definite - baseline_definite) / len(rows),
            "parent_macro_definite_query_coverage_increase_absolute": parent_macro_increase,
            "false_clear_count": false_clear,
            "occupied_true_positive_count": occupied_tp,
            "occupied_false_positive_count": occupied_fp,
            "occupied_precision": precision,
            "occupied_precision_wilson_lower_95": _wilson_lower(occupied_tp, occupied_total),
            "evaluable_parent_count": evaluable_parents,
            "parent_results": parent_rows,
            "selected_config_by_held_parent": selected_by_parent,
            "train_selection_receipts": train_selection_receipts,
            "gates": gates,
            "unknown_is_negative": False,
            "selection_used_held_parent_labels": False,
            "training_steps": 0,
            "network_requests": 0,
            "claim_ceiling": "Fit-only nested-parent CPU hypothesis canary; no promotion, deployment, device, product, or safety claim.",
            "unique_successor": "NEW_PARENT_DISJOINT_UNTOUCHED_R7_CONFIRMATION_LOCK" if passed else "R7_FAILURE_ANALYSIS_OR_NEW_TASK_LOCK",
        }
    )


__all__ = [
    "CANARY_RESULT_SCHEMA", "LABEL_FRAME_SCHEMA", "REDUCER_ID", "SOURCE_FRAME_SCHEMA", "R7CanaryError",
    "build_label_frame_record", "build_source_frame_record", "candidate_configs", "predict_query_state",
    "run_lopo_canary", "validate_label_frame_record", "validate_source_frame_record",
]
