"""Evidence-bound candidate correspondence between masks and detector instances.

This module deliberately does not create truth labels.  It scores one-frame
component/detection candidates, preserves missing evidence as ``ABSTAIN``, and
applies a deterministic one-to-one assignment after pair scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


MATCH = "MATCH"
NO_MATCH = "NO_MATCH"
ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class CorrespondenceThresholds:
    minimum_pair_score: float = 0.58
    minimum_score_margin: float = 0.08
    minimum_present_evidence: int = 2
    minimum_mask_iou: float = 0.03
    minimum_component_coverage: float = 0.20
    minimum_box_coverage: float = 0.05
    maximum_foot_point_distance_pixels: float = 28.0
    strong_no_match_foot_point_distance_pixels: float = 72.0
    minimum_temporal_iou: float = 0.10
    minimum_flow_iou: float = 0.10
    depth_relative_tolerance: float = 0.20
    depth_absolute_tolerance: float = 0.25
    detection_track_iou: float = 0.10
    maximum_track_frame_gap: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CorrespondenceThresholds":
        if value is None:
            return cls()
        names = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs: dict[str, Any] = {}
        for name in names:
            if name in value:
                kwargs[name] = value[name]
        result = cls(**kwargs)
        if result.minimum_present_evidence < 1:
            raise ValueError("minimum_present_evidence must be positive")
        if result.maximum_track_frame_gap < 0:
            raise ValueError("maximum_track_frame_gap must be non-negative")
        for name in names - {"minimum_present_evidence", "maximum_track_frame_gap"}:
            if not math.isfinite(float(getattr(result, name))):
                raise ValueError(f"threshold {name} must be finite")
        return result


@dataclass(frozen=True)
class EvidenceWeights:
    mask_iou: float = 0.22
    component_coverage: float = 0.18
    box_coverage: float = 0.10
    foot_point: float = 0.18
    depth_consistency: float = 0.10
    temporal_continuity: float = 0.08
    optical_flow: float = 0.06
    class_compatibility: float = 0.08

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "EvidenceWeights":
        if value is None:
            return cls()
        names = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs = {name: value[name] for name in names if name in value}
        result = cls(**kwargs)
        if any(float(getattr(result, name)) < 0 for name in names):
            raise ValueError("evidence weights must be non-negative")
        if sum(float(getattr(result, name)) for name in names) <= 0:
            raise ValueError("at least one evidence weight must be positive")
        return result


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _normalise_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ")
    return " ".join(text.split()) or None


def _as_box(value: Any, *, label: str = "box") -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{label} must contain four coordinates")
    coords = tuple(float(item) for item in value)
    if not all(_finite(item) for item in coords):
        raise ValueError(f"{label} must contain finite coordinates")
    x0, y0, x1, y1 = coords
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{label} must have positive area")
    return coords


def _box_area(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = _as_box(box)
    return max(0.0, (x1 - x0) * (y1 - y0))


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """IoU for half-open rectangles in one coordinate system."""

    lx0, ly0, lx1, ly1 = _as_box(left, label="left box")
    rx0, ry0, rx1, ry1 = _as_box(right, label="right box")
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = _box_area((lx0, ly0, lx1, ly1)) + _box_area((rx0, ry0, rx1, ry1)) - intersection
    return float(intersection / union) if union else 0.0


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_value = np.asarray(left, dtype=bool)
    right_value = np.asarray(right, dtype=bool)
    if left_value.shape != right_value.shape or left_value.ndim != 2:
        raise ValueError("masks must be two-dimensional and have equal shape")
    union = int(np.count_nonzero(left_value | right_value))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(left_value & right_value) / union)


def rasterize_box(
    box: Sequence[float],
    shape: tuple[int, int],
    *,
    source_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Rasterize a source or analysis-grid half-open box conservatively."""

    x0, y0, x1, y1 = _as_box(box)
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("mask shape must be positive")
    if source_shape is not None:
        source_height, source_width = int(source_shape[0]), int(source_shape[1])
        if source_height <= 0 or source_width <= 0:
            raise ValueError("source shape must be positive")
        x0 *= width / source_width
        x1 *= width / source_width
        y0 *= height / source_height
        y1 *= height / source_height
    ix0 = max(0, min(width, int(math.floor(x0))))
    ix1 = max(0, min(width, int(math.ceil(x1))))
    iy0 = max(0, min(height, int(math.floor(y0))))
    iy1 = max(0, min(height, int(math.ceil(y1))))
    mask = np.zeros((height, width), dtype=bool)
    if ix1 > ix0 and iy1 > iy0:
        mask[iy0:iy1, ix0:ix1] = True
    return mask


def _foot_point_for_mask(mask: np.ndarray) -> tuple[float, float] | None:
    value = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(value)
    if not len(xs):
        return None
    bottom = float(np.percentile(ys, 90))
    bottom_x = xs[ys >= bottom]
    if not len(bottom_x):
        bottom_x = xs
    return float(np.median(bottom_x)), float(np.max(ys))


def _foot_point_for_box(box: Sequence[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = _as_box(box)
    return (float((x0 + x1) / 2.0), float(y1))


def _mask_pair_metrics(
    component_mask: np.ndarray,
    box_mask: np.ndarray,
    detection_box: Sequence[float],
) -> dict[str, float | int | None]:
    component = np.asarray(component_mask, dtype=bool)
    box_value = np.asarray(box_mask, dtype=bool)
    if component.ndim != 2 or box_value.shape != component.shape:
        raise ValueError("component and box masks must be two-dimensional and shape-aligned")
    component_pixels = int(np.count_nonzero(component))
    box_pixels = int(np.count_nonzero(box_value))
    intersection = int(np.count_nonzero(component & box_value))
    union = int(np.count_nonzero(component | box_value))
    component_y, component_x = np.nonzero(component)
    component_box = (
        (float(np.min(component_x)), float(np.min(component_y)),
         float(np.max(component_x) + 1), float(np.max(component_y) + 1))
        if component_pixels else None
    )
    bbox_overlap = bbox_iou(component_box, detection_box) if component_box else 0.0
    component_foot = _foot_point_for_mask(component)
    box_foot = _foot_point_for_box(detection_box)
    foot_distance = (
        float(math.hypot(component_foot[0] - box_foot[0], component_foot[1] - box_foot[1]))
        if component_foot is not None
        else None
    )
    return {
        "mask_iou": float(intersection / union) if union else 0.0,
        "mask_intersection_pixels": intersection,
        "component_coverage": float(intersection / component_pixels) if component_pixels else 0.0,
        "box_coverage": float(intersection / box_pixels) if box_pixels else 0.0,
        "component_pixels": component_pixels,
        "box_pixels": box_pixels,
        "component_bbox_iou": float(bbox_overlap),
        "foot_point_distance_pixels": foot_distance,
    }


def mask_box_metrics(component_mask: np.ndarray, detection_box: Sequence[float]) -> dict[str, float | int]:
    """Compute mask-vs-box geometry on the component's analysis grid."""

    component = np.asarray(component_mask, dtype=bool)
    if component.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    return _mask_pair_metrics(component, rasterize_box(detection_box, component.shape), detection_box)


def warp_mask(mask: np.ndarray, matrix_previous_to_current: Sequence[Sequence[float]]) -> np.ndarray:
    """Warp a previous analysis-grid mask with a finite 2x3 affine."""

    source = np.asarray(mask, dtype=bool)
    matrix = np.asarray(matrix_previous_to_current, dtype=np.float64)
    if source.ndim != 2 or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("mask must be 2D and affine must be finite 2x3")
    ys, xs = np.nonzero(source)
    warped = np.zeros_like(source, dtype=bool)
    if not len(xs):
        return warped
    mapped_x = np.rint(matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2]).astype(np.int64)
    mapped_y = np.rint(matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2]).astype(np.int64)
    valid = (
        (mapped_x >= 0)
        & (mapped_x < source.shape[1])
        & (mapped_y >= 0)
        & (mapped_y < source.shape[0])
    )
    warped[mapped_y[valid], mapped_x[valid]] = True
    return warped


def warp_box(box: Sequence[float], matrix_previous_to_current: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """Transform all four rectangle corners and return the enclosing box."""

    x0, y0, x1, y1 = _as_box(box)
    matrix = np.asarray(matrix_previous_to_current, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("affine must be finite 2x3")
    points = np.asarray([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float64)
    mapped_x = matrix[0, 0] * points[:, 0] + matrix[0, 1] * points[:, 1] + matrix[0, 2]
    mapped_y = matrix[1, 0] * points[:, 0] + matrix[1, 1] * points[:, 1] + matrix[1, 2]
    return (float(np.min(mapped_x)), float(np.min(mapped_y)), float(np.max(mapped_x)), float(np.max(mapped_y)))


def class_compatibility(
    component_class: Any,
    detection: Mapping[str, Any],
    mapping: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Map detector labels to the declared semantic class without using truth."""

    component = _normalise_label(component_class)
    config = mapping or {}
    label_to_semantic = {
        _normalise_label(key): _normalise_label(value)
        for key, value in dict(config.get("yolo_label_to_semantic", {})).items()
    }
    aliases: dict[str, set[str]] = {}
    for key, values in dict(config.get("segmentation_aliases", {})).items():
        normalised_key = _normalise_label(key)
        aliases[normalised_key or ""] = {
            item for item in (_normalise_label(value) for value in values) if item
        }
    direct = _normalise_label(
        detection.get("semantic_class")
        or detection.get("class_name")
        or detection.get("semantic_label")
    )
    label = _normalise_label(detection.get("label") or detection.get("name"))
    semantic = direct or (label_to_semantic.get(label) if label else None)
    if semantic is None and label is not None:
        for target, values in aliases.items():
            if label in values:
                semantic = target
                break
    if component is None or semantic is None:
        state = "UNKNOWN"
        score = None
    elif component == semantic:
        state = "COMPATIBLE"
        score = 1.0
    else:
        state = "INCOMPATIBLE"
        score = 0.0
    return {
        "state": state,
        "component_class": component,
        "detection_label": label,
        "detection_semantic_class": semantic,
        "score": score,
    }


def depth_consistency(
    component_depth: Mapping[str, Any] | None,
    detection_depth: Mapping[str, Any] | None,
    thresholds: CorrespondenceThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare optional depth summaries and preserve unknown as a third value."""

    cfg = thresholds if isinstance(thresholds, CorrespondenceThresholds) else CorrespondenceThresholds.from_mapping(thresholds)
    left = component_depth or {}
    right = detection_depth or {}
    left_cluster = left.get("depth_cluster_id")
    right_cluster = right.get("depth_cluster_id")
    left_depth = left.get("median_depth")
    right_depth = right.get("median_depth")
    if _finite(left_depth) and _finite(right_depth):
        delta = abs(float(left_depth) - float(right_depth))
        scale = max(abs(float(left_depth)), abs(float(right_depth)), 1e-6)
        relative = delta / scale
        consistent = delta <= cfg.depth_absolute_tolerance or relative <= cfg.depth_relative_tolerance
        return {
            "state": "CONSISTENT" if consistent else "INCONSISTENT",
            "score": 1.0 if consistent else 0.0,
            "component_depth_cluster_id": left_cluster,
            "detection_depth_cluster_id": right_cluster,
            "depth_delta": float(delta),
            "depth_relative_delta": float(relative),
        }
    if left_cluster is not None and right_cluster is not None:
        consistent = str(left_cluster) == str(right_cluster)
        return {
            "state": "CONSISTENT" if consistent else "INCONSISTENT",
            "score": 1.0 if consistent else 0.0,
            "component_depth_cluster_id": left_cluster,
            "detection_depth_cluster_id": right_cluster,
            "depth_delta": None,
            "depth_relative_delta": None,
        }
    return {
        "state": "UNKNOWN",
        "score": None,
        "component_depth_cluster_id": left_cluster,
        "detection_depth_cluster_id": right_cluster,
        "depth_delta": None,
        "depth_relative_delta": None,
    }


def _normalise_score(value: float | None, *, denominator: float) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value) / denominator))


def _flow_score(flow_evidence: Mapping[str, Any] | None) -> float | None:
    if not flow_evidence:
        return None
    values = [
        float(flow_evidence[name])
        for name in ("component_iou", "detection_iou", "pair_iou", "support")
        if _finite(flow_evidence.get(name))
    ]
    return float(np.mean(values)) if values else None


def score_pair(
    component: Mapping[str, Any],
    detection: Mapping[str, Any],
    *,
    thresholds: CorrespondenceThresholds | Mapping[str, Any] | None = None,
    weights: EvidenceWeights | Mapping[str, Any] | None = None,
    class_mapping: Mapping[str, Any] | None = None,
    component_depth: Mapping[str, Any] | None = None,
    detection_depth: Mapping[str, Any] | None = None,
    temporal_continuity: float | None = None,
    flow_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one auditable candidate pair row before one-to-one assignment."""

    cfg = thresholds if isinstance(thresholds, CorrespondenceThresholds) else CorrespondenceThresholds.from_mapping(thresholds)
    weight = weights if isinstance(weights, EvidenceWeights) else EvidenceWeights.from_mapping(weights)
    component_mask = np.asarray(component.get("mask"), dtype=bool)
    if component_mask.ndim != 2:
        raise ValueError("component.mask must be a two-dimensional array")
    box = detection.get("bbox_xyxy") or detection.get("box")
    if box is None:
        raise ValueError("detection must contain bbox_xyxy or box")
    cached_box_mask = detection.get("_box_mask")
    geometry = (
        _mask_pair_metrics(component_mask, np.asarray(cached_box_mask, dtype=bool), box)
        if cached_box_mask is not None
        else mask_box_metrics(component_mask, box)
    )
    compatibility = class_compatibility(
        component.get("semantic_class") or component.get("class_name") or component.get("predicted_class"),
        detection,
        class_mapping,
    )
    depth = depth_consistency(component_depth, detection_depth, cfg)
    flow_score = _flow_score(flow_evidence)
    evidence: dict[str, float | None] = {
        "mask_iou": _normalise_score(float(geometry["mask_iou"]), denominator=0.25),
        "component_coverage": _normalise_score(float(geometry["component_coverage"]), denominator=0.60),
        "box_coverage": _normalise_score(float(geometry["box_coverage"]), denominator=0.25),
        "foot_point": (
            max(0.0, min(1.0, 1.0 - float(geometry["foot_point_distance_pixels"]) / cfg.strong_no_match_foot_point_distance_pixels))
            if geometry["foot_point_distance_pixels"] is not None
            else None
        ),
        "depth_consistency": depth["score"],
        "temporal_continuity": (
            max(0.0, min(1.0, float(temporal_continuity)))
            if temporal_continuity is not None and _finite(temporal_continuity)
            else None
        ),
        "optical_flow": flow_score,
        "class_compatibility": compatibility["score"],
    }
    weights_by_name = {
        "mask_iou": weight.mask_iou,
        "component_coverage": weight.component_coverage,
        "box_coverage": weight.box_coverage,
        "foot_point": weight.foot_point,
        "depth_consistency": weight.depth_consistency,
        "temporal_continuity": weight.temporal_continuity,
        "optical_flow": weight.optical_flow,
        "class_compatibility": weight.class_compatibility,
    }
    present = [name for name, value in evidence.items() if value is not None]
    missing = [name for name, value in evidence.items() if value is None]
    denominator = sum(weights_by_name[name] for name in present)
    score = float(sum(float(evidence[name]) * weights_by_name[name] for name in present) / denominator) if denominator else 0.0
    geometry_core = (
        float(geometry["component_coverage"]) >= cfg.minimum_component_coverage
        and (
            float(geometry["mask_iou"]) >= cfg.minimum_mask_iou
            or float(geometry["box_coverage"]) >= cfg.minimum_box_coverage
            or (
                geometry["foot_point_distance_pixels"] is not None
                and float(geometry["foot_point_distance_pixels"]) <= cfg.maximum_foot_point_distance_pixels
            )
        )
    )
    strong_separation = (
        float(geometry["mask_iou"]) < cfg.minimum_mask_iou
        and float(geometry["component_coverage"]) < cfg.minimum_component_coverage
        and float(geometry["box_coverage"]) < cfg.minimum_box_coverage
        and geometry["foot_point_distance_pixels"] is not None
        and float(geometry["foot_point_distance_pixels"]) > cfg.strong_no_match_foot_point_distance_pixels
    )
    if compatibility["state"] == "INCOMPATIBLE":
        state, reason = NO_MATCH, "CLASS_INCOMPATIBLE"
    elif strong_separation:
        state, reason = NO_MATCH, "STRONG_GEOMETRIC_SEPARATION"
    elif depth["state"] == "INCONSISTENT" and not geometry_core:
        state, reason = NO_MATCH, "DEPTH_INCONSISTENT_WITH_WEAK_GEOMETRY"
    elif (
        compatibility["state"] == "COMPATIBLE"
        and geometry_core
        and len(present) >= cfg.minimum_present_evidence
        and score >= cfg.minimum_pair_score
    ):
        state, reason = MATCH, "GEOMETRY_AND_COMPATIBLE_CLASS"
    else:
        state, reason = ABSTAIN, "INSUFFICIENT_OR_AMBIGUOUS_EVIDENCE"
    return {
        "component_id": str(component.get("component_id")),
        "detection_id": str(detection.get("detection_id")),
        "component_track_id": component.get("temporal_track_id"),
        "detection_track_id": detection.get("temporal_track_id") or detection.get("track_id"),
        "semantic_class": component.get("semantic_class") or component.get("class_name") or component.get("predicted_class"),
        "detection_label": detection.get("label") or detection.get("name"),
        "state": state,
        "raw_state": state,
        "state_reason": reason,
        "selected": False,
        "score": score,
        "present_evidence": present,
        "missing_evidence": missing,
        "class_compatibility": compatibility,
        "depth_consistency": depth,
        "temporal_continuity": temporal_continuity,
        "optical_flow": dict(flow_evidence or {}) if flow_evidence else None,
        "metrics": geometry,
    }


def _pair_sort_key(row: Mapping[str, Any]) -> tuple[float, str, str]:
    return (-float(row.get("score") or 0.0), str(row.get("component_id")), str(row.get("detection_id")))


def assign_one_to_one(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    component_ids: Sequence[str] | None = None,
    detection_ids: Sequence[str] | None = None,
    minimum_score_margin: float = 0.08,
) -> dict[str, list[dict[str, Any]]]:
    """Finalize pair states and enforce one detection per component/detection per frame."""

    rows = [dict(row) for row in pair_rows]
    by_component: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_component.setdefault(str(row["component_id"]), []).append(row)
    eligible: list[dict[str, Any]] = []
    blocked_components: set[str] = set()
    for component_id, candidates in sorted(by_component.items()):
        matches = sorted((row for row in candidates if row.get("raw_state") == MATCH), key=_pair_sort_key)
        if len(matches) >= 2 and float(matches[0].get("score") or 0.0) - float(matches[1].get("score") or 0.0) < minimum_score_margin:
            blocked_components.add(component_id)
            for row in matches:
                row["state"] = ABSTAIN
                row["state_reason"] = "AMBIGUOUS_COMPONENT_CANDIDATES"
        elif matches:
            eligible.append(matches[0])
            for row in matches[1:]:
                row["state"] = ABSTAIN
                row["state_reason"] = "NOT_SELECTED_AFTER_COMPONENT_RANKING"
    assigned_components: set[str] = set()
    assigned_detections: set[str] = set()
    by_detection: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_detection.setdefault(str(row["detection_id"]), []).append(row)
    for detection_id, candidates in sorted(by_detection.items()):
        candidates = sorted(candidates, key=_pair_sort_key)
        if len(candidates) >= 2 and float(candidates[0].get("score") or 0.0) - float(candidates[1].get("score") or 0.0) < minimum_score_margin:
            for row in candidates:
                row["state"] = ABSTAIN
                row["state_reason"] = "AMBIGUOUS_DETECTION_REUSE"
            continue
        winner = candidates[0]
        component_id = str(winner["component_id"])
        assigned_components.add(component_id)
        assigned_detections.add(detection_id)
        winner["state"] = MATCH
        winner["selected"] = True
        winner["state_reason"] = "ONE_TO_ONE_SELECTED_MATCH"
        for row in candidates[1:]:
            row["state"] = ABSTAIN
            row["state_reason"] = "DETECTION_ALREADY_ASSIGNED"
    component_universe = set(component_ids or by_component)
    detection_universe = set(detection_ids or {str(row["detection_id"]) for row in rows})
    component_rows: list[dict[str, Any]] = []
    for component_id in sorted(component_universe):
        candidates = by_component.get(component_id, [])
        selected = next((row for row in candidates if row.get("selected")), None)
        if selected is not None:
            state = MATCH
            reason = "ONE_TO_ONE_SELECTED_MATCH"
            detection_id = selected.get("detection_id")
        elif not candidates:
            state, reason, detection_id = ABSTAIN, "NO_DETECTION_CANDIDATE", None
        elif component_id in blocked_components or any(row.get("state_reason") == "AMBIGUOUS_DETECTION_REUSE" for row in candidates):
            state, reason, detection_id = ABSTAIN, "CORRESPONDENCE_AMBIGUOUS", None
        elif all(row.get("state") == NO_MATCH for row in candidates):
            state, reason, detection_id = NO_MATCH, "ALL_CANDIDATES_NO_MATCH", None
        else:
            state, reason, detection_id = ABSTAIN, "NO_UNIQUE_MATCH", None
        source = selected or (candidates[0] if candidates else {})
        component_rows.append({
            "component_id": component_id,
            "detection_id": detection_id,
            "component_track_id": source.get("component_track_id"),
            "detection_track_id": source.get("detection_track_id") if selected else None,
            "semantic_class": source.get("semantic_class"),
            "state": state,
            "state_reason": reason,
            "score": source.get("score") if selected else None,
            "selected": bool(selected),
        })
    detection_rows: list[dict[str, Any]] = []
    for detection_id in sorted(detection_universe):
        candidates = [row for row in rows if str(row["detection_id"]) == detection_id]
        selected = next((row for row in candidates if row.get("selected")), None)
        detection_rows.append({
            "detection_id": detection_id,
            "component_id": selected.get("component_id") if selected else None,
            "detection_track_id": (selected or (candidates[0] if candidates else {})).get("detection_track_id"),
            "state": MATCH if selected else ABSTAIN,
            "state_reason": "ONE_TO_ONE_SELECTED_MATCH" if selected else "NO_UNIQUE_COMPONENT_MATCH",
            "score": selected.get("score") if selected else None,
            "selected": bool(selected),
        })
    return {
        "pair_rows": rows,
        "component_rows": component_rows,
        "detection_rows": detection_rows,
    }


def annotate_frame(
    components: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    *,
    thresholds: CorrespondenceThresholds | Mapping[str, Any] | None = None,
    weights: EvidenceWeights | Mapping[str, Any] | None = None,
    class_mapping: Mapping[str, Any] | None = None,
    component_depth: Mapping[str, Mapping[str, Any]] | None = None,
    detection_depth: Mapping[str, Mapping[str, Any]] | None = None,
    temporal_by_pair: Mapping[tuple[str, str], float | None] | None = None,
    flow_by_pair: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Score and assign all candidate pairs for one frame."""

    rows: list[dict[str, Any]] = []
    for component in sorted(components, key=lambda item: str(item.get("component_id"))):
        component_id = str(component.get("component_id"))
        for detection in sorted(detections, key=lambda item: str(item.get("detection_id"))):
            detection_id = str(detection.get("detection_id"))
            key = (component_id, detection_id)
            rows.append(
                score_pair(
                    component,
                    detection,
                    thresholds=thresholds,
                    weights=weights,
                    class_mapping=class_mapping,
                    component_depth=(component_depth or {}).get(component_id),
                    detection_depth=(detection_depth or {}).get(detection_id),
                    temporal_continuity=(temporal_by_pair or {}).get(key),
                    flow_evidence=(flow_by_pair or {}).get(key),
                )
            )
    cfg = thresholds if isinstance(thresholds, CorrespondenceThresholds) else CorrespondenceThresholds.from_mapping(thresholds)
    return assign_one_to_one(
        rows,
        component_ids=[str(item.get("component_id")) for item in components],
        detection_ids=[str(item.get("detection_id")) for item in detections],
        minimum_score_margin=cfg.minimum_score_margin,
    )
