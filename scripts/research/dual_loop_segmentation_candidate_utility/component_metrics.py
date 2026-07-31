"""Pixel and connected-component metrics for candidate utility R0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover - the fallback keeps the contract portable
    _ndimage = None


@dataclass
class Component:
    """A connected component with its analysis-grid mask retained for matching."""

    index: int
    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]


def _ratio(numerator: int, denominator: int, *, empty_value: float | None = None) -> float | None:
    if denominator:
        return float(numerator / denominator)
    return empty_value


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _erode_8(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    result = np.ones(mask.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            result &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return result


def pixel_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Return confusion, overlap, false-positive area and boundary-F1 metrics."""

    pred = np.asarray(predicted, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("predicted and truth must be two-dimensional masks with equal shape")
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    tn = int(pred.size - tp - fp - fn)
    both_empty = tp + fp + fn == 0
    precision = _ratio(tp, tp + fp, empty_value=1.0 if both_empty else None)
    recall = _ratio(tp, tp + fn, empty_value=1.0 if both_empty else None)
    iou = _ratio(tp, tp + fp + fn, empty_value=1.0 if both_empty else None)
    f1 = _f1(precision, recall)

    pred_boundary = pred & ~_erode_8(pred)
    truth_boundary = target & ~_erode_8(target)
    boundary_tp = int(np.count_nonzero(pred_boundary & truth_boundary))
    boundary_fp = int(np.count_nonzero(pred_boundary & ~truth_boundary))
    boundary_fn = int(np.count_nonzero(~pred_boundary & truth_boundary))
    boundary_empty = not truth_boundary.any() and not pred_boundary.any()
    boundary_precision = _ratio(boundary_tp, boundary_tp + boundary_fp, empty_value=1.0 if boundary_empty else None)
    boundary_recall = _ratio(boundary_tp, boundary_tp + boundary_fn, empty_value=1.0 if boundary_empty else None)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": int(np.count_nonzero(pred)),
        "truth_pixels": int(np.count_nonzero(target)),
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "false_positive_area_fraction": float(fp / pred.size),
        "boundary_tp": boundary_tp,
        "boundary_fp": boundary_fp,
        "boundary_fn": boundary_fn,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": _f1(boundary_precision, boundary_recall),
    }


def connected_components(mask: np.ndarray, *, connectivity: int = 8) -> list[Component]:
    """Extract deterministic row-major connected components with a portable fallback."""

    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")
    if _ndimage is not None:
        structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
        )
        labels, count = _ndimage.label(value, structure=structure)
        components: list[Component] = []
        for label_index, bounds in enumerate(_ndimage.find_objects(labels), start=1):
            if bounds is None:
                continue
            local = labels[bounds] == label_index
            area = int(np.count_nonzero(local))
            if area == 0:
                continue
            y_slice, x_slice = bounds
            component_mask = np.zeros_like(value, dtype=bool)
            component_mask[y_slice, x_slice] = local
            components.append(
                Component(
                    index=len(components),
                    mask=component_mask,
                    area=area,
                    bbox=(x_slice.start, y_slice.start, x_slice.stop, y_slice.stop),
                )
            )
        return components
    height, width = value.shape
    visited = np.zeros_like(value, dtype=bool)
    offsets: tuple[tuple[int, int], ...] = (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if connectivity == 4
        else ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    )
    components: list[Component] = []
    for y in range(height):
        for x in range(width):
            if not value[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            min_x = max_x = x
            min_y = max_y = y
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and value[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            component_mask = np.zeros_like(value, dtype=bool)
            ys, xs = zip(*pixels)
            component_mask[np.asarray(ys), np.asarray(xs)] = True
            components.append(
                Component(
                    index=len(components),
                    mask=component_mask,
                    area=len(pixels),
                    bbox=(min_x, min_y, max_x + 1, max_y + 1),
                )
            )
    return components


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_value = np.asarray(left, dtype=bool)
    right_value = np.asarray(right, dtype=bool)
    if left_value.shape != right_value.shape:
        raise ValueError("mask shapes must match")
    union = int(np.count_nonzero(left_value | right_value))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(left_value & right_value) / union)


def component_metrics(predicted: np.ndarray, truth: np.ndarray, *, hit_rule: str = "intersection") -> dict[str, Any]:
    """Score predicted and truth components by the frozen positive-intersection rule."""

    if hit_rule != "intersection":
        raise ValueError("R0 only supports the frozen positive-intersection component rule")
    predicted_components = connected_components(predicted)
    truth_components = connected_components(truth)
    predicted_hits = [
        any(np.count_nonzero(component.mask & target.mask) > 0 for target in truth_components)
        for component in predicted_components
    ]
    truth_hits = [
        any(np.count_nonzero(component.mask & candidate.mask) > 0 for candidate in predicted_components)
        for component in truth_components
    ]
    predicted_count = len(predicted_components)
    truth_count = len(truth_components)
    return {
        "predicted_component_count": predicted_count,
        "truth_component_count": truth_count,
        "hit_predicted_component_count": int(sum(predicted_hits)),
        "hit_truth_component_count": int(sum(truth_hits)),
        "component_precision": _ratio(sum(predicted_hits), predicted_count, empty_value=1.0 if truth_count == 0 else None),
        "component_recall": _ratio(sum(truth_hits), truth_count, empty_value=1.0 if predicted_count == 0 else None),
        "false_activation_component_count": int(predicted_count - sum(predicted_hits)),
    }


def bbox_gap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    """Euclidean gap between two half-open rectangles in analysis-grid pixels."""

    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    dx = max(rx0 - lx1, lx0 - rx1, 0)
    dy = max(ry0 - ly1, ly0 - ry1, 0)
    return float((dx * dx + dy * dy) ** 0.5)


def component_records(
    candidate_by_class: dict[str, np.ndarray],
    truth_mask: np.ndarray,
    yolo_mask: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    *,
    source_id: str,
    frame_id: int,
) -> list[dict[str, Any]]:
    """Create the component ledger rows required by the R0 contract."""

    yolo_components = connected_components(yolo_mask)
    records: list[dict[str, Any]] = []
    for class_name, class_mask in candidate_by_class.items():
        for component in connected_components(class_mask):
            overlap = int(np.count_nonzero(component.mask & truth_mask))
            nearest_gap = min(
                (bbox_gap(component.bbox, other.bbox) for other in yolo_components),
                default=None,
            )
            values = confidence[component.mask]
            margins = margin[component.mask]
            records.append(
                {
                    "component_id": f"{source_id}:{frame_id}:{class_name}:{component.index}",
                    "source_id": source_id,
                    "frame_id": int(frame_id),
                    "class_name": class_name,
                    "component_index": int(component.index),
                    "area_pixels": int(component.area),
                    "bbox_xyxy": list(component.bbox),
                    "top1_confidence_median": float(np.median(values)) if values.size else None,
                    "top1_top2_margin_median": float(np.median(margins)) if margins.size else None,
                    "truth_intersection_pixels": overlap,
                    "truth_intersects": bool(overlap > 0),
                    "truth_iou": mask_iou(component.mask, truth_mask) if overlap else 0.0,
                    "nearest_yolo_box_distance_pixels": nearest_gap,
                    "persistence_frames": None,
                    "temporal_track_id": None,
                }
            )
    return records


def aggregate_confusion(metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sum frame confusion counts and derive a stable aggregate metric row."""

    rows = list(metrics)
    if not rows:
        raise ValueError("cannot aggregate an empty metric collection")
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tn = sum(int(row["tn"]) for row in rows)
    predicted = sum(int(row["predicted_pixels"]) for row in rows)
    truth = sum(int(row["truth_pixels"]) for row in rows)
    total = tp + fp + fn + tn
    empty = tp + fp + fn == 0
    precision = _ratio(tp, tp + fp, empty_value=1.0 if empty else None)
    recall = _ratio(tp, tp + fn, empty_value=1.0 if empty else None)
    iou = _ratio(tp, tp + fp + fn, empty_value=1.0 if empty else None)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": predicted,
        "truth_pixels": truth,
        "pixel_count": total,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": _f1(precision, recall),
        "false_positive_area_fraction": float(fp / total),
        "mean_frame_precision": float(np.mean([row["precision"] for row in rows if row["precision"] is not None])) if any(row["precision"] is not None for row in rows) else None,
        "mean_frame_recall": float(np.mean([row["recall"] for row in rows if row["recall"] is not None])) if any(row["recall"] is not None for row in rows) else None,
    }
