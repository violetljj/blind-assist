"""Truth-blind producer for the target-local background-warp residual R0.

The input JSONL contains only frame/detection identity and image paths. Truth,
event labels, poses and predecessor outputs are rejected before any image is
opened. The implementation is intentionally offline and Development-only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable

import cv2
import numpy as np

from .common import (
    ABSTENTION_REASONS,
    CONTRACT_RELATIVE_PATH,
    FB_ERROR_MAX_PX,
    FORBIDDEN_INPUT_KEYS,
    IMPLEMENTATION_ID,
    MAX_CONDITION_NUMBER,
    MAX_DT_NS,
    MAX_QUADRANT_FRACTION,
    MIN_INLIER_RATIO,
    MIN_SPATIAL_QUADRANTS,
    MIN_SURVIVING_POINTS,
    PROTOCOL_ID,
    RANSAC_CONFIDENCE,
    RANSAC_MAX_ITERS,
    RANSAC_REPROJECTION_MAX_PX,
    RANSAC_SEED,
    RING_CONFIGS,
    SHI_TOMASI_MODEL_ID,
    SIMILARITY_MODEL_ID,
    canonical_json,
    contract_sha256,
    detection_identity_payload,
    input_identity_payload,
    manifest_hash,
    parameter_set_id,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)


INPUT_SCHEMA = "blindassist.target_local_warp_residual_input.v1"
OUTPUT_SCHEMA = "blindassist.target_local_warp_residual_pair.v1"
RECEIPT_SCHEMA = "blindassist.target_local_warp_residual_producer_receipt.v1"


@dataclass(frozen=True)
class Similarity:
    matrix: np.ndarray
    translation: np.ndarray
    scale: float


def _finite_bbox(value: Any) -> bool:
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return len(values) == 4 and all(math.isfinite(item) for item in values) and values[2] > values[0] and values[3] > values[1]


def _bbox_inside(value: list[float], shape: tuple[int, int], *, strict: bool) -> bool:
    height, width = shape
    x0, y0, x1, y1 = value
    if strict:
        return x0 > 0 and y0 > 0 and x1 < width and y1 < height
    return x0 >= 0 and y0 >= 0 and x1 <= width and y1 <= height


def _bbox_array(value: Any) -> list[float]:
    return [float(item) for item in value]


def _dynamic_boxes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("dynamic boxes must be a list")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) - {"bbox", "dynamic"}:
            raise ValueError("dynamic box contains unsupported fields")
        if item.get("dynamic") is not True or not _finite_bbox(item.get("bbox")):
            raise ValueError("dynamic box is invalid")
        result.append({"bbox": _bbox_array(item["bbox"]), "dynamic": True})
    return result


def _read_luma(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        luma = image
    elif image.ndim == 3 and image.shape[2] == 3:
        luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"unsupported image shape for {path}: {image.shape}")
    if luma.dtype != np.uint8:
        raise ValueError(f"luma must be uint8: {path}")
    return luma


def _mask_box(shape: tuple[int, int], bbox: list[float], dilation_px: float) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = bbox
    left = max(0, int(math.floor(x0 - dilation_px)))
    top = max(0, int(math.floor(y0 - dilation_px)))
    right = min(width, int(math.ceil(x1 + dilation_px)))
    bottom = min(height, int(math.ceil(y1 + dilation_px)))
    mask = np.zeros(shape, dtype=np.uint8)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = 255
    return mask


def ring_mask(
    shape: tuple[int, int],
    target_bbox: list[float],
    dynamic_boxes: list[dict[str, Any]],
    ring_config_id: str,
) -> tuple[np.ndarray, int]:
    height, width = shape
    d = max(target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1])
    config = RING_CONFIGS[ring_config_id]
    inner = math.ceil(config["r_inner_over_d"] * d)
    outer = math.ceil(config["r_outer_over_d"] * d)
    mask = _mask_box(shape, target_bbox, outer)
    mask[ _mask_box(shape, target_bbox, inner) > 0 ] = 0
    for item in dynamic_boxes:
        mask[_mask_box(shape, item["bbox"], inner) > 0] = 0
    return mask, int(np.count_nonzero(mask))


def _track_points(previous: np.ndarray, current: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    criteria = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03
    current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=criteria,
    )
    if current_points is None or forward_status is None:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        current_points,
        None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=criteria,
    )
    if backward is None or backward_status is None:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
    first = points.reshape(-1, 2).astype(np.float64)
    second = current_points.reshape(-1, 2).astype(np.float64)
    reverse = backward.reshape(-1, 2).astype(np.float64)
    fb_error = np.linalg.norm(reverse - first, axis=1)
    height, width = current.shape
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(first).all(axis=1)
        & np.isfinite(second).all(axis=1)
        & np.isfinite(fb_error)
        & (fb_error <= FB_ERROR_MAX_PX)
        & (second[:, 0] >= 0)
        & (second[:, 0] <= width)
        & (second[:, 1] >= 0)
        & (second[:, 1] <= height)
    )
    return first[valid], second[valid], fb_error[valid]


def _fit_two_points(source: np.ndarray, target: np.ndarray) -> Similarity | None:
    source_delta = source[1] - source[0]
    target_delta = target[1] - target[0]
    source_norm = float(np.linalg.norm(source_delta))
    target_norm = float(np.linalg.norm(target_delta))
    if source_norm <= 1e-12 or target_norm <= 1e-12:
        return None
    cosine = float(np.dot(source_delta, target_delta) / (source_norm * target_norm))
    sine = float((source_delta[0] * target_delta[1] - source_delta[1] * target_delta[0]) / (source_norm * target_norm))
    matrix = (target_norm / source_norm) * np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = target[0] - matrix @ source[0]
    scale = target_norm / source_norm
    if not np.isfinite(matrix).all() or not np.isfinite(translation).all() or not math.isfinite(scale) or scale <= 0:
        return None
    return Similarity(matrix, translation, scale)


def _apply(transform: Similarity, points: np.ndarray) -> np.ndarray:
    return points @ transform.matrix.T + transform.translation


def _refit_similarity(source: np.ndarray, target: np.ndarray) -> Similarity | None:
    if len(source) < 2:
        return None
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    denominator = float(np.sum(source_centered * source_centered))
    if denominator <= 0 or not np.isfinite(denominator):
        return None
    scale = float(np.sum(singular_values) / denominator)
    translation = target_mean - scale * rotation @ source_mean
    matrix = scale * rotation
    if not np.isfinite(matrix).all() or not np.isfinite(translation).all() or not math.isfinite(scale) or scale <= 0:
        return None
    return Similarity(matrix, translation, scale)


def ransac_similarity(source: np.ndarray, target: np.ndarray) -> tuple[Similarity | None, np.ndarray, float | None]:
    if len(source) < 2 or len(source) != len(target):
        return None, np.zeros((len(source),), dtype=bool), None
    candidates: list[tuple[int, float, tuple[int, int], Similarity, np.ndarray]] = []
    for index, pair in enumerate(itertools.combinations(range(len(source)), 2)):
        if index >= RANSAC_MAX_ITERS:
            break
        transform = _fit_two_points(source[list(pair)], target[list(pair)])
        if transform is None:
            continue
        residual = np.linalg.norm(_apply(transform, source) - target, axis=1)
        inliers = np.isfinite(residual) & (residual <= RANSAC_REPROJECTION_MAX_PX)
        count = int(inliers.sum())
        median = float(np.median(residual[inliers])) if count else math.inf
        candidates.append((count, median, pair, transform, inliers))
    if not candidates:
        return None, np.zeros((len(source),), dtype=bool), None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, _, _, _, inliers = candidates[0]
    transform = _refit_similarity(source[inliers], target[inliers])
    if transform is None:
        return None, inliers, None
    residual = np.linalg.norm(_apply(transform, source) - target, axis=1)
    final_inliers = np.isfinite(residual) & (residual <= RANSAC_REPROJECTION_MAX_PX)
    median = float(np.median(residual[final_inliers])) if final_inliers.any() else None
    return transform, final_inliers, median


def _condition_number(points: np.ndarray) -> float | None:
    if len(points) < 2:
        return None
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)
    values = np.linalg.eigvalsh(covariance)
    if not np.isfinite(values).all() or values[0] <= 0:
        return None
    return float(values[-1] / values[0])


def _spatial_support(points: np.ndarray, bbox: list[float]) -> dict[str, Any]:
    center = np.asarray([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
    quadrants = [int(point[0] >= center[0]) * 2 + int(point[1] >= center[1]) for point in points]
    counts = np.bincount(np.asarray(quadrants, dtype=np.int64), minlength=4) if len(quadrants) else np.zeros((4,), dtype=np.int64)
    fraction = float(counts.max() / len(points)) if len(points) else 1.0
    return {"occupied_quadrants": int(np.count_nonzero(counts)), "max_quadrant_fraction": fraction}


def _warp_bbox(transform: Similarity, bbox: list[float], shape: tuple[int, int]) -> list[float] | None:
    height, width = shape
    corners = np.asarray([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]], dtype=np.float64)
    warped = _apply(transform, corners)
    if not np.isfinite(warped).all() or np.any(warped < 0) or np.any(warped[:, 0] > width) or np.any(warped[:, 1] > height):
        return None
    result = [float(warped[:, 0].min()), float(warped[:, 1].min()), float(warped[:, 0].max()), float(warped[:, 1].max())]
    if not _finite_bbox(result) or not _bbox_inside(result, shape, strict=False):
        return None
    return result


def _base_row(row: dict[str, Any], ring_config_id: str, repo_root: Path | None) -> dict[str, Any]:
    return {
        "schema": OUTPUT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "parameter_set_id": parameter_set_id(ring_config_id, repo_root),
        "source_id": row.get("source_id"),
        "session_id": row.get("session_id"),
        "sequence_id": row.get("sequence_id"),
        "parent_event_id": row.get("parent_event_id"),
        "previous_source_frame_id": row.get("previous_source_frame_id"),
        "current_source_frame_id": row.get("current_source_frame_id"),
        "previous_frame_shape": row.get("previous_frame_shape"),
        "current_frame_shape": row.get("current_frame_shape"),
        "previous_captured_at_ns": row.get("captured_at_ns_previous"),
        "current_captured_at_ns": row.get("captured_at_ns_current"),
        "target_id": row.get("target_id"),
        "track_epoch": row.get("track_epoch"),
        "previous_bbox": row.get("previous_bbox"),
        "current_bbox": row.get("current_bbox"),
        "ring_config_id": ring_config_id,
        "model_id": SIMILARITY_MODEL_ID,
        "input_manifest_sha256": None,
        "detection_manifest_sha256": None,
        "predicted_bbox_cam": None,
        "raw_rate_per_s": None,
        "residual_rate_per_s": None,
        "feature_count": 0,
        "surviving_track_count": 0,
        "inlier_count": 0,
        "inlier_ratio": None,
        "median_forward_backward_error_px": None,
        "median_reprojection_error_px": None,
        "spatial_support": {"occupied_quadrants": 0, "max_quadrant_fraction": 1.0},
        "condition_number": None,
        "ring_area_px": 0,
        "dynamic_mask_box_count_previous": 0,
        "dynamic_mask_box_count_current": 0,
        "quality": "ABSTAIN",
        "abstention_reason": None,
    }


def _first_reason(row: dict[str, Any], previous: np.ndarray | None, current: np.ndarray | None, ring_config_id: str) -> str | None:
    try:
        previous_time = int(row["captured_at_ns_previous"])
        current_time = int(row["captured_at_ns_current"])
        if not (current_time > previous_time and current_time - previous_time <= MAX_DT_NS):
            return "INPUT_TIMESTAMP_INVALID"
    except (KeyError, TypeError, ValueError, OverflowError):
        return "INPUT_TIMESTAMP_INVALID"
    try:
        if int(row["current_frame_index"]) != int(row["previous_frame_index"]) + 1:
            return "FRAME_ADJACENCY_INVALID"
    except (KeyError, TypeError, ValueError, OverflowError):
        return "FRAME_ADJACENCY_INVALID"
    if previous is None or current is None or previous.shape != current.shape or len(previous.shape) != 2:
        return "IMAGE_SHAPE_MISMATCH"
    try:
        expected_previous_shape = [int(item) for item in row["previous_frame_shape"]]
        expected_current_shape = [int(item) for item in row["current_frame_shape"]]
    except (KeyError, TypeError, ValueError, OverflowError):
        return "IMAGE_SHAPE_MISMATCH"
    if expected_previous_shape != list(previous.shape) or expected_current_shape != list(current.shape):
        return "IMAGE_SHAPE_MISMATCH"
    if row.get("target_id") is None or row.get("track_epoch") is None or row.get("track_reset") is True:
        return "TRACK_ID_MISMATCH"
    if not _finite_bbox(row.get("previous_bbox")) or not _finite_bbox(row.get("current_bbox")):
        return "BOX_INVALID"
    previous_bbox = _bbox_array(row["previous_bbox"])
    current_bbox = _bbox_array(row["current_bbox"])
    if not _bbox_inside(previous_bbox, previous.shape, strict=True) or not _bbox_inside(current_bbox, current.shape, strict=True):
        return "BOX_BOUNDARY_TRUNCATED"
    try:
        _dynamic_boxes(row.get("previous_dynamic_bboxes", []))
        _dynamic_boxes(row.get("current_dynamic_bboxes", []))
    except ValueError:
        return "DYNAMIC_MASK_INVALID"
    return None


def process_pair(row: dict[str, Any], previous: np.ndarray | None, current: np.ndarray | None, ring_config_id: str, repo_root: Path | None = None) -> dict[str, Any]:
    result = _base_row(row, ring_config_id, repo_root)
    result["input_manifest_sha256"] = sha256_bytes(canonical_json(input_identity_payload(row)))
    result["detection_manifest_sha256"] = sha256_bytes(canonical_json(detection_identity_payload(row)))
    reason = _first_reason(row, previous, current, ring_config_id)
    if reason is not None:
        result["abstention_reason"] = reason
        return result
    assert previous is not None and current is not None
    previous_bbox = _bbox_array(row["previous_bbox"])
    current_bbox = _bbox_array(row["current_bbox"])
    previous_dynamic = _dynamic_boxes(row.get("previous_dynamic_bboxes", []))
    current_dynamic = _dynamic_boxes(row.get("current_dynamic_bboxes", []))
    result["dynamic_mask_box_count_previous"] = len(previous_dynamic)
    result["dynamic_mask_box_count_current"] = len(current_dynamic)
    dt = (int(row["captured_at_ns_current"]) - int(row["captured_at_ns_previous"])) / 1_000_000_000.0
    raw_rate = math.log((current_bbox[3] - current_bbox[1]) / (previous_bbox[3] - previous_bbox[1])) / dt
    result["raw_rate_per_s"] = float(raw_rate) if math.isfinite(raw_rate) else None
    mask, area = ring_mask(previous.shape, previous_bbox, previous_dynamic, ring_config_id)
    result["ring_area_px"] = area
    if area <= 0:
        result["abstention_reason"] = "RING_EMPTY_OR_LOW_AREA"
        return result
    features = cv2.goodFeaturesToTrack(previous, maxCorners=80, qualityLevel=0.01, minDistance=5.0, blockSize=5, mask=mask)
    result["feature_count"] = int(0 if features is None else len(features))
    if features is None or len(features) == 0:
        result["abstention_reason"] = "FEATURE_COUNT_LOW"
        return result
    first, second, fb_errors = _track_points(previous, current, features)
    if len(first):
        for item in current_dynamic:
            keep = ~(_mask_box(current.shape, item["bbox"], math.ceil(RING_CONFIGS[ring_config_id]["r_inner_over_d"] * max(previous_bbox[2] - previous_bbox[0], previous_bbox[3] - previous_bbox[1]))) > 0)[np.clip(second[:, 1].astype(int), 0, current.shape[0] - 1), np.clip(second[:, 0].astype(int), 0, current.shape[1] - 1)]
            first, second, fb_errors = first[keep], second[keep], fb_errors[keep]
    result["surviving_track_count"] = int(len(first))
    if len(fb_errors):
        result["median_forward_backward_error_px"] = float(np.median(fb_errors))
    if len(first) < MIN_SURVIVING_POINTS:
        result["abstention_reason"] = "LK_TRACK_COUNT_LOW"
        return result
    support = _spatial_support(first, previous_bbox)
    result["spatial_support"] = support
    if support["occupied_quadrants"] < MIN_SPATIAL_QUADRANTS or support["max_quadrant_fraction"] > MAX_QUADRANT_FRACTION:
        result["abstention_reason"] = "SPATIAL_SUPPORT_LOW"
        return result
    condition = _condition_number(first)
    result["condition_number"] = condition
    if condition is None or condition > MAX_CONDITION_NUMBER:
        result["abstention_reason"] = "GEOMETRY_DEGENERATE"
        return result
    transform, inliers, median_reprojection = ransac_similarity(first, second)
    result["inlier_count"] = int(inliers.sum())
    result["inlier_ratio"] = float(inliers.sum() / len(first))
    result["median_reprojection_error_px"] = median_reprojection
    if result["inlier_ratio"] < MIN_INLIER_RATIO:
        result["abstention_reason"] = "RANSAC_INLIER_RATIO_LOW"
        return result
    if median_reprojection is None or median_reprojection > RANSAC_REPROJECTION_MAX_PX:
        result["abstention_reason"] = "REPROJECTION_ERROR_HIGH"
        return result
    if transform is None or not np.isfinite(transform.matrix).all() or not np.isfinite(transform.translation).all() or transform.scale <= 0:
        result["abstention_reason"] = "TRANSFORM_INVALID"
        return result
    predicted = _warp_bbox(transform, previous_bbox, previous.shape)
    if predicted is None:
        result["abstention_reason"] = "PREDICTED_BOX_INVALID"
        return result
    predicted_height = predicted[3] - predicted[1]
    residual_rate = math.log((current_bbox[3] - current_bbox[1]) / predicted_height) / dt
    if not math.isfinite(residual_rate):
        result["abstention_reason"] = "NUMERIC_NONFINITE"
        return result
    result["predicted_bbox_cam"] = predicted
    result["residual_rate_per_s"] = float(residual_rate)
    result["quality"] = "PASS"
    return result


def _reject_forbidden_inputs(rows: Iterable[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        forbidden = sorted(set(row) & FORBIDDEN_INPUT_KEYS)
        if forbidden:
            raise ValueError(f"truth firewall rejected input row {index}: {forbidden}")


def _write_exclusive(path: Path, payload: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    temporary.replace(path)


def run(input_path: Path, output_path: Path, receipt_path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    _reject_forbidden_inputs(rows)
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("producer refuses existing output or receipt")
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        previous_path = Path(row["previous_image"])
        current_path = Path(row["current_image"])
        if not row.get("previous_image_sha256") or not row.get("current_image_sha256"):
            raise ValueError("input row must bind previous/current image SHA-256")
        if sha256_file(previous_path) != row["previous_image_sha256"] or sha256_file(current_path) != row["current_image_sha256"]:
            raise ValueError("input image SHA-256 mismatch")
        previous = _read_luma(previous_path)
        current = _read_luma(current_path)
        for ring_config_id in RING_CONFIGS:
            output_rows.append(process_pair(row, previous, current, ring_config_id, repo_root))
    output_text = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output_rows)
    _write_exclusive(output_path, output_text)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "stage": "DEVELOPMENT",
        "claim_ceiling": "DEVELOPMENT_SIGNAL_DIAGNOSTIC_ONLY",
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "contract_sha256": contract_sha256(repo_root),
        "truth_read": False,
        "forbidden_outcome_fields_rejected": True,
        "input_path_sha256": sha256_file(input_path),
        "input_manifest_sha256": manifest_hash(rows, input_identity_payload),
        "detection_manifest_sha256": manifest_hash(rows, detection_identity_payload),
        "input_row_count": len(rows),
        "output_row_count": len(output_rows),
        "output_sha256": sha256_file(output_path),
        "ring_config_ids": list(RING_CONFIGS),
        "model_ids": [SHI_TOMASI_MODEL_ID, SIMILARITY_MODEL_ID],
    }
    _write_exclusive(receipt_path, json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    receipt = run(args.input, args.output, args.receipt, args.repo_root)
    print(json.dumps({"status": receipt["status"], "output_rows": receipt["output_row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
