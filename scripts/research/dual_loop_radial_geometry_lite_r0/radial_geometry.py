"""Frozen two-frame causal geometry arms for LITE Development R0."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import cv2
import numpy as np


PROTOCOL_ID = "DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0"
IMPLEMENTATION_ID = "DUAL_LOOP_RADIAL_GEOMETRY_LITE_R0_IMPL_R0"
TTL_NS = 100_000_000
ARM_BBOX = "BBOX_LOG_AREA_GROWTH"
ARM_FLOW = "ROI_SPARSE_RADIAL_FLOW"
ARMS = (ARM_BBOX, ARM_FLOW)
PARAMETERS: dict[str, Any] = {
    "causal_lookback_frames": 1,
    "ttl_ns": TTL_NS,
    "global_motion_compensation": False,
    "bbox": {
        "formula": "0.5*d_log_area/dt",
    },
    "flow": {
        "max_corners": 80,
        "quality_level": 0.01,
        "min_distance_px": 5,
        "block_size_px": 5,
        "lk_window_px": [15, 15],
        "lk_max_level": 2,
        "lk_termination_count": 20,
        "lk_termination_epsilon": 0.03,
        "fb_error_max_px": 1.5,
        "current_roi_expansion_fraction": 0.10,
        "minimum_radius_px": 2.0,
        "minimum_surviving_tracks": 8,
        "minimum_previous_roi_quadrants": 2,
        "quality_track_reference": 24,
        "quality_mad_reference_per_s": 0.10,
    },
}
PARAMETER_SHA256 = hashlib.sha256(
    json.dumps(PARAMETERS, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class FrameObservation:
    source_frame_id: str
    captured_at_ns: int
    target_id: str
    track_epoch: str
    region: str
    roi_xywh_normalized: tuple[float, float, float, float]
    gray: np.ndarray
    history_reset: bool = False


def _quality(score: float, **components: Any) -> dict[str, Any]:
    return {"score": float(min(1.0, max(0.0, score))), "components": components}


def _base_output(current: FrameObservation, arm_id: str) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "parameter_sha256": PARAMETER_SHA256,
        "arm_id": arm_id,
        "capture_id": "REVEL_DYNAMIC_V1",
        "source_frame_id": current.source_frame_id,
        "captured_at_ns": int(current.captured_at_ns),
        "available_at_ns": int(current.captured_at_ns),
        "target_id": current.target_id,
        "track_epoch": current.track_epoch,
        "region": current.region,
        "signed_approach_rate_per_s": None,
        "quality": _quality(0.0),
        "ttl_ns": TTL_NS,
        "valid_until_ns": int(current.captured_at_ns + TTL_NS),
        "abstention_reason": None,
    }


def apply_consumer_time(
    row: dict[str, Any],
    consumer_timestamp_ns: int,
) -> dict[str, Any]:
    """Apply the frozen TTL rule without renewing capture-anchored validity."""
    result = dict(row)
    if int(consumer_timestamp_ns) <= int(result["valid_until_ns"]):
        return result
    result["signed_approach_rate_per_s"] = None
    result["quality"] = _quality(
        0.0,
        stale_at_consumer_timestamp_ns=int(consumer_timestamp_ns),
        original_quality=result.get("quality"),
    )
    result["abstention_reason"] = "STALE_RESULT"
    return result


def _abstain(
    current: FrameObservation,
    arm_id: str,
    reason: str,
    **quality_components: Any,
) -> dict[str, Any]:
    output = _base_output(current, arm_id)
    output["quality"] = _quality(0.0, **quality_components)
    output["abstention_reason"] = reason
    return output


def _roi_pixels(
    roi_xywh_normalized: tuple[float, float, float, float],
    shape: tuple[int, ...],
) -> tuple[float, float, float, float] | None:
    if len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    cx, cy, box_width, box_height = roi_xywh_normalized
    values = (cx, cy, box_width, box_height)
    if not all(math.isfinite(value) for value in values) or box_width <= 0.0 or box_height <= 0.0:
        return None
    return (
        float(cx * width),
        float(cy * height),
        float(box_width * width),
        float(box_height * height),
    )


def _common_history_reason(
    previous: FrameObservation | None,
    current: FrameObservation,
) -> tuple[str | None, float | None]:
    if current.history_reset or previous is None or previous.track_epoch != current.track_epoch:
        return "INSUFFICIENT_HISTORY", None
    delta_ns = int(current.captured_at_ns) - int(previous.captured_at_ns)
    if delta_ns <= 0 or delta_ns > TTL_NS:
        return "HISTORY_GAP", None
    return None, delta_ns / 1e9


def bbox_log_area_growth(
    previous: FrameObservation | None,
    current: FrameObservation,
) -> dict[str, Any]:
    reason, delta_t_s = _common_history_reason(previous, current)
    if reason is not None:
        return _abstain(current, ARM_BBOX, reason)
    assert previous is not None and delta_t_s is not None
    previous_roi = _roi_pixels(previous.roi_xywh_normalized, previous.gray.shape)
    current_roi = _roi_pixels(current.roi_xywh_normalized, current.gray.shape)
    if previous_roi is None or current_roi is None:
        return _abstain(current, ARM_BBOX, "INVALID_ROI")
    previous_area = previous_roi[2] * previous_roi[3]
    current_area = current_roi[2] * current_roi[3]
    if previous_area <= 0.0 or current_area <= 0.0:
        return _abstain(current, ARM_BBOX, "INVALID_ROI")
    estimate = 0.5 * (math.log(current_area) - math.log(previous_area)) / delta_t_s
    if not math.isfinite(estimate):
        return _abstain(current, ARM_BBOX, "NONFINITE_ESTIMATE")
    output = _base_output(current, ARM_BBOX)
    output["signed_approach_rate_per_s"] = float(estimate)
    output["quality"] = _quality(
        1.0,
        box_pair_valid=True,
        delta_t_ns=int(current.captured_at_ns - previous.captured_at_ns),
    )
    return output


def _rectangle_mask(shape: tuple[int, ...], roi: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = int(shape[0]), int(shape[1])
    cx, cy, box_width, box_height = roi
    left = max(0, int(math.floor(cx - box_width / 2.0)))
    top = max(0, int(math.floor(cy - box_height / 2.0)))
    right = min(width, int(math.ceil(cx + box_width / 2.0)))
    bottom = min(height, int(math.ceil(cy + box_height / 2.0)))
    if right <= left or bottom <= top:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


def _inside_expanded_roi(
    points: np.ndarray,
    roi: tuple[float, float, float, float],
    shape: tuple[int, ...],
    expansion_fraction: float,
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    cx, cy, box_width, box_height = roi
    half_width = box_width * 0.5 * (1.0 + expansion_fraction)
    half_height = box_height * 0.5 * (1.0 + expansion_fraction)
    return (
        (points[:, 0] >= max(0.0, cx - half_width))
        & (points[:, 0] < min(float(width), cx + half_width))
        & (points[:, 1] >= max(0.0, cy - half_height))
        & (points[:, 1] < min(float(height), cy + half_height))
    )


def roi_sparse_radial_flow(
    previous: FrameObservation | None,
    current: FrameObservation,
) -> dict[str, Any]:
    reason, delta_t_s = _common_history_reason(previous, current)
    if reason is not None:
        return _abstain(current, ARM_FLOW, reason)
    assert previous is not None and delta_t_s is not None
    if previous.gray.ndim != 2 or current.gray.ndim != 2:
        return _abstain(current, ARM_FLOW, "INVALID_ROI")
    previous_roi = _roi_pixels(previous.roi_xywh_normalized, previous.gray.shape)
    current_roi = _roi_pixels(current.roi_xywh_normalized, current.gray.shape)
    if previous_roi is None or current_roi is None:
        return _abstain(current, ARM_FLOW, "INVALID_ROI")
    mask = _rectangle_mask(previous.gray.shape, previous_roi)
    if mask is None:
        return _abstain(current, ARM_FLOW, "INVALID_ROI")

    flow = PARAMETERS["flow"]
    features = cv2.goodFeaturesToTrack(
        previous.gray,
        maxCorners=int(flow["max_corners"]),
        qualityLevel=float(flow["quality_level"]),
        minDistance=float(flow["min_distance_px"]),
        mask=mask,
        blockSize=int(flow["block_size_px"]),
        useHarrisDetector=False,
    )
    detected = 0 if features is None else int(len(features))
    if features is None or detected < int(flow["minimum_surviving_tracks"]):
        return _abstain(current, ARM_FLOW, "FEATURES_LT_8", detected_features=detected)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(flow["lk_termination_count"]),
        float(flow["lk_termination_epsilon"]),
    )
    current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous.gray,
        current.gray,
        features,
        None,
        winSize=tuple(flow["lk_window_px"]),
        maxLevel=int(flow["lk_max_level"]),
        criteria=criteria,
    )
    if current_points is None or forward_status is None:
        return _abstain(current, ARM_FLOW, "SURVIVING_TRACKS_LT_8", detected_features=detected, surviving_tracks=0)
    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current.gray,
        previous.gray,
        current_points,
        None,
        winSize=tuple(flow["lk_window_px"]),
        maxLevel=int(flow["lk_max_level"]),
        criteria=criteria,
    )
    if backward_points is None or backward_status is None:
        return _abstain(current, ARM_FLOW, "SURVIVING_TRACKS_LT_8", detected_features=detected, surviving_tracks=0)

    previous_points = features.reshape(-1, 2).astype(np.float64)
    tracked_points = current_points.reshape(-1, 2).astype(np.float64)
    backward = backward_points.reshape(-1, 2).astype(np.float64)
    fb_error = np.linalg.norm(backward - previous_points, axis=1)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(previous_points).all(axis=1)
        & np.isfinite(tracked_points).all(axis=1)
        & np.isfinite(fb_error)
        & (fb_error <= float(flow["fb_error_max_px"]))
        & _inside_expanded_roi(
            tracked_points,
            current_roi,
            current.gray.shape,
            float(flow["current_roi_expansion_fraction"]),
        )
    )
    previous_center = np.asarray(previous_roi[:2], dtype=np.float64)
    current_center = np.asarray(current_roi[:2], dtype=np.float64)
    previous_radius = np.linalg.norm(previous_points - previous_center, axis=1)
    current_radius = np.linalg.norm(tracked_points - current_center, axis=1)
    valid &= (
        (previous_radius >= float(flow["minimum_radius_px"]))
        & (current_radius >= float(flow["minimum_radius_px"]))
    )
    surviving = int(valid.sum())
    if surviving < int(flow["minimum_surviving_tracks"]):
        return _abstain(
            current,
            ARM_FLOW,
            "SURVIVING_TRACKS_LT_8",
            detected_features=detected,
            surviving_tracks=surviving,
        )

    valid_previous = previous_points[valid]
    quadrants = {
        (bool(point[0] >= previous_center[0]), bool(point[1] >= previous_center[1]))
        for point in valid_previous
    }
    occupied_quadrants = len(quadrants)
    if occupied_quadrants < int(flow["minimum_previous_roi_quadrants"]):
        return _abstain(
            current,
            ARM_FLOW,
            "SPATIAL_SUPPORT_LT_2_QUADRANTS",
            detected_features=detected,
            surviving_tracks=surviving,
            occupied_quadrants=occupied_quadrants,
        )

    track_rates = np.log(current_radius[valid] / previous_radius[valid]) / delta_t_s
    estimate = float(np.median(track_rates))
    score_mad = float(np.median(np.abs(track_rates - estimate)))
    median_fb = float(np.median(fb_error[valid]))
    if not all(math.isfinite(value) for value in (estimate, score_mad, median_fb)):
        return _abstain(current, ARM_FLOW, "NONFINITE_ESTIMATE")
    quality_score = (
        min(1.0, surviving / float(flow["quality_track_reference"]))
        * min(1.0, occupied_quadrants / 4.0)
        * max(0.0, 1.0 - median_fb / float(flow["fb_error_max_px"]))
        * (1.0 / (1.0 + score_mad / float(flow["quality_mad_reference_per_s"])))
    )
    output = _base_output(current, ARM_FLOW)
    output["signed_approach_rate_per_s"] = estimate
    output["quality"] = _quality(
        quality_score,
        detected_features=detected,
        surviving_tracks=surviving,
        occupied_quadrants=occupied_quadrants,
        median_fb_error_px=median_fb,
        score_mad_per_s=score_mad,
    )
    return output


def evaluate_pair(
    previous: FrameObservation | None,
    current: FrameObservation,
) -> list[dict[str, Any]]:
    """Return the two frozen arm rows using no state beyond one previous frame."""
    return [
        bbox_log_area_growth(previous, current),
        roi_sparse_radial_flow(previous, current),
    ]
