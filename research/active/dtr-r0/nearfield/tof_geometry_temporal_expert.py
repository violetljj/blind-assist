"""Causal, label-free ToF geometry and short-gap temporal expert.

The expert consumes only ordered 8x8 range slots, validity, timestamps and the
fixed camera/body calibration.  A missing return is never interpreted as free
space.  The temporal arm only extrapolates previously observed closing returns;
it does not predict an alert beyond the current 3 m body/head query corridors.
"""
from __future__ import annotations

import numpy as np

from contact_retina_spec import BODY_BOXES

EVENT_ORDER = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
ZONES_PER_AXIS = 8
ZONE_WIDTH_DEGREES = 45.0 / ZONES_PER_AXIS
EYE_HEIGHT_M = 1.7


def zone_center_rays() -> np.ndarray:
    angles = np.arange(ZONES_PER_AXIS, dtype=np.float64) * ZONE_WIDTH_DEGREES + ZONE_WIDTH_DEGREES / 2
    elevation, azimuth = np.meshgrid(22.5 - angles, angles - 22.5, indexing='ij')
    rays = np.stack((np.ones_like(azimuth),
                     np.tan(np.deg2rad(azimuth)),
                     np.tan(np.deg2rad(elevation))), axis=-1).reshape(64, 3)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def sanitize_packet(ranges: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.asarray(ranges, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if ranges.ndim != 3 or ranges.shape[1] != 64 or ranges.shape != valid.shape:
        raise ValueError('ranges/valid must have matching [N,64,K] shapes')
    usable = valid & np.isfinite(ranges) & (ranges > 0.0) & (ranges <= 4.0)
    return np.where(usable, ranges, 0.0), usable


def project_events(ranges: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project usable returns through zone centers into four query boxes.

    Returns signed margins (+1 alert, -1 no alert) and per-query support counts.
    A negative margin means only "no supported alert", never measured clearance.
    """
    ranges, valid = sanitize_packet(ranges, valid)
    points = ranges[..., None] * zone_center_rays()[None, :, None, :]
    points[..., 2] += EYE_HEIGHT_M
    flat_points = points.reshape(len(ranges), -1, 3)
    flat_valid = valid.reshape(len(ranges), -1)
    counts = []
    for low, high in BODY_BOXES:
        for half, start in enumerate((high[0], high[0] + 1.5)):
            end_ok = flat_points[..., 0] < start + 1.5 if half == 0 else flat_points[..., 0] <= start + 1.5
            inside = ((flat_points[..., 0] >= start) & end_ok
                      & (flat_points[..., 1] >= low[1]) & (flat_points[..., 1] <= high[1])
                      & (flat_points[..., 2] >= low[2]) & (flat_points[..., 2] <= high[2]))
            counts.append((inside & flat_valid).sum(axis=1))
    support = np.stack(counts, axis=1)
    return np.where(support >= 1, 1.0, -1.0), support


def causal_gap_fill(ranges: np.ndarray, valid: np.ndarray, clip: np.ndarray,
                    time_s: np.ndarray, *, history_frames: int = 5,
                    max_gap_frames: int = 3, max_closing_speed_mps: float = 3.0
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill short missing slots from a causal robust closing-range trend.

    At least two direct observations are required.  The median consecutive
    slope is clipped to closing motion only; static/receding returns are not
    synthesized.  No future frame or evaluator field is accessed.
    """
    ranges, valid = sanitize_packet(ranges, valid)
    clip = np.asarray(clip).astype(str)
    time_s = np.asarray(time_s, dtype=np.float64)
    if clip.shape != (len(ranges),) or time_s.shape != (len(ranges),):
        raise ValueError('clip/time_s must be [N]')
    if history_frames < 2 or max_gap_frames < 1 or max_closing_speed_mps <= 0:
        raise ValueError('invalid temporal configuration')
    filled = ranges.copy()
    filled_valid = valid.copy()
    imputed = np.zeros_like(valid)
    for i in range(len(ranges)):
        same_clip = np.flatnonzero((clip[:i] == clip[i]) & (np.arange(i) >= i - history_frames))
        if len(same_clip) < 2:
            continue
        for zone in range(ranges.shape[1]):
            for slot in range(ranges.shape[2]):
                if valid[i, zone, slot]:
                    continue
                observed = same_clip[valid[same_clip, zone, slot]]
                if len(observed) < 2:
                    continue
                last = int(observed[-1])
                if i - last > max_gap_frames:
                    continue
                recent = observed[-history_frames:]
                dt = np.diff(time_s[recent])
                if np.any(dt <= 0):
                    raise ValueError('timestamps must increase within each clip')
                slopes = np.diff(ranges[recent, zone, slot]) / dt
                slope = float(np.median(slopes))
                if not (-max_closing_speed_mps <= slope < 0.0):
                    continue
                predicted = ranges[last, zone, slot] + slope * (time_s[i] - time_s[last])
                if 0.0 < predicted <= 4.0:
                    filled[i, zone, slot] = predicted
                    filled_valid[i, zone, slot] = True
                    imputed[i, zone, slot] = True
    return filled, filled_valid, imputed


def predict(ranges: np.ndarray, valid: np.ndarray, clip: np.ndarray,
            time_s: np.ndarray) -> dict[str, np.ndarray]:
    frame, frame_support = project_events(ranges, valid)
    filled, filled_valid, imputed = causal_gap_fill(ranges, valid, clip, time_s)
    temporal, temporal_support = project_events(filled, filled_valid)
    return {
        'GEOMETRY_FRAME': frame,
        'GEOMETRY_TEMPORAL': temporal,
        'frame_support': frame_support,
        'temporal_support': temporal_support,
        'filled_ranges': filled,
        'filled_valid': filled_valid,
        'imputed': imputed,
    }
