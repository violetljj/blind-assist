"""Evaluator-only depth-edge retention; never an input to ground estimation.

Native reference edges require >=0.05 m absolute and >=log(1.04) relative
jump. Predicted edges use only the relative threshold. Same-axis, same-sign
existence matching uses a fixed Euclidean 2 px radius, without one-to-one claims.
Predicted endpoints need only be finite and positive, so positive global scale
does not change the log-edge definition. This is not object/shape truth.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from near_field import Camera


LOG_THRESHOLD = math.log(1.04)
OFFSETS = sorted((dy * dy + dx * dx, dy, dx)
                 for dy in range(-2, 3) for dx in range(-2, 3) if dy * dy + dx * dx <= 4)
KERNEL = np.asarray([[int(dy * dy + dx * dx <= 4) for dx in range(-2, 3)]
                     for dy in range(-2, 3)], dtype=np.uint8)
HEIGHT_EDGES = np.asarray((.065, .65, 1.4, 1.85))


def _stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {"count": int(values.size), "mean": float(values.mean()) if values.size else None,
            "p50": float(np.quantile(values, .5)) if values.size else None,
            "p95": float(np.quantile(values, .95)) if values.size else None}


def _nearest(query, query_sign, candidates, candidate_sign, contrast):
    """Nearest compatible candidate; sorted offsets define deterministic ties."""
    distance = np.full(query.shape, np.inf)
    matched_contrast = np.full(query.shape, np.nan)
    padded = [np.pad(array, 2, constant_values=0) for array in (candidates, candidate_sign, contrast)]
    h, w = query.shape
    for squared, dy, dx in OFFSETS:
        mask, sign, strength = (array[2 + dy:2 + dy + h, 2 + dx:2 + dx + w] for array in padded)
        take = query & ~np.isfinite(distance) & mask & (query_sign == sign)
        distance[take], matched_contrast[take] = math.sqrt(squared), strength[take]
    return distance, matched_contrast


def _pair_values(depth, first, second, *, metric_reference=False):
    a, b = depth[first], depth[second]
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if metric_reference:
        valid &= (a > .08) & (a < 12) & (b > .08) & (b < 12)
    log_jump = np.zeros(a.shape)
    absolute_jump = np.zeros(a.shape)
    log_jump[valid] = np.log(b[valid]) - np.log(a[valid])
    absolute_jump[valid] = np.abs(b[valid] - a[valid])
    return valid, absolute_jump, np.abs(log_jump), np.sign(log_jump)


def diagnose_boundary(native: np.ndarray, predicted: np.ndarray, camera: Camera) -> dict:
    """Evaluate every native near-field edge, without changing either array."""
    native, predicted = np.asarray(native, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    shape = (camera.height, camera.width)
    if native.shape != shape or predicted.shape != shape:
        raise ValueError("Depth dimensions must match Camera")
    v, u = np.mgrid[:camera.height, :camera.width]
    fx = camera.width / (2 * math.tan(math.radians(camera.hfov_deg / 2)))
    ray_up = -(v - (camera.height - 1) / 2) / fx
    pitch = math.radians(camera.pitch_deg)
    ray_x = math.cos(pitch) - math.sin(pitch) * ray_up
    ray_z = math.sin(pitch) + math.cos(pitch) * ray_up
    forward, height = native * ray_x, camera.camera_height_m + native * ray_z
    band = np.searchsorted(HEIGHT_EDGES, height, side="right") - 1
    eligible = (np.isfinite(native) & (native > .08) & (native < 12) & (ray_x > 0)
                & (forward <= 3) & (band >= 0) & (band < 3))
    totals = dict(native_edges=0, matched_native_edges=0, predicted_edges_in_neighborhood=0,
                  extraneous_predicted_edges=0, evaluable_edge_locations=0)
    by_height = {name: dict(native_edges=0, matched_native_edges=0) for name in ("low", "body", "head")}
    rows, localization, ratios_matched, ratios_at_native = {}, [], [], []
    for axis, name in ((1, "horizontal_pairs"), (0, "vertical_pairs")):
        first, second = [slice(None), slice(None)], [slice(None), slice(None)]
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        valid_n, abs_n, log_n, sign_n = _pair_values(native, first, second, metric_reference=True)
        valid_p, _, log_p, sign_p = _pair_values(predicted, first, second)
        adjacent = eligible[first] | eligible[second]
        native_edges = valid_n & adjacent & (abs_n >= .05) & (log_n >= LOG_THRESHOLD)
        neighborhood = cv2.dilate((valid_n & adjacent).astype(np.uint8), KERNEL,
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) & valid_n
        predicted_edges = neighborhood & valid_p & (log_p >= LOG_THRESHOLD)
        distances, matched_log = _nearest(native_edges, sign_n, predicted_edges, sign_p, log_p)
        reverse, _ = _nearest(predicted_edges, sign_p, native_edges, sign_n, log_n)
        matched = native_edges & np.isfinite(distances)
        counts = {"native_edges": int(native_edges.sum()), "matched_native_edges": int(matched.sum()),
                  "predicted_edges_in_neighborhood": int(predicted_edges.sum()),
                  "extraneous_predicted_edges": int((predicted_edges & ~np.isfinite(reverse)).sum()),
                  "evaluable_edge_locations": int(neighborhood.sum())}
        for key in totals:
            totals[key] += counts[key]
        # Assign each reference edge to its nearer eligible endpoint; tie goes first.
        choose_first = eligible[first] & (~eligible[second] | (forward[first] <= forward[second]))
        edge_band = np.where(choose_first, band[first], band[second])
        heights = {}
        for index, label in enumerate(by_height):
            heights[label] = {"native_edges": int((native_edges & (edge_band == index)).sum()),
                              "matched_native_edges": int((matched & (edge_band == index)).sum())}
            for key in heights[label]:
                by_height[label][key] += heights[label][key]
        if sum(row["native_edges"] for row in heights.values()) != counts["native_edges"]:
            raise AssertionError("Height assignments must partition native reference edges")
        loc = distances[matched]
        match_ratio = matched_log[matched] / log_n[matched]
        at_native = native_edges & valid_p
        raw_ratio = log_p[at_native] / log_n[at_native]
        localization.extend(loc.tolist())
        ratios_matched.extend(match_ratio.tolist())
        ratios_at_native.extend(raw_ratio.tolist())
        rows[name] = {**counts, "existence_recall": counts["matched_native_edges"] / counts["native_edges"] if counts["native_edges"] else None,
                      "unmatched_native_edges": counts["native_edges"] - counts["matched_native_edges"],
                      "matched_localization_px": _stats(loc), "by_native_height": heights,
                      "matched_log_contrast_ratio": _stats(match_ratio),
                      "same_location_log_contrast_ratio": _stats(raw_ratio),
                      "native_edge_locations_with_invalid_prediction_pair": int((native_edges & ~valid_p).sum())}
    return {"schema": "nearfield-native-boundary-retention-v1", **totals,
            "existence_recall": totals["matched_native_edges"] / totals["native_edges"] if totals["native_edges"] else None,
            "unmatched_native_edges": totals["native_edges"] - totals["matched_native_edges"],
            "matched_localization_px": _stats(localization),
            "matched_log_contrast_ratio": _stats(ratios_matched),
            "same_location_log_contrast_ratio": _stats(ratios_at_native),
            "by_native_height": by_height, "by_pair_orientation": rows,
            "thin_instance_or_shape_recall": "NOT_EVALUABLE",
            "definitions": {"native_absolute_jump_min_m": .05, "log_jump_min": LOG_THRESHOLD,
                "predicted_absolute_jump_filter": None, "native_valid_endpoint_depth_m_exclusive": [.08, 12],
                "predicted_valid_endpoint_depth": "finite and strictly positive; no metric distance clipping",
                "native_attention_forward_m": 3, "native_height_edges_m": HEIGHT_EDGES.tolist(),
                "matching_radius_px": 2, "matching_metric": "Euclidean, same pair axis and signed jump",
                "neighborhood": "2 px dilation of native-valid pair locations adjacent to native near-field eligible pixels; retain native-valid endpoints",
                "unmatched_localization": "No compatible predicted edge within 2 px; distance censored, never omitted from recall",
                "height_assignment": "Nearer native eligible endpoint, ties assigned to first endpoint"},
            "limitations": ["Synthetic native reference, not independent physical object ground truth.",
                "Existence matches may be many-to-one; no instance count or one-to-one contour correspondence.",
                "Positive global scaling preserves the log-edge definition; represented values must remain finite and positive.",
                "Contrast ratios describe absolute log-jump strength; matched ratios preserve sign but same-location ratios do not.",
                "No independent thin/pole/shape labels; never infer those recalls from occupancy components."]}
