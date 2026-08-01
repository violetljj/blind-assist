"""Frozen, truth-blind DG-SRF F0 structural operators."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np


def validate_depth_direction_canary(
    signed_normalized_margins: list[float],
    *,
    frozen_direction: str,
    minimum_consistent: int,
    minimum_median_margin: float,
) -> dict[str, Any]:
    if not signed_normalized_margins:
        raise ValueError("direction canary has no margins")
    margins = np.asarray(signed_normalized_margins, dtype=np.float64)
    if frozen_direction != "RAW_LARGER_IS_NEARER":
        raise ValueError("unsupported frozen official direction")
    positive = int(np.sum(margins > 0))
    negative = int(np.sum(margins < 0))
    consistent = positive
    oriented = margins
    median_margin = float(np.median(oriented))
    passed = (
        consistent >= int(minimum_consistent)
        and median_margin >= float(minimum_median_margin)
    )
    return {
        "direction": frozen_direction,
        "positive_scene_count": positive,
        "negative_scene_count": negative,
        "consistent_with_frozen_direction_scene_count": consistent,
        "median_normalized_near_far_margin": median_margin,
        "passed": bool(passed),
    }


def depth_health_and_proximity(
    raw_depth: np.ndarray,
    *,
    direction: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    depth = np.asarray(raw_depth, dtype=np.float64)
    finite = np.isfinite(depth)
    finite_fraction = float(np.mean(finite))
    if not np.any(finite):
        return {
            "q": 0,
            "finite_fraction": finite_fraction,
            "failure_reasons": ["NO_FINITE_OUTPUT"],
        }, np.zeros_like(depth, dtype=np.float32)

    finite_values = depth[finite]
    lower_q = float(config["normalization"]["depth_lower_quantile"])
    upper_q = float(config["normalization"]["depth_upper_quantile"])
    q_low, q_high = np.quantile(finite_values, [lower_q, upper_q])
    median = float(np.median(finite_values))
    span = float(q_high - q_low)
    epsilon = float(config["normalization"]["epsilon"])
    relative_span = span / max(abs(median), epsilon)

    oriented = depth if direction == "RAW_LARGER_IS_NEARER" else -depth
    oriented_values = oriented[finite]
    o_low, o_high = np.quantile(oriented_values, [lower_q, upper_q])
    oriented_span = float(o_high - o_low)
    normalized = np.zeros_like(oriented, dtype=np.float64)
    if oriented_span > epsilon:
        normalized[finite] = np.clip(
            (oriented[finite] - o_low) / oriented_span,
            0.0,
            1.0,
        )
    norm_std = float(np.std(normalized[finite]))
    raw_min = float(np.min(finite_values))
    raw_max = float(np.max(finite_values))
    min_plateau = float(np.mean(finite_values == raw_min))
    max_plateau = float(np.mean(finite_values == raw_max))
    extreme_plateau = max(min_plateau, max_plateau)

    health = config["depth_health"]
    reasons: list[str] = []
    if finite_fraction < float(health["minimum_finite_fraction"]):
        reasons.append("FINITE_FRACTION")
    if relative_span < float(health["minimum_relative_robust_span"]):
        reasons.append("ROBUST_SPAN")
    if norm_std < float(health["minimum_normalized_standard_deviation"]):
        reasons.append("NORMALIZED_STD")
    if extreme_plateau > float(health["maximum_extreme_plateau_fraction"]):
        reasons.append("EXTREME_PLATEAU")
    q = 0 if reasons else 1
    if q == 0:
        normalized.fill(0.0)
    return {
        "q": q,
        "finite_fraction": finite_fraction,
        "raw_q05": float(q_low),
        "raw_q95": float(q_high),
        "relative_robust_span": relative_span,
        "normalized_standard_deviation": norm_std,
        "extreme_plateau_fraction": extreme_plateau,
        "failure_reasons": reasons,
    }, normalized.astype(np.float32)


def _normalize_positive_signal(
    signal: np.ndarray,
    *,
    config: Mapping[str, Any],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    value = np.asarray(signal, dtype=np.float64)
    use = np.isfinite(value)
    if mask is not None:
        use &= np.asarray(mask, dtype=bool)
    samples = value[use]
    result = np.zeros_like(value, dtype=np.float64)
    if samples.size == 0:
        return result.astype(np.float32)
    q_low = float(config["normalization"]["signal_lower_quantile"])
    q_high = float(config["normalization"]["signal_upper_quantile"])
    low, high = np.quantile(samples, [q_low, q_high])
    epsilon = float(config["normalization"]["epsilon"])
    if float(high - low) <= epsilon:
        return result.astype(np.float32)
    result = np.clip((value - low) / float(high - low), 0.0, 1.0)
    result[~np.isfinite(result)] = 0.0
    return result.astype(np.float32)


def build_proxy_mask(shape: tuple[int, int], config: Mapping[str, Any]) -> np.ndarray:
    height, width = shape
    proxy = config["proxy_roi_ablation"]
    y_start = int(np.floor(height * float(proxy["start_y_fraction"])))
    center = (width - 1) / 2.0
    top_half = width * float(proxy["top_half_width_fraction"])
    bottom_half = width * float(proxy["bottom_half_width_fraction"])
    mask = np.zeros(shape, dtype=bool)
    denominator = max(height - 1 - y_start, 1)
    for y in range(y_start, height):
        alpha = (y - y_start) / denominator
        half = top_half + alpha * (bottom_half - top_half)
        left = max(0, int(np.ceil(center - half)))
        right = min(width, int(np.floor(center + half)) + 1)
        mask[y, left:right] = True
    return mask


def structural_scores(
    proximity: np.ndarray,
    *,
    q: int,
    yolo_mask: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    p = np.asarray(proximity, dtype=np.float32)
    if p.shape != tuple(config["analysis_shape"]):
        raise ValueError("proximity shape mismatch")

    gradient_maps: list[np.ndarray] = []
    for sigma in config["structural_signal"]["gradient_gaussian_sigmas_pixels"]:
        sigma_value = float(sigma)
        smoothed = (
            p
            if sigma_value == 0.0
            else cv2.GaussianBlur(
                p,
                (0, 0),
                sigmaX=sigma_value,
                sigmaY=sigma_value,
                borderType=cv2.BORDER_REFLECT101,
            )
        )
        kernel = int(config["structural_signal"]["gradient_sobel_kernel"])
        gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=kernel)
        gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=kernel)
        gradient_maps.append(cv2.magnitude(gx, gy))
    gradient_raw = np.mean(np.stack(gradient_maps, axis=0), axis=0)
    gradient = _normalize_positive_signal(gradient_raw, config=config)

    height, width = p.shape
    trend_config = config["structural_signal"]["surface_trend"]
    start = int(
        np.floor(height * float(trend_config["lower_image_start_fraction"]))
    )
    lower_mask = np.zeros_like(p, dtype=bool)
    lower_mask[start:, :] = True
    row_stat = np.median(p[start:, :], axis=1)
    y = np.linspace(0.0, 1.0, height - start, dtype=np.float64)
    degree = int(trend_config["polynomial_degree"])
    if row_stat.size <= degree or not np.isfinite(row_stat).all():
        raise ValueError("surface trend fit input is invalid")
    design = np.vander(y, degree + 1)
    if int(np.linalg.matrix_rank(design)) != degree + 1:
        raise ValueError("surface trend fit is rank deficient")
    coefficients = np.polyfit(y, row_stat.astype(np.float64), degree)
    if not np.isfinite(coefficients).all():
        raise ValueError("surface trend fit failed")
    trend_rows = np.polyval(coefficients, y).astype(np.float32)
    trend = np.broadcast_to(trend_rows[:, None], (height - start, width))
    delta = float(trend_config["residual_dead_zone"])
    r_plus_raw = np.zeros_like(p, dtype=np.float32)
    r_minus_raw = np.zeros_like(p, dtype=np.float32)
    r_plus_raw[start:, :] = np.maximum(0.0, p[start:, :] - trend - delta)
    r_minus_raw[start:, :] = np.maximum(0.0, trend - p[start:, :] - delta)
    r_plus = _normalize_positive_signal(
        r_plus_raw,
        config=config,
        mask=lower_mask,
    )
    r_minus = _normalize_positive_signal(
        r_minus_raw,
        config=config,
        mask=lower_mask,
    )

    q_value = float(int(q))
    d1 = q_value * p
    d2 = q_value * gradient
    d3 = q_value * (r_plus + r_minus) / 2.0
    d4 = q_value * (p + gradient + r_plus + r_minus) / 4.0
    outside_yolo = (~np.asarray(yolo_mask, dtype=bool)).astype(np.float32)
    d1 *= outside_yolo
    d2 *= outside_yolo
    d3 *= outside_yolo
    d4 *= outside_yolo
    proxy = build_proxy_mask(p.shape, config).astype(np.float32)
    lam = float(config["proxy_roi_ablation"]["lambda"])
    d5 = d4 * (lam + (1.0 - lam) * proxy)
    return {
        "D1": d1.astype(np.float32),
        "D2": d2.astype(np.float32),
        "D3": d3.astype(np.float32),
        "D4": d4.astype(np.float32),
        "D5": d5.astype(np.float32),
        "N": p.astype(np.float32),
        "E": gradient.astype(np.float32),
        "R_plus": r_plus.astype(np.float32),
        "R_minus": r_minus.astype(np.float32),
    }
