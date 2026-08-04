#!/usr/bin/env python3
"""Development-only local metric traversability field and alert projection.

This module deliberately keeps the rich geometric representation separate from
the lossy user-facing alert.  A VALID field means that the frame passed the
frozen observation/support checks below; it is not a safety or navigation
authorization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from evaluate_metric3d_clearance_field_a0 import (
    depth_to_points,
    fit_ground_plane,
)


SCHEMA = "blindassist_hftf_metric_traversability_field_r0"
AUTHORITY = "DEVELOPMENT_ONLY_SHADOW_DEMO"
DEFAULT_DIRECTION_DEGREES = tuple(range(-40, 41, 5))
DEFAULT_HORIZONS_M = (1.0, 1.5, 2.0)


@dataclass(frozen=True)
class TraversabilityPolicy:
    direction_degrees: tuple[int, ...] = DEFAULT_DIRECTION_DEGREES
    horizons_m: tuple[float, ...] = DEFAULT_HORIZONS_M
    body_half_width_m: float = 0.32
    lateral_margin_m: float = 0.10
    minimum_depth_finite_fraction: float = 0.15
    minimum_direction_support_points: int = 20
    minimum_ground_support_points: int = 80
    obstacle_height_min_m: float = 0.08
    obstacle_height_max_m: float = 2.00
    minimum_forward_m: float = 0.20
    maximum_forward_m: float = 4.00
    clearance_quantile: float = 0.02

    def validate(self) -> None:
        if not self.direction_degrees or tuple(sorted(set(self.direction_degrees))) != self.direction_degrees:
            raise ValueError("direction_degrees must be unique and increasing")
        if any(abs(value) >= 90 for value in self.direction_degrees):
            raise ValueError("direction_degrees must stay inside the forward hemisphere")
        if not self.horizons_m or any(value <= 0 for value in self.horizons_m):
            raise ValueError("horizons_m must be positive")
        if tuple(sorted(set(self.horizons_m))) != self.horizons_m:
            raise ValueError("horizons_m must be unique and increasing")
        if self.body_half_width_m <= 0 or self.lateral_margin_m < 0:
            raise ValueError("body dimensions must be non-negative")
        if not 0 < self.minimum_depth_finite_fraction <= 1:
            raise ValueError("minimum_depth_finite_fraction must be in (0, 1]")
        if self.minimum_direction_support_points <= 0:
            raise ValueError("minimum_direction_support_points must be positive")
        if not 0 < self.clearance_quantile < 0.5:
            raise ValueError("clearance_quantile must be in (0, 0.5)")


def _unknown(
    reasons: list[str],
    *,
    depth_summary: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "status": "UNKNOWN",
        "unknown_reasons": list(dict.fromkeys(reasons)),
        "calibrated_depth": depth_summary,
        "ground_plane": None,
        "clearance_profile": [],
        "sweep_envelopes": [],
        "intrusion_regions": [],
        "best_observed_clearance_direction": None,
        "temporal_trend": {"status": "UNKNOWN_NO_COMPARABLE_PREVIOUS_FIELD"},
        "quality": quality,
        "claim_ceiling": (
            "development-only observed geometry; UNKNOWN is not clear; "
            "no safety, navigation, or production authority"
        ),
    }


def _state_for_horizon(
    clearance_m: float | None,
    observed_forward_m: float,
    horizon_m: float,
    support_points: int,
    policy: TraversabilityPolicy,
) -> str:
    if clearance_m is not None and clearance_m <= horizon_m:
        return "OCCUPIED_OBSERVED"
    if (
        support_points >= policy.minimum_direction_support_points
        and observed_forward_m >= horizon_m
    ):
        return "CLEAR_OBSERVED"
    return "UNKNOWN_SUPPORT"


def _intrusion_regions(profile: list[dict[str, Any]]) -> list[dict[str, Any]]:
    occupied = [
        index
        for index, item in enumerate(profile)
        if item["nearest_intrusion_m"] is not None
    ]
    if not occupied:
        return []
    groups: list[list[int]] = [[occupied[0]]]
    for index in occupied[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    regions = []
    for region_id, indices in enumerate(groups, 1):
        cells = [profile[index] for index in indices]
        nearest = min(float(cell["nearest_intrusion_m"]) for cell in cells)
        regions.append(
            {
                "region_id": region_id,
                "theta_min_deg": cells[0]["theta_deg"],
                "theta_max_deg": cells[-1]["theta_deg"],
                "nearest_intrusion_m": nearest,
                "status": "OBSERVED_CLASS_FREE_INTRUSION",
                "claim_ceiling": "region geometry only; object class and safe bypass unknown",
            }
        )
    return regions


def _temporal_trend(
    profile: list[dict[str, Any]],
    previous_field: dict[str, Any] | None,
    timestamp_ns: int | None,
    previous_timestamp_ns: int | None,
) -> dict[str, Any]:
    if not previous_field or previous_field.get("status") != "VALID":
        return {"status": "UNKNOWN_NO_COMPARABLE_PREVIOUS_FIELD"}
    previous = {
        int(item["theta_deg"]): item.get("nearest_intrusion_m")
        for item in previous_field.get("clearance_profile", [])
    }
    deltas = []
    cells = []
    for item in profile:
        theta = int(item["theta_deg"])
        current = item.get("nearest_intrusion_m")
        old = previous.get(theta)
        if current is None or old is None:
            continue
        delta = float(current) - float(old)
        deltas.append(delta)
        cells.append({"theta_deg": theta, "delta_m": delta})
    if not deltas:
        return {"status": "UNKNOWN_NO_PAIRED_DIRECTION_CLEARANCE"}
    median_delta = float(np.median(np.asarray(deltas, dtype=np.float64)))
    elapsed_ns = (
        int(timestamp_ns) - int(previous_timestamp_ns)
        if timestamp_ns is not None and previous_timestamp_ns is not None
        else None
    )
    if median_delta < -0.05:
        label = "OBSERVED_CLEARANCE_SHRINKING"
    elif median_delta > 0.05:
        label = "OBSERVED_CLEARANCE_EXPANDING"
    else:
        label = "OBSERVED_CLEARANCE_STABLE"
    return {
        "status": "VALID_OBSERVED_DELTA",
        "label": label,
        "median_delta_m": median_delta,
        "elapsed_ns": elapsed_ns,
        "paired_direction_count": len(deltas),
        "directions": cells,
        "claim_ceiling": "frame-to-frame observed trend; not motion or collision prediction",
    }


def build_metric_traversability_field(
    calibrated_depth_m: np.ndarray,
    intrinsics: np.ndarray,
    *,
    metric_scale: dict[str, Any],
    source_model: str,
    timestamp_ns: int | None = None,
    previous_field: dict[str, Any] | None = None,
    previous_timestamp_ns: int | None = None,
    image_quality: dict[str, Any] | None = None,
    policy: TraversabilityPolicy = TraversabilityPolicy(),
) -> dict[str, Any]:
    """Build a fail-closed, continuous-direction observed traversability field."""

    policy.validate()
    depth = np.asarray(calibrated_depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("calibrated_depth_m must be a 2-D array")
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    finite = np.isfinite(depth) & (depth > 0)
    finite_fraction = float(np.mean(finite)) if depth.size else 0.0
    finite_values = depth[finite]
    metric_valid = metric_scale.get("status") == "VALID"
    depth_summary = {
        "available": metric_valid,
        "unit": "m" if metric_valid else None,
        "source_model": source_model,
        "metric_scale_status": metric_scale.get("status"),
        "scale": metric_scale.get("scale"),
        "anchor_age_ns": metric_scale.get("anchor_age_ns"),
        "anchor_source": metric_scale.get("anchor_source"),
        "finite_fraction": finite_fraction,
        "p05_m": float(np.quantile(finite_values, 0.05)) if metric_valid and len(finite_values) else None,
        "p50_m": float(np.quantile(finite_values, 0.50)) if metric_valid and len(finite_values) else None,
        "p95_m": float(np.quantile(finite_values, 0.95)) if metric_valid and len(finite_values) else None,
        "retention": "summary only; caller may retain the calibrated depth artifact separately",
    }
    quality = {
        "image_quality": image_quality or {"status": "NOT_PROVIDED", "pass": None},
        "depth_finite_fraction": finite_fraction,
        "depth_support_pass": finite_fraction >= policy.minimum_depth_finite_fraction,
        "ground_support_pass": False,
        "direction_support_fraction": 0.0,
        "overall_confidence": 0.0,
    }
    reasons = []
    if image_quality is not None and not bool(image_quality.get("pass")):
        reasons.append("UNKNOWN_IMAGE_QUALITY")
    if metric_scale.get("status") != "VALID":
        reasons.append(str(metric_scale.get("status", "UNKNOWN_METRIC_SCALE")))
    if finite_fraction < policy.minimum_depth_finite_fraction:
        reasons.append("UNKNOWN_INSUFFICIENT_DEPTH_SUPPORT")
    if reasons:
        return _unknown(reasons, depth_summary=depth_summary, quality=quality)

    points, pixels = depth_to_points(depth, matrix)
    plane = fit_ground_plane(points, pixels, depth.shape[0])
    if plane is None:
        return _unknown(
            ["UNKNOWN_GROUND_PLANE"], depth_summary=depth_summary, quality=quality
        )
    up, camera_height, plane_residual = plane
    optical_forward = np.asarray([0.0, 0.0, 1.0])
    forward_axis = optical_forward - float(np.dot(optical_forward, up)) * up
    forward_norm = float(np.linalg.norm(forward_axis))
    if forward_norm <= 1e-6:
        return _unknown(
            ["UNKNOWN_GROUND_FORWARD"], depth_summary=depth_summary, quality=quality
        )
    forward_axis /= forward_norm
    lateral_axis = np.cross(forward_axis, up)
    lateral_axis /= np.linalg.norm(lateral_axis)

    heights = points @ up + camera_height
    base_forward = points @ forward_axis
    base_lateral = points @ lateral_axis
    valid_forward = (
        (base_forward >= policy.minimum_forward_m)
        & (base_forward <= policy.maximum_forward_m)
    )
    obstacle_height = (
        (heights >= policy.obstacle_height_min_m)
        & (heights <= policy.obstacle_height_max_m)
    )
    half_width = policy.body_half_width_m + policy.lateral_margin_m
    profile = []
    for theta_deg in policy.direction_degrees:
        angle = math.radians(theta_deg)
        heading_forward = base_forward * math.cos(angle) + base_lateral * math.sin(angle)
        heading_lateral = base_lateral * math.cos(angle) - base_forward * math.sin(angle)
        corridor = (
            valid_forward
            & (heading_forward >= policy.minimum_forward_m)
            & (np.abs(heading_lateral) <= half_width)
        )
        support_distances = heading_forward[corridor]
        support_points = int(len(support_distances))
        observed_forward = (
            float(np.quantile(support_distances, 0.90)) if support_points else 0.0
        )
        intrusions = heading_forward[corridor & obstacle_height]
        clearance = (
            float(np.quantile(intrusions, policy.clearance_quantile))
            if len(intrusions) >= policy.minimum_direction_support_points
            else None
        )
        profile.append(
            {
                "theta_deg": theta_deg,
                "nearest_intrusion_m": clearance,
                "risk_score": (
                    max(0.0, min(1.0, 1.0 - clearance / policy.maximum_forward_m))
                    if clearance is not None
                    else None
                ),
                "known_score": min(
                    1.0,
                    support_points / (2.0 * policy.minimum_direction_support_points),
                ),
                "intrusion_points": int(len(intrusions)),
                "support_points": support_points,
                "observed_forward_m": observed_forward,
                "support_status": (
                    "SUPPORTED"
                    if support_points >= policy.minimum_direction_support_points
                    else "UNKNOWN_SUPPORT"
                ),
                "provenance": {
                    "source_model": source_model,
                    "geometry": "calibrated_depth_plus_depth_ransac",
                    "metric_scale_source": metric_scale.get("anchor_source"),
                },
            }
        )

    sweep_envelopes = []
    for horizon in policy.horizons_m:
        cells = []
        for item in profile:
            cells.append(
                {
                    "theta_deg": item["theta_deg"],
                    "state": _state_for_horizon(
                        item["nearest_intrusion_m"],
                        item["observed_forward_m"],
                        horizon,
                        item["support_points"],
                        policy,
                    ),
                }
            )
        sweep_envelopes.append(
            {
                "horizon_m": horizon,
                "body_half_width_m": policy.body_half_width_m,
                "lateral_margin_m": policy.lateral_margin_m,
                "directions": cells,
            }
        )

    supported = [item for item in profile if item["support_status"] == "SUPPORTED"]
    direction_support_fraction = len(supported) / len(profile)
    observable = [
        item
        for item in supported
        if item["nearest_intrusion_m"] is not None
    ]
    best = (
        max(observable, key=lambda item: float(item["nearest_intrusion_m"]))
        if observable
        else None
    )
    ground_score = max(0.0, min(1.0, 1.0 - plane_residual / 0.08))
    overall_confidence = float(
        min(finite_fraction / 0.90, 1.0) * ground_score * direction_support_fraction
    )
    quality.update(
        {
            "ground_support_pass": True,
            "direction_support_fraction": direction_support_fraction,
            "overall_confidence": overall_confidence,
        }
    )
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "status": "VALID",
        "unknown_reasons": [],
        "calibrated_depth": depth_summary,
        "ground_plane": {
            "source": "depth_ransac",
            "normal_camera": [float(value) for value in up],
            "camera_height_m": float(camera_height),
            "median_residual_m": float(plane_residual),
        },
        "clearance_profile": profile,
        "sweep_envelopes": sweep_envelopes,
        "intrusion_regions": _intrusion_regions(profile),
        "best_observed_clearance_direction": (
            {
                "theta_deg": best["theta_deg"],
                "nearest_intrusion_m": best["nearest_intrusion_m"],
                "status": "DEMO_CANDIDATE_NOT_SAFE_DIRECTION",
            }
            if best is not None
            else None
        ),
        "temporal_trend": _temporal_trend(
            profile,
            previous_field,
            timestamp_ns,
            previous_timestamp_ns,
        ),
        "quality": quality,
        "claim_ceiling": (
            "development-only observed geometry; CLEAR_OBSERVED is bounded image support, "
            "not traversability truth; no safety, navigation, or production authority"
        ),
    }


class AlertMapper:
    """Lossy shadow/demo projection from the rich field to terse alerts."""

    def __init__(self, alert_horizon_m: float = 1.5) -> None:
        if not math.isfinite(alert_horizon_m) or alert_horizon_m <= 0:
            raise ValueError("alert_horizon_m must be finite and positive")
        self.alert_horizon_m = float(alert_horizon_m)

    def map(self, field: dict[str, Any]) -> dict[str, Any]:
        base = {
            "authority": "SHADOW_DEMO_ONLY",
            "alert_horizon_m": self.alert_horizon_m,
            "claim_ceiling": "not connected to user alerts or safe-route guidance",
        }
        if field.get("status") != "VALID":
            return {
                **base,
                "status": "SILENT_UNKNOWN",
                "primary_alert": None,
                "observed_opening_hint": None,
                "unknown_reasons": field.get("unknown_reasons", ["UNKNOWN_FIELD"]),
            }
        center_profile = [
            item
            for item in field.get("clearance_profile", [])
            if abs(int(item["theta_deg"])) <= 10
        ]
        if not center_profile or any(
            item.get("support_status") != "SUPPORTED" for item in center_profile
        ):
            return {
                **base,
                "status": "SILENT_UNKNOWN",
                "primary_alert": None,
                "observed_opening_hint": None,
                "unknown_reasons": ["UNKNOWN_CENTER_SUPPORT"],
            }
        occupied = []
        for item in field.get("clearance_profile", []):
            clearance = item.get("nearest_intrusion_m")
            if clearance is not None and float(clearance) <= self.alert_horizon_m:
                occupied.append(item)
        center = [item for item in occupied if abs(int(item["theta_deg"])) <= 10]
        left = [item for item in occupied if int(item["theta_deg"]) < -10]
        right = [item for item in occupied if int(item["theta_deg"]) > 10]
        if center:
            direction = "CENTER_RISK"
            chinese = f"前方约 {min(float(item['nearest_intrusion_m']) for item in center):.1f} 米有侵入。"
        elif left and not right:
            direction = "LEFT_RISK"
            chinese = "左前方有近距离侵入。"
        elif right and not left:
            direction = "RIGHT_RISK"
            chinese = "右前方有近距离侵入。"
        elif left or right:
            direction = "BOTH_SIDES_RISK"
            chinese = "两侧有近距离侵入。"
        else:
            direction = "SILENT_NO_NEAR_INTRUSION_OBSERVED"
            chinese = None
        best = field.get("best_observed_clearance_direction")
        hint = (
            {
                "theta_deg": best["theta_deg"],
                "text_zh": "该方向观测净空较大，仅供演示，不代表安全方向。",
                "status": "DEMO_CANDIDATE_NOT_SAFE_DIRECTION",
            }
            if best is not None
            else None
        )
        return {
            **base,
            "status": direction,
            "primary_alert": chinese,
            "observed_opening_hint": hint,
            "unknown_reasons": [],
        }
