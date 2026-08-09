#!/usr/bin/env python3
"""Deterministic interval reducer for the Assistive Geometry R2 F0 canary.

The reducer consumes continuous factor evidence only. It contains no learned
parameters and is the sole producer of clearance intervals and tri-state task
geometry for the F0 synthetic mechanics gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable


FRAME_SCHEMA = "blindassist_assistive_geometry_r2_factor_frame_v1"
OUTPUT_SCHEMA = "blindassist_assistive_geometry_r2_geometry_state_v1"
REDUCER_VERSION = "geometry_r2_interval_reducer_f0_v1"
STATES = ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
FORBIDDEN_FACTOR_KEYS = {
    "clearance",
    "clearance_m",
    "direct_clearance",
    "occupancy",
    "occupancy_logit",
    "task_confidence",
    "final_state",
    "free",
    "blocked",
    "unknown_logit",
}


class ReducerError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise ReducerError(code, message, **context)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _scan_forbidden(value: Any, path: str = "factors") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            require(normalized not in FORBIDDEN_FACTOR_KEYS, "FORBIDDEN_FACTOR_FIELD", "learned factor input contains a final-task shortcut", path=f"{path}.{key}")
            _scan_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden(child, f"{path}[{index}]")


@dataclass(frozen=True)
class Band:
    name: str
    x_min_m: float
    x_max_m: float
    include_max: bool

    def contains(self, x_m: float) -> bool:
        if self.include_max:
            return self.x_min_m <= x_m <= self.x_max_m
        return self.x_min_m <= x_m < self.x_max_m

    def overlaps(self, lo_m: float, hi_m: float) -> bool:
        if self.include_max:
            return hi_m >= self.x_min_m and lo_m <= self.x_max_m
        return hi_m >= self.x_min_m and lo_m < self.x_max_m


@dataclass(frozen=True)
class Profile:
    bands: tuple[Band, ...]
    horizons_m: tuple[float, ...]
    obstacle_evidence_threshold: float
    min_boundary_coverage: float
    max_support_slope_deg: float
    support_unknown_sigma_m: float
    max_horizon_m: float


def load_profile(raw: dict[str, Any]) -> Profile:
    require(isinstance(raw, dict), "PROFILE_INVALID", "reducer profile must be an object")
    bands_raw = raw.get("bands")
    require(isinstance(bands_raw, list) and len(bands_raw) == 3, "PROFILE_BANDS_INVALID", "exactly three body-swept bands are required")
    bands: list[Band] = []
    for index, item in enumerate(bands_raw):
        require(isinstance(item, dict), "PROFILE_BAND_INVALID", "band must be an object", index=index)
        lo = item.get("x_min_m")
        hi = item.get("x_max_m")
        require(finite_number(lo) and finite_number(hi) and float(lo) < float(hi), "PROFILE_BAND_RANGE_INVALID", "band range must be finite and increasing", index=index)
        bands.append(Band(str(item.get("name")), float(lo), float(hi), bool(item.get("include_max", False))))
    require(tuple(band.name for band in bands) == ("left", "center", "right"), "PROFILE_BAND_ORDER_INVALID", "bands must be left, center, right")
    for left, right in zip(bands, bands[1:]):
        require(left.x_max_m == right.x_min_m, "PROFILE_BAND_GAP_OR_OVERLAP", "band ownership must be seamless", left=left.name, right=right.name)
        require(not left.include_max, "PROFILE_BAND_BOUNDARY_OVERLAP", "only the last band may include its maximum", band=left.name)
    require(bands[-1].include_max, "PROFILE_LAST_BAND_OPEN", "right band must include the outer maximum")
    horizons_raw = raw.get("horizons_m")
    require(isinstance(horizons_raw, list) and horizons_raw, "PROFILE_HORIZONS_INVALID", "horizons are required")
    horizons = tuple(float(item) for item in horizons_raw)
    require(all(math.isfinite(item) and item > 0 for item in horizons) and tuple(sorted(horizons)) == horizons and len(set(horizons)) == len(horizons), "PROFILE_HORIZON_ORDER_INVALID", "horizons must be finite, positive and strictly increasing")
    threshold = float(raw.get("obstacle_evidence_threshold"))
    min_coverage = float(raw.get("min_boundary_coverage"))
    max_slope = float(raw.get("max_support_slope_deg"))
    support_unknown = float(raw.get("support_unknown_sigma_m"))
    require(0 < threshold < 1 and 0 < min_coverage <= 1, "PROFILE_THRESHOLD_INVALID", "profile probabilities must be within range")
    require(0 < max_slope < 90 and support_unknown > 0, "PROFILE_SUPPORT_LIMIT_INVALID", "support limits must be positive")
    return Profile(tuple(bands), horizons, threshold, min_coverage, max_slope, support_unknown, horizons[-1])


def band_for_lateral(profile: Profile, x_m: float) -> str | None:
    require(finite_number(x_m), "LATERAL_COORDINATE_INVALID", "lateral coordinate must be finite")
    owners = [band.name for band in profile.bands if band.contains(float(x_m))]
    require(len(owners) <= 1, "BAND_OWNERSHIP_OVERLAP", "lateral coordinate has multiple owners", x_m=x_m, owners=owners)
    return owners[0] if owners else None


def _all_unknown(frame_id: str, profile: Profile, reason: str, factor_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": OUTPUT_SCHEMA,
        "reducer_version": REDUCER_VERSION,
        "frame_id": frame_id,
        "factor_identity": factor_identity or {},
        "bands": [
            {
                "band": band.name,
                "clearance_interval_m": {"lower": None, "upper": None, "right_censored": False},
                "cells": [
                    {"horizon_m": horizon, "state": "UNKNOWN", "reason_codes": [reason]}
                    for horizon in profile.horizons_m
                ],
            }
            for band in profile.bands
        ],
    }


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3 or not all(finite_number(item) for item in value):
        return None
    return tuple(float(item) for item in value)


def _support_uncertainty(frame: dict[str, Any], profile: Profile) -> tuple[float | None, str | None]:
    support = frame.get("support")
    geometry = frame.get("input_geometry")
    if not isinstance(support, dict) or not isinstance(geometry, dict):
        return None, "MISSING_SUPPORT_OR_INPUT_GEOMETRY"
    if support.get("valid") is not True:
        return None, "SUPPORT_INVALID"
    normal = _vector3(support.get("normal_camera"))
    gravity = _vector3(geometry.get("gravity_up_camera"))
    if normal is None or gravity is None or geometry.get("gravity_valid") is not True:
        return None, "SUPPORT_OR_GRAVITY_INVALID"
    normal_norm = math.sqrt(sum(item * item for item in normal))
    gravity_norm = math.sqrt(sum(item * item for item in gravity))
    if abs(normal_norm - 1.0) > 1e-6 or abs(gravity_norm - 1.0) > 1e-6:
        return None, "SUPPORT_OR_GRAVITY_NOT_UNIT"
    dot = max(-1.0, min(1.0, sum(left * right for left, right in zip(normal, gravity))))
    slope_deg = math.degrees(math.acos(dot))
    if slope_deg > profile.max_support_slope_deg:
        return None, "SUPPORT_ORIENTATION_UNSUPPORTED"
    fields = ("normal_sigma_rad", "camera_height_m", "height_sigma_m", "residual_sigma_m")
    if not all(finite_number(support.get(field)) for field in fields):
        return None, "SUPPORT_NONFINITE"
    normal_sigma = float(support["normal_sigma_rad"])
    camera_height = float(support["camera_height_m"])
    height_sigma = float(support["height_sigma_m"])
    residual_sigma = float(support["residual_sigma_m"])
    if normal_sigma < 0 or height_sigma < 0 or residual_sigma < 0 or camera_height <= 0:
        return None, "SUPPORT_RANGE_INVALID"
    uncertainty = height_sigma + residual_sigma + profile.max_horizon_m * math.tan(min(normal_sigma, math.radians(45.0)))
    if uncertainty >= profile.support_unknown_sigma_m:
        return None, "SUPPORT_UNCERTAINTY_TOO_HIGH"
    return uncertainty, None


def _scale_interval(frame: dict[str, Any]) -> tuple[tuple[float, float] | None, str | None]:
    factor = frame.get("depth_scale")
    if not isinstance(factor, dict) or factor.get("valid") is not True:
        return None, "DEPTH_SCALE_INVALID"
    scale = factor.get("scale_m")
    sigma = factor.get("scale_sigma_m")
    if not finite_number(scale) or not finite_number(sigma):
        return None, "DEPTH_SCALE_NONFINITE"
    scale = float(scale)
    sigma = float(sigma)
    if scale <= 0 or sigma < 0 or scale - sigma <= 0:
        return None, "DEPTH_SCALE_RANGE_INVALID"
    return (scale - sigma, scale + sigma), None


def _factor_identity(frame: dict[str, Any]) -> dict[str, Any]:
    identity = frame.get("factor_identity")
    return dict(identity) if isinstance(identity, dict) else {}


def _obstacle_intervals(obstacle: dict[str, Any], scale_interval: tuple[float, float], support_uncertainty_m: float) -> tuple[dict[str, float] | None, str | None]:
    if obstacle.get("depth_valid") is not True:
        return None, "LOCAL_DEPTH_MISSING"
    numeric = (
        "depth_shape_forward",
        "depth_shape_sigma",
        "lateral_center_m",
        "lateral_half_width_m",
        "boundary_sigma_m",
        "evidence_probability",
        "evidence_sigma",
    )
    if not all(finite_number(obstacle.get(field)) for field in numeric):
        return None, "OBSTACLE_FACTOR_NONFINITE"
    shape = float(obstacle["depth_shape_forward"])
    shape_sigma = float(obstacle["depth_shape_sigma"])
    center = float(obstacle["lateral_center_m"])
    half_width = float(obstacle["lateral_half_width_m"])
    boundary_sigma = float(obstacle["boundary_sigma_m"])
    probability = float(obstacle["evidence_probability"])
    probability_sigma = float(obstacle["evidence_sigma"])
    if shape <= 0 or shape_sigma < 0 or shape - shape_sigma <= 0 or half_width < 0 or boundary_sigma < 0 or probability_sigma < 0 or not 0 <= probability <= 1:
        return None, "OBSTACLE_FACTOR_RANGE_INVALID"
    scale_lo, scale_hi = scale_interval
    forward_lo = scale_lo * (shape - shape_sigma) - support_uncertainty_m
    forward_hi = scale_hi * (shape + shape_sigma) + support_uncertainty_m
    nominal_lo = center - half_width
    nominal_hi = center + half_width
    guaranteed_lo = nominal_lo + boundary_sigma
    guaranteed_hi = nominal_hi - boundary_sigma
    return {
        "forward_lo": max(0.0, forward_lo),
        "forward_hi": max(0.0, forward_hi),
        "lateral_possible_lo": nominal_lo - boundary_sigma,
        "lateral_possible_hi": nominal_hi + boundary_sigma,
        "lateral_guaranteed_lo": guaranteed_lo,
        "lateral_guaranteed_hi": guaranteed_hi,
        "evidence_lo": max(0.0, probability - probability_sigma),
        "evidence_hi": min(1.0, probability + probability_sigma),
    }, None


def _missing_region_overlaps(obstacle: dict[str, Any], band: Band) -> bool:
    center = obstacle.get("lateral_center_m")
    half_width = obstacle.get("lateral_half_width_m")
    boundary_sigma = obstacle.get("boundary_sigma_m")
    if not finite_number(center) or not finite_number(half_width) or not finite_number(boundary_sigma):
        return True
    lo = float(center) - float(half_width) - float(boundary_sigma)
    hi = float(center) + float(half_width) + float(boundary_sigma)
    return band.overlaps(lo, hi)


def reduce_frame(frame: dict[str, Any], raw_profile: dict[str, Any]) -> dict[str, Any]:
    """Reduce one factor frame to deterministic interval task geometry."""

    profile = load_profile(raw_profile)
    require(isinstance(frame, dict), "FRAME_INVALID", "factor frame must be an object")
    _scan_forbidden(frame)
    frame_id = str(frame.get("frame_id", ""))
    require(frame.get("schema") == FRAME_SCHEMA and frame_id, "FRAME_SCHEMA_INVALID", "factor frame schema or identity is invalid")
    identity = _factor_identity(frame)
    geometry = frame.get("input_geometry")
    if not isinstance(geometry, dict) or geometry.get("k_valid") is not True or geometry.get("transform_valid") is not True:
        return _all_unknown(frame_id, profile, "INPUT_GEOMETRY_INVALID", identity)
    orientation = geometry.get("orientation")
    if orientation not in {"portrait", "landscape"}:
        return _all_unknown(frame_id, profile, "INPUT_ORIENTATION_INVALID", identity)
    scale_interval, scale_reason = _scale_interval(frame)
    if scale_interval is None:
        return _all_unknown(frame_id, profile, str(scale_reason), identity)
    support_uncertainty, support_reason = _support_uncertainty(frame, profile)
    if support_uncertainty is None:
        return _all_unknown(frame_id, profile, str(support_reason), identity)
    boundary = frame.get("boundary")
    if not isinstance(boundary, dict) or boundary.get("valid") is not True:
        return _all_unknown(frame_id, profile, "BOUNDARY_FACTOR_INVALID", identity)
    coverage = boundary.get("coverage")
    if not finite_number(coverage) or float(coverage) < profile.min_boundary_coverage:
        return _all_unknown(frame_id, profile, "BOUNDARY_COVERAGE_INCOMPLETE", identity)
    obstacles = boundary.get("obstacles")
    require(isinstance(obstacles, list), "OBSTACLE_LIST_INVALID", "boundary obstacles must be a list")

    resolved: list[tuple[dict[str, Any], dict[str, float] | None, str | None]] = []
    for obstacle in obstacles:
        require(isinstance(obstacle, dict), "OBSTACLE_FACTOR_INVALID", "obstacle factor must be an object")
        intervals, reason = _obstacle_intervals(obstacle, scale_interval, float(support_uncertainty))
        resolved.append((obstacle, intervals, reason))

    band_results: list[dict[str, Any]] = []
    for band in profile.bands:
        possible_intervals: list[dict[str, float]] = []
        missing_overlap = False
        for obstacle, intervals, reason in resolved:
            if intervals is None:
                missing_overlap = missing_overlap or (reason == "LOCAL_DEPTH_MISSING" and _missing_region_overlaps(obstacle, band))
                if reason != "LOCAL_DEPTH_MISSING":
                    missing_overlap = True
                continue
            possible_lateral = band.overlaps(intervals["lateral_possible_lo"], intervals["lateral_possible_hi"])
            if possible_lateral and intervals["evidence_hi"] >= profile.obstacle_evidence_threshold:
                possible_intervals.append(intervals)

        if missing_overlap:
            clearance = {"lower": None, "upper": None, "right_censored": False}
        elif possible_intervals:
            clearance = {
                "lower": min(item["forward_lo"] for item in possible_intervals),
                "upper": min(item["forward_hi"] for item in possible_intervals),
                "right_censored": False,
            }
        else:
            clearance = {"lower": profile.max_horizon_m, "upper": None, "right_censored": True}

        cells: list[dict[str, Any]] = []
        for horizon in profile.horizons_m:
            definite_occupied = False
            possible_occupied = missing_overlap
            for intervals in possible_intervals:
                guaranteed_lateral = (
                    intervals["lateral_guaranteed_hi"] > intervals["lateral_guaranteed_lo"]
                    and band.overlaps(intervals["lateral_guaranteed_lo"], intervals["lateral_guaranteed_hi"])
                )
                if intervals["forward_lo"] <= horizon:
                    possible_occupied = True
                if (
                    guaranteed_lateral
                    and intervals["evidence_lo"] >= profile.obstacle_evidence_threshold
                    and intervals["forward_hi"] <= horizon
                ):
                    definite_occupied = True
            if definite_occupied:
                state = "OCCUPIED_OBSERVED"
                reasons = ["POSITIVE_OCCUPANCY_EVIDENCE"]
            elif possible_occupied:
                state = "UNKNOWN"
                reasons = ["GEOMETRY_INTERVAL_STRADDLES_DECISION"] if not missing_overlap else ["LOCAL_FACTOR_MISSING"]
            else:
                state = "CLEAR_OBSERVED"
                reasons = ["NO_POSSIBLE_OCCUPANCY_WITHIN_HORIZON"]
            cells.append({"horizon_m": horizon, "state": state, "reason_codes": reasons})
        band_results.append({"band": band.name, "clearance_interval_m": clearance, "cells": cells})

    return {
        "schema": OUTPUT_SCHEMA,
        "reducer_version": REDUCER_VERSION,
        "frame_id": frame_id,
        "factor_identity": identity,
        "bands": band_results,
    }


def state_map(output: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    return {str(item["band"]): tuple(str(cell["state"]) for cell in item["cells"]) for item in output["bands"]}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def iter_cells(output: dict[str, Any]) -> Iterable[tuple[str, float, str]]:
    for band in output["bands"]:
        for cell in band["cells"]:
            yield str(band["band"]), float(cell["horizon_m"]), str(cell["state"])
