"""Spatially registered multi-zone ToF scale anchors for RGB metric depth."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from metric_scale_anchor import MetricScaleAnchor

REGISTRATION_SCHEMA = "hftf_tof_rgb_registration_r0"
FRAME_SCHEMA = "hftf_multizone_tof_frame_r0"


@dataclass(frozen=True)
class TofRgbRegistration:
    registration_id: str
    tof_sensor_id: str
    rgb_calibration_id: str
    transform_rgb_from_tof: np.ndarray
    rays_by_zone: dict[str, np.ndarray]


@dataclass(frozen=True)
class TofAnchorPolicy:
    max_rgb_tof_skew_ns: int
    max_sigma_m: float
    minimum_zones: int
    minimum_bands: int
    maximum_scale_mad: float
    minimum_range_m: float = 0.20
    maximum_range_m: float = 4.00
    depth_patch_radius_px: int = 2

    def validate(self) -> None:
        if self.max_rgb_tof_skew_ns < 0:
            raise ValueError("maximum RGB/ToF skew must be nonnegative")
        if not math.isfinite(self.max_sigma_m) or self.max_sigma_m <= 0:
            raise ValueError("maximum ToF sigma must be finite and positive")
        if self.minimum_zones < 3:
            raise ValueError("at least three ToF zones are required")
        if not 1 <= self.minimum_bands <= 3:
            raise ValueError("minimum band count must be between one and three")
        if not math.isfinite(self.maximum_scale_mad) or self.maximum_scale_mad < 0:
            raise ValueError("maximum scale MAD must be finite and nonnegative")
        if not 0 < self.minimum_range_m < self.maximum_range_m:
            raise ValueError("invalid admitted ToF range")
        if self.depth_patch_radius_px < 0:
            raise ValueError("depth patch radius must be nonnegative")


def load_registration(path: Path) -> TofRgbRegistration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != REGISTRATION_SCHEMA:
        raise ValueError("unsupported ToF/RGB registration schema")
    if payload.get("admitted") is not True:
        raise ValueError("ToF/RGB registration is not admitted")
    registration_id = str(payload["registration_id"])
    sensor_id = str(payload["tof_sensor_id"])
    rgb_calibration_id = str(payload["rgb_calibration_id"])
    if not registration_id or not sensor_id or not rgb_calibration_id:
        raise ValueError("registration identities must be non-empty")
    transform = np.asarray(payload["transform_rgb_from_tof"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("ToF/RGB transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError("ToF/RGB transform must have a rigid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5
    ):
        raise ValueError("ToF/RGB rotation must be orthonormal with determinant one")
    rays = {}
    for item in payload["zones"]:
        zone_id = str(item["zone_id"])
        ray = np.asarray(item["ray_tof_unit"], dtype=np.float64)
        if not zone_id or zone_id in rays or ray.shape != (3,):
            raise ValueError("zone IDs must be unique and rays must be 3-vectors")
        norm = float(np.linalg.norm(ray))
        if not np.all(np.isfinite(ray)) or norm <= 0 or not math.isclose(
            norm, 1.0, abs_tol=1e-5
        ):
            raise ValueError("ToF rays must be finite unit vectors")
        rays[zone_id] = ray
    if len(rays) < 3:
        raise ValueError("registration requires at least three zones")
    return TofRgbRegistration(
        registration_id=registration_id,
        tof_sensor_id=sensor_id,
        rgb_calibration_id=rgb_calibration_id,
        transform_rgb_from_tof=transform,
        rays_by_zone=rays,
    )


def load_tof_frames(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    previous: dict[str, int] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != FRAME_SCHEMA:
            raise ValueError(f"line {line_number}: unsupported ToF frame schema")
        sequence = str(row["sequence_id"])
        timestamp = int(row["timestamp_ns"])
        if not sequence or not str(row["clock_domain"]):
            raise ValueError(f"line {line_number}: sequence and clock domain are required")
        if previous.get(sequence, -1) >= timestamp:
            raise ValueError(f"line {line_number}: ToF timestamps must increase")
        if not isinstance(row.get("zones"), list) or not row["zones"]:
            raise ValueError(f"line {line_number}: ToF frame requires zones")
        grouped[sequence].append(row)
        previous[sequence] = timestamp
    if not grouped:
        raise ValueError("ToF file contains no frames")
    return dict(grouped)


class TofFrameStream:
    def __init__(self, frames: dict[str, list[dict[str, Any]]]) -> None:
        self.pending = {sequence: deque(rows) for sequence, rows in frames.items()}

    def take_available(self, sequence_id: str, timestamp_ns: int) -> list[dict[str, Any]]:
        queue = self.pending.get(sequence_id)
        available = []
        while queue and int(queue[0]["timestamp_ns"]) <= timestamp_ns:
            available.append(queue.popleft())
        return available


def _patch_depth(depth: np.ndarray, u: float, v: float, radius: int) -> float | None:
    height, width = depth.shape
    x = round(u)
    y = round(v)
    if x < 0 or x >= width or y < 0 or y >= height:
        return None
    values = depth[
        max(0, y - radius) : min(height, y + radius + 1),
        max(0, x - radius) : min(width, x + radius + 1),
    ]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if len(values) else None


def estimate_tof_scale_anchor(
    depth: np.ndarray,
    intrinsics_fx_fy_cx_cy: list[float],
    rgb_timestamp_ns: int,
    rgb_clock_domain: str,
    frame: dict[str, Any],
    registration: TofRgbRegistration,
    rgb_calibration_id: str,
    policy: TofAnchorPolicy,
) -> tuple[MetricScaleAnchor | None, dict[str, Any]]:
    policy.validate()
    if depth.ndim != 2:
        raise ValueError("candidate depth must be a 2D array")
    if str(frame["tof_sensor_id"]) != registration.tof_sensor_id:
        raise ValueError("ToF sensor identity differs from registration")
    if str(frame["registration_id"]) != registration.registration_id:
        raise ValueError("ToF frame registration identity mismatch")
    if rgb_calibration_id != registration.rgb_calibration_id:
        raise ValueError("RGB calibration identity differs from registration")
    if str(frame["clock_domain"]) != rgb_clock_domain:
        return None, {"status": "UNKNOWN_TOF_RGB_CLOCK_DOMAIN_MISMATCH"}
    tof_timestamp = int(frame["timestamp_ns"])
    skew_ns = int(rgb_timestamp_ns) - tof_timestamp
    if skew_ns < 0:
        return None, {"status": "UNKNOWN_FUTURE_TOF_FRAME", "skew_ns": skew_ns}
    if skew_ns > policy.max_rgb_tof_skew_ns:
        return None, {"status": "UNKNOWN_TOF_RGB_SKEW", "skew_ns": skew_ns}

    fx, fy, cx, cy = (float(value) for value in intrinsics_fx_fy_cx_cy)
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError("RGB intrinsics must be finite with positive focal lengths")
    transform = registration.transform_rgb_from_tof
    ratios = []
    bands = set()
    rejection_counts: dict[str, int] = defaultdict(int)
    seen_zones = set()
    for zone in frame["zones"]:
        zone_id = str(zone["zone_id"])
        if zone_id in seen_zones:
            raise ValueError("ToF frame contains duplicate zone IDs")
        seen_zones.add(zone_id)
        ray = registration.rays_by_zone.get(zone_id)
        if ray is None:
            rejection_counts["unregistered_zone"] += 1
            continue
        if str(zone.get("status")) != "VALID":
            rejection_counts["invalid_status"] += 1
            continue
        distance = float(zone["range_m"])
        sigma = float(zone["sigma_m"])
        if (
            not math.isfinite(distance)
            or not policy.minimum_range_m <= distance <= policy.maximum_range_m
        ):
            rejection_counts["range"] += 1
            continue
        if not math.isfinite(sigma) or sigma < 0 or sigma > policy.max_sigma_m:
            rejection_counts["sigma"] += 1
            continue
        point = transform @ np.asarray([*(ray * distance), 1.0])
        if point[2] <= 0:
            rejection_counts["behind_rgb"] += 1
            continue
        u = fx * point[0] / point[2] + cx
        v = fy * point[1] / point[2] + cy
        candidate_z = _patch_depth(depth, u, v, policy.depth_patch_radius_px)
        if candidate_z is None:
            rejection_counts["no_candidate_depth"] += 1
            continue
        ratios.append(float(point[2]) / candidate_z)
        normalized_u = u / depth.shape[1]
        bands.add("left" if normalized_u < 1 / 3 else "center" if normalized_u < 2 / 3 else "right")

    diagnostic = {
        "tof_timestamp_ns": tof_timestamp,
        "rgb_timestamp_ns": int(rgb_timestamp_ns),
        "skew_ns": skew_ns,
        "valid_zone_pairs": len(ratios),
        "covered_bands": sorted(bands),
        "rejection_counts": dict(rejection_counts),
    }
    if len(ratios) < policy.minimum_zones:
        return None, {**diagnostic, "status": "UNKNOWN_INSUFFICIENT_TOF_ZONES"}
    if len(bands) < policy.minimum_bands:
        return None, {**diagnostic, "status": "UNKNOWN_INSUFFICIENT_TOF_BAND_COVERAGE"}
    scale = float(statistics.median(ratios))
    mad = float(statistics.median(abs(value - scale) for value in ratios))
    if mad > policy.maximum_scale_mad:
        return None, {
            **diagnostic,
            "status": "UNKNOWN_TOF_SCALE_DISAGREEMENT",
            "scale": scale,
            "scale_mad": mad,
        }
    anchor = MetricScaleAnchor(
        timestamp_ns=tof_timestamp,
        scale=scale,
        pair_count=len(ratios),
        median_abs_ratio_residual=mad,
        source=f"multizone_tof:{registration.tof_sensor_id}:{registration.registration_id}",
    )
    return anchor, {
        **diagnostic,
        "status": "VALID_TOF_SCALE_ANCHOR",
        "scale": scale,
        "scale_mad": mad,
    }
