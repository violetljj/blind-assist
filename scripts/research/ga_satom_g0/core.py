"""Truth-isolated GA-SATOM G0 ground-anchor estimator and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


ANCHOR_ZONE_IDS = tuple(
    [f"r7c{column}" for column in range(8)]
    + [f"r6c{column}" for column in range(2, 6)]
)


@dataclass(frozen=True)
class GroundAnchorPolicy:
    total_zone_budget: int = 64
    anchor_zone_ids: tuple[str, ...] = ANCHOR_ZONE_IDS
    minimum_range_m: float = 0.20
    maximum_range_m: float = 4.00
    maximum_sigma_m: float = 0.05
    support_tolerance_m: float = 0.06
    minimum_support_zones: int = 6
    minimum_support_fraction: float = 0.50
    maximum_height_sigma_m: float = 0.05

    def validate(self) -> None:
        if self.total_zone_budget != 64:
            raise ValueError("G0 total information budget must remain 64 zones")
        if len(self.anchor_zone_ids) != 12 or len(set(self.anchor_zone_ids)) != 12:
            raise ValueError("G0 anchor pattern must contain exactly 12 unique zones")
        if not set(self.anchor_zone_ids).issubset({f"r{row}c{column}" for row in range(8) for column in range(8)}):
            raise ValueError("G0 anchor zone lies outside the 8x8 sensor grid")
        if not 0 < self.minimum_range_m < self.maximum_range_m:
            raise ValueError("invalid G0 range bounds")
        if not 0 < self.maximum_sigma_m <= self.support_tolerance_m:
            raise ValueError("invalid G0 sigma/support tolerance")
        if not 1 <= self.minimum_support_zones <= len(self.anchor_zone_ids):
            raise ValueError("invalid G0 minimum support count")
        if not 0 < self.minimum_support_fraction <= 1:
            raise ValueError("invalid G0 minimum support fraction")
        if self.maximum_height_sigma_m <= 0:
            raise ValueError("invalid G0 height uncertainty gate")

    @property
    def anchor_budget_fraction(self) -> float:
        return len(self.anchor_zone_ids) / self.total_zone_budget


@dataclass(frozen=True)
class ZoneMeasurement:
    zone_id: str
    origin_rgb_m: np.ndarray
    ray_rgb_unit: np.ndarray
    range_m: float | None
    sigma_m: float | None
    status: str

    def validate(self) -> None:
        origin = np.asarray(self.origin_rgb_m, dtype=np.float64)
        ray = np.asarray(self.ray_rgb_unit, dtype=np.float64)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("zone origin must be a finite 3-vector")
        if ray.shape != (3,) or not np.all(np.isfinite(ray)):
            raise ValueError("zone ray must be a finite 3-vector")
        if not math.isclose(float(np.linalg.norm(ray)), 1.0, abs_tol=1e-5):
            raise ValueError("zone ray must have unit length")


@dataclass(frozen=True)
class MeasurementFrame:
    parent_id: str
    episode_id: str
    frame_id: str
    timestamp_ns: int
    gravity_down_rgb_unit: np.ndarray
    zones: tuple[ZoneMeasurement, ...]

    def validate(self) -> None:
        if not self.parent_id or not self.episode_id or not self.frame_id or self.timestamp_ns < 0:
            raise ValueError("measurement identity is incomplete")
        gravity = np.asarray(self.gravity_down_rgb_unit, dtype=np.float64)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity must be a finite 3-vector")
        if not math.isclose(float(np.linalg.norm(gravity)), 1.0, abs_tol=1e-5):
            raise ValueError("gravity must have unit length")
        zone_ids = [zone.zone_id for zone in self.zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("frame contains duplicate zone IDs")
        expected = {f"r{row}c{column}" for row in range(8) for column in range(8)}
        if set(zone_ids) != expected:
            raise ValueError("G0 frame must retain the complete physical 8x8 zone budget")
        for zone in self.zones:
            zone.validate()


@dataclass(frozen=True)
class TruthFrame:
    parent_id: str
    episode_id: str
    frame_id: str
    reference_rgb_camera_height_m: float
    reference_height_uncertainty_m: float
    ground_labels: dict[str, str]

    def validate(self) -> None:
        if not self.parent_id or not self.episode_id or not self.frame_id:
            raise ValueError("truth identity is incomplete")
        if not 0.5 <= self.reference_rgb_camera_height_m <= 2.2:
            raise ValueError("reference height outside the G0 capture contract")
        if not 0 <= self.reference_height_uncertainty_m <= 0.01:
            raise ValueError("reference height uncertainty exceeds G0 authority")
        if set(self.ground_labels) != set(ANCHOR_ZONE_IDS):
            raise ValueError("truth must label every frozen G0 anchor zone exactly once")
        if any(value not in {"GROUND", "NON_GROUND", "UNRESOLVED"} for value in self.ground_labels.values()):
            raise ValueError("unsupported evaluator-only ground label")


def estimate_ground_anchor(frame: MeasurementFrame, policy: GroundAnchorPolicy) -> dict[str, Any]:
    """Estimate RGB-camera height without receiving evaluator truth."""
    frame.validate()
    policy.validate()
    by_id = {zone.zone_id: zone for zone in frame.zones}
    if not set(policy.anchor_zone_ids).issubset(by_id):
        raise ValueError("frame does not contain the complete frozen anchor pattern")
    gravity = np.asarray(frame.gravity_down_rgb_unit, dtype=np.float64)
    admitted = []
    rejected: dict[str, int] = {}
    for zone_id in policy.anchor_zone_ids:
        zone = by_id[zone_id]
        reason = None
        if zone.status != "VALID":
            reason = "status"
        elif zone.range_m is None or not math.isfinite(float(zone.range_m)):
            reason = "range_missing"
        elif not policy.minimum_range_m <= float(zone.range_m) <= policy.maximum_range_m:
            reason = "range_bounds"
        elif zone.sigma_m is None or not math.isfinite(float(zone.sigma_m)):
            reason = "sigma_missing"
        elif not 0 <= float(zone.sigma_m) <= policy.maximum_sigma_m:
            reason = "sigma"
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        point = np.array(zone.origin_rgb_m, dtype=np.float64, copy=True)
        point += np.asarray(zone.ray_rgb_unit, dtype=np.float64) * float(zone.range_m)
        drop = float(np.dot(gravity, point))
        if not math.isfinite(drop) or not 0.5 <= drop <= 2.2:
            rejected["height_bounds"] = rejected.get("height_bounds", 0) + 1
            continue
        admitted.append((zone_id, drop, float(zone.sigma_m)))
    minimum = max(
        policy.minimum_support_zones,
        int(math.ceil(policy.minimum_support_fraction * len(policy.anchor_zone_ids))),
    )
    base = {
        "parent_id": frame.parent_id,
        "episode_id": frame.episode_id,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "anchor_zone_budget": len(policy.anchor_zone_ids),
        "total_zone_budget": policy.total_zone_budget,
        "anchor_budget_fraction": policy.anchor_budget_fraction,
        "admitted_before_consensus": len(admitted),
        "rejection_counts": rejected,
    }
    if len(admitted) < minimum:
        return {**base, "status": "UNKNOWN_INSUFFICIENT_ANCHOR_ZONES", "valid": False, "support_zone_ids": []}
    offsets = np.asarray([row[1] for row in admitted], dtype=np.float64)
    center = float(np.median(offsets))
    support = [row for row in admitted if abs(row[1] - center) <= policy.support_tolerance_m]
    if len(support) < minimum:
        return {**base, "status": "UNKNOWN_GROUND_CONSENSUS", "valid": False, "support_zone_ids": [row[0] for row in support]}
    support_offsets = np.asarray([row[1] for row in support], dtype=np.float64)
    height = float(np.median(support_offsets))
    residual_mad = float(np.median(np.abs(support_offsets - height)))
    robust_standard_error = 1.4826 * residual_mad / math.sqrt(len(support))
    sensor_sigma = float(np.median([row[2] for row in support]))
    height_sigma = float(math.hypot(robust_standard_error, sensor_sigma))
    if height_sigma > policy.maximum_height_sigma_m:
        return {
            **base, "status": "UNKNOWN_GROUND_HEIGHT_UNCERTAIN", "valid": False,
            "support_zone_ids": [row[0] for row in support], "height_m": height,
            "height_sigma_m": height_sigma, "support_residual_mad_m": residual_mad,
        }
    return {
        **base, "status": "VALID_GROUND_ANCHOR", "valid": True,
        "support_zone_ids": [row[0] for row in support], "height_m": height,
        "height_sigma_m": height_sigma, "support_residual_mad_m": residual_mad,
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _parent_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    errors = [abs(float(row["height_m"]) - float(row["reference_height_m"])) for row in valid]
    signed_errors = [float(row["height_m"]) - float(row["reference_height_m"]) for row in valid]
    false_support, labelled_support = 0, 0
    catastrophic = 0
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in valid:
        labels = row["ground_labels"]
        support_labels = [labels.get(zone_id, "UNRESOLVED") for zone_id in row["support_zone_ids"]]
        resolved = [label for label in support_labels if label != "UNRESOLVED"]
        false_support += sum(label == "NON_GROUND" for label in resolved)
        labelled_support += len(resolved)
        non_ground_fraction = (sum(label == "NON_GROUND" for label in resolved) / len(resolved)) if resolved else 0.0
        if abs(float(row["height_m"]) - float(row["reference_height_m"])) > 0.20 or non_ground_fraction > 0.25:
            catastrophic += 1
        by_episode.setdefault(str(row["episode_id"]), []).append(row)
    jitters = []
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: int(row["timestamp_ns"]))
        jitters.extend(
            abs(float(right["height_m"]) - float(left["height_m"]))
            for left, right in zip(episode_rows, episode_rows[1:])
        )
    median_error = float(np.median(signed_errors)) if signed_errors else None
    return {
        "frames": len(rows),
        "valid_frames": len(valid),
        "coverage": len(valid) / len(rows),
        "height_mae_m": _mean(errors),
        "height_error_mad_m": (
            float(np.median(np.abs(np.asarray(signed_errors) - median_error))) if signed_errors else None
        ),
        "false_ground_support_rate": false_support / labelled_support if labelled_support else None,
        "resolved_support_zones": labelled_support,
        "catastrophic_false_anchor_rate": catastrophic / len(valid) if valid else None,
        "temporal_jitter_p95_m": _quantile(jitters, 0.95),
    }


def evaluate_g0(
    measurements: Sequence[MeasurementFrame],
    truth: Sequence[TruthFrame],
    policy: GroundAnchorPolicy | None = None,
) -> dict[str, Any]:
    if not measurements:
        raise ValueError("G0 measurements are required")
    policy = policy or GroundAnchorPolicy()
    policy.validate()
    truth_by_id = {}
    for item in truth:
        item.validate()
        key = (item.parent_id, item.episode_id, item.frame_id)
        if key in truth_by_id:
            raise ValueError("duplicate G0 truth identity")
        truth_by_id[key] = item
    rows = []
    for frame in measurements:
        key = (frame.parent_id, frame.episode_id, frame.frame_id)
        target = truth_by_id.get(key)
        if target is None:
            raise ValueError("G0 measurement lacks exact truth identity")
        estimate = estimate_ground_anchor(frame, policy)
        rows.append(
            {
                **estimate,
                "reference_height_m": target.reference_rgb_camera_height_m,
                "reference_height_uncertainty_m": target.reference_height_uncertainty_m,
                "ground_labels": target.ground_labels,
            }
        )
    if len(rows) != len(truth_by_id):
        raise ValueError("G0 truth contains unmatched identities")
    by_parent_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_parent_rows.setdefault(str(row["parent_id"]), []).append(row)
    by_parent = {parent: _parent_metrics(parent_rows) for parent, parent_rows in sorted(by_parent_rows.items())}
    metric_names = (
        "coverage", "height_mae_m", "height_error_mad_m", "false_ground_support_rate",
        "catastrophic_false_anchor_rate", "temporal_jitter_p95_m",
    )
    macro = {name: _mean([row[name] for row in by_parent.values() if row[name] is not None]) for name in metric_names}
    worst = {}
    for name in metric_names:
        values = [row[name] for row in by_parent.values() if row[name] is not None]
        worst[name] = (min(values) if name == "coverage" else max(values)) if values else None
    available = lambda name: macro[name] is not None and worst[name] is not None
    gates = {
        "parent_count_at_least_8": len(by_parent) >= 8,
        "parent_macro_coverage_at_least_0_95": available("coverage") and float(macro["coverage"]) >= 0.95,
        "worst_parent_coverage_at_least_0_90": available("coverage") and float(worst["coverage"]) >= 0.90,
        "parent_macro_height_mae_at_most_0_08m": available("height_mae_m") and float(macro["height_mae_m"]) <= 0.08,
        "worst_parent_height_mae_at_most_0_12m": available("height_mae_m") and float(worst["height_mae_m"]) <= 0.12,
        "worst_parent_height_error_mad_at_most_0_08m": available("height_error_mad_m") and float(worst["height_error_mad_m"]) <= 0.08,
        "parent_macro_false_ground_support_at_most_0_02": available("false_ground_support_rate") and float(macro["false_ground_support_rate"]) <= 0.02,
        "worst_parent_false_ground_support_at_most_0_05": available("false_ground_support_rate") and float(worst["false_ground_support_rate"]) <= 0.05,
        "worst_parent_catastrophic_false_anchor_at_most_0_01": available("catastrophic_false_anchor_rate") and float(worst["catastrophic_false_anchor_rate"]) <= 0.01,
        "worst_parent_temporal_jitter_p95_at_most_0_08m": available("temporal_jitter_p95_m") and float(worst["temporal_jitter_p95_m"]) <= 0.08,
    }
    passed = all(gates.values())
    return {
        "schema": "blindassist.ga_satom_g0.evaluation.v1",
        "status": "GA_SATOM_G0_GROUND_ANCHOR_OBSERVABILITY_PASS" if passed else "GA_SATOM_G0_GROUND_ANCHOR_OBSERVABILITY_FAIL_CLOSE",
        "passed": passed,
        "policy": {
            "total_zone_budget": policy.total_zone_budget,
            "anchor_zone_ids": list(policy.anchor_zone_ids),
            "anchor_budget_fraction": policy.anchor_budget_fraction,
        },
        "causality": {"candidate_truth_access": False, "truth_role": "evaluator_only"},
        "parents": sorted(by_parent),
        "frames": len(rows),
        "parent_macro": macro,
        "worst_parent": worst,
        "by_parent": by_parent,
        "gates": gates,
        "claim_ceiling": "physical metric ground-anchor observability only; no SATOM arm, task headroom, training, Android, product, safety, or paper claim",
    }
