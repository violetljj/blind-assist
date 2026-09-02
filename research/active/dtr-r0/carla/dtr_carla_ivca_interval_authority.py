"""Interval-born, transport-sustained collision authority.

This module is deliberately independent from evaluator truth and from the
X94 persistence implementation.  It contributes two small contracts:

1. derive every continuous contact component between one transported convex
   footprint and the issued wearer route; and
2. allow a current measured interval to create authority while allowing an
   unchanged X94 receipt to sustain, but never create, that authority.

The geometry reuses the frozen X24 route tube and X25 footprint distance.  It
adds no detector, association, route, lifecycle, or fitted score threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

import dtr_carla_x24_plan_route_core as route
ROOT_TOLERANCE_S = 1e-6
MINIMIZER_ITERATIONS = 96
ROOT_ITERATIONS = 64
MERGE_TOLERANCE_S = 2e-6

CURRENT_MEASURED = "CURRENT_MEASURED"
TRANSPORT_RENEWAL = "TRANSPORT_RENEWAL"
EPSILON = 1e-9


@dataclass(frozen=True)
class CollisionInterval:
    entry_s: float
    exit_s: float
    overlap_duration_s: float
    minimum_clearance_m: float
    minimum_clearance_time_s: float


@dataclass(frozen=True)
class CollisionIntervalSet:
    components: tuple[CollisionInterval, ...]
    total_overlap_duration_s: float
    earliest_entry_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": [asdict(component) for component in self.components],
            "total_overlap_duration_s": self.total_overlap_duration_s,
            "earliest_entry_s": self.earliest_entry_s,
        }


@dataclass(frozen=True)
class IntervalReceipt:
    plan_receipt_sha256: str
    parent_id: str
    carrier_id: str
    representation: str
    interval_set: CollisionIntervalSet
    authorized_component_index: int
    provenance: str

    @property
    def authorized_component(self) -> CollisionInterval:
        return self.interval_set.components[self.authorized_component_index]


def fixed_contract() -> dict[str, Any]:
    return {
        "mechanism": "INTERVAL_BORN_TRANSPORT_SUSTAINED_COLLISION_AUTHORITY",
        "tube_radius_m": route.DEFAULT_TUBE_RADIUS_M,
        "route_horizon_s": route.DEFAULT_ROUTE_HORIZON_S,
        "root_tolerance_s": ROOT_TOLERANCE_S,
        "minimizer_iterations": MINIMIZER_ITERATIONS,
        "root_iterations": ROOT_ITERATIONS,
        "onset_rule": "CURRENT_MEASURED_INTERVAL_RECEIPT_ONLY",
        "persistence_rule": "UNCHANGED_X94_RECEIPT_MAY_RENEW_EXISTING_AUTHORITY_ONLY",
        "missing_representation_rule": "MISSING_IS_NOT_CONTRADICTION",
        "transport_reseed_allowed": False,
        "x94_hold_window_changed": False,
    }


def _minimum_time(function: Any, start_s: float, end_s: float) -> float:
    """Minimize a convex distance function on one constant-velocity segment."""

    if end_s <= start_s + ROOT_TOLERANCE_S:
        return start_s
    left = float(start_s)
    right = float(end_s)
    inverse_phi = (math.sqrt(5.0) - 1.0) * 0.5
    first = right - inverse_phi * (right - left)
    second = left + inverse_phi * (right - left)
    first_value = float(function(first))
    second_value = float(function(second))
    for _ in range(MINIMIZER_ITERATIONS):
        if first_value <= second_value:
            right = second
            second = first
            second_value = first_value
            first = right - inverse_phi * (right - left)
            first_value = float(function(first))
        else:
            left = first
            first = second
            first_value = second_value
            second = left + inverse_phi * (right - left)
            second_value = float(function(second))
    candidates = (start_s, end_s, first, second, (left + right) * 0.5)
    return min(candidates, key=lambda value: float(function(value)))


def _point_segment_distance(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    edge = second - first
    denominator = float(np.dot(edge, edge))
    if denominator <= EPSILON:
        return float(np.linalg.norm(point - first))
    ratio = float(np.dot(point - first, edge) / denominator)
    projection = first + min(1.0, max(0.0, ratio)) * edge
    return float(np.linalg.norm(point - projection))


def _point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    cross = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        relative = point - first
        cross.append(float(edge[0] * relative[1] - edge[1] * relative[0]))
    return all(value >= -EPSILON for value in cross) or all(value <= EPSILON for value in cross)


def _point_polygon_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    if _point_in_convex_polygon(point, polygon):
        return 0.0
    return min(
        _point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _segment_distance_at(
    footprint_xy: np.ndarray,
    target_velocity_xy: np.ndarray,
    segment: route.RouteSegment,
    offset_s: float,
) -> float:
    wearer = np.asarray(segment.start_position_xy, dtype=np.float64) + np.asarray(
        segment.velocity_xy, dtype=np.float64
    ) * (offset_s - segment.start_offset_s)
    footprint = footprint_xy + target_velocity_xy[None, :] * offset_s
    return _point_polygon_distance(wearer, footprint)


def _entry_root(function: Any, outside_s: float, inside_s: float, radius_m: float) -> float:
    low = float(outside_s)
    high = float(inside_s)
    for _ in range(ROOT_ITERATIONS):
        middle = (low + high) * 0.5
        if float(function(middle)) <= radius_m:
            high = middle
        else:
            low = middle
    return high


def _exit_root(function: Any, inside_s: float, outside_s: float, radius_m: float) -> float:
    low = float(inside_s)
    high = float(outside_s)
    for _ in range(ROOT_ITERATIONS):
        middle = (low + high) * 0.5
        if float(function(middle)) <= radius_m:
            low = middle
        else:
            high = middle
    return low


def _merge_components(values: Sequence[CollisionInterval]) -> tuple[CollisionInterval, ...]:
    output: list[CollisionInterval] = []
    for value in sorted(values, key=lambda item: (item.entry_s, item.exit_s)):
        if not output or value.entry_s > output[-1].exit_s + MERGE_TOLERANCE_S:
            output.append(value)
            continue
        previous = output[-1]
        if value.minimum_clearance_m < previous.minimum_clearance_m:
            minimum_clearance = value.minimum_clearance_m
            minimum_time = value.minimum_clearance_time_s
        else:
            minimum_clearance = previous.minimum_clearance_m
            minimum_time = previous.minimum_clearance_time_s
        entry = previous.entry_s
        exit_s = max(previous.exit_s, value.exit_s)
        output[-1] = CollisionInterval(
            entry_s=entry,
            exit_s=exit_s,
            overlap_duration_s=max(0.0, exit_s - entry),
            minimum_clearance_m=minimum_clearance,
            minimum_clearance_time_s=minimum_time,
        )
    return tuple(output)


def transported_interval_set(
    footprint_xy: Sequence[Sequence[float]],
    target_velocity_xy: Sequence[float],
    route_segments: Sequence[route.RouteSegment],
    *,
    tube_radius_m: float = route.DEFAULT_TUBE_RADIUS_M,
) -> CollisionIntervalSet:
    """Return all continuous route-contact components over the frozen horizon.

    Each X24 route segment and the target footprint use constant velocity.  The
    point-to-translated-convex-polygon distance is convex on that segment, so a
    bounded minimizer followed by two bracketed roots recovers even a contact
    interval narrower than X25's former 50 ms sample step.
    """

    footprint = np.asarray(footprint_xy, dtype=np.float64).reshape(-1, 2)
    velocity = np.asarray(target_velocity_xy, dtype=np.float64).reshape(2)
    if len(footprint) < 3 or not np.isfinite(footprint).all():
        raise ValueError("footprint_xy must contain at least three finite points")
    if not np.isfinite(velocity).all():
        raise ValueError("target_velocity_xy must be finite")
    radius = float(tube_radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("tube_radius_m must be positive and finite")

    raw_components: list[CollisionInterval] = []
    for segment in route_segments:
        start = float(segment.start_offset_s)
        end = float(segment.end_offset_s)
        if end < start:
            raise ValueError("route segment end precedes start")

        def distance_at(offset_s: float) -> float:
            return _segment_distance_at(footprint, velocity, segment, offset_s)

        minimum_time = _minimum_time(distance_at, start, end)
        minimum_distance = float(distance_at(minimum_time))
        if minimum_distance > radius + EPSILON:
            continue
        entry = (
            start
            if float(distance_at(start)) <= radius + EPSILON
            else _entry_root(distance_at, start, minimum_time, radius)
        )
        exit_s = (
            end
            if float(distance_at(end)) <= radius + EPSILON
            else _exit_root(distance_at, minimum_time, end, radius)
        )
        raw_components.append(
            CollisionInterval(
                entry_s=entry,
                exit_s=exit_s,
                overlap_duration_s=max(0.0, exit_s - entry),
                minimum_clearance_m=minimum_distance - radius,
                minimum_clearance_time_s=minimum_time,
            )
        )

    components = _merge_components(raw_components)
    return CollisionIntervalSet(
        components=components,
        total_overlap_duration_s=sum(value.overlap_duration_s for value in components),
        earliest_entry_s=components[0].entry_s if components else None,
    )


def measured_receipt(
    *,
    plan_receipt_sha256: str,
    parent_id: str,
    carrier_id: str,
    representation: str,
    interval_set: CollisionIntervalSet,
    carrier_authorized: bool,
    explicit_contradiction: bool = False,
) -> IntervalReceipt | None:
    """Sign the earliest current measured component, or abstain."""

    if not carrier_authorized or explicit_contradiction or not interval_set.components:
        return None
    if not all((plan_receipt_sha256, parent_id, carrier_id, representation)):
        raise ValueError("measured interval receipt identity fields must be non-empty")
    return IntervalReceipt(
        plan_receipt_sha256=plan_receipt_sha256,
        parent_id=parent_id,
        carrier_id=carrier_id,
        representation=representation,
        interval_set=interval_set,
        authorized_component_index=0,
        provenance=CURRENT_MEASURED,
    )


def authorize_frame(
    *,
    previous_receipt: IntervalReceipt | None,
    current_measured_receipt: IntervalReceipt | None,
    x94_arm: Mapping[str, Any],
) -> tuple[bool, IntervalReceipt | None, str]:
    """Apply strict measured birth and narrow X94 persistence semantics."""

    if current_measured_receipt is not None:
        if current_measured_receipt.provenance != CURRENT_MEASURED:
            raise ValueError("current receipt must have CURRENT_MEASURED provenance")
        plan_receipt = str(x94_arm.get("plan_receipt_sha256") or "")
        if plan_receipt != current_measured_receipt.plan_receipt_sha256:
            return False, None, "PLAN_RECEIPT_CHANGED"
        return True, current_measured_receipt, CURRENT_MEASURED

    used = bool(x94_arm.get("x94_one_frame_full_dropout_continuity_used"))
    if not used:
        return False, None, "NO_CURRENT_INTERVAL"
    if previous_receipt is None:
        return False, None, "TRANSPORT_CANNOT_BIRTH"
    if previous_receipt.provenance != CURRENT_MEASURED:
        return False, None, "TRANSPORT_CANNOT_RESEED"
    plan_receipt = str(x94_arm.get("plan_receipt_sha256") or "")
    parent_ids = {str(value) for value in x94_arm.get("x94_continuity_parent_ids", [])}
    if plan_receipt != previous_receipt.plan_receipt_sha256:
        return False, None, "PLAN_RECEIPT_CHANGED"
    if previous_receipt.parent_id not in parent_ids:
        return False, None, "PARENT_WITNESS_MISMATCH"
    renewed = IntervalReceipt(
        plan_receipt_sha256=previous_receipt.plan_receipt_sha256,
        parent_id=previous_receipt.parent_id,
        carrier_id=previous_receipt.carrier_id,
        representation=previous_receipt.representation,
        interval_set=previous_receipt.interval_set,
        authorized_component_index=previous_receipt.authorized_component_index,
        provenance=TRANSPORT_RENEWAL,
    )
    return True, renewed, TRANSPORT_RENEWAL


def self_check() -> dict[str, Any]:
    contract = fixed_contract()
    return {
        "contract": contract,
        "transport_cannot_reseed": not contract["transport_reseed_allowed"],
        "x94_hold_window_unchanged": not contract["x94_hold_window_changed"],
    }
