"""Pure metric route geometry for the CARLA X23/X24 predictor arms.

The module intentionally performs no file access and imports no evaluator
code.  All positions and velocities are two-dimensional values in one shared
metric frame.  Plan waypoint times and ``now_s`` must use the same clock.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "dtr-c1-plan-receipt-v1"
PLAN_COORDINATE_FRAME = "ANCHOR_FORWARD_RIGHT"
C2_PLAN_SCHEMA = "dtr-c2-plan-receipt-v1"
C2_PLAN_COORDINATE_FRAME = "LAYOUT_FORWARD_RIGHT"
SUPPORTED_PLAN_CONTRACTS = {
    PLAN_SCHEMA: PLAN_COORDINATE_FRAME,
    C2_PLAN_SCHEMA: C2_PLAN_COORDINATE_FRAME,
}

AUTHORITY_NO_PLAN = "NO_PLAN"
AUTHORITY_INVALID_RECEIPT = "INVALID_RECEIPT"
AUTHORITY_WRONG_SESSION = "WRONG_SESSION"
AUTHORITY_FUTURE_ISSUED = "FUTURE_ISSUED"
AUTHORITY_EXPIRED = "EXPIRED"
AUTHORITY_VALID = "VALID"

ROUTE_MODE_ISSUED_PLAN = "ISSUED_PLAN"
ROUTE_MODE_OBSERVED_CV = "OBSERVED_CV_FALLBACK"

DEFAULT_PLAN_POSITION_RESIDUAL_M = 0.45
DEFAULT_PLAN_VELOCITY_DIRECTION_ERROR_DEG = 25.0
DEFAULT_ROUTE_HORIZON_S = 3.0
DEFAULT_TUBE_RADIUS_M = 0.65
DEFAULT_MIN_CLOSING_SPEED_MPS = 0.05

_EPSILON = 1e-9

Vec2 = tuple[float, float]


@dataclass(frozen=True)
class Waypoint:
    time_s: float
    position_xy: Vec2


@dataclass(frozen=True)
class RouteSegment:
    """One constant-velocity wearer segment, indexed from the current time."""

    start_offset_s: float
    end_offset_s: float
    start_position_xy: Vec2
    velocity_xy: Vec2


@dataclass(frozen=True)
class RouteSelection:
    """Causal route decision and diagnostics for one prediction timestamp."""

    mode: str
    authority: str
    receipt_valid: bool
    receipt_sha256: str | None
    plan_position_residual_m: float | None
    plan_velocity_direction_error_deg: float | None
    fallback_reason: str | None
    mode_changed: bool

    @property
    def plan_admitted(self) -> bool:
        return self.mode == ROUTE_MODE_ISSUED_PLAN


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _vec2(value: Sequence[float], label: str) -> Vec2:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two numbers")
    return (_finite_float(value[0], f"{label}[0]"), _finite_float(value[1], f"{label}[1]"))


def canonical_json_bytes(value: Any) -> bytes:
    """Match the canonical receipt hashing used by the C1 plan builder."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def receipt_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_sha256"}


def compute_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(receipt_payload(receipt))).hexdigest().upper()


def _valid_from_s(receipt: Mapping[str, Any]) -> float:
    """Normalize C1's explicit start and C2's issue-time start."""

    raw = receipt.get("valid_from_s", receipt.get("issued_at_s"))
    return _finite_float(raw, "valid_from_s")


def validate_plan_receipt(receipt: Mapping[str, Any]) -> tuple[Waypoint, ...]:
    """Validate identity, structure, time ordering, and the canonical hash."""

    if not isinstance(receipt, Mapping):
        raise ValueError("plan receipt must be an object")
    schema = receipt.get("schema_version")
    if schema not in SUPPORTED_PLAN_CONTRACTS:
        raise ValueError("unexpected plan receipt schema")
    if receipt.get("coordinate_frame") != SUPPORTED_PLAN_CONTRACTS[schema]:
        raise ValueError("unsupported plan coordinate frame")
    for key in ("plan_id", "session_id"):
        if not isinstance(receipt.get(key), str) or not str(receipt[key]).strip():
            raise ValueError(f"plan receipt {key} must be non-empty")

    issued_at_s = _finite_float(receipt.get("issued_at_s"), "issued_at_s")
    valid_from_s = _valid_from_s(receipt)
    expires_at_s = _finite_float(receipt.get("expires_at_s"), "expires_at_s")
    if issued_at_s > valid_from_s + _EPSILON or valid_from_s > expires_at_s + _EPSILON:
        raise ValueError("plan receipt authority interval is not ordered")

    raw_waypoints = receipt.get("time_parameterized_waypoints")
    if not isinstance(raw_waypoints, Sequence) or isinstance(raw_waypoints, (str, bytes)):
        raise ValueError("time_parameterized_waypoints must be a sequence")
    if len(raw_waypoints) < 2:
        raise ValueError("a plan requires at least two waypoints")
    waypoints: list[Waypoint] = []
    for index, raw in enumerate(raw_waypoints):
        if not isinstance(raw, Mapping):
            raise ValueError(f"waypoint {index} must be an object")
        waypoint = Waypoint(
            time_s=_finite_float(raw.get("time_s"), f"waypoint[{index}].time_s"),
            position_xy=(
                _finite_float(raw.get("forward_m"), f"waypoint[{index}].forward_m"),
                _finite_float(raw.get("right_m"), f"waypoint[{index}].right_m"),
            ),
        )
        if waypoints and waypoint.time_s <= waypoints[-1].time_s + _EPSILON:
            raise ValueError("plan waypoint times must be strictly increasing")
        waypoints.append(waypoint)

    supplied_hash = receipt.get("receipt_sha256")
    if not isinstance(supplied_hash, str) or not hmac.compare_digest(
        supplied_hash.upper(), compute_receipt_sha256(receipt)
    ):
        raise ValueError("plan receipt hash mismatch")
    return tuple(waypoints)


def plan_authority(
    receipt: Mapping[str, Any] | None,
    *,
    session_id: str,
    now_s: float,
) -> str:
    """Return the causal authority state without ever consulting execution truth."""

    now = _finite_float(now_s, "now_s")
    if receipt is None:
        return AUTHORITY_NO_PLAN
    try:
        validate_plan_receipt(receipt)
    except ValueError:
        return AUTHORITY_INVALID_RECEIPT
    if str(receipt["session_id"]) != str(session_id):
        return AUTHORITY_WRONG_SESSION
    if now + _EPSILON < _valid_from_s(receipt):
        return AUTHORITY_FUTURE_ISSUED
    if now > float(receipt["expires_at_s"]) + _EPSILON:
        return AUTHORITY_EXPIRED
    return AUTHORITY_VALID


def _interpolate_waypoints(waypoints: Sequence[Waypoint], time_s: float) -> Vec2 | None:
    time_value = _finite_float(time_s, "time_s")
    if time_value < waypoints[0].time_s - _EPSILON or time_value > waypoints[-1].time_s + _EPSILON:
        return None
    if time_value <= waypoints[0].time_s + _EPSILON:
        return waypoints[0].position_xy
    if time_value >= waypoints[-1].time_s - _EPSILON:
        return waypoints[-1].position_xy
    for first, second in zip(waypoints, waypoints[1:]):
        if first.time_s - _EPSILON <= time_value <= second.time_s + _EPSILON:
            ratio = (time_value - first.time_s) / (second.time_s - first.time_s)
            return (
                first.position_xy[0] + ratio * (second.position_xy[0] - first.position_xy[0]),
                first.position_xy[1] + ratio * (second.position_xy[1] - first.position_xy[1]),
            )
    return None


def interpolate_plan_position(receipt: Mapping[str, Any], time_s: float) -> Vec2 | None:
    return _interpolate_waypoints(validate_plan_receipt(receipt), time_s)


def _waypoint_velocity(waypoints: Sequence[Waypoint], time_s: float) -> Vec2 | None:
    time_value = _finite_float(time_s, "time_s")
    if time_value < waypoints[0].time_s - _EPSILON or time_value > waypoints[-1].time_s + _EPSILON:
        return None
    for index, (first, second) in enumerate(zip(waypoints, waypoints[1:])):
        is_last = index == len(waypoints) - 2
        if time_value < second.time_s - _EPSILON or (is_last and time_value <= second.time_s + _EPSILON):
            duration = second.time_s - first.time_s
            return (
                (second.position_xy[0] - first.position_xy[0]) / duration,
                (second.position_xy[1] - first.position_xy[1]) / duration,
            )
    return None


def plan_velocity_at(receipt: Mapping[str, Any], time_s: float) -> Vec2 | None:
    return _waypoint_velocity(validate_plan_receipt(receipt), time_s)


def observed_cv_position(
    current_position_xy: Sequence[float],
    current_velocity_xy: Sequence[float],
    *,
    now_s: float,
    query_time_s: float,
) -> Vec2:
    position = _vec2(current_position_xy, "current_position_xy")
    velocity = _vec2(current_velocity_xy, "current_velocity_xy")
    delta_s = _finite_float(query_time_s, "query_time_s") - _finite_float(now_s, "now_s")
    if delta_s < -_EPSILON:
        raise ValueError("observed-CV route cannot query the past")
    delta_s = max(0.0, delta_s)
    return (position[0] + velocity[0] * delta_s, position[1] + velocity[1] * delta_s)


def velocity_direction_error_degrees(first_xy: Sequence[float], second_xy: Sequence[float]) -> float:
    first = _vec2(first_xy, "first_velocity_xy")
    second = _vec2(second_xy, "second_velocity_xy")
    first_speed = math.hypot(*first)
    second_speed = math.hypot(*second)
    if first_speed <= _EPSILON and second_speed <= _EPSILON:
        return 0.0
    if first_speed <= _EPSILON or second_speed <= _EPSILON:
        return 180.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_speed * second_speed)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def route_mode_changed(previous_mode: str | None, current_mode: str) -> bool:
    if current_mode not in {ROUTE_MODE_ISSUED_PLAN, ROUTE_MODE_OBSERVED_CV}:
        raise ValueError(f"unknown current route mode: {current_mode}")
    if previous_mode is None:
        return False
    if previous_mode not in {ROUTE_MODE_ISSUED_PLAN, ROUTE_MODE_OBSERVED_CV}:
        raise ValueError(f"unknown previous route mode: {previous_mode}")
    return previous_mode != current_mode


def select_route(
    receipt: Mapping[str, Any] | None,
    *,
    session_id: str,
    now_s: float,
    wearer_position_xy: Sequence[float],
    wearer_velocity_xy: Sequence[float],
    previous_mode: str | None = None,
    maximum_plan_position_residual_m: float = DEFAULT_PLAN_POSITION_RESIDUAL_M,
    maximum_plan_velocity_direction_error_deg: float = DEFAULT_PLAN_VELOCITY_DIRECTION_ERROR_DEG,
) -> RouteSelection:
    """Admit the issued plan only when authority and current adherence agree."""

    now = _finite_float(now_s, "now_s")
    wearer_position = _vec2(wearer_position_xy, "wearer_position_xy")
    wearer_velocity = _vec2(wearer_velocity_xy, "wearer_velocity_xy")
    maximum_residual = _nonnegative_float(
        maximum_plan_position_residual_m, "maximum_plan_position_residual_m"
    )
    maximum_direction_error = _nonnegative_float(
        maximum_plan_velocity_direction_error_deg,
        "maximum_plan_velocity_direction_error_deg",
    )
    if maximum_direction_error > 180.0:
        raise ValueError("maximum_plan_velocity_direction_error_deg must be <= 180")

    authority = plan_authority(receipt, session_id=session_id, now_s=now)
    receipt_valid = authority != AUTHORITY_INVALID_RECEIPT and receipt is not None
    receipt_hash = str(receipt.get("receipt_sha256")) if receipt is not None else None
    residual: float | None = None
    direction_error: float | None = None
    mode = ROUTE_MODE_OBSERVED_CV
    fallback_reason: str | None = authority

    if authority == AUTHORITY_VALID and receipt is not None:
        waypoints = validate_plan_receipt(receipt)
        plan_position = _interpolate_waypoints(waypoints, now)
        plan_velocity = _waypoint_velocity(waypoints, now)
        if plan_position is None or plan_velocity is None:
            fallback_reason = "PLAN_TIME_OUT_OF_RANGE"
        else:
            residual = math.dist(wearer_position, plan_position)
            direction_error = velocity_direction_error_degrees(wearer_velocity, plan_velocity)
            residual_ok = residual <= maximum_residual + _EPSILON
            direction_ok = direction_error <= maximum_direction_error + _EPSILON
            if residual_ok and direction_ok:
                mode = ROUTE_MODE_ISSUED_PLAN
                fallback_reason = None
            elif not residual_ok and not direction_ok:
                fallback_reason = "POSITION_AND_VELOCITY_ADHERENCE_FAILED"
            elif not residual_ok:
                fallback_reason = "POSITION_ADHERENCE_FAILED"
            else:
                fallback_reason = "VELOCITY_ADHERENCE_FAILED"

    return RouteSelection(
        mode=mode,
        authority=authority,
        receipt_valid=receipt_valid,
        receipt_sha256=receipt_hash,
        plan_position_residual_m=residual,
        plan_velocity_direction_error_deg=direction_error,
        fallback_reason=fallback_reason,
        mode_changed=route_mode_changed(previous_mode, mode),
    )


def _plan_route_segments(
    receipt: Mapping[str, Any],
    *,
    now_s: float,
    horizon_s: float,
) -> tuple[RouteSegment, ...]:
    waypoints = validate_plan_receipt(receipt)
    now = _finite_float(now_s, "now_s")
    horizon = _nonnegative_float(horizon_s, "horizon_s")
    end_time = min(now + horizon, float(receipt["expires_at_s"]), waypoints[-1].time_s)
    output: list[RouteSegment] = []
    for first, second in zip(waypoints, waypoints[1:]):
        overlap_start = max(now, first.time_s)
        overlap_end = min(end_time, second.time_s)
        if overlap_end <= overlap_start + _EPSILON:
            continue
        duration = second.time_s - first.time_s
        velocity = (
            (second.position_xy[0] - first.position_xy[0]) / duration,
            (second.position_xy[1] - first.position_xy[1]) / duration,
        )
        elapsed = overlap_start - first.time_s
        start_position = (
            first.position_xy[0] + velocity[0] * elapsed,
            first.position_xy[1] + velocity[1] * elapsed,
        )
        output.append(
            RouteSegment(
                start_offset_s=overlap_start - now,
                end_offset_s=overlap_end - now,
                start_position_xy=start_position,
                velocity_xy=velocity,
            )
        )
    if not output and end_time >= now - _EPSILON:
        position = _interpolate_waypoints(waypoints, now)
        velocity = _waypoint_velocity(waypoints, now)
        if position is not None and velocity is not None:
            output.append(RouteSegment(0.0, 0.0, position, velocity))
    return tuple(output)


def build_route_segments(
    selection: RouteSelection,
    *,
    receipt: Mapping[str, Any] | None,
    now_s: float,
    wearer_position_xy: Sequence[float],
    wearer_velocity_xy: Sequence[float],
    horizon_s: float = DEFAULT_ROUTE_HORIZON_S,
) -> tuple[RouteSegment, ...]:
    """Build the selected time-aligned wearer trajectory for collision math."""

    horizon = _nonnegative_float(horizon_s, "horizon_s")
    if selection.mode == ROUTE_MODE_ISSUED_PLAN:
        if receipt is None or not selection.plan_admitted or not selection.receipt_valid:
            raise ValueError("issued-plan route requires an admitted receipt")
        if str(receipt.get("receipt_sha256")) != selection.receipt_sha256:
            raise ValueError("route selection and plan receipt identities differ")
        return _plan_route_segments(receipt, now_s=now_s, horizon_s=horizon)
    if selection.mode != ROUTE_MODE_OBSERVED_CV:
        raise ValueError(f"unknown route mode: {selection.mode}")
    return (
        RouteSegment(
            start_offset_s=0.0,
            end_offset_s=horizon,
            start_position_xy=_vec2(wearer_position_xy, "wearer_position_xy"),
            velocity_xy=_vec2(wearer_velocity_xy, "wearer_velocity_xy"),
        ),
    )


def _first_circle_entry_s(
    relative_position_xy: Vec2,
    relative_velocity_xy: Vec2,
    *,
    duration_s: float,
    radius_m: float,
    minimum_closing_speed_mps: float,
) -> float | None:
    x, y = relative_position_xy
    vx, vy = relative_velocity_xy
    duration = _nonnegative_float(duration_s, "duration_s")
    radius = _positive_float(radius_m, "radius_m")
    minimum_closing = _nonnegative_float(minimum_closing_speed_mps, "minimum_closing_speed_mps")
    distance = math.hypot(x, y)
    speed_squared = vx * vx + vy * vy
    if distance <= radius + _EPSILON:
        closing = math.hypot(vx, vy) if distance <= _EPSILON else -(x * vx + y * vy) / distance
        return 0.0 if closing + _EPSILON >= minimum_closing else None
    if speed_squared <= _EPSILON:
        return None
    b = 2.0 * (x * vx + y * vy)
    c = x * x + y * y - radius * radius
    discriminant = b * b - 4.0 * speed_squared * c
    if discriminant < -_EPSILON:
        return None
    square_root = math.sqrt(max(0.0, discriminant))
    roots = sorted(((-b - square_root) / (2.0 * speed_squared), (-b + square_root) / (2.0 * speed_squared)))
    for root in roots:
        if root < -_EPSILON or root > duration + _EPSILON:
            continue
        entry_s = min(duration, max(0.0, root))
        entry_x, entry_y = x + vx * entry_s, y + vy * entry_s
        entry_distance = math.hypot(entry_x, entry_y)
        closing = -(entry_x * vx + entry_y * vy) / max(_EPSILON, entry_distance)
        if closing + _EPSILON >= minimum_closing:
            return entry_s
    return None


def first_metric_tube_entry_s(
    target_position_xy: Sequence[float],
    target_velocity_xy: Sequence[float],
    route_segments: Sequence[RouteSegment],
    *,
    tube_radius_m: float = DEFAULT_TUBE_RADIUS_M,
    minimum_closing_speed_mps: float = DEFAULT_MIN_CLOSING_SPEED_MPS,
) -> float | None:
    """Return the first target/wearer tube entry across time-aligned segments."""

    target_position = _vec2(target_position_xy, "target_position_xy")
    target_velocity = _vec2(target_velocity_xy, "target_velocity_xy")
    previous_end = -math.inf
    for index, segment in enumerate(route_segments):
        start = _nonnegative_float(segment.start_offset_s, f"route_segments[{index}].start_offset_s")
        end = _nonnegative_float(segment.end_offset_s, f"route_segments[{index}].end_offset_s")
        if end + _EPSILON < start:
            raise ValueError("route segment ends before it starts")
        if start + _EPSILON < previous_end:
            raise ValueError("route segments must be time ordered and non-overlapping")
        previous_end = end
        wearer_start = _vec2(segment.start_position_xy, f"route_segments[{index}].start_position_xy")
        wearer_velocity = _vec2(segment.velocity_xy, f"route_segments[{index}].velocity_xy")
        target_start = (
            target_position[0] + target_velocity[0] * start,
            target_position[1] + target_velocity[1] * start,
        )
        local_entry = _first_circle_entry_s(
            (target_start[0] - wearer_start[0], target_start[1] - wearer_start[1]),
            (target_velocity[0] - wearer_velocity[0], target_velocity[1] - wearer_velocity[1]),
            duration_s=max(0.0, end - start),
            radius_m=tube_radius_m,
            minimum_closing_speed_mps=minimum_closing_speed_mps,
        )
        if local_entry is not None:
            return start + local_entry
    return None


def first_selected_route_entry_s(
    selection: RouteSelection,
    *,
    receipt: Mapping[str, Any] | None,
    now_s: float,
    wearer_position_xy: Sequence[float],
    wearer_velocity_xy: Sequence[float],
    target_position_xy: Sequence[float],
    target_velocity_xy: Sequence[float],
    horizon_s: float = DEFAULT_ROUTE_HORIZON_S,
    tube_radius_m: float = DEFAULT_TUBE_RADIUS_M,
    minimum_closing_speed_mps: float = DEFAULT_MIN_CLOSING_SPEED_MPS,
) -> float | None:
    segments = build_route_segments(
        selection,
        receipt=receipt,
        now_s=now_s,
        wearer_position_xy=wearer_position_xy,
        wearer_velocity_xy=wearer_velocity_xy,
        horizon_s=horizon_s,
    )
    return first_metric_tube_entry_s(
        target_position_xy,
        target_velocity_xy,
        segments,
        tube_radius_m=tube_radius_m,
        minimum_closing_speed_mps=minimum_closing_speed_mps,
    )


__all__ = [
    "AUTHORITY_EXPIRED",
    "AUTHORITY_FUTURE_ISSUED",
    "AUTHORITY_INVALID_RECEIPT",
    "AUTHORITY_NO_PLAN",
    "AUTHORITY_VALID",
    "AUTHORITY_WRONG_SESSION",
    "DEFAULT_MIN_CLOSING_SPEED_MPS",
    "DEFAULT_PLAN_POSITION_RESIDUAL_M",
    "DEFAULT_PLAN_VELOCITY_DIRECTION_ERROR_DEG",
    "DEFAULT_ROUTE_HORIZON_S",
    "DEFAULT_TUBE_RADIUS_M",
    "PLAN_COORDINATE_FRAME",
    "PLAN_SCHEMA",
    "ROUTE_MODE_ISSUED_PLAN",
    "ROUTE_MODE_OBSERVED_CV",
    "RouteSegment",
    "RouteSelection",
    "Waypoint",
    "build_route_segments",
    "canonical_json_bytes",
    "compute_receipt_sha256",
    "first_metric_tube_entry_s",
    "first_selected_route_entry_s",
    "interpolate_plan_position",
    "observed_cv_position",
    "plan_authority",
    "plan_velocity_at",
    "receipt_payload",
    "route_mode_changed",
    "select_route",
    "validate_plan_receipt",
    "velocity_direction_error_degrees",
]
