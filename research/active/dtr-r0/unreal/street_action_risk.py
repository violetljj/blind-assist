"""Compare unissued motion candidates against current sensor-derived footprints.

The state is shared across queries. This module never reads plans, actor truth,
scenario names or future observations, and never grants an issued-plan credential.
No predicted intersection is only a statement about the supplied footprints.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

EPS = 1e-9
HORIZON_S = 3.0
STEP_S = 0.2
# Physical corridor half-width used by the existing depth controller. This is
# deliberately an action footprint, not the retained X73 0.65 m alert tube.
BODY_RADIUS_M = 0.30
MAX_EVIDENCE_AGE_S = 0.60
SIDE_TARGETS_M = (-0.72, 0.72, -0.52, 0.52)


@dataclass(frozen=True)
class Candidate:
    name: str
    vx: float
    vy: float
    target_y: float
    wait: bool = False


def _clip(value, slope, low, high):
    """Restrict [low, high] to value + slope*t >= 0."""
    if abs(slope) <= EPS:
        return (low, high) if value >= -EPS else None
    crossing = -value / slope
    if slope > 0:
        low = max(low, crossing)
    else:
        high = min(high, crossing)
    return (low, high) if low <= high + EPS else None


def _halfspaces(constraints, duration):
    interval = (0.0, duration)
    for value, slope in constraints:
        interval = _clip(value, slope, *interval)
        if interval is None:
            return None
    return interval


def _disc_interval(p, velocity, center, radius, duration):
    dx, dy = p[0] - center[0], p[1] - center[1]
    a = velocity[0] ** 2 + velocity[1] ** 2
    b = 2 * (dx * velocity[0] + dy * velocity[1])
    c = dx * dx + dy * dy - radius * radius
    if a <= EPS * EPS:
        return (0.0, duration) if c <= EPS else None
    discriminant = b * b - 4 * a * c
    if discriminant < -EPS:
        return None
    root = math.sqrt(max(0.0, discriminant))
    low, high = max(0.0, (-b - root) / (2 * a)), min(duration, (-b + root) / (2 * a))
    return (low, high) if low <= high + EPS else None


def _merge(intervals):
    result = []
    for low, high in sorted(intervals):
        if result and low <= result[-1][1] + EPS:
            result[-1][1] = max(high, result[-1][1])
        else:
            result.append([low, high])
    return result


def _convex_polygon(value):
    polygon = tuple(tuple(float(v) for v in p) for p in value)
    if len(polygon) < 3 or any(len(p) != 2 or not all(map(math.isfinite, p)) for p in polygon):
        raise ValueError("A footprint must have finite 2D vertices")
    area2 = sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(polygon, polygon[1:] + polygon[:1]))
    if abs(area2) <= EPS:
        raise ValueError("Degenerate footprint")
    if area2 < 0:
        polygon = tuple(reversed(polygon))
    for a, b, c in zip(polygon, polygon[1:] + polygon[:1], polygon[2:] + polygon[:2]):
        cross = (b[0]-a[0]) * (c[1]-b[1]) - (b[1]-a[1]) * (c[0]-b[0])
        if cross < -EPS or math.dist(a, b) <= EPS:
            raise ValueError("Footprint must be a nondegenerate convex polygon")
    return polygon


def sweep_intervals(position, velocity, polygon, radius, duration):
    """Exact line sweep against a convex polygon dilated by a circular body.

The rounded dilation is the union of the polygon, edge strips and vertex discs.
It avoids both time-grid tunnelling and square-expanded corner false positives.
"""
    polygon = _convex_polygon(polygon)
    if radius < 0 or duration < 0 or not all(map(math.isfinite, (*position, *velocity, radius, duration))):
        raise ValueError("Invalid finite sweep")
    intervals, interior = [], []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        ex, ey = b[0] - a[0], b[1] - a[1]
        px, py = position[0] - a[0], position[1] - a[1]
        interior.append((ex * py - ey * px, ex * velocity[1] - ey * velocity[0]))
        length = math.hypot(ex, ey)
        along = (px * ex + py * ey) / length
        along_v = (velocity[0] * ex + velocity[1] * ey) / length
        normal = (ex * py - ey * px) / length
        normal_v = (ex * velocity[1] - ey * velocity[0]) / length
        strip = _halfspaces(((along, along_v), (length-along, -along_v),
                             (radius-normal, -normal_v), (radius+normal, normal_v)), duration)
        if strip is not None:
            intervals.append(strip)
        disc = _disc_interval(position, velocity, a, radius, duration)
        if disc is not None:
            intervals.append(disc)
    inside = _halfspaces(interior, duration)
    if inside is not None:
        intervals.append(inside)
    return _merge(intervals)


def rollout(candidate, x, y, horizon=HORIZON_S):
    """The first segment is the command actually sent; later ones re-center it."""
    segments = []
    elapsed = 0.0
    while elapsed < horizon - EPS:
        if candidate.wait:
            vx = vy = 0.0
        elif elapsed <= EPS:
            vx, vy = candidate.vx, candidate.vy
        else:
            error = candidate.target_y - y
            vy = max(-0.65, min(0.65, error * 2.0))
            vx = 0.35 if abs(error) > 0.09 else 1.0
        duration = min(STEP_S, horizon - elapsed)
        segments.append((elapsed, duration, (x, y), (vx, vy)))
        x, y = x + vx * duration, y + vy * duration
        elapsed += duration
    return segments


def candidates_for(nominal, *, x, y, corridors):
    candidates = [Candidate("DEPTH_NOMINAL", float(nominal["vx_mps"]), float(nominal["vy_mps"]),
                            float(nominal["target_y_m"]),
                            wait=nominal["vx_mps"] == 0 and nominal["vy_mps"] == 0)]
    # Depth's immediate brake, invalid depth and terminal stop retain precedence.
    if nominal["action"] in ("ARRIVED", "WAIT_UNKNOWN_DEPTH", "BRAKE_IMMINENT", "BRAKE"):
        return candidates
    candidates.append(Candidate("WAIT", 0.0, 0.0, y, wait=True))
    front = corridors["front_obstacle_m"]
    required = min(3.3, front + 1.0) if front is not None else 3.3
    for target in SIDE_TARGETS_M:
        if corridors["clearance_m"].get(str(target), 0.0) <= required:
            continue
        error = target - y
        if abs(error) <= 0.09:
            continue
        vx = 0.35 if front is None or front > 1.0 else 0.0
        candidate = Candidate(f"SIDE_{target}", vx, max(-0.65, min(0.65, error*2)), target)
        if any((c.vx, c.vy, c.target_y) == (candidate.vx, candidate.vy, candidate.target_y) for c in candidates):
            continue
        candidates.append(candidate)
    return candidates


def supported_tracks(frame, now_s):
    try:
        stamp = float(frame.get("time_s", math.nan)) if isinstance(frame, Mapping) else math.nan
    except (TypeError, ValueError, OverflowError):
        stamp = math.nan
    if not math.isfinite(stamp):
        return [], "UNKNOWN_NO_CURRENT_FOOTPRINT_FRAME", 0
    if abs(stamp - now_s) > EPS:
        return [], "UNKNOWN_FOOTPRINT_FRAME_TIME_MISMATCH", 0
    admitted, rejected = [], 0
    for item in frame.get("tracks", ()):
        try:
            age = float(item["evidence_age_s"])
            if item["disposition"] not in ("MEASURED", "HOLD") or not 0 <= age <= MAX_EVIDENCE_AGE_S + EPS:
                raise ValueError("Unsupported temporal evidence")
            polygon = _convex_polygon(item["footprint_xy"])
            velocity = (float(item["velocity_forward_mps"]), float(item["velocity_right_mps"]))
            if not all(map(math.isfinite, velocity)):
                raise ValueError("Nonfinite velocity")
            admitted.append({"track_id": str(item["track_id"]), "polygon": polygon, "velocity": velocity,
                             "disposition": item["disposition"], "evidence_age_s": age})
        except (KeyError, TypeError, ValueError, OverflowError):
            rejected += 1
    state = "SUPPORTED_LOCAL_FOOTPRINTS" if admitted else "UNKNOWN_NO_USABLE_FOOTPRINTS"
    return admitted, state, rejected


def evaluate_candidates(candidates, frame, *, t, x, y):
    tracks, support, rejected = supported_tracks(frame, t)
    rows = []
    for candidate in candidates:
        segments = rollout(candidate, x, y)
        hits = []
        for track in tracks:
            intervals = []
            tv = track["velocity"]
            for offset, duration, position, velocity in segments:
                relative_position = (position[0] - tv[0]*offset, position[1] - tv[1]*offset)
                relative_velocity = (velocity[0] - tv[0], velocity[1] - tv[1])
                intervals.extend((offset+a, offset+b) for a, b in sweep_intervals(
                    relative_position, relative_velocity, track["polygon"], BODY_RADIUS_M, duration))
            intervals = _merge(intervals)
            if intervals:
                hits.append({"track_id": track["track_id"], "intervals_s": intervals,
                             "disposition": track["disposition"], "evidence_age_s": track["evidence_age_s"]})
        progress = sum(duration*velocity[0] for _, duration, _, velocity in segments)
        rows.append({"candidate": candidate.name, "vx_mps": candidate.vx, "vy_mps": candidate.vy,
                     "target_y_m": candidate.target_y, "progress_m": progress,
                     "conflicts": hits, "first_entry_s": min((h["intervals_s"][0][0] for h in hits), default=None),
                     "support_state": support,
                     "state": "PREDICTED_INTERSECTION" if hits else
                         "NO_INTERSECTION_FOR_SUPPORTED_FOOTPRINTS" if tracks else "UNKNOWN"})
    return {"schema": "street-hypothetical-action-risk-v1", "authority": "UNISSUED_ACTION_HYPOTHESES",
            "global_observability": "UNKNOWN", "body_radius_m": BODY_RADIUS_M, "horizon_s": HORIZON_S,
            "source_time_s": t, "admitted_tracks": len(tracks), "rejected_tracks": rejected, "candidates": rows}


def choose_candidate(evaluation, *, enabled):
    rows = evaluation["candidates"]
    nominal = rows[0]
    if not enabled or not nominal["conflicts"]:
        return nominal, "DEPTH_NOMINAL_RETAINED"
    free = [r for r in rows if not r["conflicts"]]
    if free:
        selected = max(free, key=lambda r: (r["progress_m"], -abs(r["target_y_m"]), r["candidate"] == "DEPTH_NOMINAL"))
        return selected, "AVOID_SUPPORTED_NOMINAL_INTERSECTION"
    # With no conflict-free candidate, preserve depth's action unless another
    # delays the first supported intersection. This is not a safety guarantee.
    selected = max(rows, key=lambda r: (r["first_entry_s"], -r["progress_m"]))
    if selected["first_entry_s"] <= nominal["first_entry_s"] + EPS:
        return nominal, "NO_CANDIDATE_IMPROVES_SUPPORTED_ENTRY"
    return selected, "DELAY_SUPPORTED_ENTRY_ALL_CANDIDATES_CONFLICT"
