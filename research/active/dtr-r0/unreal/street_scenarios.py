"""Deterministic street Development scenarios and evaluator-only swept truth.

All coordinates are meters relative to the route anchor, +X forward, +Y right.
This stdlib module runs both in UE Python and the external evaluator. It must
never be imported by the sensor/model worker. Shapes are declared experimental
collision proxies, not human injury, gait, or certified traversability truth.
The caller supplies measured scene surface height and renders these same bounds.
"""
from __future__ import annotations

import copy
import math

SCHEMA = "street-scripted-proxy-v1"
TRAVERSABLE_RELIEF_M = 0.006
FOOT_ZONE_TOP_M = 0.35
EGO_RADIUS_M = 0.28
FOOT_RADIUS_M = 0.24
EGO_HEIGHT_M = 1.75
ARMS = ("OPEN_LOOP", "ASSISTED")


def _actor(actor_id, kind, x, y, height, radius=None, half_extents=None, waypoints=None):
    actor = {"id": actor_id, "kind": kind, "shape": "disc" if radius else "box",
             "x_m": x, "y_m": y, "base_m": 0.0, "height_m": height,
             "yaw_deg": 0.0, "waypoints": waypoints or [[0.0, x, y]],
             "truth_authority": "DECLARED_SYNTHETIC_GEOMETRY_PROXY"}
    if radius:
        actor["radius_m"] = radius
    else:
        actor["half_extents_m"] = list(half_extents)
    return actor


def scenario_catalog():
    """Return fresh mutable specs. Expected controls belong only to evaluator."""
    result = []
    for family in ("occluded_crossing", "sudden_stop", "narrow_passing", "low_obstacle"):
        for contact in (True, False):
            control = "collision" if contact else "near_miss"
            actors = []
            if family == "occluded_crossing":
                # Same crossing speed, 2 s phase shift for the near-miss control.
                points = ([[0, 4, -4], [8, 4, 4]] if contact else
                          [[0, 4, -6], [8, 4, 2]])
                actors.append(_actor("pedestrian", "pedestrian", 4, points[0][2], 1.75,
                                     radius=0.28, waypoints=points))
                actors.append(_actor("occluder", "occluder", 2.4, -1.65, 2.0,
                                     half_extents=(0.7, 0.5)))
            elif family == "sudden_stop":
                points = ([[0, 2.4, 0], [2, 4.4, 0], [8, 4.4, 0]] if contact else
                          [[0, 2.4, 0], [8, 10.4, 0]])
                actors.append(_actor("pedestrian", "pedestrian", 2.4, 0, 1.75,
                                     radius=0.28, waypoints=points))
            elif family == "narrow_passing":
                lateral = 0.42 if contact else 0.85
                actors.append(_actor("pedestrian", "pedestrian", 7, lateral, 1.75,
                                     radius=0.28, waypoints=[[0, 7, lateral], [8, -1, lateral]]))
                actors.extend([
                    _actor("passage_left", "barrier", 4, -1.0, 1.1, half_extents=(2.0, 0.15)),
                    _actor("passage_right", "barrier", 4, 1.45, 1.1, half_extents=(2.0, 0.15)),
                ])
            else:
                actors.append(_actor("low_obstacle", "low_obstacle", 4, 0 if contact else 1.0,
                                     0.12, half_extents=(0.22, 0.35)))
            # Common negative geometry control, only 4 mm above measured ground.
            actors.append(_actor("tactile_ground", "tactile_ground", 4, 0, 0.004,
                                 half_extents=(4, 0.2)))
            result.append({"schema": SCHEMA, "id": family + "_" + control,
                           "family": family, "control": control, "duration_s": 8.0,
                           "dt_s": 0.2, "ego_speed_mps": 1.0,
                           "ego_start": {"x_m": 0.0, "y_m": 0.0},
                           "arms": list(ARMS), "actors": actors,
                           "expected_open_loop_contact": contact,
                           "expected_contact_type": ("FOOT_TRIP_PROXY" if family == "low_obstacle"
                                                     else "BODY_COLLISION_PROXY") if contact else None})
    return result


def actors_at(scenario, time_s, surface_height_m=0.0):
    """Piecewise-linear scripted actor poses, with base at measured surface."""
    result = []
    for source in scenario["actors"]:
        actor = copy.deepcopy(source)
        points = source["waypoints"]
        x, y = points[-1][1:3]
        if time_s <= points[0][0]:
            x, y = points[0][1:3]
        else:
            for a, b in zip(points, points[1:]):
                if time_s <= b[0]:
                    u = (time_s - a[0]) / (b[0] - a[0])
                    x, y = a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2])
                    break
        actor.update(x_m=x, y_m=y, base_m=surface_height_m + source["base_m"])
        result.append(actor)
    return result


def _linear_nonnegative(value, slope, low, high):
    if abs(slope) < 1e-14:
        return (low, high) if value >= -1e-12 else None
    crossing = -value / slope
    if slope > 0:
        low = max(low, crossing)
    else:
        high = min(high, crossing)
    return (low, high) if low <= high + 1e-12 else None


def _rect_hit(p, v, hx, hy, low, high):
    for pos, speed, half in zip(p, v, (hx, hy)):
        interval = _linear_nonnegative(half - pos, -speed, low, high)
        if interval is None:
            return None
        interval = _linear_nonnegative(half + pos, speed, *interval)
        if interval is None:
            return None
        low, high = interval
    return low


def _disc_hit(p, v, radius, low, high):
    a = v[0] ** 2 + v[1] ** 2
    b = 2 * (p[0] * v[0] + p[1] * v[1])
    c = p[0] ** 2 + p[1] ** 2 - radius ** 2
    if a < 1e-20:
        return low if c <= 1e-12 else None
    discriminant = b * b - 4 * a * c
    if discriminant < -1e-12:
        return None
    root = math.sqrt(max(0.0, discriminant))
    entry, leave = (-b - root) / (2 * a), (-b + root) / (2 * a)
    start, end = max(low, entry), min(high, leave)
    return start if start <= end + 1e-12 else None


def _horizontal_hit(ego0, ego1, actor0, actor1, radius, low, high):
    p = (ego0["x_m"] - actor0["x_m"], ego0["y_m"] - actor0["y_m"])
    q = (ego1["x_m"] - actor1["x_m"], ego1["y_m"] - actor1["y_m"])
    v = (q[0] - p[0], q[1] - p[1])
    if actor0["shape"] == "disc":
        if actor0["radius_m"] != actor1["radius_m"]:
            raise ValueError("Sweeps require fixed proxy radius")
        return _disc_hit(p, v, radius + actor0["radius_m"], low, high)
    if actor0["shape"] != "box":
        raise ValueError("Unsupported truth shape")
    if (actor0.get("yaw_deg", 0) != actor1.get("yaw_deg", 0) or
            actor0["half_extents_m"] != actor1["half_extents_m"]):
        raise ValueError("Box sweeps require fixed orientation and extents")
    angle = math.radians(actor0.get("yaw_deg", 0))
    co, si = math.cos(angle), math.sin(angle)
    p = (co * p[0] + si * p[1], -si * p[0] + co * p[1])
    v = (co * v[0] + si * v[1], -si * v[0] + co * v[1])
    hx, hy = actor0["half_extents_m"]
    # Exact rectangle Minkowski sum with the disc: faces plus round corners.
    hits = [_rect_hit(p, v, hx + radius, hy, low, high),
            _rect_hit(p, v, hx, hy + radius, low, high)]
    for x in (-hx, hx):
        for y in (-hy, hy):
            hits.append(_disc_hit((p[0] - x, p[1] - y), v, radius, low, high))
    hits = [hit for hit in hits if hit is not None]
    return min(hits) if hits else None


def contact_between(ego0, ego1, actor0, actor1, surface_height_m=0.0):
    """First continuous contact during one linear-motion step, or None.

    Bases/heights are world meters. Ego base defaults to measured scene surface.
    Call at all script breakpoints: endpoint interpolation cannot represent a
    within-step change of velocity. Fixed vertical interval radii approximate
    torso and feet separately; this does not simulate individual footfalls.
    """
    if actor0["id"] != actor1["id"] or actor0["shape"] != actor1["shape"]:
        raise ValueError("Sweep endpoints must describe the same proxy")
    if actor0["height_m"] != actor1["height_m"]:
        raise ValueError("Sweeps require fixed proxy height")
    if ego0.get("radius_m", EGO_RADIUS_M) != ego1.get("radius_m", EGO_RADIUS_M):
        raise ValueError("Sweeps require fixed wearer radius")
    ebase0 = ego0.get("base_m", surface_height_m)
    ebase1 = ego1.get("base_m", surface_height_m)
    abase0, abase1 = actor0.get("base_m", surface_height_m), actor1.get("base_m", surface_height_m)
    # Relief touching the threshold is traversable by declaration, not a trip.
    top0, top1 = abase0 + actor0["height_m"], abase1 + actor1["height_m"]
    if max(top0 - ebase0, top1 - ebase1) <= TRAVERSABLE_RELIEF_M + 1e-12:
        return None
    contacts = []
    for label, bottom, top, radius in (
            ("BODY_COLLISION_PROXY", FOOT_ZONE_TOP_M, ego0.get("height_m", EGO_HEIGHT_M),
             ego0.get("radius_m", EGO_RADIUS_M)),
            ("FOOT_TRIP_PROXY", TRAVERSABLE_RELIEF_M, FOOT_ZONE_TOP_M,
             ego0.get("foot_radius_m", FOOT_RADIUS_M))):
        interval = _linear_nonnegative(top0 - ebase0 - bottom,
                                       (top1 - ebase1) - (top0 - ebase0), 0.0, 1.0)
        if interval is None:
            continue
        interval = _linear_nonnegative(ebase0 + top - abase0,
                                       (ebase1 - abase1) - (ebase0 - abase0), *interval)
        if interval is None:
            continue
        fraction = _horizontal_hit(ego0, ego1, actor0, actor1, radius, *interval)
        if fraction is not None:
            contacts.append({"actor_id": actor0["id"], "contact_type": label,
                             "first_fraction": max(0.0, min(1.0, fraction)),
                             "authority": "CONTINUOUS_SWEPT_DECLARED_PROXY",
                             "wearer_radius_m": radius})
    # At simultaneous contacts the body label is preferred by declared order.
    return min(contacts, key=lambda c: c["first_fraction"]) if contacts else None


def contacts_for_step(scenario, time0, time1, ego0, ego1, surface_height_m=0.0):
    """All per-actor first contacts, splitting script turns exactly."""
    if time1 <= time0:
        raise ValueError("Contact step must have positive duration")
    times = sorted({time0, time1} | {p[0] for a in scenario["actors"] for p in a["waypoints"]
                                   if time0 < p[0] < time1})
    found = {}
    for start, end in zip(times, times[1:]):
        poses = []
        for t in (start, end):
            u = (t - time0) / (time1 - time0)
            pose = dict(ego0)
            for key in ("x_m", "y_m", "base_m"):
                default = surface_height_m if key == "base_m" else 0.0
                pose[key] = ego0.get(key, default) + u * (ego1.get(key, default) - ego0.get(key, default))
            poses.append(pose)
        for a, b in zip(actors_at(scenario, start, surface_height_m),
                        actors_at(scenario, end, surface_height_m)):
            contact = contact_between(*poses, a, b, surface_height_m)
            if contact and a["id"] not in found:
                contact["time_s"] = start + contact["first_fraction"] * (end - start)
                contact["first_fraction"] = (contact["time_s"] - time0) / (time1 - time0)
                found[a["id"]] = contact
    return sorted(found.values(), key=lambda c: c["time_s"])


def validate_catalog():
    """One deterministic geometry contrast; no algorithm/safety performance claim."""
    rows = []
    for spec in scenario_catalog():
        duration = spec["duration_s"]
        ego0 = dict(spec["ego_start"])
        ego1 = dict(ego0, x_m=ego0["x_m"] + duration * spec["ego_speed_mps"])
        contacts = contacts_for_step(spec, 0.0, duration, ego0, ego1)
        kinds = {c["contact_type"] for c in contacts}
        passed = (bool(contacts) == spec["expected_open_loop_contact"] and
                  (not contacts or spec["expected_contact_type"] in kinds) and
                  all(c["actor_id"] != "tactile_ground" for c in contacts))
        rows.append({"scenario_id": spec["id"], "passed": passed, "contacts": contacts})
    return {"schema": SCHEMA, "passed": all(r["passed"] for r in rows), "scenarios": rows,
            "claim": "Synthetic declared-proxy contrast only; not algorithm or human safety evidence"}


if __name__ == "__main__":
    import json
    result = validate_catalog()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
