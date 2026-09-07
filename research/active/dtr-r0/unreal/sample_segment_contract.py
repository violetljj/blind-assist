"""Evaluator-only native-bound proxies for the Willow sample segment.

Inputs must enumerate every native static-mesh component and every instance with
a unique id and world-space bounds in metres. This module cannot prove that the
exporter supplied a complete roster. AABB contact is conservative proxy contact,
not triangle/surface contact or an Unreal physics result. Never send this roster
or evaluation output to the policy/model as observations.
"""
import math


def _vector(value, size, name):
    result = [float(v) for v in value]
    if len(result) != size or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def convert_components(rows, region=(22., 46., -6., 6.), floor_m=.12,
                       ground_upper_m=.145, wearer_height_m=1.8):
    """Return serializable clipped obstacles, full native roster and exclusions.

    region is (xmin, xmax, ymin, ymax). ground_upper_m is an absolute world-Z
    threshold, defaulting to 25 mm above the nominal floor. Collision-disabled
    visible geometry remains an obstacle. Instance bounds must already be
    transformed into world coordinates, rather than a component aggregate.
    """
    region = _vector(region, 4, "region")
    floor_m, ground_upper_m, wearer_height_m = _vector(
        [floor_m, ground_upper_m, wearer_height_m], 3, "height settings")
    if region[0] >= region[1] or region[2] >= region[3]:
        raise ValueError("region must have positive area")
    if wearer_height_m <= 0 or not floor_m <= ground_upper_m < floor_m + wearer_height_m:
        raise ValueError("invalid floor/ground/wearer height settings")
    obstacles, exclusions, native, ids = [], [], [], set()
    for row in rows:
        identity = str(row["id"])
        if not identity or identity in ids:
            raise ValueError("component/instance ids must be nonempty and unique")
        ids.add(identity)
        center = _vector(row["center_m"], 3, "center_m")
        extent = _vector(row["extent_m"], 3, "extent_m")
        if any(v < 0 for v in extent):
            raise ValueError("extent_m must be nonnegative")
        low = [c - e for c, e in zip(center, extent)]
        high = [c + e for c, e in zip(center, extent)]
        record = {"id": identity, "asset": str(row["asset"]),
                  "center_m": center, "extent_m": extent,
                  "min_m": low, "max_m": high,
                  "collision_enabled": row.get("collision_enabled")}
        native.append(record)
        reason = None
        if high[0] < region[0] or low[0] > region[1] or high[1] < region[2] or low[1] > region[3]:
            reason = "OUTSIDE_REGION"
        elif high[2] <= ground_upper_m:
            reason = "FLOOR_OR_BELOW_GROUND_TOLERANCE"
        elif low[2] >= floor_m + wearer_height_m:
            reason = "OVERHEAD_ABOVE_WEARER"
        if reason:
            exclusions.append({"id": identity, "reason": reason})
            continue
        clipped_low = [max(low[0], region[0]), max(low[1], region[2]), low[2]]
        clipped_high = [min(high[0], region[1]), min(high[1], region[3]), high[2]]
        obstacles.append({**record, "min_m": clipped_low, "max_m": clipped_high,
                          "center_m": [(a + b) / 2 for a, b in zip(clipped_low, clipped_high)],
                          "extent_m": [(b - a) / 2 for a, b in zip(clipped_low, clipped_high)]})
    return {"schema": "willow_sample_native_aabb_v1", "region_m": region,
            "floor_m": floor_m, "ground_upper_m": ground_upper_m,
            "wearer_height_m": wearer_height_m,
            "native_components": native, "obstacles": obstacles, "exclusions": exclusions,
            "coverage": {"closed_world": "SUPPLIED_STATIC_MESH_COMPONENT_AND_INSTANCE_ROSTER_ONLY",
                         "export_completeness": "NOT_VERIFIED_BY_THIS_MODULE",
                         "geometry": "NATIVE_WORLD_AABB_CONSERVATIVE_PROXY_NOT_TRIANGLE_TRUTH",
                         "collision_disabled_geometry": "INCLUDED",
                         "floor_exclusion": "max_z <= ground_upper_m",
                         "overhead_exclusion": "min_z >= floor_m + wearer_height_m",
                         "dynamic_and_non_static_mesh_geometry": "NOT_COVERED",
                         "authority": "EVALUATOR_ONLY"}}


def _point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    t = 0. if length2 == 0 else max(0., min(1., ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length2))
    return math.hypot(point[0] - start[0] - t * dx, point[1] - start[1] - t * dy)


def _segment_box_distance(start, end, low, high):
    # Slab intersection followed by endpoint/edge distances. The resulting
    # radius offset has circular corners, unlike an expanded rectangular box.
    enter, leave = 0., 1.
    intersects = True
    for axis in (0, 1):
        delta = end[axis] - start[axis]
        if delta == 0:
            if not low[axis] <= start[axis] <= high[axis]:
                intersects = False
                break
        else:
            a, b = sorted(((low[axis] - start[axis]) / delta, (high[axis] - start[axis]) / delta))
            enter, leave = max(enter, a), min(leave, b)
            if enter > leave:
                intersects = False
                break
    if intersects:
        return 0.
    corners = [(low[0], low[1]), (low[0], high[1]), (high[0], high[1]), (high[0], low[1])]
    distances = [_point_segment_distance(corner, start, end) for corner in corners]
    for point in (start, end):
        distances.append(math.hypot(max(low[0] - point[0], 0., point[0] - high[0]),
                                    max(low[1] - point[1], 0., point[1] - high[1])))
    return min(distances)


def evaluate_route(points_xy, contract, radius_m=.28):
    """Exact continuous XY circle sweep against static AABB projections.

    A clear result is scoped to the supplied roster, not physical free-space
    certification. A route whose swept footprint leaves the clipped region is
    UNKNOWN; any observed proxy contact remains reported. Single points work.
    Negative clearance means circle overlap, capped at -radius inside a box;
    it is not penetration depth. Vertical overlap follows conversion settings.
    """
    points = [_vector(point, 2, "route point") for point in points_xy]
    radius_m = float(radius_m)
    if not points or not math.isfinite(radius_m) or radius_m < 0:
        raise ValueError("nonempty route and finite nonnegative radius required")
    xmin, xmax, ymin, ymax = contract["region_m"]
    covered = all(xmin + radius_m <= x <= xmax - radius_m and
                  ymin + radius_m <= y <= ymax - radius_m for x, y in points)
    segments = list(zip(points, points[1:])) or [(points[0], points[0])]
    contacts, minimum = [], None
    for index, (start, end) in enumerate(segments):
        for box in contract["obstacles"]:
            clearance = _segment_box_distance(start, end, box["min_m"], box["max_m"]) - radius_m
            minimum = clearance if minimum is None else min(minimum, clearance)
            if clearance <= 1e-12:
                contacts.append({"segment_index": index, "id": box["id"], "min_clearance_m": clearance})
    return {"status": "PROXY_CONTACT" if contacts else "CLEAR_WITHIN_SUPPLIED_ROSTER" if covered else "UNKNOWN",
            "contact": bool(contacts), "contacts": contacts,
            "min_clearance_m": minimum, "route_footprint_within_region": covered,
            "coverage": contract["coverage"]}
