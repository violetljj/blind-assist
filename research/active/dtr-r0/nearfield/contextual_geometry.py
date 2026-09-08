"""Continuous target-cuboid contact in a straight +X route, not all-map truth.

The wearer envelopes are world-axis-aligned. Cuboids may have UE Rotator roll
about X: local Y=(0, cos(r), -sin(r)), local Z=(0, sin(r), cos(r)).
An exact rectangle SAT in YZ and a continuous interval in X suffice for this
restricted rotation. Open structures remain unions of their supplied solids.
"""
import math
from collections.abc import Mapping

from contact_retina_spec import BODY_BOXES


REGIONS = ('BODY', 'HEAD')


def _finite(value, description):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{description} must be finite') from exc
    if not math.isfinite(number):
        raise ValueError(f'{description} must be finite')
    return number


def _vector(value, description):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'{description} must contain three finite coordinates')
    return tuple(_finite(v, description) for v in value)


def _solid(obj, index):
    if not isinstance(obj, Mapping):
        raise ValueError('Each solid must be a cuboid mapping')
    if 'mesh_asset' in obj or 'primitive_asset' in obj:
        raise ValueError('Mesh/primitive assets are not exact supplied solid cuboids')
    try:
        center = _vector(obj['center_m'], 'center_m')
        size = _vector(obj['size_m'], 'size_m')
    except KeyError as exc:
        raise ValueError('Each cuboid requires center_m and size_m') from exc
    if any(v <= 0 for v in size):
        raise ValueError('Cuboid dimensions must be positive')
    rotation = obj.get('rotation_deg', {})
    if not isinstance(rotation, Mapping):
        raise ValueError('rotation_deg must be a pitch/yaw/roll mapping')
    if set(rotation) - {'pitch', 'yaw', 'roll'}:
        raise ValueError('Unknown rotation component')
    pitch, yaw, roll = (_finite(rotation.get(k, 0.), f'rotation {k}')
                        for k in ('pitch', 'yaw', 'roll'))
    if pitch != 0 or yaw != 0:
        raise ValueError('Only cuboid roll about world X is supported')
    angle = math.radians(roll)
    c, s = math.cos(angle), math.sin(angle)
    return dict(center=center, half=tuple(v / 2 for v in size),
                u=(c, -s), v=(s, c),
                name=str(obj.get('name', f'unnamed_solid_{index}')))


def _contact_distance(origin, bounds, solid, horizon):
    low, high = bounds
    body_center = tuple(origin[k] + (low[k] + high[k]) / 2 for k in range(3))
    body_half = tuple((high[k] - low[k]) / 2 for k in range(3))
    center, half, u, v = (solid[k] for k in ('center', 'half', 'u', 'v'))
    delta = (center[1] - body_center[1], center[2] - body_center[2])
    # These four normals are the complete rectangle SAT. With rotation around X
    # only, the remaining 3-D cross-product normals are duplicates of these or X.
    for ay, az in ((1., 0.), (0., 1.), u, v):
        separation = abs(delta[0] * ay + delta[1] * az)
        body_radius = body_half[1] * abs(ay) + body_half[2] * abs(az)
        solid_radius = (half[1] * abs(ay * u[0] + az * u[1])
                        + half[2] * abs(ay * v[0] + az * v[1]))
        if separation > body_radius + solid_radius:
            return None
    enter = center[0] - half[0] - (origin[0] + high[0])
    leave = center[0] + half[0] - (origin[0] + low[0])
    distance = max(0., enter)
    return distance if distance <= min(horizon, leave) else None


def _validated_geometry(objects, wearer, horizon):
    horizon = _finite(horizon, 'horizon')
    if horizon < 0:
        raise ValueError('horizon must be nonnegative')
    if not isinstance(wearer, Mapping):
        raise ValueError('wearer must be a world-frame pose mapping')
    try:
        origin = tuple(_finite(wearer[k], f'wearer {k}') for k in ('x', 'y', 'z'))
    except KeyError as exc:
        raise ValueError('wearer requires x, y, z coordinates') from exc
    for component in ('pitch', 'yaw', 'roll'):
        if _finite(wearer.get(component, 0.), f'wearer {component}') != 0:
            raise ValueError('Wearer envelope requires a straight world-X axis-aligned pose')
    solids = [_solid(obj, i) for i, obj in enumerate(objects)]
    return origin, horizon, solids


def target_contact(objects, wearer, horizon=3.):
    """Sweep BODY/HEAD over [0, horizon] metres; touching counts as contact.

    ``first_contact_distance_m`` is ordered BODY, HEAD. Witness names list every
    supplied solid contacting each region within the horizon, in input order;
    they include supports without interpreting names, materials or target roles.
    No collision volume or intrusion-depth estimate is inferred.
    """
    origin, horizon, solids = _validated_geometry(objects, wearer, horizon)
    first, witnesses = [], {}
    for region, bounds in zip(REGIONS, BODY_BOXES):
        hits = [(distance, solid['name']) for solid in solids
                if (distance := _contact_distance(origin, bounds, solid, horizon)) is not None]
        first.append(min((d for d, _ in hits), default=None))
        witnesses[region] = [name for _, name in hits]
    relation = {(False, False): 'CLEAR', (True, False): 'BODY_ONLY',
                (False, True): 'HEAD_ONLY', (True, True): 'BOTH'}[
                    tuple(distance is not None for distance in first)]
    return dict(relation=relation, first_contact_distance_m=first,
                object_witness_names=witnesses,
                authority='TARGET_SOLID_CUBOID_UNION_X_ROLL_SAT_NOT_ALL_MAP_OR_MESH_TRUTH')


def _clip_half_plane(polygon, axis, boundary, keep_above):
    """Sutherland-Hodgman clipping against one closed axis-aligned half-plane."""
    if not polygon:
        return []
    clipped = []
    previous = polygon[-1]
    previous_inside = (previous[axis] >= boundary if keep_above
                       else previous[axis] <= boundary)
    for current in polygon:
        current_inside = (current[axis] >= boundary if keep_above
                          else current[axis] <= boundary)
        if current_inside != previous_inside:
            fraction = (boundary - previous[axis]) / (current[axis] - previous[axis])
            point = [previous[k] + fraction * (current[k] - previous[k]) for k in (0, 1)]
            point[axis] = boundary
            clipped.append(tuple(point))
        if current_inside:
            clipped.append(current)
        previous, previous_inside = current, current_inside
    return clipped


def _clipped_lateral_interval(origin, bounds, solid, horizon):
    low, high = bounds
    center, half, u, v = (solid[k] for k in ('center', 'half', 'u', 'v'))
    if (center[0] + half[0] < origin[0] + low[0]
            or center[0] - half[0] > origin[0] + high[0] + horizon):
        return None
    polygon = [(center[1] + sy * half[1] * u[0] + sz * half[2] * v[0],
                center[2] + sy * half[1] * u[1] + sz * half[2] * v[1])
               for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for axis in (0, 1):
        polygon = _clip_half_plane(polygon, axis, origin[axis + 1] + low[axis + 1], True)
        polygon = _clip_half_plane(polygon, axis, origin[axis + 1] + high[axis + 1], False)
    if not polygon:
        return None
    return min(p[0] for p in polygon), max(p[0] for p in polygon)


def _union_width(intervals):
    total = 0.
    right = None
    for low, high in sorted(intervals):
        total += max(0., high - (low if right is None else max(low, right)))
        right = high if right is None else max(right, high)
    return total


def intrusion_metrics(objects, wearer, horizon=3.):
    """Union lateral occupancy after exact YZ clipping of reachable cuboids.

    For each BODY/HEAD rectangle, clip each X-roll cuboid's YZ polygon, project
    the clipped polygon onto Y, and union these intervals. X reachability is
    intersection with the envelope's swept X interval. Widths from different X
    distances may contribute to the same union: this is projected occupancy
    somewhere in the horizon, not a simultaneous cross-section, volume, collision
    severity or probability. Touching along a horizontal edge can have nonzero
    projected width even though its clipped polygon area is zero.
    """
    origin, horizon, solids = _validated_geometry(objects, wearer, horizon)
    regions = {}
    for region, bounds in zip(REGIONS, BODY_BOXES):
        intervals = [interval for solid in solids
                     if (interval := _clipped_lateral_interval(
                         origin, bounds, solid, horizon)) is not None]
        region_width = bounds[1][1] - bounds[0][1]
        width = min(region_width, _union_width(intervals))
        regions[region] = dict(lateral_intrusion_width_m=width,
                               lateral_coverage_ratio=width / region_width)
    return dict(regions=regions,
                authority=('TARGET_SOLID_CUBOID_UNION_CLIPPED_YZ_LATERAL_OCCUPANCY_'
                           'SOMEWHERE_IN_HORIZON_NOT_VOLUME_SEVERITY_PROBABILITY_OR_ALL_MAP_TRUTH'))
