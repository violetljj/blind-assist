"""Evaluator-only exact cuboid contact in fixed camera-aligned query volumes.

Distances are camera-axial metres, not historical body travel distances. The
scene solids include hidden faces; visibility is a separate reference. This
module accepts no predicted depth and does not approximate rotated cuboids by
their camera-axis enclosing boxes.
"""
import math

import numpy as np
from scipy.optimize import linprog

BANDS = {'BODY': (-1.05, -.30), 'HEAD': (-.30, .15)}
WIDTHS = {'BODY': (.36, .56, .76), 'HEAD': (.24, .36, .48)}
MIN_AXIAL_M, MAX_AXIAL_M, HORIZON_M, MAX_FULL_WIDTH_M = .5, 4., 3., 1.


def camera_rotation(camera):
    """Camera forward/right/up to world XYZ; UE positive pitch looks up."""
    yaw, pitch, roll = [math.radians(float(camera.get(k, 0))) for k in ('yaw', 'pitch', 'roll')]
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    return np.array([[cp*cy, sr*sp*cy-cr*sy, -(cr*sp*cy+sr*sy)],
                     [cp*sy, sr*sp*sy+cr*cy, cy*sr-cr*sp*sy],
                     [sp, -sr*cp, cr*cp]], dtype=np.float64)


def _cubes(frame, extra_bounds):
    for item in (*frame['native_bounds'], *extra_bounds):
        center = np.asarray(item['center_m'], dtype=np.float64)
        if 'extent_m' in item:
            half = np.asarray(item['extent_m'], dtype=np.float64)
        else:
            half = np.asarray(item['size_m'], dtype=np.float64)/2
        if center.shape != (3,) or half.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(half).all() or (half <= 0).any():
            raise ValueError('Require finite positive world-axis cube extents')
        yield center, half


def _query(frame, part):
    if part not in BANDS:
        raise ValueError('Part must be BODY or HEAD')
    camera = frame['camera']
    origin = np.array([camera[k] for k in ('x', 'y', 'z')], dtype=np.float64)
    rotation = camera_rotation(camera)
    if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
        raise ValueError('Camera pose must be finite')
    # Four camera frustum planes, with no Euclidean-range substitution.
    tx, tz = math.tan(math.radians(22.5)), math.tan(math.radians(20.))
    fov = np.array([[-tx, 1, 0], [-tx, -1, 0], [-tz, 0, 1], [-tz, 0, -1]])
    return origin, rotation, fov


def _solve(objective, A, b, bounds):
    result = linprog(objective, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                     options={'primal_feasibility_tolerance': 1e-9,
                              'dual_feasibility_tolerance': 1e-9})
    if result.status == 2:
        return None
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Contact oracle LP failed: '+str(result.message))
    # Fail a numerical/solver defect instead of silently granting contact.
    if np.max(A @ result.x-b) > 2e-8:
        raise RuntimeError('Contact oracle LP returned an infeasible point')
    return float(result.fun)


def oracle(frame_manifest, part, width, extra_bounds=()):
    """Minimum camera x in [.5,4] at the supplied full width, or no contact.

frame_manifest uses capture manifest fields camera and native_bounds. Extra
world-axis solids accept either native {center_m, extent_m} or receipt
background/floor {center_m, size_m}. No native pixels or predictions are read.
"""
    if not np.isfinite(width) or not 0 <= width <= MAX_FULL_WIDTH_M:
        raise ValueError('Full width must be finite and in [0,1] metres')
    origin, rotation, fov = _query(frame_manifest, part)
    values = []
    for center, half in _cubes(frame_manifest, extra_bounds):
        # world = origin + rotation @ camera_point: exact OBB constraints.
        A = np.concatenate((rotation, -rotation, fov))
        b = np.r_[center+half-origin, origin-center+half, np.zeros(4)]
        value = _solve([1., 0., 0.], A, b,
                       [(MIN_AXIAL_M, MAX_AXIAL_M), (-width/2, width/2), BANDS[part]])
        if value is not None:
            values.append(value)
    return min(values) if values else None


def oracle_width(frame, part, extra_bounds=()):
    """Critical full width 2*min|camera y| within [.5,3]m, bounded by 1m.

None means no intersection in the fixed height/FoV/horizon/maximum-width
scope. It is evaluator scope absence, not public CLEAR or predictor infinity.
"""
    origin, rotation, fov = _query(frame, part)
    values = []
    for center, half in _cubes(frame, extra_bounds):
        spatial = np.concatenate((rotation, -rotation, fov))
        A = np.column_stack((spatial, np.zeros(len(spatial))))
        A = np.vstack((A, [0., 1., 0., -1.], [0., -1., 0., -1.]))
        b = np.r_[center+half-origin, origin-center+half, np.zeros(6)]
        value = _solve([0., 0., 0., 2.], A, b,
                       [(MIN_AXIAL_M, HORIZON_M), (None, None), BANDS[part], (0., MAX_FULL_WIDTH_M/2)])
        if value is not None:
            values.append(max(0., value))
    return min(values) if values else None
