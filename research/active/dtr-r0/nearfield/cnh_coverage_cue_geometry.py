"""Pure NumPy finite-ray geometry for the frozen coverage-cue diagnostic.

Sphere cells are finite 2.5 cm-radius world quadrature cells, not a guarantee
about every point of the corridor. Visibility is exact for the retained rays,
with no occlusion or photon/return-strength model.
"""
from functools import lru_cache
from pathlib import Path
import ast
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FROZEN_RAY_SOURCE = ROOT / 'artifacts.local/work/cnh-track-a-v5-20260928/data/source/cnh_route_sensor.py'
OUT = ROOT / 'artifacts.local/work/cnh-coverage-cue-geometry-20261005'


@lru_cache(None)
def rays():
    """Return (zone-major directions [16384,3], tangent-grid metadata).

    Metadata includes q[128], edge, step, and grid_directions[128,128,3].
    Arrays are read-only. The ordering equals angular_rays(16).reshape(-1,3).
    """
    edge = np.tan(np.deg2rad(45. / 2))
    step = 2 * edge / 128
    q = -edge + (np.arange(128) + .5) * step
    yy, xx = np.meshgrid(q, q, indexing='ij')
    xyz = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    grid = xyz / np.linalg.norm(xyz, axis=-1)[..., None]
    zone = grid.reshape(8, 16, 8, 16, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 3)
    for value in (q, grid, zone):
        value.flags.writeable = False
    return zone, dict(q=q, edge=edge, step=step, grid_directions=grid)


def rotation_yaw_pitch(yaw_deg, pitch_deg=-10.):
    """Sensor-to-travel rotation Ry(yaw) @ Rx(pitch); X right, Y down."""
    yaw, pitch = np.broadcast_arrays(np.deg2rad(yaw_deg), np.deg2rad(pitch_deg))
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    result = np.empty(yaw.shape + (3, 3))
    result[..., 0, 0] = cy
    result[..., 0, 1] = sy * sp
    result[..., 0, 2] = sy * cp
    result[..., 1, 0] = 0
    result[..., 1, 1] = cp
    result[..., 1, 2] = -sp
    result[..., 2, 0] = -sy
    result[..., 2, 1] = cy * sp
    result[..., 2, 2] = cy * cp
    return result


def rodrigues(rotvec_rad):
    """Vectorized exponential map, including an exact zero-vector identity."""
    v = np.asarray(rotvec_rad, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError('rotvec_rad must end in dimension 3')
    angle = np.linalg.norm(v, axis=-1)
    k = np.zeros(v.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2] = -v[..., 2], v[..., 1]
    k[..., 1, 0], k[..., 1, 2] = v[..., 2], -v[..., 0]
    k[..., 2, 0], k[..., 2, 1] = -v[..., 1], v[..., 0]
    a = np.sinc(angle / np.pi)
    b = .5 * np.sinc(angle / (2*np.pi)) ** 2
    return np.eye(3) + a[..., None, None] * k + b[..., None, None] * (k @ k)


def _hit(points, directions, radius2):
    dot = np.sum(points[:, None, :] * directions, axis=-1)
    distance2 = np.sum(points * points, axis=-1)
    # Forward half-rays: closest point is the origin if dot<0.
    perpendicular2 = distance2[:, None] - np.maximum(dot, 0.) ** 2
    return perpendicular2 <= radius2 + 1e-14


def visible_spheres(local_points, radius=.025):
    """Exact finite-ray intersection for points [...,3], same leading shape.

    The nearest 3x3 tangent-grid candidates give a fast *positive* certificate.
    They are not assumed to contain the angular nearest ray. For negatives,
    an analytic sphere-projection rectangle rejects disjoint grids; unresolved
    points get all 16384 rays. Thus the optimization cannot omit a hit.
    """
    points = np.asarray(local_points, dtype=float)
    if points.shape[-1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError('Need finite [...,3] sphere centers')
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('Need finite nonnegative sphere radius')
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    if not len(flat):
        return np.empty(shape, dtype=bool)
    zone, meta = rays()
    grid, q, step = meta['grid_directions'], meta['q'], meta['step']
    radius2 = radius**2
    distance2 = np.sum(flat*flat, axis=-1)
    result = distance2 <= radius2 + 1e-14
    active = (~result) & (flat[:, 2] + radius >= 0.)
    ids = np.flatnonzero(active)
    p = flat[ids]
    # z near zero goes to the exact fallback; division here is only indexing.
    safez = np.where(p[:, 2] > 1e-12, p[:, 2], 1.)
    tangent = p[:, :2] / safez[:, None]
    index = np.rint((tangent-q[0])/step).clip(0, 127).astype(np.int64)
    offsets = np.array([-1, 0, 1])
    xx = np.clip(index[:, 0, None] + np.tile(offsets, 3), 0, 127)
    yy = np.clip(index[:, 1, None] + np.repeat(offsets, 3), 0, 127)
    hit = _hit(p, grid[yy, xx], radius2).any(-1)
    result[ids[hit]] = True
    remain = ids[~hit]
    p = flat[remain]
    z = p[:, 2]
    regular = z > radius
    possible = ~regular
    if np.any(regular):
        pr = p[regular]
        zr = pr[:, 2]
        denom = zr*zr-radius2
        center = pr[:, :2]*zr[:, None]/denom[:, None]
        extent = radius*np.sqrt(np.maximum(pr[:, :2]**2+zr[:, None]**2-radius2, 0.))/denom[:, None]
        lower, upper = center-extent, center+extent
        lo = np.ceil((lower-q[0])/step - 1e-12).astype(np.int64)
        hi = np.floor((upper-q[0])/step + 1e-12).astype(np.int64)
        overlaps = ((np.maximum(lo, 0) <= np.minimum(hi, 127)).all(-1))
        possible[regular] = overlaps
    # This uncommon fallback certifies all negatives as well as positives.
    # Keep temporary dot/perpendicular arrays below ~8 MB each.
    ambiguous = remain[possible]
    for start in range(0, len(ambiguous), 64):
        batch = ambiguous[start:start+64]
        dot = flat[batch] @ zone.T
        perpendicular2 = distance2[batch, None] - np.maximum(dot, 0.)**2
        result[batch] = (perpendicular2 <= radius2 + 1e-14).any(-1)
    return result.reshape(shape)


def _full_grid(points, radius):
    flat = np.asarray(points).reshape(-1, 3)
    zone, _ = rays()
    result = []
    for start in range(0, len(flat), 64):
        p = flat[start:start+64]
        dot = p @ zone.T
        dist = np.sum(p*p, axis=-1)[:, None] - np.maximum(dot, 0.)**2
        result.append((dist <= radius**2+1e-14).any(-1))
    return np.concatenate(result)


def check():
    """Full-grid parity, retained source parity, physical checks and CPU timing."""
    started = time.perf_counter()
    zone, meta = rays()
    text = FROZEN_RAY_SOURCE.read_text(encoding='utf8')
    tree = ast.parse(text)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'angular_rays')
    ns = dict(np=np, FOV_DEG=45.)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(FROZEN_RAY_SOURCE), 'exec'), ns)
    retained, _ = ns['angular_rays'](16)
    np.testing.assert_array_equal(zone, retained.reshape(-1, 3))
    rng = np.random.default_rng(20261005)
    random = np.column_stack((rng.uniform(-1.5, 1.5, 1200), rng.uniform(-1.5, 1.5, 1200), rng.uniform(-.2, 3., 1200)))
    # Boundary samples straddle finite FOV edges, corners, behind and origin.
    q = meta['q']
    z = np.repeat(np.array([.001, .02, .03, .1, .9, 1.13, 1.4, 3.]), 128)
    tangent = np.tile(q, 8)
    boundary = np.column_stack((z*meta['edge'] + rng.uniform(-.035, .035, len(z)), z*tangent, z))
    points = np.concatenate((random, boundary, boundary[:, [1, 0, 2]], [[0, 0, 0], [0, 0, -.02], [0, 0, -.03]]))
    checks = []
    for radius in (0., .0025, .025, .05):
        actual = visible_spheres(points, radius)
        expected = _full_grid(points, radius)
        differences = int(np.count_nonzero(actual != expected))
        if differences:
            raise AssertionError((radius, differences))
        checks.append(dict(radius_m=radius, centers=len(points), differences=differences))
    rotations = rotation_yaw_pitch(np.linspace(-60, 60, 21))
    np.testing.assert_allclose(rotations @ rotations.swapaxes(-1,-2), np.broadcast_to(np.eye(3), rotations.shape), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(rotations), 1., atol=1e-14)
    np.testing.assert_allclose(rodrigues([[0,0,0],[0,np.pi/2,0]]), rotation_yaw_pitch([0,90], 0), atol=1e-14)
    benchmark = rng.normal(size=(80, 11, 16, 3, 3))*.05
    benchmark[..., 0] += np.tile(np.array([-.275,-.225,.225,.275]), 4)[None,None,:,None]
    benchmark[..., 1] += np.tile(np.array([-.045,.265,.54,.78]), 4)[None,None,:,None]
    benchmark[..., 2] += np.array([1.29,1.13,.97])[None,None,None,:]
    local = np.einsum('...j,...jk->...k', benchmark, rotation_yaw_pitch(15))
    t = time.perf_counter()
    for _ in range(10):
        visible_spheres(local)
    elapsed = (time.perf_counter()-t)/10
    result = dict(status='PASS', backend='numpy CPU; no torch imports', retained_rays=16384,
        ray_source=str(FROZEN_RAY_SOURCE), ray_source_sha256=hashlib.sha256(FROZEN_RAY_SOURCE.read_bytes()).hexdigest(),
        ray_order='8x8 zones, each16x16 subrays', ray_parity='bitwise equal', full_grid_checks=checks,
        rotation_orthonormality='PASS', benchmark_centers=42240, benchmark_seconds_per_call=elapsed,
        exactness='3x3 hit certificate; analytic projection rectangle reject; ambiguous all16384 fallback',
        elapsed_seconds=time.perf_counter()-started)
    return result


if __name__ == '__main__':
    result = check()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'geometry_engineering_check.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, indent=2))
