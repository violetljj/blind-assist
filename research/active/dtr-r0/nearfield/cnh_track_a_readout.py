"""Frozen Track A H3-only geometric readout; poses are world-from-ToF.

No visibility, scene geometry, raw128 or labels enter temporal accumulation.
Call separately per sequence. Calibrated bias is subtracted before accumulation.
"""
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation
from cnh_route_sensor import angular_rays, H3, RAW_BIN_M

SHAPE = (8, 8, 16)
WIDTH = RAW_BIN_M * H3.sub_sample
START = RAW_BIN_M * H3.start_bin
END = START + 16 * WIDTH
EDGE = np.tan(np.deg2rad(22.5))
BOXES = np.array([([x-.3, y0, .3], [x+.3, y1, 3.])
                  for x in (-.3, 0., .3) for y0, y1 in ((-.2, .42), (.42, .9))])


def _hist(a):
    a = np.asarray(a)
    if a.ndim != 4 or a.shape[1:] != SHAPE or not np.isfinite(a).all():
        raise ValueError('Finite H3 [N,8,8,16] required; raw128 forbidden')
    return a


def _poses(p, n):
    p = np.asarray(p, dtype=float)
    if p.shape != (n, 4, 4) or not np.isfinite(p).all():
        raise ValueError('Finite world_from_tof [N,4,4] required')
    if not np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-10):
        raise ValueError('Invalid homogeneous pose')
    r = p[:, :3, :3]
    if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-8) or not np.allclose(np.linalg.det(r), 1, atol=1e-8):
        raise ValueError('Proper rigid rotations required')
    return p


@lru_cache(maxsize=1)
def cell_points():
    rays, _ = angular_rays(4)
    centers = START + (np.arange(16)+.5)*WIDTH
    return (rays.reshape(64, 16, 3)[:, None] * centers[None, :, None, None]).reshape(1024, 16, 3)


def _transform(points, t):
    return points @ t[:3, :3].T + t[:3, 3]


def _indices(points):
    r = np.linalg.norm(points, axis=-1)
    xy = np.divide(points[..., :2], points[..., 2, None],
                   out=np.full(points.shape[:-1]+(2,), np.inf), where=points[..., 2, None] > 0)
    valid = (points[..., 2] > 0) & (np.abs(xy) < EDGE).all(-1) & (r >= START) & (r < END)
    safe = np.where(np.isfinite(xy), xy, 0)
    colrow = np.clip(np.floor((safe+EDGE)/(2*EDGE)*8), 0, 7).astype(int)
    bins = np.clip(np.floor((r-START)/WIDTH), 0, 15).astype(int)
    return (colrow[..., 1]*8+colrow[..., 0])*16+bins, valid


def transport(hist, current_from_past):
    """Forward count splat and backward fractional FOV coverage, each [8,8,16]."""
    hist = _hist(np.asarray(hist)[None])[0]
    t = _poses(np.asarray(current_from_past)[None], 1)[0]
    points = cell_points()
    idx, valid = _indices(_transform(points, t))
    mass = np.broadcast_to(hist.reshape(1024, 1)/16, (1024, 16))
    out = np.bincount(idx[valid], weights=mass[valid], minlength=1024).reshape(SHAPE)
    _, covered = _indices(_transform(points, np.linalg.inv(t)))
    return out, covered.mean(-1).reshape(SHAPE)


def accumulate(hist, poses, k=4):
    """Causal mean of current plus <=k-1 transported historical H3 frames.

    W measures FOV/window coverage, not occlusion. Static sensor expectation
    is not invariant to viewpoint; no photometric compensation is claimed.
    """
    hist = _hist(hist)
    poses = _poses(poses, len(hist))
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('Positive integer k required')
    if k == 1:
        return hist.copy(), np.ones(hist.shape)
    out, weights = np.empty(hist.shape, dtype=float), np.ones(hist.shape)
    for i in range(len(hist)):
        start = max(0, i-k+1)
        total = hist[i].astype(float).copy()
        for j in range(start, i):
            if np.array_equal(poses[j], poses[i]):
                total += hist[j]
                weights[i] += 1
            else:
                warped, covered = transport(hist[j], np.linalg.inv(poses[i]) @ poses[j])
                total += warped
                weights[i] += covered
        out[i] = total / weights[i]
        # Identical repeated observations under zero motion remain bitwise exact.
        if all(np.array_equal(poses[j], poses[i]) and np.array_equal(hist[j], hist[i]) for j in range(start, i)):
            out[i] = hist[i]
    return out, weights


def fit_calib_bias(hist, splits, mask=None):
    hist = _hist(hist)
    splits = np.asarray(splits)
    if splits.shape != (len(hist),):
        raise ValueError('One split per frame required')
    selected = splits == 'calib' if mask is None else np.asarray(mask)
    if selected.shape != splits.shape or selected.dtype != bool or not selected.any():
        raise ValueError('Nonempty boolean calibration selection required')
    if np.any(selected & (splits != 'calib')):
        raise ValueError('Bias fitting may only use calib frames')
    return np.median(hist[selected], axis=0)


def subtract_bias(hist, bias):
    hist, bias = _hist(hist), np.asarray(bias)
    if bias.shape != SHAPE or not np.isfinite(bias).all():
        raise ValueError('Finite external calib bias [8,8,16] required')
    return hist - bias


def query_weights(T_Q_tof):
    t = _poses(np.asarray(T_Q_tof)[None], 1)[0]
    rays, area = angular_rays(16)
    rays = rays.reshape(64, -1, 3) @ t[:3, :3].T
    area = area.reshape(64, -1)
    area /= area.sum(-1, keepdims=True)
    origin = t[:3, 3]
    edges = START + np.arange(17)*WIDTH
    weights = []
    for low, high in BOXES:
        near, far = np.zeros(rays.shape[:2]), np.full(rays.shape[:2], np.inf)
        for axis in range(3):
            d = rays[..., axis]
            parallel = np.abs(d) < 1e-12
            a = np.divide(low[axis]-origin[axis], d, out=np.full(d.shape, -np.inf), where=~parallel)
            b = np.divide(high[axis]-origin[axis], d, out=np.full(d.shape, np.inf), where=~parallel)
            near = np.maximum(near, np.minimum(a, b))
            far = np.minimum(far, np.maximum(a, b))
            if not low[axis] <= origin[axis] <= high[axis]:
                far = np.where(parallel, -np.inf, far)
        overlap = np.maximum(0, np.minimum(far[..., None], edges[1:])-np.maximum(near[..., None], edges[:-1]))/WIDTH
        weights.append(np.einsum('zr,zrb->zb', area, overlap))
    return np.clip(weights, 0, 1)


def noisy_poses(poses, seed, dt=.2):
    """Assumed noisy ego-motion, integrated causally; not an IMU-only claim."""
    poses = _poses(poses, len(poses))
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError('Positive sample interval required')
    rng = np.random.default_rng(seed)
    scale = rng.uniform(-.2, .2)
    bias = rng.choice([-1., 1.], 3)*np.deg2rad(1.)
    out = poses.copy()
    for i in range(1, len(poses)):
        delta = np.linalg.inv(poses[i-1]) @ poses[i]
        delta[:3, 3] *= 1+scale+rng.normal(0, .02)
        delta[:3, :3] = delta[:3, :3] @ Rotation.from_rotvec(bias*dt+rng.normal(0, np.deg2rad(.1), 3)).as_matrix()
        out[i] = out[i-1] @ delta
    return out
