"""Grid-unit Chambolle–Pock solver for height (m) on a 5 cm grid."""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

CELL = 0.05
SHAPE = (54, 60)
ALPHA1, ALPHA0, BETA, GAMMA, ETA, DELTA = 0.10, 0.22, 0.010, 0.018, 0.075, 0.010
STEPS = 140
STEP = 0.22
SQRT2 = np.sqrt(2.0)


def dx(x):
    out = np.zeros_like(x)
    out[..., :-1] = x[..., 1:] - x[..., :-1]
    return out


def dy(x):
    out = np.zeros_like(x)
    out[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return out


def tx(p):
    """Exact transpose of the truncated forward difference."""
    out = np.zeros_like(p)
    out[..., :-1] -= p[..., :-1]
    out[..., 1:] += p[..., :-1]
    return out


def ty(p):
    out = np.zeros_like(p)
    out[..., :-1, :] -= p[..., :-1, :]
    out[..., 1:, :] += p[..., :-1, :]
    return out


def operator(x):
    u, vx, vy, j = x
    a, b = dx(u) - vx, dy(u) - vy
    a[:, -1] = 0
    b[-1] = 0
    return np.stack((a, b, dx(vx), dy(vy), (dy(vx) + dx(vy)) / SQRT2,
                     dx(j), dy(j), u + j, vx, vy))


def adjoint(y):
    p, q, r, s, t, a, b, d, f, g = y
    masked_p, masked_q = p.copy(), q.copy()
    masked_p[:, -1] = 0
    masked_q[-1] = 0
    return np.stack((tx(p) + ty(q) + d, -masked_p + tx(r) + ty(t) / SQRT2 + f,
                     -masked_q + ty(s) + tx(t) / SQRT2 + g, tx(a) + ty(b) + d))


def project_ball(values, radius):
    values /= np.maximum(np.sqrt(np.sum(values**2, axis=0)) / radius, 1.0)


def solve(height, valid, slopes=None, slope_valid=None, steps=STEPS):
    """Masked height observation, optional measured ground slopes in m per cell."""
    if not np.any(valid):
        return np.zeros((4, *height.shape), np.float32)
    w = valid.astype(np.float32)
    h = np.where(valid, height, 0.).astype(np.float32)
    near = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    x = np.zeros((4, *h.shape), np.float32)
    x[0] = gaussian_filter(h[tuple(near)], 2.0)
    x[1:3] = np.stack((dx(x[0]), dy(x[0])))
    xbar = x.copy()
    dual = np.zeros((10, *h.shape), np.float32)
    for _ in range(steps):
        dual += STEP * operator(xbar)
        for sl, radius in ((slice(0, 2), ALPHA1), (slice(2, 5), ALPHA0), (slice(5, 7), GAMMA)):
            project_ball(dual[sl], radius)
        dual[7] = w * (dual[7] - STEP * h) / (1 + STEP)
        if slopes is None:
            dual[8:] = 0
        else:
            dual[8:] -= STEP * slopes
            dual[8:] /= 1 + STEP * DELTA / ETA
            project_ball(dual[8:], ETA)
            dual[8:] *= slope_valid
        updated = x - STEP * adjoint(dual)
        j = updated[3]
        j[:] = np.sign(j) * np.maximum(abs(j) - STEP * BETA, 0.)
        xbar = 2 * updated - x
        x = updated
    return x
