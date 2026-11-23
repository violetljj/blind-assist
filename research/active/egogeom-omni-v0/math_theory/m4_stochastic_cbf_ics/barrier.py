"""Finite-horizon separation and causal, sparse two-view motion. SI units."""
from __future__ import annotations

from dataclasses import dataclass
from math import inf, sqrt

import cv2
import numpy as np
from scipy.optimize import minimize_scalar


def head_on_distance(closing, radius=0.45, delay=0.65, brake=2.5, side=1.5):
    """Exact deterministic head-on boundary with unconstrained free lateral space."""
    if closing < 0 or radius <= 0 or delay < 0 or brake <= 0 or side < 0:
        raise ValueError("Invalid physical parameter")
    if side == 0:
        return radius + closing * delay + closing**2 / (2 * brake)
    end = sqrt(2 * radius / side)

    def needed(t):
        y = 0.5 * side * t**2
        return closing * (delay + t) - 0.5 * brake * t**2 + sqrt(max(0, radius**2 - y**2))

    best = minimize_scalar(lambda t: -needed(t), bounds=(0, end), method="bounded",
                           options={"xatol": 1e-12})
    return max(needed(0), needed(end), -best.fun)


def contact_time(p, v, radius=0.45):
    """First disk contact along a constant-velocity ray, infinity on a miss."""
    c = float(p @ p) - radius**2
    if c <= 0:
        return 0.0
    a, b = float(v @ v), float(p @ v)
    disc = b*b - a*c
    if a <= 0 or b >= 0 or disc < 0:
        return inf
    return c / (-b + sqrt(disc))


class EscapeMargin:
    """Best of six admissible controls; the Lipschitz deduction covers inter-grid time."""

    def __init__(self, horizon=2.0, delay=0.65, radius=0.45, brake=2.5, side=1.5):
        if horizon <= 0 or delay < 0 or radius <= 0 or min(brake, side) < 0:
            raise ValueError("Invalid horizon or geometry")
        self.horizon, self.radius = horizon, radius
        self.times = np.linspace(0, horizon, 41)
        self.dt = self.times[1] - self.times[0]
        acc = np.array([(a, b) for a in (0., brake) for b in (0., side, -side)])
        self.acc, self.delay = acc, delay
        active = np.maximum(self.times - delay, 0)
        self.shift = 0.5 * active[None, :, None]**2 * acc[:, None, :]
        self.max_dv = active[-1] * np.linalg.norm(acc, axis=1)

    def clearance(self, p, v, allowance=0.0):
        track = p + self.times[:, None] * v + self.shift
        sampled = np.sqrt(np.min(np.sum(track * track, axis=-1), axis=1))
        lower = sampled - 0.5 * self.dt * (np.linalg.norm(v) + self.max_dv)
        return float(lower.max() - self.radius - allowance)

    def exact_clearance(self, p, v):
        """Continuous-time minimum for each fixed maneuver, from cubic stationary points."""
        wait = min(self.horizon,self.delay)
        v2 = float(v @ v)
        nearest = np.clip(-p @ v / max(v2,1e-20),0,wait)
        pre_min = np.linalg.norm(p + nearest*v)
        p0 = p + wait*v
        duration = self.horizon-wait
        best = -inf
        for a in self.acc:
            coeff = [.5*(a@a), 1.5*(a@v), v2+a@p0, p0@v]
            roots = np.roots(coeff)
            real = roots.real[abs(roots.imag) < 1e-8]
            tt = np.r_[0.,duration,real[(0<real)&(real<duration)]][:,None]
            dist = np.linalg.norm(p0 + tt*v + .5*tt**2*a, axis=1)
            best = max(best,min(pre_min,dist.min()))
        return float(best-self.radius)


@dataclass
class View:
    gray: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray


def unproject(xy, view):
    """The frozen cache samples full-resolution pixel centers at 4*i exactly."""
    idx = np.rint(xy).astype(int)
    h, w = view.depth.shape
    z = view.depth[idx[:, 1].clip(0, h-1), idx[:, 0].clip(0, w-1)]
    ray = (4 * xy - view.K[:2, 2]) / np.diag(view.K)[:2]
    cam = np.column_stack([ray * z[:, None], z])
    return cam @ view.R.T, z


def weighted_motion(positions, velocities, variance):
    """Diagonal-isotropic GLS; weighted sample scatter records excess dispersion."""
    weights = 1 / np.maximum(variance, 1e-9)
    weights /= weights.sum()
    p = weights @ positions
    v = weights @ velocities
    residual = velocities - v
    scatter = (weights[:, None] * residual).T @ residual
    cov_v = np.eye(2) / np.sum(1 / np.maximum(variance, 1e-9)) + scatter
    return p, v, cov_v


def sparse_motion(before, now, heading, corridor_d, dt):
    """Find range-local sparse support; missing matches fall back to static world geometry."""
    if not np.isfinite(corridor_d) or dt <= 0:
        return None
    fw = heading[:2].copy()
    norm = np.linalg.norm(fw)
    if norm < 1e-6:
        return None
    fw /= norm
    basis = np.array([fw, [fw[1], -fw[0]]]).T
    ego = ((now.t - before.t)[:2] @ basis) / dt
    static_v = -ego
    # The support and the reported distance share the frozen corridor anchor.
    fallback = (np.array([corridor_d, 0.0]), static_v,
                np.eye(2) * 0.15**2, np.eye(2) * 0.40**2, "static_fallback")
    xy0 = cv2.goodFeaturesToTrack(before.gray, 80, 0.005, 2, blockSize=3)
    if xy0 is None or len(xy0) < 3:
        return fallback
    lk = {"winSize": (9, 9), "maxLevel": 2, "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)}
    xy1, ok1, _ = cv2.calcOpticalFlowPyrLK(before.gray, now.gray, xy0, None, **lk)
    back, ok0, _ = cv2.calcOpticalFlowPyrLK(now.gray, before.gray, xy1, None, **lk)
    xy0, xy1 = xy0[:, 0], xy1[:, 0]
    height, width = now.gray.shape
    valid = (ok0[:, 0] > 0) & (ok1[:, 0] > 0) & (np.linalg.norm(back[:, 0] - xy0, axis=1) < 0.8)
    valid &= ((xy1 >= 0) & (xy1 <= [width-1, height-1])).all(axis=1)
    if valid.sum() < 3:
        return fallback
    old, z0 = unproject(xy0[valid], before)
    new, z1 = unproject(xy1[valid], now)
    p = new[:, :2] @ basis
    # World motion first removes camera translation, then subtract wearer velocity
    # exactly once to recover obstacle-relative velocity in the current local axes.
    world_v = (new - old + now.t - before.t)[:, :2] @ basis / dt
    ok = (np.isfinite(z0) & np.isfinite(z1) & (z0 > .15) & (z1 > .15)
          & (abs(p[:, 0] - corridor_d) < 0.45) & (abs(p[:, 1]) < 0.55)
          & (np.linalg.norm(world_v, axis=1) < 8))
    if ok.sum() < 3:
        return fallback
    # A heuristic per-point depth error model. Camera errors and depth errors may correlate.
    z0, z1, p, world_v = z0[ok], z1[ok], p[ok], world_v[ok]
    sd0, sd1 = .03 + .015 * z0**2, .03 + .015 * z1**2
    var = (sd0**2 + sd1**2) / dt**2 + .05**2
    med = np.median(world_v, axis=0)
    residual = np.linalg.norm(world_v - med, axis=1)
    keep = residual <= max(0.4, 2.5 * float(np.median(residual)))
    if keep.sum() < 3:
        return fallback
    pos, vel, cov = weighted_motion(p[keep], world_v[keep], var[keep])
    # Resolve static support explicitly: static obstacles still approach a moving wearer.
    if np.linalg.norm(vel) < .25:
        vel = np.zeros(2)
        mode = "matched_static"
    else:
        mode = "matched_moving_candidate"
    cov_p = np.eye(2) / np.sum(1 / sd1[keep]**2)
    return pos, vel - ego, cov_p, cov, mode
