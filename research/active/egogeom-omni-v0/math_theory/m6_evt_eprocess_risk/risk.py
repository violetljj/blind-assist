"""Dimensionless geometric evidence and a distance-block sequential detector."""
import math
import numpy as np


class InvariantEProcessBEV:
    """A clip starts with a fresh instance. The geometry score never reads a label."""
    def __init__(self):
        self.grid = None
        self.previous = None

    def step(self, depth_s4, dh_s4, nrm_s4, ls_s4, K, R_wc, t_w, strict_fwd):
        # The original camera rays were computed at full-grid indices + 0.5.
        sub = np.s_[::2, ::2]
        z = depth_s4[sub].ravel()
        if self.grid is None or self.grid[0].shape != z.shape:
            v, u = np.indices(depth_s4[sub].shape, dtype=np.float32)
            self.grid = ((8*u + .5).ravel(), (8*v + .5).ravel())
        u, v = self.grid
        xyz = np.stack(((u-K[0, 2])*z/K[0, 0], (v-K[1, 2])*z/K[1, 1], z), axis=1) @ R_wc.T
        valid = np.isfinite(xyz).all(axis=1) & (z > 0)
        height = xyz[:, 2]
        floor = valid & (height < 0) & (np.abs(nrm_s4[1][sub].ravel()) >= .65)
        if floor.sum() < 12:
            floor = valid & (height < 0)
        if floor.sum() < 12:
            self.previous = None
            return 0.
        ground = np.percentile(height[floor], 15)
        scale = -ground
        dh = dh_s4[sub].ravel()
        floor_dh = np.median(dh[floor])
        ls = ls_s4[sub].ravel()
        # A common additive shift in log-uncertainty or the learned height head cancels.
        conf = np.exp(-np.clip(ls - np.median(ls[valid]), -2, 2))
        rel_h = (height-ground)/scale
        height_head = (dh-floor_dh)/scale
        vertical = np.clip(1.-nrm_s4[1][sub].ravel()**2, 0, 1)
        weight = (np.clip(rel_h/.22, 0, 1) * np.clip(height_head/.16, 0, 1)
                  * vertical * conf)
        keep = valid & (rel_h >= .06) & (rel_h <= 1.3) & (weight > 0)
        world = xyz[keep, :2] + t_w[:2]
        fw = np.asarray(strict_fwd[:2], dtype=float)
        norm = np.linalg.norm(fw)
        if norm < 1e-7:
            self.previous = None
            return 0.
        fw = fw/norm
        rt = np.array([fw[1], -fw[0]])
        def bins(points, evidence):
            offset = (points-t_w[:2])/scale
            forward = offset @ fw
            lateral = offset @ rt
            mask = (np.abs(lateral) <= .22) & (forward >= .2) & (forward < 3.)
            indices = ((forward[mask]-.2)/.2).astype(int).clip(0,13)
            counts = np.bincount(indices, weights=evidence[mask], minlength=14)
            return counts[:-1]+counts[1:]
        now = bins(world, weight[keep])
        past = np.zeros_like(now) if self.previous is None else bins(*self.previous)
        self.previous = world, weight[keep]
        support = .35*now + .65*np.minimum(now, past)
        proximity = 1/(1+(.4 + .2*np.arange(len(support)))**2)
        return float(np.max(-np.expm1(-support/10.) * proximity))


class DistanceBlockBet:
    """Assume P(hit|observed past,H0)<=p0. At most one emission per block."""
    def __init__(self, p0=.1, p1=.55, block_m=2., alpha=.01):
        if not (0 < p0 < p1 < 1 and block_m > 0 and 0 < alpha < 1):
            raise ValueError("invalid betting parameters")
        self.good = math.log(p1/p0)
        self.bad = math.log((1-p1)/(1-p0))
        self.cut = -math.log(alpha)
        self.block_m = block_m
        self.block = None

    def step(self, hit, travelled_m):
        block = math.floor(travelled_m / self.block_m)
        if self.block != block:
            self.block, self.log_wealth, self.fired = block, 0., False
        self.log_wealth += self.good if hit else self.bad
        emit = self.log_wealth >= self.cut and not self.fired
        self.fired |= emit
        return self.log_wealth, emit
