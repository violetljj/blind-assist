"""Causal log-scale update from projective depth registration and metric camera motion."""
from __future__ import annotations

import math

import numpy as np


class LieScaleIEKF:
    """Scalar Lie-group measurement filter. K is in full-resolution pixels.

    The caller supplies calibrated camera-to-world poses in metres. A new
    instance is required at a trajectory boundary. The covariance proxy is
    a regularizer for this local fit, not a calibrated uncertainty interval.
    """

    def __init__(self, initial_scale: float = 1.0):
        if not 0.25 <= initial_scale <= 4.0:
            raise ValueError("initial scale outside supported range [0.25, 4]")
        self.log_scale = math.log(initial_scale)
        self.variance = 0.5**2
        self.accepted = 0
        self.attempts = 0
        self.anchor = None

    @property
    def scale(self):
        return math.exp(self.log_scale)

    def _remember(self, dep, dh, nrm, ls, K, R, t):
        h, w = dep.shape
        v, u = np.mgrid[2:h-1:5, 2:w-1:5]
        z = dep[v, u]
        mask = np.isfinite(z) & (z > 0.35) & (z < 12)
        v, u, z = v[mask], u[mask], z[mask]
        # The saved tensors are sliced at full-resolution pixel indices 4*i.
        ray_x = (4*u + 0.5 - K[0, 2]) / K[0, 0]
        ray_y = (4*v + 0.5 - K[1, 2]) / K[1, 1]
        xyz = np.column_stack((ray_x*z, ray_y*z, z))
        reliable = (np.abs(dh[v, u]) < 0.06) & (np.abs(nrm[1, v, u]) > 0.92)
        self.anchor = (xyz @ R.T, t.copy(), R.copy(), reliable, ls[v, u].copy())

    def step(self, depth_s4, dh_s4, nrm_s4, ls_s4, K, R_wc, t_wc):
        dep, dh, nrm, ls = depth_s4, dh_s4, nrm_s4, ls_s4
        if self.anchor is None:
            self._remember(dep, dh, nrm, ls, K, R_wc, t_wc)
            return self.scale
        xyz_w, t_old, R_old, g_old, ls_old = self.anchor
        trans = (t_old - t_wc) @ R_wc
        baseline = np.linalg.norm(trans)
        # With negligible motion there is no metric scale observation. Refresh
        # a keyframe after large rotation so that it can later share the view.
        if baseline < 0.18:
            if np.sum(R_old * R_wc) < 2.4:
                self._remember(dep, dh, nrm, ls, K, R_wc, t_wc)
            return self.scale
        self.attempts += 1
        if baseline <= 1.5 and np.sum(R_old * R_wc) > 2.0:
            self._register(xyz_w @ R_wc, trans, g_old, ls_old, dep, dh, nrm, ls, K)
        self._remember(dep, dh, nrm, ls, K, R_wc, t_wc)
        return self.scale

    def _register(self, rotated, trans, g_old, ls_old, dep, dh, nrm, ls, K):
        shift = trans / self.scale
        q = rotated + shift
        z = q[:, 2]
        denom = np.maximum(z, 0.25)
        fx, fy = K[0, 0] / 4, K[1, 1] / 4
        u = fx * q[:, 0] / denom + (K[0, 2] - 0.5) / 4
        v = fy * q[:, 1] / denom + (K[1, 2] - 0.5) / 4
        valid = (z > 0.35) & (u >= 0) & (v >= 0) & (u < dep.shape[1]-1) & (v < dep.shape[0]-1)
        if valid.sum() < 16:
            return
        z, q, u, v = z[valid], q[valid], u[valid], v[valid]
        i, j = v.astype(int), u.astype(int)
        du, dv = u-j, v-i
        d00, d01, d10, d11 = dep[i, j], dep[i, j+1], dep[i+1, j], dep[i+1, j+1]
        dx = (1-dv)*(d01-d00) + dv*(d11-d10)
        dy = (1-du)*(d10-d00) + du*(d11-d01)
        target = (1-dv)*(d00 + du*(d01-d00)) + dv*(d10 + du*(d11-d10))
        # r = q_z - D(pi(q)), dq/d(log s) = -shift.
        jx = fx*(shift[0] - q[:, 0]*shift[2]/z)/z
        jy = fy*(shift[1] - q[:, 1]*shift[2]/z)/z
        deriv = -shift[2] + dx*jx + dy*jy
        residual = z - target
        noise = np.maximum(0.08, np.maximum(target, 0.35) *
                           np.exp(np.clip(0.5*(ls[i,j] + ls_old[valid]), -3.0, 1.0)))
        smooth = ((np.abs(dx)+np.abs(dy)) < 0.16*np.maximum(target, 0.35))
        use = smooth & np.isfinite(deriv+residual+noise) & (target > 0.35) & (target < 12.0)
        if use.sum() < 16:
            return
        now_ground = (np.abs(dh[i,j]) < 0.06) & (np.abs(nrm[1,i,j]) > 0.92)
        # Ground tracking is favored only when both frames agree. Other
        # surfaces are needed to see horizontal translation on a flat floor.
        weights = (1 + (now_ground & g_old[valid]).astype(float))[use]
        err, jac, sd = residual[use], deriv[use], noise[use]
        weights /= sd**2 * np.maximum(1.0, np.abs(err)/(1.5*sd))
        # Keep correlated samples in one frame from contributing unlimited precision.
        weights *= min(1.0, 12.0/len(weights))
        info = float(np.dot(weights, jac**2))
        if info < 0.15:
            return
        prior_var = min(0.5**2, self.variance + 0.08**2)
        innovation = -np.dot(weights * jac, err)/(1/prior_var + info)
        self.log_scale = float(np.clip(self.log_scale + np.clip(innovation, -0.4, 0.4),
                                       -math.log(4.0), math.log(4.0)))
        self.variance = 1/(1/prior_var + info)
        self.accepted += 1
