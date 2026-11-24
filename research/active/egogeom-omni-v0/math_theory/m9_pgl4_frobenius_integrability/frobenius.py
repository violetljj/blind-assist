"""Positive inverse-depth affine correction from oriented normal edge integrals."""
import numpy as np


def camera_normal(level, rotation):
    gravity = -rotation[2]
    gravity = gravity / np.linalg.norm(gravity)
    forward = np.array([0., 0., 1.]) - gravity * gravity[2]
    if np.linalg.norm(forward) <= 1.e-3:
        forward = np.array([0., -1., 0.]) + gravity * gravity[1]
    forward /= np.linalg.norm(forward)
    right = np.cross(gravity, forward)
    right /= np.linalg.norm(right)
    return np.einsum("ji,jhw->ihw", np.stack((right, gravity, forward)), level)


def edge_equations(disparity, x, y, normal):
    nx, ny, nz = normal
    dot = nx*x + ny*y + nz
    safe = np.abs(dot) > .22
    dot = np.where(safe, dot, 1.)
    omega_x, omega_y = -nx/dot, -ny/dot
    ex = .5*(omega_x[:, 1:] + omega_x[:, :-1]) * np.diff(x, axis=1)
    ey = .5*(omega_y[1:] + omega_y[:-1]) * np.diff(y, axis=0)
    curl = ex[:-1] + ey[:, 1:] - ex[1:] - ey[:, :-1]
    # Edge integrals share a quadrilateral. A low curl is a consistency cue only.
    cell_ok = (safe[:-1, :-1] & safe[:-1, 1:] & safe[1:, :-1] & safe[1:, 1:])
    confidence = np.where(cell_ok, 1 / (1 + (curl/.035)**2), 0.)
    s = np.s_[:-1, :-1]
    columns, targets, weights = [], [], []
    for e, j in ((ex[:-1], np.s_[:-1, 1:]), (ey[:, :-1], np.s_[1:, :-1])):
        k = np.exp(-np.clip(e, -4., 4.))
        rhs = k*disparity[s] - disparity[j]
        delta = np.abs(np.log(disparity[s]/disparity[j]))
        w = confidence * (delta < .22) * (np.abs(e) < .35)
        w /= 1 + (rhs/(.025 + .09*disparity[s]))**2
        columns.append(np.stack((1-k, x[j]-k*x[s], y[j]-k*y[s]), axis=-1).reshape(-1, 3))
        targets.append(rhs.ravel())
        weights.append(w.ravel())
    return np.vstack(columns), np.concatenate(targets), np.concatenate(weights)


class FrobeniusFilter:
    def __init__(self):
        self.beta = np.zeros(3)
        self.pixels = None

    def step(self, depth, normal, intrinsics, rotation):
        if self.pixels is None or self.pixels[0].shape != depth.shape:
            vv, uu = np.indices(depth.shape, dtype=np.float32)
            self.pixels = (4*uu+.5, 4*vv+.5)
        u, v = self.pixels
        x = (u-intrinsics[0, 2])/intrinsics[0, 0]
        y = (v-intrinsics[1, 2])/intrinsics[1, 1]
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.all():
            # Unsupported finite-difference neighborhoods keep the source values.
            return depth.copy(), False, self.beta.copy()
        sample = np.s_[2::4, 2::4]
        n = camera_normal(normal[:, sample[0], sample[1]], rotation)
        A, b, w = edge_equations(1/depth[sample], x[sample], y[sample], n)
        ok = np.isfinite(b) & np.isfinite(w) & (w > .02) & np.isfinite(A).all(1)
        updated = False
        if ok.sum() >= 24:
            A, b, w = A[ok], b[ok], w[ok]
            gram = (A.T*w)@A
            ev = np.linalg.eigvalsh(gram)
            if ev[0] > max(1.e-7, .01*ev[-1]):
                # Ridge in inverse metres, toward identity. Smooth causal state within clip.
                beta = np.linalg.solve(gram + .1*np.trace(gram)/3*np.eye(3), (A.T*w)@b)
                self.beta += .25*(np.clip(beta, [-.15,-.12,-.12], [.15,.12,.12])-self.beta)
                updated = True
        b0, bx, by = self.beta
        rel = (b0 + bx*x + by*y)*depth
        shrink = min(1., .35/max(float(np.max(np.abs(rel))), 1.e-8))
        return (depth/(1 + shrink*rel)).astype(np.float32), updated, shrink*self.beta
