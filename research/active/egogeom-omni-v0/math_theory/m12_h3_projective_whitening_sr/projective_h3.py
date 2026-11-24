"""Whitened H3 contrasts and profiled, signed-data Gaussian likelihoods."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import helmert, solve_triangular
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parents[1]
REPO = ROUTE.parents[2]
WORK = REPO / "artifacts.local" / "work"
OUT = WORK / ROUTE.name / "math_theory" / HERE.name
WIDTH = 8 * 0.0375348
BINS = 16


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ar_cov(rho=0.25):
    idx = np.arange(BINS)
    return rho ** abs(idx[:, None] - idx)


def contrast(cov):
    """13 x 16. Two raw bins are discarded, then DC removed and noise whitened."""
    k = np.pad(helmert(14, full=False), ((0, 0), (2, 0)))
    l = np.linalg.cholesky(k @ cov @ k.T)
    return solve_triangular(l, k, lower=True)


def features(h, w, has_tof=1.0):
    """(...,16) -> (...,13) linear and unit contrasts. Broadcast availability mask."""
    y = np.where(np.asarray(has_tof, dtype=bool)[..., None], h, 0.) @ w.T
    norm = np.linalg.norm(y, axis=-1, keepdims=True)
    q = y / np.where(norm > 0, norm, 1.)
    return y, q


def torch_features(h, w, has_tof):
    """Use precomputed whitening weights in the input tensor dtype and device."""
    import torch
    y = torch.where(has_tof.bool().unsqueeze(-1), h, 0.0) @ w.T
    norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    q = y / torch.where(norm > 0, norm, 1.0)
    return y, q


def pulse(z, width, sigma):
    """Unit Gaussian IRF integrated over fixed bins; z and sigma in metres."""
    borders = (np.arange(17) * width - np.asarray(z)[..., None]) / sigma
    return np.where(borders[..., :-1] > 0, -np.diff(ndtr(-borders)), np.diff(ndtr(borders)))


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 0, n, 1.)


class EchoBank:
    """Profile positive coefficients of a background atom and one candidate front.

    For near-collinear atoms use a one-ray fit at machine rank tolerance.
    Small positive cone apertures are evaluated in an orthonormal basis.
    """
    def __init__(self, w, z2, z1_grid, sigma):
        self.bg = unit(pulse(z2, WIDTH, sigma) @ w.T)
        front = unit(pulse(z1_grid, WIDTH, sigma) @ w.T)
        self.cos = np.clip(front @ self.bg, -1., 1.)
        # Two-pass Gram-Schmidt avoids amplifying cancellation near collinearity.
        orth = front - self.cos[:, None] * self.bg
        orth -= (orth @ self.bg)[:, None] * self.bg
        self.sin = np.linalg.norm(orth, axis=-1)
        self.axis = orth / np.where(self.sin > 1e-12, self.sin, 1.)[:, None]
        self.front = front

    def fits(self, y):
        """Squared projection energies. Last output axis indexes front distance."""
        a = (y @ self.bg)[..., None]
        f = y @ self.front.T
        b = y @ self.axis.T
        interior = (self.sin > 1e-12) & (b > 0) & (a * self.sin > b * self.cos)
        energy = np.maximum(np.maximum(a, 0.) ** 2, np.maximum(f, 0.) ** 2)
        energy = np.where(interior, a*a + b*b, energy)
        return np.maximum(a, 0.) ** 2, energy

    def gain(self, y):
        one, two = self.fits(y)
        return np.maximum(np.max(two - one, axis=-1), 0.)


class BlindBoxBank:
    """One vs two unknown coarse-bin locations, after removal of U.

    All nonzero rays have indices 2..15. A zero ray, or two echoes in
    the same bin, adds no direction. No RGB prior is available here.
    """
    def __init__(self, w):
        self.rays = unit(w.T[2:])
        self.i, self.j = np.triu_indices(len(self.rays), 1)
        self.r = (self.rays[self.i] * self.rays[self.j]).sum(1)

    def score(self, y):
        dots = y @ self.rays.T
        best_one = np.maximum(dots.max(-1), 0.) ** 2
        a, b = dots[..., self.i], dots[..., self.j]
        ab, ba = a - self.r*b, b - self.r*a
        energy = (a*ab + b*ba) / (1. - self.r**2)
        best_interior = np.where((ab > 0) & (ba > 0), energy, 0.).max(-1)
        return np.maximum(best_interior - best_one, 0.)


def shifted_shared_ratio_profile(observations, shifts, z1_grid, z2, ratios, w, sigma):
    """Finite dictionary of shared front/background ratios and known axial shifts.

    observations has shape (batch,frames,16), with independent unit-variance
    whitened noise. Returns per-grid twice-log-likelihood gains over background.
    Histograms are aligned by translating forward templates, without inverse
    interpolation or reconstruction of samples that the firmware discarded.
    """
    if observations.shape[1] != len(shifts) or len(shifts) == 0:
        raise ValueError("Each physical observation needs a known displacement.")
    y = observations @ w.T
    front = pulse(z1_grid[:, None] - shifts, WIDTH, sigma) @ w.T
    background = pulse(z2 - shifts, WIDTH, sigma) @ w.T
    null = np.maximum(np.einsum("nmd,md->nm", y, unit(background)), 0.) ** 2
    best = np.zeros((len(observations), len(z1_grid)))
    for ratio in ratios:
        if ratio < 0:
            raise ValueError("Echo amplitudes must be nonnegative.")
        ray = unit(background[None, :, :] + ratio * front)
        energy = np.zeros_like(best)
        for m in range(len(shifts)):
            dots = y[:, m] @ ray[:, m].T
            energy += np.maximum(dots, 0.) ** 2 - null[:, m, None]
        best = np.maximum(best, energy)
    return best
