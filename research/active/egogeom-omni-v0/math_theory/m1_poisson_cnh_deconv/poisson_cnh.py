"""16-bin counting model; distances in metres, amplitudes in expected photons."""

from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import ndtr, xlogy

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parents[1]
WORK = ROUTE.parents[2] / "artifacts.local" / "work" / ROUTE.name
OUT = WORK / "math_theory" / HERE.name
XTALK = np.r_[0.94, 0.06, np.zeros(14)]
DC = np.ones(16)


def kernel(z, sigma=0.10, width=0.25, derivative=0):
    t = (np.arange(17) * width - np.asarray(z)[..., None]) / sigma
    if derivative == 0:
        # Use survival tails on the right to avoid the 1 - 1 cancellation.
        return np.where(t[..., :-1] > 0, -np.diff(ndtr(-t)), np.diff(ndtr(t)))
    pdf = np.exp(-t * t / 2) / np.sqrt(2 * np.pi)
    hermite = [1, t, t * t - 1][derivative - 1]
    return -np.diff(pdf * hermite) / sigma**derivative


def experiment(z1, gap, photons, ratio, rho, sbr=0.5, sigma=0.10):
    a2 = photons / (1 + ratio)
    a1 = ratio * a2
    b = photons / (16 * sbr)
    xt = rho / (1 - rho) * (photons + 16 * b)
    z2 = z1 + gap
    f1, f2 = kernel([z1, z2], sigma)
    d1, d2 = kernel([z1, z2], sigma, derivative=1)
    mean = b + xt * XTALK + a1 * f1 + a2 * f2
    jac = np.stack([a1 * d1, a2 * d2, f1, f2, XTALK, DC], axis=1)
    return mean, jac


def residual(vector, design):
    return vector - design @ np.linalg.lstsq(design, vector, rcond=None)[0]


def front_information(mean, jac):
    weighted = jac / np.sqrt(mean[:, None])
    efficient = residual(weighted[:, 0], weighted[:, 1:])
    return efficient @ efficient, efficient


def finite_null_threshold(alpha, bins=16):
    # Markov applied to exp(t * sum of independent Poisson divergences).
    def objective(t):
        return 2 * (bins * np.log((1 + t) / (1 - t)) - np.log(alpha)) / t

    return minimize_scalar(objective, bounds=(1e-8, 1 - 1e-8), method="bounded").fun


def saturated_stat(y, mean):
    return 2 * (xlogy(y, y / mean) - y + mean).sum(axis=-1)


class SlidingPoisson:
    """Correct nonnegative amplitudes and slide locations jointly after atom insertion."""

    def __init__(self, penalty=0.025, lo=0.15, hi=4.0, sigma=0.10):
        self.lo, self.hi, self.sigma = lo, hi, sigma
        self.penalty = penalty
        self.grid = np.linspace(lo, hi, 309)
        self.bank = kernel(self.grid, sigma)

    def _correct(self, y, mass, positions, kind):
        n = len(positions)

        def objective(t):
            a = t[:n]
            ph = kernel(t[n + 2 :], self.sigma)
            pred = (a @ ph + t[n] * XTALK + t[n + 1]).clip(1e-14)
            if kind == "poisson":
                err = 1 - y / pred
                val = (pred - xlogy(y, pred)).sum() + self.penalty * a.sum()
                pen = self.penalty
            else:
                err = pred - y
                val = 0.5 * (err @ err)
                pen = 0
            dz = kernel(t[n + 2 :], self.sigma, derivative=1) @ err
            grad = np.r_[ph @ err + pen, XTALK @ err, err.sum(), a * dz]
            return val, grad

        bounds = [(0, None)] * (n + 1) + [(1e-12, None)] + [(self.lo, self.hi)] * n
        fit = minimize(
            objective,
            np.r_[mass, positions],
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 180, "ftol": 5e-13, "gtol": 1e-7},
        )
        return fit.x[: n + 2], fit.x[n + 2 :]

    def fit(self, counts, max_atoms=8, dense_certificate=False):
        exposure = max(1.0, counts.sum())
        y = counts / exposure
        pos = np.array([])
        mass, pos = self._correct(y, np.array([y[0], np.median(y)]), pos, "poisson")
        for _ in range(max_atoms):
            q = (
                y / (mass[:-2] @ kernel(pos, self.sigma) + mass[-2] * XTALK + mass[-1])
                - 1
            )
            values = self.bank @ q
            i = values.argmax()
            refine = minimize_scalar(
                lambda z, q=q: -(kernel(z, self.sigma) @ q),
                bounds=(
                    self.grid[max(i - 1, 0)],
                    self.grid[min(i + 1, len(values) - 1)],
                ),
                method="bounded",
            )
            val, loc = max((values[i], self.grid[i]), (-refine.fun, refine.x))
            if val <= self.penalty + 1e-5:
                break
            mass, pos = self._correct(
                y, np.r_[mass[:-2], 0.01, mass[-2:]], np.r_[pos, loc], "poisson"
            )
        order = np.argsort(pos)
        a, pos = mass[:-2][order], pos[order]
        keep = a > 1e-8
        a, pos = a[keep], pos[keep]
        q = y / (a @ kernel(pos, self.sigma) + mass[-2] * XTALK + mass[-1]) - 1
        cert = {}
        if dense_certificate:
            grid = np.linspace(self.lo, self.hi, 4097)
            curvature = (
                2 * np.exp(-0.5) * abs(q).sum() / (np.sqrt(2 * np.pi) * self.sigma**2)
            )
            max_corr = (kernel(grid, self.sigma) @ q).max()
            cert = {
                "upper_atom_violation": float(
                    max_corr + curvature * (grid[1] - grid[0]) ** 2 / 8 - self.penalty
                ),
                "nuisance_scores": [float(q @ XTALK), float(q.sum())],
                "max_support_slack": float(
                    abs(kernel(pos, self.sigma) @ q - self.penalty).max()
                )
                if len(pos)
                else None,
            }
        return pos, a * exposure, cert

    def l2(self, counts):
        y = counts / max(counts.sum(), 1)
        # Initialize each baseline twice, once by tail peaks and once by a
        # residual matched filter; do not seed with true foreground/background.
        tail = y.copy()
        tail[:2] = 0
        pos2 = self.grid[np.argmax(self.bank @ tail)]
        rest = (tail - tail.max() * kernel(pos2, self.sigma)).clip(0)
        pos1 = self.grid[np.argmax(self.bank @ rest)]
        starts = [[pos1, pos2], [max(self.lo, pos2 - 0.75), pos2]]
        trials = []
        for start in starts:
            m, z = self._correct(y, [0.05, 0.12, y[0], np.median(y)], start, "l2")
            fit = m[:2] @ kernel(z, self.sigma) + m[2] * XTALK + m[3]
            trials.append((np.sum((fit - y) ** 2), float(z.min())))
        return min(trials)[1]


def front_from_atoms(z, a, counts):
    # Strongest two peaks; retain the nearer of the two, including single-
    # peak failures in the error denominator. Floor at 8% filters speckle.
    if len(a) == 0:
        return (1 + np.argmax(counts[1:])) * 0.25, 0.0
    groups = np.split(np.arange(len(z)), np.where(np.diff(z) > 0.12)[0] + 1)
    z = np.array([np.average(z[ix], weights=a[ix]) for ix in groups])
    a = np.array([a[ix].sum() for ix in groups])
    keep = np.flatnonzero(a >= 0.08 * a.max())
    big = keep[np.argsort(a[keep])[-2:]]
    if len(big) < 2:
        return float(z[big[0]]), 0.0
    i, j = big[np.argsort(z[big])]
    return float(z[i]), float(
        min(a[i], a[j]) / max(a[i], a[j]) * min((z[j] - z[i]) / 0.45, 1)
    )
