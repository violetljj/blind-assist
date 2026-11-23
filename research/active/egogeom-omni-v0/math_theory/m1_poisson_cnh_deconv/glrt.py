"""Continuous multistart likelihood fits. Report numerical estimates, not global certificates."""

import numpy as np
from poisson_cnh import (
    DC,
    XTALK,
    experiment,
    finite_null_threshold,
    kernel,
    saturated_stat,
)
from scipy.optimize import minimize
from scipy.special import xlogy


class PoissonGLRT:
    def __init__(self, low=0.15, high=4.0, sigma=0.10):
        self.low, self.high, self.sigma = low, high, sigma
        self.grid = np.linspace(low, high, 64)
        self.bank = kernel(self.grid, sigma)

    def objectives(self, y, alt):
        def loss(theta):
            if alt:
                front, frac, weak, extra, xt, dc = theta
                back = front + frac * (self.high - front)
                f, g = kernel([front, back], self.sigma)
                df, dg = kernel([front, back], self.sigma, derivative=1)
                pred = weak * f + (weak + extra) * g + xt * XTALK + dc
                jac = np.stack(
                    [
                        weak * df + (weak + extra) * (1 - frac) * dg,
                        (weak + extra) * (self.high - front) * dg,
                        f + g,
                        g,
                        XTALK,
                        DC,
                    ]
                )
            else:
                back, amp, xt, dc = theta
                g = kernel(back, self.sigma)
                pred = amp * g + xt * XTALK + dc
                jac = np.stack(
                    [amp * kernel(back, self.sigma, derivative=1), g, XTALK, DC]
                )
            pred = pred.clip(1e-15)
            return (pred - xlogy(y, pred)).sum(), jac @ (1 - y / pred)

        return loss

    def estimate(self, counts):
        total = max(1, counts.sum())
        y = counts / total
        proxy = (y - np.median(y)).clip(0)
        proxy[:2] = 0
        peaks = self.bank @ proxy

        def fit(obj, starts, bounds):
            results = [
                minimize(
                    obj,
                    start,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 100, "ftol": 2e-12},
                )
                for start in starts
            ]
            return min(results, key=lambda fit: fit.fun)

        lo, hi = self.low, self.high
        m = (0, None)
        floor = (1e-12, None)
        back0 = self.grid[peaks.argmax()]
        second = self.grid[
            np.argmax(np.where(abs(self.grid - back0) > 0.23, peaks, -np.inf))
        ]
        null = fit(
            self.objectives(y, False),
            [[b, max(proxy.max(), 0.01), y[0], np.median(y)] for b in [back0, second]],
            [(lo, hi), m, m, floor],
        )
        back, amp, xt, b = null.x
        q = y / (amp * kernel(back, self.sigma) + xt * XTALK + b) - 1
        allowed = self.grid < back - 1e-5
        if not allowed.any():
            return 0.0
        fronts = self.grid[allowed]
        scores = self.bank[allowed] @ q
        a = fronts[scores.argmax()]
        c = fronts[np.argmax(np.where(abs(fronts - a) > 0.20, scores, -np.inf))]
        starts = [
            [z, (back - z) / (hi - z), min(0.005, amp / 2), max(amp - 0.005, 0), xt, b]
            for z in (a, c)
        ]
        alt = fit(self.objectives(y, True), starts, [(lo, hi), (0, 1), m, m, m, floor])
        return max(0.0, 2 * total * (null.fun - alt.fun))


def profile_check(trials=2000):
    rng = np.random.default_rng(14905)
    mean, _ = experiment(1.0, 1.3, 500, 0, 0.74)
    cutoff = finite_null_threshold(0.05)
    solver = PoissonGLRT()
    draws = rng.poisson(mean, size=(trials, 16))
    t = np.array([solver.estimate(y) for y in draws])
    return {
        "trials": trials,
        "null_z2_m": 2.3,
        "estimated_profile_T_p95": float(np.quantile(t, 0.95)),
        "estimated_profile_T_above_finite_cut": int((t > cutoff).sum()),
        "guaranteed_GLRT_exceedance_upper_count": int(
            (saturated_stat(draws, mean) > cutoff).sum()
        ),
        "profile_estimates_above_true_null_saturated_bound": int(
            (t > saturated_stat(draws, mean) + 1e-8).sum()
        ),
    }
