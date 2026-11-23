"""Independent symbolic, local-information, Poisson-null and SFW stress checks."""

import json
import os
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import sympy as sp
from poisson_cnh import (
    DC,
    OUT,
    XTALK,
    SlidingPoisson,
    experiment,
    finite_null_threshold,
    front_from_atoms,
    front_information,
    kernel,
    residual,
    saturated_stat,
)
from scipy.linalg import null_space
from scipy.optimize import brentq
from scipy.stats import norm
from threadpoolctl import threadpool_limits


def symbolic():
    z, u, v, s = sp.symbols("z u v s", positive=True)
    cdf = lambda x: (1 + sp.erf((x - z) / (sp.sqrt(2) * s))) / 2
    pdf = lambda x: sp.exp(-((x - z) ** 2) / (2 * s * s)) / (sp.sqrt(2 * sp.pi) * s)
    assert sp.simplify(sp.diff(cdf(v) - cdf(u), z) - pdf(u) + pdf(v)) == 0
    x, y, a = sp.symbols("x y a", nonzero=True)
    g = sp.Matrix([[x, y], [y, a]])
    assert sp.simplify(g.inv()[0, 0] - 1 / (x - y * y / a)) == 0
    d = sp.Symbol("d")
    f = sp.symbols("f:5")
    p = sum(f[i] * d**i / sp.factorial(i) for i in range(5))
    hermite = sp.expand(f[1] - 2 * (p - f[0]) / d + sp.diff(p, d))
    assert hermite.coeff(d, 2) == f[3] / 6
    return {
        "integrated_irf_jacobian": True,
        "block_schur_scalar_identity": True,
        "hermite_residual": str(hermite),
    }


def information_checks():
    out = []
    for z1 in (0.45, 1.04):
        for ratio in (0.15, 0.35, 1):
            sigma = 0.14
            mean0, _ = experiment(z1, 0, 1000, ratio, 0.74, sigma=sigma)
            a1 = 1000 * ratio / (1 + ratio)
            whiten = np.sqrt(mean0)
            d0, d1, d2, d3 = (kernel(z1, sigma, derivative=i) for i in range(4))
            base = np.stack([DC, XTALK, d0, d1, d2], axis=1)
            leading = 36 / (
                a1**2
                * np.linalg.norm(residual(d3 / whiten, base / whiten[:, None])) ** 2
            )
            gaps = np.array([0.002, 0.001, 0.0005, 0.00025])
            front, spacing = [], []
            for gap in gaps:
                mean, j = experiment(z1, gap, 1000, ratio, 0.74, sigma=sigma)
                inf, _ = front_information(mean, j)
                front.append(1 / inf)
                # Direct projection avoids subtracting strongly correlated position variances.
                delta = (j[:, 1] - j[:, 0]) / 2 / np.sqrt(mean)
                nuisance = np.column_stack([j[:, 0] + j[:, 1], j[:, 2:]]) / np.sqrt(
                    mean[:, None]
                )
                spacing.append(1 / np.linalg.norm(residual(delta, nuisance)) ** 2)
            out.append(
                {
                    "z1_m": z1,
                    "ratio": ratio,
                    "front_loglog_slope": float(
                        np.polyfit(np.log(gaps), np.log(front), 1)[0]
                    ),
                    "gap_loglog_slope": float(
                        np.polyfit(np.log(gaps), np.log(spacing), 1)[0]
                    ),
                    "front_coeff_ratio_at_smallest": front[-1]
                    * gaps[-1] ** 4
                    / leading,
                }
            )
    return out


def null_and_information_grid(rng):
    rows = []
    limit = finite_null_threshold(0.05)
    for photons in (50, 200, 1000, 5000):
        for rho in (0.05, 0.50, 0.74, 0.85):
            for r in (0.05, 0.15, 0.35):
                for gap in (0.15, 0.45, 0.85, 1.50):
                    mean, jac = experiment(1.04, gap, photons, r, rho)
                    inf, score = front_information(mean, jac)
                    y = rng.poisson(mean, (2000, 16))
                    oracle = ((y - mean) / np.sqrt(mean)) @ score / np.sqrt(inf)
                    null, _ = experiment(1.04, gap, photons, 0, rho)
                    upper = saturated_stat(rng.poisson(null, (2000, 16)), null)
                    rows.append(
                        {
                            "N": photons,
                            "rho": rho,
                            "ratio": r,
                            "gap_m": gap,
                            "front_crsd_m": 1 / np.sqrt(inf),
                            "oracle_rmse_over_crsd": float(np.sqrt(np.mean(oracle**2))),
                            "saturated_null_exceedances": int((upper >= limit).sum()),
                        }
                    )
    return rows, limit


def rice(rng):
    # Null nuisance: amplitude, background location, xtalk and background.
    mean, j = experiment(1.0, 1.3, 500, 0, 0.74)
    whiten = np.sqrt(mean)
    null = j[:, [1, 3, 4, 5]] / whiten[:, None]
    zgrid = np.linspace(0.35, 2.05, 513)
    complement = null_space(null.T)
    directions = (kernel(zgrid) / whiten) @ complement
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    arc = np.arccos(
        np.clip((directions[1:] * directions[:-1]).sum(axis=1), -1, 1)
    ).sum()
    level = brentq(
        lambda u: norm.sf(u) + arc * np.exp(-u * u / 2) / (2 * np.pi) - 0.05, 1, 6
    )
    trials = 2000
    gaussian = rng.normal(size=(trials, directions.shape[1]))
    return {
        "scan_interval_m": [0.35, 2.05],
        "arc_length_grid_approx": float(arc),
        "asymptotic_T_threshold": float(level**2),
        "gaussian_score_scan_exceedance": float(
            ((gaussian @ directions.T).max(axis=1) > level).mean()
        ),
        "trials": trials,
    }


def sfw_stress(rng, trials):
    z1, gap, photons, r, rho = 1.04, 0.85, 200, 0.15, 0.74
    mean, jac = experiment(z1, gap, photons, r, rho)
    crsd = 1 / np.sqrt(front_information(mean, jac)[0])
    solver = SlidingPoisson()
    errs = []
    cert = None
    for i, y in enumerate(rng.poisson(mean, (trials, 16))):
        z, a, diag = solver.fit(y, dense_certificate=(i == 0))
        if i == 0:
            cert = diag
        pred, _ = front_from_atoms(z, a, y)
        errs.append(pred - z1)
    errs = np.array(errs)
    return {
        "N": photons,
        "ratio": r,
        "rho": rho,
        "SBR": 0.5,
        "gap_m": gap,
        "front_m": z1,
        "trials": trials,
        "mae_m": float(abs(errs).mean()),
        "rmse_m": float(np.sqrt((errs**2).mean())),
        "bias_m": float(errs.mean()),
        "crsd_m": crsd,
        "rmse_over_crsd": float(np.sqrt((errs**2).mean()) / crsd),
        "certificate_first_draw": cert,
    }


def layer_check():
    import torch
    from unrolled import FiveStepCNH

    torch.set_num_threads(1)
    torch.manual_seed(1905)
    x = torch.rand(1, 16, 8, 8, requires_grad=True) * 350
    layer = FiveStepCNH().eval()
    out = layer(x)
    (grad,) = torch.autograd.grad(out.square().mean(), x)
    assert torch.isfinite(grad).all() and grad.abs().max() > 0
    with torch.inference_mode():
        for _ in range(20):
            layer(x)
        samples = []
        for _ in range(11):
            start = time.perf_counter()
            for _ in range(200):
                layer(x)
            samples.append((time.perf_counter() - start) * 5)
    return {
        "cpu_single_thread_ms_per_64_zones_p50_of_batch_means": float(
            np.median(samples)
        ),
        "p90_ms": float(np.quantile(samples, 0.9)),
        "batches": 11,
        "calls_per_batch": 200,
        "output_shape": list(out.shape),
        "finite_input_gradient": True,
    }


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--sfw-trials", type=int, default=2000)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(20261124)
        result = {
            "symbolic": symbolic(),
            "small_gap": information_checks(),
            "rice": rice(rng),
        }
        result["grid"], result["finite_T_threshold_5pct"] = null_and_information_grid(
            rng
        )
        from glrt import profile_check

        result["continuous_GLRT"] = profile_check()
        result["unrolled"] = layer_check()
        result["sfw_stress"] = sfw_stress(rng, args.sfw_trials)
    (OUT / "theory_verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print({k: v for k, v in result.items() if k != "grid"}, flush=True)


if __name__ == "__main__":
    main()
