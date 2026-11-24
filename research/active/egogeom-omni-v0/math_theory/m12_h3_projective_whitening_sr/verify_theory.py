"""Symbolic quotient, exact box-sampling nulls, and a calibrated smooth-IRF experiment."""

import numpy as np
import sympy as sp
from scipy.optimize import nnls
from scipy.stats import norm, rankdata
from threadpoolctl import threadpool_limits

from projective_h3 import (BINS, HERE, WIDTH, EchoBank, ar_cov, contrast, features,
                           pulse, save, sha, shifted_shared_ratio_profile, torch_features)


def verify_projection():
    # Tridiagonal positive precision gives a non-Euclidean projector in exact arithmetic.
    u = sp.ones(BINS, 1).row_join(sp.eye(BINS)[:, :2])
    d = sp.eye(BINS) * 3
    for i in range(15):
        d[i+1, i] = d[i, i+1] = 1
    p = sp.eye(BINS) - u * (u.T*d*u).inv() * u.T*d
    a, b, c, g = sp.symbols("a b c g", real=True)
    h = sp.Matrix(sp.symbols("h:16"))
    assert p*u == sp.zeros(16, 3) and p.T*d == d*p and p*p == p
    assert (p*(g*h+u*sp.Matrix([a, b, c]))-g*p*h).expand() == sp.zeros(16, 1)
    assert p.rank() == 13
    w = contrast(ar_cov())
    assert np.allclose(w @ ar_cov() @ w.T, np.eye(13))
    signal = np.r_[np.ones(1), np.zeros(12)]
    rng = np.random.default_rng(160)
    eps = rng.standard_normal((60000, 13))
    probs = {}
    for gain in (.2, 1., 5.):
        means = (eps + gain * signal)[:, 0]
        observed = float((means > 0).mean())
        expected = float(norm.cdf(gain))
        assert abs(observed - expected) < .008
        probs[str(gain)] = {"positive_hemisphere": observed, "exact": expected}
    return {"rank": 13, "dimension_of_regular_orbit_quotient": 12,
            "fixed_noise_gain_counterexample": probs}


def box_sampling():
    n, native_per_bin = 128, 8
    s = np.zeros((16, n), dtype=int)
    for j in range(16):
        s[j, 8*j:8*j+8] = 1
    designs = {m: np.vstack([np.roll(s, k, axis=1) for k in range(m)]) for m in (1, 3, 5, 7, 8)}
    summary = {}
    for count, design in designs.items():
        sv = np.linalg.svd(design, compute_uv=False)
        rank = int((sv > 1e-10).sum())
        unique = len(np.unique(design.T, axis=0))
        if count < native_per_bin:
            assert unique < n
        summary[count] = {"rank": rank, "distinct_single_spike_columns": unique,
                          "positive_singular_min": float(sv[rank-1]), "singular_max": float(sv[0])}
    assert summary[1]["rank"] == 16 and summary[8]["rank"] == 121
    x, xp = np.zeros(n), np.zeros(n)
    x[::8], xp[1::8] = 1, 1
    assert np.array_equal(designs[8] @ x, designs[8] @ xp)
    # Five uniform coarse offsets cannot distinguish two continuous positions.
    shifts = np.arange(5) * WIDTH / 5
    z = 4 * WIDTH
    bins = lambda z0: np.floor((z0 - shifts)/WIDTH).astype(int)
    z1, z2 = z + WIDTH * .021, z + WIDTH * .179
    assert np.array_equal(bins(z1), bins(z2))
    return {"circular_native_grid": summary, "minimum_null_support_all_shifts": 32,
            "continuous_indistinguishable_pair_m": [z1, z2],
            "five_uniform_phases_minimax_abs_lower_bound_m": WIDTH / 10,
            "uniform_cell_midpoint_worst_error_m": {m: WIDTH/(2*m) for m in (1, 3, 5, 8)}}


def local_bound(z1, z2, shifts, w, amp, noise, sigma):
    info = 0.
    for shift in shifts:
        t = (np.arange(17) * WIDTH - (z1 - shift)) / sigma
        deriv = -np.diff(norm.pdf(t)) / sigma
        jac = amp * (w @ deriv)
        nuisance = np.stack([w @ pulse(z-shift, WIDTH, sigma) for z in (z1, z2)], axis=1)
        residual = jac - nuisance @ np.linalg.lstsq(nuisance, jac, rcond=1e-12)[0]
        info += float(residual @ residual) / noise**2
    return 1./np.sqrt(info)


def auc(positive, negative):
    n, m = len(positive), len(negative)
    return float((rankdata(np.r_[positive, negative])[:n].sum() - n*(n+1)/2) / (n*m))


def calibrated_experiment():
    rng = np.random.default_rng(251124)
    r = ar_cov()
    w = contrast(r)
    z2, sigma, noise = 2.7, .065, .083
    # Geometric grid / IRF chosen before sampling. Physical echoes stay beyond bin1.
    z_grid = np.linspace(.84, 2.4, 313)
    shifts = np.arange(5) * WIDTH / 5
    banks = [[EchoBank(w0, z2-s, z_grid-s, sigma) for s in shifts]
             for w0 in (w, contrast(np.eye(16)))]
    n = 500
    z1 = rng.uniform(.9, 2.25, n)
    gains = rng.uniform(.2, 5., (n, len(shifts)))
    eps = rng.multivariate_normal(np.zeros(16), noise**2*r, (2, n, len(shifts)))
    scores = [[], []]
    shared_scores = []
    recovered = []
    raw_argmax_mae = None
    for present in (0, 1):
        clean = (1-.2*present)*pulse(z2-shifts, WIDTH, sigma) + (
            present*.33*pulse(z1[:, None]-shifts, WIDTH, sigma))
        clean = np.broadcast_to(clean, (n, 5, 16)).copy()
        # 82% of positive clean energy allocated to bin zero before DC/noise.
        clean[..., 0] += .82/.18 * clean.sum(-1)
        obs = gains[..., None]*(clean+rng.uniform(-.2, .2, (n, 5, 1)))+eps[present]
        shared = shifted_shared_ratio_profile(obs, shifts, z_grid, z2,
                         np.geomspace(.025, 2., 18), w, sigma)
        shared_scores.append(shared.max(-1))
        if present:
            raw_argmax_mae = float(np.mean(abs((obs[:, 0].argmax(-1)+.5)*WIDTH-z1)))
            shared_mae = float(np.mean(abs(z_grid[shared.argmax(-1)]-z1)))
        for j, w0 in enumerate((w, contrast(np.eye(16)))):
            y = obs @ w0.T
            improvements = np.zeros((n, len(z_grid)))
            for i, bank in enumerate(banks[j]):
                one, two = bank.fits(y[:, i])
                improvements += (two - one) / noise**2
                if present and j == 0 and i in (0, 2, 4):
                    recovered.append({"frames": i+1, "mae_m": float(abs(z_grid[improvements.argmax(-1)]-z1).mean())})
            if j==0:
                assert np.all(shared.max(-1)/noise**2 <= improvements.max(-1)+1e-7)
            scores[j].append(improvements.max(-1))
    # Independent NNLS comparator on one background with non-collinear atoms.
    bank = banks[0][0]
    y = rng.normal(size=13)
    _, analytic = bank.fits(y)
    manual = []
    for f in bank.front:
        design = np.stack((f, bank.bg), axis=1)
        _, residual = nnls(design, y)
        manual.append(y @ y-residual**2)
    assert np.max(abs(analytic - manual)) < 1e-7
    crlbs = {m: local_bound(1.35, z2, np.arange(m)*WIDTH/m, w, .33, noise, sigma)
             for m in (1, 3, 5, 8)}
    return {"per_class": n, "front_amplitude": .33, "background_range_m": z2,
            "fixed_noise_sd": noise, "assumed_known_IRF_sigma_m": sigma,
            "raw_argmax_single_mae_m": raw_argmax_mae, "positive_fixed_denominator": recovered,
            "shared_ratio_5f_mae_m":shared_mae, "shared_ratio_5f_auc":auc(shared_scores[1], shared_scores[0]),
            "whitened_5f_auc": auc(scores[0][1], scores[0][0]),
            "identity_cov_5f_auc": auc(scores[1][1], scores[1][0]),
            "conditional_crlb_SD_m_at_1p35_unit_gain_every_frame": crlbs}


def check_torch():
    import torch
    rng = np.random.default_rng(89)
    h = rng.normal(size=(3, 64, 16))
    mask = np.broadcast_to(np.array([1,0,1])[:,None], (3, 64)).copy()
    w = contrast(ar_cov())
    ht = torch.tensor(h, requires_grad=True)
    y, q = torch_features(ht, torch.tensor(w), torch.tensor(mask))
    yn, qn = features(h, w, mask)
    assert np.allclose(y.detach().numpy(), yn) and np.allclose(q.detach().numpy(), qn)
    assert torch.count_nonzero(q[1]).item() == 0 and torch.count_nonzero(y[1]).item() == 0
    q.sum().backward()
    assert torch.isfinite(ht.grad).all() and torch.count_nonzero(ht.grad[1]).item() == 0
    return {"numpy_torch_agree": True, "absent_output_and_gradient_exact_zero": True}


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        report = {"symbolic": verify_projection(), "sampling": box_sampling(),
                  "synthetic": calibrated_experiment(), "physical_layer": check_torch(),
                  "code_sha256": {p: sha(HERE/p) for p in ("verify_theory.py","projective_h3.py")}}
    save("theory_verification.json", report)
    print(report)
