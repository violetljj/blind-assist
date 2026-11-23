"""Run bounded-score algebra, adversarial counterexamples and paired synthetic trials."""
import argparse
import json
import time

import numpy as np
import sympy as sp
from controller import (
    OUT,
    RiskConformalController,
    dr_gradient,
    initialize,
    issue,
    learn,
)
from numba import njit


@njit
def run_stress(scores, masks, alpha, shifts, delay=5, observe_prob=0.7):
    horizon = len(scores)
    slots = delay + 1
    prior, rate, step, mass, advice = initialize(horizon, slots, observe_prob)
    played = np.empty(horizon)
    fixed_played = np.empty(horizon)
    fixed_q = 0.0
    errors = np.zeros((2, horizon), np.int64)
    for t in range(horizon):
        # Exact same before-decision and after-decision ordering as live replay.
        q = issue(t, prior, rate, mass, advice)
        played[t] = q
        fixed_played[t] = fixed_q
        errors[0, t] = scores[t] > q
        errors[1, t] = scores[t] > fixed_q
        s = t - delay
        if s >= 0:
            g = dr_gradient(played[s], alpha, observe_prob, alpha, masks[s], scores[s])
            learn(s, played[s], g, mass, advice, rate, step)
            if masks[s]:
                fixed_q += 0.05 * (int(scores[s] > fixed_played[s]) - alpha)
    worst = np.zeros(2)
    for s in shifts:
        for k in range(2):
            worst[k] = max(worst[k], errors[k, s:s + 20].sum() / 20)
    return errors.sum(axis=1) / horizon, worst


def algebra():
    e, h, m, p, a = sp.symbols("e h m p a", positive=True)
    # Bernoulli truth and observation; e may depend on observed predecision history.
    dr_mean = sum(
        po * py * (p - a + o * (y - p) / h)
        for o, po in ((0, 1 - e), (1, e)) for y, py in ((0, 1 - m), (1, m))
    )
    dr_err2 = sum(
        po * py * ((o / h - 1) * (y - p)) ** 2
        for o, po in ((0, 1 - e), (1, e)) for y, py in ((0, 1 - m), (1, m))
    )
    assert sp.simplify(dr_mean - (m - a) - (e / h - 1) * (m - p)) == 0
    assert sp.simplify(dr_err2 - (1 - 2 * e / h + e / h ** 2) * (m * (1 - m) + (m - p)**2)) == 0
    return {"double_robust_bias": str(sp.factor(dr_mean - (m - a))), "conditional_second_moment": "verified by exact summation"}


def fixed_point_check():
    gen = np.random.default_rng(431)
    prior, eta, step, mass, advice = initialize(10000, 6, 0.7)
    max_error = 0.0
    min_multiplier = 1.0
    for t in range(2400):
        q = issue(t, prior, eta, mass, advice)
        j = t % 6
        w = mass[j] * eta
        max_error = max(max_error, abs(np.dot(w, advice[j] - q)) / np.sum(w))
        assert -2.0000001 <= q <= 2.0000001
        s = gen.uniform(-1, 1)
        g = dr_gradient(q, .05, .7, .05, gen.random() < .7, s)
        mult = 1.0 + eta * g * (advice[j] - q)
        min_multiplier = min(min_multiplier, float(mult.min()))
        assert g * (advice[j, -2] - q) >= g - 1e-12
        assert g * (advice[j, -1] - q) >= -g - 1e-12
        learn(t, q, g, mass, advice, eta, step)
    assert max_error < 1e-12 and min_multiplier >= .5
    return {"max_mass_balance_residual": float(max_error), "min_prod_factor": min_multiplier}


def latency_and_feedback_check():
    controller = RiskConformalController(6000, max_delay=5)
    frames = []
    start = time.perf_counter()
    for t in range(6000):
        token = controller.predict(2 + t % 3, .5, propensity=.7)
        frames.append(token)
        if t >= 5:
            controller.reveal(frames[t-5], True, 2 if t % 10 < 7 else None)
    us = (time.perf_counter() - start) * 1e6 / 6000
    for token in frames[-5:]:
        controller.reveal(token, True, 2)
    assert controller.resolved == 6000 and controller.audit_upper() >= 0
    try:
        controller.reveal(frames[-1], True, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate feedback accepted")
    # First five following outputs cannot be influenced by different hidden outcomes.
    inv = []
    for y in (0., 12.):
        c = RiskConformalController(20)
        seq = []
        for t in range(12):
            seq.append(c.predict(3, .4, propensity=.7))
            if t >= 5:
                c.reveal(seq[t-5], True, y)
        inv.append(np.array([p.q for p in seq]))
    assert np.array_equal(inv[0][:6], inv[1][:6])
    assert np.any(inv[0][6:] != inv[1][6:])
    return {"warm_host_mean_us_predict_and_due_feedback": round(us, 2), "below_50us_on_this_run": bool(us < 50)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("development", "fresh"), default="development")
    args = parser.parse_args()
    seed_start = 91000 if args.panel == "development" else 1731000
    result = {"algebra": algebra(), "specialist_invariants": fixed_point_check(), "latency": latency_and_feedback_check()}
    horizon = 10000
    shifts = np.array([1700, 3300, 5000, 6700, 8300])
    means = np.repeat([0., 1.8, -0.8, 1.5, -0.4, 2.0], np.diff(np.r_[0, shifts, horizon]))
    result["design"] = {"T": horizon, "replicates_per_alpha": 100, "shift_positions": shifts.tolist(), "delay": 5, "audit_probability": .7, "base_fixed_aci_step": .05, "measure": "violation of model lower range; paired worst 20-step fraction per trajectory"}
    result["design"].update(panel=args.panel, seed_start=seed_start)
    result["stress"] = {}
    for alpha in (.02, .05, .1):
        runs = []
        for seed in range(100):
            rng = np.random.default_rng(seed_start + seed)
            raw = means + rng.normal(size=horizon)
            rates, worst = run_stress(raw / (1.0 + abs(raw)), rng.random(horizon) < .7, alpha, shifts)
            runs.append([*rates, *worst])
        r = np.array(runs)
        reduction = 1 - r[:, 2].mean() / r[:, 3].mean()
        result["stress"][str(alpha)] = {
            "sa_mean": float(r[:, 0].mean()), "fixed_mean": float(r[:, 1].mean()),
            "sa_range": [float(r[:, 0].min()), float(r[:, 0].max())],
            "sa_global_within_0.008_count": int(np.sum(abs(r[:, 0] - alpha) <= .008)),
            "fixed_global_within_0.008_count": int(np.sum(abs(r[:, 1] - alpha) <= .008)),
            "mean_of_worst_postshift_20step_rate_sa": float(r[:, 2].mean()),
            "mean_of_worst_postshift_20step_rate_fixed": float(r[:, 3].mean()),
            "relative_reduction": float(reduction), "passes_35pct": bool(reduction >= .35),
            "sa_absolute_worst": float(r[:, 2].max()), "fixed_absolute_worst": float(r[:, 3].max()),
        }
    result["indistinguishable_mnar_worlds"] = {
        "both_observed_O1E1": .25, "both_observed_O1E0": .25, "both_unobserved": .5,
        "world_a_true_risk": .25, "world_b_true_risk": .75}
    OUT.mkdir(parents=True, exist_ok=True)
    filename = "theory_verification.json" if args.panel == "development" else "theory_verification_fresh.json"
    (OUT / filename).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
