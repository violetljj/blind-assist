"""Reproduce restricted HJI grids, symbolic checks and paired Monte Carlo."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import sympy as sp
from barrier import EscapeMargin, contact_time, head_on_distance, weighted_motion
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import beta, norm

OUT = Path(__file__).resolve().parents[5] / "artifacts.local/work/egogeom-omni-v0/math_theory/m4_stochastic_cbf_ics"
R, TAU, BRAKE, SIDE = .45, .65, 2.5, 1.5


def symbolic():
    x, y, vx, vy, tau, eta, q = sp.symbols("x y vx vy tau eta q", real=True)
    a = sp.symbols("a", positive=True)
    z = sp.Matrix([x, y, vx, vy])
    pullback = sp.Matrix([[1,0,tau,0],[0,1,0,tau]])
    h = (pullback*z).dot(pullback*z) - R**2
    assert sp.hessian(h, z) == 2 * pullback.T * pullback
    # 1D stopping halfspace at v<0.
    stop = x - R + tau*vx - vx**2/(2*a)
    assert sp.diff(stop, vx, 2) == -1/a
    B = sp.exp(-eta * stop)
    assert sp.simplify(sp.diff(B, vx, 2) / B -
                       (eta**2 * sp.diff(stop, vx)**2 - eta * sp.diff(stop, vx, 2))) == 0
    t = sp.symbols("t", positive=True)
    distance = q*(tau+t) - a*t**2/2 + sp.sqrt(R**2 - SIDE**2*t**4/4)
    derivative = q-a*t-SIDE**2*t**3/(2*sp.sqrt(R**2-SIDE**2*t**4/4))
    assert sp.simplify(sp.diff(distance, t) - derivative) == 0
    return {"predictor_hessian": "2 L^T L, L=[I,tau I]", "stop_hessian_vv": "-1/a",
            "negative_drift_counterexample_hit_probability": float(2*norm.cdf(-.1))}


def grid_value(axes, horizon=1.8, exact_coast=False):
    """First-order LF backward reach tube, time-to-go; outer constant extrapolation."""
    n = len(axes)
    axes = [a.astype(np.float32) for a in axes]
    fields = np.meshgrid(*axes, indexing="ij", sparse=True)
    pos = np.abs(fields[0]) if n == 2 else np.hypot(fields[0], fields[1])
    # Longitudinal slice uses a halfspace, planar case uses a disk.
    payoff = (fields[0] if n == 2 else pos) - R
    val = np.broadcast_to(payoff, [len(a) for a in axes]).copy()
    delta = [float(a[1]-a[0]) for a in axes]
    velocity = fields[n//2:]
    phases = [("control", horizon - TAU)]
    if not exact_coast:
        phases.append(("coast", TAU))
    for phase, duration in phases:
        speed_caps = [np.max(np.abs(a)) for a in axes[n//2:]]
        controls = [BRAKE, SIDE][:n//2] if phase == "control" else [0.0] * (n//2)
        alpha = speed_caps + controls
        steps = int(np.ceil(duration * sum(a/h for a,h in zip(alpha, delta)) / .8))
        dt = duration / steps
        for _ in range(steps):
            rhs = np.zeros_like(val)
            for i, (h, cap) in enumerate(zip(delta, alpha)):
                if cap == 0:
                    continue
                plus = np.take(val, np.arange(1, len(axes[i])+1), axis=i, mode="clip")
                minus = np.take(val, np.arange(-1, len(axes[i])-1), axis=i, mode="clip")
                center = (plus-minus) / (2*h)
                viscosity = abs(velocity[i]) if exact_coast and i < n//2 else cap
                rhs += (viscosity/(2*h))*(plus - 2*val + minus)
                if i < n//2:
                    rhs += velocity[i] * center
                elif i == n//2:
                    rhs += BRAKE * np.maximum(center, 0)
                else:
                    rhs += SIDE * np.abs(center)
            val = np.minimum(payoff, val + dt*rhs)
    return val


def root_location(x, margin):
    candidates = np.flatnonzero((margin[:-1] <= 0) & (margin[1:] >= 0) & (x[1:] > 0))
    if len(candidates) == 0:
        return None
    i = candidates[-1]
    return float(x[i] - margin[i]*(x[i+1]-x[i]) / max(margin[i+1]-margin[i], 1e-12))


def grids():
    horizon = 1.8
    slicing = []
    for hx, hv in [(0.1, .25), (.05, .125)]:
        axes = [np.arange(-8, 22+hx/2, hx), np.arange(-9, 6+hv/2, hv)]
        V = grid_value(axes, horizon)
        row = []
        for c in [0, 1.5, 6]:
            tbrake = min(c/BRAKE, horizon-TAU)
            exact = R + c*(TAU+tbrake) - BRAKE*tbrake**2/2
            approx = root_location(axes[0], V[:, np.argmin(abs(axes[1]+c))])
            row.append({"closing_mps": c, "analytic_m": exact, "lf_m": approx,
                            "abs_error_m": None if approx is None else abs(approx-exact)})
        slicing.append({"dx_m": hx, "dv_mps": hv, "points": row})
    axes = [np.arange(-8, 20.01, .5), np.arange(-6, 6.01, .5),
            np.arange(-9, 6.01, .75), np.arange(-4, 4.01, .5)]
    V = grid_value(axes, horizon)
    planar = []
    for c in [0, 1.5, 6]:
        # Post-delay horizon exceeds lateral crossing time sqrt(2*R/SIDE).
        cut = V[:, np.argmin(abs(axes[1])), np.argmin(abs(axes[2]+c)), np.argmin(abs(axes[3]))]
        exact = head_on_distance(c)
        approx = root_location(axes[0], cut)
        planar.append({"closing_mps": c, "exact_m": exact, "lf_m": approx,
                           "abs_error_m": None if approx is None else abs(approx-exact)})
    crossing_state = [1.0, 1.0, -.75, -1.0]
    idx = tuple(np.argmin(abs(ax-v)) for ax,v in zip(axes,crossing_state))
    post_interp = RegularGridInterpolator(axes,grid_value(axes,horizon,exact_coast=True),
                                          bounds_error=False,fill_value=np.nan)

    def coast_pullback(ps, vs):
        # Deterministic dead time permits a continuous-time disk-intersection test.
        v2 = np.sum(vs**2,axis=1)
        time_to_nearest = np.clip(-np.sum(ps*vs,axis=1)/np.maximum(v2,1e-20),0.,TAU)
        before = np.linalg.norm(ps + time_to_nearest[:,None]*vs,axis=1)-R
        after = post_interp(np.column_stack([ps+TAU*vs,vs]))
        return np.minimum(before,after)

    for row in planar:
        c = row["closing_mps"]
        x = np.linspace(0.,14.,701)
        margin = coast_pullback(np.column_stack([x,np.zeros_like(x)]),np.tile([-c,0.],(len(x),1)))
        loc = root_location(x,margin)
        row["local_lf_exact_coast_m"] = loc
        row["local_lf_exact_coast_abs_error_m"] = None if loc is None else abs(loc-row["exact_m"])
    cross = np.array(crossing_state)[None,:]
    return {"horizon_s":horizon, "boundary":"clipped outer ghosts", "signed_margin_hji_2d":slicing,
            "signed_margin_hji_4d":{"shape":[len(a) for a in axes], "head_on":planar,
            "crossing_state":crossing_state, "crossing_lf_margin_m":float(V[idx]),
            "crossing_local_lf_exact_coast_m":float(coast_pullback(cross[:,:2],cross[:,2:])[0]),
            "crossing_policy_lower_bound_m":EscapeMargin(horizon=horizon).clearance(
                np.array(crossing_state[:2]),np.array(crossing_state[2:]))}}


def monte_carlo(c, rng, n=10000):
    """Pair policies at identical initial states and Brownian increments, position noise."""
    T, dt, risk = 3.2, .01, .01
    qmax = .015 + .0003*15**2
    tube = 2*qmax*np.sqrt(T*np.log(4/risk))
    d = head_on_distance(c, R+tube)
    # Position diffusion is bounded globally. Start exactly at the analytic trigger.
    pos = np.zeros((2,n,2))
    pos[:,:,0] = d
    vel = np.zeros_like(pos)
    vel[:,:,0] = -c
    hit = np.zeros((2,n), dtype=bool)
    trigger = np.full((2,n), np.inf)
    trigger[0] = 0
    for i in range(round(T/dt)):
        t = i*dt
        trigger[1] = np.where((pos[1,:,0] <= 2.) & np.isinf(trigger[1]), t, trigger[1])
        on = (t + 1e-10 >= trigger+TAU)
        eps = rng.normal(size=(n,2)) * np.sqrt(dt)
        sd = .015 + .0003*np.minimum((pos**2).sum(axis=-1), 15**2)
        pos += vel*dt + sd[:,:,None]*eps
        vel += on[:,:,None] * np.array([BRAKE,SIDE]) * dt
        hit |= (pos**2).sum(axis=-1) <= R**2
    records = []
    for name, hh in zip(["analytic_trigger","fixed_2m"],hit):
        k = int(hh.sum())
        records.append({"policy":name,"grid_hits":k,"paths":n,
                        "hit_probability_upper95_one_sided":float(beta.ppf(.95,k+1,n-k)) if k<n else 1.})
    return {"closing_mps": c, "start_m": d, "noise_tube_m": tube, "time_step_s": dt, "horizon_s": T,
                "qmax_m_per_sqrt_s": qmax, "continuous_model_certified_risk_at_start": risk, "results": records}


def main():
    rng = np.random.default_rng(24112026)
    sym = symbolic()
    phys = {str(c):head_on_distance(c) for c in (0.,1.,1.5,6.)}
    assert phys["6.0"] > R+6*TAU and phys["6.0"] < head_on_distance(6,side=0)
    assert np.isclose(phys["0.0"], R)
    p,v = np.array([2.,0.]),np.array([-6.,0.])
    assert np.isclose(contact_time(p,v), (2.-R)/6)
    pp,vv,cov = weighted_motion(np.array([[1.,2.],[1.,2.]]), np.tile(v,(2,1)), np.ones(2))
    assert np.allclose(pp,[1,2]) and np.allclose(vv,v) and np.allclose(cov, np.eye(2)/2)
    engine = EscapeMargin()
    gaps = []
    for _ in range(300):
        x = rng.uniform([-1.,-3.],[12.,3.])
        vel = rng.uniform([-7.,-3.],[2.,3.])
        gap = engine.exact_clearance(x,vel)-engine.clearance(x,vel)
        assert -1e-8 <= gap <= (np.linalg.norm(vel)+max(engine.max_dv))*engine.dt/2+1e-8
        gaps.append(gap)
    start = perf_counter()
    for _ in range(3000):
        engine.clearance(p,v)
    latency_ms = (perf_counter()-start)*1000/3000
    results = {"symbolic": sym, "head_on_threshold_m": phys, "same_frame_risk_cpu_mean_ms": latency_ms,
                   "random_maneuver_time_grid_checks": 300, "observed_max_grid_conservatism_m": max(gaps),
                   "lf": grids(), "paired_em": [monte_carlo(c,rng) for c in [0., 1.5, 6.]]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"theory_verification.json").write_text(json.dumps(results, indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(results,allow_nan=False))


if __name__ == "__main__":
    main()
