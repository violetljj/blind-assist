"""Symbolic adjoints, exact counterexamples, and a fixed synthetic shallow-obstacle panel."""
import os
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import sympy as sp
from scipy.ndimage import minimum_filter
from tgv import CELL, SHAPE, STEP, STEPS, BETA, GAMMA, solve, operator, adjoint

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[4] / "artifacts.local/work/egogeom-omni-v0/math_theory" / HERE.name


def symbolic():
    # Finite 3x3 free-edge operator, symbolic entries; E is in packed Frobenius coordinates.
    d = sp.Matrix([[-1, 1, 0], [0,-1,1], [0,0,0]])
    dx, dy = sp.kronecker_product(sp.eye(3), d), sp.kronecker_product(d, sp.eye(3))
    v = sp.Matrix(sp.symbols("v:18"))
    q = sp.Matrix(sp.symbols("q:27"))
    E = dx.row_join(sp.zeros(9)).col_join(sp.zeros(9).row_join(dy)).col_join(
        dy.row_join(dx) / sp.sqrt(2))
    adj_q = (dx.T*q[:9,:] + dy.T*q[18:,:]/sp.sqrt(2)).col_join(
        dy.T*q[9:18,:] + dx.T*q[18:,:]/sp.sqrt(2))
    assert (q.dot(E*v) - v.dot(adj_q)).expand() == 0
    a,b,c,x,y = sp.symbols("a b c x y")
    h = a+b*x+c*y
    assert sp.expand(h.subs(x,x+1)-h-b)==0 and sp.expand(h.subs(y,y+1)-h-c)==0
    qxx, qxy, qyy, delta = sp.symbols("qxx qxy qyy delta")
    quadratic = (qxx*x*x + 2*qxy*x*y + qyy*y*y)/2
    for axis, diagonal in ((x,qxx), (y,qyy)):
        truncation = quadratic.subs(axis, axis+delta) - quadratic - delta*sp.diff(quadratic,axis)
        assert sp.expand(truncation - diagonal*delta**2/2) == 0
    H,N,P,gamma,beta,k = sp.symbols("H N P gamma beta k", positive=True)
    energy = N*(H-k)**2/2 + (beta*N + gamma*P)*k
    minimizer = sp.solve(sp.diff(energy,k),k)[0]
    assert sp.simplify(minimizer - H + beta + gamma*P/N)==0
    return {"adjoint_symbolic":True, "affine_residual":0, "quadratic_forward_error_exact":True,
            "positive_template_minimizer":str(minimizer),
            "metric_forward_difference_column_norm_squared_lower_bound":2 / CELL**2,
            "proved_scaled_step_product_bound":float(STEP**2*18)}


def numeric():
    rng = np.random.default_rng(112410)
    a = rng.normal(size=(4,7,9))
    y = rng.normal(size=(10,7,9))
    err = abs(np.vdot(operator(a),y) - np.vdot(a,adjoint(y)))
    assert err < 1e-10
    row, col = np.indices(SHAPE)
    x, z = (col+.5)*CELL-1.5, (row+.5)*CELL+.3
    affine = .06*x - .02*z
    # Exact primal feasible zero-energy affine solution (free boundaries).
    candidate = np.stack((affine, np.full(SHAPE, .06*CELL), np.full(SHAPE,-.02*CELL), np.zeros(SHAPE)))
    assert np.max(abs(operator(candidate)[:7])) < 1e-10
    valid = np.ones(SHAPE, bool)
    rows = []
    for trial in range(12):
        bend = .12*(x/1.5)**2
        base = np.tan(np.deg2rad(6))*z + bend
        # Independent analytic ground tangent on a sloped, crowned smooth road.
        slopes = CELL * np.stack((.24*x/(1.5**2), np.ones_like(z)*np.tan(np.deg2rad(6))))
        obstacle = (abs(x - rng.uniform(-.3,.3)) < .125) & (abs(z-rng.uniform(.9,1.7)) < .125)
        depression = (abs(x-rng.uniform(-.3,.3)) < .125) & (abs(z-rng.uniform(2.,2.5)) < .125)
        bias = .012*np.cos(1.5*z + rng.uniform(-1,1))*np.sin(1.7*x)
        obs = base+bias+rng.normal(0,.002,size=SHAPE)
        local = obs - minimum_filter(obs, size=19)
        start = perf_counter()
        plain = solve(obs, valid, slopes, valid)
        elapsed = (perf_counter()-start)*1000
        target = obs + .08*obstacle - .10*depression
        j = solve(target,valid,slopes, valid & ~obstacle & ~depression)[3]
        rows.append({"empty_local_5_15cm":bool(((local>=.05)&(local<.15)).any()),
                     "empty_tgv_5_15cm":bool(((plain[3]>=.05)&(plain[3]<.15)).any()),
                     "positive_peak_m":float(j[obstacle].max()), "negative_trough_m":float(j[depression].min()),
                     "positive_median_m":float(np.median(j[obstacle])), "negative_median_m":float(np.median(j[depression])),
                     "runtime_ms":elapsed})
    # Identical observations for a flat empty road and a block wholly in a missing footprint.
    hidden = (abs(x)<.13)&(abs(z-1.3)<.13)
    flat = np.zeros(SHAPE)
    unseen = np.where(hidden,.08,0.)
    assert np.array_equal(flat[~hidden],unseen[~hidden])
    probe = solve(target,valid,slopes,valid & ~obstacle & ~depression, steps=4*STEPS)[3]
    return {"operator_adjoint_error":float(err),"observed_flat_vs_hidden_block_difference":0.,
            "stress_trials":rows, "threshold_demo_m":{"H":.08,"N":25,"anisotropic_perimeter_demo":20,
            "shrunken_height":.08-BETA-GAMMA*20/25},
            "max_jump_difference_140_vs_560_m_last_synthetic":float(np.max(abs(probe-j)))}


if __name__ == "__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    result={"symbolic":symbolic(),"numerical":numeric(),
            "code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),HERE/"tgv.py")}}
    (OUT / "theory_verification.json").write_text(json.dumps(result, indent=1),encoding="utf8")
    print(json.dumps(result, indent=1))
