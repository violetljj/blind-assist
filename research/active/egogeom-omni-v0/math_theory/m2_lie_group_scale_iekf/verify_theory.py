"""Run M2 symbolic checks, stress trajectories and a single-thread CPU benchmark."""
import argparse
import json
import math
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import sympy as sp
from lie_scale import LieScaleIEKF


def cross(v):
    x, y, z = v
    return sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def symbolic():
    # Symbolic matrix navigation block, without a numeric orientation or input.
    I = sp.eye(5)
    def element(k):
        X = I.copy()
        X[:3,:3] = sp.Matrix(3, 3, sp.symbols(f'r{k}_0:9'))
        X[:3,3:] = sp.Matrix(3, 2, sp.symbols(f'x{k}_0:6'))
        return X

    a, g, w = (sp.Matrix(sp.symbols(f'{k}0:3')) for k in ('a','g','w'))
    def flow(X):
        M = sp.zeros(5)
        M[:3,:3] = X[:3,:3] * cross(w)
        M[:3,3] = X[:3,:3] * a + g
        M[:3,4] = X[:3,3]
        return M

    X, Y = element(1), element(2)
    affine = (flow(X*Y)-flow(X)*Y-X*flow(Y)+X*flow(I)*Y).applyfunc(sp.expand)
    assert affine == sp.zeros(5)
    # Lie derivatives of a 1-D translational subsystem under constant attitude.
    p, v, s, b, acc, g1 = sp.symbols('p v s b acc g', nonzero=True)
    state = sp.Matrix([p,v,s,b,g1])
    f = sp.Matrix([v,acc-b+g1,0,0,0])
    y = p/s
    L1 = sp.Matrix([y]).jacobian(state).dot(f)
    L2 = sp.Matrix([L1]).jacobian(state).dot(f)
    O = sp.Matrix([y,L1,L2]).jacobian(state)
    assert sp.simplify(L2 - (acc-b+g1)/s) == 0 and O.rank() == 3
    # This reduced matrix has observable scale under changing acceleration,
    # with an unresolved b-g offset; constant nonzero acceleration loses scale.
    def excitation_rank(av):
        return int(sp.Matrix([[ai,1,-1] for ai in av]).rank())

    ranks = [excitation_rank([1,1,1]), excitation_rank([1,2,-1])]
    assert ranks == [1,2]
    # 3D known attitude: generic case (exact rational rotations).
    Rx = sp.Matrix([[1,0,0],[0,0,-1],[0,1,0]])
    Ry = sp.Matrix([[0,0,1],[0,1,0],[-1,0,0]])
    D = sp.Matrix.vstack(*[a_i.row_join(R).row_join(-sp.eye(3)) for R,a_i in
                           [(sp.eye(3),sp.Matrix([1,0,0])),(Rx,sp.Matrix([0,1,0])),
                            (Ry,sp.Matrix([0,0,1])),(Rx*Ry,sp.Matrix([2,-1,1]))]])
    assert D.rank() == 7
    assert Rx*cross([0,1,0]) != cross([0,1,0])*Rx
    # Verify the analytic chain rule for the local image plane d=c+i*u+j*v.
    l, cx, cy, fx, fy, c, ix, iy = sp.symbols('l cx cy fx fy c ix iy')
    base = sp.Matrix(sp.symbols('q0:3'))
    shift = sp.exp(-l)*sp.Matrix(sp.symbols('t0:3'))
    q = base + shift
    e = q[2] - (c + ix*(cx+fx*q[0]/q[2]) + iy*(cy+fy*q[1]/q[2]))
    j = -shift[2]+ix*fx*(shift[0]-q[0]*shift[2]/q[2])/q[2]+iy*fy*(shift[1]-q[1]*shift[2]/q[2])/q[2]
    assert sp.simplify(e.diff(l)-j) == 0
    return {"group_affine_no_bias": True, "ordinary_additive_bias_counterexample": True,
            "one_dimensional_L0_L1_L2_rank_out_of_5": int(O.rank()),
            "constant_varying_acceleration_ranks_out_of_3": ranks,
            "excited_known_orientation_rank_out_of_7": int(D.rank()), "registration_jacobian": True}


def synthetic():
    rng = np.random.default_rng(241124)
    shape = (72, 96)
    dh = np.zeros(shape, np.float32)
    nrm = np.zeros((3,*shape), np.float32)
    nrm[2] = 1
    ls = np.full(shape,-3.0,np.float32)
    K = np.array([[230.,0,192.],[0,230.,144.],[0,0,1.]])
    R = np.eye(3)
    scenarios, times = [], []
    for s0 in np.geomspace(0.4,2.5,7):
        hits, final, accepted = [], [], []
        for _ in range(16):
            flt = LieScaleIEKF(s0)
            errs = []
            for i in range(22):
                # Known per-frame metric pose. Frontal fixed wall at world z=9m.
                t = np.array([0,0,.30*i])
                depth = (9-t[2]+rng.normal(0,.004,size=shape)).astype(np.float32)
                depth[2:8,4:14] += .6*math.sin(i*.8)
                t0 = perf_counter_ns()
                shat = flt.step(depth,dh,nrm,ls,K,R,t)
                times.append((perf_counter_ns()-t0)/1e6)
                errs.append(abs(shat-1.0))
            hits.append(next((i+1 for i in range(len(errs)) if max(errs[i:])<.05), None))
            final.append(errs[-1])
            accepted.append(flt.accepted)
        scenarios.append({"s0":float(s0),"repeats":len(hits),"stable_5pct_by_frame14":sum(h is not None and h<=14 for h in hits),
                          "max_first_stably_within_5pct_frame":max(hits) if all(h is not None for h in hits) else None,
                          "max_final_abs_relative_error":max(final),"min_updates":min(accepted)})
    stationary = LieScaleIEKF(1.7)
    spinning = LieScaleIEKF(1.7)
    flat_parallel = LieScaleIEKF(1.7)
    wall = np.full(shape,4.,np.float32)
    for i in range(50):
        stationary.step(wall,dh,nrm,ls,K,R,np.zeros(3))
        co, si = math.cos(i*.12), math.sin(i*.12)
        yaw = np.array([[co,-si,0],[si,co,0],[0,0,1.]])
        spinning.step(wall,dh,nrm,ls,K,yaw,np.zeros(3))
        flat_parallel.step(wall,dh,nrm,ls,K,R,np.array([.2*i,0,0]))
    assert stationary.scale == spinning.scale == flat_parallel.scale == 1.7
    assert stationary.accepted == spinning.accepted == flat_parallel.accepted == 0
    assert stationary.variance == spinning.variance == flat_parallel.variance == .25
    return {"synthetic_static_wall_with_moving_patch":scenarios,
            "unexcited_hold_checks":["stationary", "pure_rotation", "parallel_translation_infinite_uniform_wall"],
            "latency_ms_all_calls":{"median":float(np.median(times)),"p95":float(np.percentile(times,95)),
                                    "mean":float(np.mean(times)),"n":len(times)}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[5] /
                   "artifacts.local/work/egogeom-omni-v0/math_theory/m2_lie_group_scale_iekf/theory_verification.json")
    args = p.parse_args()
    result = {"sympy":symbolic(), **synthetic(), "python":platform.python_version(),"numpy":np.__version__,
              "sympy_version":sp.__version__}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
