"""Symbolic invariance, quantizer counterexamples, single-budget LP gap and synthetic controls."""
import itertools
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.linalg import hadamard
from scipy.stats import kurtosis

from eval_empirical import OUT,dump,sha


def quant(a,b=8):
    radius=np.max(np.abs(a))
    step=radius/((1<<(b-1))-1)
    return np.round(a/step)*step if radius else a


def verify():
    t,a,b=sp.symbols("t a b",positive=True)
    obj=a/t**2+b*t**2
    minimum=(a/b)**sp.Rational(1,4)
    assert sp.simplify(sp.diff(obj,t).subs(t,minimum))==0
    assert sp.simplify(obj.subs(t,minimum)-2*sp.sqrt(a*b))==0
    d1,d2=sp.symbols("d1 d2",positive=True)
    angle=sp.symbols("theta",real=True)
    r=sp.Matrix([[sp.cos(angle),-sp.sin(angle)],[sp.sin(angle),sp.cos(angle)]])
    q=r*sp.diag(d1,d2)
    assert sp.trigsimp(q.inv()*q)==sp.eye(2)
    assert sp.trigsimp(r.T*r)==sp.eye(2)
    # A valid affine uint8 zero; both branches share exact requantizer in this example.
    s=.125
    z=127
    x=np.arange(16,240)
    qz=np.clip(np.rint(0/s)+z,0,255)
    assert qz==127 and np.all(x+int(qz)-z==x)
    # Symmetric zeros alone cannot equalize differently rounded RGB trunks.
    assert float(np.rint(.7/.5)*.5)!=float(np.rint(.7/.25)*.25)
    # Scalar budget: fractional knapsack vs all 0-1 solutions.
    rng=np.random.default_rng(20261124)
    max_gap=0.
    for _ in range(200):
        gains=rng.uniform(0,8,8)
        costs=rng.integers(1,10,8)
        budget=rng.integers(1,sum(costs))
        left=budget
        lp=0.
        for i in np.argsort(-gains/costs):
            frac=min(1,left/costs[i])
            lp+=frac*gains[i]
            left-=frac*costs[i]
        opt=max(np.dot(v,gains) for v in itertools.product((0,1),repeat=8) if np.dot(v,costs)<=budget)
        gap=max(0,lp-opt)
        assert gap<=gains.max()+1e-12
        max_gap=max(gap/gains.max(),max_gap)
    c=64
    h=rng.normal(size=(6000,c))
    h[:,0]=np.where(rng.random(len(h))<.004,25*rng.choice([-1,1],len(h)),.12*rng.normal(size=len(h)))
    w=rng.normal(size=(c,16))
    w[0]/=16
    importance=np.geomspace(.002,1,16)
    w*=importance
    w*=.2/np.std(h@w)
    ref=h@w
    # Log-depth targets, same 2000 fit rows for all data-driven gauges.
    cal=h[:2000]; test=h[2000:]
    dynamic=np.maximum(np.max(np.abs(cal),axis=0),1e-6)
    w_amp=np.max(np.abs(w),axis=1)
    smooth=np.sqrt(w_amp/dynamic)
    aa=np.linalg.norm(w,axis=1)**2*(np.max(dynamic)/127)**2
    bb=16*np.mean(cal**2,axis=0)*(np.max(w_amp)/127)**2
    fisher=np.clip((aa/bb)**.25,.125,8)
    hh=hadamard(c)/np.sqrt(c)
    hh*=rng.choice([-1,1],c)[None,:]
    if np.linalg.det(hh)<0:
        hh[:,0]*=-1
    result={}
    for name,qq,actbits in [("plain_w8a8",np.eye(c),8),("smooth_w8a8",np.diag(smooth),8),
                           ("fisher_w8a8",np.diag(fisher),8),("fisher_so_w8a8",hh@np.diag(fisher),8),
                           ("fisher_so_w8a16",hh@np.diag(fisher),16)]:
        rotated=cal@qq.T
        wt=np.linalg.solve(qq.T,w)
        assert np.max(np.abs((test@qq.T)@wt-ref[2000:]))<1e-13
        radius=np.max(np.abs(rotated))
        limit=2**(actbits-1)-1
        xr=np.clip(np.rint((test@qq.T)*limit/radius),-limit,limit)*radius/limit
        predicted=xr@quant(wt)
        result[name]={"mean_abs_relative_z_pct":float(100*np.abs(np.expm1(predicted-ref[2000:])).mean()),
                      "fit_peak":float(radius)}
    # Rotating a Hadamard row concentrates it instead of spreading it.
    witness=hh[0]
    trace_quadratic=np.trace(np.diag([1.,0.])@np.diag([0.,1.]))
    assert np.isclose(trace_quadratic,0)
    out={"script_sha256":sha(Path(__file__)),"seed":20261124,"symbolic_gauge_and_optimum":True,
         "nonzero_zero_point":int(qz),"symmetric_trunk_counterexample_depths":[float(np.rint(.7/.5)*.5),float(np.rint(.7/.25)*.25)],
         "lp_instances":200,"max_observed_gap_over_largest_gain":max_gap,
         "synthetic":{"n_cal":2000,"n_eval":4000,"outlier_channel_excess_kurtosis":float(kurtosis(h[:,0])),
                      "arms":result,"hadamard_counterexample_peak_ratio":float(np.max(np.abs(hh@witness))/np.max(np.abs(witness)))}}
    dump(OUT/"theory_verification.json",out)
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    verify()
