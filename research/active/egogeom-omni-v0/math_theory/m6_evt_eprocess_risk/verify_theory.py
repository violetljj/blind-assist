"""Conditional spending bound, dependence failure, and finite-head extreme diagnostics."""
import json
from pathlib import Path

import numpy as np
import sympy as sp
from risk import InvariantEProcessBEV

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[4]/"artifacts.local/work/egogeom-omni-v0/math_theory"/HERE.name


def simulation():
    rng=np.random.default_rng(62024)
    p,q,r=sp.symbols("p q r",positive=True)
    factor=r*q/p+(1-r)*(1-q)/(1-p)
    assert sp.simplify(factor.subs(r,p))==1
    assert sp.simplify(sp.diff(factor,r)-(q-p)/(p*(1-p)))==0
    s,z,b,h,d=sp.symbols("s z b h d",positive=True)
    assert sp.simplify(((s*d+b)-(s*h+b))/(s*z)-(d-h)/z)==0
    # 500 walks, 5,000 2-m blocks each; 6 ordered observations per block.
    n=500
    count_valid=np.zeros(n,int)
    count_bad=np.zeros(n,int)
    state=np.zeros(n,bool)
    null_q,alternative=.10,.55
    for _ in range(5000):
        persistent=rng.random(n)<null_q
        wealth=np.ones(n)
        crossed=np.zeros(n,bool)
        for _ in range(6):
            state=rng.random(n)<np.where(state,.1,.05)
            wealth*=np.where(state,alternative/null_q,(1-alternative)/(1-null_q))
            crossed|=wealth>=100.
        count_valid+=crossed
        count_bad+=persistent  # all six factors large or all small, same marginal 0.1
    # Spatial tiles share a Pareto scale and a slope shock; emulate a bounded BEV
    # histogram transfer, not fitted tails of the four development domains.
    scale=1+rng.pareto(2.,size=(30000,1))
    bank=scale*rng.lognormal(0,.25,size=(len(scale),8))
    bank*=np.where(rng.random((len(scale),1))<.03,3.,1.)
    counts=np.clip(bank*2,0,1728)
    zmid=.5+.2*np.arange(8)
    old=np.max((1-np.exp(-counts/14))/(zmid+.15),axis=1)+.06*(np.log1p(counts/14)/(zmid+.4)**2).sum(axis=1)
    renorm=np.max(-np.expm1(-counts/np.maximum(counts.mean(axis=1,keepdims=True),.1))/(1+zmid**2),axis=1)
    def hill(x,k):
        v=np.sort(x)
        return float(np.mean(np.log(v[-k:]/v[-k-1])))
    rng=np.random.default_rng(62027)
    a=rng.uniform(.5,3,(72,96)).astype("f")
    normal=rng.normal(size=(3,72,96)).astype("f")
    normal/=np.linalg.norm(normal,axis=0)
    dh=rng.normal(size=a.shape)
    ls=rng.normal(size=a.shape)
    K=np.array([[240,0,192],[0,240,144],[0,0,1.]])
    rot=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    orig,changed=InvariantEProcessBEV(),InvariantEProcessBEV()
    errs=[]
    original_scores=[]
    for i in range(5):
        t=np.array([.03*i,.02*i,0])
        one=orig.step(a,dh,normal,ls,K,rot,t,[0,1,0])
        two=changed.step(3*a,3*dh+2,normal,ls+1,K,rot,3*t+np.array([4,2,3]),[0,1,0])
        errs.append(abs(one-two))
        original_scores.append(one)
    assert max(original_scores)>.01 and max(errs)<1e-5
    valid=count_valid.sum()/5000
    invalid=count_bad.sum()/5000
    assert valid<5<invalid
    return dict(seed=62024,walks=500,km_per_walk=10,block_m=2,alpha=.01,
                conditional_null_rate_per_km=valid,invalid_marginal_only_rate_per_km=invalid,
                finite_horizon_bound=5.,similarity_max_abs=max(errs),original_scores=original_scores,
                toy_spatial_hill={str(k):[hill(scale.ravel(),k),hill(old,k),hill(renorm,k)] for k in (100,500,2000)},
                hill_columns=["latent_Pareto","bounded_BEV_transfer","bounded_normalized"])


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    data=simulation()
    (OUT/"theory_verification.json").write_text(json.dumps(data,indent=2),encoding="utf8")
    print(data)
