"""Exact counterexample and paired simulation of a fully specified finite stopping problem."""
import json
from pathlib import Path
import numpy as np
import sympy as sp
from policy import (DIST, PROB, PRICE, LOSS, SIGMA, hazards, positive_likelihood,
                    delay_risk, solve, observe)

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[4]/"artifacts.local/work/egogeom-omni-v0/math_theory"/HERE.name


def paired():
    # For each world sample natural collision time once. All policies see identical
    # potential measurements and collision uniforms while alive.
    rng=np.random.default_rng(82024)
    n=20000
    H=rng.random(n)<.25
    start=32
    ids=np.arange(start,-1,-1)
    d=DIST[ids]
    h=hazards()[ids]
    lk=positive_likelihood()[ids]
    hit=((rng.random((n,len(ids)))<h)&H[:,None])
    impact=np.where(H,np.argmax(hit,axis=1),len(ids)+5)
    z=(np.where(H[:,None],-.5,.5) + rng.normal(size=(n,len(ids)))*(SIGMA*d**2))
    positive=z<0.
    post=np.full(n,.25)
    posterior=[]
    for k in range(len(ids)):
        if k:
            prior=post*(1.-h[k-1])/np.maximum(1.-post*h[k-1],1e-15)
        else:
            prior=post
        post=observe(prior,positive[:,k],lk[k])
        posterior.append(post.copy())
    posterior=np.stack(posterior,axis=1)
    value,adv,intervals=solve()
    _,fine,_=solve(grid=np.linspace(0,1,4001))
    advmat=np.column_stack([np.interp(posterior[:,k],PROB,adv[i]) for k,i in enumerate(ids)])
    # A no-table approximation uses a quadratic fit to the nonempty lower boundary.
    good=[i for i,b in enumerate(intervals) if b is not None]
    poly=np.polyfit(DIST[good],[intervals[i][0] for i in good],2)
    approx=np.clip(np.polyval(poly,d),0,1)
    active=(d>=DIST[min(good)]) & (d<=DIST[max(good)])
    def assess(mask):
        issued=mask & (np.arange(len(ids))[None,:]<=impact[:,None])
        first=np.where(issued.any(axis=1),np.argmax(issued,axis=1),len(ids)+10)
        collision=H & (impact < first+3)
        beep=issued.any(axis=1)
        return dict(clear_beeps=int((beep&~H).sum()),clear=int((~H).sum()),
                    collision=int(collision.sum()),obstacles=int(H.sum()),beeps=int(beep.sum()),
                    expected_total_cost=float(np.mean(PRICE*beep+LOSS*collision)))
    # Range proxies in metres come from each likelihood's underlying signed residual.
    # Under H1 the mean range is d; under H0 it is d+1. All rules share the same sample.
    y=d+z+.5
    methods={"optimal_grid":assess(advmat>=0.),"quadratic_boundary":assess((posterior>=approx)&active),
             "fixed_2m":assess(y<=2.),"gaussian_95_lower_2m":assess(y-1.645*SIGMA*d**2<=2.)}
    curves={}
    for method,stat in [("fixed",y),("lower95",y-1.645*SIGMA*d**2)]:
        table=[dict(threshold=float(t),**assess(stat<=t)) for t in np.linspace(-3,9,97)]
        matched=[x for x in table if x["collision"]<=methods["optimal_grid"]["collision"]]
        # Randomize the threshold once per encounter. Both expected counts are then
        # affine in the mixture; enumerate all two-point extreme solutions at the target.
        target=methods["optimal_grid"]["collision"]
        mixtures=[]
        for lo in table:
            for hi in table:
                gap=hi["collision"]-lo["collision"]
                if gap<=0 or not lo["collision"]<=target<=hi["collision"]:
                    continue
                frac=(target-lo["collision"])/gap
                clears=(1-frac)*lo["clear_beeps"]+frac*hi["clear_beeps"]
                mixtures.append(dict(expected_clear_beeps=clears,expected_collisions=target,
                                     threshold_lo=lo["threshold"],threshold_hi=hi["threshold"],
                                     probability_hi=frac))
        curves[method]=dict(grid=table,best_no_more_collisions=min(matched,key=lambda x:x["clear_beeps"]) if matched else None,
                            matched_collision_mixture=min(mixtures,key=lambda m:m["expected_clear_beeps"]) if mixtures else None)
    return dict(n=n,seed=82024,methods=methods,matched_curves=curves,
                value_grid_refinement_sup_marginal_cost=float(np.max(abs(fine[:,::4]-adv))),
                lower_boundary_quadratic_max_gap=float(max(abs(np.clip(np.polyval(poly,DIST[i]),0,1)-intervals[i][0]) for i in good)),
                quadratic_coefficients=poly.tolist(),threshold_intervals=intervals,distances=DIST.tolist())


def verify():
    R=sp.Rational
    A=[(1,R(1,2)),(3,R(1,2))]
    B=[(R(1,2),R(2,9)),(2,R(5,9)),(R(7,2),R(2,9))]
    for law in (A,B):
        assert sum(p for x,p in law)==1 and sum(x*p for x,p in law)==2
        assert sum((x-2)**2*p for x,p in law)==1
    tail=[sum(p for x,p in law if x<=R(11,10)) for law in (A,B)]
    assert tail[0]>R(3,10)>tail[1]
    d,c=sp.symbols("d c",positive=True)
    assert sp.simplify(sp.diff(c*d**4,d)-4*c*d**3)==0
    return dict(equal_moments=[2,1],collision_probabilities=list(map(str,tail)),simulation=paired())


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    data=verify()
    (OUT/"theory_verification.json").write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf8")
    print(data["simulation"]["methods"])
