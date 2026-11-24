"""Freeze both label-free range selections on exactly the same single-view clusters."""
import itertools
from pathlib import Path

import numpy as np
from point_support import starts
from predict import ART, HERE, OUT, dump, sha
from support import CausalRayReadout, axes

STEP_M=.05
BASELINE_M=.05
TOL_CELLS=2


def select(centres, positions):
    # True means this view supplied >=18 returns, regardless of how many.
    votes=np.zeros((len(centres),82),bool)
    for i,z in enumerate(centres):
        cells=np.floor(z/STEP_M+.5).astype(int)
        votes[i,cells]=True
    any_cell=np.flatnonzero(votes.any(0))
    nearest=float(any_cell[0]*STEP_M) if len(any_cell) else np.inf
    paired=np.zeros(votes.shape[1],bool)
    for i,j in itertools.combinations(range(len(centres)),2):
        if np.linalg.norm(positions[i]-positions[j])<BASELINE_M:
            continue
        # Mark the smaller observed bin of each stable pair; tolerance includes equality.
        paired |= votes[i]&votes[j]
        for shift in range(1,TOL_CELLS+1):
            paired[:-shift] |= ((votes[i,:-shift]&votes[j,shift:]) |
                                (votes[j,:-shift]&votes[i,shift:]))
    stable=np.flatnonzero(paired)
    persistent=float(stable[0]*STEP_M) if len(stable) else np.inf
    assert nearest<=persistent
    return nearest,persistent


def clip_predict(version,name):
    path=ART/"math_theory"/(version+"_stream")/"aria_gen2"/(name+".npz")
    with np.load(path) as obs:
        dep,K,R,t,fw=[obs[k] for k in ("depth_s4","K","poses_R","poses_t","strict_fwd")]
    history=CausalRayReadout()
    result=[]
    for i in range(len(dep)):
        history.step(dep[i],K[i],R[i],t[i],fw[i])
        forward,side=axes(R[i],fw[i])
        centres,positions=[],[]
        for _,pos,points in history.history:
            delta=points-t[i,:2]
            z=delta@forward
            mask=(abs(delta@side)<=.30)&(z>=.30)&(z<=4.)
            centres.append(starts(z[mask]))
            positions.append(pos)
        result.append(select(centres,positions))
    out=OUT/version/"selector_predictions"/(name+".npz")
    out.parent.mkdir(exist_ok=True)
    nearest,persistent=np.array(result).T
    np.savez_compressed(out,nearest=nearest,persistent=persistent)
    return {"name":name,"input_sha256":sha(path),"prediction_sha256":sha(out)}


def main():
    delta=np.array([.05,0.,0.])
    loc=np.zeros(3)
    assert select([np.array([1.]),np.array([1.1])],[loc,delta])==(1.,1.)
    assert select([np.array([1.]),np.array([1.15])],[loc,delta])==(1.,np.inf)
    assert select([np.array([1.,1.,1.]),np.array([1.])],[loc,loc])==(1.,np.inf)
    assert select([np.array([])],[loc])==(np.inf,np.inf)
    report={"protocol_sha256":sha(HERE/"SELECTOR.md"),"code_sha256":{p.name:sha(p) for p in
              (Path(__file__),HERE/"point_support.py",HERE/"support.py")},"versions":{}}
    for ver in ("v4","v41"):
        names=sorted(p.stem for p in (ART/"math_theory"/(ver+"_stream")/"aria_gen2").glob("*.npz"))
        assert len(names)==19
        report["versions"][ver]=[clip_predict(ver,name) for name in names]
    dump(OUT/"selector_predictions.json",report)


if __name__=="__main__":
    main()
