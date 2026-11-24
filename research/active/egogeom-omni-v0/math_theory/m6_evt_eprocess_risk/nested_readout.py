"""Exploratory offline trained readouts, with group-out calibration inside each outer fold."""
import argparse
import os
for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import replay
from risk import InvariantEProcessBEV

HERE=Path(__file__).resolve().parent


def features(c,stream):
    path=stream/c.obs["domain"]/(c.obs["name"]+".npz")
    with np.load(path) as f:
        arrays=[f[k] for k in ("depth_s4","dh_s4","nrm_s4","ls_s4","K","poses_R","poses_t","strict_fwd")]
    engine=InvariantEProcessBEV()
    base=np.column_stack([c.raw,c.confirmed,np.minimum(c.bev_d,12),c.obs["pred"],c.obs["sigma"]])
    result=[]
    for dep,dh,nrm,ls,K,R,t,fw in zip(*arrays,strict=True):
        score=engine.step(dep,dh,nrm,ls,K,R,t,fw)
        # Spatial context: within each fixed image region the same quantiles
        # summarize depth, predicted height, vertical face evidence and uncertainty.
        # No source ID, camera intrinsics, filenames, labels or timing enter the classifier.
        cells=[score]
        for ys,xs in [(slice(12,60,4),slice(28,68,4)),(slice(44,72,4),slice(0,96,4))]:
            region=np.stack((dep[ys,xs],dh[ys,xs],1.-nrm[1,ys,xs]**2,ls[ys,xs]))
            cells.extend(np.quantile(region,[.2,.5,.8],axis=(1,2)).ravel())
        result.append(cells)
    return base,np.column_stack((base,result))


def fit(clips,arm):
    x,y=[],[]
    for c in clips:
        valid=c.episodes[0]["valid"]
        truth=c.obs["rows"][:,0]
        include=valid & ((truth<=4.)|c.clear)
        x.append(c.feature[arm][include])
        y.append(truth[include]<=4.)
    return HistGradientBoostingClassifier(max_iter=45,max_leaf_nodes=7,l2_regularization=3.,
               min_samples_leaf=35,early_stopping=False,random_state=624).fit(np.concatenate(x),np.concatenate(y))


def predict(model,c,arm):
    return replay.confirm_score(model.predict_proba(c.feature[arm])[:,1])


def evaluate(clips,arm):
    cases=[]
    fit_rows=[]
    timing=[]
    for outer in (0,1):
        train=[c for c in clips if c.fold!=outer]
        test=[c for c in clips if c.fold==outer]
        for source in sorted({c.source for c in train}):
            inner=[c for c in train if c.source!=source]
            model=fit(inner,arm)
            for c in train:
                if c.source==source:
                    c.inner_score=predict(model,c,arm)
        cut,rec=replay.fit_threshold(train,"inner_score",True)
        fit_rows.append(dict(test_fold=outer,inner_calibration=rec,train_sources=sorted({c.source for c in train})))
        frozen=fit(train,arm)
        for c in test:
            c.outer_score=predict(frozen,c,arm)
            cases.append((c,c.count(c.outer_score>=cut)))
        # Sequential deployment cost for the classifier alone, one sample per call.
        for row in test[0].feature[arm][:64]:
            t=time.perf_counter_ns()
            frozen.predict_proba(row[None,:])
            timing.append((time.perf_counter_ns()-t)*1e-6)
    return dict(folds=fit_rows,pooled=replay.score(sum(x for _,x in cases)),
                domains={d:replay.score(sum(x for c,x in cases if c.obs["domain"]==d))
                         for d in sorted({c.obs["domain"] for c,_ in cases})},
                inference_only_ms_p50_p95=np.percentile(timing,[50,95]).tolist())


def clear_transition_audit(clips):
    """Truth is confined to this retrospective audit, never used for test actions."""
    rows=[]
    for f in (0,1):
        train=[c for c in clips if c.fold!=f]
        clear=np.concatenate([c.feature[1][c.clear,5] for c in train])
        cut=float(np.quantile(clear,.90,method="higher"))
        for dom in sorted({c.obs["domain"] for c in clips}):
            n=hits=previous_hit=both=0
            for c in clips:
                if c.fold!=f or c.obs["domain"]!=dom:
                    continue
                h=c.feature[1][:,5]>cut
                eligible=c.clear[1:] & c.clear[:-1] & h[:-1]
                n+=int(c.clear.sum())
                hits+=int((h&c.clear).sum())
                previous_hit+=int(eligible.sum())
                both+=int((eligible&h[1:]).sum())
            rows.append(dict(fold=f,domain=dom,clear=n,hits=hits,
                             consecutive_clear_with_prev_hit=previous_hit,hits_after_hit=both))
    return rows


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--version",choices=("v4","v41"),default="v4")
    version=ap.parse_args().version
    clips,stream=replay.materialize(version)
    for c in clips:
        c.feature=features(c,stream)
    old,new=evaluate(clips,0),evaluate(clips,1)
    report=dict(version=version,code_sha256=replay.digest(Path(__file__)),
                manifest_sha256=replay.digest(stream/"manifest.json"),
                legacy_features=old,geometry_and_context=new,clear_transition_audit=clear_transition_audit(clips),
                scope="supervised exploratory follow-up, leave-one-source-out inner cuts; no online labels")
    replay.save(replay.OUT.parent/HERE.name/("nested_readout_"+version+".json"),report)
    print(version,old["pooled"],new["pooled"])
