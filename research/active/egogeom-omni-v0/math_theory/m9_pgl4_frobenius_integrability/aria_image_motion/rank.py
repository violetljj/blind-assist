"""Fit on one walk, calibrate on its reserved clips, and score the opposite walk."""
import json
from pathlib import Path
import numpy as np
from build import GEOM, VIS, OLD, OUT, HERE, PACKED, dump, sha
from evaluate import Clip, BW_SPEEDS_PROXY, make_episodes, rec, windows
from selector_eval import actions
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ARMS=("geometry","aligned","mirrored")


def features(x, arm):
    g,v=len(GEOM),len(VIS)
    if arm=="geometry":
        return x[:,:g]
    if arm=="aligned":
        return x[:,:g+v]
    return np.column_stack((x[:,:g],x[:,g+v:]))


def reference(name):
    with np.load(PACKED/(name+".npz")) as archive:
        rows=archive["eval_rows"]
    episodes=make_episodes(name,rows,BW_SPEEDS_PROXY,kind="blind")
    valid=episodes[0]["valid"]
    clear=valid&(rows[:,0]>4.)&(rows[:,2]<=.5)
    position=np.cumsum(np.where(valid,rows[:,3],0.),dtype=float)
    zero=np.zeros(len(rows))
    source=name.rsplit("_c",1)[0]
    c=Clip({"name":name},"aria",source,int(source.endswith("1")),zero,zero,zero,episodes,clear,position)
    # Common helper reports unknowns. Here geometry-known means a range candidate exists.
    return c


def training_arrays(clips, samples):
    X,Y,weights=[],[],[]
    for c in clips:
        name=c.obs["name"]
        x,_,fi,counts=samples[name]
        gt=c.episodes[0]["rows"][:,0]
        danger=c.episodes[0]["valid"]&(gt>=.30)&(gt<=4.)
        use=(c.clear|danger)[fi]
        X.append(x[use])
        yy=danger[fi]&(np.abs(x[:,0]-gt[fi])<=.55)
        Y.append(yy[use])
        weights.append(1./counts[fi[use]])
    x,y,w=np.concatenate(X),np.concatenate(Y),np.concatenate(weights)
    assert y.any() and not y.all()
    return x,y,w/w.mean()


def predict(model,arm,sample):
    x,offset,_,counts=sample
    p=model.predict_proba(features(x,arm))[:,1]
    score=np.full(len(counts),-np.inf)
    dist=np.full(len(counts),np.inf)
    for i in np.flatnonzero(counts):
        begin,end=offset[i:i+2]
        j=begin+np.argmax(p[begin:end])
        score[i],dist[i]=p[j],x[j,0]
    return score,dist


def fit_cut(clips,scores):
    # Any known clear run with an alarm costs at least six walker charges.
    # This bounds the number of clear segments touched by every legal lower cut.
    km=rec(sum(c.count(np.zeros(len(c.clear),bool)) for c in clips))["walker_km"]
    limit=int(np.floor(5*km/len(clips[0].episodes)))
    peaks=[]
    for c in clips:
        s=scores[c.obs["name"]]
        edges=np.diff(np.r_[False,c.clear,False].astype(int))
        peaks.extend(s[l:r].max() for l,r in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    border=sorted(peaks,reverse=True)[limit] if len(peaks)>limit else -np.inf
    options=np.unique(np.concatenate([scores[c.obs["name"]] for c in clips]))
    options=options[np.isfinite(options)&(options>border)]
    best=None
    for cut in np.r_[np.inf,options]:
        stats=rec(sum(c.count(scores[c.obs["name"]]>=cut) for c in clips))
        if stats["fa_per_km"]>5 or stats["clear_warnings_per_km"]>5:
            continue
        tie=(-stats["timely"],stats["repeated_clear_warnings"],stats["clear_alert_frames"],
             stats["alert_frames"],-cut)
        if best is None or tie<best[0]:
            best=(tie,cut,stats)
    return best[1:]


def cut_text(cut):
    return float(cut) if np.isfinite(cut) else "off"


def ver_samples(version,ledger):
    samples={}
    for row in ledger["files"]:
        name=row["name"]
        path=OUT/version/(name+".npz")
        assert sha(path)==row["features"][version]["sha256"]
        with np.load(path) as archive:
            x,off=archive["features"],archive["offsets"]
        counts=np.diff(off)
        assert len(counts)==150 and (counts>=0).all()
        samples[name]=x,off,np.repeat(np.arange(len(counts)),counts),counts
    return samples


def build_predictions(version,ledger):
    samples=ver_samples(version,ledger)
    splits=[]
    for test in ("aria_walk_0","aria_walk_1"):
        other=sorted(n for n in samples if not n.startswith(test+"_"))
        tar=sorted(n for n in samples if n.startswith(test+"_"))
        cal=[n for n in other if int(n.rsplit("_c",1)[1])%3==1]
        fit=[n for n in other if n not in cal]
        training=[reference(n) for n in fit]
        calibration=[reference(n) for n in cal]
        x,y,w=training_arrays(training,samples)
        split=dict(test=test,fit=fit,calibration=cal,holdout=tar,
                   train_candidates=len(x),train_positive=int(y.sum()),arms={})
        for arm in ARMS:
            model=HistGradientBoostingClassifier(max_iter=45,max_leaf_nodes=7,l2_regularization=3.,
                          min_samples_leaf=35,early_stopping=False,random_state=6151)
            model.fit(features(x,arm),y,sample_weight=w)
            cal_scores={n:predict(model,arm,samples[n])[0] for n in cal}
            cut,stats=fit_cut(calibration,cal_scores)
            hashes={}
            for n in tar:
                score,dist=predict(model,arm,samples[n])
                dest=OUT/version/arm/(n+".npz")
                dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,score=score,distance=dist,alert=score>=cut)
                hashes[n]=sha(dest)
            split["arms"][arm]=dict(cut=cut_text(cut),calibration=stats,predictions=hashes)
        splits.append(split)
    dump(OUT/version/"folds.json",dict(splits=splits,feature_receipt=sha(OUT/"features.json"),
         code=sha(Path(__file__)),protocol=sha(HERE/"PROTOCOL.md")))


def counts(clips,scores,dist,alerts):
    measured=actions(clips,alerts)
    cut,upper=fit_cut(clips,scores)
    events=[]
    errors=[]
    all_y,all_s=[],[]
    for c in clips:
        n=c.obs["name"]
        er=c.episodes[0]["rows"]
        valid=c.episodes[0]["valid"]
        hz=valid&(er[:,0]>=.3)&(er[:,0]<=4)
        ok=hz|c.clear
        all_y.extend(hz[ok])
        all_s.extend(np.where(np.isfinite(scores[n][ok]),scores[n][ok],-1))
        d=dist[n]
        errors.extend(np.abs(np.where(np.isfinite(d[hz]),d[hz],6.)-er[hz,0]))
        for walker,ep in enumerate(c.episodes):
            for lo,hi,idx in windows(ep):
                true=idx[hz[idx]]
                events.append(dict(clip=n,walker=walker,start=lo,timely=bool(alerts[n][idx].any()),
                    true_timely=bool(alerts[n][true].any()),hazard_window=bool(len(true)),
                    distance_hit=bool((np.abs(d[true]-er[true,0])<=.55).any())))
    assert sum(e["timely"] for e in events)==measured["pooled"]["timely"]
    return dict(events=events,operating=measured,correct_distance=sum(e["distance_hit"] for e in events),
                true_timely=sum(e["true_timely"] for e in events),distance_mae_fill6=float(np.mean(errors)),
                evaluated_hazard_frames=len(errors),auroc_hazard_clear=float(roc_auc_score(all_y,all_s)),
                pooled_posthoc_cut=cut_text(cut),pooled_posthoc=upper)


def evaluate(version):
    folds=json.loads((OUT/version/"folds.json").read_text())
    assert folds["feature_receipt"]==sha(OUT/"features.json")
    results={}
    for arm in ARMS:
        scores,dist,alerts={}, {}, {}
        clips=[]
        for split in folds["splits"]:
            for n,digest in split["arms"][arm]["predictions"].items():
                path=OUT/version/arm/(n+".npz")
                assert sha(path)==digest
                with np.load(path) as f:
                    scores[n],dist[n],alerts[n]=f["score"],f["distance"],f["alert"]
                c=reference(n)
                c.known=np.isfinite(dist[n])
                clips.append(c)
        result=counts(clips,scores,dist,alerts)
        assert result["evaluated_hazard_frames"]==840 and len(result["events"])==143
        assert sum(r["hazard_window"] for r in result["events"])==129
        results[arm]=result
    return results


def main():
    ledger=json.loads((OUT/"features.json").read_text())
    assert ledger["source"]["PROTOCOL.md"]==sha(HERE/"PROTOCOL.md")
    assert ledger["source"]["motion.py"]==sha(HERE/"motion.py")
    with threadpool_limits(limits=1):
        for ver in ("v4","v41"):
            build_predictions(ver,ledger)
        report={ver:evaluate(ver) for ver in ("v4","v41")}
    dump(OUT/"results.json",dict(versions=report,code=sha(Path(__file__)),features=sha(OUT/"features.json"),
         folds={v:sha(OUT/v/"folds.json") for v in report}))
    for v,arms in report.items():
        for a,res in arms.items():
            op=res["operating"]["pooled"]
            print(v,a,res["correct_distance"],op["timely"],op["clear_warnings_per_km"],
                  res["pooled_posthoc"]["timely"],res["auroc_hazard_clear"],flush=True)

if __name__=="__main__":
    main()
