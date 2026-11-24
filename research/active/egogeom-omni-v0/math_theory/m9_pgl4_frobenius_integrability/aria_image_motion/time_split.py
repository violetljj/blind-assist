"""Remove other-wearer clips overlapping a test clip on the verified common clock."""
import csv
import json
from pathlib import Path
import numpy as np
from build import ART, HERE, OUT, dump, sha
from rank import counts, cut_text, features, fit_cut, predict, reference, training_arrays, ver_samples
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


def intervals(names):
    raw=ART.parents[1]/"downloads/aria-gen2-pilot-20261117"
    info={}
    identities={}
    for walk in ("walk_0","walk_1"):
        p=raw/walk/"x/depth/pinhole_camera_parameters.json"
        traj=raw/walk/"x/closed_loop_trajectory.csv"
        with traj.open(encoding="utf8") as f:
            data=[(int(r["tracking_timestamp_us"])*1000,int(r["utc_timestamp_ns"]),r["graph_uid"])
                  for r in csv.DictReader(f)]
        t,common=np.array([r[:2] for r in data],np.int64).T
        assert np.all(np.diff(t)>0) and np.all(np.diff(common)>=0)
        frames=json.loads(p.read_text())[::6]
        identities[walk]=dict(meta=sha(p),trajectory=sha(traj),graph_ids=sorted({r[2] for r in data}))
        for n in names:
            if walk not in n:
                continue
            i=int(n.rsplit("_c",1)[1])*150
            ends=[int(frames[j]["frameTimestampNs"]) for j in (i,i+149)]
            assert t[0]<=ends[0]<ends[1]<=t[-1]
            info[n]=(np.interp(ends,t,common)*1e-9).tolist()
    assert len(info)==19 and identities["walk_0"]["graph_ids"]==identities["walk_1"]["graph_ids"]
    return info,identities


def run(ver,ledger,span):
    samples=ver_samples(ver,ledger)
    clips={n:reference(n) for n in samples}
    # Test labels are read only by the terminal evaluation; training indexes below never overlap.
    saved=[]
    packed={a:({}, {}, {}) for a in ("geometry","aligned")}
    for target in sorted(samples):
        others=[n for n in samples if clips[n].source!=clips[target].source]
        left,right=span[target]
        keep=[n for n in others if span[n][1]<left-30. or span[n][0]>right+30.]
        cal=[n for n in keep if int(n.rsplit("_c",1)[1])%3==1]
        fit=[n for n in keep if n not in cal]
        assert cal and fit
        x,y,w=training_arrays([clips[n] for n in fit],samples)
        row=dict(test=target,fit=fit,cal=cal,excluded=sorted(set(others)-set(keep)),arms={})
        for arm,(scores,dist,alerts) in packed.items():
            model=HistGradientBoostingClassifier(max_iter=45,max_leaf_nodes=7,l2_regularization=3.,
                min_samples_leaf=35,early_stopping=False,random_state=6151)
            model.fit(features(x,arm),y,sample_weight=w)
            cal_score={n:predict(model,arm,samples[n])[0] for n in cal}
            cut,metrics=fit_cut([clips[n] for n in cal],cal_score)
            scores[target],dist[target]=predict(model,arm,samples[target])
            alerts[target]=scores[target]>=cut
            p=OUT/ver/"time_split"/arm/(target+".npz")
            p.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(p,score=scores[target],distance=dist[target],alert=alerts[target])
            row["arms"][arm]=dict(cut=cut_text(cut),sha256=sha(p),cal_timely=metrics["timely"])
        saved.append(row)
    report={}
    for arm,(scores,dist,alerts) in packed.items():
        for n,c in clips.items():
            c.known=np.isfinite(dist[n])
        report[arm]=counts(list(clips.values()),scores,dist,alerts)
    return dict(splits=saved,results=report)


def main():
    ledger=json.loads((OUT/"features.json").read_text())
    span,identity=intervals([r["name"] for r in ledger["files"]])
    with threadpool_limits(limits=1):
        results={v:run(v,ledger,span) for v in ("v4","v41")}
    dump(OUT/"time_split.json",dict(versions=results,shared_time_s=span,inputs=identity,
         clip_guard_s=30,protocol=sha(HERE/"TIME_SPLIT.md"),code=sha(Path(__file__)),
         features=sha(OUT/"features.json"),original_rank_code=sha(HERE/"rank.py")))
    for v,arms in results.items():
        for a,r in arms["results"].items():
            p=r["operating"]["pooled"]
            print(v,a,r["correct_distance"],p["timely"],p["clear_warnings_per_km"],
                  r["pooled_posthoc"]["timely"],flush=True)

if __name__=="__main__":
    main()
