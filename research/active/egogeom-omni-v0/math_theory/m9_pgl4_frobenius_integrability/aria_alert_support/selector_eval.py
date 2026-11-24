"""Score the two archived range selectors under the original event and clear-road rules."""
from pathlib import Path
import numpy as np

from evaluate import Clip, BW_SPEEDS_PROXY, groups, load, make_episodes, rec, windows
from predict import ART, HERE, OUT, dump, sha


def clips_for(version, receipt):
    clips=[]
    for item in receipt["versions"][version]:
        name=item["name"]
        path=OUT/version/"selector_predictions"/(name+".npz")
        assert sha(path)==item["prediction_sha256"]
        with np.load(path) as p:
            d={k:p[k] for k in ("nearest","persistent")}
        with np.load(OUT/version/(name+".npz")) as p:
            raw,known=p["raw"],p["known"]
        source=next((ART/"zeroshot_eval/aria_g2_20clips").rglob(name+".npz"))
        with np.load(source) as p:
            rows=p["eval_rows"]
        episodes=make_episodes(name,rows,BW_SPEEDS_PROXY,kind="blind")
        valid=episodes[0]["valid"]
        clear=valid&(rows[:,0]>4)&(rows[:,2]<=.5)
        c=Clip({"name":name},"aria",name.rsplit("_c",1)[0],int(name.startswith("aria_walk_1")),
               raw,raw,d["nearest"],episodes,clear,np.cumsum(np.where(valid,rows[:,3],0.),dtype=float))
        c.known=known
        c.nearest,c.persistent=d["nearest"],d["persistent"]
        clips.append(c)
    return clips


def alarms(clips,field,cut):
    return {c.obs["name"]:getattr(c,field)<=cut for c in clips}


def fit(clips,field):
    all_d=np.concatenate([getattr(c,field) for c in clips])
    options=[]
    for cut in [-np.inf,*np.unique(all_d[np.isfinite(all_d)])]:
        masks=alarms(clips,field,cut)
        metric=rec(sum(c.count(masks[c.obs["name"]]) for c in clips))
        if metric["fa_per_km"]<=5 and metric["clear_warnings_per_km"]<=5:
            objective=(-metric["timely"],metric["repeated_clear_warnings"],metric["clear_alert_frames"],
                       metric["alert_frames"],float(cut))
            options.append((objective,cut,metric))
    _,cut,result=min(options,key=lambda e:e[0])
    return cut,result


def actions(clips,masks):
    result=groups(clips,masks)
    for key,rows in [("pooled",clips),*[(s,[c for c in clips if c.source==s]) for s in result["by_walk"]]]:
        start=segments=0
        for c in rows:
            a=masks[c.obs["name"]]
            start+=int(np.sum(a&~np.r_[False,a[:-1]]&c.clear))
            chunks=np.split(np.arange(len(a)),np.flatnonzero(np.diff(c.clear))+1)
            segments+=sum(bool(c.clear[chunk[0]] and a[chunk].any()) for chunk in chunks)
        record=result["pooled"] if key=="pooled" else result["by_walk"][key]
        record.update(physical_alarm_starts_on_clear=start,physical_clear_segments_charged=segments)
    return result


def evaluate(version,receipt):
    clips=clips_for(version,receipt)
    rows=np.concatenate([c.episodes[0]["rows"] for c in clips])
    eligible=(rows[:,0]>=.3)&(rows[:,0]<=4)&(rows[:,1]>=.6)
    clear=np.concatenate([c.clear for c in clips])
    unknown=np.concatenate([~c.episodes[0]["valid"] for c in clips])
    out={f:np.concatenate([getattr(c,f) for c in clips]) for f in ("nearest","persistent")}
    common=eligible&np.isfinite(out["nearest"])&np.isfinite(out["persistent"])
    results={}
    event_ledger=[]
    for c in clips:
        truth=c.episodes[0]["rows"][:,0]
        for w,ep in enumerate(c.episodes):
            for start,end,ok in windows(ep):
                hz=ok[(truth[ok]>=.3)&(truth[ok]<=4.)]
                event_ledger.append(dict(clip=c.obs["name"],walker=w,event_start=start,event_end=end,
                    timely_frames=ok.tolist(),timely_hazard_frames=hz.tolist(),
                    **{field:bool((np.abs(getattr(c,field)[hz]-truth[hz])<=.55).any()) for field in out}))
    assert len(event_ledger)==143 and sum(bool(r["timely_hazard_frames"]) for r in event_ledger)==129
    for field,d in out.items():
        cv={}
        fits=[]
        for fold in (0,1):
            train=[c for c in clips if c.fold!=fold]
            test=[c for c in clips if c.fold==fold]
            cut,training=fit(train,field)
            fits.append({"test_walk":test[0].source,"training_walk":train[0].source,
                         "cut_m":float(cut) if np.isfinite(cut) else "off","training":training})
            cv.update(alarms(test,field,cut))
        cut,summary=fit(clips,field)
        both=alarms(clips,field,cut)
        for r in event_ledger:
            r[field+"_walkout_alert"]=bool(cv[r["clip"]][r["timely_frames"]].any())
            r[field+"_oracle_alert"]=bool(both[r["clip"]][r["timely_frames"]].any())
        metrics=actions(clips,cv)
        assert sum(r[field+"_walkout_alert"] for r in event_ledger)==metrics["pooled"]["timely"]
        finite=np.isfinite(d)
        results[field]={
            "finite_n":int((finite&eligible).sum()),"eligible_n":int(eligible.sum()),
            "finite_mae_m":float(np.abs(d[finite&eligible]-rows[finite&eligible,0]).mean()),
            "filled6_mae_m":float(np.abs(np.where(finite[eligible],d[eligible],6.)-rows[eligible,0]).mean()),
            "common_n":int(common.sum()),"common_mae_m":float(np.abs(d[common]-rows[common,0]).mean()),
            "known_clear_has_range_n":int((finite&clear).sum()),"reference_unknown_has_range_n":int((finite&unknown).sum()),
            "range_hits_among_129":sum(r[field] for r in event_ledger),"walkout":metrics,"fits":fits,
            "fixed_2m":actions(clips,alarms(clips,field,2.)),
            "posthoc_cut_m":float(cut) if np.isfinite(cut) else "off","posthoc":actions(clips,both)
        }
    pairs={k:sum((r["nearest"],r["persistent"])==v for r in event_ledger if r["timely_hazard_frames"])
           for k,v in {"both":(True,True),"nearest_only":(True,False),"persistent_only":(False,True),"neither":(False,False)}.items()}
    return {"arms":results,"paired_distance_hits":pairs,"events":event_ledger}


def main():
    path=OUT/"selector_predictions.json"
    receipt=load(path)
    assert sha(HERE/"SELECTOR.md")==receipt["protocol_sha256"]
    result={ver:evaluate(ver,receipt) for ver in ("v4","v41")}
    dump(OUT/"selector_report.json",{"versions":result,"prediction_receipt_sha256":sha(path),
                                     "code_sha256":sha(Path(__file__))})
    for v,row in result.items():
        print(v,row["paired_distance_hits"],{a:r["walkout"]["pooled"]["timely"] for a,r in row["arms"].items()})


if __name__=="__main__":
    main()
