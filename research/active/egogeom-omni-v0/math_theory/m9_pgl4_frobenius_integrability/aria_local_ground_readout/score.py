"""Score frozen outputs, applying original episodes and both clear-road budgets."""
import json
import sys
from pathlib import Path
from readout import HERE,MATH,M9,OUT,PACK,dump,sha
import numpy as np

sys.path.insert(0,str(M9/"aria_alert_support"))
from selector_eval import actions,fit,Clip,BW_SPEEDS_PROXY,make_episodes,windows


def clip_for(name,key,labels):
    rows=labels[name]
    eps=make_episodes(name,rows,BW_SPEEDS_PROXY,kind="blind")
    valid=eps[0]["valid"]
    clear=valid&(rows[:,0]>4)&(rows[:,2]<=.5)
    path=OUT/"output"/key/(name+".npz")
    with np.load(path) as f:
        d,cov,support=f["readout"].T
        timing=f["time_ms"]
    finite=np.isfinite(d)
    known=cov>=.6
    pred=np.where(known&finite,d,np.inf)
    assert not np.any(known & np.isnan(d)) and np.all((support[finite]>=8)&(d[finite]>=.3)&(d[finite]<=4))
    c=Clip({"name":name},"aria",name.rsplit("_c",1)[0],int(name.startswith("aria_walk_1")),
           pred,pred,pred,eps,clear,np.cumsum(np.where(valid,rows[:,3],0.),dtype=float))
    c.known=known
    c.nearest=pred
    c.cover=cov
    c.time=timing
    c.rawfinite=finite
    return c


def stat(clips):
    names=["hazard","clear","unknown"]
    matrix=np.zeros((3,3),int)
    raw=np.zeros(3,int)
    for c in clips:
        rows=c.episodes[0]["rows"]
        hazard=c.episodes[0]["valid"]&(rows[:,0]<=4)
        a=np.where(hazard,0,np.where(c.clear,1,2))
        b=np.where(np.isfinite(c.nearest),0,np.where(c.known,1,2))
        np.add.at(matrix,(a,b),1)
        np.add.at(raw,a[c.rawfinite],1)
    return dict(row_and_column_order=names,counts=matrix.tolist(),raw_range_before_coverage=raw.tolist())


def ranges(clips,common):
    error=[]; finite_error=[]; common_error=[]
    for c in clips:
        name=c.obs["name"]
        rows=c.episodes[0]["rows"]
        hazard=c.episodes[0]["valid"] & (rows[:,0]>=.3) & (rows[:,0]<=4)
        keep=hazard&np.isfinite(c.nearest)
        error.extend(abs(np.where(np.isfinite(c.nearest[hazard]),c.nearest[hazard],6.)-rows[hazard,0]))
        finite_error.extend(abs(c.nearest[keep]-rows[keep,0]))
        both=hazard&common[name]
        common_error.extend(abs(c.nearest[both]-rows[both,0]))
    return {k:{"n":len(x),"mae_m":float(np.mean(x)) if len(x) else None} for k,x in
            [("fill_6",error),("finite",finite_error),("three_way_common",common_error)]}


def main():
    meta=json.loads((OUT/"inventory.json").read_text())
    labels={}
    for n in meta["clips"]:
        with np.load(PACK/(n+".npz")) as p:
            labels[n]=p["eval_rows"]
    names=list(labels)
    arms=("model","scaled","stereo")
    result={}
    by_arm={}
    ledgers={}
    receipts={}
    for arm in arms:
        held=[]; fits={}; cv={}
        def cachepath(n,key):
            p=OUT/"output"/key/(n+".npz")
            receipts[str(p.relative_to(OUT))]=sha(p)
        for name in names:
            fold=meta["folds"][name]
            k=fold["scale_key"] if arm=="scaled" else arm
            for n in [name,*fold["train"]]:
                a,b=meta["clips"][name],meta["clips"][n]
                if n!=name:
                    assert a["walk"]!=b["walk"] and (a["utc_stop_ns"]+30_000_000_000<=b["utc_start_ns"]
                                        or b["utc_stop_ns"]+30_000_000_000<=a["utc_start_ns"])
                cachepath(n,k)
            train=[clip_for(n,k,labels) for n in fold["train"]]
            heldout=clip_for(name,k,labels)
            cut,training=fit(train,"nearest")
            fits[name]={"cut_m":float(cut) if np.isfinite(cut) else None,"training":training}
            cv[name]=heldout.nearest<=cut
            held.append(heldout)
        by_arm[arm]=held
        fixed={str(d):actions(held,{c.obs["name"]:c.nearest<=d for c in held}) for d in (2,4)}
        operating=actions(held,cv)
        events=[]
        for c in held:
            n=c.obs["name"]
            y=c.episodes[0]["rows"][:,0]
            for w,ep in enumerate(c.episodes):
                for begin,end,ok in windows(ep):
                    hz=ok[(y[ok]>=.30)&(y[ok]<=4.)]
                    events.append({"clip":n,"walker":w,"start":begin,"end":end,"timely_frames":ok.tolist(),
                        "timely_hazard_frames":hz.tolist(),"range_match":bool((np.abs(c.nearest[hz]-y[hz])<=.55).any()),
                        "heldout_any":bool(cv[n][ok].any()),"heldout_hazard":bool(cv[n][hz].any()),
                        **{f"fixed_{v}m":bool((c.nearest[ok]<=v).any()) for v in (2,4)},
                        **{f"fixed_{v}m_hazard":bool((c.nearest[hz]<=v).any()) for v in (2,4)}})
        assert len(events)==operating["pooled"]["avoidable"]==143
        assert sum(bool(e["timely_hazard_frames"]) for e in events)==129
        assert sum(e["heldout_any"] for e in events)==operating["pooled"]["timely"]
        strata=stat(held)
        assert np.array_equal(np.array(strata["counts"]).sum(axis=1),[840,954,1056])
        result[arm]={"heldout":operating,"fits":fits,"fixed_8_voxel":fixed,"states":strata,
                     "states_by_walk":{s:stat([c for c in held if c.source==s]) for s in sorted({c.source for c in held})},
                     "geometry_cpu_ms_p50_p95":np.percentile(np.concatenate([c.time for c in held]),[50,95]).tolist(),
                     "hazard_only_timely_heldout":sum(e["heldout_hazard"] for e in events),
                     "range_hits_in_129_events":sum(e["range_match"] for e in events)}
        ledgers[arm]=events
    common={n:np.logical_and.reduce([np.isfinite(by_arm[a][i].nearest) for a in arms]) for i,n in enumerate(names)}
    for a in arms:
        result[a]["range"]=ranges(by_arm[a],common)
    dump(OUT/"results.json",{"arms":result,"prediction_sha256":receipts,"events":ledgers,
         "truth_pack_sha256":{n:sha(PACK/(n+".npz")) for n in names},"inventory_sha256":sha(OUT/"inventory.json"),
         "code_sha256":{p.name:sha(p) for p in (Path(__file__),M9/"aria_alert_support/selector_eval.py",
           M9/"aria_alert_support/evaluate.py",MATH/"m3_online_conformal_safety/eval_alert_operating_point.py",
           MATH/"m3_online_conformal_safety/eval_on_v4.py")}})
    for a,r in result.items():
        print(a,"held",r["heldout"]["pooled"]["timely"],r["heldout"]["pooled"]["fa_per_km"],
              r["heldout"]["pooled"]["clear_warnings_per_km"],"range",r["range"],"states",r["states"],flush=True)


if __name__=="__main__":
    main()
