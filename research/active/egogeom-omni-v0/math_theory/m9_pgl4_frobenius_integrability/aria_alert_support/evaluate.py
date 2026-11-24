"""Align GT-only distance capacity with scorer-exact windows after features are archived."""
from collections import Counter, defaultdict
import json
import sys
from pathlib import Path

from predict import ART, HERE, MATH, M6, OUT, dump, sha
from support import axes, best, intervals, world_rays
import numpy as np

M3=MATH/"m3_online_conformal_safety"
sys.path.insert(0,str(M3))
from eval_alert_operating_point import Clip, confirm_score, summarize
from eval_on_v4 import BW_SPEEDS_PROXY, make_episodes, score_episode


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def windows(ep):
    rows,valid=ep["rows"],ep["valid"]
    haz=valid&(rows[:,0]<=4)
    result=[]
    i=0
    while i<len(haz):
        if not haz[i]:
            i+=1
            continue
        j=i
        while j+1<len(haz) and (haz[j+1] or (j+2<len(haz) and haz[j+2] and valid[j+1])):
            j+=1
        eligible=np.arange(i,j+1)[valid[i:j+1] & (rows[i:j+1,0]>=ep["dreq"])]
        if haz[i:j+1].sum()>=2 and len(eligible):
            result.append((i,j,eligible))
        i=j+1
    assert len(result)==score_episode(ep,np.zeros(len(haz),bool))[1]
    return result


def truth_capacity(obs, rows):
    K,R,t,fw = (obs[k] for k in ("K","poses_R","poses_t","strict_fwd"))
    shape=obs["gray_s4"].shape[1:]
    rays=[world_rays(shape,k,r) for k,r in zip(K,R)]
    eligible=(rows[:,0]>=.3)&(rows[:,0]<=4)&(rows[:,1]>=.6)
    caps={}
    for i in np.flatnonzero(eligible):
        forward,side=axes(R[i],fw[i])
        bounds=[intervals(rays[j],t[j]-t[i],forward,side) for j in range(max(0,i-5),i+1)]
        target=rows[i,0]
        one=best(*bounds[-1],target)
        views=max(best(a,b,target) for a,b in bounds)
        all_views=best(np.sort(np.concatenate([v[0] for v in bounds])),
                       np.sort(np.concatenate([v[1] for v in bounds])),target)
        caps[int(i)]={"now_rays":one,"best_single_view_rays":views,"union_rays":all_views}
    return caps


def read_clips(version,caps):
    stream=ART/"math_theory"/(version+"_stream")
    provenance=load(OUT/"predictions.json")["versions"][version]
    clips=[]
    for row in provenance["clips"]:
        name=row["name"]
        fold=int(name.startswith("aria_walk_1"))
        with np.load(OUT/version/(name+".npz")) as p:
            preds={k:p[k] for k in p.files}
        pack=next((ART/"zeroshot_eval/aria_g2_20clips").rglob(name+".npz"))
        with np.load(pack) as labels:
            rows=labels["eval_rows"]
        episodes=make_episodes(name,rows,BW_SPEEDS_PROXY,kind="blind")
        valid=episodes[0]["valid"]
        clear=valid&(rows[:,0]>4)&(rows[:,2]<=.5)
        pos=np.cumsum(np.where(valid,rows[:,3],0.),dtype=float)
        clip=Clip({"name":name},"aria",name.rsplit("_c",1)[0],fold,preds["raw"],
                  confirm_score(preds["raw"]),preds["bev"],episodes,clear,pos)
        for key,val in preds.items():
            setattr(clip,key,val)
        if version=="v4":
            with np.load(stream/"aria_gen2"/(name+".npz")) as obs:
                caps[name]=truth_capacity(obs,rows)
        else:
            other=ART/"math_theory/v4_stream/aria_gen2"/(name+".npz")
            with np.load(other) as a,np.load(stream/"aria_gen2"/(name+".npz")) as b:
                for k in ("K","poses_R","poses_t","strict_fwd"):
                    np.testing.assert_array_equal(a[k],b[k])
        clips.append(clip)
    return clips


def rec(count):
    r=summarize(count)
    r["timely"]=r["avoidable"]-r["late"]
    return r


def threshold_choices(clips, name):
    scores=np.unique(np.concatenate([getattr(c,name) for c in clips]))
    if name=="normalized":
        scores=scores[scores>0]
    return [np.inf,*scores[::-1]]


def fit(clips,field):
    pool=[]
    for cut in threshold_choices(clips,field):
        counts=sum(c.count(getattr(c,field)>=cut) for c in clips)
        s=rec(counts)
        if s["clear_warnings_per_km"]<=5 and s["fa_per_km"]<=5:
            objective=(-s["timely"],s["repeated_clear_warnings"],s["clear_alert_frames"],s["alert_frames"],-cut)
            pool.append((objective,cut,s))
    _,cut,s=min(pool,key=lambda entry:entry[0])
    return cut,s


def groups(clips, actions):
    def collect(items):
        result=rec(sum(c.count(actions[c.obs["name"]]) for c in items))
        unknown_sent=unknown_all=clear_unknown=geom_unknown=0
        known_alerts=0
        for c in items:
            a=actions[c.obs["name"]]
            unknown=~c.episodes[0]["valid"]
            unknown_all+=int(unknown.sum())
            unknown_sent+=int((unknown&a).sum())
            geom_unknown+=int((~c.known).sum())
            clear_unknown+=int((c.clear&~c.known).sum())
            known_alerts+=int((a&c.known).sum())
        clear_m=sum(float(c.episodes[0]["rows"][c.clear,3].sum()) for c in items)
        result.update(known_clear_unique_travel_m=clear_m,known_clear_six_walker_km=6*clear_m/1000,
                      reference_unknown_frames=unknown_all,alerts_on_reference_unknown=unknown_sent,
                      geometry_unknown_frames=geom_unknown,known_clear_geometry_unknown=clear_unknown,
                      alerts_on_geometry_known=known_alerts)
        return result
    return {"pooled":collect(clips),"by_walk":{walk:collect([c for c in clips if c.source==walk])
                 for walk in sorted({c.source for c in clips})}}


def serial_cut(x):
    return "never" if np.isposinf(x) else float(x)


def archived_cut(x):
    return {"always":-np.inf,"never":np.inf}.get(x,x)


def evaluate(version,caps):
    clips=read_clips(version,caps)
    prior=load(ART/"math_theory"/M6.name/("v5_empirical_report.json" if version=="v4" else "v5_empirical_report_v41.json"))
    actions={}
    fits={}
    for arm,field in (("m3_archived","confirmed"),("m6_archived","m6")):
        old=prior["baseline"]["confirmed_shared"] if arm.startswith("m3") else prior["geometry"]
        cuts=[archived_cut(item["training"]["threshold"]) for item in old["fits"]]
        actions[arm]={c.obs["name"]:getattr(c,field)>=cuts[c.fold] for c in clips}
        assert groups(clips,actions[arm])["pooled"]["timely"]==old["domains"]["aria_gen2"]["timely"]
    for field in ("m6","normalized"):
        arm=field+"_walkout"
        actions[arm]={}
        fits[arm]=[]
        for f in (0,1):
            train=[c for c in clips if c.fold!=f]
            test=[c for c in clips if c.fold==f]
            cut,s=fit(train,field)
            fits[arm].append({"test_walk":test[0].source,"training_walk":train[0].source,
                              "clean_budget_per_km":5,"cut":serial_cut(cut),"training":s})
            actions[arm].update({c.obs["name"]:getattr(c,field)>=cut for c in test})
    actions["bev2m"]={c.obs["name"]:c.bev<=2. for c in clips}
    ceilings={}
    for field in ("confirmed","m6","normalized"):
        cut,stats=fit(clips,field)
        ceilings[field]={"cut":serial_cut(cut),"posthoc":stats}
    assert ceilings["m6"]["posthoc"]["timely"]==prior["aria_frontier"]["exact_best_at_5_clear_per_km"]
    scores={arm:groups(clips,masks) for arm,masks in actions.items()}
    ray_ref=load(OUT.parent/"ray_bound.json")
    bounds={(r["clip"],r["frame"]):r for r in ray_ref["rows"]}
    boxed={(r["clip"],r["frame"]):r for r in load(OUT.parent/"box_bound.json")["results"]}
    oracle={(r["clip"],r["frame"]):r for r in load(OUT.parent/"oracle_aria.json")["rows"]}
    events=[]
    unique_clear=np.concatenate([c.m6[c.clear] for c in clips])
    clear_peaks=[float(c.m6[c.clear].max()) for c in clips if c.clear.any()]
    for c in clips:
        name=c.obs["name"]
        rows=c.episodes[0]["rows"]
        valid=c.episodes[0]["valid"]
        for i,cap in caps[name].items():
            assert (cap["now_rays"]<18)==(bounds[(name,i)]["unrestricted_support_upper"]<18)
        for w,ep in enumerate(c.episodes):
            for i,j,ok in windows(ep):
                hz=ok[(rows[ok,0]>=.3)&(rows[ok,0]<=4)]
                hist=[caps[name][k] for k in hz]
                witnessed=[]
                for k in hz:
                    old=oracle[(name,k)]["best"]
                    hit=any(v["error_m"] is not None and v["error_m"]<=.55 for v in old.values())
                    witnessed.append(hit or "witness_coefficients" in boxed.get((name,k),{}))
                peak=float(c.m6[ok].max())
                posthoc=c.normalized>=archived_cut(ceilings["normalized"]["cut"])
                measures={"has_hazard_frame":bool(len(hz)),
                     "normalized_posthoc_hit_any":bool(posthoc[ok].any()),
                     "normalized_posthoc_hit_hazard":bool(posthoc[hz].any()),
                     "has_clear_bridge":bool(c.clear[ok].any()),
                     "only_bridge_timely":bool(not len(hz)),
                     "any_now_truth_ray18":any(g["now_rays"]>=18 for g in hist),
                     "any_history_view_truth_ray18":any(g["best_single_view_rays"]>=18 for g in hist),
                     "any_history_union_truth_ray18":any(g["union_rays"]>=18 for g in hist),
                     "any_v4_pgl_distance_oracle":any(witnessed),
                     "any_v4_certified_impossible_frame":any(bounds[(name,k)]["unrestricted_support_upper"]<18
                           or boxed.get((name,k),{}).get("certified_impossible",False) for k in hz),
                     "any_actual_one_distance_055":bool((np.abs(c.one[hz]-rows[hz,0])<=.55).any()),
                     "any_actual_bev_distance_055":bool((np.abs(c.bev[hz]-rows[hz,0])<=.55).any()),
                     "any_live_obstacle_evidence":bool(c.live_hit[hz].any()),
                     "any_history_obstacle_evidence":bool((c.normalized[hz]>0).any()),
                     "any_geometry_known":bool(c.known[ok].any()),
                     "any_geometry_current_known":bool((c.current_cap[ok]>=18).any())}
                events.append({"clip":name,"source":c.source,"walker":w,"event_start":i,"event_end":j,
                               "dreq_m":ep["dreq"],"timely_frames":ok.tolist(),"timely_hazard_frames":hz.tolist(),
                               "m6_peak":peak,"known_clear_frames_at_least_m6_peak":int(sum(unique_clear>=peak)),
                               "known_clear_clips_at_least_m6_peak":sum(p>=peak for p in clear_peaks),
                               "diagnostics":measures,"detected":{a:bool(mask[name][ok].any()) for a,mask in actions.items()}})
            # Same accepted masks must match the original scorer in every walk profile and clip.
            for arm,masks in actions.items():
                exp=[r for r in events if r["clip"]==name and r["walker"]==w]
                late,avoid,*_=score_episode(ep,masks[name])
                assert len(exp)==avoid and sum(r["detected"][arm] for r in exp)==avoid-late
    assert len(events)==143
    summary={k:sum(r["diagnostics"][k] for r in events) for k in events[0]["diagnostics"]}
    for arm in actions:
        assert sum(r["detected"][arm] for r in events)==scores[arm]["pooled"]["timely"]
    stats={"version":version,"walkout_fits":fits,"operating_points":scores,"aria_posthoc_best":ceilings,
           "event_diagnostics":summary,"unique_avoidable_segments":len({(r["clip"],r["event_start"]) for r in events}),
           "unique_timely_frames":len({(r["clip"],k) for r in events for k in r["timely_frames"]}),
           "unique_timely_hazard_frames":len({(r["clip"],k) for r in events for k in r["timely_hazard_frames"]}),
           "known_clear_clips_above_timely_m6_peak_p0_p50_p100":np.percentile(
               [r["known_clear_clips_at_least_m6_peak"] for r in events],[0,50,100]).tolist(),
           "cpu_readout_ms_p50_p95":np.percentile(np.concatenate([c.time_ms for c in clips]),[50,95]).tolist(),
           "all_frames":sum(len(c.raw) for c in clips),"events":events}
    strata={}
    for c in clips:
        valid=c.episodes[0]["valid"]
        for typ,idx in (("hazard",valid&(c.episodes[0]["rows"][:,0]<=4)),("known_clear",c.clear),
                        ("reference_unknown",~valid)):
            item=strata.setdefault(typ,Counter())
            item["frames"]+=int(idx.sum())
            item["current_geometric_unavailable"]+=int((idx&(c.current_cap<18)).sum())
            item["history_geometric_unavailable"]+=int((idx&~c.known).sum())
            item["current_obstacle_evidence"]+=int((idx&c.live_hit).sum())
            item["history_obstacle_evidence"]+=int((idx&(c.normalized>0)).sum())
    stats["observable_strata"]=strata
    return stats


def main():
    caps={}
    pred_manifest=OUT/"predictions.json"
    results={v:evaluate(v,caps) for v in ("v4","v41")}
    files=[Path(__file__),HERE/"predict.py",HERE/"support.py",M3/"eval_alert_operating_point.py",
           MATH.parents[1]/"egoclear-v0"/"eval_closed_loop.py",OUT.parent/"ray_bound.json",
           OUT.parent/"box_bound.json",OUT.parent/"oracle_aria.json",pred_manifest]
    packs=sorted((ART/"zeroshot_eval/aria_g2_20clips").rglob("*.npz"))
    dump(OUT/"v5_aria_alert_support.json",{"sha256":{p.name:sha(p) for p in files},
              "truth_pack_sha256":{p.name:sha(p) for p in packs},"results":results,"frame_oracle_caps":caps})
    for v,r in results.items():
        print(v,r["event_diagnostics"],{a:s["pooled"]["timely"] for a,s in r["operating_points"].items()},flush=True)


if __name__=="__main__":
    main()
