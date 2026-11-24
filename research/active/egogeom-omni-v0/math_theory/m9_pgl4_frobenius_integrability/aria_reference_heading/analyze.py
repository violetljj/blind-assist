"""Evaluate alternative references on fixed frames/events and also recount each new task."""
from collections import Counter
import hashlib
import json
import sys

import numpy as np

from rebuild import ART, HERE, M9, OUT, ROUTE, STAGES, sha

SOURCE = ROUTE.parent/"egoclear-v0"
sys.path.insert(0,str(SOURCE))
from eval_closed_loop import H, make_episodes, score_episode

ARMS = STAGES
LABELS = ("near","clear","unknown")
# Frozen proxy in eval_cross_glasses_zeroshot.py. Cross-check resulting event
# identifiers, participant dreq and eligible frame indices with the archived scorer.
SPEEDS={f"p{i}":np.linspace(.78+.04*i,1.15+.04*i,10).tolist() for i in range(6)}
PAIRS=(("frozen","strict_full"),("strict_full","strict_past_points_full_ground"),
       ("strict_past_points_full_ground","strict_past"),("strict_full","strict_past"),
       ("frozen","strict_past"),("frozen","centered_past"),("centered_past","strict_past"))


def label(rows):
    d,c=rows[:,:2].T
    valid=(~np.isnan(d)) & (c>=.6)
    return np.where(valid,np.where(d<=H,0,1),2)


def percentile(arr):
    if not len(arr):
        return {"n":0}
    return {"n":len(arr),"mean":float(np.mean(arr)),
            **dict(zip(("p50","p90","p95","max"),map(float,np.percentile(arr,[50,90,95,100]))))}


def times(ep):
    rows,valid=ep["rows"],ep["valid"]
    hazard=valid & (rows[:,0]<=H)
    out=[];i=0
    while i<len(rows):
        if not hazard[i]:
            i+=1;continue
        j=i
        while j+1<len(rows) and (hazard[j+1] or (j+2<len(rows) and hazard[j+2] and valid[j+1])):
            j+=1
        ix=np.arange(i,j+1)
        if hazard[ix].sum()>=2:
            timely=ix[valid[ix] & (rows[ix,0]>=ep["dreq"])]
            if len(timely):
                out.append((i,j,timely.tolist(),timely[hazard[timely]].tolist()))
        i=j+1
    assert len(out)==score_episode(ep,np.zeros(len(rows),bool))[1]
    return out


def fixed_frames(clips,s0,s1):
    a,b=(np.concatenate([c[k] for c in clips]) for k in (s0,s1))
    ia,ib=label(a),label(b)
    matrix=np.bincount(3*ia+ib,minlength=9).reshape(3,3)
    both=(ia==0)&(ib==0)
    delta=b[both,0]-a[both,0]
    angles=np.concatenate([c["heading_angle_deg"] for c in clips])
    r=dict(matrix_rows_before_columns_after=matrix.tolist(),categories=LABELS,
           common_near_absolute_distance_shift_m=percentile(abs(delta)),
           common_near_signed_distance_shift_m=percentile(delta),
           common_near_distance_shift_over_055=int((abs(delta)>.55).sum()),
           common_near_distance_shift_over_01=int((abs(delta)>.1).sum()),
           fixed_near_retains_within_055=int((abs(delta)<=.55).sum()),
           angles_deg_for_near_to_clear=percentile(angles[(ia==0)&(ib==1)]),
           angles_deg_for_clear_to_near=percentile(angles[(ia==1)&(ib==0)]),
           angles_deg_for_originally_known=percentile(angles[ia!=2]))
    if s0=="frozen" and s1=="strict_full":
        d=(ia!=ib)&(ia!=2)&(ib!=2)
        start_flips=0
        for c in clips:
            x,y=label(c[s0])[:2],label(c[s1])[:2]
            start_flips+=int(np.sum((x!=y)&(x!=2)&(y!=2)))
        r["start_of_clip_hazard_clear_flips"]=start_flips
        r["all_hazard_clear_flips"]=int(d.sum())
    return r


def event_stats(clips,archive):
    fresh={}; paired={}; samples={}
    for arm in ("frozen",*ARMS):
        fresh_stats=Counter()
        fixed_stats=Counter()
        for c in clips:
            rows=c[arm]
            labels=label(rows)
            eps=make_episodes(c["name"],rows,SPEEDS,kind="blind")
            for walker,ep in enumerate(eps):
                new=times(ep)
                fresh_stats["avoidable"]+=len(new)
                fresh_stats["has_hazard_timely"]+=sum(bool(haz) for _,_,_,haz in new)
                fresh_stats["only_clear_bridge_timely"]+=sum(not haz for _,_,_,haz in new)
                timely_set={i for _,_,ok,_ in new for i in ok}
                hazard_set={i for _,_,_,hh in new for i in hh}
                old=archive.get((c["name"],walker),[])
                if arm=="frozen":
                    assert [(lo,hi,ok,haz) for lo,hi,ok,haz in new]==[
                        (v["event_start"],v["event_end"],v["timely_frames"],v["timely_hazard_frames"]) for v in old]
                for v in old:
                    assert abs(ep["dreq"]-v["dreq_m"])<1e-12
                    old_h=set(v["timely_hazard_frames"])
                    eligible=set(v["timely_frames"])
                    iv=np.array(sorted(old_h),dtype=int)
                    vv=np.array(sorted(eligible),dtype=int)
                    fixed_stats["original_143"]+=1
                    fixed_stats["same_old_timely_index_in_any_new_timely_window"]+=bool(eligible&timely_set)
                    fixed_stats["same_old_timely_index_in_new_hazard_timely_window"]+=bool(eligible&hazard_set)
                    fixed_stats["some_frame_in_original_interval_in_any_new_timely_window"]+=bool(set(range(v["event_start"],v["event_end"]+1))&timely_set)
                    fixed_stats["original_14_only_bridge"]+=not old_h
                    fixed_stats["original_14_with_some_old_timely_index_in_new_hazard_timely_window"]+=(not old_h and bool(eligible&hazard_set))
                    if len(iv):
                        fixed_stats["original_129"]+=1
                        fixed_stats["same_original_hazard_timely_index_still_near"]+=bool(np.any(labels[iv]==0))
                        fixed_stats["same_original_hazard_timely_index_still_near_and_above_dreq"]+=bool(
                            np.any((labels[iv]==0)&(rows[iv,0]>=ep["dreq"])))
                        fixed_stats["same_original_hazard_timely_index_still_in_new_hazard_timely_window"]+=bool(old_h&hazard_set)
                        fixed_stats["same_original_hazard_timely_index_still_within_055"]+=bool(
                            np.any((labels[iv]==0)&(np.abs(rows[iv,0]-c["frozen"][iv,0])<=.55)))
                    samples.setdefault(arm,[]).append({"clip":c["name"],"walker":walker,"start":v["event_start"],
                         "old_timely_indices":vv.tolist(),"new_labels_on_old_timely_indices":labels[vv].tolist(),
                         "same_old_hazard_new_timely_hazard_indices":sorted(old_h&hazard_set)})
        fresh[arm]=dict(fresh_stats);paired[arm]=dict(fixed_stats)
    return fresh,paired,samples


def main():
    path=ART/"math_theory"/M9.name/"aria_alert_support/v5_aria_alert_support.json"
    old=json.loads(path.read_text(encoding="utf-8"))
    events=old["results"]["v4"]["events"]
    assert len(events)==143 and sum(bool(v["timely_hazard_frames"]) for v in events)==129
    archive={}
    for v in events:
        archive.setdefault((v["clip"],v["walker"]),[]).append(v)
    clips=[];receipts={}
    build_source=(HERE/"rebuild.py").read_bytes()
    build_hashes={hashlib.sha256(build_source+tail).hexdigest() for tail in (b"",b"\n")}
    for file in sorted(OUT.glob("aria_walk_*.npz")):
        rec=json.loads(file.with_suffix(".json").read_text(encoding="utf-8"))
        assert rec["rebuild_sha256"] in build_hashes
        assert old["truth_pack_sha256"][file.name]==rec["frozen_pack_sha256"]
        assert sha(file)==rec["output_sha256"]
        receipts[file.stem]=sha(file.with_suffix(".json"))
        with np.load(file,allow_pickle=False) as f: c={k:f[k] for k in f.files}
        clips.append(dict(name=file.stem,walk=rec["walk"],**c))
    assert len(clips)==19
    result={}
    for group in ("pooled","walk_0","walk_1"):
        chosen=clips if group=="pooled" else [c for c in clips if c["walk"]==group]
        fresh,fixed,samples=event_stats(chosen,archive)
        result[group]={"frame_transitions":{s0+"=>"+s1:fixed_frames(chosen,s0,s1) for s0,s1 in PAIRS},
                       "recounted_events":fresh,"fixed_original_events":fixed}
        if group=="pooled":
            result["paired_event_detail"]=samples
    assert np.sum(result["pooled"]["frame_transitions"]["frozen=>strict_full"]["matrix_rows_before_columns_after"],axis=1).tolist()==[840,954,1056]
    out={"frames":2850,"clips":19,"receipt_sha256":receipts,"results":result,
         "source_sha256":{p.name:sha(p) for p in (path,HERE/"rebuild.py",HERE/"analyze.py",SOURCE/"eval_closed_loop.py",
                                                 ROUTE/"eval_cross_glasses_zeroshot.py")}}
    (OUT/"reference_comparison.json").write_text(json.dumps(out,indent=1,allow_nan=False)+"\n",encoding="utf-8")
    print(result["pooled"]["frame_transitions"]["frozen=>strict_full"],flush=True)


if __name__=="__main__":
    main()
