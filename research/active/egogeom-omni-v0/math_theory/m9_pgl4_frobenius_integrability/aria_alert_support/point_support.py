"""Audit whether predicted returns support the right range before choosing a hazard score."""
import json
from pathlib import Path

import numpy as np
from predict import ART, HERE, OUT, dump, sha
from support import CausalRayReadout, axes


def starts(arr):
    z=np.sort(arr)
    enough=np.searchsorted(z,z+.30,side="right")-np.arange(len(z))>=18
    return z[enough]


def pack(arrays):
    # Store the exact union of target intervals attainable to 0.55 m.
    merged=[]
    for a in arrays:
        z=np.sort(a)
        if not len(z):
            merged.append(np.empty((0,2)))
            continue
        gap=np.flatnonzero(z[1:]-.55>z[:-1]+.55)+1
        merged.append(np.column_stack((z[np.r_[0,gap]]-.55,z[np.r_[gap-1,len(z)-1]]+.55)))
    return np.vstack(merged),np.cumsum([0]+[len(a) for a in merged])


def make_points(version,name):
    stream=ART/"math_theory"/(version+"_stream")/"aria_gen2"/(name+".npz")
    with np.load(stream) as f:
        depth,K,R,t,fw=[f[k] for k in ("depth_s4","K","poses_R","poses_t","strict_fwd")]
    eng=CausalRayReadout()
    gathered={k:[] for k in ("current","any_view","union")}
    for i in range(len(depth)):
        eng.step(depth[i],K[i],R[i],t[i],fw[i])
        forward,side=axes(R[i],fw[i])
        by_view=[]
        for _,_,points in eng.history:
            rel=points-t[i,:2]
            z=rel@forward
            ok=(abs(rel@side)<=.3)&(z>=.3)&(z<=4.)
            by_view.append(z[ok])
        starts_by_view=[starts(a) for a in by_view]
        gathered["current"].append(starts_by_view[-1])
        gathered["any_view"].append(np.concatenate(starts_by_view))
        gathered["union"].append(starts(np.concatenate(by_view)))
    out=OUT/version/"range_hypotheses"/(name+".npz")
    out.parent.mkdir(exist_ok=True)
    arrays={}
    for key,values in gathered.items():
        arrays[key],arrays[key+"_offset"]=pack(values)
    np.savez_compressed(out,**arrays)
    return out


def main():
    ledger=json.loads((OUT/"v5_aria_alert_support.json").read_text())["results"]
    result={}
    for version in ("v4","v41"):
        events=ledger[version]["events"]
        # Include all 19 clips even when an entire clip has no avoidable event.
        names=sorted(p.stem for p in (OUT/version).glob("*.npz"))
        files={n:make_points(version,n) for n in names}
        matches={}
        for name in names:
            src=next((ART/"zeroshot_eval/aria_g2_20clips").rglob(name+".npz"))
            with np.load(src) as truth:
                er=truth["eval_rows"]
            with np.load(files[name]) as obs:
                bags={k:(obs[k],obs[k+"_offset"]) for k in ("current","any_view","union")}
            for i in np.flatnonzero((er[:,0]>=.3)&(er[:,0]<=4)&(er[:,1]>=.6)):
                good={}
                for arm,(pointers,offsets) in bags.items():
                    z=pointers[offsets[i]:offsets[i+1]]
                    good[arm]=bool(((z[:,0]<=er[i,0]) & (er[i,0]<=z[:,1])).any())
                matches[name,int(i)]=good
        event_rows=[]
        for e in events:
            hz=e["timely_hazard_frames"]
            hit={arm:any(matches[e["clip"],i][arm] for i in hz) for arm in ("current","any_view","union")}
            event_rows.append({k:e[k] for k in ("clip","walker","event_start")}|hit|
                              {"blind_live":bool(hz and not e["diagnostics"]["any_now_truth_ray18"])})
        total={arm:sum(r[arm] for r in event_rows) for arm in ("current","any_view","union")}
        blind={arm:sum(r["blind_live"] and r[arm] for r in event_rows) for arm in total}
        result[version]={"event_counts":total,"among_13_live_ray_failures":blind,"events":event_rows,
                         "candidate_sha256":{n:sha(p) for n,p in files.items()}}
        print(version,total,blind,flush=True)
    dump(OUT/"predicted_range_support.json",{"code_sha256":sha(Path(__file__)),
                                             "alignment_sha256":sha(OUT/"v5_aria_alert_support.json"),"versions":result})


if __name__=="__main__":
    main()
