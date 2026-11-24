"""Check full candidate identities on two clips, exact threshold bounds, and paired changes."""
import json
from pathlib import Path

import numpy as np
from build import ART, GEOM, HERE, OLD, OUT, VIS, dump, sha
from point_support import starts
from rank import ARMS, fit_cut, reference, rec
from support import CausalRayReadout, axes
from threadpoolctl import threadpool_limits


def compare_cells(ver,name):
    with np.load(ART/"math_theory"/(ver+"_stream")/"aria_gen2"/(name+".npz")) as f:
        dep,K,R,t,fw=[f[k] for k in ("depth_s4","K","poses_R","poses_t","strict_fwd")]
    with np.load(OUT/ver/(name+".npz")) as f:
        rows,off=f["features"],f["offsets"]
    bank=CausalRayReadout()
    for i in range(len(dep)):
        bank.step(dep[i],K[i],R[i],t[i],fw[i])
        forward,side=axes(R[i],fw[i])
        starts_all=[]
        for _,_,pts in bank.history:
            rel=pts-t[i,:2]
            z=rel@forward
            use=(np.abs(rel@side)<=.30)&(z>=.30)&(z<=4.)
            starts_all.extend(starts(z[use]))
        expected=np.unique(np.floor(np.array(starts_all)/.05+.5).astype(int))
        observed=np.rint(rows[off[i]:off[i+1],0]/.05).astype(int)
        np.testing.assert_array_equal(observed,expected)
    return len(dep)


def exhaustive(clips,scores):
    pool=np.unique(np.hstack([scores[c.obs["name"]] for c in clips]))
    best=None
    for cut in np.r_[np.inf,pool[np.isfinite(pool)]]:
        met=rec(sum(c.count(scores[c.obs["name"]]>=cut) for c in clips))
        if met["fa_per_km"]<=5 and met["clear_warnings_per_km"]<=5:
            order=(-met["timely"],met["repeated_clear_warnings"],met["clear_alert_frames"],
                   met["alert_frames"],-cut)
            if best is None or order<best[0]:
                best=(order,cut)
    return best[1]


def main():
    report=json.loads((OUT/"results.json").read_text())["versions"]
    checks={}
    with threadpool_limits(limits=1):
        for ver,arms in report.items():
            frames=sum(compare_cells(ver,f"aria_walk_{i}_c00") for i in (0,1))
            names=sorted(p.stem for p in (OUT/ver).glob("*.npz"))
            clips=[reference(n) for n in names]
            verified={}
            for a in ARMS:
                scores={}
                for n in names:
                    with np.load(OUT/ver/a/(n+".npz")) as f:
                        scores[n]=f["score"]
                cut,_=fit_cut(clips,scores)
                assert exhaustive(clips,scores)==cut
                verified[a]=float(cut) if np.isfinite(cut) else "off"
            counts={k:[] for k in ("hazard_matching","hazard_nonmatching","clear")}
            for c in clips:
                with np.load(OUT/ver/(c.obs["name"]+".npz")) as f:
                    x,off=f["features"],f["offsets"]
                gt=c.episodes[0]["rows"][:,0]
                for i in np.flatnonzero(c.episodes[0]["valid"]):
                    b=x[off[i]:off[i+1]]
                    if not len(b):
                        continue
                    right=np.abs(b[:,0]-gt[i])<=.55
                    groups=({"clear":b} if c.clear[i] else
                            {"hazard_matching":b[right],"hazard_nonmatching":b[~right]} if .30<=gt[i]<=4 else {})
                    for grp,items in groups.items():
                        if len(items):
                            counts[grp].append(items[:,len(GEOM)+VIS.index("reliable_flow")].mean())
            pairs=list(zip(arms["geometry"]["events"],arms["aligned"]["events"]))
            rescued=[(b["clip"],b["start"]) for a,b in pairs if b["distance_hit"] and not a["distance_hit"]]
            lost=[(a["clip"],a["start"]) for a,b in pairs if a["distance_hit"] and not b["distance_hit"]]
            checks[ver]=dict(full_candidate_frames_checked=frames,all_cuts_brute_checked=verified,
                reliable_fraction_frame_equal={k:dict(frames=len(y),mean=float(np.mean(y))) for k,y in counts.items()},
                rescued_combinations=len(rescued),rescued_distinct_events=sorted(set(rescued)),
                lost_combinations=len(lost),lost_distinct_events=sorted(set(lost)))
    dump(OUT/"score_audit.json",dict(versions=checks,results=sha(OUT/"results.json"),code=sha(Path(__file__))))
    print(json.dumps(checks,indent=1))

if __name__=="__main__":
    main()
