"""Oracle-independent ray support necessary condition, with enlarged float32 bounds."""
import json
from pathlib import Path

import numpy as np
from eval_empirical import ART, OUTPUT, sha, store


def upper_count(depth, intr, rot, heading, truth, bounded):
    # Same ray centres as frozen corridor readout. Shape corrections use their own head coordinates.
    v,u = np.indices(depth.shape,dtype=float)
    x,y = (4*u+2-intr[0,2])/intr[0,0], (4*v+2-intr[1,2])/intr[1,1]
    ray = np.stack((x,y,np.ones_like(x)),axis=-1)@rot.T
    fw=np.array([*heading[:2],0.],float)
    if np.linalg.norm(fw)<1e-4:
        fw = rot[:,2].copy()
        fw[2] = 0.
    fw = fw/max(np.linalg.norm(fw),1e-6)
    f = ray@fw
    lateral = np.abs(ray@np.array([fw[1],-fw[0],0.]))
    eps = .0001  # metre margins far larger than FP32 roundoff in this capped range
    zlo = np.full_like(f,.25-eps)
    zhi = np.full_like(f,10.+eps)
    if bounded:
        # Boxes for a, b0, bx, by in oracle.py; independent interval relaxation per ray.
        xx,yy = (4*u+.5-intr[0,2])/intr[0,0], (4*v+.5-intr[1,2])/intr[1,1]
        extra = .15 + .12*(np.abs(xx)+np.abs(yy)) + eps
        denom_lo = .4/depth-extra
        denom_hi = 2.5/depth+extra
        zlo = np.maximum(zlo,1/denom_hi-eps)
        zhi = np.minimum(zhi,np.where(denom_lo>eps,1/np.maximum(denom_lo,eps)+eps,np.inf))
    zhi = np.minimum(zhi,(.30+eps)/np.maximum(lateral-eps,1.e-12))
    d0 = np.maximum(.30-eps,(f-eps)*zlo)
    d1 = np.minimum(4.+eps,(f+eps)*zhi)
    lower = np.maximum(max(.30, truth-.55)-eps,d0-.30-eps)
    upper = np.minimum(min(4., truth+.55)+eps,d1+eps)
    ok = (zhi>=zlo)&(f+eps>0)&(d1>=d0)&(upper>=lower)
    starts, ends = np.sort(lower[ok]), np.sort(upper[ok])
    if not len(starts):
        return 0
    return int(np.max(np.arange(1,len(starts)+1)-np.searchsorted(ends,starts,side="left")))


def main():
    path = OUTPUT/"oracle_aria.json"
    oracle = json.loads(path.read_text())
    by_name = {}
    for row in oracle["rows"]:
        by_name.setdefault(row["clip"],[]).append(row)
    cohort = []
    for name, rows in by_name.items():
        with np.load(ART/"math_theory/v4_stream/aria_gen2"/(name+".npz")) as obs:
            depth,K,R,fw = (obs[k] for k in ("depth_s4","K","poses_R","strict_fwd"))
        for row in rows:
            i=row["frame"]
            args = depth[i],K[i],R[i],fw[i],row["target_m"]
            unrestricted = upper_count(*args,False)
            bounded = upper_count(*args,True)
            best = min(x["error_m"] for x in row["best"].values() if x["error_m"] is not None)
            if best<=.55:
                assert unrestricted>=18 and bounded>=18
            cohort.append({"clip":name,"frame":i,"unrestricted_support_upper":unrestricted,"box_support_upper":bounded,
                           "certified_scalar_failure":row["continuous_scalar_impossible"],"grid_error_m":best})
    summary={}
    for tag in ("unrestricted_support_upper","box_support_upper"):
        summary[tag] = {"impossible":sum(r[tag]<18 for r in cohort),
                        "impossible_in_certified_scalar13":sum(r[tag]<18 and r["certified_scalar_failure"] for r in cohort)}
    store(OUTPUT/"ray_bound.json",{"oracle_sha256":sha(path),"code_sha256":sha(Path(__file__)),
                                  "summary":summary,"rows":cohort})
    print(summary)


if __name__=="__main__":
    main()
