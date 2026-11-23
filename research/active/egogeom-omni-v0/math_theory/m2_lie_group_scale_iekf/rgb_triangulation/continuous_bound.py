"""Continuous-scale necessary condition for an 18-point corridor hit near truth.

Relax ground/obstacle-height rejection and first-hit ordering. A negative result
certifies impossibility; a positive result is deliberately inconclusive.
"""
import json
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parents[6]/"artifacts.local/work/egogeom-omni-v0"
STREAM = ART/"math_theory/v4_stream"
DEST = ART/"math_theory/m2_lie_group_scale_iekf"


def possible(depth,K,R,heading,target,tol=.55):
    v,u = np.mgrid[:depth.shape[0],:depth.shape[1]]
    z = depth.ravel().astype(float)
    unit = np.isfinite(z) & (z>0)
    z,u,v = z[unit],(4*u.ravel()[unit]+2.),(4*v.ravel()[unit]+2.)
    xyz = np.column_stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z)) @ R.T
    fw = np.array([*heading[:2],0.],dtype=float)
    if np.linalg.norm(fw)<1.e-4:
        fw = R[:,2].copy()
        fw[2] = 0.
    fw /= max(np.linalg.norm(fw),1.e-6)
    f = xyz@fw
    x = abs(xyz@np.array([fw[1],-fw[0],0.]))
    # Enlarge all bounds by a small epsilon for float32 projection arithmetic.
    eps = 2.e-5
    keep = f>.1
    f,x,z = f[keep],x[keep],z[keep]
    a = np.maximum.reduce((np.full_like(z,.4-eps),(.25-eps)/z,(.30-eps)/f))
    b = np.minimum.reduce((np.full_like(z,2.5+eps),(10+eps)/z,(4+eps)/f,(.30+eps)/np.maximum(x,1.e-10)))
    viable = np.flatnonzero(a<=b)
    idx = viable[np.argsort(f[viable])]
    a,b,f = a[idx],b[idx],f[idx]
    lower = np.maximum(a,(target-tol-eps)/f)
    upper = np.minimum(b,(target+tol+eps)/f)
    for j in np.flatnonzero(lower<=upper):
        # Include ties within the numerical tolerance even if they precede j.
        fst = np.searchsorted(f,f[j]-eps/lower[j])
        stop = np.searchsorted(f,f[j]+(.30+eps)/lower[j],side="right")
        if stop-fst<18:
            continue
        sl = slice(fst,stop)
        left = np.maximum(lower[j],a[sl])
        right = np.minimum(np.minimum(upper[j],b[sl]),(.30+eps)/np.maximum(f[sl]-f[j],1.e-10))
        ok = left <= right
        if ok.sum()<18:
            continue
        # At each left endpoint count closed intervals alive there.
        starts = np.sort(left[ok])
        ends = np.sort(right[ok])
        overlap = np.arange(1,len(starts)+1)-np.searchsorted(ends,starts,side="left")
        if np.max(overlap)>=18:
            return True
    return False


def main():
    failures,total = [],0
    for src in sorted((ART/"zeroshot_eval/aria_g2_20clips").rglob("*.npz")):
        with np.load(src) as truth:
            er = truth["eval_rows"]
        with np.load(STREAM/"aria_gen2"/src.name) as c:
            dep,K,R,fw = (c[k] for k in ("depth_s4","K","poses_R","strict_fwd"))
            dbase = c["d_1f_strict"]
        for i in np.flatnonzero((er[:,0]>=.30)&(er[:,0]<=4)&(er[:,1]>=.6)):
            total += 1
            if not possible(dep[i],K[i],R[i],fw[i],er[i,0]):
                assert not abs(dbase[i]-er[i,0])<=.55
                failures.append({"clip":src.stem,"frame":int(i),"baseline_finite":bool(np.isfinite(dbase[i]))})
    report = {"eligible":total,"certified_no_scalar_within_0.55m":len(failures),
              "certified_on_baseline_finite":sum(e["baseline_finite"] for e in failures),"witnesses":failures}
    path = DEST/"rgb_triangulation/continuous_bound.json"
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print({k:v for k,v in report.items() if k!="witnesses"})


if __name__=="__main__":
    main()
