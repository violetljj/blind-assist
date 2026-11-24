"""Evaluator-only finite witness search. Never imported by the causal filter."""
import itertools
from pathlib import Path

from eval_empirical import ART, OLD, OUTPUT, sha, store
from eval_on_v4 import corridor_dist_from_depth
import numpy as np
from scipy.stats import qmc


def candidates():
    scales = np.geomspace(.4,2.5,33)
    scales[16] = 1.
    for s in scales:
        yield "scalar", np.array([1/s,0.,0.,0.])
    for s,b in itertools.product(scales[::4],[-.15,-.075,.075,.15]):
        yield "offset", np.array([1/s,b,0.,0.])
    for s, bx, by in itertools.product(scales[::8],[-.12,0,.12],[-.12,0,.12]):
        if bx or by:
            yield "tilt", np.array([1/s,0.,bx,by])
    for u in qmc.Sobol(4,scramble=True,seed=20261124).random_base2(7):
        yield "joint", np.array([np.exp(np.log(.4)+u[0]*np.log(6.25)),
                                  .30*u[1]-.15,.24*u[2]-.12,.24*u[3]-.12])


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    old_witnesses = OLD/"continuous_bound.json"
    scalar_report = OLD/"oracle_aria.json"
    cert = {(r["clip"],r["frame"]) for r in __import__("json").loads(old_witnesses.read_text())["witnesses"]}
    old_scalar = __import__("json").loads(scalar_report.read_text())["fixed_support"]["all_eligible"]
    grid = list(candidates())
    arms = ("scalar","offset","tilt","joint")
    rows, identities = [], []
    for src in sorted((ART/"zeroshot_eval/aria_g2_20clips").rglob("*.npz")):
        obs = ART/"math_theory/v4_stream/aria_gen2"/src.name
        identities.append({"source":src.stem,"pack_sha256":sha(src),"stream_sha256":sha(obs)})
        with np.load(src) as p:
            er = p["eval_rows"]
        with np.load(obs) as p:
            depth, K, rot, fw, ls = (p[k] for k in ("depth_s4","K","poses_R","strict_fwd","ls_s4"))
        h,w = depth.shape[1:]
        full,unc = [np.empty((4*h,4*w),np.float32) for _ in range(2)]
        v,u = np.indices((h,w),dtype=np.float32)
        for i in np.flatnonzero((er[:,0]>=.3)&(er[:,0]<=4)&(er[:,1]>=.6)):
            target = float(er[i,0])
            best = {arm:{"error_m":None,"coefficient":None,"readout_m":None} for arm in arms}
            x,y = (4*u+.5-K[i,0,2])/K[i,0,0], (4*v+.5-K[i,1,2])/K[i,1,1]
            inv = 1/depth[i]
            unc[::4,::4] = ls[i]
            for arm,coef in grid:
                a,b,bx,by = coef
                if arm=="scalar":
                    # Match M2's float64 scalar multiplication, then cast to full float32.
                    full[::4,::4] = depth[i]*np.float64(1/a)
                else:
                    disp = a*inv + b + bx*x + by*y
                    if not np.all(disp>0):
                        continue
                    full[::4,::4] = 1/disp
                dist = corridor_dist_from_depth(full,None,unc,K[i],rot[i],fw[i],False)[0]
                estimate = min(6.,dist)
                error = abs(estimate-target)
                old = best[arm]["error_m"]
                if old is None or error<old:
                    best[arm] = {"error_m":error,"coefficient":coef.tolist(),"readout_m":float(dist) if np.isfinite(dist) else None}
            rows.append({"clip":src.stem,"frame":int(i),"target_m":target,"best":best,
                         "continuous_scalar_impossible":(src.stem,int(i)) in cert})
    def errors(row, subset):
        return min(row["best"][arm]["error_m"] for arm in subset if row["best"][arm]["error_m"] is not None)
    summaries = {}
    for name,subset in (("scalar",["scalar"]),("scalar_offset",["scalar","offset"]),("scalar_tilt",["scalar","tilt"]),
                        ("all",list(arms))):
        e = np.array([errors(row,subset) for row in rows])
        summaries[name] = {"mae_filled6_m":float(e.mean()),"over_055":int(sum(e>.55)),
                            "rescued_scalar_grid":sum(errors(r,["scalar"])>.55 and errors(r,subset)<=.55 for r in rows),
                            "rescued_scalar_cert":sum(r["continuous_scalar_impossible"] and errors(r,subset)<=.55 for r in rows)}
    assert len(rows)==840 and sum(r["continuous_scalar_impossible"] for r in rows)==13
    assert summaries["scalar"]["over_055"]==old_scalar["no_grid_scale_within_0.55m"]
    assert abs(summaries["scalar"]["mae_filled6_m"]-old_scalar["oracle_frame_mae_m"])<1e-6
    store(OUTPUT/"oracle_aria.json",{"candidates":{k:sum(t==k for t,_ in grid) for k in arms},
                                     "input_sha256":identities,"old_witness_sha256":sha(old_witnesses),
                                     "old_scalar_sha256":sha(scalar_report),"code_sha256":sha(Path(__file__)),
                                     "summary":summaries,"rows":rows})
    print(summaries)


if __name__=="__main__":
    main()
