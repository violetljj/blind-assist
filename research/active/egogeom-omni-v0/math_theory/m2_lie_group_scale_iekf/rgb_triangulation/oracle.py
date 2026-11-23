"""Label-only Aria diagnostic. Its fitted choices are never inputs to the RGB estimator."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_on_v4 import ART, DEST, STREAM, LieScaleIEKF, corridor_dist_from_depth


def grid_diagnostics(depth, target, fixed, clip, scales, old):
    deficit = ~np.isfinite(depth) | (depth > 6)
    errors = np.abs(np.where(deficit, 6.0, depth) - target[:, None])
    old_errors = np.abs(np.minimum(np.where(np.isfinite(old), old, 6.0), 6.0) - target)
    use = np.flatnonzero(fixed)
    grouped_errors = []
    clip_choices = {}
    for c in np.unique(clip[use]):
        rows = use[clip[use] == c]
        best = int(errors[rows].mean(0).argmin())
        grouped_errors.extend(errors[rows, best])
        clip_choices[c] = float(scales[best])
    best_global = int(errors[use].mean(0).argmin())
    unit = int(np.argmin(abs(scales - 1)))
    return {"frames":len(use), "unit_mae_m":float(errors[use,unit].mean()),
            "prior_M2_mae_m":float(old_errors[use].mean()),
            "oracle_global_mae_m":float(errors[use,best_global].mean()),"global_scale":float(scales[best_global]),
            "oracle_clip_mae_m":float(np.mean(grouped_errors)),"clip_scales":clip_choices,
            "oracle_frame_mae_m":float(errors[use].min(1).mean()),
            "no_grid_scale_within_0.55m":int(np.sum(errors[use].min(1)>.55)),
            "no_grid_scale_returns_finite":int(np.sum(deficit[use].all(1)))}


def main(dest=None):
    folder = ART/"zeroshot_eval/aria_g2_20clips"
    dest = DEST/"rgb_triangulation" if dest is None else dest
    dest.mkdir(exist_ok=True,parents=True)
    scales = np.geomspace(0.4,2.5,33)
    scales[16] = 1.0
    outputs, actual, old_outputs, clip_names = [], [], [], []
    for src in sorted(folder.rglob("*.npz")):
        with np.load(src) as d:
            er = d["eval_rows"]
        truth = er[:,0]
        eligible = (truth>=.30) & (truth<=4) & (er[:,1]>=.6)
        with np.load(STREAM/"aria_gen2"/src.name) as c:
            dep,dh,nrm,ls,K,R,t,fw = (c[k] for k in ("depth_s4","dh_s4","nrm_s4","ls_s4","K","poses_R","poses_t","strict_fwd"))
            unit_d = c["d_1f_strict"]
        old = LieScaleIEKF()
        full = np.zeros((dep.shape[1]*4, dep.shape[2]*4),np.float32)
        full_ls = np.empty_like(full)
        for i in range(len(dep)):
            s_old = old.step(dep[i],dh[i],nrm[i],ls[i],K[i],R[i],t[i])
            if not eligible[i]:
                continue
            full_ls[::4,::4] = ls[i]
            values = []
            for scale in (*scales, s_old):
                full[::4,::4] = dep[i]*scale
                distance, _ = corridor_dist_from_depth(full,None,full_ls,K[i],R[i],fw[i],False)
                values.append(distance)
            distances = np.array(values[:-1])
            assert distances[16] == unit_d[i]
            outputs.append(distances)
            old_outputs.append(values[-1])
            actual.append(truth[i])
            clip_names.append(src.stem)
    outputs, actual, old, clip = map(np.asarray,(outputs,actual,old_outputs,clip_names))
    masks = {"all_eligible":np.ones(len(actual),bool),"baseline_finite":np.isfinite(outputs[:,16]),
             "all_grid_values_finite":np.isfinite(outputs).all(1)}
    res = {k:grid_diagnostics(outputs,actual,v,clip,scales,old) for k,v in masks.items() if v.any()}
    info = {"scale_grid":scales.tolist(),"cohort":"Aria 19 previously scored clips",
            "missing_prediction_m":6.0,"fixed_support":res}
    (dest/"oracle_aria.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    for key,q in res.items():
        print(key,{k:v for k,v in q.items() if k!="clip_scales"})


if __name__ == "__main__":
    main()
