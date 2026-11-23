"""CODa 4–10 m full-image motion observability, no frozen corridor seeds."""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[6]
WORK = REPO/"artifacts.local/work/egogeom-omni-v0"
SRC = WORK/"zeroshot_eval/coda_zeroshot_shard000"
CACHE = WORK/"math_theory/v4_stream/coda_campus"
OUT = WORK/"math_theory/m4_stochastic_cbf_ics/far_motion"
BOUNDS = [(4.,5.1), (5.1,7.86), (7.86,10.)]
LAYERS = ["geometry", "matches", "world_speed", "reprojection", "clustered_motion"]
LK = {"winSize": (9,9), "maxLevel": 2, "criteria": (3,15,.03)}


def camera_points(uv, z, K):
    return np.column_stack([(uv*4-K[:2,2])/np.diag(K)[:2] * z[:,None],z])


def project(cam, K):
    return (cam[:,:2]/np.maximum(cam[:,2:3],.001)*np.diag(K)[:2]+K[:2,2])/4


def inside(uv, z, hw):
    return np.isfinite(uv).all(axis=1) & (z>.1) & (uv>=0).all(axis=1) & (uv<=np.array(hw[::-1])-1).all(axis=1)


def sample(z, xy):
    ij = np.rint(xy).astype(int)
    return z[ij[:,1].clip(0,z.shape[0]-1),ij[:,0].clip(0,z.shape[1]-1)]


def cell_grid(x, f, weights=None):
    g = np.zeros((5,15),dtype=float)
    use = np.isfinite(x)&np.isfinite(f)&(x>=-1.5)&(x<1.5)&(f>=4.1)&(f<5.1)
    ij = (np.floor((f[use]-4.1)/.2).astype(int).clip(0,4),
          np.floor((x[use]+1.5)/.2).astype(int).clip(0,14))
    np.maximum.at(g,ij,1. if weights is None else weights[use])
    return g


def correspondences(gray0,gray,uv_grid,dis):
    if dis is None:
        xy0 = cv2.goodFeaturesToTrack(gray0,220,.003,2,blockSize=3)
        if xy0 is None:
            return np.empty((0,2)),np.empty((0,2))
        xy1, ok1, _ = cv2.calcOpticalFlowPyrLK(gray0,gray,xy0,None,**LK)
        xyback, ok0, _ = cv2.calcOpticalFlowPyrLK(gray,gray0,xy1,None,**LK)
        xy0, xy1 = xy0[:,0],xy1[:,0]
        valid = (ok0[:,0]>0)&(ok1[:,0]>0)&(np.linalg.norm(xyback[:,0]-xy0,axis=1)<=.6)
    else:
        u,v = uv_grid.T
        texture = cv2.cornerMinEigenVal(gray0,3)
        use = (u%2==0)&(v%2==0)&(texture[v,u]>=1e-4)
        xy0 = uv_grid[use].astype(np.float32)
        pos = xy0.astype(int)
        forward = dis.calc(gray0,gray,None)
        reverse = dis.calc(gray,gray0,None)
        shift = forward[pos[:,1],pos[:,0]]
        xy1 = xy0+shift
        if not len(xy1):
            return xy0,xy1
        backward = cv2.remap(reverse,xy1[:,None,:],None,cv2.INTER_LINEAR)[:,0]
        valid = np.linalg.norm(shift+backward,axis=1)<=.6
    valid &= inside(xy1,np.ones(len(xy1)),gray.shape)
    return xy0[valid],xy1[valid]


def extract(prev, cur, fwd, uv_grid,dis):
    gray0, z0, K0, R0, t0 = prev
    gray, z1, K1, R1, t1 = cur
    fw = fwd[:2] / max(1.e-8,np.linalg.norm(fwd[:2]))
    fw = np.array([fw[0],fw[1],0.])
    right = np.array([fw[1],-fw[0],0.])
    basis = np.vstack([right,fw]).T
    cloud = camera_points(uv_grid,z1.ravel(),K1) @ R1.T
    xf = cloud@basis
    # Fixed obstacle-height slab relative to camera. No label or corridor D required.
    height = (cloud[:,2]>=-.55)&(cloud[:,2]<=1.1)
    xf = xf[height]
    grids = {key:np.zeros((5,15)) for key in LAYERS}
    grids["geometry"] = cell_grid(xf[:,0],xf[:,1])
    counts = Counter()
    def count_ranges(key,x,f):
        for j,(lo,hi) in enumerate(BOUNDS):
            n = int(((abs(x)<1.5)&(f>=lo)&(f<hi)).sum())
            counts[f"{key}_{j}_points"] += n
            counts[f"{key}_{j}_frames"] += bool(n)
    count_ranges("geometry",xf[:,0],xf[:,1])
    # Center of each labelled BEV cell at camera height, transformed via odometry.
    f_cent,x_cent = np.meshgrid(4.2+.2*np.arange(5),-1.4+.2*np.arange(15),indexing="ij")
    level = x_cent.ravel()[:,None]*right+f_cent.ravel()[:,None]*fw
    curr_cam = level @ R1
    prev_cam = (level+t1-t0) @ R0
    fov = (inside(project(curr_cam,K1),curr_cam[:,2],gray.shape)&
           inside(project(prev_cam,K0),prev_cam[:,2],gray0.shape)).reshape(5,15)
    rear = x_cent.ravel()[:,None]*right-f_cent.ravel()[:,None]*fw
    rear0,rear1 = (rear+t1-t0)@R0,rear@R1
    rear_visible = (inside(project(rear0,K0),rear0[:,2],gray0.shape)&
                    inside(project(rear1,K1),rear1[:,2],gray.shape))
    counts["level_geometry_test_cells"] += int(fov.size)
    counts["forward_centers_in_both_fov"] += int(fov.sum())
    counts["rear_mirrored_centers_in_both_fov"] += int(rear_visible.sum())
    xy0,xy1 = correspondences(gray0,gray,uv_grid,dis)
    if not len(xy0):
        return grids,fov,counts
    d0,d1 = sample(z0,xy0),sample(z1,xy1)
    old = camera_points(xy0,d0,K0) @ R0.T
    new = camera_points(xy1,d1,K1) @ R1.T
    obs = (np.isfinite(d0)&np.isfinite(d1)&(d0>.1)&(d1>.1)&(new[:,2]>=-.55)&(new[:,2]<=1.1))
    new,old,xy1 = new[obs],old[obs],xy1[obs]
    x,f = (new@basis).T
    grids["matches"] = cell_grid(x,f)
    count_ranges("matches",x,f)
    rigid_current = (old+t0-t1)@R1
    xy_rigid = project(rigid_current,K1)
    both = inside(xy_rigid,rigid_current[:,2],gray.shape)
    err_px = np.linalg.norm(xy_rigid-xy1,axis=1)
    vw = (new-old+t1-t0)@basis/.2
    speed = np.linalg.norm(vw,axis=1)
    fast = both&(speed>=.75)
    distinct = both&(err_px>=1.)  # one S4 pixel = 4 input pixels
    grids["world_speed"] = cell_grid(x[fast],f[fast])
    grids["reprojection"] = cell_grid(x[distinct],f[distinct])
    count_ranges("world_speed",x[fast],f[fast])
    count_ranges("reprojection",x[distinct],f[distinct])
    # Score exact cells with maximum compensated residual (unobserved cells stay zero).
    grids["residual_score"] = cell_grid(x[both],f[both],err_px[both])
    active = fast & distinct & (abs(x)<2.5)&(f>=3.5)&(f<10.5)
    coords = np.column_stack([x[active],f[active]])
    if len(coords):
        delta = np.linalg.norm(coords[:,None]-coords,axis=-1)
        delta_v = np.linalg.norm(vw[active,None]-vw[active],axis=-1)
        group = ((delta<=.8)&(delta_v<=1.0)).sum(axis=1)>=2
        picked = coords[group]
        grids["clustered_motion"] = cell_grid(picked[:,0],picked[:,1])
        count_ranges("clustered_motion",picked[:,0],picked[:,1])
    # Known rigid counterfactual, same camera motion. Model truth for this control
    # follows only from the reprojected old geometry, without current image/depth.
    rigid_w = rigid_current @ R1.T
    xr,fr = (rigid_w@basis).T
    roi = both & (abs(xr)<1.5)&(fr>=4)&(fr<10)&(rigid_w[:,2]>=-.55)&(rigid_w[:,2]<=1.1)
    exact_v = (rigid_w-old+t1-t0)@basis/.2
    uncomp_v = (rigid_w-old)@basis/.2
    imposed = camera_points(xy_rigid,sample(z1,xy_rigid),K1)@R1.T
    depth_only_v = (imposed-old+t1-t0)@basis/.2
    counts["rigid_points"] += int(roi.sum())
    for key,vec in [("rigid_compensated",exact_v),("rigid_without_translation",uncomp_v),
                    ("imposed_rigid_correspondence_real_depth",depth_only_v)]:
        counts[key+"_above_075"] += int((roi&(np.linalg.norm(vec,axis=1)>=.75)).sum())
    return grids,fov,counts


def evaluate_clip(p,dis):
    with np.load(CACHE/p.name) as stream:
        arrays = [stream[k] for k in ("gray_s4","depth_s4","K","poses_R","poses_t")]
        fwd = stream["causal_fwd"]
    h,w = arrays[1].shape[-2:]
    u,v = np.meshgrid(np.arange(w),np.arange(h))
    uv = np.column_stack([u.ravel(),v.ravel()])
    clip_frames = len(fwd)
    predictions = {k:np.zeros((clip_frames,5,15)) for k in LAYERS+["residual_score"]}
    visible = np.zeros((clip_frames,5,15),dtype=bool)
    total = Counter()
    for i in range(1,clip_frames):
        old = [value[i-1] for value in arrays]
        new = [value[i] for value in arrays]
        grids,visible[i],counts = extract(old,new,fwd[i],uv,dis)
        total.update(counts)
        for k,g in grids.items():
            predictions[k][i] = g
    # Truth enters only after this clip's complete prediction stream.
    with np.load(p) as truth:
        heading = truth["valid_heading"].astype(bool)
        static = truth["gt_bev_static"][:,:,-5:,:].any(axis=1)
        dyn = truth["gt_bev_dynamic"][:,:,-5:,:].any(axis=1)
    heading[0] = False
    valid = heading[:,None,None]
    labels = {"dyn_only":dyn&~static&valid,"static_only":static&~dyn&valid}
    total["all_frames"] += clip_frames
    total["paired_frames"] += clip_frames-1
    total["valid_heading_pairs"] += int(heading.sum())
    total["unlabelled_or_empty_cell_frames"] += int((valid&~dyn&~static).sum())
    total["both_static_dynamic_cell_frames"] += int((valid&dyn&static).sum())
    scores,classes = [],[]
    for name,mask in labels.items():
        total[name] += int(mask.sum())
        m = mask & visible
        total[name+"_in_fov"] += int(m.sum())
        for k in LAYERS:
            exact = predictions[k] > 0
            # The tolerant arm permits one neighbouring BEV bin (0.2m per axis).
            near = maximum_filter(exact,size=(1,3,3),mode="constant")
            total[name+"_exact_"+k] += int((m&exact).sum())
            total[name+"_near_"+k] += int((m&near).sum())
        scores.extend(predictions["residual_score"][m])
        classes.extend([int(name=="dyn_only")]*int(m.sum()))
    return total,np.array(classes),np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow",choices=["klt","dense"],default="klt")
    choice = parser.parse_args().flow
    cv2.setNumThreads(1)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST) if choice=="dense" else None
    total = Counter()
    records,yy,ss = {},[],[]
    paths = sorted(SRC.glob("*.npz"))
    if len(paths)!=24:
        raise ValueError("Expected 24 CODa development clips")
    for p in paths:
        counts,y,s = evaluate_clip(p,dis)
        total.update(counts)
        records[p.stem] = dict(counts)
        yy.extend(y)
        ss.extend(s)
    OUT.mkdir(parents=True,exist_ok=True)
    by_seq = {}
    for name,counts in records.items():
        by_seq.setdefault(name.split("_")[1],Counter()).update(counts)
    per_seq = {}
    for seq,c in by_seq.items():
        per_seq[seq] = {k:c[k] for k in ("dyn_only","static_only","dyn_only_exact_clustered_motion",
                                         "static_only_exact_clustered_motion","dyn_only_exact_matches","static_only_exact_matches")}
    res = {"clips":len(records),"flow":choice,"sequences":per_seq,"counts":dict(total),"clip_counts":records,
           "depth_capacity_intervals_m":BOUNDS,"cell_score_auroc":float(roc_auc_score(yy,ss)) if len(set(yy))==2 else None}
    name = "far_motion_report.json" if choice=="klt" else "dense_far_motion_report.json"
    (OUT/name).write_text(json.dumps(res,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in res.items() if k!="clip_counts"}))


if __name__=="__main__":
    main()
