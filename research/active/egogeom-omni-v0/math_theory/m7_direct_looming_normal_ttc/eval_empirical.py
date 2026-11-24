"""Frozen CODa Development replay. Predictions precede label access in every clip."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'm4_stochastic_cbf_ics/far_motion'
sys.path.insert(0, str(OLD))
import eval_far_motion as base
from direct import affine_ic, gradients, level_to_cam, static_warp

OUT = base.WORK / 'math_theory' / HERE.name
DT = .2
ARMS = ('dis', 's4', 's4_fused', 'full', 'full_fused', 'full_fused_k3','dis_k3')
HIGH = 'full'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode(blob, offsets, orig_hw, i):
    bgr = cv2.imdecode(blob[offsets[i]:offsets[i+1]], cv2.IMREAD_COLOR)
    oh, ow = orig_hw
    ar = ow / oh
    ih = 288 if ar >= 4/3 else max(288, int(round(384 / ar)))
    iw = 384 if ar < 4/3 else max(384, int(round(288 * ar)))
    img = cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_AREA if iw < bgr.shape[1] else cv2.INTER_LINEAR)
    x, y = (iw - 384)//2, (ih - 288)//2
    return cv2.cvtColor(img[y:y+288, x:x+384], cv2.COLOR_BGR2GRAY)


def evaluate(path, dis):
    obs = base.CACHE / path.name
    with np.load(obs) as a:
        gray, depth, normal, K, R, t, fwd = (a[k] for k in
            ('gray_s4','depth_s4','nrm_s4','K','poses_R','poses_t','causal_fwd'))
    # Access to packed RGB only; truth arrays are loaded below after all predictions.
    with np.load(path) as a:
        blob, offsets, orig_hw = (a[k] for k in ('rgb_blob','rgb_offsets','orig_hw'))
    n,h,w = gray.shape
    uv = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).reshape(-1,2)
    preds = {key: np.zeros((n,5,15)) for key in ARMS}
    support = {key: np.zeros((n,5,15), bool) for key in ('dis','s4','full')}
    visible = np.zeros((n,5,15), bool)
    counts, timings = Counter(), defaultdict(list)
    full0 = decode(blob, offsets, orig_hw, 0)
    for i in range(1,n):
        full = decode(blob, offsets, orig_hw, i)
        counts['max_full_gray_bridge_delta'] = max(counts['max_full_gray_bridge_delta'],
            int(np.abs(full[::4, ::4].astype(int)-gray[i]).max()))
        start = time.perf_counter()
        xy0, xy1 = base.correspondences(gray[i-1],gray[i],uv,dis)
        fw = fwd[i].copy()
        fw[2] = 0.
        fw = fw.astype(float) / max(1.e-8, np.linalg.norm(fw))
        right = np.array([fw[1],-fw[0],0.])
        basis = np.column_stack([right,fw])
        z0 = base.sample(depth[i-1], xy0)
        z1 = base.sample(depth[i], xy1)
        old = base.camera_points(xy0,z0,K[i-1]) @ R[i-1].T
        new = base.camera_points(xy1,z1,K[i]) @ R[i].T
        obs_ok = np.isfinite(z0)&np.isfinite(z1)&(z0>.1)&(z1>.1)&(new[:,2]>=-.55)&(new[:,2]<=1.1)
        old,new,xy0,xy1,z0 = (v[obs_ok] for v in (old,new,xy0,xy1,z0))
        rigid_cam = (old+t[i-1]-t[i]) @ R[i]
        rigid_px = base.project(rigid_cam,K[i])
        both = base.inside(rigid_px,rigid_cam[:,2],gray[i].shape)
        x,f = (new @ basis).T
        rw = rigid_cam @ R[i].T
        xr,fr = (rw @ basis).T
        roi = both&(np.abs(xr)<1.5)&(fr>=4)&(fr<10)&(rw[:,2]>=-.55)&(rw[:,2]<=1.1)
        imposed = base.camera_points(rigid_px, base.sample(depth[i],rigid_px), K[i]) @ R[i].T
        diff = np.linalg.norm((imposed-old+t[i]-t[i-1]) @ basis / DT,axis=1)
        counts['rigid_total'] += int(roi.sum())
        counts['depth_diff_above_075'] += int((roi&(diff>=.75)).sum())
        counts['far_roi_pairs'] += int(np.any((f>=7.86)&(f<10)&(np.abs(x)<1.5)))
        error = np.linalg.norm(rigid_px-xy1, axis=1)
        preds['dis'][i] = base.cell_grid(x[both],f[both],error[both])
        support['dis'][i] = base.cell_grid(x[both],f[both]).astype(bool)
        fc,xc = np.meshgrid(4.2+.2*np.arange(5),-1.4+.2*np.arange(15),indexing='ij')
        world = fc.ravel()[:,None]*fw+xc.ravel()[:,None]*right
        c1,c0 = world@R[i], (world+t[i]-t[i-1])@R[i-1]
        visible[i] = (base.inside(base.project(c1,K[i]),c1[:,2],(h,w)) &
                      base.inside(base.project(c0,K[i-1]),c0[:,2],(h,w))).reshape(5,15)
        shared_ms = (time.perf_counter()-start)*1000
        timings['shared_DIS_geometry_ms'].append(shared_ms)
        candidate = both & (np.abs(x)<1.5) & (f>=4.1)&(f<10)
        selected = np.flatnonzero(candidate | roi)
        for mode, scale, im0, im1 in [('s4',1,gray[i-1],gray[i]),('full',4,full0,full)]:
            start = time.perf_counter()
            radial = np.zeros(len(x))
            live = np.zeros(len(x), bool)
            control_ms = 0.
            if len(selected):
                idx = np.rint(xy0[selected]).astype(int)
                normals = level_to_cam(normal[i-1,:,idx[:,1],idx[:,0]],R[i-1])
                k0, k1 = (v.copy() for v in (K[i-1],K[i]))
                k0[:2] *= scale/4
                k1[:2] *= scale/4
                points = xy0[selected]*scale
                cc,jac,plane = static_warp(points,z0[selected],normals,k0,k1,R[i-1],R[i],t[i-1],t[i])
                plane &= (np.linalg.norm(cc-rigid_px[selected]*scale,axis=1)<.02*scale)
                g = gradients(im0)
                for locked in (False,True):
                    subset = np.flatnonzero(plane & (roi[selected] if locked else candidate[selected]))
                    if not len(subset):
                        continue
                    ix = selected[subset]
                    initial = cc[subset] if locked else xy1[ix]*scale
                    fit_start = time.perf_counter()
                    ell, sd, good = affine_ic(im0,im1,g,points[subset],initial,jac[subset],3.*scale,locked)
                    if locked:
                        # Exact center, live radiance. Both signs to compare to a speed magnitude.
                        speed = np.abs(np.expm1(-np.clip(ell,-2,2)))*rigid_cam[ix,2] / DT
                        counts[mode+'_rigid_supported'] += int(good.sum())
                        counts[mode+'_rigid_above_075'] += int((good&(speed>=.75)).sum())
                        counts[mode+'_matched_depth_above_075'] += int((good&(diff[ix]>=.75)).sum())
                        control_ms += (time.perf_counter()-fit_start)*1000
                    else:
                        live[ix] = good
                        radial[ix] = np.where(good,np.maximum(ell,0.)/np.maximum(sd,.001),0.)
                far = live&(f>=7.86)&(f<10)
                counts[mode+'_far_points'] += int(far.sum())
                counts[mode+'_far_pairs'] += int(far.any())
                counts[mode+'_far_positive_z196_pairs'] += int((far&(radial>=1.96)).any())
            preds[mode][i] = base.cell_grid(x[live],f[live],radial[live])
            support[mode][i] = base.cell_grid(x[live],f[live]).astype(bool)
            fused = np.hypot(error[both],radial[both])
            preds[mode+'_fused'][i] = base.cell_grid(x[both],f[both],fused)
            all_ms = shared_ms + 1000*(time.perf_counter()-start)
            timings[mode+'_joint_prediction_and_control_ms'].append(all_ms)
            timings[mode+'_prediction_ms'].append(all_ms-control_ms)
        # Three most recent pair scores at registered world-cell centers. A fixed
        # denominator prevents support dropout from increasing the score.
        avg = np.stack([preds[k][i].ravel() for k in ('full_fused', 'dis')])
        for age in (1,2):
            j = i-age
            if j < 1:
                continue
            back = world + t[i]-t[j]
            fj = fwd[j].copy()
            fj[2] = 0.
            fj /= max(np.linalg.norm(fj), 1.e-8)
            rj = np.array([fj[1],-fj[0],0.])
            gx = np.floor((back@rj + 1.5)/.2).astype(int)
            gz = np.floor((back@fj - 4.1)/.2).astype(int)
            ok = (gx>=0)&(gx<15)&(gz>=0)&(gz<5)
            for index,key in enumerate(('full_fused','dis')):
                avg[index,ok] += preds[key][j,gz[ok],gx[ok]]
        for index,key in enumerate(('full_fused_k3','dis_k3')):
            preds[key][i] = avg[index].reshape(5,15)/3.
        full0 = full
    with np.load(path) as a:
        valid = a['valid_heading'].astype(bool)
        s,d = (a[k][:,:,-5:,:].any(axis=1) for k in ('gt_bev_static','gt_bev_dynamic'))
    valid[0] = False
    include = valid[:,None,None] & visible & (s^d)
    counts['frames'] = n
    counts['heading_pairs'] = int(valid.sum())
    val = np.column_stack([d[include],*[preds[k][include] for k in ARMS],
                           *[support[k][include] for k in ('dis','s4','full')]])
    return counts,val,timings


def summary(data):
    y = data[:,0].astype(bool)
    arms = {}
    for j, k in enumerate(ARMS,1):
        arms[k] = float(roc_auc_score(y,data[:,j])) if len(np.unique(y))==2 else None
    idx = data[:,-2:].astype(bool).all(axis=1) & data[:,-3].astype(bool)
    common_y = y[idx]
    common = {k: float(roc_auc_score(common_y,data[idx,j])) if len(np.unique(common_y))==2 else None
              for j,k in enumerate(ARMS,1)}
    return dict(dynamic=int(y.sum()),static=int((~y).sum()),fixed_auroc=arms,
                common_dynamic=int(common_y.sum()), common_static=int((~common_y).sum()), common_auroc=common)


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    source_paths = sorted(base.SRC.glob('*.npz'))
    if len(source_paths)!=24:
        raise ValueError('Expected 24 CODa clips')
    total, times, groups, files = Counter(), defaultdict(list), defaultdict(list), {}
    for p in source_paths:
        c, table, elapsed = evaluate(p,dis)
        bridge = max(total['max_full_gray_bridge_delta'],c.pop('max_full_gray_bridge_delta'))
        total.update(c)
        total['max_full_gray_bridge_delta'] = bridge
        for key,dt in elapsed.items():
            times[key].extend(dt)
        groups[p.stem.split('_')[1]].append(table)
        files[p.stem] = {'stream_sha256':digest(base.CACHE/p.name),'pack_sha256':digest(p)}
        print(p.stem,c['rigid_total'],flush=True)
    per_seq = {k:summary(np.concatenate(a)) for k,a in groups.items()}
    all_data = np.concatenate([table for group in groups.values() for table in group])
    exclude = np.concatenate([table for k,group in groups.items() if k!='seq13' for table in group])
    result = dict(denominator='all visible exact M4 labelled cells; no output scores zero; common has all 3 supports',
                  clips=files,counts=total,per_sequence=per_seq,all=summary(all_data),excluding_13=summary(exclude),
                  timings_ms_p50_p95={k:np.percentile(v,[50,95]).tolist() for k,v in times.items()},
                  source_code_sha256={p.name:digest(p) for p in (Path(__file__),HERE/'direct.py',OLD/'eval_far_motion.py')},
                  alert_fa_km=None,clear_repeat_km=None,distance_mae=None)
    if (result['all']['dynamic'],result['all']['static'],total['rigid_total'],total['depth_diff_above_075'])!=(1030,5027,153952,107603):
        raise ValueError(f'M4 denominator mismatch: {result["all"]} {total}')
    (OUT/'v5_empirical_report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'all':result['all'],'counts':total,'timing':result['timings_ms_p50_p95']}))


if __name__ == '__main__':
    main()
