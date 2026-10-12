"""Paired anchor readout from full-float cached depth, consumed data only."""
from __future__ import annotations
import copy
import json
import time
from collections import Counter
import numpy as np
from atom_camera import AtomCamera
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_small_model_dev_20261012 as S
import real_rgb_query_feasibility_dev_20261012 as F
from fusion_refinement_cache_dev_20261013 import OUT
from fusion_anchor_candidates_dev_20261013 import fit_quality, numerical_checks

def project(disp, valid, fit, corridor_rays):
    if fit is None:
        return dict(available=False, rgb=[False, False], px=[0, 0], occupancy=np.zeros((2, 6, 8), int).tolist(), centroid=[None, None])
    a, b = fit
    inv = (disp.ravel() - b) / a
    good = valid.ravel() & np.isfinite(inv) & (inv > 1e-3)
    points = corridor_rays[good] / inv[good, None]
    occ, counts, centers = [], [], []
    for lo, hi in F.BOXES.values():
        inside = (np.abs(points[:, 0]) <= .30) & (points[:, 1] >= lo) & (points[:, 1] <= hi) & (points[:, 2] >= .30) & (points[:, 2] <= 2.)
        p = points[inside]
        hist = np.histogram2d(p[:, 0], p[:, 2], bins=(np.linspace(-.3,.3,7), np.linspace(.3,2,9)))[0].astype(int)
        occ.append(hist.tolist()); counts.append(int(len(p)))
        centers.append(np.median(p, axis=0).tolist() if len(p) else None)
    return dict(available=True, rgb=[n >= 150 for n in counts], px=counts, occupancy=occ, centroid=centers)

def main():
    start = time.perf_counter()
    manifest = json.loads((OUT / 'cache_manifest.json').read_text())
    cam = AtomCamera(); K = cam.K_upright
    Rct = np.array(json.loads((X.OUT / 'camera_tof_extrinsic.json').read_text())['rotation'])
    w,h = cam.upright_size
    vv,uu = np.mgrid[:h,:w]
    dirs = np.stack(((uu-K[0,2])/K[0,0],(vv-K[1,2])/K[1,1],np.ones_like(uu)), -1).reshape(-1,3)
    rays = X.zone_rays(); rng = np.random.default_rng(20261013)
    results = dict(height=manifest['height'], checks=numerical_checks(), sessions={})
    # Sort by full session name, identical to frozen smoke's traversal. Office
    # segments precede hall; reset RNG at first hall so its baseline reproduces
    # the existing nine-session smoke, not a changed RANSAC stream.
    for key,sess in manifest['sessions'].items():
        if key == '072853': rng = np.random.default_rng(20261013)
        last = None
        arms = {k:[] for k in ('baseline','fresh_only','quality_floor','quality_tof')}
        for r in sess['frames']:
            if time.perf_counter()-start > 2200: raise TimeoutError('CPU budget reserve')
            row = {k:v for k,v in r.items() if k not in ('cache','dist')}
            fits = {k:None for k in arms}; sources = {k:None for k in arms}
            diag = {}
            if 'cache' in r:
                with np.load(OUT/r['cache']) as payload:
                    disp, valid = payload['disp'],payload['valid']
                dz,zfac = S.zone_disparity(disp,K,Rct,rays)
                dist = np.asarray(r['dist']); keep = np.isfinite(dist)&(dist>.3)&(dist<4)
                inv = np.full(64,np.nan); inv[keep] = 1/(dist[keep]*zfac[keep])
                tof = S.fit_affine(dz,inv); diag = fit_quality(dz,inv,tof)
                # Quality-only fallback draws from a copied stream; cannot alter baseline.
                extra_rng = copy.deepcopy(rng)
                rot = F.rot(r['pitch'],r['roll']); corridor_rays = dirs @ (rot @ Rct.T).T
                floor = S.floor_anchor(disp,dirs,rot,Rct,manifest['height'],rng) if tof is None else None
                fresh = tof or floor
                fits['baseline'] = fresh or last; fits['fresh_only'] = fresh
                src = 'tof' if tof else 'floor' if floor else None
                sources['baseline'] = src or ('reused' if last else None); sources['fresh_only'] = src
                if fresh: last = fresh
                if diag['accepted']:
                    fits['quality_floor'] = fits['quality_tof'] = tof
                    sources['quality_floor'] = sources['quality_tof'] = 'tof'
                else:
                    if tof is not None: floor = S.floor_anchor(disp,dirs,rot,Rct,manifest['height'],extra_rng)
                    fits['quality_floor'] = floor; sources['quality_floor'] = 'floor' if floor else None
                memo = {}
                for name in arms:
                    fit = fits[name]
                    identity = tuple(fit) if fit is not None else None
                    if identity not in memo: memo[identity] = project(disp,valid,fit,corridor_rays)
                    arms[name].append(dict(row, **memo[identity], fit=fit, anchor=sources[name], quality=diag))
            else:
                for name in arms:
                    arms[name].append(dict(row, available=False, rgb=[False,False], px=[0,0], occupancy=np.zeros((2,6,8),int).tolist(), centroid=[None,None], fit=None, anchor=None,quality={}))
        results['sessions'][key] = dict(label=sess['label'],session=sess['session'],arms=arms)
        print(key, {a:dict(coverage=sum(r['available'] for r in fr),rgb=np.sum([r['rgb'] for r in fr],axis=0).tolist(),sources=dict(Counter(r['anchor'] for r in fr))) for a,fr in arms.items()},flush=True)
        (OUT / 'readout.partial.json').write_text(json.dumps(results),encoding='utf8')
    results['wall_s'] = time.perf_counter()-start
    (OUT / 'readout.json').write_text(json.dumps(results),encoding='utf8')
    # Exact same hall baseline against existing frozen smoke. No rewrite.
    smoke=json.loads((F.ROOT/'artifacts.local/work/frozen-fusion-eval-v1-smoke-20261012/frozen_fusion_eval_v1.json').read_text())
    mismatches=[]
    for key,s in results['sessions'].items():
        if s['session'] not in smoke['sessions']: continue
        old={r['tof_index']:r for r in smoke['sessions'][s['session']]['frames']}
        for r in s['arms']['baseline']:
            o=old[r['i']]
            for field in ('rgb','tof_strong','pose_ok'):
                if r[field]!=o[field]: mismatches.append([key,r['i'],field,r[field],o[field]])
    (OUT/'baseline_parity.json').write_text(json.dumps(dict(mismatches=mismatches),indent=1),encoding='utf8')
    print('parity mismatches',len(mismatches),'wall',results['wall_s'],flush=True)

if __name__=='__main__': main()
