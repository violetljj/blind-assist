"""Consumed deployment replay and input-only near-bin fidelity diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import sim_bin1_diagnosis as B
import real_near_residual_dev_20261012 as R

P, ROOT, JOINT = B.P, B.ROOT, B.JOINT
OUT = ROOT/'artifacts.local/work/cnh-near-residual-dev-20261012/simulation'
METHODS = ('spatial_median', 'fixed_template')

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')

def template():
    return np.load(R.OUT/'templates.npz')['all']

def fidelity():
    """Incremental injected signal retention, not model detection or physics rendering."""
    t = template(); records = []
    rng = np.random.default_rng(20261012)
    noise = rng.normal(0, 1, (128,8,8,16)).astype(np.float32)
    # Paired subtraction removes the background response from the intervention.
    for background in ('zero', 'unit_noise', 'template_plus_noise'):
        z = np.zeros_like(noise) if background == 'zero' else noise.copy()
        if background == 'template_plus_noise': z[...,1] += t
        for count in (1,4,16,32,48,64):
            mask = np.zeros((8,8), bool); mask.flat[:count] = True
            for amplitude in (1.,3.,10.):
                delta = np.zeros_like(z); delta[...,1] = mask*amplitude
                for method in METHODS:
                    before = R.transform(z, method, t)
                    after = R.transform(z+delta, method, t)
                    change = after[...,1]-before[...,1]
                    retained = change[:,mask]/amplitude
                    records.append(dict(background=background,zones=count,amplitude=amplitude,method=method,
                        mean_injected_support_retention=float(retained.mean()),
                        min_frame_mean_retention=float(retained.mean(-1).min()),
                        max_frame_mean_retention=float(retained.mean(-1).max()),
                        target_support_positive_fraction=float((after[...,1][:,mask]>0).mean()),
                        target_support_mean_after=float(after[...,1][:,mask].mean()),
                        target_support_mean_before=float(before[...,1][:,mask].mean()),
                        off_support_delta_mean=float(change[:,~mask].mean()) if count<64 else None,
                        other_bins_unchanged=bool(np.array_equal((after-before)[...,[0,*range(2,16)]],delta[...,[0,*range(2,16)]]))))
    save(OUT/'input_fidelity.json', dict(scope='Input-only bin1 (nominal 0.3-0.6m) positive-signal injection; no physical rendering or detector inference.',
        frames=128,noise_sd=1,seed=20261012,template_source=str(R.OUT/'templates.npz'),records=records))
    print('FIDELITY',len(records),flush=True)

def score(method):
    dest = OUT/method; dest.mkdir(parents=True,exist_ok=True)
    target = dest/'scores.npz'
    if target.exists(): raise FileExistsError(f'Preserve existing scores: {target}')
    EV=P._ev(); P._no_local(); folder=P.EVAL/'hold_deploy'
    z=np.asarray(EV.normalized(folder)).astype(np.float32); n,k=z.shape[:2]
    with np.load(folder/'geometry.npz') as g:
        sensor,query,public=g['sensor'],g['public_query'],g['true_delta']+g['public_eps']
    import cnh_robust_stack_dev_20261012 as RS
    seed=40+list(P.HOLD_CONDITIONS).index('deploy'); pclean=P.HOLD_CONDITIONS['deploy'][3]
    # Preserve baseline float16 quantization before applying the intervention.
    z=(z+RS._nuisance_scene(np.random.default_rng(seed),n,k,pclean)).astype(np.float16)
    z=R.transform(z,method,template())
    rec=dict(scope='Consumed Development',method=method,nuisance_seed=seed,p_clean=pclean,
        models_and_cuts=str(JOINT/'real_manifest.json'),status='running',
        transform_sha256=hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(),
        template_sha256=hashlib.sha256((R.OUT/'templates.npz').read_bytes()).hexdigest())
    save(dest/'receipt.json',rec)
    m3=sorted((JOINT/'models/M3').glob('model_seed*.pt'))
    ordinary=[JOINT/'models'/f'ordinary_seed{s}.pt' for s in P.SEEDS]
    original=EV.paths; EV.paths=lambda arm:(m3,ordinary)
    result=None; began=time.monotonic()
    try:
        for value in np.unique(public):
            sel=np.flatnonzero(public==value); r=P.rot(value)
            part=EV.infer(z[sel],sensor@r,query@r,arms=('no_bin0',))['no_bin0']
            if result is None:
                result={key:np.empty((3,n)+v.shape[2:] if key=='ordinary_raw' else (n,)+v.shape[1:],v.dtype) for key,v in part.items()}
            for key,v in part.items():
                if key=='ordinary_raw': result[key][:,sel]=v
                else: result[key][sel]=v
            print('SCORED',method,float(value),len(sel),round(time.monotonic()-began,2),flush=True)
    finally:
        EV.paths=original
        import torch
        torch.cuda.empty_cache()
    np.savez_compressed(target,**result)
    rec.update(status='complete',score_seconds=time.monotonic()-began);save(dest/'receipt.json',rec)

def summarize(method):
    EV=P._ev(); dest=OUT/method
    cuts=json.loads((JOINT/'real_manifest.json').read_text())['cuts']
    rows=EV.C.read(P.EVAL/'hold_deploy/scene_rows.json')
    with np.load(P.EVAL/'hold_deploy/geometry.npz') as g: category=g['category']
    grades={}
    for name,path in [('original',JOINT/'scores/hold_deploy_new.npz'),('candidate',dest/'scores.npz')]:
        s=dict(np.load(path)); strong=P._strong(s,cuts)
        light=EV.T.predict_s(JOINT/'S',EV.s_features(s,cuts)).mean(0)
        grades[name]=dict(strong=strong,all=EV.TM.apply_gate(strong,light,cuts['s_tau']))
    report=dict(scope='Consumed synthetic Development; same geometry, weights, cuts and nuisance. No contact events closer than 0.6m in this cohort.',method=method)
    payload={}
    for mode in ('strong','all'):
        base,candidate=grades['original'][mode],grades['candidate'][mode]
        report[mode]=EV.describe(candidate,base,category,rows)[0]
        report[mode]['original']=EV.describe(base,base,category,rows)[0]
        for q in ('HEAD','BODY'):
            x=report[mode][q]
            assert x['baseline_timely']+x['rescue']-x['loss']==x['timely']
        payload[mode+'_original']=base;payload[mode+'_candidate']=candidate
    np.savez_compressed(dest/'paired_grades.npz',**payload)
    save(dest/'comparison.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fidelity','score','summary','run']);ap.add_argument('--method',choices=METHODS);a=ap.parse_args()
    if a.stage=='fidelity':fidelity()
    else:
        if a.method is None:ap.error('--method required for model stages')
        if a.stage in ('score','run'):score(a.method)
        if a.stage in ('summary','run'):summarize(a.method)
