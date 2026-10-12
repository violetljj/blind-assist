"""One consumed deploy-hold paired input-mask diagnostic; no fitting or promotion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'research/active/dtr-r0/nearfield'))
import cnh_pose_floor_dr_dev_20261012 as P

JOINT=ROOT/'artifacts.local/work/cnh-pose-floor-noise-dr-dev-20261012'
OUT=ROOT/'artifacts.local/work/cnh-noise-ablation-dev-20261012/diagnosis/sim_dropbin1'

def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((JOINT/'real_manifest.json').read_text())
    assert manifest['drop_bins']==[0] and manifest['cuts']['local_tau']>1e100
    manifest.update(candidate='joint_dropbin1_diagnostic_only',drop_bins=[0,1],diagnostic_only=True,
                    default_candidate=False,intervention='Same frozen joint weights and cuts; additionally zero native bin1 before all extractors.')
    save(OUT/'real_manifest.json',manifest)
    seed=40+list(P.HOLD_CONDITIONS).index('deploy')
    receipt=dict(development_only=True,condition='deploy',nuisance_seed=seed,p_clean=P.HOLD_CONDITIONS['deploy'][3],
                 folder=str(P.EVAL/'hold_deploy'),baseline_scores=str(JOINT/'scores/hold_deploy_new.npz'),
                 model_folder=str(JOINT),local_disabled=True,intervention='z[...,1]=0; all models/cuts/poses/random draws fixed',
                 protocol='Paired timely rescue/loss per query, strong/all grades, far+clear; radial target proximity groups diagnostic only.')
    save(OUT/'receipt.json',receipt)
    return receipt

def score():
    rec=prepare();target=OUT/'scores.npz'
    if target.exists():return dict(np.load(target))
    EV=P._ev();P._no_local()
    folder=P.EVAL/'hold_deploy'
    z=np.asarray(EV.normalized(folder)).astype(np.float32);n,k=z.shape[:2]
    with np.load(folder/'geometry.npz') as g:
        sensor,query,public=g['sensor'],g['public_query'],g['true_delta']+g['public_eps']
    import cnh_robust_stack_dev_20261012 as RS
    z=(z+RS._nuisance_scene(np.random.default_rng(rec['nuisance_seed']),n,k,rec['p_clean'])).astype(np.float16)
    z[...,1]=0
    m3=sorted((JOINT/'models/M3').glob('model_seed*.pt'))
    ordinary=[JOINT/'models'/f'ordinary_seed{s}.pt' for s in P.SEEDS]
    original=EV.paths;EV.paths=lambda arm:(m3,ordinary)
    result=None;began=time.monotonic()
    try:
        for value in np.unique(public):
            sel=np.flatnonzero(public==value);r=P.rot(value)
            part=EV.infer(z[sel],sensor@r,query@r,arms=('no_bin0',))['no_bin0']
            if result is None:
                result={key:np.empty((3,n)+v.shape[2:] if key=='ordinary_raw' else (n,)+v.shape[1:],v.dtype) for key,v in part.items()}
            for key,v in part.items():
                if key=='ordinary_raw':result[key][:,sel]=v
                else:result[key][sel]=v
            print('SCORED_PUBLIC',value,len(sel),'elapsed',round(time.monotonic()-began,2),flush=True)
    finally:
        EV.paths=original
        import torch
        torch.cuda.empty_cache()
    np.savez_compressed(target,**result)
    rec['score_seconds']=time.monotonic()-began;save(OUT/'receipt.json',rec)
    return result

def summarize():
    EV=P._ev();E,T,TM,FM,CM=EV.E,EV.T,EV.TM,EV.FM,EV.CM
    cuts=json.loads((JOINT/'real_manifest.json').read_text())['cuts']
    rows=EV.C.read(P.EVAL/'hold_deploy/scene_rows.json')
    with np.load(P.EVAL/'hold_deploy/geometry.npz') as g:
        category=g['category'];sensor=g['sensor']
    grades={}
    for name,path in [('original',JOINT/'scores/hold_deploy_new.npz'),('dropbin1',OUT/'scores.npz')]:
        s=dict(np.load(path));strong=P._strong(s,cuts)
        light=T.predict_s(JOINT/'S',EV.s_features(s,cuts)).mean(0)
        grades[name]=dict(strong=strong,all=TM.apply_gate(strong,light,cuts['s_tau']))
    distances=[]
    for row in rows:
        box=row.get('target_box')
        if not box or not row.get('presence',True):distances.append(np.inf);continue
        lo,hi=np.asarray(box['lo']),np.asarray(box['hi'])
        centers=sensor[3:14,:3,3]
        delta=np.maximum(np.maximum(lo-centers,centers-hi),0)
        distances.append(float(np.linalg.norm(delta,axis=1).min()))
    distances=np.asarray(distances)
    groups={'all':np.ones(len(rows),bool),'target_nearest_lt0.6m':distances<.6,
            'target_nearest_0.6_to1.2m':(distances>=.6)&(distances<1.2),'target_nearest_ge1.2m':np.isfinite(distances)&(distances>=1.2)}
    report=dict(scope='Consumed synthetic Development; distances are evaluator-only nearest authored AABB distance over timely sensor poses.',groups={})
    payload=dict(nearest_target_m=distances,category=category)
    for mode in ('strong','all'):
        base,candidate=grades['original'][mode],grades['dropbin1'][mode]
        description,_=EV.describe(candidate,base,category,rows)
        description['original_far_clear']=EV.describe(base,base,category,rows)[0]['far_clear']
        report[mode]=description
        a=CM.first_notice(FM.replay_gap1(base));b=CM.first_notice(FM.replay_gap1(candidate))
        ah=(a>=0)&(a<11);bh=(b>=0)&(b<11)
        payload.update({mode+'_original_hit':ah,mode+'_dropbin1_hit':bh})
        report['groups'][mode]={}
        for label,mask in groups.items():
            table={}
            for q,name in enumerate(('HEAD','BODY')):
                selected=mask&(category[:,q]=='contact');x,y=ah[selected,:,q],bh[selected,:,q]
                table[name]=dict(denominator=int(x.size),original=int(x.sum()),dropbin1=int(y.sum()),rescue=int((~x&y).sum()),loss=int((x&~y).sum()))
            report['groups'][mode][label]=table
    np.savez_compressed(OUT/'paired_events.npz',**payload)
    save(OUT/'comparison.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','score','summarize','run']);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='score':score()
    elif a.stage=='summarize':summarize()
    else:score();summarize()
