"""Pose factorial on consumed displacement data; exact counts, no training."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import cnh_displacement_ceiling as D
import cnh_displacement_ceiling_render as R
import cnh_displacement_ceiling_evaluate as E

OUT=D.OUT.parent/'cnh-pose-factorial-20261004'
PRIOR=D.OUT.parent/'cnh-unknown-target-reference-20261004'
RUN='CNH_POSE_FACTORIAL_20261004'
TIDS=[0,1,4,5,6]
FRAMES=D.FRAMES
PAST=np.concatenate([np.arange(max(0,f-7),f+1) for f in FRAMES])
START=np.cumsum([0]+[min(8,int(f)+1) for f in FRAMES])


def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    for key,sub in [('TEMP','tmp'),('TMP','tmp'),('CUPY_CACHE_DIR','cupy-cache')]:
        path=OUT/sub;path.mkdir(exist_ok=True);os.environ[key]=str(path)


def read(p):return E.read(p)
def save(p,v):return D.SS.save(p,v)


def plan():
    if (OUT/'PLAN.json').exists():raise FileExistsError('Existing PLAN is immutable')
    import cnh_pose_factorial_m3 as M
    import cnh_margin_confirm as MC
    files=[D.OUT/'result.json',PRIOR/'scores/all_scores.npz',PRIOR/'scenes.json']
    for u in D.UNITS:
        files.extend(D.OUT/f'{sub}/unit{u}.{ext}' for sub,ext in [('observations','npz'),('truth','json'),('templates','npz'),('scores','npz')])
    files.extend(MC.model_paths('M3'));files.append(D.BIAS)
    save(OUT/'PLAN.json',dict(run=RUN,created_utc=datetime.now(timezone.utc).isoformat(),units=D.UNITS,
        role='EXPLORE consumed synthetic Development pure measurement; no training/new photons/changed noise law; old results read-only',
        input_sha256={str(p):E.sha(p) for p in files},
        source_sha256={str(p):E.sha(p) for p in [Path(__file__),Path(M.__file__),Path(R.__file__),Path(E.__file__)]},
        m3=dict(pose_lambda=.25,dt=.2,pose_seed_prefix=2026100306,K=4,history=8,frames=FRAMES.tolist(),five_seed_mean=True,threshold=D.THRESHOLD),
        arms=['A_M3_noisy','B_M3_true','M3_quarter','C_oracle8_true','oracle1_true','oracle2_true','oracle4_true','D_oracle8_belief','oracle2_belief','oracle4_belief'],
        belief=dict(formula='sensor[f] @ inv(noisy[k,f]) @ noisy[k,past]; lambda1 uses stored noisy, quarter uses coupled scaled noise draws before integration',
            candidate_indices=TIDS,positive_indices=[0,1],negative_indices=[2,3,4],
            rendering='Entire scene including background at each anchored belief pose; sub16, noise_scale0; no latent counts or estimated true displacement identity supplied to score',
            windows=[2,4,8],anchors=FRAMES.tolist(),history_exposures_per_K=int(len(PAST)),past=PAST.tolist(),anchor_offsets=START.tolist(),
            unknowns='This oracle retains exact target size/rho, background, true current anchor and nominal sensor model; estimated relative motion only is intervened'),
        check=dict(lambda0_units=[110001,110008],criterion='Same belief render/score path lambda0 reproduces every retained oracle8 score in >=2scenes with max abs error<1e-8; failure stops D'),
        primary='Within-scene macroAUC at1.2-2.1m, inside1/2 vs outside10/15/20;48equal independent scenes',
        bootstrap=dict(n=1000,seed=2026100402,unit='whole scenes;K/delta/frames paired; fixed scores and descriptive thresholds'),
        visibility='Full13 retained frames, fraction of inside1/2 branch-frame pairs with >=1 first-visible target ray; vf>=.5=36IN vs12OUT. This is a return-visibility proxy, not a strict geometric FOV/occlusion classifier.',
        matching='Two separate outside15/20 primary/domain workpoints; A frozen >=threshold integer budgets; largest whole-score tie groups with strict > threshold for other arms; global48 thresholds retained in strata and full13frame sequences',
        sequence='First alarm over all13 retained frames; shallow inside1/2 xK4, timely if first front>=.9; lead=(front_first-.5)/.8 conditioned on timely alarm; outside15/20 first-stops counted separately. Descriptive early rescues>=2.1m and direct-return visibility',
        branch=dict(low='D-A AUC<=.03 and both workpoints inner1/2 four recall differences<=.05 -> POSE_LIMITED',
            high='D-A AUC>=.06 or any four recall differences>=.10 -> HEADROOM_SURVIVES_POSE',otherwise='INTERMEDIATE'),
        quarter_oracle='Run only if D is POSE_LIMITED or INTERMEDIATE; M3 quarter runs unconditionally; quarter shrinks scale,bias,translation-stepnoise,rotation-stepnoise with same RNG draws',
        interpretation='Plug-in misaligned likelihood may overconfidently fail; POSE_LIMITED does not prove marginalized pose inference has no headroom. B input differs from noisy-pose training; all estimates are assumed noise not device calibration.',
        decision_mapping=dict(HEADROOM_SURVIVES_POSE='Propose per-frame hist+relative-pose temporal pilot, far-range weighting, new displacement evaluation batch, >=3seeds perarm; no automatic training',
            POSE_LIMITED='Recommend pause readout search; report lambda response and pose accuracy requirement conditional on assumed noise',INTERMEDIATE='Report for discussion; no automatic work'),
        budget=dict(compute_seconds=3600,render_workers=4,full_scope='48scenes,K4,all13anchors and5hypotheses',
            reduction='Only prospectively via PLAN_AMENDMENT before reduced computation if budget forecast exceeds60min; restrict anchors .9-2.6m orK2paired withA; never silently drop scenes',
            cpu_noise_check_seconds=16.30)))
    print('PLAN frozen',E.sha(OUT/'PLAN.json'),flush=True)


def unit_inputs(u):
    p=read(OUT/'PLAN.json')
    for sub,ext in [('observations','npz'),('truth','json'),('templates','npz'),('scores','npz')]:
        path=D.OUT/f'{sub}/unit{u}.{ext}'
        assert E.sha(path)==p['input_sha256'][str(path)]


def render_one(u,lam):
    tick=time.monotonic();unit_inputs(u)
    truth=read(D.OUT/'truth'/f'unit{u}.json')
    with np.load(D.OUT/'observations'/f'unit{u}.npz') as ob:
        sensor=ob['sensor'].copy();noisy=ob['noisy'].copy();ambient=ob['ambient'].copy()
    if lam==0:noisy=np.repeat(sensor[None],4,axis=0)
    elif lam==.25:
        posefile=OUT/'m3/noisy_quarter'/f'unit{u}.npz'
        pose_receipt=read(OUT/'m3'/f'unit{u}.json')
        relative=str(posefile.relative_to(OUT)).replace('\\','/')
        assert E.sha(posefile)==pose_receipt['output_sha256'][relative]
        with np.load(posefile) as quarter:
            np.testing.assert_array_equal(sensor,quarter['sensor'])
            noisy=quarter['noisy_quarter'].copy()
    elif lam!=1:raise ValueError('Unsupported lambda')
    expected=np.empty((4,5,len(PAST),8,8,16),np.float64)
    for k in range(4):
        poses=np.concatenate([sensor[f]@np.linalg.inv(noisy[k,f])@noisy[k,max(0,f-7):f+1] for f in FRAMES])
        for j,di in enumerate(TIDS):
            quiet=R.expected(dict(poses=poses,boxes=truth['boxes'][di]))
            np.testing.assert_array_equal(quiet['ambient'],ambient[PAST])
            expected[k,j]=quiet['expectation']
    folder=OUT/('lambda0-check' if lam==0 else f'belief-lambda{lam:g}')
    folder.mkdir(exist_ok=True)
    path=folder/f'unit{u}.npz'
    if path.exists():raise FileExistsError('Do not overwrite rendered templates')
    np.savez(path,expected=expected,past=PAST,offsets=START)
    return dict(unit=u,lambda_value=lam,seconds=time.monotonic()-tick,path=str(path),sha256=E.sha(path))


def render(lam,workers):
    if lam not in (1,.25):raise ValueError('Use check stage for lambda0')
    assert read(OUT/'lambda0_check.json')['status']=='PASS'
    path=OUT/f'render_lambda{lam:g}.json'
    if path.exists():raise FileExistsError('Rendering already completed')
    tick=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(render_one,u,lam) for u in D.UNITS]
        for job in as_completed(jobs):
            records.append(job.result());print('RENDER',lam,len(records),'/48',records[-1]['unit'],flush=True)
    save(path,dict(status='COMPLETE',plan_sha256=E.sha(OUT/'PLAN.json'),seconds=time.monotonic()-tick,workers=workers,units=sorted(records,key=lambda x:x['unit'])))


def belief_scores(u,path):
    import cnh_unknown_target_gpu as G
    with np.load(path) as tmp:expected=tmp['expected']
    with np.load(D.OUT/'observations'/f'unit{u}.npz') as ob:hist=ob['hist'];ambient=ob['ambient'][PAST]
    scores={w:np.empty((7,4,13),np.float64) for w in (2,4,8)}
    for k in range(4):
        ll=G.skellam_logpmf_signed_gpu(hist[:,k,PAST][:,None],expected[k][None],8*ambient[None,None,...,None]).sum(axis=(-3,-2,-1))
        for ai in range(13):
            end=START[ai+1]
            for w in scores:
                begin=max(START[ai],end-w)
                total=ll[:,:,begin:end].sum(-1)
                scores[w][:,k,ai]=logsumexp(total[:,:2],axis=1)-np.log(2)-logsumexp(total[:,2:],axis=1)+np.log(3)
    return scores


def check():
    tick=time.monotonic();records=[]
    with np.load(PRIOR/'scores/all_scores.npz') as old:oracle=old['oracle8'].copy()
    for u in read(OUT/'PLAN.json')['check']['lambda0_units']:
        receipt=render_one(u,0)
        scores=belief_scores(u,receipt['path'])
        i=D.UNITS.index(u)
        error=float(np.max(np.abs(scores[8]-oracle[i])))
        with np.load(receipt['path']) as tmp, np.load(D.OUT/'templates'/f'unit{u}.npz') as old:
            expected=old['expected'][TIDS][:,PAST]
            expectation_error=float(np.max(np.abs(tmp['expected']-expected[None])))
        assert error<1e-8,(u,error)
        records.append(dict(**receipt,oracle8_max_abs_error=error,expectation_max_abs_error=expectation_error))
    save(OUT/'lambda0_check.json',dict(status='PASS',plan_sha256=E.sha(OUT/'PLAN.json'),seconds=time.monotonic()-tick,units=records))
    print('LAMBDA0 PASS',records,flush=True)


def assemble(lam):
    tick=time.monotonic()
    receipt=read(OUT/f'render_lambda{lam:g}.json')
    assert receipt['status']=='COMPLETE' and receipt['plan_sha256']==E.sha(OUT/'PLAN.json')
    scorepath=OUT/'scores/all_scores.npz';(OUT/'scores').mkdir(exist_ok=True)
    if lam==1:
        if scorepath.exists():raise FileExistsError('Scores already assembled')
        with np.load(PRIOR/'scores/all_scores.npz') as old:
            assert old['units'].tolist()==D.UNITS
            scores=dict(A_M3_noisy=old['M3_smooth'].copy(),C_oracle8_true=old['oracle8'].copy())
        pending={name:[] for name in ['B_M3_true','M3_quarter','oracle1_true','oracle2_true','oracle4_true','D_oracle8_belief','oracle2_belief','oracle4_belief']}
        scenes=[]
    else:
        with np.load(scorepath) as source:scores={key:source[key].copy() for key in source.files if key!='units'}
        pending={'oracle8_quarter':[]}
    for i,u in enumerate(D.UNITS):
        unit_inputs(u);record=receipt['units'][i];assert record['unit']==u and E.sha(record['path'])==record['sha256']
        belief=belief_scores(u,record['path'])
        if lam==1:
            for w in (2,4,8):pending['D_oracle8_belief' if w==8 else f'oracle{w}_belief'].append(belief[w])
            truth=read(D.OUT/'truth'/f'unit{u}.json')
            with np.load(OUT/'m3'/f'unit{u}.npz') as m:
                for key,outkey in [('raw_true','B_M3_true'),('raw_quarter','M3_quarter')]:pending[outkey].append(E.m3_smooth(m[key])[...,truth['group']])
            with np.load(D.OUT/'observations'/f'unit{u}.npz') as ob, np.load(D.OUT/'templates'/f'unit{u}.npz') as tmp:
                ll=E.frame_likelihood(ob['hist'],tmp['expected'],ob['ambient'])
                visible=(tmp['object_id'][:,FRAMES]==0).any(axis=(-3,-2,-1))
            for w in (1,2,4):pending[f'oracle{w}_true'].append(E.mixture_llr(ll,[0,1],[3,4,5],w).reshape(7,4,16)[...,FRAMES])
            scenes.append(dict(unit=u,group=truth['group'],context=truth['context'],front_range_m=np.array(truth['front_range_m'])[FRAMES].tolist(),target_visible_frame=visible.tolist(),visibility_fraction=float(visible[:2].mean())))
        else:pending['oracle8_quarter'].append(belief[8])
        print('SCORE',lam,i+1,'/48',u,flush=True)
    scores.update({key:np.stack(value) for key,value in pending.items()})
    if lam==.25:
        # Keep the primary D result and its score file immutable.
        scorepath=OUT/'scores/all_scores_with_quarter.npz'
        if scorepath.exists():raise FileExistsError('Quarter scores already completed')
    np.savez(scorepath,units=np.array(D.UNITS),**scores)
    if lam==1:save(OUT/'scenes.json',scenes)
    import cnh_unknown_target_gpu as G
    scoring_receipt=dict(status='COMPLETE',plan_sha256=E.sha(OUT/'PLAN.json'),seconds=time.monotonic()-tick,
        source_sha256={str(p):E.sha(p) for p in [Path(__file__),Path(G.__file__),Path(E.__file__)]},
        m3_receipt_sha256=E.sha(OUT/'m3/inference_receipt.json'),
        render_receipt_sha256=E.sha(OUT/f'render_lambda{lam:g}.json'),
        output_sha256={str(scorepath.relative_to(OUT)).replace('\\','/'):E.sha(scorepath),'scenes.json':E.sha(OUT/'scenes.json')})
    save(OUT/f'scores_lambda{lam:g}_receipt.json',scoring_receipt)
    if lam==1:save(OUT/'scores_receipt.json',scoring_receipt)


if __name__=='__main__':
    setup();p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['plan','check','render','assemble']);p.add_argument('--lambda-value',type=float,default=1);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    {'plan':plan,'check':check,'render':lambda:render(a.lambda_value,a.workers),'assemble':lambda:assemble(a.lambda_value)}[a.stage]()
