"""Consumed Development unknown-size/rho diagnostic; no training or reader fitting."""
import argparse
import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import cnh_displacement_ceiling as D  # loads the frozen sensor snapshot first
import cnh_displacement_ceiling_render as R
import cnh_displacement_ceiling_evaluate as E
from scipy.special import logsumexp

OLD = D.OUT
OUT = OLD.parent/'cnh-unknown-target-reference-20261004'
RUN = 'CNH_UNKNOWN_TARGET_REFERENCE_20261004'
SHAPES = [(w, None, depth) for w in (.09, .11) for depth in (.09, .15)] + [(.02, .04, .04)]
RHOS = [.22+(.65-.22)*(i+.5)/3 for i in range(3)]
WEIGHTS = np.array([.9/12]*12+[.1/3]*3)
TEMPLATE_IDS = [0, 1, 2, 4, 5, 6]


def read(p):
    return E.read(p)


def plan():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'PLAN.json'
    if path.exists():
        raise FileExistsError('Resume the existing frozen plan, do not overwrite')
    sources = [Path(__file__), Path(R.__file__), Path(D.SS.__file__), Path(D.S.__file__), D.S.SOURCE/'cnh_route_sensor.py']
    inputs = {}
    for unit in D.UNITS:
        for sub, ext in [('truth','json'), ('observations','npz'), ('templates','npz'), ('scores','npz'), ('evaluation_units','json')]:
            p = OLD/sub/f'unit{unit}.{ext}'
            inputs[str(p.relative_to(OLD)).replace('\\','/')] = E.sha(p)
    record = dict(run=RUN, created_utc=datetime.now(timezone.utc).isoformat(), units=D.UNITS,
        role='EXPLORE consumed Development measurement only; no new training/readout; cohort remains evaluation-only',
        old_root=str(OLD), input_sha256=inputs, old_plan_sha256=E.sha(OLD/'PLAN.json'), old_result_sha256=E.sha(OLD/'result.json'),
        source_sha256={str(p):E.sha(p) for p in sources},
        prior=dict(name='fixed extended discrete hypothesis prior', shapes=SHAPES, rhos=RHOS,
            grid_weights=WEIGHTS.tolist(), regular_mass=.9, tiny_mass=.1,
            regular_height_by_group=[.36,.34], vertical_center_by_group=[.08,.67], front_z=.65,
            generator='width U(.08,.12), depth U(.06,.18), rho U(.22,.65), group height point mass .36/.34; no tiny branch',
            quadrature='regular width/depth two equal-mass midpoint bins, rho three equal-mass midpoint bins; tiny is an explicit added pressure component',
            limits='Finite grid can lose likelihood through mismatch. A low U result is not an impossibility proof or strict information bound.'),
        positions=D.INTRUSION_CM.tolist(), frames=D.FRAMES.tolist(), windows=[8,12],
        class_mixture='Within each class grid weights sum1; inside templates1/2cm equally weighted, outside10/15/20cm equally weighted; inside5/outside5 rendered but excluded from class mixtures',
        temporal_mixture='Accumulate each fixed size/rho/displacement log likelihood over causal window before mixing; no framewise nuisance resampling',
        privileges='Exact background geometry/rho, true sensor poses, side, HEAD/BODY band and center, front position .65, exact photon model; no true target width/depth/rho read for U',
        likelihood='Exact signed Skellam P(E+8ambient)-P(8ambient), zero-noise templates, renderer sub16',
        oracle='User authorized reconstruction of frozen oracle scores from existing hist/templates solely because old output did not retain frame scores; no rerender or model inference',
        m3='Reuse raw stored logits and inherited deterministic smoothing, no model inference or threshold change',
        matching='Two descriptive workpoints separately calibrated on outside15 and outside20 branch scores in each distance domain; target is M3 frozen >=.8557642486787612 frame rate. Candidate strict > thresholds include largest whole tie groups under same integer false-alert count; no subgroup calibration.',
        bootstrap=dict(n=1000, seed=2026100401, unit='48 independent scenes; K4/frames/positions remain paired inside scenes; fixed descriptive thresholds; excludes seed/calibration uncertainty'),
        branch=dict(low='U8 minus M3_smooth primary macroAUC <=.03 AND all inner1/2cm recall differences at both workpoints <=.05: NEAR_REALISTIC_REFERENCE; recommend pause variant search conditional on this reference',
            high='AUC difference >=.06 OR any inner1/2cm recall difference at either workpoint >=.10: REALISTIC_HEADROOM; locate strata only, no automatic training',
            intermediate='Otherwise INTERMEDIATE. Rules use signed point differences, not equivalence tests.'),
        execution='All CPU template renders must finish before GPU scientific likelihood computation; rho affine interpolation verified independently with opaque rho0 target retained',
        measurement_budget_seconds=2400, real_replay='Located; waiting for device session with xtalk/bias, ambient, geometry, empty corridor; no further01/04 analysis')
    E.save(path, record)
    print('PLAN frozen', E.sha(path), flush=True)


def validate_inputs(unit):
    p=read(OUT/'PLAN.json')
    for folder,ext in [('truth','json'),('observations','npz'),('templates','npz'),('scores','npz'),('evaluation_units','json')]:
        rel=f'{folder}/unit{unit}.{ext}'
        assert E.sha(OLD/rel)==p['input_sha256'][rel], rel


def candidate(background, poses, side, group, shape, penetration, rho):
    w,h,depth=shape
    h=(.36,.34)[group] if h is None else h
    yc=(.08,.67)[group]
    a=.30-penetration/100
    xlo,xhi=(a,a+w) if side>0 else (-a-w,-a)
    box=dict(lo=[xlo,yc-h/2,.65],hi=[xhi,yc+h/2,.65+depth],rho=rho)
    for b in background:
        overlap=np.minimum(box['hi'],b['hi'])-np.maximum(box['lo'],b['lo'])
        assert not (overlap>0).all(), 'Fixed candidate intersects background'
    return dict(poses=poses, boxes=[box,*copy.deepcopy(background)])


def render_one(unit):
    started=time.monotonic()
    validate_inputs(unit)
    truth=read(OLD/'truth'/f'unit{unit}.json')
    with np.load(OLD/'observations'/f'unit{unit}.npz') as ob:
        poses=ob['sensor'].copy()
    # Only side identity is read from target; no dimensions/rho/center copied.
    side=1 if truth['boxes'][0][0]['lo'][0]>0 else -1
    background=truth['boxes'][0][1:]
    assert all(branch[1:]==background for branch in truth['boxes'])
    endpoints=np.empty((5,7,2,16,8,8,16),np.float64)
    for si,shape in enumerate(SHAPES):
        for di,p in enumerate(D.INTRUSION_CM):
            for ri,rho in enumerate([0.,.65]):
                result=R.expected(candidate(background,poses,side,truth['group'],shape,p,rho))
                endpoints[si,di,ri]=result['expectation']
                if si==di==ri==0:
                    ambient=result['ambient'].copy()
                else:
                    np.testing.assert_array_equal(ambient,result['ambient'])
    np.savez(OUT/'templates'/f'unit{unit}.npz', endpoints=endpoints, ambient=ambient)
    return dict(unit=unit, seconds=time.monotonic()-started, sha256=E.sha(OUT/'templates'/f'unit{unit}.npz'))


def render(workers):
    started=time.monotonic()
    assert read(OUT/'PLAN.json')['run']==RUN
    (OUT/'templates').mkdir(exist_ok=True)
    # Engineering check uses neither scientific cohort scores nor scene truth.
    poses=np.repeat(np.eye(4)[None],2,axis=0)
    poses[:,:3,:3]=D.S.ry(15)@D.S.rx(-10)
    poses[:,2,3]=[-1.2,-1.04]
    back=[dict(lo=[-8,-3,4.5],hi=[8,2,4.7],rho=.35)]
    values=[R.expected(candidate(back,poses,1,0,SHAPES[0],1,r))['expectation'] for r in [0.,.65,RHOS[1]]]
    interpolated=values[0]+RHOS[1]/.65*(values[1]-values[0])
    np.testing.assert_allclose(values[2],interpolated,atol=1e-9,rtol=1e-12)
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(render_one,u):u for u in D.UNITS}
        for f in as_completed(futures):
            records.append(f.result())
            print('RENDER',len(records),'/48',records[-1]['unit'],round(records[-1]['seconds'],2),flush=True)
    E.save(OUT/'render_receipt.json',dict(status='COMPLETE',plan_sha256=E.sha(OUT/'PLAN.json'),
        seconds=time.monotonic()-started, workers=workers, units=sorted(records,key=lambda x:x['unit']),
        rho_affine_max_error=float(np.max(np.abs(values[2]-interpolated)))))


def score():
    started=time.monotonic()
    receipt=read(OUT/'render_receipt.json')
    assert receipt['status']=='COMPLETE' and len(receipt['units'])==48
    assert receipt['plan_sha256']==E.sha(OUT/'PLAN.json')
    # GPU only imported after every render and its receipt is complete.
    for name,folder in [('TEMP','tmp'),('TMP','tmp'),('CUPY_CACHE_DIR','cupy-cache')]:
        directory=OUT/folder
        directory.mkdir(exist_ok=True)
        os.environ[name]=str(directory)
    import cnh_unknown_target_gpu as G
    numerical_check=read(OUT/'gpu_check.json')
    assert numerical_check['status']=='PASS_SYNTHETIC_AND_RENDERED'
    assert numerical_check['module_sha256']==E.sha(Path(G.__file__))
    assert numerical_check['cpu_reference_module_sha256']==E.sha(Path(E.__file__))
    assert numerical_check['rendered_prior_sha256']==E.sha(OUT/'PLAN.json')
    assert numerical_check['rendered_templates_sha256']==receipt['units'][0]['sha256']
    assert numerical_check['rendered_observation_sha256']==read(OUT/'PLAN.json')['input_sha256'][f'observations/unit{D.UNITS[0]}.npz']
    all_scores={name:[] for name in ['M3_smooth','oracle8','oracle12','U8','U12']}
    scenes=[]
    parity_errors=[]
    for index,unit in enumerate(D.UNITS):
        validate_inputs(unit)
        path=OUT/'templates'/f'unit{unit}.npz'
        assert E.sha(path)==receipt['units'][index]['sha256']
        truth=read(OLD/'truth'/f'unit{unit}.json')
        with np.load(path) as templates:
            ends=templates['endpoints']
            # [shape,rho,position,frame,row,col,radial] -> [candidate,frame,...]
            utemplates=np.stack([ends[:,:,0]+rho/.65*(ends[:,:,1]-ends[:,:,0]) for rho in RHOS],axis=1)
            utemplates=utemplates[:,:,TEMPLATE_IDS].reshape(90,16,8,8,16)
            ambient=templates['ambient'].copy()
        with np.load(OLD/'observations'/f'unit{unit}.npz') as ob:
            hist=ob['hist'].copy()
            np.testing.assert_array_equal(ambient,ob['ambient'])
        likelihood=G.frame_likelihood(hist,utemplates,ambient)
        ll=likelihood.reshape(28,15,6,16)
        logweights=np.log(WEIGHTS)[None,:,None,None]
        for window in [8,12]:
            totals=E.rolling_sum(ll,window)+logweights
            value=logsumexp(totals[:,:,:2].reshape(28,30,16),axis=1)-np.log(2)
            value-=logsumexp(totals[:,:,3:].reshape(28,45,16),axis=1)-np.log(3)
            all_scores[f'U{window}'].append(value.reshape(7,4,16)[...,D.FRAMES])
        # Keep the original CPU numerical implementation for exact frozen-score
        # reconstruction, including its floating-point ties.
        with np.load(OLD/'templates'/f'unit{unit}.npz') as old:
            oll=E.frame_likelihood(hist,old['expected'],ambient)
        for window in [8,12]:
            value=E.mixture_llr(oll,[0,1],[3,4,5],window).reshape(7,4,16)[...,D.FRAMES]
            all_scores[f'oracle{window}'].append(value)
        with np.load(OLD/'scores'/f'unit{unit}.npz') as raw:
            all_scores['M3_smooth'].append(E.m3_smooth(raw['raw'])[...,truth['group']])
        # Original per-scene AUC receipt verifies reconstruction parity.
        oldscene=read(OLD/'evaluation_units'/f'unit{unit}.json')['scene']
        front=np.array(truth['front_range_m'])[D.FRAMES]
        for key,cells in oldscene['cells'].items():
            lo,hi=(1.2,2.1) if key=='primary1.2-2.1m' else map(float,key[:-1].split('-'))
            keep=(front>=lo)&(front<hi)
            for arm in ['oracle8','oracle12','M3_smooth']:
                actual=E.auc_cell(all_scores[arm][-1],keep)['auc']
                previous=cells[arm]['auc']
                if previous is not None:
                    parity_errors.append(abs(actual-previous))
        scenes.append(dict(unit=unit,group=truth['group'],context=truth['context'],front_range_m=front.tolist()))
        print('SCORE',index+1,'/48',unit,flush=True)
    assert max(parity_errors)<1e-10, max(parity_errors)
    (OUT/'scores').mkdir(exist_ok=True)
    target=OUT/'scores/all_scores.npz'
    np.savez(target,units=np.array(D.UNITS),**{arm:np.stack(value) for arm,value in all_scores.items()})
    E.save(OUT/'scenes.json',scenes)
    E.save(OUT/'scores_receipt.json',dict(status='COMPLETE',plan_sha256=E.sha(OUT/'PLAN.json'),
        seconds=time.monotonic()-started, numerical_check=numerical_check, original_auc_max_error=max(parity_errors),
        output_sha256={'scores/all_scores.npz':E.sha(target),'scenes.json':E.sha(OUT/'scenes.json')},
        source_sha256={str(p):E.sha(p) for p in [Path(__file__),Path(G.__file__),Path(E.__file__)]}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['plan','render','score'])
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    {'plan':plan,'render':lambda:render(args.workers),'score':score}[args.stage]()
