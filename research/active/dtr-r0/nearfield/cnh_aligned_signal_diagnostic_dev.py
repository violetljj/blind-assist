"""Privileged analytic target-present/absent photon diagnostic on old payloads.

No model inference, sampled new observations, threshold selection or training.
An absent scene removes the target; existing clear-target scenes are never used
as the absent reference. Negative mean differences from occlusion are retained.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import time

import numpy as np
import cnh_displacement_ceiling_render as R

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'artifacts.local/work/cnh-aligned-boundary-dev-20261008'
OUT = ROOT/'artifacts.local/work/cnh-aligned-shapes-dev-20261008/signal'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def separation(mu0,mu1,ambient):
    delta=mu1-mu0
    variance=.5*(mu0+mu1)+16*ambient[...,None]
    term=np.divide(delta**2,variance,out=np.zeros_like(delta),where=variance>0)
    if np.any((variance==0)&(delta!=0)):raise ValueError('Nonzero delta with zero variance')
    return term.sum(axis=(-3,-2,-1)),delta


def stats(a):
    a=np.asarray(a,float)
    return None if not a.size else dict(n=int(a.size),min=float(a.min()),p25=float(np.quantile(a,.25)),
        median=float(np.median(a)),p75=float(np.quantile(a,.75)),max=float(a.max()))


def run(bright=False):
    out = OUT/'v2' if bright else OUT
    started=time.monotonic(); cpu=time.process_time()
    out.mkdir(parents=True,exist_ok=False)
    def check():
        if time.monotonic()-started>=60:raise TimeoutError('60 second wall budget reached')
    planfile=OLD/'PLAN.json'; old=json.loads(planfile.read_text(encoding='utf8'))
    rows=old['scene_rows'];ids=np.array([i for i,r in enumerate(rows) if r['rho']==.25 or (bright and r['side']==.04 and r['rho']==.65)],int)
    physical=[OLD/'physical'/f'pitch_{p}.npz' for p in (-10.,old['candidate_pitch_deg'])]
    source={str(Path(m.__file__).resolve()):sha(m.__file__) for name,m in list(sys.modules.items())
            if name.startswith('cnh_') and getattr(m,'__file__',None)}
    source[str(Path(__file__).resolve())]=sha(__file__)
    inputs={str(p.relative_to(ROOT)):sha(p) for p in (planfile,OLD/'evaluated.npz',*physical)}
    if bright:
        inputs[str((OUT/'signal_arrays.npz').relative_to(ROOT))]=sha(OUT/'signal_arrays.npz')
        inputs[str((OUT/'source_v1.py').relative_to(ROOT))]=sha(OUT/'source_v1.py')
    # Already-consumed outcomes: this is a prespecified descriptive calculation,
    # not a blind scientific selection or independent confirmation.
    plan=dict(lane='EXPLORE privileged analytic description on consumed payloads',
        wall_budget_seconds=60,source_sha256=source,input_sha256=inputs,
        scene_ids=ids.tolist(),size_rho_pairs=[[.04,.25],[.04,.65],[.10,.25]] if bright else [[.04,.25],[.10,.25]],
        absent='Reuse v1 saved background_expectation; no render' if bright else 'Frozen R.expected(original background, original physical sensor poses), target removed, 16 frames per existing pitch only',
        formula='d2_frame=sum_bins((mu1-mu0)^2 / (.5*(mu0+mu1)+16*ambient)); delta signed including occlusion',
        noise='Signed=Poisson(mu+8ambient)-Poisson(8ambient); Var=mu+16ambient; denominator averages the two marginal variances',
        windows='Single frame, cumulative prefix, and past at most8 exposure frames. Output f3..f13 before original .9m timely deadline; f14/15 preserved separately.',
        strata='Contact scenes only: original target-height M3 timely/not timely at frozen original theta, K4 replicas. Geometry expectation reused across replicas; counts are descriptive noise replicas, not independent objects.',
        alarm_contract='Baseline and candidate both fixed original theta; candidate_fixed_timely, never candidate_matched_timely',
        exclusions='Known target/background/pose privilege; Gaussian diagonal standardized separation proxy is not exact Bayes accuracy or deployable detector. No SNR cutoff, causal attribution, learning feasibility, or unique failure mechanism inferred.')
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
    if bright:
        (out/'source.py').write_bytes(Path(__file__).read_bytes())
        with np.load(OUT/'signal_arrays.npz') as saved:
            saved_background=saved['background_expectation']
    # Narrow arithmetic check (including a shadow/negative delta term).
    a=np.full((1,8,8,16),3.);b=a.copy();b[0,0,0,0]=5.;b[0,0,0,1]=1.
    d,_=separation(a,b,np.full((1,8,8),4.))
    np.testing.assert_allclose(d,[4/68+4/66],rtol=0,atol=1e-15)
    arrays=[];backgrounds=[]
    for arm,p in enumerate(physical):
        check()
        with np.load(p) as z:
            sensor=z['sensor'];ambient=z['ambient'];mu1=z['expectation'][ids]
        if bright:
            mu0=saved_background[arm]
        else:
            bg=R.expected(dict(poses=sensor,boxes=old['background']))
            np.testing.assert_array_equal(bg['ambient'],ambient)
            mu0=bg['expectation']
        d2,delta=separation(mu0[None],mu1,ambient[None])
        window=np.stack([d2[:,max(0,f-7):f+1].sum(-1) for f in range(16)],-1)
        cumulative=np.cumsum(d2,-1)
        arrays.append(dict(d2=d2,window=window,cumulative=cumulative,
            positive_delta=np.maximum(delta,0).sum((-3,-2,-1)),negative_delta=np.minimum(delta,0).sum((-3,-2,-1)),
            nonzero_bins=(delta!=0).sum((-3,-2,-1))))
        backgrounds.append(mu0)
        check()
    # Only now associate existing evaluator outcome strata.
    with np.load(OLD/'evaluated.npz') as z:
        category=z['category'][ids]
        timely=np.stack([z['baseline_timely'][ids],z['candidate_fixed_timely'][ids]])
    summaries=[];records=[]
    for arm,pitch in enumerate((-10.,old['candidate_pitch_deg'])):
        a=arrays[arm]
        for side,rho in ([(.04,.25),(.04,.65),(.10,.25)] if bright else [(.04,.25),(.10,.25)]):
            selected=np.array([rows[i]['side']==side and rows[i]['rho']==rho for i in ids])
            for q,height in enumerate(('HEAD','BODY')):
                for status in (True,False):
                    examples=[]
                    for j,i in enumerate(ids):
                        if not selected[j] or category[j,q]!='contact':continue
                        for k in range(timely.shape[2]):
                            if bool(timely[arm,j,k,q])!=status:continue
                            r=dict(pitch_deg=pitch,scene_id=int(i),replica=k,side_m=side,rho=rho,height=height,timely=status,
                                d_max_frame_predeadline=float(np.sqrt(a['d2'][j,3:14].max())),
                                d_max_past8_predeadline=float(np.sqrt(a['window'][j,3:14].max())),
                                d_prefix_at_deadline=float(np.sqrt(a['cumulative'][j,13])),
                                d_past8_at_deadline=float(np.sqrt(a['window'][j,13])),
                                negative_delta_predeadline=float(a['negative_delta'][j,:14].sum()),
                                positive_delta_predeadline=float(a['positive_delta'][j,:14].sum()))
                            records.append(r);examples.append(r)
                    summaries.append(dict(pitch_deg=pitch,side_m=side,rho=rho,height=height,timely=status,
                        replicas=len(examples),distinct_scenes=len(set(r['scene_id'] for r in examples)),
                        d_max_frame=stats([r['d_max_frame_predeadline'] for r in examples]),
                        d_max_past8=stats([r['d_max_past8_predeadline'] for r in examples]),
                        d_prefix=stats([r['d_prefix_at_deadline'] for r in examples])))
    np.savez_compressed(out/'signal_arrays.npz',scene_ids=ids,
        background_expectation=np.stack(backgrounds),**{k:np.stack([a[k] for a in arrays]) for k in arrays[0]})
    with (out/'contact_strata.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    check()
    result=dict(status='COMPLETE',scene_probes=len(ids),contact_replica_rows=len(records),summaries=summaries,
        arithmetic_check='PASS',background_ambient_parity='v1 exact check retained' if bright else 'exact',background_exposures=0 if bright else 32,
        wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
        interpretation=plan['exclusions'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run(bright='--bright' in sys.argv[1:])
