"""Post-hoc EXPLORE replay of the past-position-smoothed torso-bias gate.

Only the new gait correction is inferred. Original observations, window draws,
M3 scores, and comparison arms remain frozen; the first pilot is preserved.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import cnh_torso_bias_replay_dev as B

OUT=B.OUT/'gait_mechanism'
SEED=B.SEED


def new_spent():
    return sum(float(B.A.read(p)['seconds']) for p in (OUT/'replay_attempts').glob('attempt*.json'))


def shared_spent():
    return B.spent_seconds()+new_spent()


def load_bank():
    with np.load(OUT/'series.npz',allow_pickle=False) as f:
        d={k:f[k] for k in f.files}
    with np.load(B.OUT/'series.npz',allow_pickle=False) as f:
        for k in ('e1','torso','oracle','valid','file_index','frame_index'):
            np.testing.assert_array_equal(d[k],f[k])
    windows=B.A.read(B.OUT/'windows.json')
    for w in windows:
        ix=int(w['start'])+np.arange(192)
        if not np.isfinite(d['corrected_gait'][ix]).all():
            raise ValueError('Gait correction missing from original common window')
    return d,windows


def freeze():
    paths=[OUT/'series.npz',B.OUT/'windows.json',B.OUT/'replay_plan.json',
        B.OUT/'PLAN.json',OUT/'PLAN.json',HERE/'cnh_torso_gait_bias_dev.py',Path(__file__),
        HERE/'cnh_torso_bias_replay_dev.py',*B.A.M3_MODELS]
    plan=dict(task='CNH_TORSO_GAIT_BIAS_REPLAY_DEV_20261007',lane='POSTHOC EXPLORE',
        reason='First pilot gate admitted few updates and correction barely changed raw torso RMS; change past displacement signal, not thresholds after evaluation',
        units=B.A.UNITS,arm='corrected_gait',original_arms=B.ARMS,seed=SEED,
        draws='Identical original heldout P06..P10 window IDs/signs/native indices, checked per sequence against original unit files',
        query_sign='Negative of signed estimator-minus-future-pelvis1.5m truth error',
        freeze='Original M3,stored inputs,smoothing,geometry,control/main/startup definitions retained; infer only corrected_gait',
        budget=dict(shared_replay_seconds=1800,shared_analysis_seconds=600,
                    accounting='Sum original and gait replay attempt seconds, including failed resumed attempts; setup separate'),
        thresholds='Original E1 actual evaluation integer mainFA budgets at2.5%/5%; gait smallest whole-tie feasible threshold; separate cal-only2.5%/5%',
        uncertainty='1000 evaluation-unit paired draws; fixed thresholds; no source-window,participant,calibration,training or postselection uncertainty',
        scope='Observed first pilot evaluation motivated this mechanism: not fresh confirmation, hardware, or glasses torso evidence',
        hashes={str(p.relative_to(B.A.ROOT)):B.A.sha(p) for p in paths})
    path=OUT/'replay_plan.json'
    if path.exists():
        if B.A.read(path)!=B.json_value(plan):raise ValueError('Frozen gait replay input/source changed')
    else:B.save(path,plan)


def run_unit(rn,d,windows,unit,available):
    import cnh_cvr_pilot as CP
    dest=OUT/'replay_units'/f'unit{unit}.npz'
    if dest.exists():return
    tick=time.monotonic();status='FAILED';error=None
    def check():
        if time.monotonic()-tick>=available:raise TimeoutError('Shared1800-second replay budget reached')
    try:
        original=B.OUT/'replay_units'/f'unit{unit}.npz'
        with np.load(original,allow_pickle=False) as f:old={k:f[k] for k in f.files}
        obs,stored=B.A.stored(unit)
        try:
            C=len(obs['noisy_center']);draws=[B.sample(d,windows,unit,c) for c in range(C)]
            ix=np.stack([v[2] for v in draws]);sg=np.array([v[1] for v in draws]);wi=np.array([v[0] for v in draws])
            np.testing.assert_array_equal(ix,old['native_series_index']);np.testing.assert_array_equal(wi,old['window']);np.testing.assert_array_equal(sg,old['sign'])
            for a in ('e1','torso','oracle'):np.testing.assert_array_equal(d[a][ix]*sg[:,None],old[a+'_heading_err'])
            error_deg=d['corrected_gait'][ix]*sg[:,None]
            q=np.repeat(obs['public_query'][None],C,0).copy()
            for c in range(C):
                for f in range(16):q[c,f,:3,:3]=CP.rotation(float(-error_deg[c,f]),'y')@obs['public_query'][f,:3,:3]
            raw=np.empty((3,C,13,2),np.float32)
            for s,a in enumerate(B.A.ANGLES):
                check();ex=rn.eng.N.extrinsic(a);raw[s]=rn.raw(obs['z1'][s],obs['noisy_center']@ex,q@ex)
            check()
            result=dict(unit=unit,raw=raw,heading_err=error_deg,query_err=-error_deg,
                window=wi,sign=sg,native_series_index=ix,native_frame_index=d['frame_index'][ix],
                file_index=d['file_index'][ix],seconds=time.monotonic()-tick,
                original_unit_sha256=B.A.sha(original))
            for k in ('updated','bias_deg','turn_group'):
                if k in d:result[k]=d[k][ix]
            B.atomic_npz(dest,**result);status='COMPLETE'
            print('gait unit',unit,round(result['seconds'],2),'seconds shared',round(shared_spent()+result['seconds'],2),flush=True)
        finally:obs.close();stored.close()
    except BaseException as exc:error=repr(exc);raise
    finally:
        attempt=len(list((OUT/'replay_attempts').glob('attempt*.json')))
        B.save(OUT/'replay_attempts'/f'attempt{attempt:04d}_unit{unit}.json',dict(unit=unit,status=status,error=error,seconds=time.monotonic()-tick))


def run(limit):
    d,w=load_bank();freeze();setup=time.monotonic();B.A.OUT=OUT/'replay_runtime';rn=B.H.Runner()
    B.save(OUT/'replay_attempts'/f'setup{int(time.time()*1000)}.json',dict(seconds=time.monotonic()-setup,device=rn.eng.torch.cuda.get_device_name()))
    try:
        for u in B.A.UNITS[:limit] if limit else B.A.UNITS:
            if (OUT/'replay_units'/f'unit{u}.npz').exists():continue
            remaining=1800-shared_spent()
            past=[B.A.read(p)['seconds'] for p in (OUT/'replay_attempts').glob('attempt*.json') if B.A.read(p)['status']=='COMPLETE']
            if remaining<=0 or (past and remaining<max(past[-3:])):
                print('SHARED_BUDGET_STOP remaining',remaining,flush=True);break
            run_unit(rn,d,w,u,remaining)
    finally:rn.eng.torch.cuda.empty_cache()


def analyze():
    tick=time.monotonic();old_result=B.A.read(B.OUT/'replay_result.json')
    inherited=old_result['cpu_seconds']
    if (B.OUT/'replay_audit.json').exists():inherited+=B.A.read(B.OUT/'replay_audit.json')['seconds']
    if (OUT/'replay_result.json').exists():raise FileExistsError('Preserve gait analysis')
    def check():
        if inherited+time.monotonic()-tick>=600:raise TimeoutError('Shared600-second analysis budget reached')
    with np.load(B.OUT/'replay_ledger.npz',allow_pickle=False) as f:z={k:f[k] for k in f.files}
    uid=z['unit'];cfg=z['config'];n=len(uid);newscore=np.empty((2,n,13));err=np.empty((n,16));cache={}
    for j,(u,c) in enumerate(zip(uid,cfg)):
        check();u,c=int(u),int(c)
        if u not in cache:
            cache.clear()
            with np.load(OUT/'replay_units'/f'unit{u}.npz',allow_pickle=False) as f:cache[u]={k:f[k] for k in f.files}
            if str(cache[u]['original_unit_sha256'])!=B.A.sha(B.OUT/'replay_units'/f'unit{u}.npz'):
                raise ValueError('Original unit input changed')
        f=cache[u]
        sm=B.R.smooth(f['raw'][:,c]).max(-1);newscore[0,j]=sm[0];newscore[1,j]=sm[1:].max(0)
        err[j]=f['heading_err'][c]
        for k in ('window','sign','native_frame_index','file_index'):
            np.testing.assert_array_equal(f[k][c],z[k][j])
    ev=z['evaluation'];cal=z['calibration'];evt=ev&z['contact'];ctl=ev&z['control'];cc=cal&z['control']
    rr=np.arange(n);di=np.clip(z['deadline']+3,0,15);age=z['native_frame_index'][rr,di]/60.;turn=z['turn_group'][rr,di]
    groups={'all':evt,'age_under1':evt&(age<1),'age_1to2':evt&(age>=1)&(age<2),'age_2to4':evt&(age>=2)&(age<4),'age_4plus':evt&(age>=4)}
    groups.update({name:evt&(turn==i) for i,name in enumerate(('straight','slowturn','onset','other'))})
    uu,inv=np.unique(uid[ev],return_inverse=True);rng=np.random.default_rng(SEED+1)
    weights=np.array([np.bincount(rng.integers(len(uu),size=len(uu)),minlength=len(uu)) for _ in range(1000)])
    def compare(candidate,base,mask):
        delta=np.where(mask,candidate.astype(int)-base.astype(int),0)[ev]
        totals=np.bincount(inv,weights=delta,minlength=len(uu))
        return dict(events=int(mask.sum()),candidate_timely=int(candidate[mask].sum()),baseline_timely=int(base[mask].sum()),
            gains=int((mask&candidate&~base).sum()),losses=int((mask&~candidate&base).sum()),diff=int(totals.sum()),
            ci95=np.quantile(weights@totals,[.025,.975]).tolist())
    metrics={};thresholds=[];saved={}
    for si,sensor in enumerate(('single','dual')):
        for cap in (.025,.05):
            for selection in ('matched_actual','calibrated'):
                check();policy=f'{selection}/{sensor}/{cap:.3f}'
                e1=next(t for t in old_result['thresholds'] if t['policy']==policy and t['arm']=='e1')
                budget=e1['actual_fa_count'] if selection=='matched_actual' else int(np.floor(cap*int(cc.sum())*10))
                mask=ctl if selection=='matched_actual' else cc
                th=B.integer_budget_threshold(newscore[si,mask,2:12],budget)
                thresholds.append(dict(policy=policy,**th))
                alarm,full,main=B.outcomes(newscore[si],th['threshold'],z['contact'],z['deadline'])
                comp={}
                for name,ai in [('e1',1),('raw_torso',2),('old_corrected',3)]:
                    bf=z[policy+'/timely_all'][ai];bm=z[policy+'/timely_main'][ai]
                    comp[name]=dict(all_before=compare(full,bf,evt),main=compare(main,bm,evt),
                        groups={g:dict(all_before=compare(full,bf,m),main=compare(main,bm,m)) for g,m in groups.items()},
                        baseline_FA=old_result['metrics'][policy][str(z['arms'][ai])]['false_alarm'])
                    saved[policy+'/'+name+'/gain_all']=evt&full&~bf;saved[policy+'/'+name+'/loss_all']=evt&~full&bf
                    saved[policy+'/'+name+'/gain_main']=evt&main&~bm;saved[policy+'/'+name+'/loss_main']=evt&~main&bm
                metrics[policy]=dict(timely_all=int(full[evt].sum()),timely_main=int(main[evt].sum()),events=int(evt.sum()),
                    false_alarm=B.alarm_cost(alarm,ctl),calibration_false_alarm=B.alarm_cost(alarm,cc),comparisons=comp)
                saved[policy+'/alarm']=alarm;saved[policy+'/timely_all']=full;saved[policy+'/timely_main']=main
    check();ledger=OUT/'replay_ledger.npz'
    B.atomic_npz(ledger,unit=uid,config=cfg,score=newscore,heading_err=err,evaluation=ev,calibration=cal,
        contact=z['contact'],control=z['control'],deadline=z['deadline'],window=z['window'],sign=z['sign'],
        native_frame_index=z['native_frame_index'],file_index=z['file_index'],turn_group=z['turn_group'],**saved)
    result=dict(status='COMPLETE',lane='POSTHOC EXPLORE',units=len(np.unique(uid)),evaluation_units=len(uu),sequences=n,
        events=int(evt.sum()),controls=int(ctl.sum()),shared_gpu_seconds=shared_spent(),seconds=time.monotonic()-tick,
        shared_analysis_seconds=inherited+time.monotonic()-tick,metrics=metrics,thresholds=thresholds,
        error_rms=float(np.sqrt((err[ev,3:]**2).mean())),
        provenance=dict(ledger_sha256=B.A.sha(ledger),original_ledger_sha256=B.A.sha(B.OUT/'replay_ledger.npz'),
            original_result_sha256=B.A.sha(B.OUT/'replay_result.json'),series_sha256=B.A.sha(OUT/'series.npz'),
            plan_sha256=B.A.sha(OUT/'replay_plan.json'),source_sha256=B.A.sha(__file__)),
        limits=['Posthoc mechanism selected after first pilot P06..P10 outcome; no fresh confirmation',
            'Same overlapping source windows and participants reused across synthetic units; unitCI omits this dependency',
            'Fixed chosen thresholds; no calibration,training,participant or postselection uncertainty',
            'Matched evaluation integerFA is descriptive; calibrated targets do not equate actual evaluationFA',
            'Real source errors injected onto scripted passive simulation; no BlindWays obstacle or glasses torso measurement',
            'Native future1.5m error truth transferred to frozen current-travel query; turn-context transport is approximate',
            'No new side-query/direction-dependent coverage gate or real three-state validation'])
    check();B.save(OUT/'replay_result.json',result)
    for p,r in metrics.items():print(p,'timely',r['timely_main'],'FA',r['false_alarm']['main'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('run','analyze'));parser.add_argument('--limit',type=int,choices=(1,2))
    args=parser.parse_args()
    run(args.limit) if args.stage=='run' else analyze()


if __name__=='__main__':main()
