"""Frozen confirmation EMA on stored physical-yaw Development observations.

Whole-scheme comparison: EMA has noisy 5Hz, window-reset inputs; native
torso/gait/E1 have ideal 60Hz full-clip history. No rerender or parameter search.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from pathlib import Path
import time
import numpy as np
import cnh_torso_head_yaw_dev as Y
import cnh_torso_head_yaw_analysis_dev as A
import cnh_torso_bias_replay_dev as B
import cnh_real_head_confirm as RC

OUT = Y.OUT / 'ema_compare_20261008'
BUDGET = dict(prepare=120., run=1200., analyze=180.)
ARMS = (*Y.ARMS, 'ema')


def decision(diffs):
    if len(diffs) != 4: raise ValueError('Four challenges required')
    if all(d >= 0 for d in diffs) and any(d > 0 for d in diffs):
        return 'SUPPORT_COLD_START_SCREEN'
    if all(d <= 0 for d in diffs) and any(d < 0 for d in diffs):
        return 'LOWER_GAIT_PRIORITY'
    if all(d == 0 for d in diffs): return 'ALL_ZERO_RETAIN_NO_AUTOMATIC_FOLLOWUP'
    return 'MIXED_TRADEOFF_NO_AUTOMATIC_FOLLOWUP'


def ema_queries(sensor, noisy, pelvis_origin):
    """Keep confirmation noisy-relative rotation, add common ideal pelvis origin.

    q translation = R_est^-1(head_true - pelvis_true), where R_est yaw is
    noisy yaw - frozen ema_rel. Rotational query is exactly RC.rel_query(rel).
    Thus confirmation yaw noise is retained; the ideal origin is disclosed.
    """
    import cnh_cvr_pilot as CP
    rel = np.stack([RC.ema_rel(n) for n in noisy])
    heading = np.stack([[RC.wrap(RC.A_yaw(n)-r) for n,r in zip(ns,rs)]
                        for ns,rs in zip(noisy,rel)])
    q = np.stack([[RC.rel_query(r) for r in rs] for rs in rel])
    for c in range(len(sensor)):
        for f in range(16):
            q[c,f,:3,3] = CP.rotation(-float(heading[c,f]), 'y') @ (
                sensor[c,f,:3,3] - pelvis_origin[c,f])
    return q, rel, heading


@contextmanager
def stage(name):
    receipts = OUT/'attempts'; receipts.mkdir(parents=True,exist_ok=True)
    spent = sum(B.A.read(p)['seconds'] for p in receipts.glob(name+'*.json'))
    tick=time.monotonic(); status='FAILED'; error=None
    def check():
        if spent+time.monotonic()-tick >= BUDGET[name]:
            raise TimeoutError(f'{name} cumulative wall budget {BUDGET[name]}s')
    try:
        check(); yield check; check(); status='COMPLETE'
    except BaseException as exc:
        error=repr(exc); raise
    finally:
        B.save(receipts/f'{name}{time.time_ns()}.json',dict(stage=name,status=status,
            error=error,seconds=time.monotonic()-tick,previous_seconds=spent,budget=BUDGET[name]))


def prepare():
    with stage('prepare') as check:
        paths=[Path(__file__),Y.OUT/'PLAN.json',Y.OUT/'result.json',Y.OUT/'ledger.npz',
               Y.M.OUT/'bank.npz',Y.M.OUT/'replay_plan.json',
               Y.N.HERE/'cnh_real_head_confirm.py',*B.A.M3_MODELS]
        plan=dict(task='CNH_TORSO_EMA_COMPARE_DEV_20261008',lane='EXPLORE consumed Development',
            authorization='User reviewed roundtable conclusion: input/cost audit then frozen EMA comparison; conditional cold-start only',
            goal='Decide separately per configuration whether gait warrants a cold-start screen against stronger EMA baseline',
            units=Y.UNITS,names=Y.NAMES,arms=ARMS,budgets_phase_wall_seconds=BUDGET,
            inputs=dict(e1='Ideal native60Hz head positions, past60-frame chord, clipped clip-start history; fullclip no window reset',
                torso='Ideal Xsens shoulder/pelvis torso proxy at60Hz, no added estimator noise, fullclip',
                gait='Same ideal torso plus native60Hz ideal pelvis-position causal gate; frozen selected bias state inherited from fullclip',
                ema='Existing condition-specific noisy poses5Hz (16 frames), original RC.ema_rel resets each window, m=None/dirs=NaN/tau=.5; frame0 world heading0 fallback',
                shared='Same cached photons/noisy projection, geometry, deadlines, ideal current pelvis query origin; EMA confirmation relative rotation retains noisy sensor yaw',
                noise='Stored RC.noisy_for/noisy_poses dt.2; scale Uniform[-.2,.2], increment scale N(0,.02), rotational bias signed1deg axes integrated dt plus N(0,.1deg) increments; initial true pose',
                alignment='Whole schemes under existing inputs, not matched-input isolated gait-vs-EMA mechanism'),
            ema_frozen='Call original RC.ema_rel unchanged: tau .5s, next-frame tau .25s when smoothed direction change over.4s >=10deg/s; dt.2s; no search',
            adapter='Rotation exactly RC.rel_query(ema_rel). Add translation R_est^-1*(trueHead-idealPelvis), estYaw=noisyYaw-rel. Confirmation translation was zero; origin adapter disclosed, no fullclip EMA extension.',
            primary='Each condition/sensor current E1 whole-tie <=2.5% all13 strict-control FA; match each arm to actual integer count (currently124/4992), report residual, never promise exact tie matching',
            decision_rules='For each sensor separately, four gait-minus-EMA main timely differences: all>=0 and one>0 supports cold-start screen; all<=0 and one<0 lowers gait priority; mixed or all0 retains record without automatic followup or appendix. No sum, -3 cut, CI, superiority or noninferiority claim.',
            decision_check='229 paired events,384 clear controls/4992 outputs; one event minimum change. EMA may saturate/headroom absent; all-zero not strict-win. Counts are investment rules, not statistical proof.',
            cost='Previous physical-yaw run1445.360s includes rendering and4 arms. This uses5 cached conditions and1 new arm. First-unit inference/I/O measured once and reused; no new render.',
            stop='Cumulative stage caps, changed sources, missing unit or unresolved parity stop affected run; preserve failures/partial, no partial ranking, rerender, tuning or new confirmation batch',
            deliverables='Input/cost record, cached-only EMA query/raw, five-condition per-sensor paired tables/cost residuals, focused validation, docs and master delivery',
            hashes={str(p.relative_to(B.A.ROOT)):B.A.sha(p) for p in paths})
        path=OUT/'PLAN.json'
        if path.exists():
            if B.A.read(path)!=B.json_value(plan): raise ValueError('Frozen EMA comparison plan changed')
        else: B.save(path,plan)
        parent=B.A.read(Y.OUT/'result.json')['provenance']['unit_sha256']
        for u in Y.UNITS:
            check()
            if B.A.sha(Y.OUT/'units'/f'unit{u}.npz')!=parent[str(u)]: raise ValueError('Physical-yaw source changed')
        B.save(OUT/'prepare_result.json',dict(status='COMPLETE',units_verified=96))


def run():
    with stage('run') as check:
        if not (OUT/'prepare_result.json').exists(): raise ValueError('Prepare required')
        B.A.OUT=OUT/'runtime'; rn=None
        try:
            rn=B.H.Runner(); check()
            for u in Y.UNITS:
                check(); dest=OUT/'units'/f'unit{u}.npz'
                if dest.exists(): continue
                tick=time.monotonic(); values=dict(unit=u)
                with np.load(Y.M.OUT/'units'/f'unit{u}.npz') as old, np.load(Y.OUT/'units'/f'unit{u}.npz') as phy:
                    for name in Y.NAMES:
                        f=old if name=='zero' else phy
                        prefix='' if name=='zero' else name+'/'
                        sensor=f[prefix+'sensor']; noisy=f[prefix+'noisy']; z=f[prefix+'z']
                        q,rel,heading=ema_queries(sensor,noisy,old['travel'][...,:3,3])
                        raw=Y.infer(rn,z,noisy,[q],check)[0]
                        values.update({name+'/ema_query':q,name+'/ema_rel':rel,
                                       name+'/ema_heading':heading,name+'/ema_raw':raw})
                    if u==Y.UNITS[0]:
                        # One existing new-E1 result verifies identical backend/input ordering.
                        prefix='const_pos/'
                        reference=Y.infer(rn,phy[prefix+'z'],phy[prefix+'noisy'],[phy[prefix+'e1_query']],check)[0]
                        np.testing.assert_array_equal(reference,phy[prefix+'e1_raw'])
                        B.save(OUT/'first_unit_parity.json',dict(status='PASS',unit=u,max_abs=0.))
                check(); B.atomic_npz(dest,**values)
                elapsed=time.monotonic()-tick
                B.save(OUT/'progress'/f'unit{u}.json',dict(unit=u,seconds=elapsed))
                if u==Y.UNITS[0]:
                    B.save(OUT/'cost_projection.json',dict(first_unit_seconds=elapsed,
                        projected96_seconds=elapsed*96,includes='I/O,5-condition EMA plus1 E1 parity inference,query/save; engine setup separately in run receipt'))
                print('EMA unit',u,'seconds',round(elapsed,3),flush=True)
        finally:
            if rn is not None: rn.eng.torch.cuda.empty_cache()


def analyze():
    with stage('analyze') as check:
        missing=[u for u in Y.UNITS if not (OUT/'units'/f'unit{u}.npz').exists()]
        if missing:
            B.save(OUT/'result.json',dict(status='INCOMPLETE_NO_RANKING',missing_units=missing)); return
        with np.load(Y.OUT/'ledger.npz') as f: old={k:f[k] for k in f.files}
        score=np.empty((5,2,5,3840,13),np.float64); score[:,:,:4]=old['score']
        hashes={}
        for u in Y.UNITS:
            check(); path=OUT/'units'/f'unit{u}.npz'; hashes[str(u)]=B.A.sha(path)
            mask=old['unit']==u
            with np.load(path) as f:
                for ni,name in enumerate(Y.NAMES):
                    sm=B.R.smooth(f[name+'/ema_raw']).max(-1)
                    score[ni,:,4][:,mask]=np.stack((sm[0],sm[1:].max(0)))
        contact=old['contact'];control=old['control'];deadline=old['deadline'];metrics={};thresholds=[];saved={}
        parent=B.A.read(Y.OUT/'result.json')
        for ni,name in enumerate(Y.NAMES):
            for si,sensor in enumerate(A.SENSORS):
                check();key=f'{name}/{sensor}'
                e1=B.integer_budget_threshold(score[ni,si,1,control],int(np.floor(.025*control.sum()*13)))
                target=e1['actual_fa_count'];cell={};mains=[];fulls=[]
                for ai,arm in enumerate(ARMS):
                    th=e1 if arm=='e1' else B.integer_budget_threshold(score[ni,si,ai,control],target)
                    alarm,full,main=B.outcomes(score[ni,si,ai],th['threshold'],contact,deadline)
                    cell[arm]=dict(timely_main=int(main[contact].sum()),timely_all=int(full[contact].sum()),
                        false_alarm=B.alarm_cost(alarm,control),threshold=th)
                    if ai<4:
                        ref=parent['metrics'][key+'/matched_actual'][arm]
                        for k in ('timely_main','timely_all','false_alarm'):
                            if cell[arm][k]!=ref[k]:raise ValueError('Existing arm reproduction failed')
                    mains.append(main);fulls.append(full);thresholds.append(dict(name=name,sensor=sensor,arm=arm,**th))
                for base,bi in (('ema',4),('e1',1),('torso',2)):
                    cell['gait_minus_'+base]=dict(main=Y.N.compare(mains[3],mains[bi],contact),
                        all_before=Y.N.compare(fulls[3],fulls[bi],contact))
                metrics[key]=cell;saved[key+'/timely_main']=np.array(mains);saved[key+'/timely_all']=np.array(fulls)
        decisions={s:dict(diffs=[metrics[n+'/'+s]['gait_minus_ema']['main']['diff'] for n in Y.NAMES[1:]]) for s in A.SENSORS}
        for v in decisions.values(): v['decision']=decision(v['diffs'])
        B.atomic_npz(OUT/'ledger.npz',score=score,arms=np.array(ARMS),names=np.array(Y.NAMES),
            unit=old['unit'],config=old['config'],contact=contact,control=control,deadline=deadline,**saved)
        B.save(OUT/'result.json',dict(status='COMPLETE',events=int(contact.sum()),controls=int(control.sum()),
            control_outputs=int(control.sum())*13,metrics=metrics,thresholds=thresholds,decisions=decisions,
            provenance=dict(unit_sha256=hashes,plan_sha256=B.A.sha(OUT/'PLAN.json'),ledger_sha256=B.A.sha(OUT/'ledger.npz'))))
        print('EMA_COMPLETE',decisions,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','run','analyze'));a=p.parse_args();globals()[a.stage]()
