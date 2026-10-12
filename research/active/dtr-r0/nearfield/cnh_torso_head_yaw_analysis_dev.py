"""Frozen-M3 paired head-yaw rerender analysis, consumed Development only.

Geometry and source trajectories stay fixed; head yaw changes actual simulated
observations and all arm queries. Each condition uses its newly inferred E1.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from pathlib import Path
import time
import numpy as np
import cnh_torso_native_motion_dev as M
import cnh_torso_bias_replay_dev as B
import cnh_torso_native_replay_dev as N

OUT = M.OUT / 'head_yaw'
NAMES = ('zero', 'const_pos', 'const_neg', 'pulse_pos', 'pulse_neg')
ARMS = ('exact', 'e1', 'torso', 'gait')
SENSORS = ('single', 'dual')
UNITS = tuple(range(99000, 99096))


@contextmanager
def stage():
    receipts = OUT / 'attempts'
    receipts.mkdir(parents=True, exist_ok=True)
    plan = B.A.read(OUT / 'PLAN.json')
    budget = float(plan['budgets_phase_wall_seconds']['analyze'])
    spent = sum(float(B.A.read(p)['seconds']) for p in receipts.glob('analyze*.json'))
    tick = time.monotonic(); status = 'FAILED'; error = None
    def check():
        if spent + time.monotonic() - tick >= budget:
            raise TimeoutError(f'Head-yaw cumulative analysis budget {budget}s reached')
    try:
        check(); yield check; check(); status = 'COMPLETE'
    except BaseException as exc:
        error = repr(exc); raise
    finally:
        B.save(receipts / f'analyze{time.time_ns()}.json', dict(stage='analyze', status=status,
            seconds=time.monotonic()-tick, previous_seconds=spent, budget=budget, error=error))


def load_old():
    with np.load(M.OUT / 'ledger.npz', allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def grouped(old, ev, contact, deadline):
    n = len(contact); row = np.arange(n); di = np.clip(deadline+3, 0, 15)
    frame = old['native_frame_index'][ev]; age = frame[row,di] / 60.
    seen = old['seen'][ev][row,di]
    turn = old['turn_group'][ev][row,di]
    groups = dict(all=contact, ever_updated=contact & seen,
        never_updated=contact & ~seen,
        age_1to2=contact & (age>=1) & (age<2),
        age_2to4=contact & (age>=2) & (age<4), age_4plus=contact & (age>=4))
    groups.update({g:contact & (turn==i) for i,g in enumerate(('straight','slowturn','onset','other'))})
    groups.update({str(p):contact & (old['source_pid'][ev]==p) for p in sorted(set(old['source_pid'][ev]))})
    return groups, di, age, seen


def analyze():
    with stage() as check:
        dests = [OUT / p for p in ('result.json','ledger.npz','summary.json')]
        if any(p.exists() for p in dests):
            raise FileExistsError('Preserve existing head-yaw analysis evidence')
        missing = [u for u in UNITS if not (OUT/'units'/f'unit{u}.npz').exists()]
        if missing:
            partial = dict(status='INCOMPLETE_NO_RANKING', missing_units=missing)
            B.save(OUT/'result.json', partial); B.save(OUT/'summary.json', partial)
            return partial
        old = load_old(); ev = old['evaluation']
        uid = old['unit'][ev]; cfg = old['config'][ev]
        contact = old['contact'][ev]; control = old['control'][ev]; deadline = old['deadline'][ev]
        if tuple(np.unique(uid)) != UNITS:
            raise ValueError('Frozen 96-unit evaluation cohort changed')
        n = len(uid)
        if n != 3840 or int(contact.sum()) != 229 or int(control.sum()) != 384:
            raise ValueError('Frozen source/geometry denominator changed')
        score = np.empty((len(NAMES),2,len(ARMS),n,13),np.float64)
        score[0] = old['score'][:,[0,1,2,4]][:,:,ev]
        delta = np.zeros((len(NAMES),n,16),np.float64)
        input_hashes = {}
        for u in UNITS:
            check(); mask = uid==u; path = OUT/'units'/f'unit{u}.npz'
            input_hashes[str(u)] = B.A.sha(path)
            with np.load(path,allow_pickle=False) as f:
                if int(mask.sum()) != 40 or not np.array_equal(cfg[mask],np.arange(40)):
                    raise ValueError('Saved source rows must match all40 configs')
                for ni,name in enumerate(NAMES[1:],1):
                    for ai,arm in enumerate(ARMS):
                        raw = f[name+'/'+arm+'_raw']
                        if raw.shape != (3,40,13,2): raise ValueError('Raw shape changed')
                        sm = B.R.smooth(raw).max(-1)
                        score[ni,:,ai][:,mask] = np.stack((sm[0],sm[1:].max(0)))
                    delta[ni,mask] = f[name+'/delta_deg']
        groups,di,age,seen = grouped(old,ev,contact,deadline)
        original = B.A.read(M.OUT/'replay_result.json')
        old_thresholds = original['thresholds']; metrics = {}; thresholds = []; traces = []
        saved = {}; K = int(np.floor(.025*int(control.sum())*13))
        for ni,name in enumerate(NAMES):
            check()
            for si,sensor in enumerate(SENSORS):
                e1 = B.integer_budget_threshold(score[ni,si,1,control],K)
                target = e1['actual_fa_count']
                for policy in ('matched_actual','zero_threshold'):
                    key = f'{name}/{sensor}/{policy}'
                    cell = {}; als = []; fulls = []; mains = []
                    for ai,arm in enumerate(ARMS):
                        if policy == 'matched_actual':
                            th = e1 if arm=='e1' else B.integer_budget_threshold(score[ni,si,ai,control],target)
                            record = dict(th, selection='Current-condition descriptive eval whole-tie budget')
                        else:
                            oldarm = 'corrected_gait' if arm=='gait' else arm
                            oldth = next(t for t in old_thresholds if t['policy']==f'matched_actual/{sensor}/0.025' and t['arm']==oldarm)
                            # Inherited selection counts refer to zero data. Current
                            # actual costs below are separately recounted, unconstrained.
                            record = dict(threshold=oldth['threshold'], selection='Original zero-condition per-arm threshold',
                                original_zero_selection=oldth, target_fa_count=None, residual_fa_count=None,
                                control_intervals=int(control.sum())*13, minimality_verified=False)
                        alarm,full,main = B.outcomes(score[ni,si,ai],record['threshold'],contact,deadline)
                        als.append(alarm); fulls.append(full); mains.append(main)
                        record.update(name=name,sensor=sensor,policy=policy,arm=arm,
                            evaluated_actual_fa_count=int(alarm[control].sum()))
                        if policy=='zero_threshold': record['actual_fa_count']=record['evaluated_actual_fa_count']
                        thresholds.append(record)
                        cell[arm] = dict(timely_main=int(main[contact].sum()), timely_all=int(full[contact].sum()),
                            events=int(contact.sum()),false_alarm=B.alarm_cost(alarm,control),
                            groups={g:dict(events=int(m.sum()),timely_main=int(main[m].sum()),timely_all=int(full[m].sum())) for g,m in groups.items()})
                        if ni==0:
                            oldarm = 'corrected_gait' if arm=='gait' else arm
                            reference = original['metrics'][f'matched_actual/{sensor}/0.025'][oldarm]
                            for metric in ('timely_main','timely_all','false_alarm'):
                                if cell[arm][metric] != reference[metric]:
                                    raise ValueError(f'Zero reproduction failed {sensor}/{arm}/{metric}')
                    for base,bi in (('torso',2),('e1',1)):
                        comparison = f'gait_minus_{base}'
                        cell[comparison] = {g:dict(main=N.compare(mains[3],mains[bi],m),
                            all_before=N.compare(fulls[3],fulls[bi],m)) for g,m in groups.items()}
                        for j in np.flatnonzero(contact & mains[bi] & ~mains[3]):
                            traces.append(dict(name=name,sensor=sensor,policy=policy,comparison=comparison,
                                unit=int(uid[j]),config=int(cfg[j]),window=int(old['window'][ev][j]),
                                source_pid=str(old['source_pid'][ev][j]),source_clip=str(old['source_clip'][ev][j]),
                                clip_index=int(old['clip_index'][ev][j]),native_frame_index=old['native_frame_index'][ev][j].tolist(),
                                deadline=int(deadline[j]),native_age_seconds=float(age[j]),ever_updated=bool(seen[j]),
                                delta_deg=delta[ni,j].tolist(),bias_deg=old['bias_deg'][ev][j].tolist(),
                                updated=old['updated'][ev][j].tolist(),seen=old['seen'][ev][j].tolist(),
                                thresholds={a:thresholds[-4+i]['threshold'] for i,a in enumerate(ARMS)},
                                scores={a:score[ni,si,i,j].tolist() for i,a in enumerate(ARMS)}))
                    metrics[key] = cell
                    saved[key+'/alarm']=np.asarray(als)
                    saved[key+'/timely_all']=np.asarray(fulls)
                    saved[key+'/timely_main']=np.asarray(mains)
        check()
        B.atomic_npz(OUT/'ledger.npz',score=score,names=np.array(NAMES),arms=np.array(ARMS),
            unit=uid,config=cfg,contact=contact,control=control,deadline=deadline,delta_deg=delta,
            source_pid=old['source_pid'][ev],source_clip=old['source_clip'][ev],
            native_frame_index=old['native_frame_index'][ev],turn_group=old['turn_group'][ev],
            updated=old['updated'][ev],seen=old['seen'][ev],bias_deg=old['bias_deg'][ev],**saved)
        limits = [
            'Artificial head-yaw offsets around torso-aligned proxy sensor, not measured headyaw or real-head confirmation',
            'Yaw changes sensor rotations, photons, noisy poses and every query; native ideal-head-position E1 direction stays fixed, but new E1 scores are inferred for each condition',
            'Same future-conditioned synthetic scene placement, futurepelvis1.5m truth, privileged current pelvis query origin and frozen geometry/deadline; not swept-path collision or deployable PDR',
            'Consumed Development with overlapping source clips/windows; fixed choices from prior pilots, no participant confirmation or uncertainty interval',
            'Matched evaluation thresholds are descriptive actual-FA workpoints; zero-threshold secondary costs may differ and are not same-FA comparisons',
            'Costs use one13-output replaywindow, startup0:2/main2:12/all13; timely main excludes startup, full-before includes it; no session-frequency or deployment calibration claim',
            'Empty earlycold strata remain undefined; no validated UNKNOWN/coveragegate/three-state/sidequery/hardware claim']
        result = dict(status='COMPLETE',evaluation_units=96,sequences=n,events=int(contact.sum()),
            controls=int(control.sum()),integer_all13_budget=K,control_intervals=int(control.sum())*13,
            metrics=metrics,thresholds=thresholds,loss_records=traces,
            group_counts={g:int(m.sum()) for g,m in groups.items()},
            provenance=dict(unit_sha256=input_hashes,source_sha256=B.A.sha(Path(__file__)),
                plan_sha256=B.A.sha(OUT/'PLAN.json'),ledger_sha256=B.A.sha(OUT/'ledger.npz')),
            limits=limits)
        B.save(OUT/'result.json',result)
        summary = dict(status='COMPLETE',events=229,controls=384,all13_outputs=4992,
            per_condition={key:dict(timely_main={a:v[a]['timely_main'] for a in ARMS},
                timely_all={a:v[a]['timely_all'] for a in ARMS},
                all13_fa={a:v[a]['false_alarm']['all13']['count'] for a in ARMS},
                startup_fa={a:v[a]['false_alarm']['startup']['count'] for a in ARMS},
                main_fa={a:v[a]['false_alarm']['main']['count'] for a in ARMS},
                gait_minus_torso=v['gait_minus_torso']['all'],gait_minus_e1=v['gait_minus_e1']['all']) for key,v in metrics.items()},
            primary_actual_fa_residuals=[t for t in thresholds if t['policy']=='matched_actual'],
            group_counts=result['group_counts'],limits=limits)
        check(); B.save(OUT/'summary.json',summary)
        print('HEAD_YAW_ANALYSIS_COMPLETE',flush=True)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('analyze',)); parser.parse_args()
    analyze()
