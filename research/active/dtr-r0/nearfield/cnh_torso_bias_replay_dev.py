"""Frozen-M3 error injection replay for the BlindWays causal torso-bias pilot.

This transports heading errors onto consumed synthetic passive observations. It
does not replay BlindWays scenes, a real glasses torso observation, or a validated
query-dependent three-state gate. All arms use the same sampled native window.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_active_scan_dev as A
import cnh_heading_uncertainty_dev as H
import cnh_tristate_dev as R
import cnh_tristate_event_dev as E
from cnh_event_ledger_dev import integer_budget_threshold, json_value

OUT = A.ROOT / 'artifacts.local/work/cnh-torso-bias-dev-20261007'
ARMS = ('exact', 'e1', 'torso', 'corrected', 'oracle')
SEED = 2026100741
GPU_BUDGET = 1800.
CPU_BUDGET = 600.


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(json_value(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def atomic_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Preserve {path}')
    tmp = path.with_suffix('.tmp.npz')
    if tmp.exists():
        raise FileExistsError(f'Preserve incomplete payload {tmp}')
    np.savez_compressed(tmp, **values)
    os.replace(tmp, path)


def bank():
    with np.load(OUT/'series.npz', allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    windows = A.read(OUT/'windows.json')
    if isinstance(windows, dict):
        windows = windows['windows']
    if not windows:
        raise ValueError('No heldout contiguous window')
    for w in windows:
        if w['pid'] not in ('P06', 'P07', 'P08', 'P09', 'P10'):
            raise ValueError('Only heldout participant windows may enter replay')
        start = int(w.get('series_start', w['start'])); ix = start+np.arange(192)
        if ix[-1] >= len(d['valid']) or not d['valid'][ix].all():
            raise ValueError('Window must retain all 192 valid native frames')
        if not np.all(d['file_index'][ix] == int(w['file_index'])):
            raise ValueError('Window crosses source clip')
        if not np.all(np.diff(d['frame_index'][ix]) == 1):
            raise ValueError('Window skips native frames')
        if not all(np.isfinite(d[arm][ix]).all() for arm in ARMS[1:]):
            raise ValueError('Error arm missing in common window')
    return d, windows


def sample(d, windows, unit, config):
    rng = np.random.default_rng([SEED, unit, config])
    wi = int(rng.integers(len(windows))); sign = 1. if rng.random() < .5 else -1.
    w = windows[wi]; start = int(w.get('series_start', w['start']))
    ix = start+12*np.arange(16)
    return wi, sign, ix


def frozen_plan():
    paths = [OUT/'series.npz', OUT/'windows.json', OUT/'PLAN.json', OUT/'result_angles.json',
             HERE/'cnh_torso_bias_dev.py', Path(__file__),
             HERE/'cnh_heading_uncertainty_dev.py', HERE/'cnh_active_scan_dev.py',
             HERE/'cnh_tristate_event_dev.py', HERE/'cnh_event_ledger_dev.py',
             *A.M3_MODELS]
    hashes = {str(p.relative_to(A.ROOT)): A.sha(p) for p in paths}
    plan = dict(task='CNH_TORSO_BIAS_REPLAY_DEV_20261007', lane='EXPLORE',
        units=A.UNITS, arms=ARMS, seed=SEED,
        gpu_budget_seconds=GPU_BUDGET, cpu_analysis_budget_seconds=CPU_BUDGET,
        window='192 contiguous valid 60Hz frames; sample every12 -> 16 at5Hz; P06..P10; common window/sign across arms',
        sign='Native error is estimated travel minus future-pelvis1.5m truth; query-relative injection is negative of signed heading error',
        frozen='Stored observations, noisy poses, M3 five seeds,13 output frames,causal5-frame smoothing,event/control geometry; no train',
        primary='Evaluation E1 threshold smallest whole-tie feasible <=2.5% maincontrol frames[2:12]; candidate whole-tie feasible threshold at actual E1 alarm-count budget; always report residual',
        secondary='Separate calibration98000..98047 thresholds at2.5% and5%; evaluation99000..99095 actual FA;5% descriptive matching',
        startup='Control alarms outputs[:2],main[2:12],all13; timely all-before and excluding startup[2:deadline]',
        trystate='Frozen theta and max tau with inherited query-invariant r3 gate; proxy only, NOT real direction-dependent three-state validation',
        bootstrap='1000 paired evaluation-unit resamples at fixed selected thresholds; no calibration,training,participant or selection uncertainty',
        budget_accounting='Cumulative monotonic seconds for every completed or failed replay attempt, including resumed attempts; setup and analysis separately recorded',
        evidence='Consumed synthetic Development error injection from heldout participants in one existing dataset; not fresh simulation confirmation or hardware',
        source_sha256=hashes)
    path = OUT/'replay_plan.json'
    if path.exists():
        old = A.read(path)
        if old != json_value(plan):
            raise ValueError('Frozen replay inputs or source changed; preserve run and investigate')
    else:
        save(path, plan)
    return plan


def spent_seconds():
    return sum(float(A.read(p)['seconds']) for p in (OUT/'replay_attempts').glob('attempt*.json'))


def run_unit(rn, d, windows, unit, available):
    import cnh_cvr_pilot as CP
    dest = OUT/'replay_units'/f'unit{unit}.npz'
    if dest.exists():
        return
    tick = time.monotonic(); status = 'FAILED'; error = None
    def check():
        if time.monotonic()-tick >= available:
            raise TimeoutError('Cumulative 1800-second GPU replay budget reached')
    try:
        obs, stored = A.stored(unit)
        try:
            C = len(obs['noisy_center']); pq = obs['public_query']
            draws = [sample(d, windows, unit, c) for c in range(C)]
            ix = np.stack([v[2] for v in draws]); sg = np.array([v[1] for v in draws])
            result = dict(unit=unit, mode=unit % 3, window=np.array([v[0] for v in draws]),
                sign=sg, native_series_index=ix, native_frame_index=d['frame_index'][ix],
                file_index=d['file_index'][ix], turn_group=d['turn_group'][ix],
                updated=d['updated'][ix], bias_deg=d['bias_deg'][ix]*sg[:, None],
                passive_gate=A.r3_gate(unit, C))
            for arm in ARMS:
                check()
                heading_error = np.zeros((C, 16), float) if arm == 'exact' else d[arm][ix]*sg[:, None]
                query_error = -heading_error
                q = np.repeat(pq[None], C, 0).copy()
                for c in range(C):
                    for f in range(16):
                        q[c, f, :3, :3] = CP.rotation(float(query_error[c, f]), 'y')@pq[f, :3, :3]
                raw = np.empty((3, C, 13, 2), np.float32)
                for branch, angle in enumerate(A.ANGLES):
                    check(); ex = rn.eng.N.extrinsic(angle)
                    raw[branch] = rn.raw(obs['z1'][branch], obs['noisy_center']@ex, q@ex)
                result[f'{arm}_raw'] = raw
                result[f'{arm}_heading_err'] = heading_error
                result[f'{arm}_query_err'] = query_error
            result['exact_vs_stored_max_abs'] = float(np.abs(result['exact_raw']-stored['reference']).max())
            result['exact_vs_stored_alarm_flips'] = int(((R.smooth(result['exact_raw']).max(-1) >= R.THRESHOLD) !=
                (R.smooth(stored['reference']).max(-1) >= R.THRESHOLD)).sum())
            check(); result['seconds'] = time.monotonic()-tick
            atomic_npz(dest, **result); status = 'COMPLETE'
            print('unit', unit, round(result['seconds'], 2), 'seconds parity', result['exact_vs_stored_max_abs'], flush=True)
        finally:
            obs.close(); stored.close()
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        attempt = len(list((OUT/'replay_attempts').glob('*.json')))
        save(OUT/'replay_attempts'/f'attempt{attempt:04d}_unit{unit}.json',
             dict(unit=unit,status=status,error=error,seconds=time.monotonic()-tick,unix=time.time()))


def run(limit):
    d, windows = bank(); frozen_plan()
    setup = time.monotonic()
    # Engine's established cache routing is canonical artifacts.local; pin this run.
    A.OUT = OUT/'replay_runtime'
    rn = H.Runner()
    save(OUT/'replay_attempts'/f'setup{int(time.time()*1000)}.json',
         dict(status='SETUP', seconds=time.monotonic()-setup, backend='CUDA fusedFP32 + AMPFP16',
              device=rn.eng.torch.cuda.get_device_name(),torch=rn.eng.torch.__version__))
    try:
        for unit in A.UNITS[:limit] if limit else A.UNITS:
            if (OUT/'replay_units'/f'unit{unit}.npz').exists():
                continue
            available = GPU_BUDGET-spent_seconds()
            durations = [A.read(p)['seconds'] for p in (OUT/'replay_attempts').glob('attempt*.json')
                         if A.read(p)['status']=='COMPLETE']
            if available <= 0 or (durations and available < max(durations[-3:])):
                print('BUDGET_STOP remaining', available, flush=True)
                break
            run_unit(rn, d, windows, unit, available)
    finally:
        rn.eng.torch.cuda.empty_cache()
    print('cumulative GPU-attempt seconds',spent_seconds(),flush=True)


def outcomes(score, theta, contact, deadline):
    alarm = score >= theta; n = len(score)
    full = np.zeros(n,bool); main = np.zeros(n,bool)
    rr = np.flatnonzero(contact)
    full[rr] = np.maximum.accumulate(alarm,1)[rr,deadline[rr]]
    primary = alarm.copy(); primary[:,:2] = False
    main[rr] = np.maximum.accumulate(primary,1)[rr,deadline[rr]]
    return alarm, full, main


def alarm_cost(alarm, control):
    n = int(control.sum())
    return {name:dict(count=int(alarm[control,sl].sum()), denominator=n*length,
                rate=None if not n else float(alarm[control,sl].mean()))
            for name,sl,length in [('startup',slice(0,2),2),('main',slice(2,12),10),('all13',slice(None),13)]}


def analyze(limit):
    tick = time.monotonic(); wanted = A.UNITS[:limit] if limit else A.UNITS
    paths = {u:OUT/'replay_units'/f'unit{u}.npz' for u in wanted}
    missing = [u for u,p in paths.items() if not p.exists()]
    if missing:
        raise ValueError(f'Incomplete requested cohort: {missing}; preserve completed units and resume')
    suffix = f'_smoke{limit}' if limit else ''
    result_path = OUT/f'replay_result{suffix}.json'
    ledger_path = OUT/f'replay_ledger{suffix}.npz'
    if result_path.exists() or ledger_path.exists():
        raise FileExistsError('Analysis outputs already exist; preserve prior evidence')
    rows = A.read(A.R3/'rows.json'); keep = [i for i,r in enumerate(rows) if r['unit'] in paths]
    sub = [rows[i] for i in keep]; n = len(sub)
    geo = {k:v[keep] for k,v in E.load_geometry(rows).items()}
    uid = np.array([r['unit'] for r in sub]); cfg = np.array([r['config'] for r in sub])
    contact,control = geo['contact'],geo['control']; deadline = E.causal_index(geo['fraction'])
    if np.any((deadline[contact]<0)|(deadline[contact]>12)):
        raise ValueError('Contact deadline outside frozen outputs')
    scores = np.empty((2,len(ARMS),n,13)); gates = np.empty((2,n,13),bool)
    metas = {k:[] for k in ('window','sign','native_frame_index','file_index','turn_group','updated','bias_deg')}
    errs = {a:[] for a in ARMS}; parity=[]; flips=[]; cache={}
    for j,r in enumerate(sub):
        if time.monotonic()-tick > CPU_BUDGET:
            raise TimeoutError('600-second CPU analysis budget reached')
        u,c = r['unit'],r['config']
        if u not in cache:
            cache.clear()
            with np.load(paths[u],allow_pickle=False) as z:
                cache[u] = {k:z[k] for k in z.files}
            parity.append(float(cache[u]['exact_vs_stored_max_abs'])); flips.append(int(cache[u]['exact_vs_stored_alarm_flips']))
        z=cache[u]
        for ai,a in enumerate(ARMS):
            sm=R.smooth(z[f'{a}_raw'][:,c]).max(-1)
            scores[0,ai,j]=sm[0];scores[1,ai,j]=sm[1:].max(0);errs[a].append(z[f'{a}_heading_err'][c])
        gates[:,j]=z['passive_gate'][c].T
        for k in metas:metas[k].append(z[k][c])
    metas={k:np.asarray(v) for k,v in metas.items()};errs={k:np.asarray(v) for k,v in errs.items()}
    evaluation=uid>=99000; calibration=uid<99000
    # Smoke may contain calibration units only; these exploratory numbers never claim eval status.
    selected=evaluation if evaluation.any() else np.ones(n,bool)
    evaluation_control=control&selected;calibration_control=control&calibration
    eventmask=contact&selected;eval_units,inv=np.unique(uid[selected],return_inverse=True)
    rng=np.random.default_rng(SEED+1);weights=np.zeros((1000,len(eval_units)),np.int16)
    for b in range(1000):
        weights[b]=np.bincount(rng.integers(len(eval_units),size=len(eval_units)),minlength=len(eval_units))
    def paired(candidate,base):
        delta=np.where(eventmask,candidate.astype(int)-base.astype(int),0)[selected]
        totals=np.bincount(inv,weights=delta,minlength=len(eval_units))
        return dict(diff=int(totals.sum()),ci95=np.quantile(weights@totals,[.025,.975]).tolist(),
                    scope='Fixed selected thresholds; paired evaluation units only; calibration and participants not resampled')
    thresholds=[]; metrics={};saved={};tau=float(np.load(A.R3/'online.npz')['thresholds'][-1])
    groups={'all':eventmask}
    deadline_native=np.clip(deadline+3,0,15)
    turn=metas['turn_group'][np.arange(n),deadline_native]
    updated=metas['updated'][np.arange(n),deadline_native]
    groups.update({name:eventmask&(turn==i) for i,name in enumerate(('straight','slowturn','onset','other'))})
    groups.update({'updated_at_deadline':eventmask&updated,'not_updated_at_deadline':eventmask&~updated})
    native_t=metas['native_frame_index'][np.arange(n),deadline_native]/60.
    groups.update({'clip_first5s':eventmask&(native_t<5),'clip_5to15s':eventmask&(native_t>=5)&(native_t<15),'clip_after15s':eventmask&(native_t>=15)})
    for si,sensor in enumerate(('single','dual')):
        for cap in (.025,.05):
            if not evaluation_control.any():
                continue
            values=scores[si,1,evaluation_control,2:12]
            e1_th=integer_budget_threshold(values,int(np.floor(cap*values.size)))
            target=e1_th['actual_fa_count'];policy=f'matched_actual/{sensor}/{cap:.3f}'
            alarms=[];fulls=[];mains=[]
            for ai,a in enumerate(ARMS):
                th=e1_th if a=='e1' else integer_budget_threshold(scores[si,ai,evaluation_control,2:12],target)
                al,fu,ma=outcomes(scores[si,ai],th['threshold'],contact,deadline)
                alarms.append(al);fulls.append(fu);mains.append(ma)
                thresholds.append(dict(policy=policy,arm=a,selection='descriptive evaluation integer budget',**th))
            cell={}
            for ai,a in enumerate(ARMS):
                fu,ma=fulls[ai],mains[ai];base_fu,base_ma=fulls[1],mains[1]
                cell[a]=dict(timely_all=int(fu[eventmask].sum()),timely_excluding_startup=int(ma[eventmask].sum()),
                    events=int(eventmask.sum()),false_alarm=alarm_cost(alarms[ai],evaluation_control),
                    gains_all=int((eventmask&fu&~base_fu).sum()),losses_all=int((eventmask&~fu&base_fu).sum()),
                    gains_main=int((eventmask&ma&~base_ma).sum()),losses_main=int((eventmask&~ma&base_ma).sum()),
                    paired_timely_all=paired(fu,base_fu),paired_timely_main=paired(ma,base_ma),
                    groups={name:dict(events=int(m.sum()),timely_all=int(fu[m].sum()),timely_main=int(ma[m].sum()),
                        gains=int((m&ma&~base_ma).sum()),losses=int((m&~ma&base_ma).sum())) for name,m in groups.items()})
            metrics[policy]=cell;saved[policy+'/alarm']=np.asarray(alarms);saved[policy+'/timely_all']=np.asarray(fulls);saved[policy+'/timely_main']=np.asarray(mains)
            if calibration_control.any() and evaluation.any():
                policy=f'calibrated/{sensor}/{cap:.3f}';cell={};als=[];fus=[];mas=[]
                for ai,a in enumerate(ARMS):
                    v=scores[si,ai,calibration_control,2:12]
                    th=integer_budget_threshold(v,int(np.floor(cap*v.size)))
                    al,fu,ma=outcomes(scores[si,ai],th['threshold'],contact,deadline)
                    thresholds.append(dict(policy=policy,arm=a,selection='separate calibration integer budget',**th))
                    cell[a]=dict(timely_all=int(fu[eventmask].sum()),timely_excluding_startup=int(ma[eventmask].sum()),events=int(eventmask.sum()),
                                 false_alarm=alarm_cost(al,evaluation_control),calibration_false_alarm=alarm_cost(al,calibration_control))
                    als.append(al);fus.append(fu);mas.append(ma)
                for ai,a in enumerate(ARMS):
                    cell[a].update(paired_timely_all=paired(fus[ai],fus[1]),paired_timely_main=paired(mas[ai],mas[1]),
                        gains_main=int((eventmask&mas[ai]&~mas[1]).sum()),losses_main=int((eventmask&~mas[ai]&mas[1]).sum()))
                metrics[policy]=cell;saved[policy+'/alarm']=np.asarray(als);saved[policy+'/timely_all']=np.asarray(fus);saved[policy+'/timely_main']=np.asarray(mas)
        proxy={}
        for ai,a in enumerate(ARMS):
            al,fu,ma=outcomes(scores[si,ai],R.THRESHOLD,contact,deadline)
            clear=~al&gates[si]&(scores[si,ai]<=tau);unknown=~al&~clear
            rr=np.flatnonzero(eventmask);silent=np.zeros(n,bool);silent[rr]=~fu[rr]&clear[rr,deadline[rr]]
            proxy[a]=dict(timely=int(fu[eventmask].sum()),silent=int(silent[eventmask].sum()),
                unknown_miss=int((eventmask&~fu&~silent).sum()),events=int(eventmask.sum()),
                unknown_control_count=int(unknown[evaluation_control,2:12].sum()),control_intervals=int(evaluation_control.sum())*10,
                false_alarm=alarm_cost(al,evaluation_control),gate='Inherited query-invariant proxy; real three-state validity untested')
            saved[f'proxy/{sensor}/{a}/silent']=silent;saved[f'proxy/{sensor}/{a}/unknown']=unknown
        metrics[f'frozen_theta_proxy/{sensor}']=proxy
    payload=dict(unit=uid,config=cfg,scores=scores,gate=gates,deadline=deadline,arms=np.array(ARMS),
                 evaluation=evaluation,calibration=calibration,**geo,**metas,**saved)
    payload.update({f'{a}_heading_err':err for a,err in errs.items()})
    atomic_npz(ledger_path,**payload)
    result=dict(status='SMOKE_COMPLETE' if limit else 'COMPLETE',units=len(paths),sequences=n,
        calibration_units=int(len(np.unique(uid[calibration]))),evaluation_units=int(len(np.unique(uid[evaluation]))),
        events=int(eventmask.sum()),controls=int(evaluation_control.sum()),full_cohort_events=int(contact.sum()),full_cohort_controls=int(control.sum()),
        exact_parity=dict(max_abs_logit=max(parity),branch_query_alarm_flips=sum(flips)),
        gpu_cumulative_seconds=spent_seconds(),cpu_seconds=time.monotonic()-tick,
        thresholds=thresholds,metrics=metrics,
        heading_rms={a:float(np.sqrt((err[selected,3:]**2).mean())) for a,err in errs.items()},
        update_sample_fraction=float(metas['updated'][selected].mean()),
        provenance=dict(series_sha256=A.sha(OUT/'series.npz'),windows_sha256=A.sha(OUT/'windows.json'),
                        source_sha256=A.sha(__file__),ledger_sha256=A.sha(ledger_path)),
        limits=['Consumed synthetic Development error injection; not real BlindWays obstacles or new confirmation',
            'Native errors relative to evaluator-only future-pelvis1.5m direction; oracle is privileged upper-bound diagnostic',
            'Same native windows can repeat across simulated units; unit bootstrap omits source-window and participant uncertainty',
            'Descriptive evaluation FA matching is not deployable calibration; ties can leave residual counts',
            'Fixed threshold bootstrap omits calibration,training and post-selection uncertainty',
            'Frozen r3 gate is query-invariant proxy; no direction-dependent side-query or real three-state validation',
            'Startup/main/all13 costs are separate windows; no assumed session frequency',
            'FiveHz injection transports60Hz errors onto scripted sensor observations; not measured glasses torso input'])
    if time.monotonic()-tick > CPU_BUDGET:
        raise TimeoutError('600-second CPU analysis budget reached; preserve ledger')
    save(result_path,result)
    print(json.dumps({k:result[k] for k in ('status','units','events','controls','heading_rms','exact_parity','cpu_seconds')},indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('run','analyze'))
    parser.add_argument('--limit',type=int,choices=(1,2))
    args=parser.parse_args()
    run(args.limit) if args.stage=='run' else analyze(args.limit)


if __name__=='__main__':
    main()
