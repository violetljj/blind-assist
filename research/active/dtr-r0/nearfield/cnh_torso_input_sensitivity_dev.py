"""Frozen-M3 torso-input stress replay; stored photons/sensor and labels stay fixed.

Consumed Development, not measured head motion. Fixed positive/negative challenges
are applied to the full native clip before sampling; no parameter selection.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import time
import numpy as np
import cnh_torso_native_motion_dev as M
import cnh_torso_native_replay_dev as N
import cnh_torso_gait_bias_dev as G
import cnh_torso_bias_replay_dev as B

OUT = M.OUT / 'input_sensitivity'
NAMES = ('zero', 'const_pos', 'const_neg', 'pulse_pos', 'pulse_neg')
ARMS = ('torso', 'gait')
UNITS = list(range(99000, 99096))
BUDGET = {'prepare': 120., 'run': 600., 'analyze': 120.}


def perturbation(name, n=600):
    t = np.arange(n) / 60.
    if name == 'zero': return np.zeros(n)
    sign = 1. if name.endswith('pos') else -1.
    if name.startswith('const_'): return np.full(n, sign * 5.)
    if name.startswith('pulse_'):
        return np.where((t >= 2.) & (t <= 4.), sign * 5. * (1. - np.cos(np.pi * (t - 2.))), 0.)
    raise ValueError(name)


def rotate_query(q, signed_delta_deg):
    q = np.asarray(q, float)
    delta = np.broadcast_to(np.asarray(signed_delta_deg, float), q.shape[:-2])
    r = np.broadcast_to(np.eye(4), q.shape).copy()
    rad = np.radians(-delta); c, s = np.cos(rad), np.sin(rad)
    r[..., 0, 0] = c; r[..., 0, 2] = s
    r[..., 2, 0] = -s; r[..., 2, 2] = c
    return r @ q


def estimate_clip(bank, ci, name, config):
    torso = M.T.B.wrap(bank['torso'][ci] + perturbation(name, len(bank['torso'][ci])))
    gait, bias, updated, _ = G.causal_correct(bank['pelvis'][ci], torso, config)
    return dict(torso=torso, gait=gait, bias=bias, updated=updated, seen=np.cumsum(updated) > 0)


@contextmanager
def stage(name):
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = OUT / 'attempts'; receipts.mkdir(exist_ok=True)
    spent = sum(B.A.read(p)['seconds'] for p in receipts.glob(name + '*.json'))
    tick = time.monotonic(); status = 'FAILED'; error = None
    def check():
        if spent + time.monotonic() - tick >= BUDGET[name]:
            raise TimeoutError(f'{name} cumulative phase wall budget {BUDGET[name]}s')
    try:
        check(); yield check; check(); status = 'COMPLETE'
    except BaseException as exc:
        error = repr(exc); raise
    finally:
        B.save(receipts / f'{name}{time.time_ns()}.json', dict(stage=name, status=status,
            error=error, seconds=time.monotonic()-tick, previous_seconds=spent, budget=BUDGET[name]))


def freeze():
    paths = [M.OUT/'bank.npz', M.OUT/'ledger.npz', M.OUT/'replay_result.json',
             M.OUT/'replay_plan.json', M.OUT/'PLAN.json', G.OUT/'result_angles.json',
             N.HERE/'cnh_torso_gait_bias_dev.py', N.HERE/'cnh_torso_native_motion_dev.py',
             N.HERE/'cnh_torso_input_sensitivity_dev.py', *B.A.M3_MODELS]
    B.save(OUT/'PLAN.json', dict(task='CNH_TORSO_INPUT_SENSITIVITY_DEV_20261007',
        authorization='User 推进 after reviewed roundtable next-step recommendation',
        goal='Challenge frozen gait vs raw torso benefit under perturbed torso input, fixed sensor and observations',
        lane='EXPLORE consumed Development', units=UNITS, names=NAMES, arms=ARMS,
        budgets_phase_wall_seconds=BUDGET,
        perturbations='constant +/-5deg; pulse +/-5*(1-cos(pi*(native_age_seconds-2))) deg on 2..4s, zero outside, peak +/-10deg; artificial stress hypotheses',
        state='Full 60Hz native clip from frame0, original zero-bias initialization; no window reset; only pelvis and perturbed torso enter frozen causal estimator',
        fixed='Old sensor/noisy/z, all 96 eval units/3840 sequences/229 events/384 controls, scene geometry/labels/deadline, M3 five seeds and AMP path; no new rendering',
        policies='Each perturbation matched to unchanged E1 actual all13 FA whole-tie <=2.5%, residual preserved; secondary original zero-input per-arm thresholds; descriptive eval workpoints, not deployment calibration',
        zero_check='Recover original counts from saved scores, rebuild queries, then rerun zero-input raw/gait on first unit through new inference entry; exact match required, investigate differences and stop unresolved',
        decision_check='One event is smallest paired change; zero timely change is not noninferiority. Same sources overlap. Preserve no-update strata; query and score equality plus same threshold implies same alarm; cost changes cannot be counted as angle-only gains. Repeated net loss triggers mechanisms before lowering candidate priority; no automatic direction-priority reversal.',
        stop='Any phase limit, missing input, unresolved zero check or complete queue; retain failures/partial, no incomplete queue ranking, no automatic rerender or extra perturbations',
        adjustments='Implementation repairs only within cumulative phase caps; no thresholds/gait retuning, added units or budget expansion',
        deliverables='Frozen plan, zero-check receipt, perturbed queries/raw logits/state ledger, paired counts/loss traces, focused checks, report and scoped commit/push',
        limits='Same future-conditioned synthetic scenes and ideal pelvis origin; sensor torso-aligned proxy unchanged. Not physical headyaw, coverage/UNKNOWN/tristate/hardware or independent participant confirmation. E1 input unchanged; no true-headyaw robustness inference.',
        hashes={str(p.relative_to(B.A.ROOT)): B.A.sha(p) for p in paths}))


def load_old():
    with np.load(M.OUT/'ledger.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepare():
    with stage('prepare') as check:
        freeze(); bank, _ = M.load_bank(); old = load_old(); ev = old['evaluation']
        source = B.A.read(M.OUT/'replay_result.json')['provenance']['unit_sha256']
        for u in UNITS:
            check()
            if B.A.sha(M.OUT/'units'/f'unit{u}.npz') != source[str(u)]: raise ValueError('Source unit hash changed')
        original = B.A.read(M.OUT/'replay_result.json')
        baseline = {}
        for si, sensor in enumerate(('single', 'dual')):
            cell = original['metrics'][f'matched_actual/{sensor}/0.025']
            for arm, ai in (('torso', 2), ('corrected_gait', 4), ('e1', 1)):
                alarm, full, main = B.outcomes(old['score'][si, ai], next(t['threshold'] for t in original['thresholds'] if t['policy']==f'matched_actual/{sensor}/0.025' and t['arm']==arm), old['contact'], old['deadline'])
                count = int(main[ev & old['contact']].sum())
                if count != cell[arm]['timely_main']: raise ValueError('Saved count reproduction failed')
                baseline[f'{sensor}/{arm}'] = count
        clips = sorted(set(map(int, old['clip_index'][ev])))
        config = B.A.read(G.OUT/'result_angles.json')['config']
        data = {'clip': np.array(clips), 'names': np.array(NAMES)}
        for name in NAMES:
            rows = []
            for ci in clips:
                check(); rows.append(estimate_clip(bank, ci, name, config))
            for key in rows[0]: data[name+'/'+key] = np.stack([r[key] for r in rows])
        np.testing.assert_allclose(data['zero/gait'], bank['corrected_gait'][clips], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(data['zero/updated'], bank['updated_gait'][clips])
        B.atomic_npz(OUT/'estimates.npz', **data)
        B.save(OUT/'prepare_result.json', dict(status='COMPLETE', clips=len(clips), recovered_timely_main=baseline))
        print('PREPARE', baseline, 'clips', len(clips), flush=True)


def unit_queries(f, estimates, bank, name):
    ci = f['clip_index']; frame = f['native_frame_index']; sign = f['sign'][:,None]
    lookup = {int(c):i for i,c in enumerate(estimates['clip'])}; ix = np.array([lookup[int(c)] for c in ci])
    result = {}; states = {}
    for arm, original in (('torso','torso'), ('gait','corrected_gait')):
        yaw = estimates[name+'/'+arm][ix[:,None], frame]
        delta = sign * M.T.B.wrap(yaw - bank[original][ci[:,None], frame])
        result[arm] = rotate_query(f[original+'_query'], delta)
        states[arm+'_heading_error'] = sign * M.T.B.wrap(yaw - bank['exact'][ci[:,None], frame])
    for key in ('updated', 'seen', 'bias'):
        states[key] = estimates[name+'/'+key][ix[:,None], frame]
    return result, states


def infer(rn, f, queries, check):
    # Concatenation preserves the 40*13=520 item boundaries of the 130-item M3 batch.
    count = len(queries); C = len(f['sensor']); result = np.empty((count,3,C,13,2), np.float32)
    for branch, angle in enumerate(B.A.ANGLES):
        check(); ex = rn.eng.N.extrinsic(angle)
        raw = rn.raw(np.concatenate([f['z'][branch]]*count),
            np.concatenate([f['noisy']@ex]*count), np.concatenate([q@ex for q in queries]))
        result[:,branch] = raw.reshape(count,C,13,2)
    return result


def run():
    with stage('run') as check:
        with np.load(OUT/'estimates.npz') as z: estimates = {k:z[k] for k in z.files}
        bank, _ = M.load_bank(); B.A.OUT = OUT/'runtime'; rn = None
        try:
            rn = B.H.Runner(); check()
            with np.load(M.OUT/'units'/f'unit{UNITS[0]}.npz') as f:
                q, _ = unit_queries(f, estimates, bank, 'zero')
                for arm, oldarm in (('torso','torso'),('gait','corrected_gait')):
                    np.testing.assert_allclose(q[arm], f[oldarm+'_query'], atol=1e-12, rtol=0)
                zero = infer(rn, f, list(q.values()), check)
                differences = {arm:float(np.abs(zero[i]-f[oldarm+'_raw']).max()) for i,(arm,oldarm) in enumerate((('torso','torso'),('gait','corrected_gait')))}
                B.save(OUT/'zero_inference_check.json', dict(unit=UNITS[0], raw_max_abs=differences, exact_equal=all(v==0 for v in differences.values())))
                if any(v!=0 for v in differences.values()): raise ValueError('Unresolved zero inference difference: '+str(differences))
            for u in UNITS:
                check(); dest = OUT/'units'/f'unit{u}.npz'
                if dest.exists(): continue
                tick = time.monotonic(); values = {}; qs = []
                with np.load(M.OUT/'units'/f'unit{u}.npz') as f:
                    for name in NAMES[1:]:
                        q, states = unit_queries(f, estimates, bank, name)
                        for arm in ARMS: qs.append(q[arm]); values[name+'/'+arm+'_query'] = q[arm]
                        values.update({name+'/'+k:v for k,v in states.items()})
                    raw = infer(rn, f, qs, check)
                    for i,name in enumerate(NAMES[1:]):
                        for ai,arm in enumerate(ARMS): values[name+'/'+arm+'_raw'] = raw[2*i+ai]
                check(); B.atomic_npz(dest, **values)
                print('sensitivity unit',u,'seconds',round(time.monotonic()-tick,3),flush=True)
        finally:
            if rn is not None: rn.eng.torch.cuda.empty_cache()


def analyze():
    with stage('analyze') as check:
        missing = [u for u in UNITS if not (OUT/'units'/f'unit{u}.npz').exists()]
        if missing:
            B.save(OUT/'result.json', dict(status='INCOMPLETE_NO_RANKING', missing_units=missing)); return
        old = load_old(); ev = old['evaluation']; uid = old['unit'][ev]
        contact = old['contact'][ev]; ctl = old['control'][ev]; deadline = old['deadline'][ev]
        # The original causal weighted smoother produces float64. Preserve it:
        # rounding back to float32 can change >= at a whole-tie threshold.
        score = np.empty((5,2,2,len(uid),13), np.float64)
        score[0] = old['score'][:,[2,4]][:, :, ev]
        states = {}; querydiff = []; records = []; equality_checks = []
        for u in UNITS:
            check(); mask = uid==u
            with np.load(OUT/'units'/f'unit{u}.npz') as f:
                qd = []
                for ni,name in enumerate(NAMES[1:],1):
                    for ai,arm in enumerate(ARMS):
                        sm = B.R.smooth(f[name+'/'+arm+'_raw']).max(-1)
                        score[ni,:,ai,mask,:] = np.stack((sm[0],sm[1:].max(0)),0).transpose(1,0,2)
                    for key in ('updated','seen','bias','torso_heading_error','gait_heading_error'):
                        states.setdefault(name+'/'+key, []).append(f[name+'/'+key])
                    qd.append(np.abs(f[name+'/gait_query']-f[name+'/torso_query']).max((-1,-2)))
                querydiff.append(np.stack(qd))
        states = {k:np.concatenate(v) for k,v in states.items()}; qdiff = np.concatenate(querydiff,1)
        di = np.clip(deadline+3,0,15); frames = old['native_frame_index'][ev]; rows = np.arange(len(uid))
        zero_th = B.A.read(M.OUT/'replay_result.json')['thresholds']; metrics = {}; thresholds = []
        for ni,name in enumerate(NAMES):
            updated = old['seen'][ev] if ni==0 else states[name+'/seen']
            if ni==0:
                diff = old['gait_torso_query_max_abs'][ev]
            else:
                diff = qdiff[ni-1]
            samequery = np.all((diff<=1e-12) | (np.arange(16)[None,:] > di[:,None]),1)
            age = frames[rows,di]/60.
            groups = {'all':contact, 'updated':contact & updated[rows,di], 'not_updated':contact & ~updated[rows,di], 'same_query_through_deadline':contact & samequery,
                      'age_2to4':contact & (age>=2) & (age<4), 'age_4plus':contact & (age>=4)}
            groups.update({g:contact & (old['turn_group'][ev][rows,di]==i) for i,g in enumerate(('straight','slowturn','onset','other'))})
            groups.update({p:contact & (old['source_pid'][ev]==p) for p in sorted(set(old['source_pid'][ev]))})
            for si,sensor in enumerate(('single','dual')):
                e1 = B.integer_budget_threshold(old['score'][si,1,ev][ctl], int(.025*ctl.sum()*13))
                eligible = (np.arange(13)[None,:] <= deadline[:,None]) & contact[:,None] & samequery[:,None]
                sd = np.abs(score[ni,si,1]-score[ni,si,0])
                shared = next(t['threshold'] for t in zero_th if t['policy']==f'matched_actual/{sensor}/0.025' and t['arm']=='torso')
                eq = eligible & (sd==0)
                flips = (score[ni,si,1]>=shared) != (score[ni,si,0]>=shared)
                if np.any(flips & eq): raise AssertionError('Equal scores at shared threshold changed alarm')
                equality_checks.append(dict(name=name,sensor=sensor,same_query_events=int((contact&samequery).sum()),
                    same_query_score_max_abs=float(sd[eligible].max()) if eligible.any() else None,
                    shared_threshold_alarm_flip_outputs=int((flips&eligible).sum()), identical_score_alarm_flips=int((flips&eq).sum())))
                for policy in ('matched_actual','zero_threshold'):
                    cell = {}; mains = []; fulls = []
                    for ai,arm in enumerate(ARMS):
                        oldarm = 'torso' if ai==0 else 'corrected_gait'
                        th = B.integer_budget_threshold(score[ni,si,ai,ctl], e1['actual_fa_count']) if policy=='matched_actual' else next(t for t in zero_th if t['policy']==f'matched_actual/{sensor}/0.025' and t['arm']==oldarm)
                        alarm,full,main = B.outcomes(score[ni,si,ai], th['threshold'], contact, deadline)
                        mains.append(main); fulls.append(full)
                        cell[arm] = dict(timely_main=int(main[contact].sum()), timely_all=int(full[contact].sum()), false_alarm=B.alarm_cost(alarm,ctl))
                        thresholds.append({**th, 'name':name, 'sensor':sensor, 'policy':policy, 'arm':arm})
                    cell['gait_minus_torso'] = {g:dict(main=N.compare(mains[1],mains[0],m), all_before=N.compare(fulls[1],fulls[0],m)) for g,m in groups.items()}
                    cell['gait_minus_e1'] = N.compare(mains[1], B.outcomes(old['score'][si,1,ev], e1['threshold'],contact,deadline)[2],contact)
                    metrics[f'{name}/{sensor}/{policy}'] = cell
                    for j in np.flatnonzero(contact & mains[0] & ~mains[1]):
                        records.append(dict(name=name,sensor=sensor,policy=policy,unit=int(uid[j]),config=int(old['config'][ev][j]),source_pid=str(old['source_pid'][ev][j]),source_clip=str(old['source_clip'][ev][j]),deadline=int(deadline[j]),native_age_seconds=float(frames[j,di[j]]/60),ever_updated=bool(updated[j,di[j]]),torso_score=score[ni,si,0,j].tolist(),gait_score=score[ni,si,1,j].tolist()))
        check(); B.atomic_npz(OUT/'ledger.npz', score=score,unit=uid,config=old['config'][ev],contact=contact,control=ctl,deadline=deadline,query_diff=qdiff,**states)
        B.save(OUT/'result.json',dict(status='COMPLETE',evaluation_units=96,sequences=len(uid),events=int(contact.sum()),controls=int(ctl.sum()),metrics=metrics,thresholds=thresholds,loss_records=records,equality_checks=equality_checks,limits=B.A.read(OUT/'PLAN.json')['limits']))
        print('ANALYSIS COMPLETE', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=('prepare','run','analyze')); args=p.parse_args()
    globals()[args.stage]()
