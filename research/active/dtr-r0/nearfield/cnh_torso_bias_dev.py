"""Past-frame torso offset calibration; consumed BlindWays Development, not device proof.

No evaluator truth enters causal_correct. Published Xsens positions are not known
to be causal acquisitions. A clip boundary resets bias because session mapping is absent.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import cnh_blindways_heading as B

ROOT = B.ROOT
OUT = ROOT/'artifacts.local/work/cnh-torso-bias-dev-20261007'
HZ = 60
ARMS = ('e1', 'torso', 'corrected', 'oracle')
DEV = tuple(f'P{i:02d}' for i in range(1, 6))
EVAL = tuple(f'P{i:02d}' for i in range(6, 11))
CONFIGS = [dict(tau_seconds=t, straight_rate_deg_s=r, torso_rate_deg_s=12.,
                min_speed=.3, max_bias_rate_deg_s=3.) for t in (2., 4., 8.) for r in (3., 6.)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (OUT/name).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def torso_yaw(x):
    sh = x[:, B.J['lsh'], :2]-x[:, B.J['rsh'], :2]
    return B.yaw(np.c_[sh[:, 1], -sh[:, 0]])


def causal_correct(pelvis, torso, config):
    """Return bias-corrected yaw, bias, update mask; all arrays refer to current frame.

    Compare the fully ended 1s pelvis chord to torso at its midpoint. Gate uses
    two past .5s chords and past 1s torso rotation, never future/evaluator labels.
    """
    p = np.asarray(pelvis, float)[:, :2]; torso = np.asarray(torso, float)
    n = len(p); bias = np.zeros(n); updated = np.zeros(n, bool)
    t = np.arange(HZ, n)
    chord = p[t]-p[t-HZ]
    a, b = B.yaw(p[t]-p[t-HZ//2]), B.yaw(p[t-HZ//2]-p[t-HZ])
    eligible = ((np.linalg.norm(chord, axis=1) >= config['min_speed']) &
                (np.abs(B.wrap(a-b))/.5 <= config['straight_rate_deg_s']) &
                (np.abs(B.wrap(torso[t]-torso[t-HZ])) <= config['torso_rate_deg_s']))
    residual = B.wrap(torso[t-HZ//2]-B.yaw(chord))
    alpha = 1-np.exp(-1/(HZ*config['tau_seconds']))
    cap = config['max_bias_rate_deg_s']/HZ
    state = 0.
    for j, i in enumerate(t):
        if eligible[j]:
            delta = float(np.clip(alpha*B.wrap(residual[j]-state), -cap, cap))
            state = float(B.wrap(state+delta)); updated[i] = True
        bias[i] = state
    return B.wrap(torso-bias), bias, updated


def labels(x):
    """Evaluator-only future pelvis 1.5m; no interpolation across clip boundaries."""
    p = x[:, 0, :2]; n = len(x); ids = np.arange(n)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    future = np.searchsorted(distance, distance+1.5, side='left')
    truth = B.yaw(p[np.minimum(future, n-1)]-p)
    centered = B.yaw(p[np.minimum(ids+30, n-1)]-p[np.maximum(ids-30, 0)])
    speed = np.linalg.norm(p[np.minimum(ids+30, n-1)]-p[np.maximum(ids-30, 0)], axis=1)
    valid = (ids >= HZ) & (ids < n-HZ) & (future < n) & (speed >= .3)
    rate = np.abs(B.wrap(centered[np.minimum(ids+30, n-1)]-centered[np.maximum(ids-30, 0)]))
    groups = np.full(n, 3, np.int8)
    groups[rate < 3] = 0; groups[(rate >= 3) & (rate < 10)] = 1
    high = rate >= 10; onsets = []
    for i in np.flatnonzero(high & ~np.r_[False, high[:-1]]):
        if i >= HZ and not high[i-HZ:i].any() and valid[i]:
            onsets.append(int(i)); groups[i:min(i+HZ, n)] = 2
    return dict(truth=truth, valid=valid, turn_group=groups, speed=speed, onsets=onsets)


def errors(x, config):
    lab = labels(x); torso = torso_yaw(x)
    corrected, bias, updated = causal_correct(x[:, 0], torso, config)
    ids = np.arange(len(x)); e1 = B.yaw(x[:, 6, :2]-x[np.maximum(ids-HZ, 0), 6, :2])
    error = B.wrap(torso-lab['truth']); m = lab['valid']
    # Whole-clip circular fit to evaluator truth: deliberately privileged upper reference.
    oracle_bias = float(np.degrees(np.arctan2(np.sin(np.radians(error[m])).mean(),
                                             np.cos(np.radians(error[m])).mean()))) if m.any() else 0.
    arr = dict(e1=B.wrap(e1-lab['truth']), torso=error,
               corrected=B.wrap(corrected-lab['truth']), oracle=B.wrap(torso-oracle_bias-lab['truth']),
               updated=updated, bias_deg=bias, **lab)
    return arr


def measures(v):
    v = np.asarray(v)
    return None if not len(v) else dict(frames=len(v), seconds=len(v)/HZ,
        rms_deg=float(np.sqrt(np.mean(v*v))), median_abs_deg=float(np.median(np.abs(v))),
        gt15_fraction=float((np.abs(v)>15).mean()))


def persistent(v, mask):
    active = mask & (np.abs(v)>15); s = np.sign(v)
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1] & (s[:-1]==s[1:])])
    ends = np.flatnonzero(active & ~np.r_[active[1:] & (s[:-1]==s[1:]), False])+1
    durations = ends-starts; keep = durations >= HZ
    return dict(runs_ge1s=int(keep.sum()), frames_in_runs_ge1s=int(durations[keep].sum()))


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(B.SRC.glob('*.npy'))
    save('PLAN.json', dict(lane='EXPLORE; previously consumed real BlindWays and simulated Development',
        goal='Test identifiable slow torso offset versus causal E1/raw torso, then frozen M3 task effect',
        budget=dict(estimator_cpu_wall_seconds=600, replay_gpu_wall_seconds=1800, analysis_cpu_wall_seconds=600),
        backend='TASK_NOT_GPU_SUITABLE for scalar estimator; existing CUDA fused M3 for replay',
        split=dict(development=list(DEV), evaluation=list(EVAL),
            note='Pxx-prefix-disjoint for new estimator selection; nominal participant IDs not independently mapped by source documentation. Not fresh population confirmation; dataset previously inspected. No route/session mapping available.'),
        configs=CONFIGS, selection='Lowest participant-equal all-valid RMS on P01..P05; tie order listed. No selection on P06..P10 or alarms.',
        evaluator='Pelvis direction to first horizontal cumulative-path point >=1.5m ahead; walking centered1s speed>=.3m/s; t>=1s, t<clip_end-1s, future present. Same support every arm; invalid counts retained.',
        online='Ended1s pelvis chord vs midpoint shoulder normal; past half-chord turn and torso turn gates; wrapped lowpass residual with max3deg/s change; bias starts0 per clip. No future truth or turn labels enter updates.',
        groups='Evaluator rate of centered1s pelvis direction across1s: straight<3deg/s, slowturn3..10, other>=10; onset first crossing10 after1s below10, next1s. Cold by clip age1..2,2..4,>=4. Future labels only posthoc.',
        oracle='Whole-clip constant circular torso-minus-future-truth fit; privileged reference, not mathematical globally optimal M3 bound.',
        decision_check='No strict win gate; one rescued/lost simulated event is smallest endpoint count. Angle gain alone insufficient. Turn/onset regressions or no alarm gain trigger mechanism diagnosis, not automatic reversal of direction priority.',
        replay='144 inherited sim units; contiguous fully valid192-frame windows P06..P10 only sampled12-frame stride; deterministic paired windows/mirrors all arms. Unsynchronized with simulator turns; heldout participants do not make simulation fresh. Preserve no-update windows.',
        final_comparison='Main E1-matched actual main-window false-alarm count at2.5%; residual ties/startup/full13 output costs reported. Calibration-only threshold and5% secondary; no deployment calibration.',
        source_boundary='Xsens joint model positions; shoulder normal derived from same positions, independence unsupported. Common global yaw errors unidentifiable; only relative torso/path offset studied. Published-frame causality only; upstream processing mode unspecified. Device requires displacement with identifiable bias information.',
        files=[dict(name=p.name,pid=p.stem.split('_')[0],sha256=sha(p)) for p in files],
        source_sha256=sha(__file__), receipt_sha256=sha(B.SRC.parent/'RECEIPT.json')))
    print('PLAN', len(files), 'clips', flush=True)


def run():
    tick = time.monotonic(); plan = json.loads((OUT/'PLAN.json').read_text(encoding='utf8'))
    if (OUT/'result_angles.json').exists(): raise FileExistsError('Preserve existing result')
    assert sha(__file__)==plan['source_sha256']
    files = [B.SRC/r['name'] for r in plan['files']]
    data = []
    for p, record in zip(files, plan['files']):
        assert sha(p)==record['sha256'], p.name
        x = np.load(p)
        if x.shape != (600,24,3) or not np.isfinite(x).all(): raise ValueError(str(p))
        data.append(x)
    def check():
        if time.monotonic()-tick>plan['budget']['estimator_cpu_wall_seconds']: raise TimeoutError('CPU600s reached')
    selection = []
    for config in plan['configs']:
        bypid = {}
        for p, x in zip(files, data):
            if p.stem.split('_')[0] not in DEV: continue
            check(); d = errors(x, config); m = d['valid']
            bypid.setdefault(p.stem.split('_')[0], []).append(d['corrected'][m])
        rms = {pid: measures(np.concatenate(v))['rms_deg'] for pid,v in bypid.items()}
        selection.append(dict(config=config, participant_rms=rms, equal_participant_rms=float(np.mean(list(rms.values())))))
        print('candidate', config, round(selection[-1]['equal_participant_rms'],3), flush=True)
    chosen = min(range(len(selection)), key=lambda i:selection[i]['equal_participant_rms'])
    config = selection[chosen]['config']; packed = {k:[] for k in (*ARMS,'valid','turn_group','updated','bias_deg','file_index','frame_index')}
    per = {}; window_rows = []; onset_rows = []; offset = 0
    for fi, (p,x) in enumerate(zip(files,data)):
        check(); d = errors(x, config); pid = p.stem.split('_')[0]; n=len(x)
        for k in (*ARMS,'valid','turn_group','updated','bias_deg'): packed[k].append(d[k])
        packed['file_index'].append(np.full(n,fi,np.int32)); packed['frame_index'].append(np.arange(n,dtype=np.int16))
        per.setdefault(pid,[]).append(d)
        if pid in EVAL:
            c = np.r_[0,np.cumsum(d['valid'])]
            for start in np.flatnonzero(c[192:]-c[:-192]==192):
                window_rows.append(dict(file_index=fi,start=int(offset+start),pid=pid,clip=p.name))
            for onset in d['onsets']:
                before = slice(max(0,onset-2*HZ),onset); after=slice(onset,min(n,onset+HZ))
                changes=np.diff(np.r_[0.,d['bias_deg']])
                onset_rows.append(dict(pid=pid,clip=p.name,frame=onset,
                    pre2s_update_fraction=float(d['updated'][before].mean()),
                    pre2s_bias_change_deg=float(changes[before].sum()),
                    pre2s_abs_bias_change_deg=float(np.abs(changes[before]).sum()),
                    after={k:measures(d[k][after][d['valid'][after]]) for k in ARMS}))
        offset+=n
    all_arrays={k:np.concatenate(v) for k,v in packed.items()}
    np.savez_compressed(OUT/'series.npz',**all_arrays)
    save('windows.json',window_rows); save('turn_onsets.json',onset_rows)
    summaries={}
    for role,pids in [('development',DEV),('evaluation',EVAL)]:
        role_stats={}; participant={}
        for pid in pids:
            ds=per[pid]; m=np.concatenate([d['valid'] for d in ds]); g=np.concatenate([d['turn_group'] for d in ds])
            up=np.concatenate([d['updated'] for d in ds]); age=np.tile(np.arange(600),len(ds))
            arm={k:np.concatenate([d[k] for d in ds]) for k in ARMS}
            masks=dict(all=m,straight=m&(g==0),slowturn=m&(g==1),onset=m&(g==2),otherturn=m&(g==3),
                cold1to2=m&(age<120),age2to4=m&(age>=120)&(age<240),age4plus=m&(age>=240),never_updated_yet=m&np.concatenate([np.cumsum(d['updated'])==0 for d in ds]))
            participant[pid]=dict(clips=len(ds),valid_frames=int(m.sum()),invalid_frames=int((~m).sum()),
                updated_valid_fraction=float(up[m].mean()),groups={name:{k:measures(v[mask]) for k,v in arm.items()} for name,mask in masks.items()},
                persistent={k:persistent(v,m) for k,v in arm.items()})
        for name in participant[pids[0]]['groups']:
            role_stats[name]={}
            for k in ARMS:
                pts=[participant[pid]['groups'][name][k] for pid in pids]
                pts=[v for v in pts if v is not None]
                role_stats[name][k]=None if not pts else dict(participants=len(pts),
                    frames=sum(v['frames'] for v in pts),seconds=sum(v['seconds'] for v in pts),
                    equal_participant_rms_deg=float(np.mean([v['rms_deg'] for v in pts])),
                    pooled_rms_deg=float(np.sqrt(sum(v['rms_deg']**2*v['frames'] for v in pts)/sum(v['frames'] for v in pts))))
        summaries[role]=dict(participant=participant,groups=role_stats)
    # Difference to past pelvis displacement exposes lag-following without changing updates.
    lag_stats={}
    for lag in (.5,1.,1.5,2.):
        diffs=[]
        for p,x in zip(files,data):
            if p.stem.split('_')[0] not in EVAL:continue
            d=errors(x,config); ids=np.arange(len(x)); back=np.maximum(ids-int(lag*HZ),0)
            estimate=d['corrected']+d['truth']; displacement=B.yaw(x[:,0,:2]-x[back,0,:2])
            diffs.append(B.wrap(estimate-displacement)[d['valid']&(ids>=int(lag*HZ))])
        lag_stats[str(lag)]=measures(np.concatenate(diffs))
    check()
    save('result_angles.json',dict(status='EXPLORATORY_COMPLETE',seconds=time.monotonic()-tick,selection=selection,
        selected_index=chosen,config=config,summaries=summaries,windows=len(window_rows),onsets=len(onset_rows),
        corrected_minus_past_pelvis_chord=lag_stats,plan_sha256=sha(OUT/'PLAN.json'),series_sha256=sha(OUT/'series.npz'),
        windows_sha256=sha(OUT/'windows.json'),limits=plan['source_boundary']))
    print('RESULT',config,'windows',len(window_rows),'onsets',len(onset_rows),summaries['evaluation']['groups'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.stage=='freeze' else run()
