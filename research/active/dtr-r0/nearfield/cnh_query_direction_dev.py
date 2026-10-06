"""Head-to-travel query-direction error: how much do frozen M3 alarms rely on the exact value?

EXPLORE, consumed simulation Development (batches 98000/99000, stored passive
observations). Only the alarm-query transform changes; observations, noisy
history, M3, smoothing, alarm threshold, r3 gate and event definitions are frozen.
Conditions: exact (stored), est (head yaw from noisy sensor orientation relative
to the past-displacement travel direction, same family as the gate), bias +/-10 deg.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_active_scan_dev as A  # noqa: E402

OUT = A.ROOT/'artifacts.local/work/cnh-query-direction-dev-20261006'
CONDITIONS = ('est', 'bias+10', 'bias-10')


def yaw_of(R):
    """Yaw of a pose's forward (z) axis projected to the horizontal x-z plane, degrees."""
    f = R[:3, 2]
    return np.degrees(np.arctan2(f[0], f[2]))


def est_query(noisy, f):
    """Estimated head-to-travel query: noisy head yaw minus noisy displacement direction."""
    import cnh_cvr_pilot as CP
    a = noisy[max(0, f-5), [0, 2], 3]; b = noisy[f, [0, 2], 3]; d = b-a
    travel_yaw = np.degrees(np.arctan2(d[0], d[1]))
    rel = (yaw_of(noisy[f])-travel_yaw+180) % 360-180
    q = np.eye(4); q[:3, :3] = CP.rotation(rel, 'y')@CP.rotation(-10., 'x')
    return q, rel


def est_comp_query(noisy, f):
    """Lag-compensated travel direction: 1 s displacement direction plus half-window
    extrapolation of its own frame-to-frame change (no head yaw rate used)."""
    import cnh_cvr_pilot as CP
    def dirdeg(a, b):
        d = noisy[b, [0, 2], 3]-noisy[a, [0, 2], 3]; return np.degrees(np.arctan2(d[0], d[1]))
    w = lambda x: (x+180) % 360-180
    t5 = dirdeg(max(0, f-5), f); t5p = dirdeg(max(0, f-6), f-1)
    travel = t5+(min(f, 5)/2)*w(t5-t5p)
    rel = w(yaw_of(noisy[f])-travel)
    q = np.eye(4); q[:3, :3] = CP.rotation(rel, 'y')@CP.rotation(-10., 'x')
    return q, rel


def queries(obs, cond):
    import cnh_cvr_pilot as CP
    pq = obs['public_query']; C = len(obs['noisy_center'])
    q = np.repeat(pq[None], C, 0).copy(); err = np.zeros((C, 16))
    true_rel = np.array([yaw_of(pq[f]) for f in range(16)])
    for c in range(C):
        for f in range(16):
            if cond in ('est', 'est_comp'):
                q[c, f], rel = (est_query if cond == 'est' else est_comp_query)(obs['noisy_center'][c], f)
                err[c, f] = (rel-true_rel[f]+180) % 360-180
            else:
                b = 10. if cond == 'bias+10' else -10.
                q[c, f, :3, :3] = CP.rotation(b, 'y')@pq[f, :3, :3]; err[c, f] = b
    return q, err


def check_deadline():
    deadline = A.read(OUT/'PLAN.json')['deadline_unix']
    if (OUT/'PLAN_AMENDMENT.json').exists():
        deadline = max(deadline, A.read(OUT/'PLAN_AMENDMENT.json')['deadline_unix'])
    if time.time() > deadline:
        raise TimeoutError('query-direction wall budget reached')


def run_unit(eng, unit, conditions=CONDITIONS, sub='units'):
    from cnh_temporal_readout_model import prepare_voxels
    dest = OUT/sub/f'unit{unit}.npz'
    if dest.exists():
        return
    tick = time.monotonic(); obs, sc = A.stored(unit); torch = eng.torch
    C = len(obs['noisy_center']); out = dict(unit=unit, mode=unit % 3)
    for cond in conditions:
        q, err = queries(obs, cond); raw = np.empty((3, C, 13, 2), np.float32)
        for s, a in enumerate(A.ANGLES):
            check_deadline()
            ex = eng.N.extrinsic(a); nn = obs['noisy_center']@ex; qq = q@ex; z = obs['z1'][s]
            vox = []
            for f in range(3, 16):
                ix = np.arange(max(0, f-7), f+1)
                for b in range(0, C, 4):
                    m = (qq[b:b+4, f]@np.linalg.inv(nn[b:b+4, f]))[:, None]@nn[b:b+4][:, ix]
                    vox.append(eng.project(z[b:b+4][:, ix], m).half())
            # vox order: frame-major, config groups; reassemble to [C,13]
            v = torch.stack([torch.cat(vox[k*((C+3)//4):(k+1)*((C+3)//4)]) for k in range(13)], 1).reshape(-1, 3, 24, 17, 33)
            pred = []
            with torch.inference_mode():
                for b in range(0, len(v), 64):
                    x = prepare_voxels(v[b:b+64], eng.masks)
                    pred.append(torch.stack([n(x).float() for n in eng.nets]).mean(0))
            raw[s] = torch.cat(pred).cpu().numpy().reshape(C, 13, 2)
            del vox, v
        out[f'{cond}_raw'] = raw; out[f'{cond}_err'] = err.astype(np.float32)
    out['seconds'] = time.monotonic()-tick
    dest.parent.mkdir(parents=True, exist_ok=True); tmp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **out); os.replace(tmp, dest)
    print('unit', unit, round(out['seconds'], 1), 's', flush=True)


def freeze(hours):
    now = time.time()
    A.save(OUT/'PLAN.json', dict(task='CNH_QUERY_DIRECTION_DEV_20261006', lane='EXPLORE consumed simulation Development',
        question='How much do frozen M3 timely/silent/unknown results depend on the exact head-to-travel alarm-query transform?',
        units='98000-98047 + 99000-99095 stored passive S/L/R observations (369 events, 1255 controls)',
        conditions=dict(exact='stored reference scores', est='noisy sensor yaw minus noisy displacement direction over last<=5 frames (1 s); nominal pitch -10',
                        bias='constant +/-10 deg added to the exact head yaw (worst-case bracket)'),
        frozen='observations, noisy history, M3 5 seeds, FP32 projection (validated in active-scan), smoothing, alarm 0.8557642486787612, r3 gate, 20 tau, events/controls',
        not_changed='coverage gate already uses estimated direction; only alarm query changes',
        budget_wall_hours=hours, started_unix=now, deadline_unix=now+hours*3600, source_sha256=A.sha(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['freeze', 'run', 'check'])
    ap.add_argument('--hours', type=float, default=3.); ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--comp-mode2', action='store_true', help='lag-compensated estimate on mode2 units only')
    a = ap.parse_args()
    if a.stage == 'freeze':
        return freeze(a.hours)
    if a.stage == 'check':
        # estimator on TRUE poses must reproduce the exact query in straight modes
        for unit in (99000, 99001, 99002, 99003):
            obs, _ = A.stored(unit); sc = obs['sensor_center']
            errs = [((est_query(sc, f)[1]-yaw_of(obs['public_query'][f])+180) % 360-180) for f in range(3, 16)]
            print(unit, 'mode', unit % 3, 'true-pose estimator max |err| deg', round(float(np.max(np.abs(errs))), 3))
        return
    eng = A.Engine()
    if a.comp_mode2:
        for u in [u for u in A.UNITS if u % 3 == 2]:
            run_unit(eng, u, ('est_comp',), 'units_comp')
        return
    for u in (A.UNITS[::-1] if a.reverse else A.UNITS):
        run_unit(eng, u)


if __name__ == '__main__':
    main()
