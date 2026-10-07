"""Uncertainty-adaptive alarm/clear query width under real blind-walker heading error (EXPLORE).

Error source: BlindWays, truth = pelvis direction to the point 1.5 m further along its path (future
path inside the alarm horizon). Estimators: head E1 (1 s head displacement, current) and torso yaw
(shoulder line; body-worn reference). Causal uncertainty proxies: 1 s head-displacement speed and
the estimator's own |change| over 0.5 s. Width k per frame = 75th percentile of |error| in the proxy
bin (speed <0.6 / 0.6-0.9 / >=0.9 m/s x turn <10 / >=10 deg/s), fitted on odd participants; error
windows for the replay come only from even participants. Fixed baseline: k = overall 75th pct.
Replay: stored observations of dev batches 98000/99000, frozen M3/threshold/r3 gate/events.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_active_scan_dev as A  # noqa: E402
import cnh_blindways_heading as B  # noqa: E402

OUT = A.ROOT/'artifacts.local/work/cnh-adaptive-query-dev-20261007'
SEED = 2026100798
WIN = 96
SP_EDGES, TURN = (.6, .9), 10.
ESTS = ('head', 'torso')


def series(x):
    """60 Hz clip -> 30 Hz per-frame errors (vs future 1.5 m pelvis path) and causal proxies."""
    h, pv = x[:, 6, :2], x[:, 0, :2]; n = len(x)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pv, axis=0), axis=1))]
    t = np.arange(60, n-60); fwd = np.searchsorted(s, s[t]+1.5, side='left'); v = fwd < n
    sh = x[:, 11, :2]-x[:, 7, :2]; torso_all = np.degrees(np.arctan2(-sh[:, 0], sh[:, 1]))
    head_all = np.full(n, np.nan); head_all[60:] = B.yaw(h[60:]-h[:-60])
    FUT = np.full(len(t), np.nan); FUT[v] = B.yaw(pv[fwd[v]]-pv[t[v]])
    est = dict(head=head_all[t], torso=torso_all[t])
    speed = np.linalg.norm(h[t]-h[t-60], axis=1)  # causal 1 s head displacement
    ok = v & (np.linalg.norm(pv[np.minimum(t+30, n-1)]-pv[t-30], axis=1) >= .3)
    out = dict(ok=ok[::2], speed=speed[::2])
    for k, e in est.items():
        out[f'{k}_err'] = B.wrap(e-FUT)[::2]
        out[f'{k}_turn'] = np.r_[np.full(30, np.nan), np.abs(B.wrap(e[30:]-e[:-30]))/.5][::2]
    return out


def bins(speed, turn):
    return np.digitize(speed, SP_EDGES)*2+(np.nan_to_num(turn, nan=0.) >= TURN)


def build():
    cal, ev = {}, {}
    for f in sorted(glob.glob(str(B.SRC/'*.npy'))):
        pid = int(Path(f).stem[1:3]); d = series(np.load(f)); d['ok'][-1] = False
        tgt = cal if pid % 2 else ev
        for k, v in d.items():
            tgt.setdefault(k, []).append(v)
    cal = {k: np.concatenate(v) for k, v in cal.items()}; ev = {k: np.concatenate(v) for k, v in ev.items()}
    table, fixed = {}, {}
    for e in ESTS:
        ok = cal['ok'] & np.isfinite(cal[f'{e}_err']); b = bins(cal['speed'], cal[f'{e}_turn'])
        table[e] = [float(np.percentile(np.abs(cal[f'{e}_err'][ok & (b == j)]), 75)) for j in range(6)]
        fixed[e] = float(np.percentile(np.abs(cal[f'{e}_err'][ok]), 75))
    ok = ev['ok'] & np.isfinite(ev['head_err']) & np.isfinite(ev['torso_err'])
    c = np.r_[0, np.cumsum(ok)]; starts = np.flatnonzero(c[WIN:]-c[:-WIN] == WIN)
    for e in ESTS:
        ev[f'{e}_k'] = np.array(table[e])[bins(ev['speed'], ev[f'{e}_turn'])]
    meta = dict(table_deg=table, fixed_deg=fixed, bins='speed<0.6/0.6-0.9/>=0.9 x turn<10/>=10 (index = 2*speed_bin+turn)',
                eval_windows=int(len(starts)), eval_rms={e: float(np.sqrt(np.mean(ev[f'{e}_err'][ok]**2))) for e in ESTS},
                eval_mean_k={e: float(ev[f'{e}_k'][ok].mean()) for e in ESTS})
    return ev, starts, fixed, meta


def conditions():
    out = [('exact', None, 0, None)]
    for e in ESTS:
        out += [(f'{e}_c', e, 0, None), (f'{e}_ap', e, 1, 'adapt'), (f'{e}_am', e, -1, 'adapt'),
                (f'{e}_fp', e, 1, 'fixed'), (f'{e}_fm', e, -1, 'fixed')]
    return out


def run_unit(rn, ev, starts, fixed, unit, deadline):
    import cnh_cvr_pilot as CP
    dest = OUT/'units'/f'unit{unit}.npz'
    if dest.exists():
        return
    tick = time.monotonic(); obs, sc = A.stored(unit); C = len(obs['noisy_center']); pq = obs['public_query']
    draws = []
    for c in range(C):
        rng = np.random.default_rng([SEED, unit, c]); i = starts[rng.integers(len(starts))]; sg = 1. if rng.random() < .5 else -1.
        draws.append((i, sg))
    res = dict(unit=unit)
    for name, est, side, kind in conditions():
        if time.time() > deadline:
            raise TimeoutError('adaptive-query wall budget reached')
        q = np.repeat(pq[None], C, 0).copy()
        if est:
            ang = np.zeros((C, 16), np.float32)
            for c, (i, sg) in enumerate(draws):
                e = sg*ev[f'{est}_err'][i:i+WIN:6]
                k = ev[f'{est}_k'][i:i+WIN:6] if kind == 'adapt' else np.full(16, fixed[est] if kind else 0.)
                ang[c] = e+side*k
                for f in range(16):
                    q[c, f, :3, :3] = CP.rotation(float(ang[c, f]), 'y')@pq[f, :3, :3]
            if side == 0:
                res[f'{name}_err'] = ang
            elif side == 1:
                res[f'{name}_k'] = (ang-res[f'{est}_c_err']).astype(np.float32)
        raw = np.empty((3, C, 13, 2), np.float32)
        for s, a in enumerate(A.ANGLES):
            ex = rn.eng.N.extrinsic(a); raw[s] = rn.raw(obs['z1'][s], obs['noisy_center']@ex, q@ex)
        res[f'{name}_raw'] = raw
    res['exact_vs_stored_max_abs'] = float(np.abs(res['exact_raw']-sc['reference']).max()); res['seconds'] = time.monotonic()-tick
    dest.parent.mkdir(parents=True, exist_ok=True); tmp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **res); os.replace(tmp, dest)
    print('unit', unit, round(res['seconds'], 1), 's', flush=True)


def analyze():
    import cnh_heading_uncertainty_analyze as HA
    E, R = HA.E, HA.R
    rows = R.read(HA.R3/'rows.json')
    with np.load(HA.R3/'online.npz') as z:
        r3gate, thresholds = z['gate'][:, :, 1, :], z['thresholds']
    units = {int(f.stem[4:]): f for f in (OUT/'units').glob('unit*.npz') if not f.name.endswith('.tmp.npz')}
    keep = [i for i, r in enumerate(rows) if r['unit'] in units]; sub = [rows[i] for i in keep]; n = len(sub)
    names = [c[0] for c in conditions()]; sc = {}; cache = {}
    for j, r in enumerate(sub):
        if r['unit'] not in cache:
            cache.clear(); cache[r['unit']] = dict(np.load(units[r['unit']]))
        z = cache[r['unit']]
        for nm in names:
            s, d = HA.arm_scores(z[f'{nm}_raw'][:, r['config']])
            sc.setdefault(('single', nm), np.zeros((n, 13)))[j] = s; sc.setdefault(('dual', nm), np.zeros((n, 13)))[j] = d
    g = {k: v[keep] for k, v in E.load_geometry(rows).items()}
    contact, control = g['contact'], g['control']; ix = E.causal_index(g['fraction']); rr = np.flatnonzero(contact)
    gate = {'single': r3gate[keep, :, 0], 'dual': r3gate[keep, :, 1]}; tau = thresholds[-1]
    uid = np.array([r['unit'] for r in sub]); uu, inv = np.unique(uid, return_inverse=True)
    strat = {r['unit']: f"{r['batch']}/{r['mode']}/{r['turn']}" for r in sub}; srow = np.array([strat[u] for u in uu])
    rng = np.random.default_rng(2026100799); boot = np.zeros((1000, len(uu)), np.int16)
    for s in np.unique(srow):
        ids = np.flatnonzero(srow == s); dr = rng.integers(0, len(ids), size=(1000, len(ids)))
        for k in range(1000):
            boot[k, ids] = np.bincount(dr[k], minlength=len(ids))
    agg = lambda x: np.bincount(inv, weights=np.where(contact, x, 0.), minlength=len(uu))
    res = dict(meta=json.loads((OUT/'meta.json').read_text()), events=int(contact.sum()), controls=int(control.sum()),
               frontier={}, three_state={}, diffs={})

    def frontier(S):
        ctl = S[control][:, 2:12]; peak = np.maximum.accumulate(S, axis=1)[rr, ix[rr]]
        th = np.quantile(ctl, np.linspace(.5, .999, 400)); fa = np.array([(ctl >= t).mean() for t in th]); out = {}
        for T in (.025, .05, .10):
            cand = th[fa <= T]; best = cand[np.argmax([(peak >= t).sum() for t in cand])]
            tm = np.zeros(n, bool); tm[rr] = peak >= best; out[T] = tm
        return out

    for a in ('single', 'dual'):
        alarm_variants = {'exact': sc[a, 'exact']}
        for e in ESTS:
            alarm_variants[f'{e}/none'] = sc[a, f'{e}_c']
            alarm_variants[f'{e}/wedge-adapt'] = np.maximum.reduce([sc[a, f'{e}_{t}'] for t in ('c', 'ap', 'am')])
            alarm_variants[f'{e}/wedge-fixed'] = np.maximum.reduce([sc[a, f'{e}_{t}'] for t in ('c', 'fp', 'fm')])
        tms = {k: frontier(S) for k, S in alarm_variants.items()}
        for k, d in tms.items():
            res['frontier'][f'{a}/{k}'] = {f'{T:.3f}': int(v.sum()) for T, v in d.items()}
        for e in ESTS:
            for v in ('none', 'wedge-adapt', 'wedge-fixed'):
                for T in (.025, .05):
                    x = agg(tms[f'{e}/{v}'][T].astype(float))-agg(tms['exact'][T].astype(float))
                    res['diffs'][f'{a}/{e}/{v}-exact/{T}'] = dict(diff=float(x.sum()), ci95=np.quantile(boot@x, [.025, .975]).tolist())
            for v in ('wedge-adapt', 'wedge-fixed'):
                for T in (.025, .05):
                    x = agg(tms[f'{e}/{v}'][T].astype(float))-agg(tms[f'{e}/none'][T].astype(float))
                    res['diffs'][f'{a}/{e}/{v}-none/{T}'] = dict(diff=float(x.sum()), ci95=np.quantile(boot@x, [.025, .975]).tolist())
            # three-state at frozen theta: clear needs centre (none) or all three (union-clear) <= tau
            for v, Cl in (('none', sc[a, f'{e}_c']), ('clear-adapt', alarm_variants[f'{e}/wedge-adapt']), ('clear-fixed', alarm_variants[f'{e}/wedge-fixed'])):
                S = sc[a, f'{e}_c']; alarm = S >= R.THRESHOLD; clear = ~alarm & gate[a] & (Cl <= tau)
                timely = np.maximum.accumulate(alarm, axis=1)[rr, ix[rr]]; st = np.where(timely, 0, np.where(clear[rr, ix[rr]], 2, 1))
                res['three_state'][f'{a}/{e}/{v}'] = dict(timely=int((st == 0).sum()), unknown_miss=int((st == 1).sum()), silent=int((st == 2).sum()),
                                                          unknown_time=float((~alarm & ~clear)[control][:, 2:12].mean()), obstacle_time=float(alarm[control][:, 2:12].mean()))
        S = sc[a, 'exact']; alarm = S >= R.THRESHOLD; clear = ~alarm & gate[a] & (S <= tau)
        timely = np.maximum.accumulate(alarm, axis=1)[rr, ix[rr]]; st = np.where(timely, 0, np.where(clear[rr, ix[rr]], 2, 1))
        res['three_state'][f'{a}/exact'] = dict(timely=int((st == 0).sum()), unknown_miss=int((st == 1).sum()), silent=int((st == 2).sum()),
                                                unknown_time=float((~alarm & ~clear)[control][:, 2:12].mean()), obstacle_time=float(alarm[control][:, 2:12].mean()))
    (OUT/'result.json').write_text(json.dumps(res, indent=1)+'\n', encoding='utf8')
    print('events', res['events'], 'controls', res['controls'], json.dumps(res['meta']))
    for k, v in res['frontier'].items():
        print('F', k, list(v.values()))
    for k, v in res['three_state'].items():
        print('3S', k, v['timely'], v['unknown_miss'], v['silent'], round(100*v['unknown_time'], 2), round(100*v['obstacle_time'], 2))
    for k, v in res['diffs'].items():
        print('D', k, v['diff'], np.round(v['ci95'], 1).tolist())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['run', 'analyze']); a = ap.parse_args()
    if a.stage == 'analyze':
        return analyze()
    ev, starts, fixed, meta = build(); OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'meta.json').write_text(json.dumps(meta, indent=1)+'\n', encoding='utf8'); print(meta, flush=True)
    import cnh_heading_uncertainty_dev as HU
    rn = HU.Runner(); deadline = time.time()+1.*3600
    for u in A.UNITS:
        run_unit(rn, ev, starts, fixed, u, deadline)
    analyze()


if __name__ == '__main__':
    main()
