"""Replay real HEADS-UP travel-direction errors through the frozen CNH three-state pipeline.

Step 2 of CNH_HEADS_UP_HEADING_20261007 (EXPLORE). Per sequence of batches 98000/99000, a 3.2 s
window of the real error series (sampled every 6 frames of 30 Hz = the sim's 5 Hz) is drawn from
walking runs with a random sign (mirror), and added to the exact head-to-travel query. Everything
else is the frozen pipeline of cnh_heading_uncertainty_dev (observations, noisy history, M3 five
seeds, fused FP32 projection + FP16 M3, smoothing, alarm threshold, r3 gate, events/controls).

Conditions (k = RMS of that error series over all walking samples):
  u1  unconstrained recording, E1 (1 s displacement) vs centred 1 s chord   [frozen primary]
  u2  unconstrained, E1 vs centred 2 s chord                               [lenient truth]
  h1  hard recording (instructed head motion), E1 vs centred 1 s chord
  uE  unconstrained, E2 (sim lag compensation) vs centred 1 s chord
u1 and u2 also get +/-k queries for union-clear. u1/u2/uE share window positions and signs.
Post hoc extra (stage `extra`, after the CPU estimator search in cnh_heads_up_gait_estimator.py):
  uA  unconstrained, causal adaptive EMA of horizontal velocity (tau 0.5 s, 0.25 s when its own
      direction turns >= 10 deg/s; selected on hard) vs centred 1 s chord; same windows as u1.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_active_scan_dev as A  # noqa: E402
import cnh_heads_up_heading as H  # noqa: E402

OUT = H.WORK
SEED = 2026100791
WIN = 96  # 3.2 s at 30 Hz, sampled every 6 frames -> 16 sim frames


def pools():
    out = {}
    for s in ('unconstrained', 'hard'):
        p, fwd = H.load(s); d = H.series(s); t = d['t']
        T2, _ = H.chord_yaw(p, t-30, t+30)
        ser = dict(e1_T1=H.wrap(d['E1']-d['T']), e1_T2=H.wrap(d['E1']-T2), e2_T1=H.wrap(d['E2']-d['T']))
        if s == 'unconstrained':
            import cnh_heads_up_gait_estimator as G
            ser['ema_T1'] = H.wrap(G.ema_adaptive(p, G.jump_mask(p, fwd), .5)[t]-d['T'])
        ok = d['ok']; c = np.r_[0, np.cumsum(ok)]
        starts = np.flatnonzero(c[WIN:]-c[:-WIN] == WIN)  # windows fully inside walking runs
        rms = {k: float(np.sqrt(np.nanmean(v[ok]**2))) for k, v in ser.items()}
        out[s] = dict(starts=starts, rms=rms, **ser)
    return out


def conditions(P):
    k1, k2 = P['unconstrained']['rms']['e1_T1'], P['unconstrained']['rms']['e1_T2']
    return [('exact', None, None, 0.),
            ('u1c', 'unconstrained', 'e1_T1', 0.), ('u1p', 'unconstrained', 'e1_T1', k1), ('u1m', 'unconstrained', 'e1_T1', -k1),
            ('u2c', 'unconstrained', 'e1_T2', 0.), ('u2p', 'unconstrained', 'e1_T2', k2), ('u2m', 'unconstrained', 'e1_T2', -k2),
            ('h1c', 'hard', 'e1_T1', 0.), ('uEc', 'unconstrained', 'e2_T1', 0.)]


def window(P, pool, series, unit, config):
    rng = np.random.default_rng([SEED, unit, config, 0 if pool == 'unconstrained' else 1])
    i = P[pool]['starts'][rng.integers(len(P[pool]['starts']))]; sign = 1. if rng.random() < .5 else -1.
    return sign*P[pool][series][i:i+WIN:6]


EXTRA = [('uAc', 'unconstrained', 'ema_T1', 0.)]


def run_unit(rn, P, unit, deadline, conds=None, folder='units'):
    import cnh_cvr_pilot as CP
    dest = OUT/folder/f'unit{unit}.npz'
    if dest.exists():
        return
    tick = time.monotonic(); obs, sc = A.stored(unit); C = len(obs['noisy_center']); pq = obs['public_query']
    res = dict(unit=unit)
    for name, pool, series, k in (conds or conditions(P)):
        if time.time() > deadline:
            raise TimeoutError('heads-up replay wall budget reached')
        q = np.repeat(pq[None], C, 0).copy()
        if pool:
            err = np.stack([window(P, pool, series, unit, c) for c in range(C)]).astype(np.float32)
            for c in range(C):
                for f in range(16):
                    q[c, f, :3, :3] = CP.rotation(float(err[c, f]+k), 'y')@pq[f, :3, :3]
            if k == 0.:
                res[f'{name}_err'] = err
        raw = np.empty((3, C, 13, 2), np.float32)
        for s, a in enumerate(A.ANGLES):
            ex = rn.eng.N.extrinsic(a)
            raw[s] = rn.raw(obs['z1'][s], obs['noisy_center']@ex, q@ex)
        res[f'{name}_raw'] = raw
    if 'exact_raw' in res:
        res['exact_vs_stored_max_abs'] = float(np.abs(res['exact_raw']-sc['reference']).max())
    res['seconds'] = time.monotonic()-tick
    dest.parent.mkdir(parents=True, exist_ok=True); tmp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **res); os.replace(tmp, dest)
    print('unit', unit, round(res['seconds'], 1), 's exact-vs-stored', round(res.get('exact_vs_stored_max_abs', float('nan')), 4), flush=True)


def freeze(P, hours):
    now = time.time()
    A.save(OUT/'PLAN_STEP2.json', dict(
        task='CNH_HEADS_UP_HEADING_20261007 step 2', lane='EXPLORE; real error series replayed on consumed simulation Development',
        units='98000-98047 + 99000-99095 stored passive S/L/R observations; 369 events, 1255 controls',
        conditions=[c[0] for c in conditions(P)], k_deg=dict(u1=P['unconstrained']['rms']['e1_T1'], u2=P['unconstrained']['rms']['e1_T2']),
        error_rms_deg={s: P[s]['rms'] for s in P}, window_starts={s: int(len(P[s]['starts'])) for s in P},
        deviations_from_PLAN=['easy and hard pose files are one recording (180 deg about x); easy excluded from pools',
                              'turning split uses a 2 s rate window (0.2 s rate is gait-contaminated); replay itself is unaffected',
                              'added u2 (centred 2 s chord truth) as lenient-truth sensitivity after step 1',
                              'secondary mode-matched assignment (turning windows to mode2) not run'],
        frozen='observations, noisy history, M3 5 seeds, smoothing, alarm 0.8557642486787612, r3 gate, 20 tau, events/controls',
        budget_wall_hours=hours, started_unix=now, deadline_unix=now+hours*3600, source_sha256=A.sha(__file__)))


def analyze():
    import cnh_heading_uncertainty_analyze as HA
    E, R = HA.E, HA.R
    rows = R.read(HA.R3/'rows.json')
    with np.load(HA.R3/'online.npz') as z:
        r3gate, thresholds = z['gate'][:, :, 1, :], z['thresholds']
    units = {int(f.stem[4:]): f for f in (OUT/'units').glob('unit*.npz') if not f.name.endswith('.tmp.npz')}
    keep = [i for i, r in enumerate(rows) if r['unit'] in units]; sub = [rows[i] for i in keep]; n = len(sub)
    extra = {int(f.stem[4:]): f for f in (OUT/'units_extra').glob('unit*.npz') if not f.name.endswith('.tmp.npz')}
    has_extra = all(r['unit'] in extra for r in sub)
    names = ['exact', 'u1c', 'u1p', 'u1m', 'u2c', 'u2p', 'u2m', 'h1c', 'uEc']+(['uAc'] if has_extra else [])
    sc = {(a, nm): np.zeros((n, 13)) for a in ('single', 'dual') for nm in names}
    err = {nm: np.zeros((n, 16)) for nm in ('u1c', 'u2c', 'h1c', 'uEc')+(('uAc',) if has_extra else ())}; parity = []; cache = {}
    for j, r in enumerate(sub):
        u, cfg = r['unit'], r['config']
        if u not in cache:
            cache.clear(); cache[u] = dict(np.load(units[u])); parity.append(float(cache[u]['exact_vs_stored_max_abs']))
            if has_extra:
                cache[u] |= dict(np.load(extra[u]))
        z = cache[u]
        for nm in names:
            sc['single', nm][j], sc['dual', nm][j] = HA.arm_scores(z[f'{nm}_raw'][:, cfg])
        for nm in err:
            err[nm][j] = z[f'{nm}_err'][cfg]
    gate = {'single': r3gate[keep, :, 0], 'dual': r3gate[keep, :, 1]}
    g = {k: v[keep] for k, v in E.load_geometry(rows).items()}
    contact, control = g['contact'], g['control']; ix = E.causal_index(g['fraction']); rr = np.flatnonzero(contact)
    modes = np.array([r['mode'] for r in sub]); uid = np.array([r['unit'] for r in sub])
    uu, inv = np.unique(uid, return_inverse=True)
    strat = {r['unit']: f"{r['batch']}/{r['mode']}/{r['turn']}" for r in sub}; srow = np.array([strat[u] for u in uu])
    rng = np.random.default_rng(2026100792); boot = np.zeros((1000, len(uu)), np.int16)
    for s in np.unique(srow):
        ids = np.flatnonzero(srow == s); draw = rng.integers(0, len(ids), size=(1000, len(ids)))
        for k in range(1000):
            boot[k, ids] = np.bincount(draw[k], minlength=len(ids))
    agg = lambda x, m: np.bincount(inv, weights=np.where(m, x, 0.), minlength=len(uu))

    def evaluate(al, cl, gt, tau):
        alarm = al >= R.THRESHOLD; clear = ~alarm & gt & (cl <= tau)
        timely = np.maximum.accumulate(alarm, axis=1)[rr, ix[rr]]
        st = np.full(n, -1, np.int8); st[rr] = np.where(timely, 0, np.where(clear[rr, ix[rr]], 2, 1))
        return st, (~alarm & ~clear)[:, 2:12].sum(1)*.2, alarm[:, 2:12].sum(1)*.2

    variants = {'exact': ('exact', 'exact'), 'u1/none': ('u1c', 'u1c'), 'u1/union-clear': ('u1c', ('u1c', 'u1p', 'u1m')),
                'u2/none': ('u2c', 'u2c'), 'u2/union-clear': ('u2c', ('u2c', 'u2p', 'u2m')),
                'h1/none': ('h1c', 'h1c'), 'uE/none': ('uEc', 'uEc')} | ({'uA/none': ('uAc', 'uAc')} if has_extra else {})
    j19 = len(thresholds)-1; tau = thresholds[j19]
    groups = {'all': np.ones(n, bool)} | {f'mode{k}': modes == k for k in range(3)}
    res = dict(units=len(uu), sequences=n, events=int(contact.sum()), controls=int(control.sum()),
               exact_vs_stored_max_abs=max(parity), tau_index=j19,
               applied_error={nm: dict(rms=float(np.sqrt((e[:, 3:]**2).mean())), p50=float(np.median(np.abs(e[:, 3:]))),
                                       p90=float(np.percentile(np.abs(e[:, 3:]), 90))) for nm, e in err.items()},
               groups={}, diffs={}, frontier={})
    states = {}
    for a in ('single', 'dual'):
        for vn, (al, cl) in variants.items():
            C_ = np.maximum.reduce([sc[a, t] for t in cl]) if isinstance(cl, tuple) else sc[a, cl]
            st, us, as_ = evaluate(sc[a, al], C_, gate[a], tau); states[a, vn] = (st, us, as_)
            for gn, m in groups.items():
                G = res['groups'].setdefault(gn, dict(events=int((m & contact).sum()), controls=int((m & control).sum()), arms={}))
                ctl = m & control
                G['arms'][f'{a}/{vn}'] = dict(timely=int(((st == 0) & m).sum()), unknown_miss=int(((st == 1) & m).sum()),
                                              silent=int(((st == 2) & m).sum()), unknown_time=float(us[ctl].sum()/(2.*ctl.sum())),
                                              obstacle_time=float(as_[ctl].sum()/(2.*ctl.sum())))
        s0, u0, a0 = states[a, 'exact']
        for vn in variants:
            if vn == 'exact':
                continue
            s1, u1, a1 = states[a, vn]; d = {}
            for nm, code in (('timely', 0), ('silent', 2)):
                x = agg((s1 == code).astype(float), contact)-agg((s0 == code).astype(float), contact)
                d[nm] = dict(diff=float(x.sum()), ci95=np.quantile(boot@x, [.025, .975]).tolist())
            for nm, (v1, v0) in (('unknown_time_pp', (u1, u0)), ('obstacle_time_pp', (a1, a0))):
                x = agg(v1-v0, control); dd = agg(np.full(n, 2.), control)
                d[nm] = dict(diff=100*float(x.sum()/dd.sum()), ci95=(100*np.quantile((boot@x)/(boot@dd), [.025, .975])).tolist())
            res['diffs'][f'{a}/{vn}'] = d
        for nm in ('exact', 'u1c', 'u2c', 'h1c', 'uEc')+(('uAc',) if has_extra else ()):
            S = sc[a, nm]; ctl = S[control][:, 2:12]; peak = np.maximum.accumulate(S, axis=1)[rr, ix[rr]]
            th = np.quantile(ctl, np.linspace(.5, .999, 400))
            fa = np.array([(ctl >= t).mean() for t in th]); tim = np.array([(peak >= t).sum() for t in th])
            res['frontier'][f'{a}/{nm}'] = {f'{T:.3f}': (int(tim[fa <= T].max()) if (fa <= T).any() else None) for T in (.025, .05, .10, .15, .20)}
    (OUT/('result_step2_extra.json' if has_extra else 'result_step2.json')).write_text(json.dumps(res, indent=1, ensure_ascii=False)+'\n', encoding='utf8')
    print('units', res['units'], 'events', res['events'], 'controls', res['controls'], 'parity', round(res['exact_vs_stored_max_abs'], 4))
    print('applied error', {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in res['applied_error'].items()})
    for gn, G in res['groups'].items():
        print(f"== {gn} events {G['events']} controls {G['controls']}")
        for arm, p in G['arms'].items():
            print(f"  {arm:22s} T/U/S {p['timely']}/{p['unknown_miss']}/{p['silent']}  unknown {100*p['unknown_time']:.2f}%  obstacle {100*p['obstacle_time']:.2f}%")
    for k, v in res['diffs'].items():
        print(f"  DIFF {k:22s} timely {v['timely']['diff']:+.0f} {np.round(v['timely']['ci95'], 1)} silent {v['silent']['diff']:+.0f} {np.round(v['silent']['ci95'], 1)}"
              f" unknown {v['unknown_time_pp']['diff']:+.2f}pp obstacle {v['obstacle_time_pp']['diff']:+.2f}pp {np.round(v['obstacle_time_pp']['ci95'], 2)}")
    print('frontier: timely at control obstacle-time <= 2.5/5/10/15/20%')
    for k, row in res['frontier'].items():
        print(f'  {k:14s}', list(row.values()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['freeze', 'run', 'analyze', 'extra'])
    ap.add_argument('--hours', type=float, default=.5); a = ap.parse_args()
    if a.stage == 'analyze':
        return analyze()
    P = pools()
    if a.stage == 'freeze':
        return freeze(P, a.hours)
    import cnh_heading_uncertainty_dev as HU
    if a.stage == 'extra':
        now = time.time()
        A.save(OUT/'PLAN_STEP2_EXTRA.json', dict(
            task='CNH_HEADS_UP_HEADING_20261007 step 2 extra (post hoc)', condition='uAc',
            rationale='CPU estimator search (cnh_heads_up_gait_estimator.py; params chosen on hard, unconstrained held out): adaptive EMA lowered held-out all-walking RMS -2.66 deg [-4.12,-1.24] and turning 16.6->9.0, straight mixed across directions; replay checks whether this changes alarms',
            error_rms_deg=P['unconstrained']['rms']['ema_T1'], windows='identical positions and signs to u1',
            budget_wall_hours=.25, started_unix=now, deadline_unix=now+.25*3600, source_sha256=A.sha(__file__)))
        rn = HU.Runner(); deadline = now+.25*3600
        for u in A.UNITS:
            run_unit(rn, P, u, deadline, EXTRA, 'units_extra')
        return analyze()
    rn = HU.Runner(); deadline = A.read(OUT/'PLAN_STEP2.json')['deadline_unix']
    for u in A.UNITS:
        run_unit(rn, P, u, deadline)
    analyze()


if __name__ == '__main__':
    main()
