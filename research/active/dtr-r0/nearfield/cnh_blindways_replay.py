"""Replay BlindWays travel-direction errors through the frozen CNH pipeline (EXPLORE, dev units).

Same mechanics as cnh_heads_up_replay (stored observations of batches 98000/99000, error added to
the exact head-to-travel query, frozen M3/threshold/r3 gate/events). Error series from
cnh_blindways_heading (head-position E1 and EMA vs centred 1 s chord), downsampled 60 -> 30 Hz so a
3.2 s window is 96 frames sampled every 6. Pools: bw (all walking windows of all clips), bwfast
(windows with mean centred speed >= 0.8 m/s).
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_active_scan_dev as A  # noqa: E402
import cnh_blindways_heading as B  # noqa: E402
import cnh_heads_up_replay as Rp  # noqa: E402

OUT = B.OUT


def pools():
    e1, em, ok, sp, pid = [], [], [], [], []
    for f in sorted(glob.glob(str(B.SRC/'*.npy'))):
        d = B.clip(np.load(f)); n = len(d['ok'])
        e1.append(d['e1'][::2]); em.append(d['ema'][::2]); ok.append(np.r_[d['ok'][::2][:-1], False]); sp.append(d['speed'][::2])
        pid += [Path(f).stem.split('_')[0]]*len(d['ok'][::2])
    e1, em, ok, sp = map(np.concatenate, (e1, em, ok, sp)); pid = np.array(pid)
    c = np.r_[0, np.cumsum(ok)]; starts = np.flatnonzero(c[Rp.WIN:]-c[:-Rp.WIN] == Rp.WIN)
    mean_sp = np.array([sp[i:i+Rp.WIN].mean() for i in starts])
    base = dict(e1_T1=e1, ema_T1=em)
    P = dict(bw=dict(starts=starts, **base), bwfast=dict(starts=starts[mean_sp >= .8], **base))
    meta = dict(windows_all=int(len(starts)), windows_fast=int((mean_sp >= .8).sum()),
                participants=sorted(set(pid[starts].tolist())), rms={k: float(np.sqrt(np.mean(v[ok]**2))) for k, v in base.items()})
    return P, meta


CONDS = [('exact', None, None, 0.), ('b1c', 'bw', 'e1_T1', 0.), ('bAc', 'bw', 'ema_T1', 0.), ('f1c', 'bwfast', 'e1_T1', 0.)]


def analyze():
    import cnh_heading_uncertainty_analyze as HA
    E, R = HA.E, HA.R
    rows = R.read(HA.R3/'rows.json')
    with np.load(HA.R3/'online.npz') as z:
        r3gate, thresholds = z['gate'][:, :, 1, :], z['thresholds']
    units = {int(f.stem[4:]): f for f in (OUT/'units').glob('unit*.npz') if not f.name.endswith('.tmp.npz')}
    keep = [i for i, r in enumerate(rows) if r['unit'] in units]; sub = [rows[i] for i in keep]; n = len(sub)
    names = [c[0] for c in CONDS]; sc = {(a, nm): np.zeros((n, 13)) for a in ('single', 'dual') for nm in names}; cache = {}; err = {nm: [] for nm in names[1:]}
    for j, r in enumerate(sub):
        if r['unit'] not in cache:
            cache.clear(); cache[r['unit']] = dict(np.load(units[r['unit']]))
        z = cache[r['unit']]
        for nm in names:
            sc['single', nm][j], sc['dual', nm][j] = HA.arm_scores(z[f'{nm}_raw'][:, r['config']])
        for nm in err:
            err[nm].append(z[f'{nm}_err'][r['config']])
    g = {k: v[keep] for k, v in E.load_geometry(rows).items()}
    contact, control = g['contact'], g['control']; ix = E.causal_index(g['fraction']); rr = np.flatnonzero(contact)
    gate = {'single': r3gate[keep, :, 0], 'dual': r3gate[keep, :, 1]}
    uid = np.array([r['unit'] for r in sub]); uu, inv = np.unique(uid, return_inverse=True)
    strat = {r['unit']: f"{r['batch']}/{r['mode']}/{r['turn']}" for r in sub}; srow = np.array([strat[u] for u in uu])
    rng = np.random.default_rng(2026100797); boot = np.zeros((1000, len(uu)), np.int16)
    for s in np.unique(srow):
        ids = np.flatnonzero(srow == s); dr = rng.integers(0, len(ids), size=(1000, len(ids)))
        for k in range(1000):
            boot[k, ids] = np.bincount(dr[k], minlength=len(ids))
    agg = lambda x: np.bincount(inv, weights=np.where(contact, x, 0.), minlength=len(uu))
    res = dict(events=int(contact.sum()), controls=int(control.sum()),
               applied_error={nm: float(np.sqrt((np.array(e)[:, 3:]**2).mean())) for nm, e in err.items()}, max_tau={}, frontier={}, diffs={})
    tau = thresholds[-1]
    for a in ('single', 'dual'):
        peaks = {}
        for nm in names:
            S = sc[a, nm]; alarm = S >= R.THRESHOLD; clear = ~alarm & gate[a] & (S <= tau)
            timely = np.maximum.accumulate(alarm, axis=1)[rr, ix[rr]]
            st = np.where(timely, 0, np.where(clear[rr, ix[rr]], 2, 1))
            res['max_tau'][f'{a}/{nm}'] = dict(timely=int((st == 0).sum()), unknown_miss=int((st == 1).sum()), silent=int((st == 2).sum()),
                                              unknown_time=float((~alarm & ~clear)[control][:, 2:12].mean()), obstacle_time=float(alarm[control][:, 2:12].mean()))
            ctl = S[control][:, 2:12]; peak = np.maximum.accumulate(S, axis=1)[rr, ix[rr]]
            th = np.quantile(ctl, np.linspace(.5, .999, 400)); fa = np.array([(ctl >= t).mean() for t in th])
            row = {}
            for T in (.025, .05, .10):
                okk = fa <= T; t_best = th[okk][np.argmax([(peak >= t).sum() for t in th[okk]])]
                tm = np.zeros(n, bool); tm[rr] = peak >= t_best; peaks[nm, T] = tm; row[f'{T:.3f}'] = int(tm.sum())
            res['frontier'][f'{a}/{nm}'] = row
        for nm in names[1:]:
            for T in (.025, .05):
                x = agg(peaks[nm, T].astype(float))-agg(peaks['exact', T].astype(float))
                res['diffs'][f'{a}/{nm}-exact/{T}'] = dict(diff=float(x.sum()), ci95=np.quantile(boot@x, [.025, .975]).tolist())
    (OUT/'result_replay.json').write_text(json.dumps(res, indent=1)+'\n', encoding='utf8')
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['run', 'analyze']); a = ap.parse_args()
    if a.stage == 'analyze':
        return analyze()
    P, meta = pools(); OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'replay_pools.json').write_text(json.dumps(meta, indent=1)+'\n', encoding='utf8'); print(meta, flush=True)
    import cnh_heading_uncertainty_dev as HU
    rn = HU.Runner(); deadline = time.time()+.5*3600
    Rp.OUT = OUT
    for u in A.UNITS:
        Rp.run_unit(rn, P, u, deadline, CONDS, 'units')
    analyze()


if __name__ == '__main__':
    main()
