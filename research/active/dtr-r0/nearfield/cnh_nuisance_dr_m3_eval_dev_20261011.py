"""Compare default vs nuisance-DR M3 on the consumed sim hold under three input conditions.

Conditions reproduce cnh_real_nuisance_injection_dev_20261011 exactly (same seed and draw
order). LOCAL is analytic and input-only, so its scores are reused from that run. For each
backbone and condition, the raised-M3 cut is matched exactly and the LOCAL cut (floored at its
original value) recalibrated offline to the reference clear-slot counts of the original default on clean inputs (negative-only selection).
Writes a real-replay manifest for the nuisance backbone with cuts calibrated on the
'v5 residual' and 'real-like' conditions. Development description only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent); sys.path.insert(0, '.')
import numpy as np
import cnh_no_bin0_dr_eval_20261011 as EV

ROOT = EV.C.ROOT
NEW = ROOT / 'artifacts.local/work/cnh-nuisance-dr-m3-dev-20261011'
NUIS = ROOT / 'artifacts.local/work/real-contract-v5-dev-20261011/nuisance'
PREV = EV.OUT
RA, LT = 14.973200828798356, 4.625390338985158
NAMES = ('sim_original', 'real_like_nuisance', 'v5_residual', 'baseline_only_on_clean')


def conditions():
    W = PREV / 'data/hold'
    z0 = np.asarray(EV.normalized(W)).astype(np.float32)
    n, k = z0.shape[:2]
    rng = np.random.default_rng(20261011)
    t1 = rng.uniform(0, 3, (n, k, 1, 1, 1)); zone = rng.uniform(.5, 1.5, (n, k, 1, 8, 8)); dc = rng.uniform(-.2, .5, (n, k, 1, 1, 1))

    def nuisance(z, resid, baseline):
        z = z.copy()
        tt = t1 if not resid else rng.uniform(-.6, .9, (n, k, 1, 1, 1))
        for b, f in ((1, 1.), (2, .43), (3, .3)):
            z[..., b] += tt * f * zone
        z[..., 1:] += dc[..., None]
        if baseline:
            z = z - np.median(z[..., 3:16], axis=-1, keepdims=True)
        return z.astype(np.float16)
    out = {'sim_original': z0.astype(np.float16)}
    out['real_like_nuisance'] = nuisance(z0, False, False)
    out['v5_residual'] = nuisance(z0, True, True)
    out['baseline_only_on_clean'] = (z0 - np.median(z0[..., 3:16], axis=-1, keepdims=True)).astype(np.float16)
    return out


def smooth(x):
    out = np.empty_like(x, dtype=float)
    for t in range(x.shape[-2]):
        y = x[..., max(0, t - 4):t + 1, :]; w = 2. ** np.arange(y.shape[-2]); out[..., t, :] = (y * w[:, None]).sum(-2) / w.sum()
    return out


def main():
    category = np.load(PREV / 'hold_grades_notifications.npz')['category']
    sel = lambda a, c: a.transpose(0, 3, 1, 2)[category == c]
    m3_paths = sorted((NEW / 'models/M3').glob('model_seed*.pt'))
    if len(m3_paths) != 5:
        raise FileNotFoundError('Need five nuisance M3 seeds')
    _, ordinary = EV.paths('no_bin0')
    original_paths = EV.paths
    EV.paths = lambda arm: (m3_paths, ordinary)
    try:
        with np.load(PREV / 'data/hold/geometry.npz') as g:
            sensor, query = g['sensor'], g['public_query']
        new_scores = {}
        if all((NEW / f'hold_m3_{name}.npy').exists() for name in NAMES):
            new_scores = {name: np.load(NEW / f'hold_m3_{name}.npy') for name in NAMES}
        else:
            for name, z in conditions().items():
                new_scores[name] = EV.infer(z, sensor, query, arms=('no_bin0',))['no_bin0']['m3_raw']
                np.save(NEW / f'hold_m3_{name}.npy', new_scores[name])
    finally:
        EV.paths = original_paths
    ref = np.load(NUIS / 'sim_original.npz')
    nA = int(sel(smooth(ref['m3_raw']) >= RA, 'clear').sum()); nL = int(sel(smooth(ref['local_raw']) >= LT, 'clear').sum())

    def cut(score, count):
        v = np.sort(sel(score, 'clear').ravel())[::-1]
        return float(v[count]) if count < len(v) else float('-inf')
    rows = {}
    for backbone in ('default', 'nuisance_dr'):
        for name in NAMES:
            d = np.load(NUIS / f'{name}.npz')
            m = smooth(d['m3_raw'] if backbone == 'default' else new_scores[name]); l = smooth(d['local_raw'])
            ra, lt = cut(m, nA), max(LT, cut(l, nL))   # raised matched exactly (backbone logit scales differ)
            A, L = m >= ra, l >= lt; S = A | L
            rows[f'{backbone}|{name}'] = dict(raised_cut=ra, local_cut=lt,
                contact_timely=float(sel(S, 'contact')[..., :11].any(-1).mean()),
                contact_timely_m3=float(sel(A, 'contact')[..., :11].any(-1).mean()),
                clear_events=float(sel(S, 'clear')[..., :11].any(-1).mean()),
                pass_events=float(sel(S, 'pass')[..., :11].any(-1).mean()))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    base = json.loads((PREV / 'main_real_manifest.json').read_text(encoding='utf8'))
    for cond in ('v5_residual', 'real_like_nuisance'):
        man = dict(base)
        man['m3_paths'] = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in m3_paths]
        r = rows[f'nuisance_dr|{cond}']
        man['cuts'] = dict(base['cuts'], raised_tau=r['raised_cut'], m3_tau=r['raised_cut'], local_tau=r['local_cut'],
                           rule=f'Offline sim-hold negative recalibration under {cond}; Development only')
        (NEW / f'real_manifest_{cond}.json').write_text(json.dumps(man, indent=1), encoding='utf8')
    (NEW / 'eval_receipt.json').write_text(json.dumps(dict(reference_clear_slots=dict(raised=nA, local=nL), rows=rows,
        m3_sha256={p.name: sha(p) for p in m3_paths}, code_sha256=sha(__file__)), indent=1), encoding='utf8')
    for key, r in rows.items():
        print(f"{key:40s} timely {r['contact_timely']:.3f} (M3 {r['contact_timely_m3']:.3f})  clear {r['clear_events']:.3f}  pass {r['pass_events']:.3f}  cuts {r['raised_cut']:.2f}/{r['local_cut']:.2f}")


if __name__ == '__main__':
    main()
