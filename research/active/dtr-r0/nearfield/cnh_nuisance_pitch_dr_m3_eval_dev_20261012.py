"""Assumed-pitch robustness on the consumed sim hold: nuisance-DR M3 vs nuisance+pitch-DR M3.

Under the 'real_like_nuisance' condition (cnh_nuisance_dr_m3_eval_dev_20261011.conditions),
projection uses an assumed pose rotated by delta about the sensor x axis
(sensor'=sensor@Rx, query'=query@Rx gives corridor_from_sensor@Rx). M3-only strong chain;
each backbone's raised cut is fixed at its delta=0 value matched to the reference total
strong clear-slot count, then held for all deltas (one deployment threshold).
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
import cnh_nuisance_dr_m3_eval_dev_20261011 as V
from cnh_cvr_pilot import rotation

ROOT = EV.C.ROOT
OUT = ROOT / 'artifacts.local/work/cnh-nuisance-pitch-dr-m3-dev-20261012'
BACKBONES = {'nuisance_dr': ROOT / 'artifacts.local/work/cnh-nuisance-dr-m3-dev-20261011/models/M3', 'nuisance_pitch_dr': OUT / 'models/M3'}
DELTAS = (-10, -5, 0, 5, 10)


def main():
    category = np.load(V.PREV / 'hold_grades_notifications.npz')['category']
    sel = lambda a, c: a.transpose(0, 3, 1, 2)[category == c]
    ref = np.load(V.NUIS / 'sim_original.npz')
    total = int(sel((V.smooth(ref['m3_raw']) >= V.RA) | (V.smooth(ref['local_raw']) >= V.LT), 'clear').sum())
    with np.load(V.PREV / 'data/hold/geometry.npz') as g:
        sensor, query = g['sensor'], g['public_query']
    cond = V.conditions()
    _, ordinary = EV.paths('no_bin0')
    original = EV.paths
    scores = {}
    try:
        for name, folder in BACKBONES.items():
            paths = sorted(folder.glob('model_seed*.pt'))
            if len(paths) != 5:
                raise FileNotFoundError(folder)
            EV.paths = lambda arm, p=paths: (p, ordinary)
            for delta in DELTAS:
                r = np.eye(4); r[:3, :3] = rotation(delta, 'x')
                for c in (('real_like_nuisance', 'sim_original') if delta == 0 else ('real_like_nuisance',)):
                    m = EV.infer(cond[c], sensor @ r, query @ r, arms=('no_bin0',))['no_bin0']['m3_raw']
                    scores[f'{name}|{c}|{delta}'] = m
                    np.save(OUT / f'hold_m3_{name}_{c}_{delta}.npy', m)
    finally:
        EV.paths = original
    rows = {}
    for name in BACKBONES:
        m0 = V.smooth(scores[f'{name}|real_like_nuisance|0'])
        cut = float(np.sort(sel(m0, 'clear').ravel())[::-1][total])
        for key, m in scores.items():
            if not key.startswith(name + '|'):
                continue
            A = V.smooth(m) >= cut
            rows[key] = dict(cut=cut, contact_timely=float(sel(A, 'contact')[..., :11].any(-1).mean()),
                             clear_events=float(sel(A, 'clear')[..., :11].any(-1).mean()),
                             pass_events=float(sel(A, 'pass')[..., :11].any(-1).mean()))
            r = rows[key]
            print(f"{key:44s} timely {r['contact_timely']:.3f} clear {r['clear_events']:.3f} pass {r['pass_events']:.3f} cut {cut:.2f}", flush=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    base = json.loads((ROOT / 'artifacts.local/work/cnh-nuisance-dr-m3-dev-20261011/real_manifest_m3_only.json').read_text(encoding='utf8'))
    paths = sorted(BACKBONES['nuisance_pitch_dr'].glob('model_seed*.pt'))
    base['m3_paths'] = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in paths]
    c = rows['nuisance_pitch_dr|real_like_nuisance|0']['cut']
    base['cuts'] = dict(base['cuts'], raised_tau=c, m3_tau=c, rule='M3-only strong; nuisance+pitch DR; cut matched at delta=0 real_like; Development only')
    (OUT / 'real_manifest_m3_only.json').write_text(json.dumps(base, indent=1), encoding='utf8')
    (OUT / 'eval_receipt.json').write_text(json.dumps(dict(reference_total_clear_slots=total, rows=rows,
        code_sha256=sha(__file__)), indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
