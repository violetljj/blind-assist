"""Summary of the ADT cane-reach relabelling (EXPLORE, 2026-10-15).

Strata: walking (speed >= 0.30 m/s) and walking without manipulation (no ADT dynamic object centre within 0.8 m of the
camera; added after visual checks showed carried items and the wearer's arms still labelled as cane-missed).
For v6 (2/3 confirmed) alerts on any box and on HEAD only (height-only filtering, NOA-like), report how alerts split into
CANE (cane finds it first), MISS (cane-missed hazard) and NEG, and MISS recall within 1.5 m.
Usage: adt_cane_reach_summary_dev_20261015.py <cane_reach.json> <adt_eval.json> <prep dir> <adt data dir> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import adt_cane_reach_dev_20261015 as M
import frozen_fusion_eval_v2 as V2

MANIP_M = .8


def nearest_dynamic(prep_seq, adt_seq):
    boxes = M.dynamic_boxes(adt_seq)
    with np.load(prep_seq / 'frames.npz') as Z:
        tns = Z['t_ns']
    T = np.load(prep_seq / 'poses.npz')['T_sc']; d = np.full(len(tns), np.inf)
    for ts, R, tr, lo, hi in boxes.values():
        k = np.clip(np.searchsorted(ts, tns), 0, len(ts) - 1) if ts[0] >= 0 else np.zeros(len(tns), int)
        ctr = tr[k] + np.einsum('nij,j->ni', R[k], (lo + hi) / 2)
        d = np.minimum(d, np.linalg.norm(ctr - T[:, :3, 3], axis=1))
    return d


def main(cane, evaluation, prep, adt, out):
    L = json.loads(Path(cane).read_text(encoding='utf8'))['labels']; R = json.loads(Path(evaluation).read_text(encoding='utf8'))
    nd = {s: nearest_dynamic(Path(prep) / s, Path(adt) / s) for s in L}
    res = {}
    for sub in ('walk', 'walk_noManip'):
        c = {}
        for s, v in R.items():
            fr, lab, spd = v['frames'], L[s]['rows'], L[s]['speed']
            al = [np.asarray(V2.confirm([f['arms']['chain'][q][1] for f in fr]), bool) for q in (0, 1)]
            for k, f in enumerate(fr):
                if spd[k] < M.WALK or (sub == 'walk_noManip' and nd[s][k] < MANIP_M):
                    continue
                n = [f['truth'][q]['n'] for q in (0, 1)]
                if max(n) >= 150 and lab[k] and not lab[k].get('held_only'):
                    cat = 'CANE' if lab[k]['support_vox'] >= M.MIN_VOX else 'MISS'; near = lab[k]['d_body'] <= 1.5
                elif max(n) == 0:
                    cat, near = 'NEG', False
                else:
                    continue
                for arm, a in (('any', bool(al[0][k] or al[1][k])), ('head', bool(al[0][k]))):
                    c[f'{arm}|{cat}'] = c.get(f'{arm}|{cat}', 0) + a
                    if near:
                        c[f'{arm}_near|{cat}'] = c.get(f'{arm}_near|{cat}', 0) + a
                c[f'n|{cat}'] = c.get(f'n|{cat}', 0) + 1
                if near:
                    c[f'n_near|{cat}'] = c.get(f'n_near|{cat}', 0) + 1
        res[sub] = c
        print(sub, json.dumps(c, sort_keys=True))
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:6])
