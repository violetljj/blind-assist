"""Direction A confirmation on unseen SANPO chest sessions (DIRECTION_A_CONFIRM_SANPO_CHEST_20261117.md): the adopted
chain (ToF -> own ground x known mount height -> last) with UniDepth-V2-S at the phone budget 490x644.
Usage:
  run_confirm.py zones <out json>             intake only: valid ToF zones per frame from the emulator (no model output)
  run_confirm.py run <out dir> [--smoke N]    formal run and frozen judgement
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import adapters
import core
from methods import Methods
from run_a2 import evaluate, judge

NAME = 'sanpo_chest_seg'
PIXELS = 315560                     # 490x644 phone budget (DIRECTION_A3)
ARMS = ('tof_dav2s_out', 'unidepth_s', 'tof_unidepth_out', 'hybrid_out', 'hybrid_out_hm5', 'hybrid_out_hp5')
SCALED = ('tof_unidepth_out', 'hybrid_out', 'hybrid_out_hm5', 'hybrid_out_hp5')
MIN_POS = 30                        # common positive frames per band (LOW, BODY) for a comparison to be evaluable
MIN_BLOCKS = 10                     # sessions with paired corridor AbsRel per band (LOW, BODY)


def zones(out_path):
    M = Methods(); ideal, outdoor = [], []
    for f in adapters.ADAPTERS[NAME]():
        ideal.append(int(np.isfinite(M._zones(f, False)[0]).sum())); outdoor.append(int(np.isfinite(M._zones(f, True)[0]).sum()))
    res = dict(frames=len(ideal), ideal=dict(median=float(np.median(ideal)), below5=int(sum(z < 5 for z in ideal))),
               outdoor=dict(median=float(np.median(outdoor)), below5=int(sum(z < 5 for z in outdoor))))
    Path(out_path).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res))


def evaluable(c):
    gaps = [f'positives_{b}' for b in ('LOW', 'BODY') if c['detect'][b]['positives'] < MIN_POS]
    gaps += [f'blocks_{b}' for b in ('LOW', 'BODY')
             if f'corridor_{b}' not in c['depth'] or c['depth'][f'corridor_{b}']['blocks'] < MIN_BLOCKS]
    return gaps


def claim(c):
    gaps = evaluable(c)
    if gaps:
        return dict(verdict='NOT_EVALUABLE', gaps=gaps)
    j = judge(c)
    j['claim'] = 'CONFIRMED' if j['verdict'] == 'ADOPT_B' else 'NOT_CONFIRMED' if j['verdict'] == 'KEEP_A' else 'UNCERTAIN'
    return j


def run(out_dir, smoke=0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261117); M = Methods(); M.set_unidepth_pixels(PIXELS)
    report = dict(protocol=core.PROTOCOL_VERSION, pixels=PIXELS, arms=ARMS, smoke=smoke)
    report[NAME], rows = evaluate(NAME, M, rng, out, smoke, methods=ARMS, scaled=SCALED)
    print('sources', json.dumps(report[NAME].get('sources')), flush=True)
    if smoke or not rows:
        (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8'); return
    comps = {f'{a}__{b}': core.compare(rows, a, b) for a, b in
             [('tof_unidepth_out', 'hybrid_out'), ('tof_dav2s_out', 'hybrid_out'), ('unidepth_s', 'hybrid_out'),
              ('hybrid_out', 'hybrid_out_hm5'), ('hybrid_out', 'hybrid_out_hp5')]}
    sub = [r for r in rows if r['source']['tof_unidepth_out'] != 'tof']
    comps['tof_insufficient_subset'] = core.compare(sub, 'tof_unidepth_out', 'hybrid_out') if sub else None
    report['comparisons'] = comps
    c1 = claim(comps['tof_insufficient_subset']) if sub else dict(verdict='NOT_EVALUABLE', gaps=['no_tof_insufficient_frames']); c2 = claim(comps['tof_dav2s_out__hybrid_out'])
    report['decision'] = dict(C1_fallback_vs_tof_only=c1, C2_chain_vs_current=c2)
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    for k, v in report['decision'].items():
        print(k, v.get('claim', v['verdict']), json.dumps({x: v[x] for x in ('primary', 'relative_change', 'checks', 'gaps') if x in v}, default=str))
    print('sources', json.dumps(report[NAME].get('sources')), Counter(r['evaluable'] for r in rows))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'zones':
        zones(a[1])
    else:
        run(a[1], int(a[a.index('--smoke') + 1]) if '--smoke' in a else 0)
