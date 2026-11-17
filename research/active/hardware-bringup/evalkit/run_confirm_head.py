"""Direction A head-worn confirmation on unseen SANPO head sessions with automatic sky
(DIRECTION_A_CONFIRM_SANPO_HEAD_20261117.md). Same arms, budget and rule as run_confirm.py; H1 on all frames (A2's D1).
Usage: run_confirm_head.py <out dir> [--smoke N]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import core
from methods import Methods
from run_a2 import evaluate
from run_confirm import ARMS, PIXELS, SCALED, claim

NAME = 'sanpo_head_noseg'


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
    report['comparisons'] = comps
    report['decision'] = dict(H1_fallback_vs_tof_only=claim(comps['tof_unidepth_out__hybrid_out']),
                              H2_chain_vs_current=claim(comps['tof_dav2s_out__hybrid_out']))
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    for k, v in report['decision'].items():
        print(k, v.get('claim', v['verdict']), json.dumps({x: v[x] for x in ('primary', 'relative_change', 'checks', 'gaps') if x in v}, default=str))


if __name__ == '__main__':
    a = sys.argv[1:]
    run(a[0], int(a[a.index('--smoke') + 1]) if '--smoke' in a else 0)
