"""Follow-up of DIRECTION_A_INDOOR_ADT_DEV_20261118.md: ToF scale + offset (hybrid_*_aff) instead of one ToF scale.
Development on seen data: ADT Apartment (indoor, ideal ToF), SANPO chest and Aria walks (outdoor-limited ToF).
Usage: run_aff_dev.py <out dir> [--smoke N]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import core
from methods import Methods
from run_a2 import evaluate, judge
from run_confirm import PIXELS, evaluable

PLAN = {'adt_apartment': (('tof_dav2s', 'hybrid_in', 'hybrid_in_aff'), ('hybrid_in', 'hybrid_in_aff'), ('planar',)),
        'sanpo_chest_seg': (('hybrid_out', 'hybrid_out_aff'), ('hybrid_out', 'hybrid_out_aff'), ()),
        'aria_gen2_walks': (('hybrid_out', 'hybrid_out_aff'), ('hybrid_out', 'hybrid_out_aff'), ())}


def verdict(c):
    gaps = evaluable(c)
    return dict(verdict='NOT_EVALUABLE', gaps=gaps) if gaps else judge(c)


def run(out_dir, smoke=0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    M = Methods(); M.set_unidepth_pixels(PIXELS); report = dict(protocol=core.PROTOCOL_VERSION, pixels=PIXELS, smoke=smoke)
    for name, (arms, scaled, skip) in PLAN.items():
        rng = np.random.default_rng(20261118)
        report[name], rows = evaluate(name, M, rng, out, smoke, methods=arms, scaled=scaled, intake_skip=skip)
        print(name, 'sources', json.dumps(report[name].get('sources')), flush=True)
        if smoke or not rows:
            continue
        old = 'hybrid_in' if name == 'adt_apartment' else 'hybrid_out'
        comps = {f'{old}__{old}_aff': core.compare(rows, old, f'{old}_aff')}
        if name == 'adt_apartment':
            comps['tof_dav2s__hybrid_in_aff'] = core.compare(rows, 'tof_dav2s', 'hybrid_in_aff')
        report[name]['comparisons'] = comps
        report[name]['verdicts'] = {k: verdict(c) for k, c in comps.items()}
        for k, v in report[name]['verdicts'].items():
            print(name, k, v['verdict'], json.dumps({x: v[x] for x in ('primary', 'relative_change', 'checks', 'gaps') if x in v}, default=str), flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    a = sys.argv[1:]
    run(a[0], int(a[a.index('--smoke') + 1]) if '--smoke' in a else 0)
