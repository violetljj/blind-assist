"""Direction A indoors on ADT Apartment (DIRECTION_A_INDOOR_ADT_DEV_20261118.md): the chain with the ideal (indoor) ToF
emulation, UniDepth-V2-S at the 490x644 phone budget. Development: these sequences fed earlier ADT diagnostics.
Usage: run_adt_indoor.py <out dir> [--smoke N]
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
from run_confirm import PIXELS, claim

NAME = 'adt_apartment'
ARMS = ('tof_dav2s', 'unidepth_s', 'tof_unidepth', 'floor_unidepth', 'hybrid_in', 'hybrid_in_hm5', 'hybrid_in_hp5')
SCALED = ('tof_unidepth', 'floor_unidepth', 'hybrid_in', 'hybrid_in_hm5', 'hybrid_in_hp5')


def run(out_dir, smoke=0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261118); M = Methods(); M.set_unidepth_pixels(PIXELS)
    report = dict(protocol=core.PROTOCOL_VERSION, pixels=PIXELS, arms=ARMS, smoke=smoke)
    report[NAME], rows = evaluate(NAME, M, rng, out, smoke, methods=ARMS, scaled=SCALED, intake_skip=('planar',))
    print('sources', json.dumps(report[NAME].get('sources')), flush=True)
    if smoke or not rows:
        (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8'); return
    comps = {f'{a}__{b}': core.compare(rows, a, b) for a, b in
             [('tof_unidepth', 'hybrid_in'), ('tof_dav2s', 'hybrid_in'), ('unidepth_s', 'hybrid_in'),
              ('unidepth_s', 'floor_unidepth'), ('tof_unidepth', 'floor_unidepth'),
              ('hybrid_in', 'hybrid_in_hm5'), ('hybrid_in', 'hybrid_in_hp5')]}
    sub = [r for r in rows if r['source']['tof_unidepth'] != 'tof']
    comps['tof_insufficient_subset'] = core.compare(sub, 'tof_unidepth', 'hybrid_in') if sub else None
    report['comparisons'] = comps
    report['decision'] = dict(I1_fallback_vs_tof_only=claim(comps['tof_unidepth__hybrid_in']),
                              I2_chain_vs_current=claim(comps['tof_dav2s__hybrid_in']))
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    for k, v in report['decision'].items():
        print(k, v.get('verdict'), json.dumps({x: v[x] for x in ('primary', 'relative_change', 'checks', 'gaps') if x in v}, default=str))


if __name__ == '__main__':
    a = sys.argv[1:]
    run(a[0], int(a[a.index('--smoke') + 1]) if '--smoke' in a else 0)
