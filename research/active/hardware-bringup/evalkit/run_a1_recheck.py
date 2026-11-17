"""Direction A experiment 1 recheck on fresh SANPO-Real head sessions (DIRECTION_A1_RECHECK_SANPO_20261117.md).
Usage: run_a1_recheck.py <out dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import core
from methods import A1_METHODS, Methods
from run_a1 import decide, evaluate, print_table

NAME = 'sanpo_head_seg'
EPS = 1e-9          # float round-off only (A1 lesson); counts are compared as the rule states
MIN_POS = 30        # a band's recall tolerance is NOT_EVALUABLE below this many positive frames
MIN_BLOCKS = 10     # primary metric needs corridor AbsRel in >= this many sessions for both bands


def judge(s):
    d = decide(s, eps=EPS)
    d_out = decide(s, 'tof_dav2s_out', 'tof_unidepth_out', eps=EPS)
    gaps = [f'recall_{b}' for b in ('LOW', 'BODY') if s['tof_dav2s'][f'detect_{b}']['positives'] < MIN_POS]
    gaps += [f'corridor_{b}_blocks' for b in ('LOW', 'BODY')
             if min(s[m][f'corridor_{b}']['blocks'] for m in A1_METHODS) < MIN_BLOCKS]
    for x in (d, d_out):     # a band without enough positives cannot fail the recall tolerance; re-evaluate without it
        if any(g.startswith('recall_') for g in gaps) and x['verdict'] == 'KEEP_TOF_DAV2S' and x['relative_change'] < .15:
            ok_fa = all(v <= .03 + EPS for v in x['fa_delta'].values())
            ok_rec = all(v >= -.02 - EPS for b, v in x['recall_delta'].items() if f'recall_{b}' not in gaps)
            x['verdict_without_unevaluable_recall'] = ('ADOPT_TOF_UNIDEPTH' if x['relative_change'] <= -.15 and ok_fa and ok_rec
                                                      else 'UNCERTAIN' if ok_fa and ok_rec else 'KEEP_TOF_DAV2S')
    if any(g.endswith('_blocks') for g in gaps):
        overall = 'NOT_EVALUABLE'
    else:
        v1 = d.get('verdict_without_unevaluable_recall', d['verdict']); v2 = d_out.get('verdict_without_unevaluable_recall', d_out['verdict'])
        overall = ('CONFIRMED' if v1 == v2 == 'ADOPT_TOF_UNIDEPTH' else 'IDEAL_ONLY' if v1 == 'ADOPT_TOF_UNIDEPTH'
                   else 'NOT_CONFIRMED' if v1 == 'KEEP_TOF_DAV2S' else 'UNCERTAIN')
    return dict(ideal=d, outdoor_limited=d_out, gaps=gaps, overall=overall)


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261117); M = Methods()
    report = dict(protocol=core.PROTOCOL_VERSION, methods=A1_METHODS, **{NAME: evaluate(NAME, M, rng, out)})
    if 'summary' in report[NAME]:
        report['decision'] = judge(report[NAME]['summary'])
        print(json.dumps(report['decision'], indent=1))
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print_table(report, [NAME])


if __name__ == '__main__':
    main(sys.argv[1])
