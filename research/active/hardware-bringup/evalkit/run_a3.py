"""Direction A experiment 3 (DIRECTION_A3_PHONE_RESOLUTION_20261117.md): the adopted scale chain at phone pixel budgets.
Usage:
  run_a3.py run <out dir> <pixels> [--smoke N]          UniDepth arms at a fixed input pixel budget
  run_a3.py compare <A2 dir> <dir 203056> <dir 315560> <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import core
from methods import Methods
from run_a2 import DATASETS, evaluate

ARMS = ('unidepth_s', 'tof_unidepth_out', 'hybrid_out')
BUDGETS = {203056: '392x518', 315560: '490x644'}


def run(out_dir, pixels, smoke=0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261117); M = Methods(); M.set_unidepth_pixels(int(pixels))
    report = dict(protocol=core.PROTOCOL_VERSION, pixels=int(pixels), arms=ARMS, smoke=smoke)
    for name in DATASETS:
        report[name], _ = evaluate(name, M, rng, out, smoke, methods=ARMS, scaled=('tof_unidepth_out', 'hybrid_out'))
        print(name, 'sources', json.dumps(report[name].get('sources')), flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')


def not_worse(c):
    """A3 rule on c = core.compare(rows, default, phone): primary rise <= 15 %, recall loss <= max(3, 2 %) frames, FA rise <= 3 %."""
    a, b = c['methods']
    prim = {m: (c['depth']['corridor_LOW']['absrel'][m] + c['depth']['corridor_BODY']['absrel'][m]) / 2 for m in (a, b)}
    checks = {}
    for band in ('LOW', 'BODY'):
        d = c['detect'][band]
        checks[band] = dict(recall_change_frames=d['tp'][b] - d['tp'][a], recall_tol_frames=max(3, .02 * d['positives']),
                            fa_change_frames=d['fp'][b] - d['fp'][a], fa_tol_frames=.03 * d['negatives'])
        checks[band]['ok'] = (checks[band]['recall_change_frames'] >= -checks[band]['recall_tol_frames']
                              and checks[band]['fa_change_frames'] <= checks[band]['fa_tol_frames'])
    rel = prim[b] / prim[a] - 1
    return dict(primary=prim, relative_change=rel, checks=checks, not_worse=rel <= .15 and all(v['ok'] for v in checks.values()))


def compare(a2_dir, d203, d316, out_path):
    res = {}
    for name in DATASETS:
        base = json.loads((Path(a2_dir) / f'rows_{name}.json').read_text(encoding='utf-8'))
        merged = base
        for pixels, d in ((203056, d203), (315560, d316)):
            rows = json.loads((Path(d) / f'rows_{name}.json').read_text(encoding='utf-8'))
            assert len(rows) == len(merged), (name, pixels, len(rows), len(merged))
            same_truth = sum(r['truth'] == q['truth'] and r['seq'] == q['seq'] and r['t'] == q['t'] for r, q in zip(rows, merged))
            res.setdefault(name, {})[f'truth_identical_{pixels}'] = same_truth
            for r, q in zip(rows, merged):
                for arm in ARMS:
                    q['alert'][f'{arm}@{BUDGETS[pixels]}'] = r['alert'][arm]
                    q['depth'][f'{arm}@{BUDGETS[pixels]}'] = r['depth'][arm]
        for arm in ARMS:
            for tag in BUDGETS.values():
                c = core.compare(merged, arm, f'{arm}@{tag}')
                res[name][f'{arm}@{tag}'] = dict(compare=c, rule=not_worse(c))
    v = {tag: all(res[n][f'hybrid_out@{tag}']['rule']['not_worse'] for n in DATASETS) for tag in BUDGETS.values()}
    res['verdict'] = 'PHONE_392x518' if v['392x518'] else 'PHONE_490x644' if v['490x644'] else 'PHONE_RESOLUTION_LOSS'
    Path(out_path).write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
    for n in DATASETS:
        print(n, {k: v for k, v in res[n].items() if k.startswith('truth')})
        for arm in ARMS:
            for tag in BUDGETS.values():
                r = res[n][f'{arm}@{tag}']['rule']
                print(f"  {arm:16s} {tag}: primary {r['primary'][arm]:.4f} -> {r['primary'][arm + '@' + tag]:.4f} ({r['relative_change']:+.1%})",
                      {b: (x['recall_change_frames'], x['fa_change_frames'], x['ok']) for b, x in r['checks'].items()}, 'not_worse', r['not_worse'])
    print('verdict', res['verdict'])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'run':
        run(a[1], a[2], int(a[a.index('--smoke') + 1]) if '--smoke' in a else 0)
    else:
        compare(*a[1:5])
