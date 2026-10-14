"""Score adt_hist_thin_dev_20261014 output (plan: adt-eval-20261013/PLAN_HIST_THIN.md).
Usage: adt_hist_thin_analysis_dev_20261014.py <sweep dir> <hist dir> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v2 as V2

ARMS = ('cov', 'A', 'B', 'B4', 'rgb', 'rgb|cov', 'rgb|A', 'rgb|B', 'rgb|B4')


def main(sweep, hist, out):
    sw, hd = Path(sweep), Path(hist); res = {}
    seqs = sorted(p.name[:-len('.truth.json')] for p in sw.glob('*.truth.json'))
    for t in (0, 30):
        c = {a: {k: [0, 0] for k in ('rec', 'rec_partial', 'rec_full', 'empty')} for a in ARMS}
        for s in seqs:
            truth = json.loads((sw / f'{s}.truth.json').read_text(encoding='utf8'))
            base = json.loads((sw / f'{s}.i{t}.json').read_text(encoding='utf8'))
            H = json.loads((hd / f'{s}.t{t}.hist.json').read_text(encoding='utf8'))
            n = len(truth); raw = {a: np.zeros((n, 2), bool) for a in ARMS}
            for i in range(n):
                for q in (0, 1):
                    rgb = sum(b[0] for b in base[i]['blobs'][q] if b[1] <= 1.8) >= 150
                    cov, A, B, B4 = base[i]['tof'][q], H[i][q][0], H[i][q][2], H[i][q][6]
                    for a, val in (('cov', cov), ('A', A), ('B', B), ('B4', B4), ('rgb', rgb), ('rgb|cov', rgb or cov), ('rgb|A', rgb or A), ('rgb|B', rgb or B), ('rgb|B4', rgb or B4)):
                        raw[a][i, q] = val
            for a in ARMS:
                M = np.stack([V2.confirm(raw[a][:, 0]), V2.confirm(raw[a][:, 1])], 1)
                for i, f in enumerate(truth):
                    for q in (0, 1):
                        tq = f['truth'][q]
                        if tq['n'] == 0:
                            ks = ['empty']
                        elif tq['n'] >= 150 and tq['z50'] <= 1.5:
                            ks = ['rec', 'rec_full' if raw['cov'][i, q] else 'rec_partial']
                        else:
                            continue
                        for k in ks:
                            c[a][k][0] += int(M[i, q]); c[a][k][1] += 1
        res[f't{t}'] = {a: {k: dict(n=v, rate=round(v[0] / max(v[1], 1), 4)) for k, v in d.items()} for a, d in c.items()}
        print(f'\n== extra down-tilt {t}: recall all / partial (coverage ToF misses) / full; empty FA  [%]')
        for a in ARMS:
            r = res[f't{t}'][a]
            print(f"{a:8s} {r['rec']['rate']*100:5.1f} / {r['rec_partial']['rate']*100:5.1f} / {r['rec_full']['rate']*100:5.1f} ; empty FA {r['empty']['rate']*100:4.1f}")
        r = res[f't{t}']['cov']
        print('denominators: positives', r['rec']['n'][1], 'partial', r['rec_partial']['n'][1], 'full', r['rec_full']['n'][1], 'empty', r['empty']['n'][1])
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:4])
