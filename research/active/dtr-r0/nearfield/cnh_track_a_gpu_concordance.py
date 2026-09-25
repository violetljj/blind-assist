"""Concordance of GPU vs CPU scale readouts (same condition): scores, selections, APs, tests."""
import json
import sys
from pathlib import Path
import numpy as np


def main(cpu_dir, gpu_dir):
    cpu_dir, gpu_dir = Path(cpu_dir), Path(gpu_dir)
    worst, units = {}, 0
    for path in sorted(cpu_dir.glob('unit*.npz')):
        gp = gpu_dir/path.name
        if not gp.exists():
            continue
        units += 1
        a, b = np.load(path), np.load(gp)
        for k in a.files:
            if a[k].dtype.kind != 'f' or k in ('witness', 'visibility'):
                continue
            d = np.abs(a[k]-b[k])/np.maximum(np.abs(a[k]), 1)
            worst[k] = max(worst.get(k, 0.), float(np.nanmax(d)))
    ra = json.loads((cpu_dir/'result.json').read_text(encoding='utf-8'))
    rb = json.loads((gpu_dir/'result.json').read_text(encoding='utf-8'))
    ap = max(abs(ra['arms'][k][g]['macro_AP']-rb['arms'][k][g]['macro_AP']) for k in ra['arms'] for g in ('HEAD', 'BODY'))
    taus = all(ra['selection'][k]['tau'] == rb['selection'][k]['tau'] for k in ra['selection'])
    tests = [(x['comparison'], x['group'], x['delta'], y['delta'], x['holm_reject_at_0.05'] == y['holm_reject_at_0.05'])
             for x, y in zip(ra['primary_tests'], rb['primary_tests'])]
    report = dict(units_compared=units, worst_rel_score_error=worst, max_macro_ap_diff=ap, tau_equal=taus,
                  primary_tests=[dict(comparison=c, group=g, cpu=d1, gpu=d2, same_decision=s) for c, g, d1, d2, s in tests],
                  all_decisions_equal=all(t[4] for t in tests))
    print(json.dumps(report, indent=1))
    return report


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
