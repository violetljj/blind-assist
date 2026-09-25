"""Summarize v1.3 pilot readout JSONs into compact tables (fast lane)."""
import json
import sys
from pathlib import Path


def fmt(x, digits=3):
    return '—' if x is None else f'{x:.{digits}f}'


def main(root):
    root = Path(root)
    rows = []
    for path in sorted(root.glob('mount*-snr*-*hz.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        for arm, groups in r['arms'].items():
            for g in ('HEAD', 'BODY'):
                m = groups[g]
                rows.append((r['mount'], r['snr'], r['rate'], arm, g, m['macro_AP'], m['AP'], m['recall'], m['FPR'],
                             m['TP'], m['positive'], m['FP'], m['negative'],
                             {k: v['AP'] for k, v in m['strata_AP'].items()}))
    print('| mount | SNR | Hz | arm | group | macro AP | pooled AP | recall | FPR | TP/pos | FP/neg |')
    print('|---|---|---|---|---|---:|---:|---:|---:|---|---|')
    for r in rows:
        print(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {fmt(r[5])} | {fmt(r[6])} | {fmt(r[7])} | {fmt(r[8])} | '
              f'{r[9]}/{r[10]} | {r[11]}/{r[12]} |')
    print('\nStrata AP (positives of stratum + all negatives):')
    for r in rows:
        print(r[:5], {k: fmt(v) for k, v in r[13].items()})


if __name__ == '__main__':
    main(sys.argv[1])
