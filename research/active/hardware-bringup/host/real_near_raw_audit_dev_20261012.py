"""Inspect raw bin0/bin1 lineage of consumed quiet phone clips; no fitting."""
from pathlib import Path
import json
import numpy as np
import real_phone_imu_replay as R

OUT = R.ROOT / 'artifacts.local/work/cnh-near-residual-dev-20261012/raw_audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rotation = np.asarray(R.load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    reports = {}
    for directory in sorted(R.RAW.iterdir()):
        key = f'{int(directory.name[17:19])+8:02d}{directory.name[19:23]}'
        if key not in ('100702', '100754', '103924'):
            continue
        d = R.load(directory, rotation)
        h = d['H'].reshape(-1, 8, 8, 16)[:, ::-1, ::-1]
        distance = d['D'].reshape(-1, 8, 8)[:, ::-1, ::-1]
        status = d['S'].reshape(-1, 8, 8)[:, ::-1, ::-1]
        rows = []
        for r, c in np.ndindex(8, 8):
            a, b = h[:, r, c, 0], h[:, r, c, 1]
            valid = np.isin(status[:, r, c], (5, 6, 9, 10))
            rows.append(dict(row=r, col=c, bin0_mean=float(a.mean()),
                             bin1_mean=float(b.mean()), bin1_sd=float(b.std()),
                             ratio_median=float(np.median(b / np.maximum(a, 1e-9))),
                             correlation=float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 0 else None,
                             firmware_valid=int(valid.sum()),
                             firmware_distance_median_mm=float(np.median(distance[valid, r, c])) if valid.any() else None))
        reports[key] = dict(frames=len(h), source=str(directory), zones=rows,
                            top_bin1=sorted(rows, key=lambda x: x['bin1_mean'], reverse=True)[:5])
    (OUT / 'raw_lineage.json').write_text(json.dumps(reports, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({k: v['top_bin1'] for k, v in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
