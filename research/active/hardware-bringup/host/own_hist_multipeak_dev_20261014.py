"""Real-CNH multi-peak detection on own head-worn captures (plan: own-blob-dump-20261013/PLAN_REAL_MULTIPEAK.md).

Consumed Development data, frozen labels. Per pose-ok ready frame: S = frozen ToF strong grade (from the v6 blob dump),
A = firmware distance (status 5/6/9/10) along the zone centre ray in a box, B = any bin >= 1 whose adapter_v5 evidence
(calibrated signed bias + noise model, as frozen v1 feeds the engine) is >= 5 and whose whole 0.30 m range interval lies
in box depth 0.3-2.0 m with the bin-centre point inside the box laterally/vertically; RGB = v6 blobs from the dump.
Usage: own_hist_multipeak_dev_20261014.py <dump dir> <raw dir>...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
import frozen_fusion_eval_v2 as V2
from real_contract_v5 import adapter_v5

THR, BIN_M = 5.0, 8 * .0375348
BROOM = ('054159', '054301', '054334', '054405', '054437', '054511')


def main(dump, *raws):
    from atom_imu import load_mounting
    F, R = V1.F, V1.R
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    arrays = np.load(V1.P.OUT / 'instrument_arrays.npz'); bias = arrays['bias'].reshape(64, 16)
    noise = V1.P.read('instrument_parameters.json')['noise']
    rays = V1.X.zone_rays(); ctr = rays.mean(1); ctr /= np.linalg.norm(ctr, axis=1, keepdims=True)   # [64,3] ToF frame, model order
    lo_r = np.arange(16) * BIN_M; hi_r = lo_r + BIN_M; mid = lo_r + BIN_M / 2
    dirs = {d.name: d for r in raws for d in Path(r).iterdir()}
    C = V1.CRITERIA; rows = []
    for jf in sorted(Path(dump).glob('capture-*.json')):
        J = json.loads(jf.read_text(encoding='utf8')); fr = J['frames']; lab = J['label']; c = lab['class']
        if not fr:
            continue
        x = R.load(dirs[jf.stem], imu_rot)
        z = adapter_v5(x['H'], x['A'], bias, noise, None, True, False).reshape(-1, 64, 16)
        dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
        raw = {k: np.zeros((len(fr), 2), bool) for k in ('S', 'A', 'B', 'RGB')}
        for j, f in enumerate(fr):
            i = f['tof_index']
            raw['S'][j] = f['tof_strong']
            if f['blobs'] is not None:
                raw['RGB'][j] = [sum(b[0] for b in f['blobs'][q][1] if b[1] <= 1.8) >= V1.MIN_PX for q in (0, 1)]
            if not f['pose_ok']:
                continue
            Rp = F.rot(f['pitch'], f['roll'])
            cc = ctr @ Rp.T                                                  # zone centre rays, corridor frame
            pA = cc * np.nan_to_num(dist[i], nan=-1.)[:, None]
            zlo, zhi = cc[:, None, 2] * lo_r[None], cc[:, None, 2] * hi_r[None]
            pB = cc[:, None, :] * mid[None, :, None]                         # [64,16,3]
            det = (z[i] >= THR); det[:, 0] = False
            for q, (lo, hi) in enumerate(F.BOXES.values()):
                inA = np.isfinite(dist[i]) & (np.abs(pA[:, 0]) <= .3) & (pA[:, 1] >= lo) & (pA[:, 1] <= hi) & (pA[:, 2] >= .3) & (pA[:, 2] <= V1.Z_MAX)
                inB = (np.abs(pB[..., 0]) <= .3) & (pB[..., 1] >= lo) & (pB[..., 1] <= hi) & (zlo >= .3) & (zhi <= V1.Z_MAX)
                raw['A'][j, q] = bool(inA.any()); raw['B'][j, q] = bool((det & inB).any())
        def conf(m):
            return np.stack([V2.confirm(m[:, 0]), V2.confirm(m[:, 1])], 1)
        sysm = {'S': conf(raw['S']), 'A': conf(raw['A']), 'B': conf(raw['B']), 'RGB': conf(raw['RGB']),
                'v6': conf(raw['S'] | raw['RGB']), 'v6+B': conf(raw['S'] | raw['RGB'] | raw['B'])}
        E = V1.END_WINDOW
        st = {k: dict(any=float(m.any(1).mean()), end=float(m[-E:].any(1).mean()), end_head=float(m[-E:, 0].mean())) for k, m in sysm.items()}
        def ok(r):
            return (None if c in ('calibration', 'unknown') else r['any'] <= C['quiet_max_alert'] if c == 'quiet' else
                    r['any'] <= C['side_max_alert'] if c == 'side' else r['end'] >= C['ahead_min_end_alert'] if c == 'ahead' else
                    r['end_head'] >= C['head_min_end_alert'] if c == 'head' else None)
        row = dict(session=jf.stem, cls=c, scene=lab.get('scene'), n=len(fr), stats=st, pass_v6=ok(st['v6']), pass_v6B=ok(st['v6+B']))
        rows.append(row)
        print(f"{jf.stem[17:23]} {c:11s} {str(lab.get('scene'))[:28]:28s} n={len(fr):3d} | end S {st['S']['end']:.2f} A {st['A']['end']:.2f} B {st['B']['end']:.2f} RGB {st['RGB']['end']:.2f} | "
              f"any S {st['S']['any']:.2f} B {st['B']['any']:.2f} v6 {st['v6']['any']:.2f} v6+B {st['v6+B']['any']:.2f} | pass {row['pass_v6']}->{row['pass_v6B']}", flush=True)
    br = [r for r in rows if r['session'][17:23] in BROOM]
    wins = sum(r['stats']['B']['end'] > r['stats']['S']['end'] for r in br)
    flips = [r['session'][17:23] for r in rows if r['pass_v6'] is True and r['pass_v6B'] is False]
    print(f'broom sessions where B end-window > S: {wins}/{len(br)}; judged pass->fail when adding B: {flips}')
    (Path(dump) / 'real_multipeak.json').write_text(json.dumps(dict(rows=rows, broom_wins=wins, flips=flips), indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:])
