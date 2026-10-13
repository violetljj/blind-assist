"""Score adt_joint_anchor_dev_20261013 output (plan: adt-eval-20261013/PLAN_JOINT_ANCHOR.md).
Usage: adt_joint_anchor_analysis_dev_20261013.py <joint dir> <prep dir> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v2 as V2
from adt_backbone_motion_analysis_dev_20261013 import curve, fa_at, score


def main(jdir, prep, out):
    seqs = sorted(Path(jdir).glob('*.json')); frames_all = []; speeds = []
    for p in seqs:
        frames_all.append(json.loads(p.read_text(encoding='utf8'))['frames']); speeds.append(np.load(Path(prep) / p.stem / 'poses.npz')['speed'])
    res = {}
    for arm in ('chain', 'joint', 'floor', 'oracle'):
        rows = []
        for frames, sp in zip(frames_all, speeds):
            a = np.zeros((len(frames), 2), bool)
            for i, f in enumerate(frames):
                x = f['arms'][arm]
                if x is not None:
                    a[i] = [sum(b[0] for b in x[q][1] if b[1] <= 1.8) >= 150 for q in (0, 1)]
            rows.append((np.stack([V2.confirm(a[:, 0]), V2.confirm(a[:, 1])], 1), frames, sp))
        s = score(rows); cv = curve(frames_all, arm)
        res[arm] = dict(v6=s, curve=cv, fa_at={str(r): fa_at(cv, r) for r in (.8, .85, .9)})
        print(f"{arm:7s} v6 FA {s['fa']['rate']:.4f} ({s['fa']['n'][0]}/{s['fa']['n'][1]}) rec {s['rec']['rate']:.4f} near {s['rec_near']['rate']:.3f} mid {s['rec_mid']['rate']:.3f} | "
              f"FA at rec>=.80/.85/.90 {[None if c is None else (c[0], c[1], round(c[2], 4), round(c[3], 3)) for c in res[arm]['fa_at'].values()]}", flush=True)
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:4])
