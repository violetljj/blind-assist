"""How much of the GT obstacle mass in the HEAD/BODY boxes does the ToF field see, and what would a second, down-tilted
ToF add? GT-only geometry on prepared ADT frames (EXPLORE; no model).

Per GT-positive query-frame (>= 150 in-box GT px, z50 <= 1.5 m): fraction of in-box GT pixels whose ray lies inside the
+-22 deg square field of the current ToF (camera<-ToF extrinsic) and of the same ToF pitched further down by 15/25/35/45
deg (about the ToF x axis); a query-frame counts as covered when >= 150 of its in-box pixels (full-res) are inside.
Usage: adt_tof_coverage_dev_20261013.py <prep dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1

TILTS = tuple(int(x) for x in __import__("os").environ.get("TILTS", "0,15,25,35,45").split(","))
HALF = np.radians(float(__import__("os").environ.get("HALF_FOV", "22")))   # 22: VL53L8 (45 deg); 30: VL53L7CX (60 deg)
Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation'])


def rx(d):
    a = np.radians(d); return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


cov = {t: [0, 0] for t in TILTS}; cov.update({f'0+{t}': [0, 0] for t in TILTS[1:]}); qn = {0: 0, 1: 0}; qc = {}
for seq in sorted(p for p in Path(sys.argv[1]).iterdir() if (p / 'frames.npz').exists() and 'sample' not in p.name):
    meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
    vv, uu = np.mgrid[0:h, 0:w]; dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
    with np.load(seq / 'frames.npz') as Z:
        dep, R = Z['depth'], Z['R_cc']
    for i in range(len(dep)):
        z = dep[i].reshape(-1) / 1000.; ok = z > .1
        pc = dirs[ok] * z[ok, None]; p = pc @ R[i].T
        ptof = pc @ Rct                                         # camera -> ToF: Rct.T @ pc
        inside = {}
        for t in TILTS:
            q_ = ptof @ rx(-t)                                  # ToF pitched down by t (y down; sign checked: +t shrank coverage)
            inside[t] = (q_[:, 2] > 0) & (np.abs(q_[:, 0]) <= np.tan(HALF) * q_[:, 2]) & (np.abs(q_[:, 1]) <= np.tan(HALF) * q_[:, 2])
        for q, (lo, hi) in enumerate(V1.F.BOXES.values()):
            m = (np.abs(p[:, 0]) <= .3) & (p[:, 1] >= lo) & (p[:, 1] <= hi) & (p[:, 2] >= .3) & (p[:, 2] <= 2.)
            if m.sum() < 150 or np.median(p[m, 2]) > 1.5:
                continue
            qn[q] += 1
            for t in TILTS:
                cov[t][0] += int((m & inside[t]).sum()); cov[t][1] += int(m.sum())
                qc.setdefault((q, t), 0); qc[(q, t)] += int((m & inside[t]).sum() >= 150)
                if t:
                    u = m & (inside[0] | inside[t]); cov[f'0+{t}'][0] += int(u.sum()); cov[f'0+{t}'][1] += int(m.sum())
                    qc.setdefault((q, f'0+{t}'), 0); qc[(q, f'0+{t}')] += int(u.sum() >= 150)
    print(seq.name, flush=True)
print('positive query-frames HEAD %d BODY %d' % (qn[0], qn[1]))
for k in cov:
    print(f'ToF {"tilt +" + str(k) if isinstance(k, int) else "current + tilt " + k[2:]:>18s}: in-box GT px covered {cov[k][0] / cov[k][1]:.2f} | '
          f'query-frames covered HEAD {qc[(0, k)] / qn[0]:.2f} BODY {qc[(1, k)] / qn[1]:.2f}')
