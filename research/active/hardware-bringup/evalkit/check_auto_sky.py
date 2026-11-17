"""Acceptance check of automatic sky masks on the seen SANPO head subset (DIRECTION_A_CONFIRM_SANPO_HEAD_20261117.md,
step 1). Evaluator side only: truth per band and outdoor-ToF >= 5 zones with human vs automatic sky, same frame seeds.
Usage: check_auto_sky.py <out json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import adapters
import core
from methods import Methods


def seed(f):
    return int(hashlib.sha256(f'{f.seq}:{f.index}'.encode()).hexdigest()[:8], 16)


def main(out_path):
    M = Methods(); same = {b: 0 for b in core.BANDS}; tof_same = 0; n = 0; iou = []
    for fh, fa in zip(adapters.sanpo_head_seg(), adapters.sanpo_head_seg_autosky()):
        assert (fh.seq, fh.index) == (fa.seq, fa.index)
        rh, ra = core.regions(fh, np.random.default_rng(seed(fh))), core.regions(fa, np.random.default_rng(seed(fa)))
        for b in core.BANDS:
            same[b] += rh.get(f'truth_{b}') == ra.get(f'truth_{b}')
        th = np.isfinite(M._zones(fh, True)[0]).sum() >= 5; ta = np.isfinite(M._zones(fa, True)[0]).sum() >= 5
        tof_same += th == ta
        sh, sa = ~np.isfinite(fh.ref_depth), ~np.isfinite(fa.ref_depth)      # removed or invalid pixels
        iou.append(float((sh & sa).sum() / max(1, (sh | sa).sum())))
        n += 1
    res = dict(frames=n, truth_same_share={b: same[b] / n for b in core.BANDS}, tof5_same_share=float(tof_same / n),
               removed_pixel_iou_median=float(np.median(iou)))
    res['pass'] = bool(all(v >= .98 for v in res['truth_same_share'].values()) and res['tof5_same_share'] >= .98)
    Path(out_path).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1])
