"""Rule change 2026-11-17 (user's choice, DIRECTION_A_CONFIRM_SANPO_HEAD_20261117.md): the unseen head data failed the
intake planarity check (0.36 < 0.40), so each session gets the same intake check on its own frames and only passing
sessions are kept: ground plane share >= 0.40 and median sensor height in 1.2-1.9 m. Evaluator side only, before any
method is run on these sessions.
Usage: head_noseg_session_filter.py <out json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import adapters
import core


def main(out_path):
    by = defaultdict(list)
    for f in adapters.sanpo_head_noseg(keep=None):
        by[f.seq].append(f)
    res = {}
    for s, fr in sorted(by.items()):
        sc = core.self_check(fr, np.random.default_rng(int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)))
        res[s] = dict(frames=len(fr), plane_share=sc['plane_share_as_optical_z'], height=sc['sensor_height_median_by_mount'].get('head'),
                      keep=bool(sc['checks']['planar'] and sc['checks']['height']))
    keep = [s for s, v in res.items() if v['keep']]
    out = dict(rule='per-session planar (>= 0.40) and head height median in 1.2-1.9 m', sessions=len(res), kept=len(keep),
               frames_kept=sum(res[s]['frames'] for s in keep), keep=keep, per_session=res)
    Path(out_path).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print({k: v for k, v in out.items() if k not in ('keep', 'per_session')})


if __name__ == '__main__':
    main(sys.argv[1])
