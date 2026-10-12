"""Frozen fusion evaluation v2 = v1 + per-query temporal confirmation (2 of the last 3 ready frames).

Frozen 2026-10-12 before the 2026-10-13 captures (see ../FROZEN_FUSION_EVAL_V1_PROTOCOL.md, section v2).
Reads the v1 output (frozen_fusion_eval_v1.json) and re-scores with identical criteria.
Usage: frozen_fusion_eval_v2.py <dir containing frozen_fusion_eval_v1.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

K, N = 2, 3
CRITERIA = dict(quiet_max_alert=.10, side_max_alert=.30, ahead_min_end_alert=.50, head_min_end_alert=.50)
END_WINDOW = 15


def confirm(seq):
    s = np.asarray(seq, int)
    return np.array([s[max(0, i - N + 1):i + 1].sum() >= K for i in range(len(s))])


def main(folder):
    v1 = json.loads((Path(folder) / 'frozen_fusion_eval_v1.json').read_text(encoding='utf8'))
    out = dict(protocol='FROZEN_FUSION_EVAL_V2', rule=f'{K} of last {N} ready frames per query', v1_frozen_sha256=v1['frozen_sha256'], sessions={})
    for name, s in v1['sessions'].items():
        fr = s['frames']; c = s['cls']
        if not fr:
            out['sessions'][name] = dict(cls=c, ready_frames=0, passed=None); continue
        head = confirm([f['fused'][0] for f in fr]); body = confirm([f['fused'][1] for f in fr]); anyq = head | body
        res = dict(scene=s.get('scene'), cls=c, ready_frames=len(fr), fused_any_alert=float(anyq.mean()),
                   end_window_any_alert=float(anyq[-END_WINDOW:].mean()), end_window_head_alert=float(head[-END_WINDOW:].mean()))
        res['pass'] = (None if c in ('calibration', 'unknown') else
                       res['fused_any_alert'] <= CRITERIA['quiet_max_alert'] if c == 'quiet' else
                       res['fused_any_alert'] <= CRITERIA['side_max_alert'] if c == 'side' else
                       res['end_window_any_alert'] >= CRITERIA['ahead_min_end_alert'] if c == 'ahead' else
                       res['end_window_head_alert'] >= CRITERIA['head_min_end_alert'] if c == 'head' else None)
        out['sessions'][name] = res
        print(f"{name[17:23]} {c:11s} fused {res['fused_any_alert']:.2f} end {res['end_window_any_alert']:.2f} head-end {res['end_window_head_alert']:.2f} -> {res['pass']}")
    judged = [v for v in out['sessions'].values() if v.get('pass') is not None]
    out['summary'] = dict(judged=len(judged), passed=sum(v['pass'] for v in judged),
                          all_ahead_detected=all(v['pass'] for v in judged if v['cls'] == 'ahead') if any(v['cls'] == 'ahead' for v in judged) else None,
                          all_quiet_pass=all(v['pass'] for v in judged if v['cls'] == 'quiet') if any(v['cls'] == 'quiet' for v in judged) else None)
    (Path(folder) / 'frozen_fusion_eval_v2.json').write_text(json.dumps(out, indent=1), encoding='utf8')
    print(json.dumps(out['summary']))


if __name__ == '__main__':
    main(sys.argv[1])
