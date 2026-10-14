"""Compare frozen v3/v6 against the pitch-fixed extrinsic rerun per segment (EXPLORE, 2026-10-14)."""
import json
from pathlib import Path
W = Path(r'E:/linnan/linnan/artifacts.local/work')
ORIG = {'v6': {t: W / f'fusion-v6-dev-20261013/{t}/frozen_fusion_eval_v6.json' for t in ('m12', 'a12', 'm13', 'a13', 'e13')},
        'v3': {'m12': W / 'fusion-v3-dev-20261013/v3_morning/frozen_fusion_eval_v3.json', 'a12': W / 'fusion-v3-dev-20261013/v3_afternoon/frozen_fusion_eval_v3.json',
               'm13': W / 'frozen-eval-capture-20261013/eval/frozen_fusion_eval_v3.json', 'a13': W / 'frozen-eval-capture-20261013b/eval/frozen_fusion_eval_v3.json',
               'e13': W / 'frozen-eval-capture-20261013c/eval/frozen_fusion_eval_v3.json'}}
out = {}
for ver in ('v6', 'v3'):
    rows = []
    for tag, op in ORIG[ver].items():
        fp = W / f'hindsight-truth-dev-20261014/pitchfix/{ver}/{tag}/frozen_fusion_eval_{ver}.json'
        if not (op.exists() and fp.exists()):
            continue
        o, f = json.loads(op.read_text(encoding='utf8'))['sessions'], json.loads(fp.read_text(encoding='utf8'))['sessions']
        for name in sorted(o):
            if name not in f: continue
            a, b = o[name], f[name]
            anc = lambda s: {k: sum(fr.get('anchor') == k for fr in s['frames']) for k in ('tof', 'robust', 'floor', None)}
            rows.append(dict(batch=tag, seg=name[17:23], cls=a['cls'], scene=(a.get('scene') or '')[:60], n=a['ready_frames'],
                             any=(a['fused_any_alert'], b['fused_any_alert']), end=(a['end_window_any_alert'], b['end_window_any_alert']),
                             rgb=(a['rgb_any'], b['rgb_any']), passed=(a['pass'], b['pass']), anchor=(anc(a), anc(b))))
    out[ver] = rows
    print(f'==== {ver}: segment  class  any-alert orig->fix  end-window orig->fix  pass orig->fix  tof-anchor frames orig->fix')
    for r in rows:
        f2 = lambda p: '/'.join('-' if v is None else f'{v:.2f}' for v in p)
        print(f"{r['batch']} {r['seg']} {r['cls']:11s} any {f2(r['any'])}  end {f2(r['end'])}  pass {r['passed'][0]}->{r['passed'][1]}  "
              f"tofanc {r['anchor'][0].get('tof',0)+r['anchor'][0].get('robust',0)}->{r['anchor'][1].get('tof',0)+r['anchor'][1].get('robust',0)}  {r['scene'][:40]}")
(W / 'hindsight-truth-dev-20261014/pitchfix/compare.json').write_text(json.dumps(out, indent=1), encoding='utf8')
