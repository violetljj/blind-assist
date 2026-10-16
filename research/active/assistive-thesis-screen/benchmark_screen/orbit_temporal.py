"""Offline ORBIT scoring of dumped per-frame scores (orbit_dump.py): arms x causal temporal accumulation, official
50-task / 200-frame sampling. CPU only, seconds. EXPLORE, 2026-10-15.

Arms: full, part@j, fuse@j = z(full) + z(part@j) (z-scored over the user's objects per frame), head:<name> (learned
matching, orbit_dump head=...), ens:<prefix> = mean z-score of all arms whose name starts with <prefix> (seed ensemble),
sum:<a>+<b> = z(a) + z(b) (e.g. whole-frame part memory + zoomed look), gate:<a>+<b>@<tau> = z(a) where the gap between
the two best z(a) is >= tau, else z(z(a) + z(b)) (look again only when unsure; the result reports the share of valid
frames that were looked at again, 'zoom_frac'), switch:<a>><b>@<tau> = z(a), replaced by z(b) on the unsure frames
(for candidate verification, whose scores already contain the whole-frame ranking).
Temporal (causal - the official rules allow the current and all earlier frames of the clutter video, no labels):
  h0      current frame only
  hN      exponential moving average of the per-frame z-scores with a half-life of N frames
  cum     running mean from the first frame
Usage: orbit_temporal.py <dump dir> <out json> [tasks=50] [arms=all] [halflives=0,5,15,30,60,120,cum]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter


def z(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)


def causal(s, h):
    if h == '0':
        return s
    if h == 'cum':
        return np.cumsum(s, 0) / np.arange(1, len(s) + 1)[:, None]
    a = 0.5 ** (1 / float(h))
    return lfilter([1 - a], [1, -a], s, axis=0) / (1 - a ** np.arange(1, len(s) + 1))[:, None]   # bias-corrected EMA


def main(dump, out, tasks=50, arms='all', halflives='0,5,15,30,60,120,cum'):
    dump = Path(dump); tasks = int(tasks); hs = halflives.split(',')
    vids = [dict(np.load(f, allow_pickle=False)) for f in sorted(dump.glob('*.npz'))]
    keys = [k for k in vids[0] if k not in ('names', 'valid', 'y', 'user') and not k.startswith(('box_', 'aux_'))]
    all_arms = keys + ['fuse@' + k.split('@')[1] for k in keys if k.startswith('part@')]
    want = all_arms if arms == 'all' else [a for x in arms.split(',') for a in (all_arms if x == 'all' else [x])]
    rng = random.Random(0)
    samples = [[sorted(rng.sample(range(int(v['valid'].sum())), min(200, int(v['valid'].sum())))) for _ in range(tasks)]
               for v in vids]
    res = {}
    for arm in want:
        for h in hs:
            accs, users, looked = [], {}, []
            for v, smp in zip(vids, samples):
                if arm.startswith('fuse@'):
                    s = z(v['full']) + z(v['part@' + arm.split('@')[1]])
                elif arm.startswith('ens:'):
                    s = np.mean([z(v[k]) for k in keys if k.startswith(arm[4:])], 0)
                elif arm.startswith('sum:'):
                    a, b = arm[4:].split('+'); s = z(v[a]) + z(v[b])
                elif arm.startswith('switch:'):
                    ab, tau = arm[7:].rsplit('@', 1); a, b = ab.split('>'); s = z(v[a]); top = np.sort(s, 1)
                    unsure = (top[:, -1] - top[:, -2]) < float(tau)
                    s = np.where(unsure[:, None], z(v[b]), s); looked.append(unsure[v['valid']])
                elif arm.startswith('gate:'):
                    ab, tau = arm[5:].rsplit('@', 1); a, b = ab.split('+'); s = z(v[a]); top = np.sort(s, 1)
                    unsure = (top[:, -1] - top[:, -2]) < float(tau)
                    s = np.where(unsure[:, None], z(z(v[a]) + z(v[b])), s); looked.append(unsure[v['valid']])
                else:
                    s = z(v[arm])
                ok = (causal(s, h).argmax(1) == int(v['y']))[v['valid']]
                a = [ok[i].mean() for i in smp]; accs += a; users.setdefault(str(v['user']), []).extend(a)
            accs = np.array(accs)
            res[f'{arm}|h{h}' if h != 'cum' else f'{arm}|cum'] = dict(
                acc=round(100 * accs.mean(), 2), ci=round(100 * 1.96 * accs.std() / np.sqrt(len(accs)), 2),
                per_user={u: round(100 * float(np.mean(x)), 1) for u, x in sorted(users.items())},
                **({'zoom_frac': round(float(np.concatenate(looked).mean()), 3)} if looked else {}))
    meta = json.loads((dump / 'meta.json').read_text())
    out_obj = dict(dump=str(dump), backbone=meta['backbone'], cfgs=meta['cfgs'], n_videos=len(vids), tasks=tasks, results=res)
    Path(out).write_text(json.dumps(out_obj, indent=1))
    for k, r in sorted(res.items(), key=lambda kv: -kv[1]['acc'])[:12]:
        print(f"{k:22s} {r['acc']:6.2f} +- {r['ci']}")


if __name__ == '__main__':
    main(*sys.argv[1:])
