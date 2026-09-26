"""Alert persistence rules on saved v3 scores (Development, fast lane, descriptive).

Rule N/M: alert at frame t if at least N of the frames t-M+1..t score >= threshold
(earlier frames of the same sequence serve as history). Events and non-event
pairs follow the v3 evaluator (sequence x query over frames t >= 3, all audit
frames). Thresholds are either the frozen per-frame calib thresholds or chosen on
calib units so the non-event sequence false-alert rate matches a target; audit is
only scored. Reports event recall, timeliness (first alert witness Z >= 1 m among
events whose obstacle came within 1 m) and the sequence false-alert rate.
"""
import json
import sys
from pathlib import Path
import numpy as np

GROUPS = (('HEAD', (0, 2, 4)), ('BODY', (1, 3, 5)))
RULES = ((1, 1), (2, 2), (2, 3), (3, 3), (3, 4))
ARMS = {'S2': lambda f: f['S2__noisy@0.75'], 'S3': lambda f: np.maximum(f['S2__noisy@0.75'], f['memory__noisy'])}


def load(readouts):
    """Per split: list of sequences (scores[T,6] per arm, labels, witness, frame)."""
    seqs = {'calib': [], 'audit': []}
    for p in sorted(Path(readouts).glob('unit*.npz')):
        with np.load(p) as f:
            split = str(f['split'])
            if split not in seqs:
                continue
            cfg, frame = f['config'], f['frame']
            scores = {a: fn(f) for a, fn in ARMS.items()}
            for c in np.unique(cfg):
                m = np.flatnonzero(cfg == c)
                m = m[np.argsort(frame[m])]
                seqs[split].append(dict(frame=frame[m], y=f['labels'][m], w=f['witness'][m],
                                        s={a: v[m] for a, v in scores.items()}))
    return seqs


def alerts(s, thr, n, m):
    over = (s >= thr).astype(int)
    c = np.cumsum(np.vstack([np.zeros((1,)+over.shape[1:], int), over]), 0)
    lo = np.maximum(np.arange(len(s))-m+1, 0)
    return (c[1:]-c[lo]) >= n


def score(seqs, arm, boxes, thr, rule):
    ev = dict(events=0, alerted=0, near=0, timely=0, late=0, never=0, lead=[], nonevent=0, false=0)
    for sq in seqs:
        a = alerts(sq['s'][arm], thr, *rule)
        t = sq['frame'] >= 3
        for q in boxes:
            pos = np.flatnonzero(t & (sq['y'][:, q] == 1))
            if not len(pos):
                ev['nonevent'] += 1
                ev['false'] += int(a[t, q].any())
                continue
            ev['events'] += 1
            hits = pos[a[pos, q]]
            near = np.nanmin(sq['w'][pos, q]) <= 1.
            ev['near'] += int(near)
            if len(hits):
                ev['alerted'] += 1
                lead = sq['w'][hits[0], q]
                if np.isfinite(lead):
                    ev['lead'].append(float(lead))
                if near:
                    ev['timely' if lead >= 1. else 'late'] += 1
            elif near:
                ev['never'] += 1
    lead = ev.pop('lead')
    return dict(ev, recall=ev['alerted']/max(1, ev['events']), false_alert_rate=ev['false']/max(1, ev['nonevent']),
                median_first_alert_m=float(np.median(lead)) if lead else None,
                near_timely=ev['timely']/max(1, ev['near']), near_late=ev['late']/max(1, ev['near']),
                near_never=ev['never']/max(1, ev['near']))


def calib_threshold(seqs, arm, boxes, rule, target):
    """Lowest threshold whose calib sequence false-alert rate is <= target."""
    grid = np.linspace(1., 12., 441)
    for thr in grid:
        if score(seqs, arm, boxes, thr, rule)['false_alert_rate'] <= target:
            return float(thr)
    return float(grid[-1])


def main(readouts, out):
    result = json.loads((Path(readouts)/'result.json').read_text(encoding='utf-8'))
    seqs = load(readouts)
    report = dict(calib_sequences=len(seqs['calib']), audit_sequences=len(seqs['audit']), rows=[])
    for arm in ARMS:
        for g, boxes in GROUPS:
            frozen = result['arms'][f'{arm}/noisy'][g]['threshold']
            for rule in RULES:
                settings = [('frozen', frozen)] + [(f'calib_FA{int(100*t)}', calib_threshold(seqs['calib'], arm, boxes, rule, t))
                                                   for t in (.10, .20)]
                for name, thr in settings:
                    r = score(seqs['audit'], arm, boxes, thr, rule)
                    report['rows'].append(dict(arm=arm, group=g, rule=f'{rule[0]}/{rule[1]}', threshold_setting=name,
                                               threshold=thr, **r))
    Path(out).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print('calib/audit sequences', report['calib_sequences'], report['audit_sequences'])
    for r in report['rows']:
        print(f"{r['arm']} {r['group']} {r['rule']} {r['threshold_setting']:10s} thr {r['threshold']:5.2f}  recall {r['recall']:.3f}"
              f"  FA {r['false_alert_rate']:.3f}  near {r['near']} timely/late/never {r['near_timely']:.2f}/{r['near_late']:.2f}/{r['near_never']:.2f}"
              f"  first {r['median_first_alert_m']:.2f}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
