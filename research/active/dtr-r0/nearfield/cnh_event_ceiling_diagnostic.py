"""Are small-target misses late warnings or never warnings? (Development, fast lane.)

Event = (sequence, query) with a positive main frame at t >= 3, as in the v3 evaluator.
Per event: size stratum, closest approach (minimum in-box witness Z over its positive
frames), first-alert witness Z at the frozen calib threshold, and the largest 4-frame
matched-filter ceiling z_mf4 over its visible positive frames (from
cnh_signal_ceiling_diagnostic outputs). Descriptive only; saved v3 scores.
"""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

GROUPS = (('HEAD', (0, 2, 4)), ('BODY', (1, 3, 5)))


def load_events(readouts, ceiling, result):
    thr = {(a, g): result['arms'][a][g]['threshold'] for a in ('S2/noisy', 'S3/noisy') for g, _ in GROUPS}
    zmax = {}
    for p in Path(ceiling).glob('unit*.json'):
        for r in json.loads(p.read_text(encoding='utf-8')):
            if r['frame'] >= 3:
                key = (r['unit'], r['config'], r['box'])
                zmax[key] = max(zmax.get(key, 0.), r['z_mf4'])
    events = []
    for p in sorted(Path(readouts).glob('unit*.npz')):
        with np.load(p) as f:
            if str(f['split']) != 'audit':
                continue
            unit = int(p.stem[4:])
            s2, s3 = f['S2__noisy@0.75'], np.maximum(f['S2__noisy@0.75'], f['memory__noisy'])
            y, main, frame, cfg, wz, strata = f['labels'], f['main'], f['frame'], f['config'], f['witness'], f['strata']
        for c in np.unique(cfg):
            m = (cfg == c) & main & (frame >= 3)
            for g, boxes in GROUPS:
                for q in boxes:
                    pos = np.flatnonzero(m & (y[:, q] == 1))
                    if not len(pos):
                        continue
                    ev = dict(unit=unit, config=int(c), box=q, group=g,
                              stratum=Counter(strata[pos, q].tolist()).most_common(1)[0][0],
                              closest=float(np.nanmin(wz[pos, q])), zmax=zmax.get((unit, int(c), q), 0.))
                    for arm, s in (('S2', s2), ('S3', s3)):
                        hits = pos[s[pos, q] >= thr[(f'{arm}/noisy', g)]]
                        ev[f'{arm}_first'] = float(wz[hits[0], q]) if len(hits) else None
                    events.append(ev)
    return events


def summarize(events, arm):
    out = {}
    for g, _ in GROUPS:
        for st in ('tiny', 'realistic', 'wide', 'all'):
            ev = [e for e in events if e['group'] == g and (st == 'all' or e['stratum'] == st)]
            if not ev:
                continue
            first = [e[f'{arm}_first'] for e in ev]
            alerted = np.array([f is not None for f in first])
            near = np.array([e['closest'] <= 1.0 for e in ev])        # obstacle came within 1 m in the sequence
            timely = np.array([f is not None and f >= 1.0 for f in first])
            zmax = np.array([e['zmax'] for e in ev])
            missed = ~alerted
            out[f'{g}/{st}'] = dict(
                events=len(ev), recall=float(alerted.mean()),
                median_first_alert_m=float(np.median([f for f in first if f is not None])) if alerted.any() else None,
                came_within_1m=int(near.sum()),
                within_1m_timely=float(timely[near].mean()) if near.any() else None,
                within_1m_late=float((alerted & ~timely)[near].mean()) if near.any() else None,
                within_1m_never=float(missed[near].mean()) if near.any() else None,
                missed=int(missed.sum()),
                missed_closest_le_1m=int((missed & near).sum()),
                missed_by_zmax=dict(lt2=int((missed & (zmax < 2)).sum()), z2_5=int((missed & (zmax >= 2) & (zmax < 5)).sum()),
                                    ge5=int((missed & (zmax >= 5)).sum())),
                missed_within_1m_zmax_lt2=int((missed & near & (zmax < 2)).sum()))
    return out


if __name__ == '__main__':
    readouts, ceiling, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    result = json.loads((readouts/'result.json').read_text(encoding='utf-8'))
    events = load_events(readouts, ceiling, result)
    report = {arm: summarize(events, arm) for arm in ('S2', 'S3')}
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
    for arm, r in report.items():
        for k, v in r.items():
            print(arm, k, json.dumps(v, ensure_ascii=False))
