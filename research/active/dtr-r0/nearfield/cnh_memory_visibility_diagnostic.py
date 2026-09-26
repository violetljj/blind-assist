"""Why near-field memory helps HEAD far less than BODY (Development diagnostic, fast lane).

For every audit positive that is invisible in the current frame (no ToF ray's first hit
lands on a label-causing object inside the query box), look back through earlier
frames: was that object part (static world, placed in the current Q frame) hit by
any ToF ray within the frozen 8-frame memory window, or only earlier? Records the
best past solid-angle fraction and its range. Evaluator-only oracle; readout scores
are the saved v3 arrays (S3 = max(S2@0.75, memory)).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from cnh_route_sensor import angular_rays
from cnh_track_a_readout import BOXES
import cnh_track_a_scale_evaluate as se
import cnh_track_a_v13_sensor as sensor_module

GROUPS = (('HEAD', [0, 2, 4]), ('BODY', [1, 3, 5]))
DIRS, W = angular_rays(16)
W = W/W.sum()


def box_hits(dist, oid, pose, wq, ids, lo, hi):
    """Solid-angle fraction and median range of rays hitting ids inside [lo, hi] of Q=wq."""
    points = pose[:3, 3]+(DIRS@pose[:3, :3].T)*dist[..., None]
    q = (points-wq[:3, 3])@wq[:3, :3]
    hit = np.isin(oid, ids) & (q >= lo).all(-1) & (q <= hi).all(-1)
    if not hit.any():
        return 0., np.nan
    return float(W[hit].sum()), float(np.median(dist[hit]))


def unit_rows(geometry, sensor, readouts, unit):
    split, records, step = se.unit_records(geometry, sensor, unit, -10, 1)
    if split != 'audit':
        return []
    with np.load(sensor/f'unit{unit:02d}-mount-10-oracle.npz') as f:
        dist_all, oid_all = f['raydistance'], f['object_id']
    with np.load(sensor/f'unit{unit:02d}-mount-10-observations.npz') as f:
        obs_config = f['config']
    with np.load(readouts/f'unit{unit:02d}.npz') as f:
        s2, mem = f['S2__noisy@0.75'], f['memory__noisy']
        index = {(int(c), int(t)): n for n, (c, t) in enumerate(zip(f['config'], f['frame']))}
    out = []
    for rec in records:
        rows = np.flatnonzero(obs_config == rec['config'])[::step]
        for i in range(len(rows)):
            n = index[(rec['config'], i)]
            for q, (lo, hi) in enumerate(BOXES):
                y = int(rec['labels'][i, q])
                row = dict(unit=unit, config=rec['config'], frame=i, box=q, label=y, main=bool(rec['main'][i]),
                           s2=float(s2[n, q]), s3=float(max(s2[n, q], mem[n, q])))
                ids = rec['contributors'][i][q]
                if y == 1 and ids:
                    wq = rec['world_from_Q'][i]
                    now, _ = box_hits(dist_all[rows[i]], oid_all[rows[i]], rec['poses'][i], wq, ids, lo, hi)
                    row['vis_now'] = now
                    if now == 0:
                        best_win, rng_win, best_old, age_last = 0., np.nan, 0., None
                        for j in range(i-1, -1, -1):
                            frac, rng = box_hits(dist_all[rows[j]], oid_all[rows[j]], rec['poses'][j], wq, ids, lo, hi)
                            if frac > 0 and age_last is None:
                                age_last = i-j
                            if i-j <= 7 and frac > best_win:
                                best_win, rng_win = frac, rng
                            if i-j > 7:
                                best_old = max(best_old, frac)
                        row |= dict(vis_window=best_win, range_window=rng_win, vis_older=best_old, frames_since_seen=age_last)
                out.append(row)
    return out


def pooled_ap(rows, group, mask):
    sel = [r for r in rows if r['main'] and r['box'] in dict(GROUPS)[group] and (r['label'] == 0 or mask(r))]
    y = np.array([r['label'] for r in sel])
    if not y.any():
        return None, 0
    return {k: float(average_precision_score(y, [r[k] for r in sel])) for k in ('s2', 's3')}, int(y.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--sensor', type=Path, required=True)
    p.add_argument('--readouts', type=Path, required=True)
    p.add_argument('--units', type=int, nargs='+', required=True)
    p.add_argument('--family', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sensor_module.FAMILY = a.family
    a.out.mkdir(parents=True, exist_ok=True)
    for u in a.units:
        target = a.out/f'unit{u}.json'
        if not target.exists() and (a.readouts/f'unit{u:02d}.npz').exists():
            target.write_text(json.dumps(unit_rows(a.geometry, a.sensor, a.readouts, u)), encoding='utf-8')
        print(u, flush=True)


def attach_scores(rows, readouts):
    """Add the saved memory score and size stratum to every row."""
    cache = {}
    for r in rows:
        if r['unit'] not in cache:
            with np.load(readouts/f"unit{r['unit']:02d}.npz") as f:
                cache[r['unit']] = (f['memory__noisy'], f['strata'],
                                    {(int(c), int(t)): n for n, (c, t) in enumerate(zip(f['config'], f['frame']))})
        mem, strata, index = cache[r['unit']]
        n = index[(r['config'], r['frame'])]
        r['mem'], r['stratum'] = float(mem[n, r['box']]), str(strata[n, r['box']])


def memory_detail(rows, z_ref):
    """Pooled S3 AP ceilings and memory-score distributions per group (descriptive)."""
    def kind(r):
        if r['label'] == 0:
            return 'negative'
        if 'vis_now' not in r:
            return 'other'
        if r['vis_now'] > 0:
            return 'visible'
        return 'seen' if r['vis_window'] > 0 else 'unseen'
    out = {}
    for g, boxes in GROUPS:
        rs = [r for r in rows if r['main'] and r['box'] in boxes and kind(r) != 'other']
        y, s3, mem = (np.array([r[k] for r in rs]) for k in ('label', 's3', 'mem'))
        kinds, strata = np.array([kind(r) for r in rs]), np.array([r['stratum'] for r in rs])
        ceil_seen, ceil_all = s3.copy(), s3.copy()
        ceil_seen[kinds == 'seen'] = np.inf
        ceil_all[(kinds == 'seen') | (kinds == 'unseen')] = np.inf
        big = np.finfo(float).max
        res = dict(pooled_S3_AP=float(average_precision_score(y, s3)),
                   ceiling_seen_ranked_first=float(average_precision_score(y, np.minimum(ceil_seen, big))),
                   ceiling_all_invisible_first=float(average_precision_score(y, np.minimum(ceil_all, big))))
        for k in ('seen', 'visible', 'negative'):
            m = kinds == k
            res[f'memory_{k}'] = dict(n=int(m.sum()), median=float(np.median(mem[m])), frac_above_ref=float(np.mean(mem[m] > z_ref)))
        for s in ('tiny', 'realistic', 'wide'):
            m = (kinds == 'seen') & (strata == s)
            if m.any():
                res[f'memory_seen_{s}'] = dict(n=int(m.sum()), median=float(np.median(mem[m])), frac_above_ref=float(np.mean(mem[m] > z_ref)))
        out[g] = res
    return out


def summarize(root, readouts=None, z_ref=3.79):
    rows = [r for p in sorted(Path(root).glob('unit*.json')) for r in json.loads(p.read_text(encoding='utf-8'))]
    strata = {
        'visible_now': lambda r: r.get('vis_now', 0) > 0,
        'invisible_seen_in_window': lambda r: r.get('vis_now', 1) == 0 and r['vis_window'] > 0,
        'invisible_seen_only_earlier': lambda r: r.get('vis_now', 1) == 0 and r['vis_window'] == 0 and r['vis_older'] > 0,
        'invisible_never_seen': lambda r: r.get('vis_now', 1) == 0 and r['vis_window'] == 0 and r['vis_older'] == 0,
    }
    out = dict(units=len({r['unit'] for r in rows}))
    for g, boxes in GROUPS:
        pos = [r for r in rows if r['main'] and r['box'] in boxes and r['label'] == 1 and 'vis_now' in r]
        out[g] = dict(positives=len(pos))
        for name, m in strata.items():
            ap, n = pooled_ap(rows, g, m)
            out[g][name] = dict(positives=n, AP=ap)
        seen = [r for r in pos if r['vis_now'] == 0 and r['vis_window'] > 0]
        if seen:
            out[g]['seen_in_window_detail'] = dict(
                median_best_fraction=float(np.median([r['vis_window'] for r in seen])),
                median_range_m=float(np.nanmedian([r['range_window'] for r in seen])),
                median_frames_since_seen=float(np.median([r['frames_since_seen'] for r in seen])))
    if readouts is not None:
        attach_scores(rows, Path(readouts))
        out['memory_detail'] = dict(z_ref=z_ref, **memory_detail(rows, z_ref))
    return out


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'summarize':
        print(json.dumps(summarize(*sys.argv[2:4]), indent=1, ensure_ascii=False))
    else:
        main()
