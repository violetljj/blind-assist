"""Height-band filtering (as commercial cane-complement products describe it) vs cane-reach rule vs learned head D8.
(EXPLORE, 2026-11-17; arms and reading rules frozen in <work>/PLAN_HEIGHTBAND.md before any result.)
Reuses existing ASE artifacts only (side-view grids, labels v2, calibrated 5-fold D8 scores, insertion runs); no retraining.
Usage:
  ase_cane_heightband_dev_20261117.py run <work>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ase_cane_reach_dev_20261015 as A  # noqa: E402
import ase_cane_sideview_dev_20261015 as V  # noqa: E402
import ase_cane_decision_dev_20261015 as D  # noqa: E402
import ase_cane_insert_dev_20261015 as I  # noqa: E402

HB_T = (.60, .70, 1.10, 1.40)
HBF_T = (.70, 1.10, 1.40)
COL0, COL1 = V.BODY_COLS
ROW_TOP = V.BODY_ROWS[1]


def band_alert(occ, t, free):
    """Nearest occupied cell with height >= t in the body corridor columns; alert if within NEAR (and free below if `free`)."""
    body = occ[int(round(t / V.CELL)):ROW_TOP, COL0:COL1]
    cols = np.flatnonzero(body.any(0))
    if not len(cols):
        return False
    c0 = COL0 + int(cols[0])
    if c0 * V.CELL > A.NEAR:
        return False
    if free:                                                          # same column end as V.decide: ceil((d + MARGIN) / CELL)
        end = int(np.ceil((c0 * V.CELL + A.MARGIN) / V.CELL))
        return not bool(occ[V.BAND_ROWS[0]:V.BAND_ROWS[1], c0:end].any())
    return True


def arms_for(occ_seq):
    out = {}
    for t in HB_T:
        out[f'HB{t:.2f}'] = np.array([band_alert(o, t, False) for o in occ_seq])
    for t in HBF_T:
        out[f'HBF{t:.2f}'] = np.array([band_alert(o, t, True) for o in occ_seq])
    return out


def records(work, k):
    """D.scene_records plus primary-stratum flag in the same frame order."""
    R = D.scene_records(work, k)
    rows = {r['frame']: r for r in json.loads((Path(work) / 'labels_v2' / f'{k}.json').read_text())['rows']}
    R['prim'] = np.array([D.strata(rows[int(fr)]) for fr in R['frames']])
    mono = (R['X'][:, 0] > 0) | (R['X'][:, 1] > 0)
    # scene_records keeps only its G decision; rebuild the kept GT occupancy (same frame filter) for the band arms
    Z = np.load(Path(work) / 'grids' / f'{k}.npz')
    keep = [j for j, fr in enumerate(Z['frames']) if Z['valid'][j] and rows.get(int(fr)) is not None and 'skip' not in rows[int(fr)]]
    gt = Z['Y'][np.array(keep, int)] > 0
    assert len(gt) == len(mono)
    R['arms_mono'] = arms_for(mono)
    R['arms_gt'] = arms_for(gt)
    return R


def frame_tally(recs, alerts):
    items = []
    for k, R in recs.items():
        a = alerts[k]
        for j in np.flatnonzero(R['prim']):
            items.append((k, R['lab'][j], R['inst'][j], bool(a[j])))
    t = D.tally(items)
    t['cane_rate'] = round(t['cane_alert'] / max(t['cane_near'], 1), 4)
    t['neg_rate'] = round(t['neg_alert'] / max(t['neg'], 1), 4)
    return t


def insertion(work, extra_arms):
    """Per-run alerts on insertion grids for band arms (mono / GT), plus existing D8/H/G from score.json."""
    work = Path(work)
    S = json.loads((work / 'insert' / 'score.json').read_text())['runs']
    per = []
    for p in S:
        g = np.load(work / 'insert' / str(p['id']) / 'grid.npz')
        mono = [(x[0] > 0) | (x[1] > 0) for x in g['X']]
        gt = [y > 0 for y in g['Y']]
        q = dict(id=p['id'], kind=p['kind'], lab=p['lab'], d=p['d'], D8=p['D8'], H=p['H'], G=p['G'])
        for name, al in arms_for(mono).items():
            q[f'{name}_mono'] = al.tolist()
        for name, al in arms_for(gt).items():
            q[f'{name}_gt'] = al.tolist()
        per.append(q)

    def summarize(arm):
        out = {}
        for grp, kinds in (('miss', I.MISS_TYPES), ('control', set(I.TYPES) - I.MISS_TYPES)):
            caught = t07 = t10 = disturbed = nnear = 0
            for p in [p for p in per if p['kind'] in kinds]:
                near = [q for q, l in enumerate(p['lab']) if l.endswith('_near')]
                if not near:
                    continue
                nnear += 1
                hit = [q for q in near if p[arm][q]]
                if grp == 'miss':
                    if hit:
                        caught += 1; t07 += p['d'][hit[0]] >= .7; t10 += p['d'][hit[0]] >= 1.
                else:
                    disturbed += bool(hit)
            out[grp] = dict(runs=nnear, caught=caught, timely_07=t07, timely_10=t10) if grp == 'miss' else dict(runs=nnear, disturbed=disturbed)
        return out
    names = ['D8', 'H', 'G'] + [f'{a}_{s}' for a in extra_arms for s in ('mono', 'gt')]
    return {a: summarize(a) for a in names}


def run(work):
    work = Path(work)
    Z = np.load(work / 'decision_event_scores.npz')
    thetas = {f: float(Z[f't{f}']) for f in range(5)}
    recs = {}
    for k in D.TRAIN:
        recs[k] = records(work, k)
        recs[k]['s'] = Z[f's{k}']
        assert len(recs[k]['s']) == len(recs[k]['lab'])
    print('records', len(recs), flush=True)
    fold = {k: k % 5 for k in D.TRAIN}
    d8 = {k: recs[k]['s'] > thetas[fold[k]] for k in D.TRAIN}
    arm_names = list(recs[D.TRAIN[0]]['arms_mono'])
    rep = dict(frame={}, event={})
    alerts = dict(D8=d8, H={k: recs[k]['H'] for k in D.TRAIN}, G={k: recs[k]['G'] for k in D.TRAIN})
    for a in arm_names:
        alerts[f'{a}_mono'] = {k: recs[k]['arms_mono'][a] for k in D.TRAIN}
        alerts[f'{a}_gt'] = {k: recs[k]['arms_gt'][a] for k in D.TRAIN}
    for name, al in alerts.items():
        rep['frame'][name] = frame_tally(recs, al)
        rep['event'][name] = D.event_metrics(recs, al)
        print(f'{name:12s} frame {rep["frame"][name]}\n{"":12s} event {rep["event"][name]}', flush=True)
    # D8 nuisance-matched to each mono band arm (pooled-score thresholds; descriptive)
    allc = np.concatenate([recs[k]['s'] for k in D.TRAIN])
    qs = np.unique(np.quantile(allc, np.linspace(.80, .9995, 160)))
    curve = []
    for q in qs:
        e = D.event_metrics(recs, {k: recs[k]['s'] > q for k in D.TRAIN})
        f = frame_tally(recs, {k: recs[k]['s'] > q for k in D.TRAIN})
        curve.append(dict(thr=float(q), **e, cane_rate=f['cane_rate'], miss_kept=f['miss_kept'], miss_objects_kept=f['miss_objects_kept']))
    rep['curve_D8'] = curve
    rep['matched'] = {}
    for a in ['H'] + [f'{n}_mono' for n in arm_names]:
        nu = rep['event'][a]['nuisance_per_min']
        ok = [c for c in curve if c['nuisance_per_min'] <= nu]
        rep['matched'][a] = dict(arm=rep['event'][a], D8_at_same_nuisance=max(ok, key=lambda c: c['nuisance_per_min']) if ok else None)
        print('matched', a, rep['matched'][a], flush=True)
    rep['insertion'] = insertion(work, arm_names)
    for a, v in rep['insertion'].items():
        print('insert', a, v, flush=True)
    (work / 'heightband_compare.json').write_text(json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)), encoding='utf8')
    print('wrote', work / 'heightband_compare.json')


if __name__ == '__main__':
    dict(run=run)[sys.argv[1]](*sys.argv[2:])
