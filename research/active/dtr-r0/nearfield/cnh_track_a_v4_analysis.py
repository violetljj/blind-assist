"""Track A scale v4 frozen alert-level and descriptive analysis (formal lane).

Inputs: readout arrays of the primary condition (cnh_track_a_scale_gpu_run outputs) and,
for the descriptive reference, per-unit outputs of cnh_memory_visibility_diagnostic and
cnh_signal_ceiling_diagnostic. Alert unit: one (sequence x query box) pair over its
frames t >= 3 (9 frames, 1.8 s at 5 Hz). A pair with no positive frame is empty and
counts as a false-alert pair if any of those frames alerts. Thresholds are chosen on
calib units only (lowest grid value whose calib false-alert rate <= budget) and frozen;
audit reports the realised false-alert rate. Near event: closest in-box witness Z <= 1 m
among its positive frames; timely: first alert while witness Z >= 1 m; never: no alert.

Alert family (Holm, alpha .05, one-sided paired unit bootstrap B=10000, seed 20260927):
  A1  BODY never-warned rate among near events, S2 minus S3 > 0, at budgets 5/10/20%.
  A2  HEAD and BODY timely rate among near events, S2 rule 1/1 minus S2 rule 2/2 > 0,
      at budgets 5/10/20%.
A claim for a hypothesis (and group) needs all three budgets to reject after Holm.
Everything else is descriptive.
"""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from cnh_alert_persistence_diagnostic import alerts
from cnh_track_a_scale_evaluate import holm

GROUPS = (('HEAD', (0, 2, 4)), ('BODY', (1, 3, 5)))
BUDGETS = (.05, .10, .20)
GRID = np.linspace(1., 12., 441)
B, SEED = 10000, 20260927
POLICIES = {'S2 1/1': ('S2', (1, 1)), 'S3 1/1': ('S3', (1, 1)), 'S2 2/2': ('S2', (2, 2))}


def load(readouts):
    seqs = {'calib': [], 'audit': []}
    for p in sorted(Path(readouts).glob('unit*.npz')):
        with np.load(p) as f:
            split = str(f['split'])
            if split not in seqs:
                continue
            cfg, frame = f['config'], f['frame']
            scores = {'S2': f['S2__noisy@0.75'], 'S3': np.maximum(f['S2__noisy@0.75'], f['memory__noisy'])}
            for c in np.unique(cfg):
                m = np.flatnonzero(cfg == c)
                m = m[np.argsort(frame[m])]
                seqs[split].append(dict(unit=int(p.stem[4:]), frame=frame[m], y=f['labels'][m], w=f['witness'][m],
                                        strata=f['strata'][m], s={k: v[m] for k, v in scores.items()}))
    return seqs


def pairs(seqs, arm, rule, boxes, thr):
    """One row per (sequence, query): unit, empty, false, near, alerted, timely, stratum."""
    out = []
    for sq in seqs:
        a = alerts(sq['s'][arm], thr, *rule)
        t = sq['frame'] >= 3
        for q in boxes:
            pos = np.flatnonzero(t & (sq['y'][:, q] == 1))
            if not len(pos):
                out.append(dict(unit=sq['unit'], empty=True, false=bool(a[t, q].any())))
                continue
            hits = pos[a[pos, q]]
            lead = sq['w'][hits[0], q] if len(hits) else np.nan
            out.append(dict(unit=sq['unit'], empty=False, near=bool(np.nanmin(sq['w'][pos, q]) <= 1.), alerted=bool(len(hits)),
                            timely=bool(len(hits) and lead >= 1.), lead=float(lead),
                            stratum=Counter(sq['strata'][pos, q].tolist()).most_common(1)[0][0]))
    return out


def fa_rate(rows):
    e = [r for r in rows if r['empty']]
    return sum(r['false'] for r in e)/max(1, len(e))


def calib_threshold(seqs, arm, rule, boxes, budget):
    lo, hi = 0, len(GRID)-1                     # false-alert rate is non-increasing in the threshold
    if fa_rate(pairs(seqs, arm, rule, boxes, GRID[hi])) > budget:
        return float(GRID[hi])
    while lo < hi:
        mid = (lo+hi)//2
        if fa_rate(pairs(seqs, arm, rule, boxes, GRID[mid])) <= budget:
            hi = mid
        else:
            lo = mid+1
    return float(GRID[lo])


def summary(rows, stratum=None):
    ev = [r for r in rows if not r['empty'] and (stratum is None or r['stratum'] == stratum)]
    near = [r for r in ev if r['near']]
    lead = [r['lead'] for r in ev if r['alerted'] and np.isfinite(r['lead'])]
    out = dict(events=len(ev), event_recall=sum(r['alerted'] for r in ev)/max(1, len(ev)), near=len(near),
               timely=sum(r['timely'] for r in near)/max(1, len(near)),
               late=sum(r['alerted'] and not r['timely'] for r in near)/max(1, len(near)),
               never=sum(not r['alerted'] for r in near)/max(1, len(near)),
               median_first_alert_m=float(np.median(lead)) if lead else None)
    if stratum is None:
        e = [r for r in rows if r['empty']]
        out |= dict(empty_pairs=len(e), false_alert_rate=fa_rate(rows))
    return out


def unit_counts(rows, key, units):
    """Per unit: (numerator, near events) for timely / never."""
    num, den = np.zeros(len(units)), np.zeros(len(units))
    idx = {u: i for i, u in enumerate(units)}
    for r in rows:
        if not r['empty'] and r['near']:
            den[idx[r['unit']]] += 1
            num[idx[r['unit']]] += r['timely'] if key == 'timely' else (not r['alerted'])
    return num, den


def paired_test(a, b, units):
    """One-sided p for pooled rate(a) - rate(b) > 0, resampling audit units."""
    (na, da), (nb, db) = a, b
    draws = np.random.default_rng(SEED).integers(0, len(units), size=(B, len(units)))
    diff = na[draws].sum(1)/np.maximum(da[draws].sum(1), 1)-nb[draws].sum(1)/np.maximum(db[draws].sum(1), 1)
    point = na.sum()/max(1, da.sum())-nb.sum()/max(1, db.sum())
    return dict(delta=float(point), ci95=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                p_one_sided=float((1+np.sum(diff <= 0))/(B+1)))


def alert_family(readouts):
    seqs = load(readouts)
    units = sorted({s['unit'] for s in seqs['audit']})
    rows, thresholds = {}, {}
    for g, boxes in GROUPS:
        for name, (arm, rule) in POLICIES.items():
            for b in BUDGETS:
                thr = calib_threshold(seqs['calib'], arm, rule, boxes, b)
                thresholds[(g, name, b)] = thr
                rows[(g, name, b)] = pairs(seqs['audit'], arm, rule, boxes, thr)
    tests = []
    for b in BUDGETS:
        g = 'BODY'
        t = paired_test(unit_counts(rows[(g, 'S2 1/1', b)], 'never', units), unit_counts(rows[(g, 'S3 1/1', b)], 'never', units), units)
        tests.append(dict(test='A1 never-warned S2 - S3', group=g, budget=b, **t))
    for g, _ in GROUPS:
        for b in BUDGETS:
            t = paired_test(unit_counts(rows[(g, 'S2 1/1', b)], 'timely', units), unit_counts(rows[(g, 'S2 2/2', b)], 'timely', units), units)
            tests.append(dict(test='A2 timely S2 1/1 - S2 2/2', group=g, budget=b, **t))
    for t, p in zip(tests, holm(np.array([t['p_one_sided'] for t in tests]))):
        t['p_holm'], t['reject'] = float(p), bool(p <= .05)
    claims = {}
    for t in tests:
        key = f"{t['test']} / {t['group']}"
        claims[key] = claims.get(key, True) and t['reject']
    desc = {f'{g} | {name} | budget {b:.2f}': dict(threshold=thresholds[(g, name, b)], all=summary(rows[(g, name, b)]),
                                                   strata={s: summary(rows[(g, name, b)], s) for s in ('tiny', 'realistic', 'wide')})
            for g, _ in GROUPS for name in POLICIES for b in BUDGETS}
    return dict(audit_units=len(units), calib_sequences=len(seqs['calib']), audit_sequences=len(seqs['audit']),
                tests=tests, claims=claims, descriptive=desc)


def reference(ceiling_root, result_json):
    """Model-conditional known-template reference (descriptive; no attribution of gaps)."""
    thr = {g: json.loads(Path(result_json).read_text(encoding='utf-8'))['arms']['S2/noisy'][g]['threshold'] for g, _ in GROUPS}
    rows = [r for p in sorted(Path(ceiling_root).glob('unit*.json')) for r in json.loads(p.read_text(encoding='utf-8'))]
    out = {}
    for g, boxes in GROUPS:
        for st in ('tiny', 'realistic', 'wide', 'all'):
            rs = [r for r in rows if r['box'] in boxes and (st == 'all' or r['stratum'] == st)]
            if not rs:
                continue
            z = np.array([r['z_mf4'] for r in rs])
            hit = np.array([r['s2'] for r in rs]) >= thr[g]
            bins = [dict(z_mf4=f'[{a},{b})', n=int(((z >= a) & (z < b)).sum()),
                         S2_recall=float(hit[(z >= a) & (z < b)].mean()) if ((z >= a) & (z < b)).any() else None)
                    for a, b in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, np.inf))]
            out[f'{g}/{st}'] = dict(n=len(rs), S2_recall=float(hit.mean()),
                                    known_template_reference_recall=float(norm.cdf(z-norm.ppf(.95)).mean()),
                                    auc_z_mf4_predicts_S2_hit=float(roc_auc_score(hit, z)) if 0 < hit.sum() < len(hit) else None,
                                    frac_z_mf4_lt2=float(np.mean(z < 2)), by_z_mf4=bins)
    return out


def main(readouts, visibility_root, ceiling_root, out):
    readouts = Path(readouts)
    report = dict(alerts=alert_family(readouts))
    from cnh_memory_visibility_diagnostic import summarize as vis_summary
    s3_head = json.loads((readouts/'result.json').read_text(encoding='utf-8'))['arms']['S3/noisy']['HEAD']['threshold']
    report['visibility'] = vis_summary(visibility_root, readouts, s3_head)
    report['model_conditional_reference'] = reference(ceiling_root, readouts/'result.json')
    Path(out).write_text(json.dumps(report, indent=1, allow_nan=False), encoding='utf-8')
    for t in report['alerts']['tests']:
        print(f"{t['test']:28s} {t['group']} budget {t['budget']:.2f}  delta {t['delta']:+.3f} "
              f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  p_holm {t['p_holm']:.4f}  {'reject' if t['reject'] else '-'}")
    print(json.dumps(report['alerts']['claims'], indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
