"""Independent posthoc audit: no imports from the analysis implementation.

Reads only fixed old train/cal labels, observation features and consumed immutable
v2 eval cache. All outputs are audit receipts in the separate task artifact tree.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[4]
V2 = REPO / 'artifacts.local/work/sync-fusion-confirm-v2-dev-20261011'
ROOT = REPO / 'artifacts.local/work/sync-fusion-evidence-audit-dev-20261011'
PLAN = Path(__file__).with_name('SYNC_FUSION_EVIDENCE_AUDIT_PLAN_DEV_20261011.json')
BANDS = ('0.3-0.8m', '0.8-1.5m', '1.5-3m')
SOURCES = ('native_perturbed', 'faro_rho030_ambient1')
GROUPS = ('both', 'tof_only', 'rgb_only', 'neither')


def load(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(entry):
    assert sha(entry['path']) == entry['sha256']
    with np.load(entry['path'], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def threshold(cell):
    return np.inf if cell['threshold_kind'] == 'positive_infinity' else cell['threshold']


def labels(path, roles):
    result = {}; visits = set()
    states = {'POS': 1, 'POSITIVE': 1, 'FREE': 0,
              'FREE_ON_SAMPLED_RAYS': 0, 'UNKNOWN': -1}
    for row in load(path)['rows']:
        if row.get('role', row.get('split')) not in roles:
            continue
        visits.add(str(row['visit_id']))
        for q in row['queries']:
            if q['state'] == 'FREE_ON_SAMPLED_RAYS':
                assert q['unknown_pixels'] == q['positive_pixels'] == 0
                assert q['free_ray_pixels'] == q['domain_pixels']
            key = (str(row.get('source_id', row.get('frame_id'))), str(q.get('name', q.get('query_id'))))
            assert key not in result
            result[key] = states[q['state']]
    return result, visits


def tie_rows(scores, y):
    """Independent exhaustive ties: direct predicates for every finite cutoff."""
    ts = [np.inf] + sorted(set(scores[np.isfinite(scores)]), reverse=True)
    out = []
    for i, t in enumerate(ts):
        p = scores >= t
        out.append(dict(tie_index=i, threshold_kind='positive_infinity' if np.isposinf(t) else 'finite',
                        threshold=None if np.isposinf(t) else float(t),
                        W=int(sum(p & (y == 1))), F=int(sum(p & (y == 0))), U=int(sum(p & (y == -1)))))
    return out


def close_record(a, b):
    assert set(a) == set(b), (set(a), set(b))
    for k in a:
        assert a[k] is None and b[k] is None or a[k] is not None and b[k] is not None and np.isclose(a[k], b[k], rtol=1e-12, atol=1e-12), (k, a[k], b[k])


def ci(v):
    x = v[np.isfinite(v)]
    return dict(lower=float(np.percentile(x, 2.5)) if len(x) else None,
                upper=float(np.percentile(x, 97.5)) if len(x) else None,
                valid_replicates=len(x), replicates=2000)


def group(t, r):
    return GROUPS[0 if t and r else 1 if t else 2 if r else 3]


def audit(root):
    plan = load(PLAN)
    assert plan['models']['recipe']['seeds'] == [955, 956, 957]
    oldtrain = load(V2 / 'train/train_seal.json')
    oldcal = load(V2 / 'cal/cal_seal.json')
    terminal = load(V2 / 'eval/eval_terminal.json')
    cachepath = V2 / 'eval/per_query.jsonl'
    assert terminal['status'] == 'COMPLETE' and sha(cachepath) == terminal['cache_sha256']
    rows = [json.loads(x) for x in cachepath.read_text('utf8').splitlines()]
    rowmap = {(r['source'], r['frame_id'], r['query_id']): r for r in rows}
    assert len(rowmap) == len(rows) == 12150
    refs_train, trainvis = labels(oldtrain['reference_path'], ('train', 'eval'))
    refs_cal, calvis = labels(oldcal['cal_reference_path'], ('cal',))
    visits = load(V2 / 'eval/eval_open.json')['visit_ids']
    assert len(trainvis) == 10 and len(calvis) == 6 and len(visits) == len(set(visits)) == 12
    assert not (trainvis & calvis or trainvis & set(visits) or calvis & set(visits))
    assert sha(oldtrain['reference_path']) == oldtrain['reference_sha256']
    assert sha(oldcal['cal_reference_path']) == oldcal['cal_reference_sha256']
    opened = load(V2 / 'eval/eval_open.json')
    assert sha(V2 / 'cal/cal_seal.json') == opened['cal_seal_sha256']
    assert sha(V2 / 'feature_manifest.json') == opened['feature_manifest_sha256']
    datasets = {(e['source'], e['role']): features(e) for e in load(V2 / 'feature_manifest.json')['outputs']}
    for source in SOURCES:
        entries = [e for e in oldtrain['feature_entries'] if e['source'] == source]
        assert [e['old_role'] for e in entries] == ['train', 'eval']
        parts = [features(e) for e in entries]
        td = datasets[source, 'train']
        for k in td:
            if k != 'feature_names':
                assert np.array_equal(td[k], np.concatenate([d[k] for d in parts])), (source, k)
        assert list(td['feature_names']) == oldtrain['feature_contracts'][source]
    assert sum(len(datasets[s, 'eval']['X']) for s in SOURCES) == len(rows)
    for source in SOURCES:
        d = datasets[source, 'eval']
        for i, (fid, qid) in enumerate(zip(d['frame_id'], d['query_id'])):
            r = rowmap[source, str(fid), str(qid)]
            assert (r['band'], r['visit_id']) == (str(d['band'][i]), str(d['visit_id'][i]))
            for m in ('tof', 'rgb'):
                raw = d[m + '_score'][i]
                assert (r['scores'][m] is None and np.isneginf(raw)) or raw == r['scores'][m]
            for cell in oldcal['cells']:
                if (cell['source'], cell['band']) != (source, r['band']):
                    continue
                raw = r['scores'][cell['method']]
                raw = -np.inf if raw is None else raw
                assert r['predictions'][cell['cell_id']] == bool(cell['status'] == 'ADOPTED' and raw >= threshold(cell))
    # Output schema-specific checks follow below.
    return check_outputs(root, datasets, refs_train, refs_cal, rows, rowmap, oldcal, visits)


def check_outputs(root, data, yt, yc, rows, rowmap, oldcal, visits):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    seal = load(root / 'cal_seal.json'); summary = load(root / 'summary.json')
    assert seal['plan_sha256'] == summary['plan_sha256'] == sha(PLAN)
    assert summary['phase'] == 'POSTHOC_ONLY' and summary['rows'] == len(rows)
    assert summary['source_sha256'] == sha(Path(__file__).with_name('sync_fusion_evidence_audit.py'))
    assert sorted(visits) == summary['eval_visits']
    assert seal['train_visits'] == load(V2 / 'train/train_seal.json')['visit_ids']
    assert seal['cal_visits'] == oldcal['cal_visit_ids'] and not seal['cal_uses_eval_labels']
    for e in load(root / 'input_seal.json')['files']:
        assert sha(e['path']) == e['sha256']
    scoremap = {}; checked_models = 0; checked_ties = 0
    for source in SOURCES:
        td = data[source, 'train']; names = list(td['feature_names'])
        y = np.array([yt[str(f), str(q)] for f, q in zip(td['frame_id'], td['query_id'])])
        known = y >= 0
        for method, cols in (('prior', [16, 17, 18, 19]), ('a', list(range(16)))):
            entries = [e for e in seal['model_entries'] if (e['source'], e['method']) == (source, method)]
            assert sorted(e['seed'] for e in entries) == [955, 956, 957]
            independent = make_pipeline(StandardScaler(), LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=1000, class_weight=None, random_state=955))
            independent.fit(td['X'][known][:, cols], y[known])
            preds = {'cal': [], 'eval': []}
            for e in entries:
                assert sha(e['path']) == e['sha256']
                assert e['features'] == [names[i] for i in cols] and e['known_train_rows'] == int(known.sum())
                m = joblib.load(e['path']); scaler, logit = list(m.named_steps.values())
                assert scaler.n_samples_seen_ == int(known.sum())
                np.testing.assert_allclose(scaler.mean_, td['X'][known][:, cols].mean(0), atol=1e-12)
                np.testing.assert_allclose(scaler.var_, td['X'][known][:, cols].var(0), atol=1e-12)
                assert logit.C == 1 and logit.penalty == 'l2' and logit.solver == 'lbfgs'
                assert logit.max_iter == 1000 and logit.class_weight is None and logit.random_state == e['seed']
                assert (logit.n_iter_ < 1000).all()
                wantlogit = list(independent.named_steps.values())[1]
                np.testing.assert_allclose(logit.coef_, wantlogit.coef_, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(logit.intercept_, wantlogit.intercept_, rtol=1e-12, atol=1e-12)
                for role in preds:
                    preds[role].append(m.predict_proba(data[source, role]['X'][:, cols])[:, 1])
                checked_models += 1
            for role in preds:
                scoremap[source, role, method] = np.mean(preds[role], axis=0, dtype=np.float64)
        for role in ('cal', 'eval'):
            d = data[source, role]
            gate = ((np.isfinite(d['tof_score']) & (d['tof_score'] >= 0)) |
                    (np.isfinite(d['rgb_score']) & (d['rgb_score'] >= 0)))
            scoremap[source, role, 'b'] = np.where(gate, scoremap[source, role, 'a'], -np.inf)
            with np.load(root / f'{source}_{role}_scores.npz', allow_pickle=False) as saved:
                for k in ('frame_id', 'query_id', 'band'):
                    assert np.array_equal(saved[k], d[k])
                for method in ('prior', 'a', 'b'):
                    np.testing.assert_allclose(saved[method], scoremap[source, role, method], rtol=1e-12, atol=1e-12)
                if role == 'cal':
                    assert np.array_equal(saved['label'], [yc[str(f), str(q)] for f, q in zip(d['frame_id'], d['query_id'])])
    assert checked_models == len(seal['model_entries']) == 12 and len(seal['cells']) == 18
    cellmap = {(c['source'], c['band'], c['method']): c for c in seal['cells']}
    for key, c in cellmap.items():
        source, band, method = key; d = data[source, 'cal']; take = d['band'] == band
        y = np.array([yc[str(f), str(q)] for f, q in zip(d['frame_id'], d['query_id'])])[take]
        scores = scoremap[source, 'cal', method][take]; table = tie_rows(scores, y)
        assert sha(c['ties_path']) == c['ties_sha256']
        saved = load(c['ties_path']); assert len(saved) == len(table)
        for expected, actual in zip(table, saved):
            expected.pop('tie_index')
            assert expected == actual, (key, expected, actual)
            checked_ties += 1
        target = {BANDS[0]: .02, BANDS[1]: .05, BANDS[2]: .1}[band]
        assert c['target'] == target
        np_, nf, nu = [int(sum(y == label)) for label in (1, 0, -1)]
        assert (c['POS'], c['FREE'], c['UNKNOWN']) == (np_, nf, nu)
        eligible = [r for r in table if r['F'] <= int(np.floor(target * nf))]
        chosen = max(eligible, key=lambda r: (r['W'], -r['F'], threshold(r))) if np_ and nf else table[0]
        assert c['status'] == ('NOT_CALIBRATABLE' if not nf else 'NO_CAL_POS' if not np_ else 'ADOPTED')
        assert (c['W'], c['F'], c['U'], threshold(c)) == (chosen['W'], chosen['F'], chosen['U'], threshold(chosen))
    auditrows = [json.loads(x) for x in (root / 'per_query_evidence.jsonl').read_text('utf8').splitlines()]
    auditmap = {(r['source'], r['frame_id'], r['query_id']): r for r in auditrows}
    assert len(auditrows) == len(auditmap) == len(rows) and set(auditmap) == set(rowmap)
    with (root / 'per_query_evidence.csv').open(newline='', encoding='utf8') as f:
        csvrows = list(csv.DictReader(f))
    assert len(csvrows) == len(auditrows)
    for cr, ar in zip(csvrows, auditrows):
        assert set(cr) == set(ar)
        for k, value in ar.items():
            assert cr[k] == ('' if value is None else str(value)), (k, cr[k], value)
    for source in SOURCES:
        d = data[source, 'eval']
        for i, (fid, qid) in enumerate(zip(d['frame_id'], d['query_id'])):
            key = (source, str(fid), str(qid)); orig = rowmap[key]; r = auditmap[key]
            for k in ('source', 'frame_id', 'query_id', 'band', 'visit_id', 'label'):
                assert r[k] == orig[k]
            t = np.isfinite(d['tof_score'][i]) and d['tof_score'][i] >= 0
            rgb = np.isfinite(d['rgb_score'][i]) and d['rgb_score'][i] >= 0
            assert (r['tof_minimum'], r['rgb_minimum'], r['evidence_group']) == (t, rgb, group(t, rgb))
            origpred = {m: orig['predictions'][next(c['cell_id'] for c in oldcal['cells'] if (c['source'], c['band'], c['method']) == (source, r['band'], m))] for m in ('logit', 'tof', 'rgb')}
            assert r['workpoint_group'] == group(origpred['tof'], origpred['rgb'])
            for m in origpred:
                assert r['original_' + m] == origpred[m]
            for m in ('tof', 'rgb'):
                assert r['raw_' + m + '_score'] == orig['scores'][m]
            assert r['tof_support_pixels_Kmean'] == d['X'][i, 2]
            assert r['tof_valid_K_count'] == d['X'][i, 9]
            inactive = 'dav' if r['band'] == BANDS[0] else 'uni'
            marginindex = list(d['feature_names']).index(inactive + '_margin16_m')
            missingindex = list(d['feature_names']).index(inactive + '_margin16_missing')
            assert r['inactive_rgb_minimum'] == (d['X'][i, marginindex] >= 0 and d['X'][i, missingindex] == 0)
            for method in ('prior', 'a', 'b'):
                sc = scoremap[source, 'eval', method][i]; c = cellmap[source, r['band'], method]
                assert r[method + '_score'] is None and np.isneginf(sc) or np.isclose(r[method + '_score'], sc, atol=1e-12, rtol=1e-12)
                assert r[method + '_prediction'] == bool(c['status'] == 'ADOPTED' and sc >= threshold(c))
    vi = {v: i for i, v in enumerate(sorted(visits))}
    draws = np.random.default_rng(20261011).integers(0, 12, size=(2000, 12))
    weights = np.stack([(draws == j).sum(1) for j in range(12)], axis=1)
    metrics = {(r['source'], r['band'], r['method']): r for r in summary['metrics']}
    pairs = {(r['source'], r['band'], r['method']): r for r in summary['paired']}
    decomps = {(r['source'], r['band'], r['category']): r for r in summary['decompositions']}
    assert len(metrics) == 36 and len(pairs) == 24 and len(decomps) == 12
    def clusters(rr, vals):
        out = np.zeros(12, dtype=int)
        for row, value in zip(rr, vals):
            out[vi[row['visit_id']]] += int(value)
        return out
    for source in SOURCES:
        for band in BANDS:
            rr = [r for r in auditrows if (r['source'], r['band']) == (source, band)]
            y = np.array([r['label'] for r in rr]); p = {}
            for method in ('tof', 'rgb', 'logit', 'prior', 'a', 'b'):
                field = 'original_' + method if method in ('tof', 'rgb', 'logit') else method + '_prediction'
                p[method] = np.array([r[field] for r in rr], bool)
                m = metrics[source, band, method]
                for label, num, den in ((1, 'W', 'POS'), (0, 'F', 'FREE'), (-1, 'U', 'UNKNOWN')):
                    assert m[num] == int(sum(p[method] & (y == label))) and m[den] == int(sum(y == label))
                assert m['W_rate'] == (m['W'] / m['POS'] if m['POS'] else None)
                assert m['F_rate'] == (m['F'] / m['FREE'] if m['FREE'] else None)
            best = max(('tof', 'rgb'), key=lambda method: (metrics[source, band, method]['W'], -metrics[source, band, method]['F'], method == 'rgb'))
            bd = weights @ clusters(rr, y == 1)
            for method in ('logit', 'prior', 'a', 'b'):
                pair = pairs[source, band, method]; rescued = p[method] & ~p[best] & (y == 1); lost = ~p[method] & p[best] & (y == 1)
                net = int(sum(rescued) - sum(lost))
                assert pair['best_single'] == best and pair['best_single_W'] == metrics[source, band, best]['W']
                assert (pair['rescue'], pair['loss'], pair['net']) == (int(sum(rescued)), int(sum(lost)), net)
                assert pair['net_rate'] == (net / int(sum(y == 1)) if sum(y == 1) else None)
                assert pair['FREE_added'] == int(sum(p[method] & ~p[best] & (y == 0)))
                assert pair['FREE_removed'] == int(sum(~p[method] & p[best] & (y == 0)))
                drawnet = weights @ clusters(rr, rescued.astype(int) - lost.astype(int))
                close_record(ci(drawnet), pair['net_CI'])
                close_record(ci(np.divide(drawnet, bd, out=np.full(2000, np.nan), where=bd > 0)), pair['net_rate_CI'])
            for category in ('evidence_group', 'workpoint_group'):
                dec = decomps[source, band, category]; assert dec['best_single'] == best
                assert [part['group'] for part in dec['parts']] == list(GROUPS)
                rescued = p['logit'] & ~p[best] & (y == 1); lost = ~p['logit'] & p[best] & (y == 1)
                groups = np.array([r[category] for r in rr])
                for part in dec['parts']:
                    mask = groups == part['group']; assert part['rows'] == int(sum(mask))
                    for label, num, den in ((1, 'W', 'POS'), (0, 'F', 'FREE'), (-1, 'U', 'UNKNOWN')):
                        assert part[num] == int(sum(mask & p['logit'] & (y == label)))
                        assert part[den] == int(sum(mask & (y == label)))
                    assert (part['rescue'], part['loss'], part['net']) == (int(sum(mask & rescued)), int(sum(mask & lost)), int(sum(mask & rescued) - sum(mask & lost)))
                assert dec['evidence_net'] == sum(part['net'] for part in dec['parts'][:3])
                assert dec['neither_net'] == dec['parts'][3]['net']
                assert dec['evidence_net'] + dec['neither_net'] == pairs[source, band, 'logit']['net']
    middle = pairs[SOURCES[0], BANDS[1], 'logit']; assert (middle['rescue'], middle['loss'], middle['net']) == (407, 22, 385)
    bmiddle = pairs[SOURCES[0], BANDS[1], 'b']; lo = bmiddle['net_rate_CI']['lower']
    assert summary['conclusion'] == ('去先验后融合收益仍成立（事后）' if lo is not None and lo > 0 else '去显式先验并门控后未证实融合收益（事后）')
    assert load(root / 'paired.json') == summary['paired'] and load(root / 'decomposition.json') == summary['decompositions']
    return dict(models_checked=checked_models, independently_refitted_recipes=4, cal_cells_checked=18,
                cal_ties_checked=checked_ties, eval_queries_checked=len(auditrows), eval_decisions_checked=3 * len(auditrows),
                metrics_checked=36, paired_rows_checked=24, decompositions_checked=12,
                bootstrap_replicates=2000, visit_clusters=12, FARO_zero_row_visits_retained=True,
                original_middle=middle, gated_middle=bmiddle, middle_evidence=decomps[SOURCES[0], BANDS[1], 'evidence_group'],
                source_cache_sha256=sha(V2 / 'eval/per_query.jsonl'), summary_sha256=sha(root / 'summary.json'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--receipt', default='independent_audit.json'); args = p.parse_args()
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        result = audit(args.root)
    result.update(status='PASS', CPU_wall_s=time.perf_counter() - started, GPU_s=0,
                  auditor_sha256=sha(__file__), plan_sha256=sha(PLAN),
                  protected_access=False, new_eval_reference_reopens=0)
    with (args.root / args.receipt).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
