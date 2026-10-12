"""Frozen v3: six-arm training, mixed calibration, exclusive eval, cache-only summary."""
import argparse
import csv
import json
import shutil
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import sync_fusion_v1_train_cal as frozen

METHODS = ('D_prime', 'B_prime', 'audit_b', 'tof', 'rgb', 'or', 'and')
MODELS = METHODS[:3]
ARMS = ('native_perturbed', 'faro_rho015_ambient1', 'faro_rho060_ambient1',
        'faro_rho030_ambient3', 'faro_rho030_ambient1', 'faro_rho030_ambient10')
BANDS = ('0.3-0.8m', '0.8-1.5m', '1.5-3m')
STATE_MAP = {'POS': 1, 'POSITIVE': 1, 'FREE': 0, 'FREE_ON_SAMPLED_RAYS': 0, 'UNKNOWN': -1}
sha, save = frozen.sha, frozen.save


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def protocol(root):
    plan = load(root / 'PLAN.json'); receipt = load(root / 'protocol_commit.json')
    if receipt['plan_sha256'] != sha(root / 'PLAN.json') or not receipt.get('pushed_before_data'):
        raise ValueError('Committed and pushed protocol binding required')
    return plan, dict(plan_sha256=receipt['plan_sha256'], protocol_commit=receipt['commit'])


def dependencies(include_observations=False):
    paths = [__file__, frozen.__file__]
    if include_observations:
        paths.extend(Path(__file__).with_name(n) for n in
                     ('sync_fusion_confirm_v3_data.py', 'sync_fusion_confirm_v3_observations.py',
                      'sync_fusion_features_v1.py'))
    return [dict(path=str(Path(p).resolve()), sha256=sha(p)) for p in paths]


def check_dependencies(entries):
    for e in entries:
        if sha(e['path']) != e['sha256']:
            raise ValueError('Sealed executed dependency changed: ' + e['path'])


def check_factory(started, budget):
    def check():
        if time.monotonic() - started >= budget - 3:
            raise TimeoutError('Allocated model command-wall budget reached')
    return check


def feature_data(path, names, training=False):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if training:
        if d['X'].shape[1] != 20:
            raise ValueError('Frozen consumed cache must have twenty original features')
        if list(map(str, d['feature_names'][:16])) != names:
            raise ValueError('Consumed first sixteen feature identities changed')
        d['X'] = d['X'][:, :16]
        d['feature_names'] = np.asarray(names)
    if d['X'].shape != (len(d['arm']), 16) or not np.isfinite(d['X']).all():
        raise ValueError('Sixteen finite interval-relative features required')
    if list(map(str, d['feature_names'])) != names:
        raise ValueError('Feature identity/order differs from registered sixteen')
    ids = list(zip(d['arm'].astype(str), d['frame_id'].astype(str), d['query_id'].astype(str)))
    if len(set(ids)) != len(ids) or not set(d['arm'].astype(str)).issubset(ARMS):
        raise ValueError('Duplicate arm/frame/query or unregistered arm')
    if not set(d['band'].tolist()).issubset((0, 1, 2)):
        raise ValueError('Band must be integer 0,1,2')
    for key in ('tof_score', 'rgb_score'):
        if not np.all(np.isfinite(d[key]) | np.isneginf(d[key])):
            raise ValueError('Baseline raw scores must be finite or -inf')
    return d


def cohort(data, visits, role):
    if len(visits) != (6 if role == 'cal' else 12) or len(set(visits)) != len(visits):
        raise ValueError('Incomplete registered new ' + role + ' cohort')
    if not set(data['visit_id'].astype(str)).issubset(visits):
        raise ValueError('Unexpected visit in features')
    native = data['arm'].astype(str) == ARMS[0]
    for v in visits:
        take = native & (data['visit_id'].astype(str) == v)
        if int(take.sum()) != 32 * 27 or len(set(data['frame_id'][take])) != 32:
            raise ValueError('Native observation cohort incomplete: ' + v)


def reference_labels(path, data, role):
    rows = load(path)['rows']; refs = {}; visits = set()
    for r in rows:
        if r.get('role', r.get('split')) != role:
            continue
        visits.add(str(r['visit_id'])); fid = str(r.get('source_id', r.get('frame_id')))
        for q in r['queries']:
            if q['state'] == 'FREE_ON_SAMPLED_RAYS' and (q['unknown_pixels'] != 0 or
                    q['positive_pixels'] != 0 or q['free_ray_pixels'] != q['domain_pixels']):
                raise ValueError('Reference FREE is not strict full sampled-domain FREE')
            key = (fid, str(q.get('name', q.get('query_id'))))
            if key in refs:
                raise ValueError('Duplicate reference query')
            refs[key] = STATE_MAP[q['state']]
    y = np.asarray([refs[(str(f), str(q))] for f, q in zip(data['frame_id'], data['query_id'])], np.int8)
    return y, sorted(visits)


def train(args):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    started = time.monotonic(); check = check_factory(started, args.budget_s)
    p, binding = protocol(args.root)
    if (args.root / 'eval/eval_open.json').exists():
        raise ValueError('Training forbidden after OPEN')
    if sha(args.features) != p['inherited']['training_features_sha256']:
        raise ValueError('Consumed training cache hash mismatch')
    d = feature_data(args.features, p['features']['names'], training=True)
    visits = sorted(set(d['visit_id'].astype(str))); y = d['y'].astype(np.int8)
    if len(visits) != 30 or set(d['arm'].astype(str)) != set(ARMS):
        raise ValueError('Exactly thirty consumed visits and six arms required')
    known = y >= 0; arm = d['arm'].astype(str); n = int(known.sum())
    weights = np.zeros(len(y)); counts = []
    for a in ARMS:
        take = known & (arm == a); na = int(take.sum())
        if not na:
            raise ValueError('Every arm needs known training rows')
        weights[take] = n / (6 * na)
        counts.append(dict(arm=a, known=na, weight_each=float(n / (6 * na)),
                           weight_sum=float(weights[take].sum()), POS=int(((arm == a) & (y == 1)).sum()),
                           FREE=int(((arm == a) & (y == 0)).sum())))
    out = args.root / 'train'; out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / 'executed_models.py'); entries = []
    with threadpool_limits(limits=1):
        for method in MODELS:
            take = known if method != 'audit_b' else known & (arm == ARMS[0])
            X, yy = d['X'][take], y[take]
            w = weights[take] if method != 'audit_b' else np.ones(int(take.sum()))
            for seed in frozen.SEEDS:
                check()
                if method == 'D_prime':
                    model = HistGradientBoostingClassifier(**frozen.HGB, random_state=seed)
                    fit_kwargs = dict(sample_weight=w)
                else:
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., penalty='l2',
                        solver='lbfgs', max_iter=1000, class_weight=None, random_state=seed))
                    fit_kwargs = dict(standardscaler__sample_weight=w, logisticregression__sample_weight=w)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always'); model.fit(X, yy, **fit_kwargs)
                path = out / 'models' / f'{method}_{seed}.joblib'; path.parent.mkdir(exist_ok=True)
                joblib.dump(model, path)
                entries.append(dict(method=method, seed=seed, path=str(path.resolve()), sha256=sha(path),
                                    known_rows=len(yy), warnings=[str(x.message) for x in caught]))
    save(out / 'train_seal.json', dict(status='TRAIN30_COMPLETE', **binding, model_entries=entries,
        feature_names=p['features']['names'], visit_ids=visits, arm_counts=counts,
        training_features_path=str(args.features.resolve()), training_features_sha256=sha(args.features),
        dependencies=dependencies(), CPU_wall_s=time.monotonic() - started, GPU_s=0, new_eval_reads=0))


def verified_train(root, binding):
    path = root / 'train/train_seal.json'; seal = load(path)
    if seal['plan_sha256'] != binding['plan_sha256']:
        raise ValueError('Train protocol mismatch')
    check_dependencies(seal['dependencies'])
    for e in seal['model_entries']:
        if sha(e['path']) != e['sha256']:
            raise ValueError('Frozen model weights changed')
    return path, seal


def scores(data, train_seal):
    import joblib
    tof, rgb = data['tof_score'].astype(float), data['rgb_score'].astype(float)
    gate = (tof >= 0) | (rgb >= 0)
    result = dict(tof=tof, rgb=rgb, or_=np.maximum(tof, rgb), and_=np.minimum(tof, rgb))
    result['or'] = result.pop('or_'); result['and'] = result.pop('and_')
    for m in MODELS:
        entries = [e for e in train_seal['model_entries'] if e['method'] == m]
        if sorted(e['seed'] for e in entries) != list(frozen.SEEDS):
            raise ValueError('Frozen seed ensemble incomplete')
        values = np.mean([joblib.load(e['path']).predict_proba(data['X'])[:, 1] for e in entries], axis=0)
        result[m] = np.where(gate, values, -np.inf)
    return result


def cache_data(data, y, values, cells=None):
    out = {k: data[k] for k in ('arm', 'visit_id', 'frame_id', 'query_id', 'band')}
    out['label'] = y
    out['tof_evidence'] = data['tof_score'] >= 0; out['rgb_evidence'] = data['rgb_score'] >= 0
    out['evidence_group'] = np.select([out['tof_evidence'] & out['rgb_evidence'], out['tof_evidence'],
                                     out['rgb_evidence']], ['both', 'tof_only', 'rgb_only'], 'neither')
    for m in METHODS:
        out[m + '_score'] = values[m]
        if cells is not None:
            pred = np.zeros(len(y), bool)
            for c in cells:
                if c['method'] == m and c['status'] == 'ADOPTED':
                    threshold = np.inf if c['threshold_kind'] == 'positive_infinity' else c['threshold']
                    pred[data['band'] == c['band']] = values[m][data['band'] == c['band']] >= threshold
            out[m + '_pred'] = pred
    return out


def calibrate(args):
    started = time.monotonic(); check = check_factory(started, args.budget_s)
    p, binding = protocol(args.root); train_path, train_seal = verified_train(args.root, binding)
    if (args.root / 'eval/eval_open.json').exists():
        raise ValueError('Calibration forbidden after OPEN')
    d = feature_data(args.features, p['features']['names']); y, visits = reference_labels(args.references, d, 'cal')
    cohort(d, visits, 'cal')
    if set(visits) & set(train_seal['visit_ids']):
        raise ValueError('Calibration overlaps training')
    out = args.root / 'cal'; out.mkdir(exist_ok=False); cells = []
    with threadpool_limits(limits=1):
        values = scores(d, train_seal)
        for b, name in enumerate(BANDS):
            take = d['band'] == b
            for m in METHODS:
                check(); ties = frozen.enumerate_ties(values[m][take], y[take])
                path = out / 'ties' / f'{m}__band{b}.json'
                save(path, dict(method=m, band=b, band_name=name, rows=ties))
                cells.append(dict(method=m, band=b, band_name=name,
                    **frozen.select(ties, y[take], p['calibration']['band_targets'][name]),
                    all_ties_path=str(path.resolve()), all_ties_sha256=sha(path)))
    cache = out / 'cache.npz'; np.savez_compressed(cache, **cache_data(d, y, values, cells))
    shutil.copyfile(__file__, out / 'executed_models.py')
    feature_manifest = args.root / 'features/manifest.json'
    if not feature_manifest.exists():
        feature_manifest = args.root / 'feature_manifest.json'
    save(out / 'cal_seal.json', dict(status='CAL_SEALED_EVAL_NOT_OPENED', **binding, cells=cells,
        cal_visit_ids=visits, model_entries=train_seal['model_entries'], dependencies=dependencies(True),
        train_seal_path=str(train_path.resolve()), train_seal_sha256=sha(train_path),
        feature_path=str(args.features.resolve()), feature_sha256=sha(args.features),
        reference_path=str(args.references.resolve()), reference_sha256=sha(args.references),
        feature_manifest_path=str(feature_manifest.resolve()), feature_manifest_sha256=sha(feature_manifest),
        cache_path=str(cache.resolve()), cache_sha256=sha(cache),
        CPU_wall_s=time.monotonic() - started, GPU_s=0, new_eval_reads=0))


def evaluate(args):
    started = time.monotonic(); check = check_factory(started, args.budget_s)
    p, binding = protocol(args.root)
    if not args.open_eval:
        raise ValueError('Explicit parent OPEN instruction and --open-eval required')
    cal_path = args.root / 'cal/cal_seal.json'; seal = load(cal_path)
    if seal['plan_sha256'] != binding['plan_sha256']:
        raise ValueError('Cal protocol mismatch')
    check_dependencies(seal['dependencies']); train_path, train_seal = verified_train(args.root, binding)
    if sha(train_path) != seal['train_seal_sha256'] or sha(seal['cache_path']) != seal['cache_sha256']:
        raise ValueError('Frozen training/cal cache changed')
    if sha(seal['feature_manifest_path']) != seal['feature_manifest_sha256']:
        raise ValueError('Frozen observation feature manifest changed')
    for c in seal['cells']:
        if sha(c['all_ties_path']) != c['all_ties_sha256']:
            raise ValueError('Frozen calibration ties changed')
    d = feature_data(args.features, p['features']['names']); visits = list(map(str, args.visits))
    cohort(d, visits, 'eval')
    if set(visits) & (set(train_seal['visit_ids']) | set(seal['cal_visit_ids'])):
        raise ValueError('Eval cohort overlaps train/cal')
    out = args.root / 'eval'; out.mkdir(exist_ok=True)
    # Exclusive receipt is created before the sole evaluator reference read.
    save(out / 'eval_open.json', dict(**binding, opened_utc=datetime.now(timezone.utc).isoformat(),
        cal_seal_sha256=sha(cal_path), feature_sha256=sha(args.features),
        visit_ids=visits, reference_path=str(args.references.resolve()), dependencies=dependencies(), one_shot=True))
    status = 'FAILED_NO_REOPEN'; error = None; nrows = 0
    try:
        y, reference_visits = reference_labels(args.references, d, 'eval')
        if set(reference_visits) != set(visits):
            raise ValueError('Eval reference visit mismatch')
        with threadpool_limits(limits=1):
            check(); values = scores(d, train_seal)
        cache = cache_data(d, y, values, seal['cells'])
        path = out / 'cache.npz'; np.savez_compressed(path, **cache)
        with (out / 'per_query.csv').open('x', newline='', encoding='utf8') as f:
            keys = list(cache); writer = csv.writer(f); writer.writerow(keys)
            for i in range(len(y)):
                writer.writerow([cache[k][i].item() for k in keys]); nrows += 1
        save(out / 'cache_seal.json', dict(cache_path=str(path.resolve()), cache_sha256=sha(path),
            per_query_sha256=sha(out / 'per_query.csv'), rows=nrows, visit_ids=visits,
            cal_seal_sha256=sha(cal_path), eval_open_sha256=sha(out / 'eval_open.json')))
        status = 'COMPLETE_CACHE_FROZEN'
    except BaseException as exc:
        error = dict(error=repr(exc), traceback=traceback.format_exc()); raise
    finally:
        save(out / 'eval_terminal.json', dict(status=status, rows_cached=nrows, error=error,
            reference_sha256=sha(args.references), CPU_wall_s=time.monotonic() - started,
            GPU_s=0, reopen_permitted=False))


def interval(values):
    v = np.asarray(values); v = v[np.isfinite(v)]
    return dict(lower=float(np.quantile(v, .025)) if len(v) else None,
                upper=float(np.quantile(v, .975)) if len(v) else None,
                valid_replicates=len(v), total_replicates=2000)


def ratio(a, b):
    return np.divide(a, b, out=np.full(np.shape(a), np.nan, float), where=np.asarray(b) > 0)


def aggregate(cache, cells, visits):
    if len(visits) != 12 or len(set(visits)) != 12:
        raise ValueError('Twelve original eval clusters required')
    draw = np.random.default_rng(20261011).integers(0, 12, (2000, 12))
    vi = {v: i for i, v in enumerate(visits)}
    yall = cache['label']; vall = np.asarray([vi[str(v)] for v in cache['visit_id']], int)
    metrics = []; paired = []; evidence = []; best = []; saturation = []
    for arm in ARMS:
        for b, band_name in enumerate(BANDS):
            take = (cache['arm'] == arm) & (cache['band'] == b); y = yall[take]; v = vall[take]
            den = np.asarray([(y == k).sum() for k in (1, 0, -1)], int)
            bd = np.asarray([[((v == j) & (y == k)).sum() for k in (1, 0, -1)]
                             for j in range(12)], float)[draw].sum(1)
            preds = {}; stats = {}; cs = {c['method']: c for c in cells if c['band'] == b}
            for m in METHODS:
                pred = cache[m + '_pred'][take]; preds[m] = pred
                counts = np.asarray([(pred & (y == k)).sum() for k in (1, 0, -1)], int); stats[m] = counts
                bc = np.asarray([[((v == j) & pred & (y == k)).sum() for k in (1, 0, -1)]
                                 for j in range(12)], float)[draw].sum(1)
                metrics.append(dict(arm=arm, band=b, band_name=band_name, method=m, cal_status=cs[m]['status'],
                    W=int(counts[0]), POS=int(den[0]), F=int(counts[1]), FREE=int(den[1]),
                    U=int(counts[2]), UNKNOWN=int(den[2]), contributing_visits=len(set(v)),
                    W_rate=float(counts[0] / den[0]) if den[0] else None,
                    F_rate=float(counts[1] / den[1]) if den[1] else None,
                    W_rate_CI=interval(ratio(bc[:, 0], bd[:, 0])), F_rate_CI=interval(ratio(bc[:, 1], bd[:, 1]))))
            chosen = max(('tof', 'rgb'), key=lambda m: (stats[m][0], -stats[m][1], m == 'rgb'))
            best.append(dict(arm=arm, band=b, method=chosen))
            for m in MODELS:
                for role, comparator in (('best_single', chosen), ('OR', 'or')):
                    rescue = (y == 1) & preds[m] & ~preds[comparator]
                    loss = (y == 1) & ~preds[m] & preds[comparator]
                    byvisit = np.asarray([(rescue & (v == j)).sum() - (loss & (v == j)).sum()
                                          for j in range(12)], float)
                    diff = byvisit[draw].sum(1); delta = int(rescue.sum() - loss.sum())
                    row = dict(arm=arm, band=b, method=m, comparator=comparator, comparator_role=role,
                        rescue=int(rescue.sum()), loss=int(loss.sum()), delta_W=delta, POS=int(den[0]),
                        delta_W_rate=float(delta / den[0]) if den[0] else None,
                        delta_W_CI=interval(diff), delta_W_rate_CI=interval(ratio(diff, bd[:, 0])))
                    for k, prefix in ((0, 'FREE'), (-1, 'UNKNOWN')):
                        row[prefix + '_added'] = int(((y == k) & preds[m] & ~preds[comparator]).sum())
                        row[prefix + '_removed'] = int(((y == k) & ~preds[m] & preds[comparator]).sum())
                    paired.append(row)
            group = cache['evidence_group'][take]
            for g in ('both', 'tof_only', 'rgb_only', 'neither'):
                gg = group == g
                for m in METHODS:
                    evidence.append(dict(arm=arm, band=b, group=g, method=m,
                        POS=int((gg & (y == 1)).sum()), FREE=int((gg & (y == 0)).sum()),
                        UNKNOWN=int((gg & (y == -1)).sum()), W=int((gg & (y == 1) & preds[m]).sum()),
                        F=int((gg & (y == 0) & preds[m]).sum()), U=int((gg & (y == -1) & preds[m]).sum()),
                        supported=int((gg & preds[m]).sum())))
            if arm == 'faro_rho030_ambient1':
                rate = float(stats['rgb'][0] / den[0]) if den[0] else None
                saturation.append(dict(arm=arm, band=b, RGB_W=int(stats['rgb'][0]), POS=int(den[0]),
                    RGB_W_rate=rate, remaining_POS=int(den[0] - stats['rgb'][0]),
                    saturated_descriptor=rate >= .95 if rate is not None else None))
    def metric(a, b, m):
        return next(r for r in metrics if (r['arm'], r['band'], r['method']) == (a, b, m))
    primary = []
    for m in MODELS:
        rr = metric(ARMS[0], 1, m)
        pp = next(r for r in paired if (r['arm'], r['band'], r['method'], r['comparator_role']) ==
                  (ARMS[0], 1, m, 'best_single'))
        evaluable = rr['cal_status'] == 'ADOPTED' and rr['POS'] > 0 and rr['FREE'] > 0
        criteria = dict(calibrated_and_denominators=evaluable, positive_difference=pp['delta_W'] > 0,
            interval_lower_positive=pp['delta_W_rate_CI']['lower'] is not None and pp['delta_W_rate_CI']['lower'] > 0,
            FREE_tolerance_pass=rr['F_rate'] is not None and rr['F_rate'] <= .075)
        primary.append(dict(**pp, FREE_rate=rr['F_rate'], FREE_tolerance=.075,
                            criteria=criteria, status='PASS' if all(criteria.values()) else
                            ('FAIL' if evaluable else 'NOT_EVALUABLE'), pass_all=all(criteria.values())))
    secondary = []
    for a in ('faro_rho030_ambient1', 'faro_rho030_ambient10'):
        for m in MODELS:
            rr = metric(a, 0, m); rgb = metric(a, 0, 'rgb')
            evaluable = rr['cal_status'] == 'ADOPTED' and rr['POS'] > 0 and rr['FREE'] > 0 and rr['contributing_visits'] > 0
            criteria = dict(calibrated_and_denominators=evaluable, nonnegative_net=rr['W'] - rgb['W'] >= 0,
                            FREE_tolerance_pass=rr['F_rate'] is not None and rr['F_rate'] <= .04)
            secondary.append(dict(arm=a, method=m, delta_W_vs_RGB=rr['W'] - rgb['W'],
                FREE_rate=rr['F_rate'], FREE_tolerance=.04, criteria=criteria,
                status='PASS' if all(criteria.values()) else ('FAIL' if evaluable else 'NOT_EVALUABLE'),
                pass_all=all(criteria.values())))
    dprimary = next(r for r in primary if r['method'] == 'D_prime')
    return dict(metrics=metrics, paired=paired, best_single=best, evidence_four=evidence,
        primary=dprimary, model_primary_comparison=primary, secondary=secondary,
        secondary_D_prime_pass=all(r['pass_all'] for r in secondary if r['method'] == 'D_prime'),
        evidence_gate_neither_supported=sum(r['supported'] for r in evidence if r['method'] in MODELS and r['group'] == 'neither'),
        faro_RGB_saturation=saturation, result='去先验、证据门控的融合在中带得到确认（真实 RGB + 半合成 ToF）'
            if dprimary['pass_all'] else '主中带确认未通过',
        bootstrap=dict(seed=20261011, replicates=2000, visit_ids=visits, unit='visit',
            comparator='fixed full-eval best single; zero-row FARO clusters retained', zero_denominator='omit; valid count retained'),
        witness_contract='Predicted query positive AND POS; not pixel witness or whole-body clearance',
        far_claim='Report only; no claim')


def summarize(args):
    started = time.monotonic(); protocol(args.root)
    out = args.root / 'eval'; seal = load(out / 'cache_seal.json')
    if sha(out / 'cache.npz') != seal['cache_sha256'] or sha(out / 'per_query.csv') != seal['per_query_sha256']:
        raise ValueError('Frozen eval cache changed')
    with np.load(out / 'cache.npz', allow_pickle=False) as z:
        cache = {k: z[k] for k in z.files}
    cells = load(args.root / 'cal/cal_seal.json')['cells']
    summary = aggregate(cache, cells, seal['visit_ids'])
    summary.update(cache_sha256=seal['cache_sha256'], CPU_wall_s=time.monotonic() - started, GPU_s=0,
                   source_sha256=sha(__file__), reference_reads=0, model_reads=0)
    save(out / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='stage', required=True)
    for name in ('train', 'cal', 'eval', 'summarize'):
        p = sub.add_parser(name); p.add_argument('--root', type=Path, required=True)
        p.add_argument('--budget-s', type=float, default=100)
        if name != 'summarize':
            p.add_argument('--features', type=Path, required=True)
        if name in ('cal', 'eval'):
            p.add_argument('--references', type=Path, required=True)
        if name == 'eval':
            p.add_argument('--visits', nargs=12, required=True); p.add_argument('--open-eval', action='store_true')
    args = parser.parse_args(); args.root = args.root.resolve()
    {'train': train, 'cal': calibrate, 'eval': evaluate, 'summarize': summarize}[args.stage](args)


if __name__ == '__main__':
    main()
