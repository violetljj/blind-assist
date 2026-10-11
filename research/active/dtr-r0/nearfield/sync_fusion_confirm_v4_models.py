"""v4: visit OOF calibration only, byte-frozen v3 evaluation and whole-tie pAUC."""
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
import sync_fusion_confirm_v3_models as v3
import sync_fusion_v1_train_cal as frozen

ARMS, BANDS = v3.ARMS, v3.BANDS
METHODS = ('D_prime', 'rgb', 'tof', 'or', 'and')
sha, save, load = frozen.sha, frozen.save, v3.load


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def csvout(path, rows):
    with Path(path).open('x', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def dependencies():
    return [dict(path=str(Path(p).resolve()), sha256=sha(p))
            for p in (__file__, v3.__file__, frozen.__file__)]


def protocol(root):
    p, binding = v3.protocol(root)
    if p['scores']['methods'] != list(METHODS):
        raise ValueError('Registered v4 method order mismatch')
    return p, binding


def frozen_train(p):
    path = Path(p['inherited']['v3_train_seal_path'])
    if sha(path) != p['inherited']['v3_train_seal_sha256']:
        raise ValueError('v3 model seal hash mismatch')
    seal = load(path)
    entries = [e for e in seal['model_entries'] if e['method'] == 'D_prime']
    if sorted(e['seed'] for e in entries) != list(frozen.SEEDS):
        raise ValueError('Three frozen v3 D-prime weights required')
    for e in entries:
        if sha(e['path']) != e['sha256']:
            raise ValueError('v3 D-prime weight bytes changed')
    if [(e['seed'], e['sha256']) for e in sorted(entries, key=lambda e:e['seed'])] != [
            (e['seed'], e['sha256']) for e in sorted(p['models']['model_entries'], key=lambda e:e['seed'])]:
        raise ValueError('Registered frozen weight bytes differ from v3 seal')
    return path, dict(seal, model_entries=entries)


def scores(d, seal):
    import joblib
    tof, rgb = d['tof_score'].astype(float), d['rgb_score'].astype(float)
    gate = (tof >= 0) | (rgb >= 0)
    values = np.mean([joblib.load(e['path']).predict_proba(d['X'])[:, 1]
                      for e in sorted(seal['model_entries'], key=lambda e: e['seed'])], axis=0)
    return dict(D_prime=np.where(gate, values, -np.inf), tof=tof, rgb=rgb,
                or_=np.maximum(tof, rgb), and_=np.minimum(tof, rgb))


def method_scores(d, seal):
    s = scores(d, seal); s['or'] = s.pop('or_'); s['and'] = s.pop('and_'); return s


def cache_data(d, y, values, cells=None, origin=None):
    result = {k: d[k] for k in ('arm', 'visit_id', 'frame_id', 'query_id', 'band')}
    result['label'] = y
    if origin is not None:
        result['cal_origin'] = np.asarray(origin)
    for m in METHODS:
        result[m + '_score'] = values[m]
        if cells is not None:
            pred = np.zeros(len(y), bool)
            for c in cells:
                if c['method'] == m and c['status'] == 'ADOPTED':
                    cut = np.inf if c['threshold_kind'] == 'positive_infinity' else c['threshold']
                    take = d['band'] == c['band']; pred[take] = values[m][take] >= cut
            result[m + '_pred'] = pred
    return result


def cohort(d, visits, count):
    if len(visits) != count or len(set(visits)) != count:
        raise ValueError('Registered cohort incomplete')
    if set(map(str, d['visit_id'])) != set(visits):
        raise ValueError('Feature visit identities differ from fixed cohort')
    for visit in visits:
        take = (d['arm'] == ARMS[0]) & (d['visit_id'] == visit)
        if take.sum() != 32 * 27 or len(set(d['frame_id'][take])) != 32:
            raise ValueError('Incomplete native frame/query cohort: ' + visit)


def crossfit(args):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    started = time.monotonic(); check = v3.check_factory(started, args.budget_s)
    p, binding = protocol(args.root); path, seal = frozen_train(p)
    if (args.root / 'eval/eval_open.json').exists():
        raise ValueError('No fitting after eval OPEN')
    if sha(args.features) != p['inherited']['training_features_sha256']:
        raise ValueError('Thirty-visit feature hash mismatch')
    d = v3.feature_data(args.features, p['features']['names'], training=True)
    visits = sorted(set(map(str, d['visit_id'])))
    if len(visits) != 30 or set(visits) != set(seal['visit_ids']):
        raise ValueError('OOF requires original thirty visits')
    folds = [list(map(str, x)) for x in np.array_split(np.random.default_rng(20261011).permutation(visits), 5)]
    out = args.root / 'oof'; out.mkdir(exist_ok=False); entries = []; fold_rows = []
    y = d['y'].astype(np.int8); arm = d['arm']; known = y >= 0
    oof = np.full(len(y), np.nan); gate = (d['tof_score'] >= 0) | (d['rgb_score'] >= 0)
    with threadpool_limits(limits=1):
        for fi, held in enumerate(folds):
            hold = np.isin(d['visit_id'], held); train = known & ~hold
            n = int(train.sum()); weights = np.zeros(len(y)); counts = []
            for a in ARMS:
                take = train & (arm == a); na = int(take.sum())
                if not na: raise ValueError('Empty training arm in OOF')
                weights[take] = n / (6 * na)
                counts.append(dict(arm=a, known=na, weight_each=n / (6 * na), weight_sum=n / 6))
            pred = []
            for seed in frozen.SEEDS:
                check(); model = HistGradientBoostingClassifier(**frozen.HGB, random_state=seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always'); model.fit(d['X'][train], y[train], sample_weight=weights[train])
                target = out / 'models' / f'fold{fi}_D_prime_{seed}.joblib'; target.parent.mkdir(exist_ok=True)
                joblib.dump(model, target); pred.append(model.predict_proba(d['X'][hold])[:, 1])
                entries.append(dict(fold=fi, seed=seed, path=str(target.resolve()), sha256=sha(target),
                                    warnings=[str(w.message) for w in caught]))
            oof[hold] = np.where(gate[hold], np.mean(pred, axis=0), -np.inf)
            fold_rows.append(dict(fold=fi, held_visit_ids=held, train_visit_ids=[v for v in visits if v not in held],
                                  train_known_rows=n, hold_rows=int(hold.sum()), arm_counts=counts))
            print('OOF fold', fi, 'wall_s', time.monotonic() - started, flush=True)
        check(); values = method_scores(d, seal); frozen_values = values['D_prime'].copy(); values['D_prime'] = oof
    if np.isnan(oof).any(): raise ValueError('OOF row assignment incomplete')
    cache = cache_data(d, y, values, origin=['old30_oof'] * len(y)); cache['D_prime_frozen_score'] = frozen_values
    target = out / 'cache.npz'; np.savez_compressed(target, **cache)
    shutil.copyfile(__file__, out / 'executed_models.py')
    save(out / 'oof_seal.json', dict(status='OOF30_COMPLETE_CAL_ONLY', **binding, folds=fold_rows,
        model_entries=entries, frozen_model_entries=seal['model_entries'], dependencies=dependencies(),
        v3_train_seal_path=str(path.resolve()), v3_train_seal_sha256=sha(path),
        training_features_path=str(args.features.resolve()), training_features_sha256=sha(args.features),
        cache_path=str(target.resolve()), cache_sha256=sha(target), visit_ids=visits,
        CPU_wall_s=time.monotonic() - started, GPU_s=0, new_eval_reads=0,
        limitation='Frozen D-prime scores on old30 are in-sample; OOF scores use fold-specific temporary calibrators, not the frozen evaluation ensemble.'))


def distribution(cache):
    rows = []
    for arm in ('POOLED_SIX_ARMS', *ARMS):
        for b, band in enumerate(BANDS):
            take = (cache['band'] == b) & ((cache['arm'] == arm) if arm != 'POOLED_SIX_ARMS' else True)
            for label in ('ALL', 1, 0, -1):
                sub = take & ((cache['label'] == label) if label != 'ALL' else True)
                for source, key in (('OOF', 'D_prime_score'), ('FROZEN_IN_SAMPLE', 'D_prime_frozen_score')):
                    val = cache[key][sub]; finite = val[np.isfinite(val)]
                    row = dict(arm=arm, band=b, band_name=band, label=str(label), score_source=source,
                               rows=len(val), finite_rows=len(finite), disabled_rows=int(np.isneginf(val).sum()))
                    row.update({f'q{round(q*100):02}': float(np.quantile(finite, q)) if len(finite) else None
                                for q in (0, .05, .25, .5, .75, .95, 1)})
                    rows.append(row)
                both = sub & np.isfinite(cache['D_prime_score']) & np.isfinite(cache['D_prime_frozen_score'])
                delta = cache['D_prime_score'][both] - cache['D_prime_frozen_score'][both]
                rows[-1]['paired_OOF_minus_frozen_mean'] = float(delta.mean()) if len(delta) else None
                rows[-2]['paired_OOF_minus_frozen_mean'] = rows[-1]['paired_OOF_minus_frozen_mean']
    return rows


def calibrate(args):
    started = time.monotonic(); p, binding = protocol(args.root); _, train = frozen_train(p)
    if (args.root / 'eval/eval_open.json').exists(): raise ValueError('No calibration after OPEN')
    oof_seal = load(args.root / 'oof/oof_seal.json'); v3.check_dependencies(oof_seal['dependencies'])
    if oof_seal['plan_sha256'] != binding['plan_sha256'] or sha(oof_seal['cache_path']) != oof_seal['cache_sha256']:
        raise ValueError('OOF seal binding changed')
    for e in oof_seal['model_entries']:
        if sha(e['path']) != e['sha256']: raise ValueError('OOF calibrator bytes changed')
    old = npz(oof_seal['cache_path']); d = v3.feature_data(args.features, p['features']['names'])
    y, visits = v3.reference_labels(args.references, d, 'cal'); cohort(d, visits, 4)
    if set(visits) & set(oof_seal['visit_ids']): raise ValueError('Old/new calibration overlap')
    with threadpool_limits(limits=1): values = method_scores(d, train)
    new = cache_data(d, y, values, origin=['new4_frozen'] * len(y))
    cache = {k: np.concatenate([old[k], new[k]]) for k in new}
    out = args.root / 'cal'; out.mkdir(exist_ok=False); cells = []
    for b, name in enumerate(BANDS):
        take = cache['band'] == b
        for m in METHODS:
            ties = frozen.enumerate_ties(cache[m + '_score'][take], cache['label'][take])
            target = out / 'ties' / f'{m}__band{b}.json'; save(target, dict(method=m, band=b, rows=ties))
            cells.append(dict(method=m, band=b, band_name=name,
                **frozen.select(ties, cache['label'][take], p['calibration']['band_targets'][name]),
                all_ties_path=str(target.resolve()), all_ties_sha256=sha(target)))
    cache = cache_data(cache, cache['label'], {m:cache[m + '_score'] for m in METHODS}, cells, cache['cal_origin'])
    target = out / 'cache.npz'; np.savez_compressed(target, **cache)
    csvout(out / 'oof_vs_frozen_distribution.csv', distribution(old))
    newdist = dict(new, D_prime_frozen_score=new['D_prime_score'])
    rows = distribution(newdist)
    for r in rows: r['score_source'] = 'NEW4_FROZEN'; r['paired_OOF_minus_frozen_mean'] = None
    csvout(out / 'new4_frozen_distribution.csv', rows[::2])
    shutil.copyfile(__file__, out / 'executed_models.py')
    save(out / 'cal_seal.json', dict(status='CAL_SEALED_EVAL_NOT_OPENED', **binding, cells=cells,
        dependencies=dependencies(), old_visit_ids=oof_seal['visit_ids'], cal_visit_ids=visits,
        oof_seal_sha256=sha(args.root / 'oof/oof_seal.json'), model_entries=train['model_entries'],
        feature_path=str(args.features.resolve()), feature_sha256=sha(args.features),
        reference_path=str(args.references.resolve()), reference_sha256=sha(args.references),
        cache_path=str(target.resolve()), cache_sha256=sha(target),
        CPU_wall_s=time.monotonic() - started, GPU_s=0, new_eval_reads=0))


def evaluate(args):
    started = time.monotonic(); p, binding = protocol(args.root); _, train = frozen_train(p)
    if not args.open_eval: raise ValueError('--open-eval required')
    cal_path = args.root / 'cal/cal_seal.json'; seal = load(cal_path); v3.check_dependencies(seal['dependencies'])
    if seal['plan_sha256'] != binding['plan_sha256'] or sha(seal['cache_path']) != seal['cache_sha256']:
        raise ValueError('Calibration seal changed')
    for c in seal['cells']:
        if sha(c['all_ties_path']) != c['all_ties_sha256']: raise ValueError('Cal ties changed')
    if sha(args.root / 'oof/oof_seal.json') != seal['oof_seal_sha256']: raise ValueError('OOF seal changed')
    old_cal_path = Path(p['inherited']['v3_cal_seal_path'])
    if sha(old_cal_path) != p['inherited']['v3_cal_seal_sha256']: raise ValueError('v3 cal seal changed')
    old_cells = load(old_cal_path)['cells']
    d = v3.feature_data(args.features, p['features']['names']); visits = list(map(str, args.visits)); cohort(d, visits, 16)
    if set(visits) & (set(seal['old_visit_ids']) | set(seal['cal_visit_ids'])): raise ValueError('Eval/cal overlap')
    out = args.root / 'eval'; out.mkdir(exist_ok=True)
    save(out / 'eval_open.json', dict(**binding, one_shot=True, opened_utc=datetime.now(timezone.utc).isoformat(),
        cal_seal_sha256=sha(cal_path), feature_sha256=sha(args.features), visit_ids=visits,
        reference_path=str(args.references.resolve()), dependencies=dependencies()))
    status = 'FAILED_NO_REOPEN'; error = None
    try:
        y, reference_visits = v3.reference_labels(args.references, d, 'eval')
        if set(reference_visits) != set(visits): raise ValueError('Eval reference visits mismatch')
        with threadpool_limits(limits=1): values = method_scores(d, train)
        cache = cache_data(d, y, values, seal['cells'])
        old = cache_data(d, y, values, old_cells)
        for m in METHODS: cache[m + '_v3_pred'] = old[m + '_pred']
        target = out / 'cache.npz'; np.savez_compressed(target, **cache)
        csvout(out / 'per_query.csv', [{k: cache[k][i].item() for k in cache} for i in range(len(y))])
        save(out / 'cache_seal.json', dict(cache_path=str(target.resolve()), cache_sha256=sha(target),
            per_query_sha256=sha(out / 'per_query.csv'), rows=len(y), visit_ids=visits,
            cal_seal_sha256=sha(cal_path), eval_open_sha256=sha(out / 'eval_open.json')))
        status = 'COMPLETE_CACHE_FROZEN'
    except BaseException as exc:
        error = dict(error=repr(exc), traceback=traceback.format_exc()); raise
    finally:
        save(out / 'eval_terminal.json', dict(status=status, error=error, reference_sha256=sha(args.references),
            CPU_wall_s=time.monotonic() - started, GPU_s=0, reopen_permitted=False))


def tie_counts(s, y, v, nv):
    order = np.flatnonzero(np.isfinite(s)); order = order[np.argsort(-s[order], kind='stable')]
    ends = np.r_[np.flatnonzero(np.diff(s[order])) + 1, len(order)] if len(order) else np.array([], int)
    cuts = np.r_[np.inf, s[order][ends-1]]; counts = []
    for label in (1, 0, -1):
        z = np.zeros((nv, len(order)), np.int32)
        if len(order): z[v[order], np.arange(len(order))] = y[order] == label
        counts.append(np.column_stack((np.zeros(nv, int), np.cumsum(z, axis=1)[:, ends-1])))
    return cuts, counts


def pauc(W, F, P, N, width=.05):
    """Integrate whole-tie budget envelope. Equal-F vertical steps have zero width."""
    W, F = np.atleast_2d(W), np.atleast_2d(F)
    P, N = np.atleast_1d(P), np.atleast_1d(N)
    cap = width * N
    nextF = np.column_stack((F[:, 1:], cap))
    dx = np.maximum(0, np.minimum(nextF, cap[:, None]) - F)
    numerator = (W * dx).sum(1)
    return np.divide(numerator, P * N * width, out=np.full(len(P), np.nan), where=(P > 0) & (N > 0))


def bootstrap_pauc(counts, weights, posden, freeden, check):
    values = np.empty(len(weights))
    for start in range(0, len(weights), 128):
        check(); ww = weights[start:start+128]
        values[start:start+128] = pauc(ww @ counts[0], ww @ counts[1], ww @ posden, ww @ freeden)
    return values


def rate(a, b): return float(a / b) if b else None


def stats(cache, take, m):
    y = cache['label'][take]; pred = cache[m + '_pred'][take]
    W, F, U = [int((pred & (y == label)).sum()) for label in (1, 0, -1)]
    P, N, Q = [int((y == label).sum()) for label in (1, 0, -1)]
    return dict(W=W, F=F, U=U, POS=P, FREE=N, UNKNOWN=Q, W_rate=rate(W, P), F_rate=rate(F, N))


def summarize(args):
    started = time.monotonic(); check = v3.check_factory(started, args.budget_s); p, binding = protocol(args.root)
    es = load(args.root / 'eval/cache_seal.json'); cal_seal = load(args.root / 'cal/cal_seal.json')
    if sha(es['cache_path']) != es['cache_sha256'] or sha(cal_seal['cache_path']) != cal_seal['cache_sha256']:
        raise ValueError('Immutable query cache changed')
    cache, cal = npz(es['cache_path']), npz(cal_seal['cache_path']); visits = es['visit_ids']; nv = len(visits)
    vi = {str(v):i for i,v in enumerate(visits)}; vv = np.array([vi[str(v)] for v in cache['visit_id']])
    draws = np.random.default_rng(20261011).integers(0, nv, (2000, nv))
    weights = np.array([np.bincount(d, minlength=nv) for d in draws], float)
    out = args.root / 'summary'; out.mkdir(exist_ok=False)
    metrics, curves, boots, deploy, paired, drift, driftvisit, visitpauc = [], [], [], [], [], [], [], []
    replicates = {}
    old_v3_cal = npz(Path(p['inherited']['v3_cal_seal_path']).parent / 'cache.npz')
    old_v3_eval = npz(Path(p['inherited']['v3_cal_seal_path']).parent.parent / 'eval/cache.npz')
    old_cuts_on_v4 = dict(cache)
    for m in METHODS: old_cuts_on_v4[m + '_pred'] = cache[m + '_v3_pred']
    for arm in ARMS:
        for b, band in enumerate(BANDS):
            take = (cache['arm'] == arm) & (cache['band'] == b); y = cache['label'][take]; v = vv[take]
            P, N = int((y == 1).sum()), int((y == 0).sum())
            pd = np.bincount(v[y == 1], minlength=nv); nd = np.bincount(v[y == 0], minlength=nv)
            base = dict(arm=arm, band=b, band_name=band, POS=P, FREE=N)
            values, means = {}, {}
            for m in METHODS:
                check(); cuts, counts = tie_counts(cache[m + '_score'][take], y, v, nv)
                W, F, U = [c.sum(0) for c in counts]; point = float(pauc(W, F, P, N)[0]) if P and N else None
                means[m] = point; values[m] = bootstrap_pauc(counts, weights, pd, nd, check)
                replicates[f'{arm}__band{b}__{m}'] = values[m]
                metrics.append(dict(**base, method=m, normalized_pAUC=point,
                                    status='EVALUABLE' if P and N else 'NOT_EVALUABLE'))
                last = np.r_[np.flatnonzero(np.diff(F)), len(F)-1]
                for i in last:
                    curves.append(dict(**base, method=m, F=int(F[i]), W=int(W[i]), U=int(U[i]),
                        F_rate=rate(F[i], N), W_rate=rate(W[i], P), threshold=None if np.isinf(cuts[i]) else float(cuts[i])))
                deploy.append(dict(arm=arm, band=b, band_name=band, method=m, **stats(cache, take, m)))
                for j, visit in enumerate(visits):
                    wp, fp = counts[0][j], counts[1][j]
                    visitpauc.append(dict(arm=arm, band=b, method=m, visit_id=visit, POS=int(pd[j]), FREE=int(nd[j]),
                        normalized_pAUC=float(pauc(wp, fp, pd[j], nd[j])[0]) if pd[j] and nd[j] else None))
                for role, cc in (('v4_pooled_cal', cal), ('v4_old30_oof', cal), ('v4_new4_frozen', cal),
                                 ('v4_eval16', cache), ('v3_cal6', old_v3_cal), ('v3_eval12', old_v3_eval)):
                    mm = (cc['arm'] == arm) & (cc['band'] == b)
                    if role == 'v4_old30_oof': mm &= cc['cal_origin'] == 'old30_oof'
                    if role == 'v4_new4_frozen': mm &= cc['cal_origin'] == 'new4_frozen'
                    drift.append(dict(arm=arm, band=b, method=m, role=role, **stats(cc, mm, m)))
                    for visit in sorted(set(cc['visit_id'][mm])):
                        vm = mm & (cc['visit_id'] == visit)
                        driftvisit.append(dict(arm=arm, band=b, method=m, role=role, visit_id=str(visit), **stats(cc, vm, m)))
                cc = old_cuts_on_v4; mm = (cc['arm'] == arm) & (cc['band'] == b)
                drift.append(dict(arm=arm, band=b, method=m, role='v3_cuts_on_v4_eval16', **stats(cc, mm, m)))
                for visit in sorted(set(cc['visit_id'][mm])):
                    vm = mm & (cc['visit_id'] == visit)
                    driftvisit.append(dict(arm=arm, band=b, method=m, role='v3_cuts_on_v4_eval16', visit_id=str(visit), **stats(cc, vm, m)))
            for m in ('rgb', 'tof', 'or', 'and'):
                d = cache['D_prime_pred'][take]; c = cache[m + '_pred'][take]
                ds, cs = stats(cache, take, 'D_prime'), stats(cache, take, m)
                paired.append(dict(arm=arm, band=b, comparator=m,
                    rescue=int(((y == 1) & d & ~c).sum()), loss=int(((y == 1) & ~d & c).sum()),
                    delta_W=ds['W']-cs['W'], delta_F=ds['F']-cs['F'],
                    FREE_added=int(((y == 0) & d & ~c).sum()), FREE_removed=int(((y == 0) & ~d & c).sum()),
                    deployment_bidirectional_dominance=bool(P and N and ds['W'] >= cs['W'] and ds['F'] <= cs['F'])))
            for comparator in ('rgb', 'or'):
                delta = values['D_prime'] - values[comparator]; valid = delta[np.isfinite(delta)]
                ci = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
                difference = means['D_prime'] - means[comparator] if means['D_prime'] is not None else None
                boots.append(dict(**base, comparator=comparator, delta_pAUC=difference,
                    lower=float(ci[0]) if ci[0] is not None else None, upper=float(ci[1]) if ci[1] is not None else None,
                    valid_replicates=len(valid), total_replicates=2000,
                    pass_lower_gt_zero=bool(ci[0] is not None and ci[0] > 0),
                    all_2000_valid=bool(len(valid) == 2000)))
            print('summary', arm, b, 'wall_s', time.monotonic() - started, flush=True)
    for b in range(3):
        for m in METHODS:
            for role, cc in (('v4_pooled_cal', cal), ('v4_old30_oof', cal), ('v4_new4_frozen', cal),
                             ('v4_eval16', cache), ('v3_cal6', old_v3_cal), ('v3_eval12', old_v3_eval),
                             ('v3_cuts_on_v4_eval16', old_cuts_on_v4)):
                mm = cc['band'] == b
                if role == 'v4_old30_oof': mm &= cc['cal_origin'] == 'old30_oof'
                if role == 'v4_new4_frozen': mm &= cc['cal_origin'] == 'new4_frozen'
                drift.append(dict(arm='POOLED_SIX_ARMS', band=b, method=m, role=role, **stats(cc, mm, m)))
    for name, rows in [('pauc.csv', metrics), ('curves.csv', curves), ('bootstrap.csv', boots),
                       ('deployment_points.csv', deploy), ('deployment_paired.csv', paired),
                       ('drift.csv', drift), ('drift_visit.csv', driftvisit), ('visit_pauc.csv', visitpauc)]:
        csvout(out / name, rows)
    np.savez_compressed(out / 'bootstrap_replicates.npz', draws=draws, **replicates)
    primary = [r for r in boots if r['arm'] == ARMS[0] and r['band'] == 1]
    evaluable = len(visits) == 16 and len(primary) == 2 and all(r['all_2000_valid'] for r in primary)
    passed = evaluable and all(r['pass_lower_gt_zero'] for r in primary)
    conclusion = (p['confirmation']['pass_claim'] if passed else
                  '主判据未通过：' + ', '.join(r['comparator'] for r in primary if not r['pass_lower_gt_zero'])
                  if evaluable else 'NOT_EVALUABLE：主指标分母或有效重抽样不足')
    plot(curves, out / 'low_free_curves.png')
    save(out / 'summary.json', dict(**binding, status='PASS' if passed else 'FAIL' if evaluable else 'NOT_EVALUABLE', conclusion=conclusion,
        primary=primary, pAUC=metrics, deployment=deploy, paired=paired,
        eval_cache_sha256=es['cache_sha256'], cal_cache_sha256=cal_seal['cache_sha256'],
        source_sha256=sha(__file__), CPU_wall_s=time.monotonic()-started, GPU_s=0,
        pAUC_convention='Whole-tie right-continuous budget envelope integrated exactly over [0,.05], normalized by .05; no interpolation or tie splitting.'))


def plot(curves, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ai, arm in enumerate((ARMS[0], ARMS[4], ARMS[5])):
        for b, band in enumerate(BANDS):
            ax = axes[ai, b]
            for m in METHODS:
                rows = [r for r in curves if r['arm'] == arm and r['band'] == b and r['method'] == m]
                if rows[0]['FREE'] and rows[0]['POS']:
                    rows = [r for r in rows if r['F_rate'] <= .05]
                    x = [r['F_rate'] * 100 for r in rows] + [5]
                    z = [r['W_rate'] * 100 for r in rows]; z.append(z[-1])
                    ax.step(x, z, where='post', label=m)
                else:
                    ax.text(.05, .2, 'POS/FREE denominator zero: N/E', transform=ax.transAxes)
            ax.set(title=arm + '\n' + band, xlim=(0, 5), ylim=(0, 102), xlabel='FREE support (%)', ylabel='Witness / POS (%)')
            ax.grid(alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc='upper center', ncol=5)
    fig.suptitle('v4 preregistered whole-tie low-FREE ranking curves', y=.995)
    fig.tight_layout(rect=(0, 0, 1, .945)); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); subs = parser.add_subparsers(dest='stage', required=True)
    for stage in ('crossfit', 'cal', 'eval', 'summarize'):
        sub = subs.add_parser(stage); sub.add_argument('--root', type=Path, required=True)
        sub.add_argument('--budget-s', type=float, default=200)
        if stage != 'summarize': sub.add_argument('--features', type=Path, required=True)
        if stage in ('cal', 'eval'): sub.add_argument('--references', type=Path, required=True)
        if stage == 'eval':
            sub.add_argument('--visits', nargs=16, required=True); sub.add_argument('--open-eval', action='store_true')
    args = parser.parse_args(); args.root = args.root.resolve()
    {'crossfit':crossfit, 'cal':calibrate, 'eval':evaluate, 'summarize':summarize}[args.stage](args)


if __name__ == '__main__': main()
