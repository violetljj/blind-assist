"""Pre-registered v2: one old-data refit, new calibration, one-shot evaluation.

The v1 sensor/query feature and model recipes are imported without modification.
New eval labels are accepted only by the explicit, exclusive `eval` command.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import traceback
from datetime import datetime, timezone

import numpy as np
from threadpoolctl import threadpool_limits
import sync_fusion_v1_train_cal as frozen

METHODS = ('logit', 'hgb', 'tof', 'rgb', 'or', 'and')
SOURCES = ('native_perturbed', 'faro_rho030_ambient1')
STATES = ('POSITIVE', 'FREE_ON_SAMPLED_RAYS', 'UNKNOWN')
STATE_MAP = {'POS': 1, 'POSITIVE': 1, 'FREE': 0,
             'FREE_ON_SAMPLED_RAYS': 0, 'UNKNOWN': -1}
BOOTSTRAP_SEED = 20261011
BOOTSTRAP_REPLICATES = 2000


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return frozen.sha(path)


def save(path, data):
    return frozen.save(path, data)


def protocol(root):
    plan = root / 'PLAN.json'
    receipt = load(root / 'protocol_commit.json')
    if not receipt.get('commit') or receipt.get('plan_sha256') != sha(plan):
        raise ValueError('Committed protocol binding required before computation')
    binding = dict(plan_sha256=sha(plan), protocol_commit=receipt['commit'],
                   protocol_receipt_sha256=sha(root / 'protocol_commit.json'))
    resource_receipt = root / 'resource_amendment_receipt.json'
    if resource_receipt.exists():
        resource = load(resource_receipt)
        amendment = Path(resource['path'])
        if not amendment.is_absolute():
            amendment = Path(__file__).resolve().parents[4] / amendment
        if sha(amendment) != resource['sha256'] or not resource['original_plan_unchanged']:
            raise ValueError('Resource amendment provenance mismatch')
        binding.update(resource_amendment_receipt_sha256=sha(resource_receipt),
                       resource_amendment_sha256=resource['sha256'],
                       resource_amendment_commit=resource['commit'])
    return load(plan), binding


def check_factory(started, budget):
    def check():
        if time.monotonic() - started >= budget - 5:
            raise TimeoutError('Allocated CPU command-wall budget reached')
    return check


def source_entries(manifest, source, role):
    if 'sources' in manifest:
        return manifest['sources'][source][role]
    entries = [e for e in manifest['outputs'] if e['source'] == source and e['role'] == role]
    if len(entries) != 1:
        raise ValueError(f'Exactly one feature entry required: {source}/{role}')
    return entries[0]


def labels(path, allowed_roles, loaded_rows=None):
    refs = load(path)['rows'] if loaded_rows is None else loaded_rows
    out = {}; visits = set()
    for row in refs:
        role = row.get('role', row.get('split'))
        if role not in allowed_roles:
            continue
        visits.add(str(row['visit_id']))
        fid = str(row.get('source_id', row.get('frame_id')))
        for q in row['queries']:
            if q['state'] == 'FREE_ON_SAMPLED_RAYS' and (
                    q['unknown_pixels'] != 0 or q['positive_pixels'] != 0 or
                    q['free_ray_pixels'] != q['domain_pixels']):
                raise ValueError('FREE must be strict full query sampled-domain FREE')
            key = (fid, str(q.get('name', q.get('query_id'))))
            if key in out:
                raise ValueError('Duplicate reference query')
            out[key] = STATE_MAP[q['state']]
    return out, sorted(visits)


def require_features(data, names):
    if list(map(str, data['feature_names'])) != names:
        raise ValueError('Frozen v1 feature order changed')
    if len(names) != 20 or not np.isfinite(data['X']).all():
        raise ValueError('Expected twenty finite features with frozen missing encoding')


def native_cohort(data, visits, frames_per_visit=32):
    """Completeness is checked from observation identities, before new labels."""
    for visit in visits:
        take = data['visit_id'].astype(str) == visit
        ids = list(zip(data['frame_id'][take].astype(str), data['query_id'][take].astype(str)))
        frames = {fid for fid, qid in ids}
        if len(ids) != frames_per_visit * 27 or len(frames) != frames_per_visit:
            return False
        if any(sum(fid == x for x, qid in ids) != 27 for fid in frames):
            return False
    return set(map(str, data['visit_id'])) == set(visits)


def train(args):
    started = time.monotonic(); check = check_factory(started, args.budget_s)
    plan, binding = protocol(args.root)
    manifest = load(args.features)
    refs, visits = labels(args.references, ('train', 'eval'))
    if len(visits) != 10:
        raise ValueError('Refit must use exactly original train6 plus consumed eval4')
    out = args.root / 'train'
    if out.exists():
        if not args.resume_mechanical or (out / 'train_seal.json').exists():
            raise FileExistsError('Fixed refit cannot be repeated')
    else:
        out.mkdir(parents=True)
    shutil.copyfile(__file__, out / ('executed_models_resume.py' if args.resume_mechanical else 'executed_models.py'))
    models = []; contracts = {}; entries = []; counts = []
    with threadpool_limits(limits=1):
        for source in SOURCES:
            inputs = [source_entries(manifest, source, role) for role in ('train', 'eval')]
            parts = [frozen.load_features(e) for e in inputs]
            names = plan['features']['names']
            for data, ids in parts:
                require_features(data, names)
                if not set(map(str, data['visit_id'])).issubset(visits):
                    raise ValueError('Feature visit outside fixed old training10')
            ids = sum((p[1] for p in parts), [])
            if len(ids) != len(set(ids)):
                raise ValueError('Refit duplicates frame/query observations')
            X = np.concatenate([p[0]['X'] for p in parts])
            y = np.asarray([refs[key] for key in ids], np.int8)
            existing = [out / 'models' / source / f'{method}_{seed}.joblib'
                        for method in ('hgb', 'logit') for seed in frozen.SEEDS]
            if any(p.exists() for p in existing):
                if not args.resume_mechanical or not all(p.exists() for p in existing):
                    raise ValueError('Partial model ensemble cannot be silently refitted')
                model_entries = [dict(source=source, method=method, seed=seed,
                    path=str(path.resolve()), sha256=sha(path), warnings=[],
                    warnings_receipt='Original fit succeeded before mechanical seal exception; no refit',
                    train_known_rows=int((y >= 0).sum()), reused_without_refit=True)
                    for (method, seed), path in zip(((method, seed) for method in ('hgb', 'logit')
                                                    for seed in frozen.SEEDS), existing)]
            else:
                _, model_entries = frozen.fit_models(X, y, source, out, check)
            models.extend(model_entries); contracts[source] = names
            entries.extend({**e, 'source': source, 'old_role': role} for role, e in zip(('train', 'eval'), inputs))
            counts.append(dict(source=source, rows=len(y), POS=int((y == 1).sum()),
                               FREE=int((y == 0).sum()), UNKNOWN=int((y == -1).sum())))
    save(out / 'train_seal.json', dict(status='OLD_TRAIN10_REFIT_COMPLETE', **binding,
        feature_contracts=contracts, feature_entries=entries, model_entries=models,
        visit_ids=visits, counts=counts, reference_sha256=sha(args.references),
        reference_path=str(args.references.resolve()), feature_manifest_sha256=sha(args.features),
        source_sha256=sha(__file__), frozen_train_recipe_sha256=sha(frozen.__file__),
        seeds=list(frozen.SEEDS), hgb_recipe=frozen.HGB,
        logit_recipe=dict(StandardScaler='old_train10 known labels only', C=1,
            penalty='l2', solver='lbfgs', max_iter=1000, class_weight=None),
        CPU_wall_s=time.monotonic() - started, GPU_s=0, new_eval_reads=0))


def predictions(data, source, train_seal):
    import joblib
    require_features(data, train_seal['feature_contracts'][source])
    scores = {name: np.asarray(data[name + '_score'], np.float64)
              for name in ('tof', 'rgb', 'or', 'and')}
    if len(data['X']) == 0:
        scores.update(logit=np.asarray([], np.float64), hgb=np.asarray([], np.float64))
        return scores
    for method in ('logit', 'hgb'):
        entries = [e for e in train_seal['model_entries'] if e['source'] == source and e['method'] == method]
        if sorted(e['seed'] for e in entries) != list(frozen.SEEDS):
            raise ValueError('Three frozen v1 seeds required')
        scores[method] = np.mean([joblib.load(e['path']).predict_proba(data['X'])[:, 1]
                                 for e in entries], axis=0, dtype=np.float64)
    if not all(np.all(np.isfinite(v) | np.isneginf(v)) for v in scores.values()):
        raise ValueError('Invalid observable score')
    return scores


def verified_train(root, binding):
    seal_path = root / 'train' / 'train_seal.json'; seal = load(seal_path)
    if seal.get('plan_sha256') != binding['plan_sha256']:
        raise ValueError('Refit protocol mismatch')
    for entry in seal['model_entries']:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Frozen refit model hash mismatch')
    return seal_path, seal


def calibrate(args):
    started = time.monotonic(); check = check_factory(started, args.budget_s)
    plan, binding = protocol(args.root); train_path, train_seal = verified_train(args.root, binding)
    manifest = load(args.features)
    if any(r.get('role', r.get('split')) != 'cal' for r in load(args.references)['rows']):
        raise ValueError('New calibration reference file must contain cal only')
    refs, visits = labels(args.references, ('cal',))
    if len(visits) != 6:
        raise ValueError('Six new cal visits required')
    if set(visits) & set(train_seal['visit_ids']):
        raise ValueError('New cal overlaps old train10')
    out = args.root / 'cal'; out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / 'executed_cal.py')
    cells = []; inputs = []
    with threadpool_limits(limits=1):
        for source in SOURCES:
            entry = source_entries(manifest, source, 'cal')
            data, ids = frozen.load_features(entry)
            if not set(map(str, data['visit_id'])).issubset(visits):
                raise ValueError('Cal feature visits outside fixed new cal6')
            if source == 'native_perturbed' and not native_cohort(data, visits):
                raise ValueError('Incomplete primary cal6 observations: NOT_EVALUABLE')
            scores = predictions(data, source, train_seal)
            y = np.asarray([refs[key] for key in ids], np.int8)
            for band in plan['calibration']['band_targets']:
                take = data['band'].astype(str) == band
                for method in METHODS:
                    check(); ties = frozen.enumerate_ties(scores[method][take], y[take])
                    target = plan['calibration']['band_targets'][band]
                    path = out / 'ties' / f'{source}__{method}__{band.replace("/", "_")}.json'
                    save(path, dict(source=source, method=method, band=band, rows=ties))
                    cells.append(dict(cell_id=f'{source}/{method}/{band}/{target:.2f}',
                        source=source, method=method, band=band,
                        **frozen.select(ties, y[take], target),
                        all_ties_path=str(path.resolve()), all_ties_sha256=sha(path)))
            np.savez_compressed(out / f'{source}_scores.npz', **scores,
                frame_id=data['frame_id'], query_id=data['query_id'], band=data['band'])
            inputs.append({**entry, 'source': source})
    save(out / 'cal_seal.json', dict(status='CAL_SEALED_EVAL_NOT_OPENED', **binding,
        train_seal_sha256=sha(train_path), train_seal_path=str(train_path.resolve()),
        model_entries=train_seal['model_entries'], feature_contracts=train_seal['feature_contracts'],
        cells=cells, feature_entries=inputs, cal_visit_ids=visits,
        cal_reference_path=str(args.references.resolve()), cal_reference_sha256=sha(args.references),
        feature_manifest_sha256=sha(args.features), source_sha256=sha(__file__),
        frozen_train_recipe_sha256=sha(frozen.__file__),
        CPU_wall_s=time.monotonic() - started, GPU_s=0, eval_reads=0))


def divide(a, b):
    return np.divide(a, b, out=np.full(np.shape(a), np.nan, float), where=np.asarray(b) > 0)


def interval(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    return dict(lower=float(np.quantile(a, .025)) if len(a) else None,
                upper=float(np.quantile(a, .975)) if len(a) else None,
                valid_replicates=len(a), total_replicates=BOOTSTRAP_REPLICATES)


def aggregate(rows, cells, visits):
    if len(visits) != 12 or len(set(visits)) != 12:
        raise ValueError('Bootstrap must retain twelve original new eval visits')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(visits), size=(BOOTSTRAP_REPLICATES, len(visits)))
    vi = {v: i for i, v in enumerate(visits)}
    metrics = []; paired = []; best = []; decisions = []; saturation = []
    for source, band in sorted({(c['source'], c['band']) for c in cells}):
        rr = [r for r in rows if r['source'] == source and r['band'] == band]
        y = np.asarray([r['label'] for r in rr], np.int8)
        v = np.asarray([vi[r['visit_id']] for r in rr], int)
        den = np.asarray([(y == k).sum() for k in (1, 0, -1)], int)
        bd = np.asarray([[( (v == j) & (y == k)).sum() for k in (1, 0, -1)]
                         for j in range(len(visits))], float)[draw].sum(1)
        preds = {}; stats = {}; cs = {c['method']: c for c in cells if c['source'] == source and c['band'] == band}
        for method in METHODS:
            c = cs[method]; p = np.asarray([r['predictions'][c['cell_id']] for r in rr], bool)
            n = np.asarray([(p & (y == k)).sum() for k in (1, 0, -1)], int)
            bc = np.asarray([[( (v == j) & p & (y == k)).sum() for k in (1, 0, -1)]
                             for j in range(len(visits))], float)[draw].sum(1)
            preds[method] = p; stats[method] = n
            metrics.append(dict(source=source, band=band, method=method, cal_status=c['status'],
                W=int(n[0]), POS=int(den[0]), F=int(n[1]), FREE=int(den[1]), U=int(n[2]), UNKNOWN=int(den[2]),
                W_rate=float(n[0] / den[0]) if den[0] else None,
                F_rate=float(n[1] / den[1]) if den[1] else None,
                W_rate_CI=interval(divide(bc[:, 0], bd[:, 0])), F_rate_CI=interval(divide(bc[:, 1], bd[:, 1]))))
        chosen = max(('tof', 'rgb'), key=lambda m: (stats[m][0], -stats[m][1], m == 'rgb'))
        best.append(dict(source=source, band=band, method=chosen,
                         rule='full-eval frozen-point max W; tie smaller F; tie RGB; identity fixed bootstrap'))
        for method in ('logit', 'hgb'):
            for role, comparator in (('best_single', chosen), ('OR', 'or')):
                rescue = (y == 1) & preds[method] & ~preds[comparator]
                loss = (y == 1) & ~preds[method] & preds[comparator]
                byvisit = np.asarray([((rescue & (v == j)).sum() - (loss & (v == j)).sum())
                                      for j in range(len(visits))], float)
                diff = byvisit[draw].sum(1)
                row = dict(source=source, band=band, method=method, comparator=comparator,
                    comparator_role=role, rescue=int(rescue.sum()), loss=int(loss.sum()),
                    delta_W=int(rescue.sum() - loss.sum()), POS=int(den[0]),
                    delta_W_rate=float((rescue.sum() - loss.sum()) / den[0]) if den[0] else None,
                    delta_W_CI=interval(diff), delta_W_rate_CI=interval(divide(diff, bd[:, 0])))
                for label, prefix in ((0, 'FREE'), (-1, 'UNKNOWN')):
                    row[prefix + '_added'] = int(((y == label) & preds[method] & ~preds[comparator]).sum())
                    row[prefix + '_removed'] = int(((y == label) & ~preds[method] & preds[comparator]).sum())
                paired.append(row)
                if role == 'best_single' and source == 'native_perturbed':
                    f_rate = float(stats[method][1] / den[1]) if den[1] else None
                    target = cs[method]['target'] * 1.5
                    lower = row['delta_W_rate_CI']['lower']
                    components = dict(calibrated=cs[method]['status'] == 'ADOPTED',
                        positive_difference=row['delta_W'] > 0, interval_lower_positive=lower is not None and lower > 0,
                        FREE_tolerance_pass=f_rate is not None and f_rate <= target + 1e-12)
                    decisions.append(dict(method=method, band=band, delta_W=row['delta_W'],
                        delta_W_CI=row['delta_W_CI'], delta_W_rate_CI=row['delta_W_rate_CI'],
                        FREE_rate=f_rate, FREE_tolerance=target, criteria=components,
                        pass_all=all(components.values()), confirmatory_primary=method == 'logit' and band == '0.8-1.5m'))
        if source == 'faro_rho030_ambient1':
            saturation.append(dict(source=source, band=band, POS=int(den[0]),
                RGB_W=int(stats['rgb'][0]), RGB_W_rate=float(stats['rgb'][0] / den[0]) if den[0] else None,
                RGB_unwitnessed_POS=int(den[0] - stats['rgb'][0]), RGB_cal_status=cs['rgb']['status'],
                saturated_descriptor=bool(stats['rgb'][0] / den[0] >= .95)
                    if den[0] > 0 and cs['rgb']['status'] == 'ADOPTED' else None,
                classification='Predeclared purely descriptive saturation at RGB W/POS>=.95'))
    primary = [d for d in decisions if d['confirmatory_primary']]
    if len(primary) != 1:
        raise ValueError('Expected exactly one native logistic middle-band confirmation')
    return dict(metrics=metrics, paired=paired, best_single=best, band_criteria=decisions,
        primary=primary[0], result='query级融合在中带得到确认（真实RGB+半合成ToF）' if primary[0]['pass_all'] else '主中带确认未通过',
        faro_RGB_saturation=saturation, FARO_role='Independent-geometry descriptive only; excluded from confirmation',
        bootstrap=dict(unit='visit', visit_ids=visits, replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED, comparator='fixed full-eval best single identity', zero_denominator='excluded, valid count retained'),
        witness_contract='query alert AND POS; same definition for all methods; not old pixel-overlap W')


def evaluate(args):
    started = time.monotonic(); check = check_factory(started, args.budget_s)
    plan, binding = protocol(args.root)
    if not args.open_eval:
        raise ValueError('Explicit parent OPEN instruction and --open-eval required')
    out = args.root / 'eval'; out.mkdir(exist_ok=True)
    if (out / 'eval_open.json').exists():
        raise FileExistsError('Eval already opened; rerun forbidden')
    cal_path = args.root / 'cal' / 'cal_seal.json'; seal = load(cal_path)
    if seal['plan_sha256'] != binding['plan_sha256']:
        raise ValueError('Cal protocol mismatch')
    if seal.get('resource_amendment_receipt_sha256') != binding.get('resource_amendment_receipt_sha256'):
        raise ValueError('Resource provenance changed after calibration')
    train_path, train_seal = verified_train(args.root, binding)
    if seal['train_seal_sha256'] != sha(train_path):
        raise ValueError('Training seal changed after calibration')
    manifest = load(args.features); entries = {source: source_entries(manifest, source, 'eval') for source in SOURCES}
    for entry in entries.values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Eval feature hash mismatch')
    visits = list(map(str, args.visits))
    if len(visits) != 12 or len(set(visits)) != 12 or set(visits) & (set(train_seal['visit_ids']) | set(seal['cal_visit_ids'])):
        raise ValueError('Twelve distinct new eval visits must be disjoint')
    native_data, _ = frozen.load_features(entries['native_perturbed'])
    require_features(native_data, seal['feature_contracts']['native_perturbed'])
    if not native_cohort(native_data, visits):
        raise ValueError('Incomplete primary eval12 observations: NOT_EVALUABLE; labels unopened')
    save(out / 'eval_open.json', dict(opened_utc=datetime.now(timezone.utc).isoformat(), **binding,
        cal_seal_sha256=sha(cal_path), feature_manifest_sha256=sha(args.features),
        eval_reference_path=str(args.references.resolve()), visit_ids=visits, one_shot=True,
        source_sha256=sha(__file__)))
    cached = []; status = 'FAILED_NO_REOPEN'; error = None
    try:
        reference_rows = load(args.references)['rows']
        if any(r.get('role', r.get('split')) != 'eval' for r in reference_rows):
            raise ValueError('New eval reference file must contain eval only')
        refs, reference_visits = labels(args.references, ('eval',), reference_rows)
        if set(reference_visits) != set(visits):
            raise ValueError('Eval reference visits differ from frozen eval12')
        with threadpool_limits(limits=1), (out / 'per_query.jsonl').open('x', encoding='utf8') as cache:
            for source in SOURCES:
                check(); data, ids = frozen.load_features(entries[source]); scores = predictions(data, source, train_seal)
                cs = [c for c in seal['cells'] if c['source'] == source]
                for i, (fid, qid) in enumerate(ids):
                    band = str(data['band'][i]); visit = str(data['visit_id'][i])
                    if visit not in visits:
                        raise ValueError('Eval feature visit outside frozen list')
                    pred = {}
                    for c in cs:
                        if c['band'] != band:
                            continue
                        cutoff = np.inf if c['threshold_kind'] == 'positive_infinity' else c['threshold']
                        pred[c['cell_id']] = bool(c['status'] == 'ADOPTED' and scores[c['method']][i] >= cutoff)
                    row = dict(source=source, frame_id=fid, query_id=qid, band=band, visit_id=visit,
                        label=int(refs[(fid, qid)]), scores={m: float(scores[m][i]) if np.isfinite(scores[m][i]) else None for m in METHODS}, predictions=pred)
                    cache.write(json.dumps(row, allow_nan=False) + '\n'); cached.append(row)
                cache.flush()
        check(); summary = aggregate(cached, seal['cells'], visits)
        save(out / 'summary.json', summary); status = 'COMPLETE'
    except BaseException as exc:
        error = dict(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save(out / 'eval_terminal.json', dict(status=status, error=error, rows_cached=len(cached),
            cache_sha256=sha(out / 'per_query.jsonl') if (out / 'per_query.jsonl').exists() else None,
            eval_reference_sha256=sha(args.references),
            CPU_wall_s=time.monotonic() - started, GPU_s=0, reopen_permitted=False))


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'cal', 'eval'):
        p = sub.add_parser(name); p.add_argument('--root', type=Path, required=True)
        p.add_argument('--features', type=Path, required=True); p.add_argument('--references', type=Path, required=True)
        p.add_argument('--budget-s', type=float, default=100)
        if name == 'eval':
            p.add_argument('--visits', nargs=12, required=True); p.add_argument('--open-eval', action='store_true')
        if name == 'train':
            p.add_argument('--resume-mechanical', action='store_true')
    args = parser.parse_args(); args.root = args.root.resolve()
    {'train': train, 'cal': calibrate, 'eval': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
