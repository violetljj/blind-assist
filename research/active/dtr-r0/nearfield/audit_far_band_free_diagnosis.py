"""Independent consumed-Development audit. No evaluator/model execution or parent imports."""
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[4]
WORK = REPO / 'artifacts.local/work'
OUT = WORK / 'far-band-free-diagnosis-dev-20261011'
PLAN = Path(__file__).with_name('FAR_BAND_FREE_DIAGNOSIS_PLAN_DEV_20261011.json')
FREE = 'FREE_ON_SAMPLED_RAYS'


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(r):
    return r['dataset'], r['source_id'], r['query_id']


def slab(K, shape, query):
    """Matrix backprojection then intersect each ray with all three box slabs."""
    yy, xx = np.indices(shape)
    pixels = np.stack((xx.ravel(), yy.ravel(), np.ones(xx.size)), axis=0)
    direction = np.linalg.solve(np.asarray(K, dtype=float), pixels).T
    direction /= direction[:, 2:3]
    limits = []
    for axis in range(3):
        d = direction[:, axis]
        nonzero = np.abs(d) > 1e-12
        lower, upper = query['low'][axis], query['high'][axis]
        a = np.full(d.shape, -np.inf)
        b = np.full(d.shape, np.inf)
        a[nonzero] = lower / d[nonzero]
        b[nonzero] = upper / d[nonzero]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        hi[~nonzero & ((lower > 0) | (upper < 0))] = -np.inf
        limits.append((lo, hi))
    entry = np.maximum.reduce([x[0] for x in limits]).reshape(shape)
    exit = np.minimum.reduce([x[1] for x in limits]).reshape(shape)
    domain = entry <= exit + 1e-12
    radial = np.linalg.norm(direction, axis=1).reshape(shape)
    return entry, exit, domain, radial


def rank(z, entry, exit, domain, order=16):
    scores = np.minimum((z-entry)[domain], (exit-z)[domain])
    scores = np.sort(scores[np.isfinite(scores)])[::-1]
    return float(scores[order-1]) if len(scores) >= order else None


def raster_faro(data, target_K, shape):
    """Homogeneous grid-camera -> world -> RGB-camera, independently sorted zbuffer."""
    if not np.isfinite(data['grid_pose']).all() or not np.isfinite(data['rgb_pose']).all():
        return np.full(shape, np.inf)
    z = data['depth']
    y, x = np.nonzero(np.isfinite(z) & (z > 0))
    p = np.stack((x, y, np.ones(len(x))))
    camera = np.linalg.solve(data['K'], p) * z[y, x]
    world = data['grid_pose'] @ np.vstack((camera, np.ones(len(x))))
    target = np.linalg.solve(data['rgb_pose'], world)
    target = target[:, target[2] > 0]
    image = np.asarray(target_K) @ target[:3]
    uv = np.rint(image[:2] / image[2]).astype(int)
    ok = (uv[0] >= 0) & (uv[0] < shape[1]) & (uv[1] >= 0) & (uv[1] < shape[0])
    flat = uv[1, ok]*shape[1] + uv[0, ok]
    distances = target[2, ok]
    order = np.lexsort((distances, flat))
    sorted_flat = flat[order]
    first = np.r_[True, np.diff(sorted_flat) != 0] if len(order) else np.array([], dtype=bool)
    result = np.full(np.prod(shape), np.inf)
    result[sorted_flat[first]] = distances[order][first]
    return result.reshape(shape)


def quantiles(values, prefix):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return {prefix+'_'+suffix: float(np.quantile(v, percentile)) if len(v) else None
            for suffix, percentile in [('p10', .1), ('median', .5), ('p90', .9), ('min', 0), ('max', 1)]}


def depth_fields(z, en, ex, dm, prefix):
    valid = dm & np.isfinite(z) & (z > 0)
    inside = valid & (z >= en) & (z <= ex)
    before = valid & (z < en)
    after = valid & (z > ex)
    return {prefix+'_valid': int(valid.sum()), prefix+'_inside': int(inside.sum()),
            prefix+'_before': int(before.sum()), prefix+'_after': int(after.sum()),
            prefix+'_missing_fraction': 1-float(valid.sum())/dm.sum(),
            **quantiles(z[after], prefix+'_after_z_m'),
            **quantiles((z-ex)[after], prefix+'_after_exit_gap_m')}


def decisions(path):
    result = {}
    for line in Path(path).open(encoding='utf8'):
        row = json.loads(line)
        if row['source'] != 'native_perturbed' or row['band'] != '1.5-3m':
            continue
        cell = {}
        for model in ('logit', 'rgb'):
            values = [v for k, v in row['predictions'].items() if k.split('/')[1] == model]
            assert len(values) == 1
            cell[model] = values[0]
        identity = row['frame_id'], row['query_id']
        assert identity not in result or result[identity] == cell, 'Conflicting repeats'
        result[identity] = cell
    return result


def contexts():
    result = []
    for dataset, source_name, geometry_name, roster_name, decision_name in [
        ('v2_eval', 'sync-fusion-confirm-v2-dev-20261011', 'sync-fusion-confirm-v2-dev-20261011', 'eval_reference_roster.json', 'sync-fusion-confirm-v2-dev-20261011'),
        ('v1_1_all', 'sync-rgb-tof-dataset-v1-dev-20261011', 'sync-rgb-tof-dataset-v1-1-dev-20261011', 'reference_roster.json', 'sync-fusion-v1-dev-20261011')]:
        base = WORK/source_name
        roster = read(base/roster_name)
        predictions = {}
        for model in ('dav2', 'unidepth'):
            predictions[model] = {}
            for suffix in ('predictions.json', 'sealed_eval/predictions.json'):
                for row in read(base/'rgb_inference'/model/suffix)['rows']:
                    predictions[model][row['source_id']] = row
        result.append(dict(dataset=dataset, roster=roster, predictions=predictions,
                           geometry={r['source_id']: r for r in read(WORK/geometry_name/'synthesis_manifest.json')['frames']},
                           decisions=decisions(WORK/decision_name/'eval/per_query.jsonl')))
    return result


def category_fields(r):
    # These fixed diagnostic cuts are independent from every prior evaluator.
    label = r['native_valid'] < 64 or (r['confidence2_fraction'] is not None and r['confidence2_fraction'] < .9) or (r['state'] == FREE and r['faro_inside'] >= 16)
    geometry = r['edge_fraction'] > 0 or r['domain_pixels'] < 64 or r['dav_top16_short_fraction'] > .5
    native_empty = r['native_after'] == r['domain_pixels']
    strong = native_empty and r['confidence2_fraction'] is not None and r['confidence2_fraction'] >= .9 and r['faro_after']/r['domain_pixels'] >= .9 and r['faro_inside'] == 0 and r['dav_inside'] >= 16
    return dict(label_suspect=bool(label), geometry_risk=bool(geometry), model_error_strong=bool(strong),
                model_disagreement_unconfirmed=bool(native_empty and r['dav_inside'] >= 16 and not strong),
                fusion_only=bool(r['logit_support'] and not r['rgb_support']), unresolved=not(label or geometry or strong),
                sparse_native=r['native_valid'] < 64,
                low_confidence=r['confidence2_fraction'] is not None and r['confidence2_fraction'] < .9,
                faro_surface_present=r['faro_inside'] >= 16,
                faro_surface_conflict=r['state'] == FREE and r['faro_inside'] >= 16,
                faro_confirms_empty=r['faro_after']/r['domain_pixels'] >= .9 and r['faro_inside'] == 0)


def run():
    started = time.perf_counter()
    checks = Counter()
    failures = []

    def equal(actual, expected, label):
        checks[label.split(':')[0]] += 1
        good = actual is None and expected is None
        if actual is not None and expected is not None:
            good = bool(np.isclose(actual, expected, rtol=1e-8, atol=1e-9)) if isinstance(expected, (float, np.floating)) else actual == expected
        if not good:
            failures.append(dict(check=label, actual=actual, expected=expected))

    plan = read(PLAN)
    selection = read(OUT/'selection.json')
    equal(selection['plan_sha256'], digest(PLAN), 'identity:plan')
    equal(selection['seed'], 20261011, 'identity:seed')
    contexts_data = contexts()
    inventory = []
    for ctx in contexts_data:
        for fr in ctx['roster']['rows']:
            with np.load(ctx['predictions']['dav2'][fr['source_id']]['path']) as data:
                z = data['depth']
            assert np.isfinite(z).all(), 'DAV NaN/inf can corrupt parent top16 selection'
            for j, q in enumerate(ctx['roster']['queries']):
                if (q['low'][2], q['high'][2]) != (1.5, 3.):
                    continue
                state = fr['queries'][j]['state']
                old = ctx['decisions'].get((fr['source_id'], q['name']))
                en, ex, dm, _ = slab(fr['depth_K'], z.shape, q)
                score = rank(z, en, ex, dm)
                rgb = old['rgb'] if ctx['dataset'] == 'v2_eval' else score is not None and score >= .24403834342956543
                logit = old['logit'] if old else None
                group = 'a' if state == FREE and (rgb or logit) else 'b' if state == FREE else 'c' if state == 'POSITIVE' else 'unknown'
                inventory.append(dict(dataset=ctx['dataset'], visit_id=str(fr['visit_id']), capture=str(fr['capture']), split=fr['role'], source_id=fr['source_id'], query_id=q['name'], query_index=j, state=state, pool_group=group, rgb_support=bool(rgb), logit_support=logit, dav_margin16_m=score))
    equal(len(inventory), len(list(csv.DictReader((OUT/'far_query_inventory.csv').open(encoding='utf8')))), 'selection:inventory_count')
    pool_by_key = {key(r): r for r in inventory}
    equal(len(pool_by_key), len(inventory), 'selection:unique_inventory')
    all_a = {key(r) for r in inventory if r['pool_group'] == 'a'}
    equal({key(r) for r in selection['rows'] if r['group'] == 'a'}, all_a, 'selection:all_a')
    equal(len({key(r) for r in selection['rows']}), len(selection['rows']), 'selection:no_replacement')
    rng = np.random.default_rng(plan['sampling']['seed'])
    expected = [dict(r, group='a') for r in inventory if r['pool_group'] == 'a']
    shortages = []
    for dataset in sorted({r['dataset'] for r in inventory}):
        aa = [r for r in expected if r['dataset'] == dataset]
        targets = Counter(r['visit_id'] for r in aa)
        for group in ('b', 'c'):
            chosen, remaining = [], {}
            for visit in sorted({r['visit_id'] for r in inventory if r['dataset'] == dataset}):
                pool = sorted((r for r in inventory if r['dataset'] == dataset and r['visit_id'] == visit and r['pool_group'] == group), key=lambda r: (r['source_id'], r['query_id']))
                random_order = rng.permutation(len(pool))
                n = min(targets[visit], len(pool))
                chosen.extend(pool[i] for i in random_order[:n])
                remaining[visit] = [pool[i] for i in random_order[n:]]
                if n < targets[visit]:
                    shortages.append(dict(dataset=dataset, group=group, visit_id=visit, target=targets[visit], available=len(pool), shortfall=targets[visit]-n))
            while len(chosen) < len(aa) and any(remaining.values()):
                visit = sorted(remaining, key=lambda v: (len(remaining[v]), v), reverse=True)[0]
                chosen.append(remaining[visit].pop())
            expected.extend(dict(r, group=group) for r in chosen)
    equal({key(r): r['group'] for r in expected}, {key(r): r['group'] for r in selection['rows']}, 'selection:seeded_strata')
    equal(selection['shortages'], shortages, 'selection:shortage_receipt')
    for selected in selection['rows']:
        original = pool_by_key[key(selected)]
        for name in original:
            equal(selected[name], original[name], 'selection:'+name)
    rows = [json.loads(line) for line in (OUT/'per_query_diagnosis.jsonl').open(encoding='utf8')]
    table = {key(r): r for r in rows}
    equal(len(table), len(rows), 'table:unique')
    equal(set(table), {key(r) for r in selection['rows']}, 'table:selection_coverage')
    conf_manifest = read(OUT/'confidence/manifest.json') if (OUT/'confidence/manifest.json').exists() else {'rows': []}
    confidences = {r['source_id']: r for r in conf_manifest['rows'] if r.get('path')}
    byframe = defaultdict(list)
    for r in rows:
        byframe[(r['dataset'], r['source_id'])].append(r)
        for name, value in category_fields(r).items():
            equal(r[name], value, 'rules:'+name)
    frame_checks = []
    for ctx in contexts_data:
        frames = {r['source_id']: r for r in ctx['roster']['rows']}
        for (dataset, sid), selected in byframe.items():
            if dataset != ctx['dataset']:
                continue
            assert time.perf_counter()-started < 200, 'Audit allocation reached'
            fr = frames[sid]
            with np.load(fr['reference_path']) as reference:
                native = reference['depth']
                K = reference['depth_K']
                equal(np.array_equal(np.asarray(fr['depth_K']), K), True, 'raw:intrinsics')
                equal(np.array_equal(reference['map_x'], np.indices(native.shape)[1]), True, 'raw:map_x')
                equal(np.array_equal(reference['map_y'], np.indices(native.shape)[0]), True, 'raw:map_y')
                equal(bool(reference['observed'].all()), True, 'raw:observed')
            raw_native = np.asarray(Image.open(fr['native_depth_path']), dtype=float)/1000.
            equal(bool(np.allclose(native, raw_native, atol=1e-6, rtol=0)), True, 'raw:native_depth')
            with np.load(ctx['geometry'][sid]['geometry_path']) as geometry:
                faro = raster_faro(geometry, K, native.shape)
                transformed = bool(np.isfinite(geometry['grid_pose']).all() and np.isfinite(geometry['rgb_pose']).all())
            rgb = {}
            for model in ('dav2', 'unidepth'):
                with np.load(ctx['predictions'][model][sid]['path']) as data:
                    rgb[model] = data['depth']
            confidence = np.asarray(Image.open(confidences[sid]['path'])) if sid in confidences else None
            if confidence is not None:
                equal(confidence.shape, native.shape, 'raw:confidence_shape')
                equal(set(np.unique(confidence)).issubset({0, 1, 2}), True, 'raw:confidence_values')
                equal(digest(confidences[sid]['path']), confidences[sid]['sha256'], 'raw:confidence_hash')
            for r in selected:
                q = ctx['roster']['queries'][r['query_index']]
                en, ex, dm, radial = slab(K, native.shape, q)
                measured = {'domain_pixels': int(dm.sum())}
                for name, z in [('native', native), ('faro', faro), ('dav', rgb['dav2']), ('uni', rgb['unidepth'])]:
                    measured.update(depth_fields(z, en, ex, dm, name))
                for model, name in [('dav2', 'dav'), ('unidepth', 'uni')]:
                    z = rgb[model]
                    pair = dm & np.isfinite(native) & (native > 0) & np.isfinite(z) & (z > 0)
                    measured.update(quantiles((z-native)[pair], name+'_minus_native_m'))
                    measured.update(quantiles(np.abs(z-native)[pair], name+'_absdiff_native_m'))
                    measured[name+'_margin16_m'] = rank(z, en, ex, dm)
                after = dm & np.isfinite(native) & (native > ex)
                measured.update(quantiles((native*radial)[after], 'native_after_ray_distance_m'))
                physical = (ex-en)*radial
                measured.update(quantiles((ex-en)[dm], 'intersection_axial_m'))
                measured.update(quantiles(physical[dm], 'intersection_ray_m'))
                margin = np.minimum(rgb['dav2'][dm]-en[dm], ex[dm]-rgb['dav2'][dm])
                top = np.argsort(margin)[-16:]
                measured['dav_top16_short_fraction'] = float((physical[dm][top] < .10).mean())
                measured['dav_top16_intersection_median_m'] = float(np.median(physical[dm][top]))
                measured['dav_margin15_minus17_m'] = rank(rgb['dav2'], en, ex, dm, 15)-rank(rgb['dav2'], en, ex, dm, 17)
                y, x = np.where(dm)
                h, w = dm.shape
                measured['edge_fraction'] = float(((x < .05*w) | (x >= .95*w) | (y < .05*h) | (y >= .95*h)).mean())
                measured['top_fraction'] = float((y < .1*h).mean())
                measured['bottom_fraction'] = float((y >= .9*h).mean())
                measured['image_center_x_fraction'] = float(x.mean()/w)
                measured['image_center_y_fraction'] = float(y.mean()/h)
                for v in range(3):
                    measured['confidence'+str(v)+'_pixels'] = int((confidence[dm] == v).sum()) if confidence is not None else None
                measured['confidence2_fraction'] = measured['confidence2_pixels']/dm.sum() if confidence is not None else None
                measured['faro_coverage_fraction'] = measured['faro_valid']/dm.sum()
                for name, value in measured.items():
                    equal(r[name], value, 'measure:'+name)
                equal(r['state'], fr['queries'][r['query_index']]['state'], 'reference:state')
                equal(r['domain_pixels'], fr['queries'][r['query_index']]['domain_pixels'], 'reference:domain')
                if r['state'] == FREE:
                    equal(measured['native_after'], measured['domain_pixels'], 'reference:strict_empty')
            frame_checks.append(dict(dataset=dataset, source_id=sid, queries=len(selected), pose_transform_available=transformed, confidence_available=confidence is not None))
    summary = read(OUT/'summary.json')
    equal(summary['plan_sha256'], digest(PLAN), 'summary:plan')
    equal(summary['selection_sha256'], digest(OUT/'selection.json'), 'summary:selection')
    for group in summary['groups']:
        subgroup = [r for r in rows if r['group'] == group['group'] and (group['dataset'] == 'all' or r['dataset'] == group['dataset'])]
        equal(group['n'], len(subgroup), 'aggregate:n')
        equal(group['visits'], len({r['visit_id'] for r in subgroup}), 'aggregate:visits')
        for category, result in group['categories'].items():
            count = sum(category_fields(r)[category] for r in subgroup)
            equal(result['count'], count, 'aggregate:'+category)
            equal(result['denominator'], len(subgroup), 'aggregate:denominator')
            equal(result['rate'], count/len(subgroup) if subgroup else None, 'aggregate:rate')
    for contrast in summary.get('visit_contrasts', []):
        dataset, control = contrast['dataset'], contrast['comparator']
        byvisit = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r['dataset'] == dataset:
                byvisit[r['visit_id']][r['group']].append(r)
        common = sorted(v for v, groups in byvisit.items() if groups['a'] and groups[control])
        equal(contrast['common_visits'], common, 'contrast:common_visits')
        for category, value in contrast['differences_pp'].items():
            differences = [sum(category_fields(r)[category] for r in byvisit[v]['a'])/len(byvisit[v]['a'])
                           - sum(category_fields(r)[category] for r in byvisit[v][control])/len(byvisit[v][control]) for v in common]
            equal(value, float(np.mean(differences))*100 if differences else None, 'contrast:'+category)
    page = (OUT/'index.html').read_text('utf8')
    panels = [r for r in rows if r['group'] == 'a']
    equal(summary['panels'], len(panels), 'pages:count')
    equal(page.count('<article '), len(panels), 'pages:articles')
    for r in panels:
        path = Path(r['panel_path'])
        equal(path.exists(), True, 'pages:present')
        if path.exists():
            with Image.open(path) as image:
                equal(image.size, (1200, 650), 'pages:dimensions')
                image.verify()
        equal(('panels/'+path.name) in page, True, 'pages:linked')
    for entry in read(OUT/'input_hashes.json')['files']:
        equal(digest(entry['path']), entry['sha256'], 'provenance:input_hash')
    result = dict(status='PASS' if not failures else 'FAIL', independent=True,
                  scope='Full selected raw-frame recomputation; no parent rays/rules/reproject imports, no model forward, no protected 480/test access',
                  inventory_queries=len(inventory), selected_queries=len(rows), raw_frames=len(frame_checks), panels=len(panels),
                  plan_sha256=digest(PLAN), script_sha256=digest(__file__), checks=dict(checks), failure_count=len(failures), failures=failures[:50],
                  observed_repair='FARO-inside >=16 is a surface conflict only for FREE; POSITIVE inside surfaces are compatible. Final semantics audited; no change to a/b sampling or support cuts.',
                  raw_frame_receipts=frame_checks, command_wall_s=time.perf_counter()-started, GPU_s=0)
    (OUT/'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: sorted(x) if isinstance(x, set) else str(x))+'\n', encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('raw_frame_receipts', 'failures')}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
