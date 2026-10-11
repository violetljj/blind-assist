"""Independent posthoc audit, reading sealed v3 caches without fitting/inference.

This implementation does not import the primary posthoc analysis. Whole-score
ties are accumulated independently in score order and by visit multiplicity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

ARMS = ('native_perturbed', 'faro_rho030_ambient1', 'faro_rho030_ambient10')
METHODS = ('D_prime', 'B_prime', 'audit_b', 'rgb', 'tof', 'or', 'and')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def equal(a, b, where):
    if b is None:
        assert a in (None, '', 'nan', 'None'), (where, a, b)
    elif isinstance(b, (int, float, np.number)):
        assert np.isclose(float(a), b, atol=1e-9, rtol=1e-10, equal_nan=True), (where, a, b)
    else:
        assert str(a) == str(b), (where, a, b)


def sorted_ties(scores, labels, visits=None, roster=None):
    """Return all whole ties, including +inf disabled; unavailable -inf excluded."""
    assert not np.isnan(scores).any() and not np.isposinf(scores).any()
    order = np.flatnonzero(np.isfinite(scores))
    order = order[np.argsort(scores[order], kind='stable')[::-1]]
    s, y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]) + 1, len(s)] if len(s) else np.array([], int)
    thresholds = np.r_[np.inf, s[ends - 1]] if len(s) else np.array([np.inf])
    counts = np.zeros((len(ends) + 1, 3), int)
    for j, label in enumerate((1, 0, -1)):
        if len(ends):
            counts[1:, j] = np.cumsum(y == label)[ends - 1]
    clusters = None
    if visits is not None:
        v = visits[order]
        clusters = np.zeros((len(ends) + 1, len(roster), 2), int)
        for i, visit in enumerate(roster):
            for j, label in enumerate((1, 0)):
                if len(ends):
                    clusters[1:, i, j] = np.cumsum((v == visit) & (y == label))[ends - 1]
    return thresholds, counts, clusters


def exact_curve(thresholds, counts):
    # Sort each exact F by descending W and threshold; first is its maximum.
    index = np.lexsort((-thresholds, -counts[:, 0], counts[:, 1]))
    f = counts[index, 1]
    return index[np.r_[True, f[1:] != f[:-1]]]


def select(thresholds, counts, cap):
    index = exact_curve(thresholds, counts)
    eligible = index[counts[index, 1] <= cap]
    return int(eligible[-1])


def weighted_bootstrap(clusters, multiplicities, free_den, pos_den, rate):
    result = np.full((len(multiplicities), 4), np.nan)
    # Every draw recomputes weighted cumulative W/F and the closest reachable F.
    # Subsequent maximum W is equivalent to selecting the last admissible tie;
    # recover the highest threshold among its same-W plateau.
    for k, m in enumerate(multiplicities):
        nf, np_ = int(m @ free_den), int(m @ pos_den)
        if not nf or not np_:
            continue
        wf = np.einsum('tvc,v->tc', clusters, m, optimize=False)
        cap = int(np.floor(rate * nf))
        eligible = np.flatnonzero(wf[:, 1] <= cap)
        i = eligible[-1]
        max_f, max_w = wf[i, 1], wf[i, 0]
        i = np.flatnonzero((wf[:, 1] == max_f) & (wf[:, 0] == max_w))[0]
        result[k] = (max_w, max_f, nf, np_)
    return result


def key(row):
    return row['arm'], int(row['band']), row['method']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('artifacts.local/work/sync-fusion-confirm-v3-dev-20261011'))
    ap.add_argument('--output', type=Path, default=Path('artifacts.local/work/sync-fusion-v3-matched-free-posthoc-dev-20261011'))
    ap.add_argument('--receipt', type=Path, default=Path(__file__).with_name('SYNC_FUSION_V3_MATCHED_FREE_POSTHOC_INDEPENDENT_AUDIT_DEV_20261011.json'))
    args = ap.parse_args()
    started = time.perf_counter()
    source, out = args.source, args.output
    plan = read_json(Path(__file__).with_name('SYNC_FUSION_V3_MATCHED_FREE_POSTHOC_PLAN_DEV_20261011.json'))
    assert read_json(out / 'PLAN.json') == plan
    assert plan['bootstrap']['seed'] == 20261011 and plan['bootstrap']['replicates'] == 2000
    seal = read_json(source / 'eval/cache_seal.json')
    assert digest(source / 'eval/cache.npz') == seal['cache_sha256']
    cal_seal = read_json(source / 'cal/cal_seal.json')
    assert digest(source / 'cal/cal_seal.json') == seal['cal_seal_sha256']
    assert digest(source / 'cal/cache.npz') == cal_seal['cache_sha256']
    data = read_npz(source / 'eval/cache.npz')
    cal = read_npz(source / 'cal/cache.npz')
    features = read_npz(source / 'features/eval.npz')
    roster = seal['visit_ids']
    draws = np.random.default_rng(20261011).integers(0, 12, (2000, 12))
    multiplicities = np.column_stack([(draws == i).sum(1) for i in range(12)])
    cell = {}
    for arm in ARMS:
        for band in range(3):
            mask = (data['arm'] == arm) & (data['band'] == band)
            y, v = data['label'][mask], data['visit_id'][mask]
            den = np.array([(y == z).sum() for z in (1, 0, -1)], int)
            fd = np.array([((v == vi) & (y == 0)).sum() for vi in roster], int)
            pd = np.array([((v == vi) & (y == 1)).sum() for vi in roster], int)
            for method in METHODS:
                thresholds, counts, clusters = sorted_ties(data[method + '_score'][mask], y, v, roster)
                cell[arm, band, method] = dict(mask=mask, y=y, v=v, den=den, fd=fd, pd=pd,
                                               thresholds=thresholds, counts=counts, clusters=clusters)
    # Output schema-specific checks follow below; no primary implementation import.
    audit_outputs(args, data, cal, features, cal_seal, cell, roster, draws, multiplicities)
    receipt = dict(status='PASS', phase='INDEPENDENT_POSTHOC_ORACLE_DESCRIPTION',
                   independent_implementation=True, source_eval_cache_sha256=seal['cache_sha256'],
                   plan_sha256=digest(Path(__file__).with_name('SYNC_FUSION_V3_MATCHED_FREE_POSTHOC_PLAN_DEV_20261011.json')),
                   GPU_s=0, download_bytes=0, model_fit_or_inference_runs=0, protected_480_test_reads=0,
                   checks=['sealed cal/eval cache identities', 'all finite whole ties and disabled +inf',
                           'maximum W at every exactly reachable F with highest-threshold tie',
                           'all fixed-level caps and reachable points', '2000 original 12-visit draws seed20261011',
                           'all method bootstrap W/F/thresholds reselected per draw',
                           'descriptive percentile intervals and denominator omissions',
                           'individual visit oracle and pooled threshold contributions',
                           'sealed original calibration drift pooled/arm/visit',
                           '1609 feature-aligned RGB-supported POS and exactly126 loss flags',
                           'every loss group reaggregated', '8 independent ray/peak observation samples',
                           'unchanged prespecified successor-design decision rule'],
                   audited_files={p.name: digest(p) for p in out.iterdir() if p.suffix in ('.csv', '.npz', '.json')},
                   wall_s=time.perf_counter() - started)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(receipt, ensure_ascii=False))


def audit_outputs(args, data, cal, features, cal_seal, cells, roster, draws, multiplicities):
    out = args.output
    rows = read_csv(out / 'all_ties.csv')
    expected_total = sum(len(c['thresholds']) for c in cells.values())
    assert len(rows) == expected_total
    assert len({key(r) + (int(r['tie']),) for r in rows}) == expected_total
    for row in rows:
        c = cells[key(row)]; i = int(row['tie'])
        for j, field in enumerate(('W', 'F', 'U')):
            equal(row[field], c['counts'][i, j], ('tie', key(row), i, field))
        equal(row['threshold'], 'disabled' if i == 0 else c['thresholds'][i], ('tie_threshold', key(row), i))
    curves = read_csv(out / 'curves.csv')
    expected = {}
    for k, c in cells.items():
        for i in exact_curve(c['thresholds'], c['counts']):
            expected[k + (int(c['counts'][i, 1]),)] = i
    assert len(curves) == len(expected)
    assert {key(r) + (int(r['F']),) for r in curves} == set(expected)
    for row in curves:
        k = key(row); c = cells[k]; i = expected[k + (int(row['F']),)]
        for j, field in enumerate(('W', 'F', 'U')):
            equal(row[field], c['counts'][i, j], ('curve', k, field))
        equal(row['threshold'], 'disabled' if not np.isfinite(c['thresholds'][i]) else c['thresholds'][i], ('curve_threshold', k))
    fixed = read_csv(out / 'fixed_points.csv'); selected = {}
    assert len(fixed) == len(cells) * 7
    assert len({key(r) + (r['level'],) for r in fixed}) == len(fixed)
    for row in fixed:
        k = key(row); c = cells[k]; level = row['level']
        cap = int(np.floor(c['den'][1] * float(level[:-1]) / 100)) if '%' in level else int(level)
        i = select(c['thresholds'], c['counts'], cap); selected[k + (level,)] = i
        equal(row['cap'], cap, ('cap', k, level))
        for j, field in enumerate(('W', 'F', 'U')):
            equal(row[field], c['counts'][i, j], ('fixed', k, level, field))
        equal(row['threshold'], 'disabled' if not np.isfinite(c['thresholds'][i]) else c['thresholds'][i], ('fixed_threshold', k, level))
    boot = read_npz(out / 'bootstrap_draws.npz')
    np.testing.assert_array_equal(boot['draw'], draws)
    boot_values = {}
    for k, c in cells.items():
        # Matrix products use an independently constructed [tie,visit,label] table.
        wc = multiplicities @ c['clusters'][:, :, 0].T
        fc = multiplicities @ c['clusters'][:, :, 1].T
        nf = multiplicities @ c['fd']; np_ = multiplicities @ c['pd']
        for rate in (.02, .05):
            cap = np.floor(rate * nf).astype(int)
            last = np.sum(fc <= cap[:, None], axis=1) - 1
            rr = np.arange(len(draws)); w = wc[rr, last]; f = fc[rr, last]
            ix = np.argmax((wc == w[:, None]) & (fc == f[:, None]), axis=1)
            prefix = f'{k[0]}__{k[1]}__{k[2]}__{rate}'
            for suffix, values in (('W', w), ('F', f), ('tie', ix)):
                np.testing.assert_array_equal(boot[prefix + '__' + suffix], values, err_msg=prefix + '__' + suffix)
            boot_values[k + (rate,)] = (w, f, nf, np_, (nf > 0) & (np_ > 0))
        print('audit bootstrap', k, flush=True)
    for row in read_csv(out / 'bootstrap.csv'):
        arm, band, comp = row['arm'], int(row['band']), row['comparator']
        rate = float(row['level'][:-1]) / 100; lev = row['level']
        a = boot_values[arm, band, 'D_prime', rate]; b = boot_values[arm, band, comp, rate]
        valid = a[4]; delta = a[0] - b[0]
        equal(row['valid'], valid.sum(), ('bootstrap_valid', arm, band, comp, rate))
        if valid.any():
            ci = np.quantile(delta[valid], (.025, .975))
            pp = np.quantile(100 * delta[valid] / a[3][valid], (.025, .975))
            for field, value in zip(('lower', 'upper', 'lower_pp', 'upper_pp'), np.r_[ci, pp]):
                equal(row[field], value, ('bootstrap_interval', arm, band, comp, rate, field))
        else:
            assert all(row[f] == '' for f in ('lower', 'upper', 'lower_pp', 'upper_pp'))
        ca, cb = cells[arm, band, 'D_prime'], cells[arm, band, comp]
        ia, ib = selected[arm, band, 'D_prime', lev], selected[arm, band, comp, lev]
        equal(row['delta_W'], ca['counts'][ia, 0] - cb['counts'][ib, 0], ('bootstrap_point', arm, band, comp, rate))
        equal(row['D_F'], ca['counts'][ia, 1], ('bootstrap_D_F', arm, band, comp, rate))
        equal(row['comparison_F'], cb['counts'][ib, 1], ('bootstrap_comparison_F', arm, band, comp, rate))
    visit_rows = read_csv(out / 'visit_consistency.csv')
    for row in visit_rows:
        arm, band, visit, lev = row['arm'], int(row['band']), row['visit_id'], row['level']
        rate = float(lev[:-1]) / 100; local = {}
        for method in ('D_prime', 'rgb', 'or'):
            c = cells[arm, band, method]; mask = c['v'] == visit
            ss = data[method + '_score'][c['mask']][mask]; yy = c['y'][mask]
            t, count, _ = sorted_ties(ss, yy)
            i = select(t, count, int(np.floor(rate * (yy == 0).sum())))
            local[method] = int(count[i, 0])
            equal(row[method + '_W'], count[i, 0], ('visit_W', arm, band, visit, lev, method))
            equal(row[method + '_F'], count[i, 1], ('visit_F', arm, band, visit, lev, method))
            ct = c['thresholds'][selected[arm, band, method, lev]]
            equal(row[method + '_pooled_W'], ((ss >= ct) & (yy == 1)).sum(), ('visit_pooled_W', arm, band, visit, lev, method))
        evaluable = int(row['FREE']) > 0 and int(row['POS']) > 0
        equal(row['D_ge_rgb'], str(local['D_prime'] >= local['rgb']) if evaluable else '', ('visit_D_ge_rgb', arm, band, visit, lev))
    original_cells = {(c['method'], c['band']): c for c in cal_seal['cells']}
    for name in ('drift.csv', 'drift_visit.csv'):
        for row in read_csv(out / name):
            method, band = row['method'], int(row['band']); src = cal if row['role'] == 'cal' else data
            mask = src['band'] == band
            if row['arm'] != 'CAL_POOLED_SIX_ARMS':
                mask &= src['arm'] == row['arm']
            if name == 'drift_visit.csv':
                mask &= src['visit_id'] == row['visit_id']
            cc = original_cells[method, band]
            cut = np.inf if cc['threshold_kind'] == 'positive_infinity' else cc['threshold']
            prediction = (src[method + '_score'][mask] >= cut) if cc['status'] == 'ADOPTED' else np.zeros(mask.sum(), bool)
            np.testing.assert_array_equal(src[method + '_pred'][mask], prediction)
            yy = src['label'][mask]
            for field, value in dict(F=(prediction & (yy == 0)).sum(), FREE=(yy == 0).sum(), W=(prediction & (yy == 1)).sum(), POS=(yy == 1).sum()).items():
                equal(row[field], value, (name, row['arm'], band, method, row.get('visit_id'), field))
    audit_loss(args, data, features)
    audit_rgb_drift_distribution(args, data, cal)
    summary = read_json(out / 'summary.json')
    a = all(cells[ARMS[0], 1, 'D_prime']['counts'][selected[ARMS[0], 1, 'D_prime', lev], 0] >= cells[ARMS[0], 1, m]['counts'][selected[ARMS[0], 1, m, lev], 0] for lev in ('1%', '2%', '5%') for m in ('rgb', 'or'))
    vc = {lev: sum(r['D_ge_rgb'] == 'True' for r in visit_rows if r['arm'] == ARMS[0] and int(r['band']) == 1 and r['level'] == lev) for lev in ('2%', '5%')}
    assert summary['rule_a'] == a and summary['visit_success'] == vc
    equal(summary['decision'], '建议进入条件仲裁器设计与新 visit 确认' if a and min(vc.values()) >= 8 else 'D′ 的双向占优不稳定，暂停融合迭代', 'decision')


def audit_rgb_drift_distribution(args, data, cal):
    """Focused optional supplemental drift quantiles from original caches."""
    path = args.output / 'rgb_free_distribution.csv'
    if not path.exists():
        return
    cut = next(c['threshold'] for c in read_json(args.source / 'cal/cal_seal.json')['cells'] if c['method'] == 'rgb' and c['band'] == 1)
    for row in read_csv(path):
        src = cal if row['role'] == 'cal' else data
        mask = (src['arm'] == ARMS[0]) & (src['band'] == 1) & (src['label'] == 0)
        score = src['rgb_score'][mask]
        fields = dict(FREE=len(score), above_zero=(score >= 0).sum(), relaxed_negative=((score >= cut) & (score < 0)).sum())
        for name, p in zip(('score_p50', 'score_p90', 'score_p95'), (.5, .9, .95)):
            fields[name] = np.quantile(score, p)
        for field, value in fields.items():
            equal(row[field], value, ('rgb_drift_distribution', row['role'], field))


def audit_loss(args, data, features):
    rows = read_csv(args.output / 'loss_queries.csv')
    mask = (data['arm'] == ARMS[0]) & (data['band'] == 1) & (data['label'] == 1) & data['rgb_pred']
    ii = np.flatnonzero(mask)
    assert len(ii) == len(rows) == 1609
    assert (~data['D_prime_pred'][ii]).sum() == 126
    assert read_csv(args.output / 'loss126.csv') == [r for r in rows if r['loss'] == 'True']
    manifest_path = args.source / 'feature_manifest.json'
    manifest = read_json(manifest_path)
    input_receipt = read_json(args.output / 'loss_input_receipts.json')
    assert digest(manifest_path) == input_receipt['feature_manifest_sha256']
    input_hashes = {r['path']: r['sha256'] for r in manifest['inputs']}
    for entry in input_receipt['inputs']:
        assert entry['sha256'] == input_hashes[entry['path']] == digest(entry['path'])
    assert digest(args.source / 'features/eval.npz') == next(r['sha256'] for r in manifest['outputs'] if r['role'] == 'eval')
    for field in ('arm', 'frame_id', 'query_id', 'band', 'visit_id'):
        np.testing.assert_array_equal(features[field], data[field])
    quart = np.quantile(data['rgb_score'][ii], (.25, .5, .75))
    rgb_cut = next(c['threshold'] for c in read_json(args.source / 'cal/cal_seal.json')['cells'] if c['method'] == 'rgb' and c['band'] == 1)
    for row, idx in zip(rows, ii):
        for field in ('visit_id', 'frame_id', 'query_id'):
            equal(row[field], data[field][idx], ('loss_identity', idx, field))
        equal(row['loss'], str(not data['D_prime_pred'][idx]), ('loss_flag', idx))
        equal(row['rgb_score'], data['rgb_score'][idx], ('rgb_score', idx))
        equal(row['rgb_cut_margin'], data['rgb_score'][idx] - rgb_cut, ('rgb_cut_margin', idx))
        equal(row['rgb_quartile'], np.searchsorted(quart, data['rgb_score'][idx], side='right') + 1, ('quartile', idx))
        X = features['X'][idx]
        for field, value in dict(tof_unknown=X[3], tof_support_Kmean=X[2], tof_valid_K=X[9]).items():
            equal(row[field], value, ('feature_binding', idx, field))
        empty = 'finite_all_after' if X[9] == 2 and X[3] < 1 and X[5] == 0 and X[4] == 0 and X[6] > 0 else 'finite_outside_mixed_or_before' if X[9] == 2 and X[3] < 1 and X[5] == 0 else 'inside_or_unknown'
        equal(row['tof_empty_proxy'], empty, ('empty_proxy', idx))
        rays = int(row['rgb_support_rays']); unknown = float(row['tof_unknown']); support = float(row['tof_support_Kmean']); peak = float(row['tof_accepted_peak_snr_max_Kmean'])
        bins = dict(rgb_support_bin='0-15' if rays < 16 else '16-63' if rays < 64 else '64-255' if rays < 256 else '256+',
                    rgb_strong=str(data['rgb_score'][idx] >= quart[1] and rays >= 16),
                    tof_unknown_bin='0-.25' if unknown < .25 else '.25-.75' if unknown < .75 else '.75-1',
                    tof_support_bin='0' if support == 0 else '0-16' if support < 16 else '16+',
                    tof_peak_bin='<3' if peak < 3 else '3-10' if peak < 10 else '10+',
                    query_xy=str(data['query_id'][idx]).split('_z')[0], evidence_group=str(data['evidence_group'][idx]))
        z = float(row['rgb_support_median_Z_m']) if row['rgb_support_median_Z_m'] else None
        bins['rgb_band_position'] = 'no_inside' if z is None else 'lower_third' if z < .8+.7/3 else 'middle_third' if z < .8+1.4/3 else 'upper_third'
        for field, value in bins.items():
            equal(row[field], value, ('loss_group_classification', idx, field))
    groups = read_csv(args.output / 'loss_groups.csv')
    expected = {}
    for field in ('rgb_quartile', 'rgb_support_bin', 'rgb_strong', 'tof_unknown_bin', 'tof_support_bin', 'tof_empty_proxy', 'tof_peak_bin', 'rgb_band_position', 'query_xy', 'evidence_group', 'visit_id'):
        for value in set(r[field] for r in rows):
            sub = [r for r in rows if r[field] == value]; losses = sum(r['loss'] == 'True' for r in sub)
            expected[field, value] = dict(loss=losses, RGB_supported_POS=len(sub), loss_fraction=losses / len(sub), share_of_126=losses / 126)
    assert len(groups) == len(expected)
    for row in groups:
        for field, value in expected[row['dimension'], row['value']].items():
            equal(row[field], value, ('loss_group', row['dimension'], row['value'], field))
    # Independently recompute cached RGB ray support and ToF raw peak on fixed
    # spread samples: four losses and four retained rows, chosen by row position.
    sample_indices = []
    for flag in ('True', 'False'):
        candidates = [i for i, r in enumerate(rows) if r['loss'] == flag]
        sample_indices.extend(candidates[i] for i in np.linspace(0, len(candidates)-1, 4, dtype=int))
    public = read_json(args.source / 'public_roster.json')
    pub = {r['source_id']: r for r in public['rows']}; queries = {q['name']: q for q in public['queries']}
    predictions = {}
    for name in ('predictions.json', 'sealed_eval/predictions.json'):
        predictions.update({r['source_id']: r for r in read_json(args.source / 'rgb_inference/dav2' / name)['rows']})
    synthesis = {r['source_id']: r for r in read_json(args.source / 'synthesis_manifest.json')['frames']}
    seal = read_json(args.source / 'frozen_readout_seal.json')
    for i in sample_indices:
        row = rows[i]; frame = pub[row['frame_id']]; query = queries[row['query_id']]
        z = read_npz(predictions[row['frame_id']]['path'])['depth']; K = np.asarray(frame['depth_K'])
        h, w = z.shape; yy, xx = np.indices((h, w))
        ki = np.linalg.inv(K)
        rx = ki[0, 0] * xx + ki[0, 1] * yy + ki[0, 2]; ry = ki[1, 0] * xx + ki[1, 1] * yy + ki[1, 2]
        low, high = np.asarray(query['low']), np.asarray(query['high'])
        with np.errstate(divide='ignore', invalid='ignore'):
            xa, xb = low[0]/rx, high[0]/rx; ya, yb = low[1]/ry, high[1]/ry
        xmin, xmax = np.minimum(xa, xb), np.maximum(xa, xb)
        ymin, ymax = np.minimum(ya, yb), np.maximum(ya, yb)
        xmin = np.where(rx == 0, -np.inf if low[0] <= 0 <= high[0] else np.inf, xmin)
        xmax = np.where(rx == 0, np.inf if low[0] <= 0 <= high[0] else -np.inf, xmax)
        ymin = np.where(ry == 0, -np.inf if low[1] <= 0 <= high[1] else np.inf, ymin)
        ymax = np.where(ry == 0, np.inf if low[1] <= 0 <= high[1] else -np.inf, ymax)
        entry = np.maximum(np.maximum(xmin, ymin), low[2]); exit = np.minimum(np.minimum(xmax, ymax), high[2]); dom = exit + 1e-12 >= entry
        inside = dom & np.isfinite(z) & (z > 0) & (z >= entry) & (z <= exit)
        equal(row['rgb_support_rays'], inside.sum(), ('independent_ray_sample', i))
        equal(row['rgb_support_median_Z_m'], float(np.median(z[inside])) if inside.any() else None, ('independent_z_sample', i))
        peaks = []
        for arm in synthesis[row['frame_id']]['arms']:
            if arm['arm'] != ARMS[0]:
                continue
            zz = read_npz(arm['path']); hh = zz['hist']; pp = np.argmax(hh, axis=-1)
            height = np.take_along_axis(hh, pp[..., None], axis=-1)[..., 0]
            background = np.take_along_axis(zz['background'], pp[..., None], axis=-1)[..., 0]
            snr = height / np.sqrt(np.maximum(height, 0) + 2*background + 1)
            accepted = (zz['coverage'] >= seal['tof_zone_coverage_min']) & (snr >= seal['tof_peak_snr_min'])
            peaks.append((snr.max(), snr[accepted].max() if accepted.any() else 0, accepted.sum()))
        for field, value in zip(('tof_peak_snr_max_Kmean', 'tof_accepted_peak_snr_max_Kmean', 'tof_accepted_zones_Kmean'), np.mean(peaks, axis=0)):
            equal(row[field], value, ('independent_peak_sample', i, field))


if __name__ == '__main__':
    main()
