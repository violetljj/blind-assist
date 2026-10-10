"""Consumed-Development leave-one-capture-out sampled-query calibration.

This changes a working point, not a model. Sampled FREE is not volume clearance.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
import numpy as np
from rgb_body_query_interval_distribution import load, utc, interval_probability
from rgb_body_query_input_diagnostic import sha, write as write_json
from rgb_body_query_reference_eval import rays, ray_interval
from rgb_body_query_scene_diagnostic import write_csv

ARMS = ('depth_ray_mean_margin_gridcal', 'geometry_mean_margin_gridcal',
        'affine_margin_gridcal', 'depth_ray_gaussian', 'geometry_gaussian')
COHORTS = ('arkit16', 'arkit_40777060', 'arkit_40777065')
FREE = 'FREE_ON_SAMPLED_RAYS'
TARGET = .05


def write(path, value):
    # JSON stays strict; unavailable support is an explicit sentinel, not zero.
    def serial(obj):
        if isinstance(obj, dict):
            return {key:serial(item) for key,item in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [serial(item) for item in obj]
        if isinstance(obj, float) and np.isneginf(obj):
            return '-Infinity'
        return obj
    write_json(path, serial(value))


def nth_score(values):
    values = np.asarray(values, np.float64).ravel()
    finite = values[np.isfinite(values)]
    return float(np.partition(finite, len(finite)-16)[len(finite)-16]) if len(finite) >= 16 else -np.inf


def select_cutoff(values):
    """Largest attainable empirical support count <= floor(.05*N), ties kept."""
    values = np.asarray(values, np.float64)
    if not len(values):
        return dict(status='NOT_EVALUABLE', negative_queries=0, cutoff=None)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError('Invalid query scores')
    allowed = int(np.floor(TARGET*len(values)))
    finite = values[np.isfinite(values)]
    if not len(finite):
        cutoff = 0.  # A finite threshold excludes unavailable (-inf) queries.
    else:
        unique, counts = np.unique(finite, return_counts=True)
        tails = np.cumsum(counts[::-1])[::-1]
        eligible = tails <= allowed
        cutoff = float(unique[np.flatnonzero(eligible)[0]]) if eligible.any() else float(np.nextafter(unique[-1], np.inf))
    fp = int((values >= cutoff).sum())
    # Independent exhaustive threshold/count check, including impossible queries.
    candidates = np.unique(np.r_[finite, np.nextafter(finite.max(), np.inf) if len(finite) else 0.])
    attainable = [int((values >= c).sum()) for c in candidates]
    best = max(n for n in attainable if n <= allowed)
    if fp != best or not np.isfinite(cutoff):
        raise AssertionError('Conservative tie selection failed')
    return dict(status='FROZEN_AFTER_CAL', cutoff=cutoff, negative_queries=len(values),
                false_support=fp, support_rate=fp/len(values), allowed_false_support=allowed,
                unavailable_queries=int(np.isneginf(values).sum()), ties_retained=True,
                exact_target_reached=fp/len(values) == TARGET, optimality_check='PASS')


def summary(records):
    out = {}
    for title, keys in [('all', []), ('band', ['distance_band'])]:
        groups = defaultdict(list)
        for r in records:
            groups[tuple(r[k] for k in keys)].append(r)
        for key, rr in groups.items():
            pos = [r for r in rr if r['reference_state'] == 'POSITIVE']
            free = [r for r in rr if r['reference_state'] == FREE]
            unknown = [r for r in rr if r['reference_state'] == 'UNKNOWN']
            out['/'.join([title]+list(key))] = dict(
                query_total=len(rr), frames=len({r['frame'] for r in rr}),
                positive_total=len(pos), positive_support=sum(r['predicted_positive'] for r in pos),
                positive_known_witness=sum(r['positive_known_witness'] for r in pos),
                free_total=len(free), free_support=sum(r['predicted_positive'] for r in free),
                free_support_rate=sum(r['predicted_positive'] for r in free)/len(free) if free else None,
                unknown_total=len(unknown), unknown_support=sum(r['predicted_positive'] for r in unknown),
                unavailable_queries=sum(r['valid_ray_count'] < 16 for r in rr))
    return out


def run(repo, root, budget):
    started = time.perf_counter()
    out = root/'query'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'execution_plan.json').exists():
        raise FileExistsError('Preserve prior probe')
    parent_plan = root/'plan.json'
    plan_hash = sha(parent_plan)
    plan = load(parent_plan)
    assert plan['query_probe']['arms'] == list(ARMS)
    assert plan['query_probe']['target_sampled_free_query_support_rate'] == TARGET
    old = repo/'artifacts.local/work/rgb-body-query-query-level-dev-20261009'
    frozen = repo/'artifacts.local/work/rgb-body-query-interval-distribution-dev-20261010'
    candidate = frozen/'candidate'
    info = load(candidate/'training_inputs.json')
    primarycal = load(candidate/'calibration.json')
    pointcal = load(frozen/'mean-probe/calibration.json')
    inherited = {a:(pointcal if a in pointcal['arms'] else primarycal)['arms'][a]['cutoff'] for a in ARMS}
    write(out/'execution_plan.json', dict(created_utc=utc(), parent_plan_sha256=plan_hash,
          source_sha256=sha(__file__), arms=ARMS, cohorts=COHORTS, target=TARGET,
          selection='Other two captures strict sampled FREE only; no positive/heldout selection',
          query_score='16th largest domain-valid score; threshold comparison >=; unavailable=-inf',
          budget_cpu_wall_s=budget, inherited_cutoffs=inherited, new_training=False, gpu_s=0,
          original_calibration_sha256=sha(candidate/'calibration.json'),
          original_point_calibration_sha256=sha(frozen/'mean-probe/calibration.json')))
    complete = False
    failure = None
    checks = Counter()
    try:
        def check():
            if time.perf_counter()-started >= budget:
                raise TimeoutError('Query probe CPU wall budget reached')
        original = load(frozen/'train-cal-sensor/dataset_manifest.json')
        original_states = Counter(q['state'] for r in original['rows'] if r['split']=='cal' for q in r['queries'])
        assert original_states[FREE] == 2
        data, coverage, identities = {}, {}, []
        for name in COHORTS:
            path = old/'fixed-grid-sensor'/name if name=='arkit16' else old/'additional-arkit-sensor'/name.removeprefix('arkit_')
            manifest = load(path/'dataset_manifest.json')
            pmpath = candidate/'predictions'/name/'predictions.json'
            pm = load(pmpath)
            refs = {(r['scan'],r['frame']):r for r in manifest['rows']}
            records = {a:[] for a in ARMS}
            identities.append(dict(cohort=name, reference_manifest_path=str(path/'dataset_manifest.json'),
                reference_manifest_sha256=sha(path/'dataset_manifest.json'),
                predictions_manifest_path=str(pmpath), predictions_manifest_sha256=sha(pmpath)))
            coverage[name] = dict(frames=len(pm['rows']), states=dict(Counter(q['state'] for r in manifest['rows'] for q in r['queries'])),
                                 by_band={})
            for row in pm['rows']:
                check()
                ref = refs[row['scan'],row['frame']]
                assert sha(ref['reference_path']) == ref['reference_sha256']
                assert sha(row['sampled_depth_path']) == row['sampled_depth_sha256']
                with np.load(ref['reference_path']) as f:
                    labels = f['labels']
                with np.load(row['sampled_depth_path']) as f:
                    dp = f['depth']
                valid = np.isfinite(dp)&(dp>0)
                zaffine = np.zeros(dp.shape)
                zaffine[valid] = np.exp(info['affine']['a']*np.log(dp[valid].astype(np.float64))+info['affine']['b'])
                dist = {}
                for a in ('depth_ray_gaussian', 'geometry_gaussian'):
                    p = row['distributions'][a]
                    assert sha(p['path']) == p['sha256']
                    with np.load(p['path']) as f:
                        dist[a] = (f['mu'], f['sigma'], f['valid'])
                rx, ry = rays(row['depth_K'], row['depth_shape'])
                for j, q in enumerate(manifest['queries']):
                    check()
                    entry, exit, domain = ray_interval(rx, ry, q)
                    lab = labels[j]
                    counts = {k:int((lab==v).sum()) for k,v in [('positive_pixels',1),('free_ray_pixels',0),('unknown_pixels',2)]}
                    assert all(ref['queries'][j][k] == v for k,v in counts.items())
                    assert int(domain.sum()) == ref['queries'][j]['domain_pixels']
                    state = 'POSITIVE' if counts['positive_pixels']>=16 else (FREE if counts['positive_pixels']==0 and counts['unknown_pixels']==0 and counts['free_ray_pixels']>=16 else 'UNKNOWN')
                    assert state == ref['queries'][j]['state']
                    checks['reference_query_counts'] += 1
                    band = f'{q["low"][2]:g}-{q["high"][2]:g}m'
                    coverage[name]['by_band'].setdefault(band, Counter())[state] += 1
                    scores = {}
                    margin = np.full(dp.shape, -np.inf)
                    mask = domain&valid
                    margin[mask] = np.minimum(zaffine[mask]-entry[mask], exit[mask]-zaffine[mask])
                    scores['affine_margin_gridcal'] = margin
                    for a,(mu,sigma,v) in dist.items():
                        mask = domain&v
                        margin = np.full(mu.shape, -np.inf)
                        z = np.exp(mu)
                        margin[mask] = np.minimum(z[mask]-entry[mask], exit[mask]-z[mask])
                        scores[a.replace('gaussian','mean_margin_gridcal')] = margin
                        cdf = interval_probability(mu,sigma,entry,exit,domain,v)
                        scores[a] = np.where(mask&(entry>0)&(exit>=entry),cdf,-np.inf)
                    for a, s in scores.items():
                        query_score = nth_score(s)
                        witness_score = nth_score(s[lab==1])
                        for threshold in [inherited[a]] + ([query_score, np.nextafter(query_score,np.inf)] if np.isfinite(query_score) else [0.]):
                            # Full mask count is independent of nth-score implementation.
                            assert (int((s>=threshold).sum())>=16) == (query_score>=threshold)
                            assert (int(((s>=threshold)&(lab==1)).sum())>=16) == (witness_score>=threshold)
                            checks['nth_score_mask_equivalence'] += 1
                        finite = s[np.isfinite(s)]
                        assert query_score == (float(np.sort(finite)[-16]) if len(finite)>=16 else -np.inf)
                        checks['independent_sorted_nth_score'] += 1
                        records[a].append(dict(cohort=name,environment=ref['environment'],scan=ref['scan'],frame=ref['frame'],
                            query=q['name'],distance_band=band,reference_state=state,valid_ray_count=len(finite),
                            query_score=query_score,known_positive_score=witness_score,**counts))
            data[name] = records
            write(out/f'{name}_scores.json', dict(arms=records, identity=identities[-1]))
        write(out/'reference_coverage.json', dict(original_cal_states=dict(original_states), cohorts=coverage, identities=identities))
        tables, pairs, frozen_folds = [], [], {}
        for held in COHORTS:
            check()
            calnames = [n for n in COHORTS if n!=held]
            threshold = {}
            for a in ARMS:
                neg = [r['query_score'] for n in calnames for r in data[n][a] if r['reference_state']==FREE]
                threshold[a] = select_cutoff(neg)
                threshold[a]['cutoff_unit'] = 'probability score' if a.endswith('gaussian') else 'optical-Z interval margin metres'
                checks['fold_arm_threshold_optimality'] += 1
            foldpath = out/f'{held}_thresholds.json'
            write(foldpath, dict(frozen_utc=utc(), held_capture=held, cal_captures=calnames, arms=threshold,
                  selection_positive_or_held_scores=False, empirical_target=TARGET))
            frozen_folds[held] = dict(path=str(foldpath), sha256=sha(foldpath))
            # Held evaluation begins only after this fold's threshold file is frozen.
            all_records = {}
            for a in ARMS:
                for mode, cutoff in [('inherited', inherited[a]), ('query_calibrated', threshold[a]['cutoff'])]:
                    rr = []
                    for r in data[held][a]:
                        rec = dict(r, arm=a, mode=mode, cutoff=cutoff,
                            predicted_positive=bool(r['query_score']>=cutoff) if cutoff is not None else False,
                            positive_known_witness=bool(r['known_positive_score']>=cutoff) if cutoff is not None else False,
                            evaluation_status='EVALUABLE' if cutoff is not None else 'NOT_EVALUABLE')
                        rr.append(rec)
                    all_records[a+'/'+mode] = rr
                    for key, v in summary(rr).items():
                        tables.append(dict(cohort=held, arm=a, mode=mode, group=key, cutoff=cutoff, **v))
            for a in ARMS:
                candidate_rows = all_records[a+'/query_calibrated']
                for baseline in [a+'/inherited'] + (['affine_margin_gridcal/query_calibrated'] if a!='affine_margin_gridcal' else []):
                    for r,b in zip(candidate_rows,all_records[baseline]):
                        pairs.append(dict(cohort=held,arm=a,baseline=baseline,frame=r['frame'],query=r['query'],distance_band=r['distance_band'],
                            reference_state=r['reference_state'],positive_rescue=r['reference_state']=='POSITIVE' and r['positive_known_witness'] and not b['positive_known_witness'],
                            positive_loss=r['reference_state']=='POSITIVE' and b['positive_known_witness'] and not r['positive_known_witness'],
                            free_removed=r['reference_state']==FREE and b['predicted_positive'] and not r['predicted_positive'],
                            free_added=r['reference_state']==FREE and r['predicted_positive'] and not b['predicted_positive']))
            write(out/f'{held}_evaluation.json', dict(frozen_threshold_sha256=sha(foldpath), records=all_records))
        write_csv(out/'summary.csv', tables)
        write_csv(out/'rescue_loss.csv', pairs)
        pair_summary = []
        for keys in [('cohort','arm','baseline'), ('cohort','arm','baseline','distance_band')]:
            groups = defaultdict(list)
            for r in pairs:
                groups[tuple(r[k] for k in keys)].append(r)
            for key, rr in groups.items():
                pair_summary.append(dict(zip(keys,key), **{k:sum(r[k] for r in rr) for k in ('positive_rescue','positive_loss','free_removed','free_added')}))
        write(out/'results.json', dict(summary=tables, rescue_loss=pair_summary, folds=frozen_folds,
            reference_coverage=coverage, original_cal_states=dict(original_states),
            limitations='Consumed Development; same camera family; related frames and boxes; sampled FREE is not volume clearance; target 5% empirical only'))
        assert sha(parent_plan) == plan_hash
        write(out/'focused_check.json', dict(status='PASS', counts=dict(checks), parent_plan_unchanged=True,
            inherited_thresholds_unchanged=True, fold_thresholds_frozen_before_held_evaluation=True))
        complete = True
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        write(out/'completion_receipt.json', dict(status='COMPLETE' if complete else 'PARTIAL', failure=failure,
            cpu_wall_s=time.perf_counter()-started, budget_cpu_wall_s=budget, gpu_s=0, download_bytes=0,
            new_training=False, source_sha256=sha(__file__), parent_plan_sha256=plan_hash, resources_released=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cpu-budget-s', type=float, default=300.)
    args = parser.parse_args()
    run(args.repo.resolve(), args.root.resolve(), args.cpu_budget_s)
