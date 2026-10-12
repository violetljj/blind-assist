"""A0 frozen log-distance correction proxy; consumed Development, no training.

Main cutoffs use other-capture strict sampled FREE only. Held-capture FREE
curves are explicitly posthoc diagnostics and never choose a delta per capture.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import time
import numpy as np
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_query_calibration_probe import (
    COHORTS, FREE, nth_score, select_cutoff, summary, write)
from rgb_body_query_reference_eval import rays, ray_interval
from rgb_body_query_scene_diagnostic import write_csv

DELTAS = (.05, .1, .2)
BASE = 'affine'
ARMS = (BASE,) + tuple(f'a0_delta_{d:g}' for d in DELTAS)


def cutoff_at_cost(values, allowed):
    """Closest attainable negative support <= budget, preserving score ties."""
    values = np.asarray(values, np.float64)
    finite = values[np.isfinite(values)]
    if not len(values):
        return None
    if not len(finite):
        return 0.
    unique, counts = np.unique(finite, return_counts=True)
    tails = np.cumsum(counts[::-1])[::-1]
    excluded = tails > allowed
    # Lowest finite cutoff permitted by FREE scores; fill cost plateaus without
    # consulting positives. At the saturated endpoint keep -inf unavailable.
    return float(np.nextafter(unique[np.flatnonzero(excluded)[-1]], np.inf)) if excluded.any() else -float(np.finfo(np.float64).max)


def evaluate(rows, cutoff):
    return [dict(r, cutoff=cutoff,
                 predicted_positive=r['query_score'] >= cutoff,
                 positive_known_witness=r['known_positive_score'] >= cutoff)
            for r in rows]


def paired(rows, baseline, cohort, arm, mode):
    pairs = []
    for r, b in zip(rows, baseline, strict=True):
        assert (r['scan'], r['frame'], r['query'], r['reference_state']) == (b['scan'], b['frame'], b['query'], b['reference_state'])
        pos, free = r['reference_state'] == 'POSITIVE', r['reference_state'] == FREE
        pairs.append(dict(cohort=cohort, arm=arm, mode=mode, scan=r['scan'],
            environment=r['environment'], frame=r['frame'], query=r['query'],
            distance_band=r['distance_band'], reference_state=r['reference_state'],
            candidate_cutoff=r['cutoff'], affine_cutoff=b['cutoff'],
            candidate_known_witness=r['positive_known_witness'], affine_known_witness=b['positive_known_witness'],
            candidate_support=r['predicted_positive'], affine_support=b['predicted_positive'],
            positive_rescue=pos and r['positive_known_witness'] and not b['positive_known_witness'],
            positive_loss=pos and b['positive_known_witness'] and not r['positive_known_witness'],
            free_removed=free and b['predicted_positive'] and not r['predicted_positive'],
            free_added=free and r['predicted_positive'] and not b['predicted_positive']))
    return pairs


def pair_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups['all'].append(r)
        groups['band/' + r['distance_band']].append(r)
    return {k: {field: sum(r[field] for r in rr)
                for field in ('positive_rescue', 'positive_loss', 'free_removed', 'free_added')}
            for k, rr in groups.items()}


def run(repo, output, budget, prior_attempt=None):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists():
        raise FileExistsError('Preserve the existing A0 run and its failures')
    started = time.perf_counter()
    prior_cpu_s = 0.
    if prior_attempt is not None:
        prior = load(prior_attempt/'completion_receipt.json')
        prior_cpu_s = prior['cpu_wall_s'] + prior.get('prior_attempt_cpu_wall_s', 0.)
        assert prior['status'] == 'PARTIAL'
        budget -= prior_cpu_s
        if budget <= 0:
            raise TimeoutError('No remaining A0 budget after prior attempt')
    frozen = repo/'artifacts.local/work/rgb-body-query-interval-distribution-dev-20261010/candidate'
    reference_root = repo/'artifacts.local/work/rgb-body-query-query-level-dev-20261009'
    previous = repo/'artifacts.local/work/rgb-body-query-migration-diagnostic-dev-20261010/query'
    info = load(frozen/'training_inputs.json')
    write(output/'plan.json', dict(run='RGB_BODY_QUERY_A0_DEV_20261010', created_utc=utc(),
        base_commit='167181b5', lane='EXPLORE_CONSUMED_DEVELOPMENT', deltas=DELTAS, cohorts=COHORTS,
        mechanism='log Z = log Z_affine + clip(mu_old_depth_ray - log Z_affine, -delta, delta)',
        main='Three leave-one-capture-out folds; other two strict sampled FREE only; attainable support <=5%; >= cutoff; ties retained',
        posthoc='Held FREE only selects largest attainable count <= same-fold affine actual FREE; full count-budget curves also saved; never main evidence',
        inputs='Frozen Depth Pro, original-train log-affine coefficients and old depth/ray mu; masks must equal',
        budgets=dict(cpu_wall_s_remaining=budget, prior_attempt_cpu_wall_s=prior_cpu_s, gpu_s=0, download_bytes=0),
        repair_lineage=None if prior_attempt is None else dict(path=str(prior_attempt), plan_sha256=sha(prior_attempt/'plan.json'),
            defect='Equal FREE support does not imply equal positive witnesses; keep affine main point fixed in matched-cost diagnostic and use lowest finite FREE-permitted cutoff for full curves'),
        optional_input_range_fallback='NOT_RUN; not required, no default q01/q99 fallback',
        decision_check=dict(unit='Paired fixed-grid sampled query; one query is smallest observable change',
            affine_targets=dict(arkit16=5, arkit_40777060=12, arkit_40777065=10),
            strong='Both additional captures rescue>loss, FREE<=affine, and posthoc curves not worse',
            mixed='Witness benefit plus extra FREE cost is a tradeoff, not automatic failure',
            negative='No additional-capture witness advantage lowers proxy training priority, cannot reject retrained residual'),
        limits='A0 clips an old absolute-distance model, not a trained residual. Consumed same-family related captures; sampled FREE not whole-volume clearance',
        source_sha256=sha(__file__), affine_training_inputs_sha256=sha(frozen/'training_inputs.json')))
    shutil.copyfile(__file__, output/'executed_bounded_residual_proxy.py')
    complete, failure = False, None
    checks = Counter()
    inputs, all_data, clip_stats = [], {}, []

    def check_budget():
        if time.perf_counter()-started >= budget:
            raise TimeoutError('A0 CPU wall budget reached')

    def checked(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Frozen input identity changed: {path}')
        inputs.append(dict(path=str(path), sha256=digest))

    try:
        for cohort in COHORTS:
            folder = reference_root/'fixed-grid-sensor'/cohort if cohort == 'arkit16' else reference_root/'additional-arkit-sensor'/cohort.removeprefix('arkit_')
            checked(folder/'dataset_manifest.json')
            checked(frozen/'predictions'/cohort/'predictions.json')
            checked(previous/f'{cohort}_scores.json')
            manifest = load(folder/'dataset_manifest.json')
            pm = load(frozen/'predictions'/cohort/'predictions.json')
            old = load(previous/f'{cohort}_scores.json')['arms']
            refs = {(r['scan'], r['frame']): r for r in manifest['rows']}
            old_lookup = {a: {(r['scan'], r['frame'], r['query']): r for r in rr} for a, rr in old.items()}
            data = {a: [] for a in ARMS}
            for row in pm['rows']:
                check_budget()
                ref = refs[row['scan'], row['frame']]
                checked(ref['reference_path'], ref['reference_sha256'])
                checked(row['sampled_depth_path'], row['sampled_depth_sha256'])
                dist_path = row['distributions']['depth_ray_gaussian']
                checked(dist_path['path'], dist_path['sha256'])
                with np.load(ref['reference_path']) as f:
                    labels = f['labels']
                with np.load(row['sampled_depth_path']) as f:
                    dp = f['depth']
                with np.load(dist_path['path']) as f:
                    mu, dr_valid = f['mu'], f['valid']
                valid = np.isfinite(dp) & (dp > 0)
                assert np.array_equal(valid, dr_valid)
                assert np.isfinite(mu[valid]).all()
                checks['matching_affine_and_depth_ray_valid_masks'] += 1
                log_affine = np.zeros(dp.shape, np.float64)
                log_affine[valid] = info['affine']['a'] * np.log(dp[valid].astype(np.float64)) + info['affine']['b']
                depths = {BASE: np.exp(log_affine)}
                assert np.array_equal(np.exp(log_affine + np.clip(mu-log_affine, 0, 0)), depths[BASE])
                correction = mu.astype(np.float64)-log_affine
                for arm, delta in zip(ARMS[1:], DELTAS, strict=True):
                    residual = np.clip(correction, -delta, delta)
                    assert np.max(np.abs(residual[valid])) <= delta
                    depths[arm] = np.exp(log_affine+residual)
                    clip_stats.append(dict(cohort=cohort, frame=row['frame'], arm=arm,
                        valid_pixels=int(valid.sum()), clipped_pixels=int((np.abs(correction[valid])>delta).sum()),
                        max_absolute_log_residual=float(np.abs(residual[valid]).max())))
                rx, ry = rays(row['depth_K'], row['depth_shape'])
                for j, q in enumerate(manifest['queries']):
                    check_budget()
                    entry, exit, domain = ray_interval(rx, ry, q)
                    lab = labels[j]
                    state = ref['queries'][j]['state']
                    pos, free, unknown = (int((lab == n).sum()) for n in (1, 0, 2))
                    expected_state = 'POSITIVE' if pos >= 16 else (FREE if pos == 0 and unknown == 0 and free >= 16 else 'UNKNOWN')
                    assert state == expected_state
                    assert all(ref['queries'][j][k] == v for k, v in [('positive_pixels', pos), ('free_ray_pixels', free), ('unknown_pixels', unknown), ('domain_pixels', int(domain.sum()))])
                    mask = valid & domain
                    for arm, depth in list(depths.items()) + [('old_depth_ray', np.exp(mu))]:
                        score = np.full(dp.shape, -np.inf)
                        score[mask] = np.minimum(depth[mask]-entry[mask], exit[mask]-depth[mask])
                        qs, ws = nth_score(score), nth_score(score[lab == 1])
                        if arm in (BASE, 'old_depth_ray'):
                            prior_arm = 'affine_margin_gridcal' if arm == BASE else 'depth_ray_mean_margin_gridcal'
                            prior = old_lookup[prior_arm][row['scan'], row['frame'], q['name']]
                            for key, value in [('query_score', qs), ('known_positive_score', ws)]:
                                pv = -np.inf if prior[key] == '-Infinity' else prior[key]
                                assert value == pv
                            checks['exact_prior_score_reproduction'] += 1
                        if arm == 'old_depth_ray':
                            continue
                        finite = score[np.isfinite(score)]
                        assert qs == (float(np.sort(finite)[-16]) if len(finite) >= 16 else -np.inf)
                        for threshold in ([qs, np.nextafter(qs, np.inf)] if np.isfinite(qs) else [0.]):
                            assert (int((score>=threshold).sum())>=16) == (qs>=threshold)
                            assert (int(((score>=threshold)&(lab==1)).sum())>=16) == (ws>=threshold)
                        checks['sorted_query_score_and_witness_equivalence'] += 1
                        data[arm].append(dict(cohort=cohort, environment=ref['environment'], scan=ref['scan'], frame=ref['frame'],
                            query=q['name'], distance_band=f'{q["low"][2]:g}-{q["high"][2]:g}m', reference_state=state,
                            valid_ray_count=len(finite), query_score=qs, known_positive_score=ws))
            assert all(len(rr) == 432 for rr in data.values())
            all_data[cohort] = data
            write(output/f'{cohort}_scores.json', dict(arms=data))
            print('A0_SCORES_COMPLETE', cohort, flush=True)
        write(output/'inputs.json', inputs)
        write_csv(output/'clip_stats.csv', clip_stats)
        results, tables, pairs, curves = {}, [], [], []
        for held in COHORTS:
            check_budget()
            cal_cohorts = [c for c in COHORTS if c != held]
            thresholds = {a: select_cutoff([r['query_score'] for c in cal_cohorts for r in all_data[c][a] if r['reference_state']==FREE]) for a in ARMS}
            threshold_path = output/f'{held}_main_thresholds.json'
            write(threshold_path, dict(frozen_utc=utc(), cal_cohorts=cal_cohorts, held_capture=held,
                selection='Other two captures strict sampled FREE only', arms=thresholds))
            main = {a: evaluate(rr, thresholds[a]['cutoff']) for a, rr in all_data[held].items()}
            main_summary = {a: summary(rr) for a, rr in main.items()}
            target = main_summary[BASE]['all']['free_support']
            assert target == dict(arkit16=5, arkit_40777060=12, arkit_40777065=10)[held]
            old_threshold = load(previous/f'{held}_thresholds.json')['arms']['affine_margin_gridcal']['cutoff']
            assert thresholds[BASE]['cutoff'] == old_threshold
            checks['exact_prior_affine_fold_reproduction'] += 1
            post = {}
            for a, rr in all_data[held].items():
                neg = [r['query_score'] for r in rr if r['reference_state']==FREE]
                post[a] = main[BASE] if a == BASE else evaluate(rr, cutoff_at_cost(neg, target))
                for cost in range(len(neg)+1):
                    cutoff = cutoff_at_cost(neg, cost)
                    sm = summary(evaluate(rr, cutoff))['all']
                    assert sm['free_support'] <= cost
                    curves.append(dict(cohort=held, arm=a, negative_count_budget=cost, cutoff=cutoff,
                        actual_free_support=sm['free_support'], free_total=sm['free_total'], positive_known_witness=sm['positive_known_witness'], positive_total=sm['positive_total']))
            post_summary = {a: summary(rr) for a, rr in post.items()}
            assert post_summary[BASE]['all']['free_support'] == target
            assert post_summary[BASE]['all']['positive_known_witness'] == main_summary[BASE]['all']['positive_known_witness']
            pair_results, curve_results = {}, {}
            base_curve = {r['negative_count_budget']:r for r in curves if r['cohort']==held and r['arm']==BASE}
            for mode, evaluation in [('main', main), ('posthoc_same_cost', post)]:
                for a, rr in evaluation.items():
                    for group, sm in summary(rr).items():
                        tables.append(dict(cohort=held, arm=a, mode=mode, group=group, cutoff=rr[0]['cutoff'], affine_actual_free_target=target, **sm))
                    if a != BASE:
                        pp = paired(rr, evaluation[BASE], held, a, mode)
                        pairs.extend(pp)
                        pair_results[a+'/'+mode] = pair_summary(pp)
            for a in ARMS[1:]:
                cc = [r for r in curves if r['cohort']==held and r['arm']==a]
                diff = [r['positive_known_witness']-base_curve[r['negative_count_budget']]['positive_known_witness'] for r in cc]
                curve_results[a] = dict(full_cost_budget_range=[0, len(cc)-1], budgets_below_affine=sum(d<0 for d in diff),
                    budgets_equal_affine=sum(d==0 for d in diff), budgets_above_affine=sum(d>0 for d in diff),
                    minimum_witness_difference=min(diff), maximum_witness_difference=max(diff),
                    low_cost_range=[0, target], low_cost_budgets_below_affine=sum(d<0 for d in diff[:target+1]),
                    all_budgets_noninferior_descriptive=all(d>=0 for d in diff),
                    interpretation='Descriptive held-FREE-selected curve, correlated consumed Development; not statistical noninferiority')
            results[held] = dict(main=main_summary, posthoc_same_cost=post_summary, paired=pair_results, curves=curve_results,
                main_threshold_sha256=sha(threshold_path), posthoc_affine_actual_free_target=target)
            write(output/f'{held}_evaluation.json', dict(main=main, posthoc_same_cost=post, results=results[held]))
            print('A0_EVALUATION_COMPLETE', held, flush=True)
        write(output/'results.json', dict(cohorts=results, deltas=DELTAS, per_capture_delta_selection=False,
            main_vs_posthoc_separate=True, input_range_fallback=False))
        write_csv(output/'summary.csv', tables)
        write_csv(output/'query_pairs.csv', pairs)
        write_csv(output/'posthoc_curves.csv', curves)
        write(output/'focused_check.json', dict(status='PASS', checks=dict(checks),
            all_nine_delta_capture_combinations=True, zero_delta_affine_identity=True,
            threshold_files_frozen_before_held_evaluation=True, masks_equal=True))
        shutil.copyfile(__file__, output/'executed_bounded_residual_proxy.py')
        complete = True
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        write(output/'completion_receipt.json', dict(status='COMPLETE' if complete else 'PARTIAL', failure=failure,
            cpu_wall_s=time.perf_counter()-started, prior_attempt_cpu_wall_s=prior_cpu_s,
            budget_cpu_wall_s_remaining=budget,
            gpu_s=0, download_bytes=0, new_training=False, resources_released=True,
            placement='TASK_NOT_GPU_SUITABLE: cached array geometry and order statistics, no model inference',
            source_sha256=sha(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-budget-s', type=float, default=600.)
    parser.add_argument('--prior-attempt', type=Path)
    args = parser.parse_args()
    run(args.repo.resolve(), args.output.resolve(), args.cpu_budget_s,
        args.prior_attempt.resolve() if args.prior_attempt else None)
