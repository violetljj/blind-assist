"""Frozen public-only head inference; separate evaluator scoring/calibration stages.

Consumed Development. Global thresholds use only original cal plus additional
strict sampled FREE. No training, downloads or evaluation-selected thresholds.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import time
import numpy as np
from rgb_body_query_interval_distribution import load, utc, public_depth
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_fixed_grid import PUBLIC
from rgb_body_query_query_calibration_probe import FREE, COHORTS, nth_score, select_cutoff, write
from rgb_body_query_reference_eval import rays, ray_interval
from rgb_body_query_bounded_residual_proxy import evaluate, paired, pair_summary
from rgb_body_query_scene_diagnostic import write_csv

ARMS = ('affine', 'old_depth_ray', 'a0_delta_0.05', 'a0_delta_0.1', 'a0_delta_0.2')
HEAD_SHA = 'e2cf2d3deed5d830dc01d62d1cf7dd9baba17255b41615c11f2f5263c722b1d4'
EVAL = ('original_validation24', 'new_3rscan64') + COHORTS


def numeric(rows):
    return [dict(r, **{k: (-np.inf if r[k] == '-Infinity' else r[k])
                      for k in ('query_score', 'known_positive_score')}) for r in rows]


def summaries(rows):
    out = {}
    for title, keys in [('all', []), ('environment', ['environment']),
                        ('band', ['distance_band']), ('environment_band', ['environment', 'distance_band'])]:
        groups = defaultdict(list)
        for r in rows: groups[tuple(r[k] for k in keys)].append(r)
        for key, rr in groups.items():
            pos = [r for r in rr if r['reference_state'] == 'POSITIVE']
            free = [r for r in rr if r['reference_state'] == FREE]
            unk = [r for r in rr if r['reference_state'] == 'UNKNOWN']
            out['/'.join([title] + list(key))] = dict(query_total=len(rr),
                frames=len({(r['scan'], r['frame']) for r in rr}), environments=len({r['environment'] for r in rr}),
                positive_total=len(pos), positive_known_witness=sum(r['positive_known_witness'] for r in pos),
                positive_support=sum(r['predicted_positive'] for r in pos), free_total=len(free),
                free_support=sum(r['predicted_positive'] for r in free), unknown_total=len(unk),
                unknown_support=sum(r['predicted_positive'] for r in unk),
                unavailable_queries=sum(r['valid_ray_count'] < 16 for r in rr))
    return out


def compare(rows, baseline, cohort, arm, baseline_arm, mode):
    pp = paired(rows, baseline, cohort, arm, mode)
    for r in pp:
        r['baseline_arm'] = baseline_arm
        for key in list(r):
            if key.startswith('affine_'):
                r['baseline_' + key.removeprefix('affine_')] = r.pop(key)
    return pp


def paths(repo):
    work = repo/'artifacts.local/work'
    root = work/'rgb-body-query-interval-distribution-dev-20261010'
    refs = work/'rgb-body-query-query-level-dev-20261009'
    cohorts = {n: refs/'fixed-grid-sensor'/n for n in EVAL[:3]}
    cohorts.update({n: refs/'additional-arkit-sensor'/n.removeprefix('arkit_') for n in EVAL[3:]})
    return work, root, root/'candidate', cohorts


def score(cohort, manifest, pm, info, check, inputs, checks):
    refs = {(r['scan'], r['frame']): r for r in manifest['rows']}
    data = {a: [] for a in ARMS}
    for row in pm['rows']:
        check(); ref = refs[row['scan'], row['frame']]
        for path, expected in [(ref['reference_path'], ref['reference_sha256']),
                               (row['sampled_depth_path'], row['sampled_depth_sha256']),
                               (row['distributions']['depth_ray_gaussian']['path'], row['distributions']['depth_ray_gaussian']['sha256'])]:
            assert sha(path) == expected
            inputs.append(dict(path=str(path), sha256=expected))
        with np.load(ref['reference_path']) as f: labels = f['labels']
        with np.load(row['sampled_depth_path']) as f: dp = f['depth']
        with np.load(row['distributions']['depth_ray_gaussian']['path']) as f: mu, sigma, drvalid = f['mu'], f['sigma'], f['valid']
        assert np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all()
        checks['finite_mu_positive_finite_sigma'] += 1
        valid = np.isfinite(dp) & (dp > 0)
        assert np.array_equal(valid, drvalid); checks['matching_masks'] += 1
        loga = np.zeros(dp.shape, np.float64)
        loga[valid] = info['affine']['a']*np.log(dp[valid].astype(np.float64))+info['affine']['b']
        depths = {'affine': np.exp(loga), 'old_depth_ray': np.exp(mu)}
        for delta in (.05, .1, .2): depths[f'a0_delta_{delta:g}'] = np.exp(loga + np.clip(mu-loga, -delta, delta))
        rx, ry = rays(row['depth_K'], row['depth_shape'])
        for j, q in enumerate(manifest['queries']):
            check(); entry, exit, domain = ray_interval(rx, ry, q); lab = labels[j]
            pos, free, unknown = [int((lab == n).sum()) for n in (1, 0, 2)]
            state = 'POSITIVE' if pos >= 16 else (FREE if pos == 0 and unknown == 0 and free >= 16 else 'UNKNOWN')
            assert state == ref['queries'][j]['state']; checks['strict_query_states'] += 1
            mask = valid & domain
            for arm, depth in depths.items():
                margin = np.full(dp.shape, -np.inf)
                margin[mask] = np.minimum(depth[mask]-entry[mask], exit[mask]-depth[mask])
                qs, ws = nth_score(margin), nth_score(margin[lab == 1])
                data[arm].append(dict(cohort=cohort, environment=ref['environment'], scan=row['scan'], frame=row['frame'],
                    query=q['name'], distance_band=f'{q["low"][2]:g}-{q["high"][2]:g}m', reference_state=state,
                    valid_ray_count=int(np.isfinite(margin).sum()), query_score=qs, known_positive_score=ws))
    return data


def head(repo, output, check, receipt, source=None, expected_frames=240):
    work, root, candidate, _ = paths(repo)
    source = source if source is not None else work/'rgb-body-query-negative-score-dev-20261010/additional-inference'
    # This stage never opens evaluator manifests or reference payloads.
    obs = load(source/'observations.json'); dp = load(source/'predictions.json')
    assert dp['status'] == 'COMPLETE' and load(source/'terminal.json')['status'] == 'COMPLETE'
    assert not (source/'writer.json').exists()
    assert len(obs['rows']) == len(dp['rows']) == expected_frames
    assert all(set(r) <= set(PUBLIC) and r['split'] == 'additional_cal' for r in obs['rows'])
    info = load(candidate/'training_inputs.json'); gpu = load(candidate/'gpu_receipt.json')
    checkpoint = candidate/'depth_ray_gaussian_final.pt'
    assert sha(checkpoint) == gpu['arms']['depth_ray_gaussian']['sha256'] == HEAD_SHA
    receipt['inputs'] = {str(p): sha(p) for p in [source/'observations.json', source/'predictions.json', candidate/'training_inputs.json', checkpoint]}
    import torch
    import gc
    torch.set_num_threads(4)
    model = tensor = raw = None
    gpu_started = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        model = torch.nn.Sequential(torch.nn.Linear(3,64), torch.nn.ReLU(), torch.nn.Linear(64,64), torch.nn.ReLU(), torch.nn.Linear(64,2)).cuda()
        state = torch.load(checkpoint, map_location='cuda', weights_only=True)
        assert state['normalization'] == info['normalization']
        # JSON acquired provenance hashes after checkpoint serialization; every
        # stored training parameter/identity must still match exactly.
        assert all(info[k] == v for k, v in state['initial'].items())
        assert set(info)-set(state['initial']) <= {'manifest_sha256', 'prediction_manifest_sha256'}
        model.load_state_dict(state['state_dict']); model.eval().requires_grad_(False)
        mean, std = [info['normalization'][k] for k in ('log_dp_mean', 'log_dp_std')]
        initial = info['sigma_initial']; sigma_logit = float(np.log((initial-.03)/(2.-initial)))
        def forward(depth, row):
            nonlocal tensor, raw
            check(); valid = np.isfinite(depth) & (depth > 0); log = np.log(np.where(valid, depth, 1.).astype(np.float64))
            rx, ry = rays(row['depth_K'], row['depth_shape'])
            inp = np.stack([(log-mean)/std, rx, ry], -1).reshape(-1,3).astype(np.float32)
            tensor = torch.tensor(inp, device='cuda')
            with torch.inference_mode(): raw = model(tensor).cpu().numpy().astype(np.float64)
            mu = np.full(depth.shape, info['geometry_mean']) + raw[:,0].reshape(depth.shape)
            sig = .03+1.97/(1.+np.exp(-(raw[:,1].reshape(depth.shape)+sigma_logit)))
            assert np.isfinite(mu).all() and np.isfinite(sig).all() and (sig > 0).all()
            tensor = raw = None
            return mu, sig, valid
        # Reproduction check reads only already-public cached model outputs.
        old = load(candidate/'predictions/cal/predictions.json')['rows'][0]
        with np.load(old['sampled_depth_path']) as f: depth = f['depth']
        mu, sig, valid = forward(depth, old)
        dist = old['distributions']['depth_ray_gaussian']
        assert sha(dist['path']) == dist['sha256']
        with np.load(dist['path']) as f:
            errors = dict(mu_max_abs=float(np.max(np.abs(f['mu']-mu))), sigma_max_abs=float(np.max(np.abs(f['sigma']-sig))))
            assert np.array_equal(f['valid'], valid)
            assert np.allclose(f['mu'], mu, atol=2e-6, rtol=0) and np.allclose(f['sigma'], sig, atol=2e-6, rtol=0)
        write(output/'reproduction_check.json', dict(status='PASS', fixed_tolerance_absolute=2e-6, **errors,
            scan=old['scan'], frame=old['frame'], parameters_changed=False))
        lookup = {(r['scan'], r['frame']): r for r in dp['rows']}; saved = []
        for r in obs['rows']:
            check(); p = lookup[r['scan'], r['frame']]; depth = public_depth(r, p)
            mu, sig, valid = forward(depth, r)
            stem = f'{r["scan"]}_{r["frame"]:06d}'
            dpath = output/f'depth_{stem}.npz'; mpath = output/f'depth_ray_gaussian_{stem}.npz'
            np.savez(dpath, depth=depth); np.savez(mpath, mu=mu, sigma=sig, valid=valid)
            saved.append(dict(r, sampled_depth_path=str(dpath), sampled_depth_sha256=sha(dpath),
                depthpro_source_path=p['path'], depthpro_source_sha256=p['sha256'],
                distributions=dict(depth_ray_gaussian=dict(path=str(mpath), sha256=sha(mpath)))))
        write(output/'predictions.json', dict(status='COMPLETE', rows=saved, evaluator_reads=False,
            queries_not_network_inputs=True, source_predictions_sha256=sha(source/'predictions.json'), checkpoint_sha256=HEAD_SHA))
        receipt['frames'] = len(saved)
    finally:
        model = tensor = raw = state = None; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        receipt.update(gpu_allocation_wall_s=time.perf_counter()-gpu_started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            cuda_bytes_after_cleanup=torch.cuda.memory_allocated(), resources_released=True)


def cached(repo, output, check, receipt):
    work, root, candidate, cohorts = paths(repo); info = load(candidate/'training_inputs.json')
    inputs, checks = [], Counter()
    for name, folder in [('cal_original', root/'train-cal-sensor')] + list(cohorts.items()):
        check(); mpath = folder/'dataset_manifest.json'; pname = 'cal' if name == 'cal_original' else name
        pm_path = candidate/'predictions'/pname/'predictions.json'
        manifest, pm = load(mpath), load(pm_path)
        inputs.extend([dict(path=str(p), sha256=sha(p)) for p in [mpath, pm_path]])
        data = score(name, manifest, pm, info, check, inputs, checks)
        if name in COHORTS:
            a0 = load(work/f'rgb-body-query-a0-dev-20261010/repair-2/{name}_scores.json')['arms']
            prev = load(work/f'rgb-body-query-migration-diagnostic-dev-20261010/query/{name}_scores.json')['arms']
            for arm in ARMS:
                prior = numeric(prev['depth_ray_mean_margin_gridcal']) if arm == 'old_depth_ray' else numeric(a0[arm])
                for r, p in zip(data[arm], prior, strict=True):
                    assert all(r[k] == p[k] for k in ('scan','frame','query','query_score','known_positive_score','reference_state'))
                    checks['exact_prior_scores'] += 1
        write(output/f'{name}_scores.json', dict(arms=data))
        print('CACHED_SCORED', name, len(pm['rows']), flush=True)
    write(output/'inputs.json', inputs); write(output/'checks.json', dict(checks)); receipt['checks'] = dict(checks)


def final(repo, output, check, receipt):
    work, root, candidate, _ = paths(repo); base = output.parent
    info = load(candidate/'training_inputs.json'); checks = Counter(); inputs = []
    manifest_path = work/'rgb-body-query-negative-reference-dev-20261010/dataset_manifest.json'
    manifest = load(manifest_path)
    head_stages = [p for p in base.glob('head*') if (p/'terminal.json').exists() and load(p/'terminal.json')['status']=='COMPLETE']
    assert len(head_stages) == 1
    head_prediction_path = head_stages[0]/'predictions.json'
    pm = load(head_prediction_path)
    inputs.extend([dict(path=str(p), sha256=sha(p)) for p in [manifest_path, head_prediction_path, candidate/'training_inputs.json']])
    for name in ('cal_original',)+EVAL:
        p = base/f'cached/{name}_scores.json'
        inputs.append(dict(path=str(p), sha256=sha(p)))
    additional = score('cal_additional', manifest, pm, info, check, inputs, checks)
    write(output/'cal_additional_scores.json', dict(arms=additional))
    data = {n: {a: numeric(rr) for a, rr in load(base/f'cached/{n}_scores.json')['arms'].items()}
            for n in ('cal_original',)+EVAL}
    cal = {a: data['cal_original'][a]+additional[a] for a in ARMS}
    assert all(len(rr) == 6912 for rr in cal.values())
    negs = [r for r in cal['affine'] if r['reference_state'] == FREE]
    assert len(negs) == 32 and len({r['environment'] for r in negs}) == 12
    cutoffs = {a: select_cutoff([r['query_score'] for r in cal[a] if r['reference_state'] == FREE]) for a in ARMS}
    write(output/'thresholds.json', dict(status='FROZEN_BEFORE_EVALUATION', frozen_utc=utc(), arms=cutoffs,
        selection='Original cal16 + additional cal240 strict sampled FREE only; no train/eval/POS/UNKNOWN; retain ties',
        cal_free_queries=32, cal_free_environments=12, near_free_queries=6, near_free_environments=1,
        training=False, distance_conditioned_calibration='NOT_RUN; near-band coverage inadequate'))
    results, tables, pairs, pair_results = {}, [], [], {}
    for name in ('cal_fit_description',)+EVAL:
        check(); scores = cal if name == 'cal_fit_description' else data[name]
        ev = {a: evaluate(rr, cutoffs[a]['cutoff']) for a, rr in scores.items()}
        write(output/f'{name}_evaluation.json', ev)
        results[name] = {a: summaries(rr) for a, rr in ev.items()}
        for a, rr in ev.items():
            for group, sm in results[name][a].items(): tables.append(dict(cohort=name, arm=a, group=group, cutoff=cutoffs[a]['cutoff'], **sm))
            if a != 'affine':
                pp = compare(rr, ev['affine'], name, a, 'affine', 'new_global_cal_vs_affine')
                pairs += pp; pair_results[f'{name}/{a}/new_vs_affine'] = pair_summary(pp)
        if name in COHORTS:
            previous = load(work/f'rgb-body-query-a0-dev-20261010/repair-2/{name}_evaluation.json')['main']
            oldhead_scores = numeric(load(work/f'rgb-body-query-migration-diagnostic-dev-20261010/query/{name}_scores.json')['arms']['depth_ray_mean_margin_gridcal'])
            oldhead_cutoff = load(work/f'rgb-body-query-migration-diagnostic-dev-20261010/query/{name}_thresholds.json')['arms']['depth_ray_mean_margin_gridcal']['cutoff']
            previous['old_depth_ray'] = evaluate(oldhead_scores, oldhead_cutoff)
            for a, rr in ev.items():
                pp = compare(rr, previous[a], name, a, a, 'new_global_vs_old_same_arm_LOCO')
                pairs += pp; pair_results[f'{name}/{a}/new_vs_old_LOCO'] = pair_summary(pp)
            results[name]['previous_same_arm_LOCO'] = {a: summaries(rr) for a, rr in previous.items()}
        elif name in EVAL:
            # Earlier 3RScan workpoints were ray-level original-cal selections;
            # they are descriptive comparators, not imaginary query LOCO.
            primary = load(candidate/'calibration.json')['arms']
            point = load(root/'mean-probe/calibration.json')['arms']
            inherited = {'affine': primary['affine_margin_gridcal']['cutoff'],
                         'old_depth_ray': point['depth_ray_mean_margin_gridcal']['cutoff']}
            results[name]['old_query_LOCO'] = 'NOT_AVAILABLE'
            for a, cutoff in inherited.items():
                oldev = evaluate(data[name][a], cutoff)
                pp = compare(ev[a], oldev, name, a, a, 'new_query_cal_vs_old_original_cal_ray_workpoint')
                pairs += pp; pair_results[f'{name}/{a}/new_vs_old_original_cal_ray_workpoint'] = pair_summary(pp)
                results[name][f'old_original_cal_ray_workpoint/{a}'] = summaries(oldev)
    # Auxiliary LOEO stability changes no global threshold or delta selection.
    loeo = []
    for environment in sorted({r['environment'] for r in negs}):
        check()
        for a in ARMS:
            trainfree = [r['query_score'] for r in cal[a] if r['reference_state']==FREE and r['environment'] != environment]
            choice = select_cutoff(trainfree)
            held = [r for r in cal[a] if r['reference_state']==FREE and r['environment'] == environment]
            heldev = evaluate(held, choice['cutoff'])
            for n in COHORTS:
                sm = summaries(evaluate(data[n][a], choice['cutoff']))['all']
                loeo.append(dict(held_negative_environment=environment, arm=a, cohort=n,
                    cal_negative_queries=len(trainfree), cutoff=choice['cutoff'], held_free_total=len(held),
                    held_free_support=sum(r['predicted_positive'] for r in heldev), **sm))
    write(output/'results.json', dict(status='COMPLETE', thresholds=cutoffs, cohorts=results, pairs=pair_results,
        auxiliary_loeo='12 negative environments; only FREE select thresholds; cannot change main or choose delta',
        limits='Development; source shift and related frames; sampled FREE is not volume clearance; cal positives fitted-in description'))
    write_csv(output/'summary.csv', tables); write_csv(output/'query_pairs.csv', pairs)
    write_csv(output/'loeo.csv', loeo); write(output/'inputs.json', inputs); write(output/'checks.json', dict(checks))
    receipt.update(cal_queries=6912, cal_free=32, eval_queries=3672, checks=dict(checks))


def run(repo, out, stage, budget, prior_attempt=None):
    out.mkdir(parents=True, exist_ok=True)
    planpath = out/'plan.json'
    if planpath.exists(): raise FileExistsError('Preserve stage payload; completed stages reusable, failed stage needs distinct attempt path')
    prior_wall = 0.
    if prior_attempt is not None:
        prior = load(prior_attempt/'terminal.json')
        assert prior['status'] == 'FAILED_PARTIAL' and prior['stage']==stage
        prior_wall = prior['stage_wall_s']+prior.get('prior_attempt_wall_s', 0.)
        budget -= prior_wall
        assert budget > 0
    write(planpath, dict(stage=stage, frozen_utc=utc(), source_sha256=sha(__file__), arms=ARMS,
        stage_budget_wall_s=budget, training=0, downloads=0, gpu_allocation_budget_s=budget if stage=='head' else 0,
        evaluator_read=stage!='head', new_inference_frames=240 if stage=='head' else 0,
        query_bounds_network_inputs=False, global_only=True, prior_attempt_wall_s=prior_wall,
        prior_attempt=None if prior_attempt is None else str(prior_attempt)))
    shutil.copyfile(__file__, out/'executed_negative_frozen_score.py')
    start = time.perf_counter(); receipt = dict(status='STARTING', stage=stage, prior_attempt_wall_s=prior_wall)
    def check():
        if time.perf_counter()-start >= budget: raise TimeoutError(f'{stage} stage wall budget reached')
    try:
        {'head': head, 'cached': cached, 'final': final}[stage](repo, out, check, receipt)
        receipt['status'] = 'COMPLETE'
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL', error=repr(exc)); raise
    finally:
        receipt.update(stage_wall_s=time.perf_counter()-start, completed_utc=utc(), source_sha256=sha(__file__))
        write(out/'terminal.json', receipt); print(receipt, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--stage', choices=('head','cached','final'), required=True)
    parser.add_argument('--prior-attempt', type=Path)
    parser.add_argument('--budget-s', type=float, required=True); args = parser.parse_args()
    run(args.repo.resolve(), args.output.resolve(), args.stage, args.budget_s, args.prior_attempt)
