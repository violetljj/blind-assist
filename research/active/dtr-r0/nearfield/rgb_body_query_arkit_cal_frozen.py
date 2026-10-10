"""Conditional frozen ARKit calibration, consumed Development.

Only new fixed-frame scores are computed. Prior 392-frame scores are reused;
new pooled calibration supplements rather than replaces original capture LOCO.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import time
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_negative_frozen_score import (
    ARMS, EVAL, paths, head, score, numeric, summaries, compare)
from rgb_body_query_normalized_margin_probe import scores as normalized_scores
from rgb_body_query_query_calibration_probe import FREE, COHORTS, select_cutoff, write
from rgb_body_query_bounded_residual_proxy import evaluate, cutoff_at_cost
from rgb_body_query_scene_diagnostic import write_csv


def pair_groups(rows):
    out = {}
    for title, keys in [('all', []), ('band', ['distance_band']),
                        ('environment', ['environment']),
                        ('environment_band', ['environment', 'distance_band'])]:
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[k] for k in keys)].append(row)
        for key, rr in groups.items():
            out['/'.join([title] + list(key))] = {
                f: sum(r[f] for r in rr) for f in
                ('positive_rescue', 'positive_loss', 'free_removed', 'free_added')}
    return out


def scoring(repo, out, reference, predictions, check, receipt):
    work, root, candidate, _ = paths(repo)
    previous = work/'rgb-body-query-negative-score-dev-20261010/frozen-score'
    info_path = candidate/'training_inputs.json'
    inputs = []
    def checked(path):
        check(); inputs.append(dict(path=str(path), sha256=sha(path)))
        return load(path)
    info, manifest, pm = checked(info_path), checked(reference), checked(predictions)
    assert len(pm['rows']) == 48 and len(manifest['queries']) == 27
    identities = {(r['scan'], r['frame']) for r in pm['rows']}
    refs = {(r['scan'], r['frame']): r for r in manifest['rows']}
    assert len(identities) == 48 and identities <= refs.keys()
    manifest = dict(manifest, rows=[refs[r['scan'], r['frame']] for r in pm['rows']])
    assert all(r['split'] == 'additional_cal' for r in manifest['rows'])
    train_ids = {(r['scan'], r['frame']) for r in info['frames']}
    train_env = {r['environment'] for r in info['frames']}
    checks = Counter(); widths = []
    new = {
        'absolute': score('cal_arkit', manifest, pm, info, check, inputs, checks),
        'normalized': normalized_scores('cal_arkit', manifest, pm, info, check, inputs, checks, widths)}
    newnear = [r for r in new['absolute']['affine']
               if r['reference_state']==FREE and r['distance_band']=='0.3-0.8m']
    assert len({r['environment'] for r in newnear})>=2
    # Calibration and all cutoffs are saved before evaluation caches are read.
    data, cals, cutoffs = {}, {}, {}
    for kind in ('absolute', 'normalized'):
        folder = previous/'normalized-probe' if kind == 'normalized' else previous/'cached'
        first = checked(folder/'cal_original_scores.json')['arms']
        additional_path = previous/('normalized-probe' if kind == 'normalized' else 'final')/'cal_additional_scores.json'
        second = checked(additional_path)['arms']
        oldcal = {a: numeric(first[a])+numeric(second[a]) for a in ARMS}
        assert all(len(rr) == 6912 for rr in oldcal.values())
        assert identities.isdisjoint({(r['scan'], r['frame']) for r in oldcal['affine']})
        assert identities.isdisjoint(train_ids)
        newenv = {r['environment'] for r in new[kind]['affine']}
        assert newenv.isdisjoint(train_env | {r['environment'] for r in oldcal['affine']})
        cal = {a: oldcal[a]+new[kind][a] for a in ARMS}
        assert all(len(rr) == 8208 for rr in cal.values())
        cals[kind] = cal
        negative = [r for r in cal['affine'] if r['reference_state'] == FREE]
        choice = {a: select_cutoff([r['query_score'] for r in cal[a] if r['reference_state'] == FREE]) for a in ARMS}
        cutoffs[kind] = choice
        dest = out/kind; dest.mkdir()
        write(dest/'cal_arkit_scores.json', dict(arms=new[kind]))
        write(dest/'cal_scores.json', dict(arms=cal))
        write(dest/'thresholds.json', dict(status='FROZEN_BEFORE_EVALUATION', frozen_utc=utc(), arms=choice,
            strict_free_queries=len(negative), negative_environments=len({r['environment'] for r in negative}),
            negative_band_counts={b: dict(queries=sum(r['distance_band']==b for r in negative),
                environments=len({r['environment'] for r in negative if r['distance_band']==b}))
                for b in ('0.3-0.8m','0.8-1.5m','1.5-3m')},
            calibration='Original16 + additional3RScan240 + fixed newARKit48; FREE only, <=5%, ties retained',
            original_capture_LOCO_preserved=True, new_calibration_is_supplement=True,
            cal_POS_fitted_in_description_only=True, normalized_is_auxiliary=kind=='normalized',
            band_upper_bounds={'0.3-0.8m': .25 if kind=='absolute' else 1.,
                               '0.8-1.5m': .35 if kind=='absolute' else 1.,
                               '1.5-3m': .75 if kind=='absolute' else 1.},
            structurally_excluded_bands={a: [b for b, cap in
                [('0.3-0.8m',.25),('0.8-1.5m',.35),('1.5-3m',.75)]
                if choice[a]['cutoff'] > (cap if kind=='absolute' else 1.)] for a in ARMS}))
    write(out/'frozen_thresholds.json', dict(status='BOTH_FROZEN_BEFORE_EVALUATION', frozen_utc=utc(), arms=cutoffs))
    # These are query-score JSON only: no old pixel payload or model is opened.
    eval_ids, eval_env = set(), set()
    for kind in ('absolute', 'normalized'):
        folder = previous/('normalized-probe' if kind=='normalized' else 'cached')
        data[kind] = {}
        for name in EVAL:
            arms = checked(folder/f'{name}_scores.json')['arms']
            data[kind][name] = {a: numeric(rr) for a, rr in arms.items()}
            eval_ids.update((r['scan'], r['frame']) for r in data[kind][name]['affine'])
            eval_env.update(r['environment'] for r in data[kind][name]['affine'])
    assert identities.isdisjoint(eval_ids)
    assert {r['environment'] for r in manifest['rows']}.isdisjoint(eval_env)
    assert len(eval_ids)==136
    write(out/'identity_checks.json', dict(status='PASS', new_frames=48, eval_frames=136,
        original_cal_frames=256, combined_cal_frames=304, new_train_cal_eval_frames_disjoint=True,
        new_reference_visits=sorted({r['visit_id'] for r in manifest['rows']}),
        new_train_cal_eval_environment_labels_disjoint=True,
        caveat='ARKit distinct visit metadata does not certify independent physical environments'))
    results = {}
    for kind in ('absolute', 'normalized'):
        dest = out/kind; cal = cals[kind]; choice = cutoffs[kind]
        table, pairs, pair_results, cohorts, loeo, matched, matched_absolute = [], [], {}, {}, [], [], []
        priorfolder = previous/('normalized-probe' if kind=='normalized' else 'final')
        priorcuts = checked(priorfolder/'thresholds.json')['arms']
        for name in ('cal_fit_description', 'cal_arkit_fit_description')+EVAL:
            check()
            ss = cal if name=='cal_fit_description' else new[kind] if name=='cal_arkit_fit_description' else data[kind][name]
            ev = {a: evaluate(rr, choice[a]['cutoff']) for a, rr in ss.items()}
            write(dest/f'{name}_evaluation.json', ev)
            cohorts[name] = {a: summaries(rr) for a, rr in ev.items()}
            for a, rr in ev.items():
                for group, summary in cohorts[name][a].items():
                    table.append(dict(cohort=name, arm=a, group=group, cutoff=choice[a]['cutoff'], **summary))
                if a!='affine':
                    pp = compare(rr, ev['affine'], name, a, f'{kind}_newcal_affine', 'newcal_vs_same_readout_affine')
                    pairs += pp; pair_results[f'{name}/{a}/vs_newcal_affine'] = pair_groups(pp)
                if name in EVAL:
                    oldev = evaluate(ss[a], priorcuts[a]['cutoff'])
                    pp = compare(rr, oldev, name, a, f'{kind}_previous32FREE_{a}', 'newcal_vs_previous32FREE_same_arm')
                    pairs += pp; pair_results[f'{name}/{a}/vs_previous32FREE'] = pair_groups(pp)
            if name in COHORTS:
                original = checked(work/f'rgb-body-query-a0-dev-20261010/repair-2/{name}_evaluation.json')['main']
                oldpath = work/f'rgb-body-query-migration-diagnostic-dev-20261010/query/{name}_thresholds.json'
                oldcut = checked(oldpath)['arms']['depth_ray_mean_margin_gridcal']['cutoff']
                original['old_depth_ray'] = evaluate(data['absolute'][name]['old_depth_ray'], oldcut)
                for a, rr in ev.items():
                    pp = compare(rr, original[a], name, a, f'original_absolute_LOCO_{a}', 'newcal_vs_original_absolute_LOCO_same_arm')
                    pairs += pp; pair_results[f'{name}/{a}/vs_original_LOCO'] = pair_groups(pp)
                    # Posthoc held-FREE same-cost affine: auxiliary only.
                    cost = cohorts[name][a]['all']['free_support']
                    mc = cutoff_at_cost([r['query_score'] for r in data[kind][name]['affine'] if r['reference_state']==FREE], cost)
                    mev = evaluate(data[kind][name]['affine'], mc)
                    matched.append(dict(cohort=name, candidate_arm=a, baseline_readout=kind, requested_free_budget=cost,
                        affine_cutoff=mc, **summaries(mev)['all']))
                    pp = compare(rr, mev, name, a, f'{kind}_heldFREE_matched_affine', 'posthoc_heldFREE_same_cost')
                    pairs += pp; pair_results[f'{name}/{a}/posthoc_samecost_affine'] = pair_groups(pp)
                    # Retain the stronger original absolute-affine score curve,
                    # including when comparing an auxiliary normalized candidate.
                    abscores = data['absolute'][name]['affine']
                    ac = cutoff_at_cost([r['query_score'] for r in abscores if r['reference_state']==FREE], cost)
                    aev = evaluate(abscores, ac)
                    matched_absolute.append(dict(cohort=name, candidate_arm=a,
                        candidate_readout=kind, baseline_readout='absolute', requested_free_budget=cost,
                        affine_cutoff=ac, **summaries(aev)['all']))
                    pp = compare(rr, aev, name, a, 'absolute_heldFREE_matched_affine', 'posthoc_heldFREE_absolute_affine_same_cost')
                    pairs += pp; pair_results[f'{name}/{a}/posthoc_samecost_absolute_affine'] = pair_groups(pp)
        negative = [r for r in cal['affine'] if r['reference_state']==FREE]
        for env in sorted({r['environment'] for r in negative}):
            check()
            for a in ARMS:
                ss = [r['query_score'] for r in cal[a] if r['reference_state']==FREE and r['environment']!=env]
                selected = select_cutoff(ss)
                held = [r for r in cal[a] if r['reference_state']==FREE and r['environment']==env]
                heldev = evaluate(held, selected['cutoff'])
                for name in COHORTS:
                    sm = summaries(evaluate(data[kind][name][a], selected['cutoff']))['all']
                    loeo.append(dict(held_negative_environment=env, arm=a, cohort=name,
                        cal_negative_queries=len(ss), cutoff=selected['cutoff'], held_free_total=len(held),
                        held_free_support=sum(r['predicted_positive'] for r in heldev), **sm))
        results[kind] = dict(status='COMPLETE', thresholds=choice, cohorts=cohorts, pairs=pair_results,
            negative_calibration=summaries(evaluate(cal['affine'], choice['affine']['cutoff'])),
            auxiliary_loeo_environments=len({r['environment'] for r in negative}),
            original_LOCO='Absolute original protocol retained, normalized comparison crosses readout definitions',
            limits='Consumed Development; fitted-in cal positives descriptive; sampled FREE not volume clearance; no newdelta/readout search')
        write(dest/'results.json', results[kind]); write_csv(dest/'summary.csv', table)
        write_csv(dest/'query_pairs.csv', pairs); write_csv(dest/'loeo.csv', loeo)
        write_csv(dest/'posthoc_matched_affine.csv', matched)
        write_csv(dest/'posthoc_matched_absolute_affine.csv', matched_absolute)
    write_csv(out/'width_diagnostics.csv', widths); write(out/'checks.json', dict(checks))
    write(out/'inputs.json', inputs); write(out/'results.json', results)
    receipt.update(frames_scored_new=48, original_frames_recomputed=0, cal_queries=8208,
        eval_queries=3672, checks=dict(checks), training=0, downloads=0)


def run(repo, out, stage, budget, source=None, reference=None, predictions=None, prior_attempt=None):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise FileExistsError('Preserve completed and failed stage payloads')
    prior = 0.
    if prior_attempt:
        earlier = load(prior_attempt/'terminal.json')
        assert earlier['status']=='FAILED_PARTIAL' and earlier['stage']==stage
        prior = earlier['stage_wall_s']+earlier.get('prior_attempt_wall_s',0.)
        budget -= prior; assert budget>0
    write(out/'plan.json', dict(frozen_utc=utc(), stage=stage, source_sha256=sha(__file__),
        reused_source_sha256={str(Path(f.__code__.co_filename).name): sha(f.__code__.co_filename)
                             for f in (head, normalized_scores)},
        budget_wall_s_remaining=budget, prior_attempt_wall_s=prior, arms=ARMS,
        new_frames=48, old_pixel_recompute=False, training=0, downloads=0,
        absolute_primary=True, normalized_auxiliary=True, new_readout_search=False,
        original_capture_LOCO_preserved=True, cal_POS_descriptive_only=True,
        gpu_budget_s=budget if stage=='head' else 0))
    shutil.copyfile(__file__, out/'executed_arkit_cal_frozen.py')
    for function in (head, normalized_scores):
        path = Path(function.__code__.co_filename)
        shutil.copyfile(path, out/f'executed_dependency_{path.name}')
    started = time.perf_counter(); receipt=dict(status='STARTING',stage=stage,prior_attempt_wall_s=prior)
    def check():
        if time.perf_counter()-started>=budget:
            raise TimeoutError(f'{stage} cumulative stage budget reached')
    try:
        if stage=='head':
            assert source is not None
            head(repo, out, check, receipt, source=source, expected_frames=48)
        else:
            assert reference is not None and predictions is not None
            scoring(repo, out, reference, predictions, check, receipt)
        receipt['status']='COMPLETE'
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL',error=repr(exc)); raise
    finally:
        receipt.update(stage_wall_s=time.perf_counter()-started,completed_utc=utc(),source_sha256=sha(__file__))
        write(out/'terminal.json',receipt); print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--stage',choices=('head','score'),required=True)
    p.add_argument('--source',type=Path); p.add_argument('--reference',type=Path); p.add_argument('--predictions',type=Path)
    p.add_argument('--prior-attempt',type=Path); p.add_argument('--budget-s',type=float,required=True)
    a=p.parse_args(); run(a.repo.resolve(),a.output.resolve(),a.stage,a.budget_s,a.source,a.reference,a.predictions,a.prior_attempt)
