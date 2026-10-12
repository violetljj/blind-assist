"""Fixed public-query distance-band calibration of frozen Development scores.

No model/pixel inputs. Calibration FREE alone selects each band cutoff;
held-FREE affine matches and LOEO are explicitly auxiliary diagnostics.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import time
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_negative_frozen_score import ARMS, EVAL, numeric, summaries, compare
from rgb_body_query_query_calibration_probe import FREE, COHORTS, select_cutoff, write
from rgb_body_query_bounded_residual_proxy import evaluate, cutoff_at_cost
from rgb_body_query_arkit_cal_frozen import pair_groups
from rgb_body_query_scene_diagnostic import write_csv

BANDS = ('0.3-0.8m', '0.8-1.5m', '1.5-3m')


def choices(rows):
    return {b: select_cutoff([r['query_score'] for r in rows
                             if r['distance_band'] == b and r['reference_state'] == FREE])
            for b in BANDS}


def band_evaluate(rows, selected):
    out = []
    for r in rows:
        cutoff = selected[r['distance_band']]['cutoff']
        out.append(dict(r, cutoff=cutoff,
            predicted_positive=cutoff is not None and r['query_score'] >= cutoff,
            positive_known_witness=cutoff is not None and r['known_positive_score'] >= cutoff))
    return out


def process(repo, out, check, receipt):
    work = repo/'artifacts.local/work'
    prior = work/'rgb-body-query-arkit-cal-dev-20261010/score'
    cached = work/'rgb-body-query-negative-score-dev-20261010/frozen-score'
    inputs = []
    def checked(path):
        check(); inputs.append(dict(path=str(path), sha256=sha(path)))
        return load(path)
    cal, cuts, globalcuts, evdata = {}, {}, {}, {}
    for kind in ('absolute', 'normalized'):
        cal[kind] = {a: numeric(rr) for a, rr in checked(prior/kind/'cal_scores.json')['arms'].items()}
        assert tuple(cal[kind]) == ARMS
        assert all(len(rr) == 8208 and set(r['distance_band'] for r in rr) == set(BANDS)
                   for rr in cal[kind].values())
        cuts[kind] = {a: choices(rr) for a, rr in cal[kind].items()}
        negative = [r for r in cal[kind]['affine'] if r['reference_state'] == FREE]
        assert len(negative) == 588
        assert [sum(r['distance_band'] == b for r in negative) for b in BANDS] == [395, 180, 13]
        globalcuts[kind] = checked(prior/kind/'thresholds.json')['arms']
        dest = out/kind; dest.mkdir()
        write(dest/'thresholds.json', dict(status='FROZEN_BEFORE_EVALUATION', frozen_utc=utc(),
            arms=cuts[kind], negative_queries=588, allowed_by_band=[19, 9, 0],
            actual_support={a: sum(s['false_support'] for s in cc.values()) for a, cc in cuts[kind].items()},
            band_from='Frozen public query geometry only, not predicted/reference point distance',
            rule='Each band nearest attainable <=5%, original tie policy, no budget redistribution',
            sparse_far='13 strict FREE: integer budget0; no relaxation; cal POS descriptive',
            original_LOCO_preserved=True, absolute_primary=True, normalized_auxiliary=True))
    write(out/'frozen_thresholds.json', dict(status='ALL30_FROZEN_BEFORE_EVAL', frozen_utc=utc(), arms=cuts))
    cal_ids = {(r['scan'], r['frame']) for r in cal['absolute']['affine']}
    eval_ids = set()
    for kind in ('absolute', 'normalized'):
        folder = cached/('cached' if kind == 'absolute' else 'normalized-probe')
        evdata[kind] = {}
        for c in EVAL:
            evdata[kind][c] = {a: numeric(rr) for a, rr in checked(folder/f'{c}_scores.json')['arms'].items()}
            assert all(len(rr) == len(evdata[kind][c]['affine']) for rr in evdata[kind][c].values())
            eval_ids.update((r['scan'], r['frame']) for r in evdata[kind][c]['affine'])
    assert len(cal_ids) == 304 and len(eval_ids) == 136 and cal_ids.isdisjoint(eval_ids)
    write(out/'identity_checks.json', dict(status='PASS', cal_frames=304, eval_frames=136,
        frame_roles_disjoint=True, original392_pixel_recompute=False, cache_only=True))
    results = {}
    for kind in ('absolute', 'normalized'):
        dest = out/kind; table, pairs, matched, loeo, loeo_cuts = [], [], [], [], []
        cohorts, pairresults = {}, {}
        for c in ('cal_fit_description',) + EVAL:
            check(); ss = cal[kind] if c == 'cal_fit_description' else evdata[kind][c]
            ev = {a: band_evaluate(rr, cuts[kind][a]) for a, rr in ss.items()}
            cohorts[c] = {a: summaries(rr) for a, rr in ev.items()}
            write(dest/f'{c}_evaluation.json', ev)
            def add_pair(a, baseline, baseline_name, mode):
                pp = compare(ev[a], baseline, c, a, baseline_name, mode)
                pairs.extend(pp); pairresults[f'{c}/{a}/{mode}'] = pair_groups(pp)
            for a in ARMS:
                for group, sm in cohorts[c][a].items():
                    table.append(dict(cohort=c, arm=a, group=group, **sm))
                add_pair(a, evaluate(ss[a], globalcuts[kind][a]['cutoff']),
                         f'{kind}_global588_{a}', 'vs_previous_global_same_arm')
                if a != 'affine':
                    add_pair(a, ev['affine'], f'{kind}_conditional_affine', 'vs_conditional_affine')
                if c in EVAL:
                    add_pair(a, evaluate(evdata['absolute'][c]['affine'], globalcuts['absolute']['affine']['cutoff']),
                             'absolute_global588_affine', 'vs_global_absolute_affine')
                if c in COHORTS:
                    raw = evdata['absolute'][c]['affine']
                    # Overall raw absolute-affine curve at candidate total FREE cost.
                    cost = cohorts[c][a]['all']['free_support']
                    mc = cutoff_at_cost([r['query_score'] for r in raw if r['reference_state'] == FREE], cost)
                    mev = evaluate(raw, mc)
                    matched.append(dict(cohort=c, candidate_arm=a, mode='overall_absolute_affine',
                        group='all', requested_free_budget=cost, cutoff=mc, **summaries(mev)['all']))
                    add_pair(a, mev, 'heldFREE_overall_absolute_affine', 'posthoc_overall_absolute_affine')
                    # Independently match each band at its actually achieved candidate cost.
                    selected = {}
                    for b in BANDS:
                        rr = [r for r in raw if r['distance_band'] == b]
                        cost = cohorts[c][a]['band/'+b]['free_support']
                        mc = cutoff_at_cost([r['query_score'] for r in rr if r['reference_state'] == FREE], cost)
                        selected[b] = dict(cutoff=mc)
                        ee = band_evaluate(rr, {b: selected[b]})
                        matched.append(dict(cohort=c, candidate_arm=a, mode='per_band_absolute_affine',
                            group=b, requested_free_budget=cost, cutoff=mc, **summaries(ee)['all']))
                    mev = band_evaluate(raw, selected)
                    matched.append(dict(cohort=c, candidate_arm=a, mode='per_band_absolute_affine',
                        group='all', requested_free_budget=cohorts[c][a]['all']['free_support'],
                        cutoff=None, **summaries(mev)['all']))
                    add_pair(a, mev, 'heldFREE_perband_absolute_affine', 'posthoc_perband_absolute_affine')
        neg = [r for r in cal[kind]['affine'] if r['reference_state'] == FREE]
        envs = sorted({r['environment'] for r in neg}); assert len(envs) == 15
        for env in envs:
            check()
            for a in ARMS:
                remaining = [r for r in cal[kind][a] if r['environment'] != env]
                cc = choices(remaining)
                held = [r for r in cal[kind][a] if r['environment'] == env and r['reference_state'] == FREE]
                heldev = band_evaluate(held, cc)
                for b in BANDS:
                    rr = [r for r in heldev if r['distance_band'] == b]
                    loeo_cuts.append(dict(held_negative_environment=env, arm=a, band=b,
                        held_free_total=len(rr), held_free_support=sum(r['predicted_positive'] for r in rr), **cc[b]))
                for c in COHORTS:
                    loeo.append(dict(held_negative_environment=env, arm=a, cohort=c,
                        held_free_total=len(held), held_free_support=sum(r['predicted_positive'] for r in heldev),
                        **summaries(band_evaluate(evdata[kind][c][a], cc))['all']))
        results[kind] = dict(status='COMPLETE', thresholds=cuts[kind], cohorts=cohorts, pairs=pairresults,
            auxiliary_loeo_environments=15, sparse_far_zero_budget=True,
            limits='Consumed Development; cal positives fitted-in; heldFREE same-cost auxiliary, no claim of independent transfer')
        write(dest/'results.json', results[kind]); write_csv(dest/'summary.csv', table)
        write_csv(dest/'query_pairs.csv', pairs); write_csv(dest/'posthoc_matched_absolute_affine.csv', matched)
        write_csv(dest/'loeo.csv', loeo); write_csv(dest/'loeo_cutoffs.csv', loeo_cuts)
    write(out/'inputs.json', inputs); write(out/'results.json', results)
    receipt.update(cal_frames=304, cal_queries=8208, cal_FREE=588, eval_frames=136, eval_queries=3672,
        thresholds=30, models_opened=0, pixels_opened=0, downloads=0, training=0, gpu=0)


def run(repo, out, budget, prior_attempt=None):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists(): raise FileExistsError('Preserve prior evidence')
    prior = 0.
    if prior_attempt:
        old = load(prior_attempt/'terminal.json'); assert old['status'] == 'FAILED_PARTIAL'
        prior = old['wall_s'] + old.get('prior_attempt_wall_s', 0.)
    assert budget > prior
    started = time.perf_counter(); receipt = dict(status='STARTING', prior_attempt_wall_s=prior)
    write(out/'plan.json', dict(frozen_utc=utc(), source_sha256=sha(__file__),
        parent_plan_sha256=sha(out.parent/'plan.json'), budget_s=budget, prior_attempt_wall_s=prior,
        ARMS=ARMS, BANDS=BANDS, rule='Each band FREE-only<=5%; no cost reallocation or readout/delta search'))
    shutil.copyfile(__file__, out/'executed_distance_cal.py')
    for f in (select_cutoff, evaluate, summaries, compare, pair_groups):
        p = Path(f.__code__.co_filename); shutil.copyfile(p, out/f'executed_dependency_{p.name}')
    def check():
        if time.perf_counter()-started+prior >= budget: raise TimeoutError('Cumulative score wall budget')
    try:
        process(repo, out, check, receipt); receipt['status'] = 'COMPLETE'
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL', error=repr(exc)); raise
    finally:
        receipt.update(wall_s=time.perf_counter()-started, completed_utc=utc(), source_sha256=sha(__file__))
        write(out/'terminal.json', receipt); print(receipt, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--budget-s', type=float, default=300.)
    p.add_argument('--prior-attempt', type=Path)
    a = p.parse_args(); run(a.repo.resolve(), a.output.resolve(), a.budget_s, a.prior_attempt)
