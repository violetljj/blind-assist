"""Frozen-score distance-band calibration alternative, consumed Development.

Executed after negative-reference preparation. Newly prepared references have
no image-depth predictions and are not used in this pilot's calibration.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_query_calibration_probe import COHORTS, FREE, select_cutoff, summary, write
from rgb_body_query_bounded_residual_proxy import ARMS, paired, pair_summary
from rgb_body_query_scene_diagnostic import write_csv


def decode(rows):
    return [dict(r, **{k: -np.inf if r[k] == '-Infinity' else r[k]
                     for k in ('query_score', 'known_positive_score')}) for r in rows]


def run(repo, a0, output, negative_reference, budget):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists():
        raise FileExistsError('Preserve distance-conditioned pilot')
    receipt = load(negative_reference/'terminal.json')
    assert receipt['status'] == 'COMPLETE'
    previous = repo/'artifacts.local/work/rgb-body-query-migration-diagnostic-dev-20261010/query'
    arms = ARMS + ('old_depth_ray',)
    write(output/'plan.json', dict(created_utc=utc(), phase='AFTER_NEGATIVE_REFERENCE_PREPARATION',
        budget_cpu_wall_s=budget, gpu_s=0, downloads=0, new_training=False,
        negative_reference_receipt_sha256=sha(negative_reference/'terminal.json'),
        new_reference_scores='NOT_RUN: no new Depth Pro or model inference; new references never assigned old scores',
        cohorts=COHORTS, arms=arms, target=.05,
        selection='For each public query Z band, strict FREE from other two captures only; all ties retained; empty band => NOT_EVALUABLE',
        comparison='Same-arm original global-cutoff point and same-fold distance-conditioned affine; complete bands and paired queries',
        limits='Exploratory calibration method changes; already consumed same-family captures; not a validation on the new reference environments',
        source_sha256=sha(__file__)))
    complete, failure = False, None
    tables, pairs, results = [], [], {}
    try:
        data, old_main, inputs = {}, {}, []
        for c in COHORTS:
            data[c] = {a: decode(rr) for a, rr in load(a0/f'{c}_scores.json')['arms'].items()}
            data[c]['old_depth_ray'] = decode(load(previous/f'{c}_scores.json')['arms']['depth_ray_mean_margin_gridcal'])
            old_main[c] = load(a0/f'{c}_evaluation.json')['main']
            old_main[c]['old_depth_ray'] = load(previous/f'{c}_evaluation.json')['records']['depth_ray_mean_margin_gridcal/query_calibrated']
            inputs.extend(dict(path=str(p), sha256=sha(p)) for p in (a0/f'{c}_scores.json', a0/f'{c}_evaluation.json', previous/f'{c}_scores.json', previous/f'{c}_evaluation.json'))
        bands = sorted({r['distance_band'] for r in data[COHORTS[0]]['affine']})
        for held in COHORTS:
            if time.perf_counter()-started >= budget:
                raise TimeoutError('Distance-conditioned probe CPU budget reached')
            cal = [c for c in COHORTS if c != held]
            thresholds = {a: {band: select_cutoff([r['query_score'] for c in cal for r in data[c][a]
                          if r['reference_state']==FREE and r['distance_band']==band]) for band in bands} for a in arms}
            path = output/f'{held}_thresholds.json'
            write(path, dict(frozen_utc=utc(), held=held, cal_captures=cal, arms=thresholds))
            evaluated = {}
            for a, rr in data[held].items():
                evaluated[a] = []
                for r in rr:
                    cutoff = thresholds[a][r['distance_band']]['cutoff']
                    if cutoff is None:
                        raise ValueError('Band not evaluable; do not label unavailable as negative')
                    evaluated[a].append(dict(r, cutoff=cutoff,
                        predicted_positive=bool(r['query_score']>=cutoff),
                        positive_known_witness=bool(r['known_positive_score']>=cutoff)))
                for group, sm in summary(evaluated[a]).items():
                    tables.append(dict(cohort=held, arm=a, group=group, **sm))
            pair_results = {}
            for a in arms:
                same = paired(evaluated[a], old_main[held][a], held, a, 'same_arm_global_cutoff')
                pairs.extend(same)
                pair_results[a+'/same_arm_global_cutoff'] = pair_summary(same)
                if a != 'affine':
                    pp = paired(evaluated[a], evaluated['affine'], held, a, 'same_band_cal_affine')
                    pairs.extend(pp)
                    pair_results[a+'/same_band_cal_affine'] = pair_summary(pp)
            results[held] = dict(arms={a:summary(rr) for a, rr in evaluated.items()}, paired=pair_results,
                threshold_sha256=sha(path))
            write(output/f'{held}_evaluation.json', dict(records=evaluated, results=results[held]))
        write(output/'inputs.json', inputs)
        write(output/'results.json', results)
        write_csv(output/'summary.csv', tables)
        write_csv(output/'query_pairs.csv', pairs)
        complete = True
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        write(output/'completion_receipt.json', dict(status='COMPLETE' if complete else 'PARTIAL', failure=failure,
            cpu_wall_s=time.perf_counter()-started, budget_cpu_wall_s=budget, gpu_s=0, downloads=0,
            new_training=False, new_inference=False, source_sha256=sha(__file__), resources_released=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--a0', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--negative-reference', type=Path, required=True)
    p.add_argument('--cpu-budget-s', type=float, default=30.)
    a = p.parse_args()
    run(a.repo.resolve(), a.a0.resolve(), a.output.resolve(), a.negative_reference.resolve(), a.cpu_budget_s)
