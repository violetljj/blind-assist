"""Cached near witness ranking diagnostics, consumed Development only.

All curves/costs use evaluator labels and are posthoc descriptions, never
replacement main thresholds. Witness-vs-FREE is not ordinary deployment AUC.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict
import shutil
import time
import numpy as np
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_negative_frozen_score import ARMS, EVAL, numeric
from rgb_body_query_query_calibration_probe import FREE, write
from rgb_body_query_bounded_residual_proxy import cutoff_at_cost
from rgb_body_query_scene_diagnostic import write_csv

RATES = (0., .05, .10, .20, .50, 1.)
NEAR = '0.3-0.8m'


def describe(vals):
    a = np.asarray(vals, np.float64); f = a[np.isfinite(a)]
    return dict(n=len(a), finite=len(f), unavailable=int(np.isneginf(a).sum()),
        quantiles=dict(zip(('min', 'q25', 'median', 'q75', 'q95', 'max'),
                          map(float, np.quantile(f, [0., .25, .5, .75, .95, 1.])))) if len(f) else None)


def pair_probability(positive, negative):
    if not len(positive) or not len(negative): return None
    n = np.sort(np.asarray(negative, np.float64)); p = np.asarray(positive, np.float64)
    less = np.searchsorted(n, p, side='left'); equal = np.searchsorted(n, p, side='right')-less
    return float((less+.5*equal).sum()/(len(p)*len(n)))


def run(repo, out, budget, prior_attempt=None):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists(): raise FileExistsError('Preserve previous receipts')
    prior = 0.
    if prior_attempt:
        old = load(prior_attempt/'terminal.json'); assert old['status'] == 'FAILED_PARTIAL'
        prior = old['wall_s']+old.get('prior_attempt_wall_s', 0.)
    assert prior < budget
    started = time.perf_counter(); receipt = dict(status='STARTING', prior_attempt_wall_s=prior)
    write(out/'plan.json', dict(frozen_utc=utc(), source_sha256=sha(__file__),
        parent_plan_sha256=sha(out.parent/'plan.json'), budget_s=budget, prior_attempt_wall_s=prior,
        rates=RATES, near=NEAR, all_arms=ARMS, curves_posthoc=True, main_cutoffs_changed=False))
    shutil.copyfile(__file__, out/'executed_near_ranking.py')
    for f in (cutoff_at_cost, numeric):
        pp = Path(f.__code__.co_filename); shutil.copyfile(pp, out/f'executed_dependency_{pp.name}')
    def check():
        if time.perf_counter()-started+prior >= budget: raise TimeoutError('Cumulative ranking wall budget')
    inputs = []
    def checked(p):
        check(); inputs.append(dict(path=str(p), sha256=sha(p))); return load(p)
    try:
        work = repo/'artifacts.local/work'; previous = work/'rgb-body-query-arkit-cal-dev-20261010/score'
        cached = work/'rgb-body-query-negative-score-dev-20261010/frozen-score'
        conditional = work/'rgb-body-query-distance-cal-dev-20261010/score'
        result, summary, curves, descriptors, counts = {}, [], [], [], []
        all_cal_ids, all_eval_ids = set(), set()
        for kind in ('absolute', 'normalized'):
            cal = {a: numeric(rr) for a, rr in checked(previous/kind/'cal_scores.json')['arms'].items()}
            globals = checked(previous/kind/'thresholds.json')['arms']
            bands = checked(conditional/kind/'thresholds.json')['arms']
            groups = {'cal_all_fitted_in': cal}
            for field, prefix in [('cohort', 'cal_source'), ('environment', 'cal_environment')]:
                for name in sorted({r[field] for r in cal['affine']}):
                    groups[prefix+'/'+name] = {a: [r for r in rr if r[field]==name] for a, rr in cal.items()}
            all_cal_ids.update((r['scan'], r['frame']) for r in cal['affine'])
            folder = cached/('cached' if kind=='absolute' else 'normalized-probe')
            for c in EVAL:
                vv = {a: numeric(rr) for a, rr in checked(folder/f'{c}_scores.json')['arms'].items()}
                groups['eval/'+c] = vv
                all_eval_ids.update((r['scan'], r['frame']) for r in vv['affine'])
            result[kind] = {}
            for group, arms in groups.items():
                check(); result[kind][group] = {}
                for arm in ARMS:
                    rr = [r for r in arms[arm] if r['distance_band']==NEAR]
                    pos = [r for r in rr if r['reference_state']=='POSITIVE']
                    free = [r for r in rr if r['reference_state']==FREE]
                    pp = [r['known_positive_score'] for r in pos]
                    pq = [r['query_score'] for r in pos]; ff = [r['query_score'] for r in free]
                    finitep = [v for v in pp if np.isfinite(v)]
                    first = max(finitep) if finitep and ff else None
                    cost = sum(v>=first for v in ff) if first is not None else None
                    values = dict(query_total=len(rr), POS=len(pos), FREE=len(free),
                        UNKNOWN=sum(r['reference_state']=='UNKNOWN' for r in rr),
                        witness_pair_probability=pair_probability(pp, ff),
                        trigger_pair_probability=pair_probability(pq, ff),
                        first_witness_cutoff_posthoc=first, first_witness_required_FREE=cost,
                        first_witness_FREE_rate=cost/len(ff) if cost is not None else None,
                        first_witnesses=sum(v>=first for v in pp) if first is not None else None)
                    result[kind][group][arm] = dict(**values, known_POS_score=describe(pp),
                        POS_trigger_score=describe(pq), FREE_query_score=describe(ff))
                    summary.append(dict(readout=kind, group=group, arm=arm, **values))
                    for tag, vals in [('known_POS', pp), ('POS_trigger', pq), ('FREE', ff)]:
                        counts.append(dict(readout=kind, group=group, arm=arm, type=tag, **describe(vals)))
                    for label, cutoff in [('global588', globals[arm]['cutoff']), ('conditional_band5', bands[arm][NEAR]['cutoff'])]:
                        descriptors.append(dict(readout=kind, group=group, arm=arm, mode=label, cutoff=cutoff,
                            POS_total=len(pp), known_witness=sum(v>=cutoff for v in pp),
                            POS_trigger=sum(v>=cutoff for v in pq), FREE_total=len(ff), FREE_support=sum(v>=cutoff for v in ff)))
                    for rate in RATES:
                        allowed = int(np.floor(rate*len(ff))) if ff else None
                        cutoff = cutoff_at_cost(ff, allowed) if ff else None
                        curves.append(dict(readout=kind, group=group, arm=arm, requested_rate=rate,
                            allowed_FREE=allowed, cutoff=cutoff, POS_total=len(pp), FREE_total=len(ff),
                            known_witness=sum(v>=cutoff for v in pp) if cutoff is not None else None,
                            FREE_support=sum(v>=cutoff for v in ff) if cutoff is not None else None,
                            unavailable_POS_witness=sum(np.isneginf(v) for v in pp),
                            role='POSTHOC_evaluator_FREE_auxiliary_not_main_calibration'))
        assert len(all_cal_ids)==304 and len(all_eval_ids)==136 and all_cal_ids.isdisjoint(all_eval_ids)
        write(out/'results.json', dict(status='COMPLETE', readouts=result,
            witness_probability_definition='P(known_POS_score > FREE_query_score)+.5P(tie), not ordinary classifier AUC',
            first_cost_role='Posthoc uses highest known_POS score, no new main threshold',
            limits='Consumed Development; query/frame correlation, missing FREE/POS not measurable'))
        write_csv(out/'summary.csv', summary); write_csv(out/'score_descriptors.csv', counts)
        write_csv(out/'held_FREE_curves.csv', curves); write_csv(out/'frozen_cutoff_descriptors.csv', descriptors)
        write(out/'inputs.json', inputs)
        receipt.update(status='COMPLETE', summary_rows=len(summary), curve_rows=len(curves),
            descriptor_rows=len(descriptors), cal_frames=304, eval_frames=136,
            GPU=0, training=0, downloads=0, model_or_pixels_opened=0, main_cutoffs_changed=False)
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL', error=repr(exc)); raise
    finally:
        receipt.update(wall_s=time.perf_counter()-started, completed_utc=utc(), source_sha256=sha(__file__))
        write(out/'terminal.json', receipt); print(receipt, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--budget-s', type=float, default=180.)
    p.add_argument('--prior-attempt', type=Path)
    a=p.parse_args(); run(a.repo.resolve(), a.output.resolve(), a.budget_s, a.prior_attempt)
