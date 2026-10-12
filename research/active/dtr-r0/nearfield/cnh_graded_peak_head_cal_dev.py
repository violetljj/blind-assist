"""Cal-only HEAD incremental-union budgets with frozen BODY light additions.

Fixed cached joint-current scores; consumed Development, no refit or prediction.
"""
from pathlib import Path
import time
import numpy as np
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E
import cnh_graded_peak_body_only_dev as B

ROOT = C.ROOT
JOINT = B.JOINT
OUT = ROOT/'artifacts.local/work/cnh-graded-peak-head-cal-dev-20261010'
FRACTIONS = (.25, .50, .75)
POLICIES = ('baseline', 'both', 'body_only', 'head25', 'head50', 'head75')


def baseline(d, si, th):
    ordinary = d['candidates'][0, si]
    strong = E.old_fusion(d['m3'], d['local']) | (ordinary >= th['addition'])
    return np.where(strong, 2, np.where(ordinary >= th['single'], 1, 0)).astype(np.int8)


def variants(base, score, theta):
    add = (base == 0) & np.isfinite(score) & (score >= theta)
    both = np.where(add, 1, base).astype(np.int8)
    body = both.copy()
    body[..., 0] = base[..., 0]
    return both, body


def calibrate_head(base, body, score, category, mask, parent_theta, fraction):
    # The joint-silent condition prices actual extra union slots on cal only;
    # it is not a runtime eligibility condition and never consumes contact truth.
    eligible = (base[..., 0] == 0) & np.isfinite(score[..., 0]) & (score[..., 0] >= parent_theta)
    marginal = eligible & ~(body > 0).any(-1)
    values = np.where(marginal[mask], score[mask, ..., 0], -np.inf)
    cat = category[mask]
    clear = (cat == 'clear').all(1)
    passed = (cat == 'pass').any(1) & ~(cat == 'contact').any(1)
    counts = dict(clear=int(marginal[mask][clear].sum()), pass_=int(marginal[mask][passed].sum()))
    targets = {name:int(np.floor(fraction*count)) for name, count in counts.items()}
    clear_cut = C.at_most(values[clear], targets['clear'])
    pass_cut = C.at_most(values[passed], targets['pass_'])
    theta = max(parent_theta, clear_cut, pass_cut)
    flagged = marginal[mask] & (score[mask, ..., 0] >= theta)
    actual_clear, actual_pass = int(flagged[clear].sum()), int(flagged[passed].sum())
    assert actual_clear <= targets['clear'] and actual_pass <= targets['pass_']
    return dict(fraction=fraction, parent_theta=parent_theta, theta=theta,
        old_HEAD_incremental_clear_slots=counts['clear'], old_HEAD_incremental_pass_slots=counts['pass_'],
        target_clear_slots=targets['clear'], target_pass_slots=targets['pass_'],
        actual_clear_slots=actual_clear, actual_pass_slots=actual_pass,
        clear_unused=targets['clear']-actual_clear, pass_unused=targets['pass_']-actual_pass,
        clear_cut=None if clear_cut == -np.inf else clear_cut,
        pass_cut=None if pass_cut == -np.inf else pass_cut,
        clear_denominator=int(clear.sum())*4*13, pass_denominator=int(passed.sum())*4*13)


def run():
    began = time.monotonic()
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Preserve previous declared attempt')
    inputs = dict(C.read(B.OUT/'PLAN.json')['inputs_sha256'])
    for path in [B.OUT/'metrics.json', B.OUT/'cal_grades.npz', B.OUT/'validation_grades.npz',
                 Path(B.__file__), Path(G.__file__), Path(C.__file__)]:
        inputs[str(path.relative_to(ROOT))] = C.sha(path)
    C.save(OUT/'PLAN.json', dict(task='CNH_GRADED_PEAK_HEAD_CAL_DEV_20261010',
        authorization='User 继续 height-specific calibration comparison', lane='EXPLORE consumed simulated Development',
        goal='Retain useful weak HEAD additions while reducing their extra union cost; BODY additions fixed',
        budget=dict(main_CPU_command_wall_seconds=90, audit_CPU_command_wall_seconds=90,
                    integration_CPU_command_wall_seconds=120, total_CPU_command_wall_seconds=300, GPU_seconds=0),
        backend='TASK_NOT_GPU_SUITABLE: tiny saved-score scalar comparisons and CSV counts, localCPU',
        inputs_sha256=inputs, source_sha256=C.sha(Path(__file__)), seeds=list(G.SEEDS), policies=POLICIES,
        fractions=FRACTIONS, primary='head50; all three fractions reported without validation winner selection',
        calibration='Original cal background9/11 only; cal8/10 not used to select cutoffs. HEAD theta>=parent score_current/c15_p64 theta. BODY unchanged. Original both HEAD marginal extra union clear/purepass slots relative to body_only determine each floor(fraction*count) cap separately; whole-score ties; smallest feasible cutoff via nextafter excluded boundary.',
        runtime='Only HEAD baseline0, finite saved score>=newtheta; no joint-silent condition in policy. BODY unchanged from parent both. Baseline strong/light retained.',
        decision_check='Each height384 contacts; clear6656/pass3328 union slots per full split.1/384 event resolution. Cost caps may be0 and ties may leave unused cap; report count rather than require strict improvement or zero loss. head50 primary retains tradeoff, not formal promotion. No utility weights or independent-confirmation gate.',
        timing='f3..13 timely, f3..15 full first; contact-query physical union distinguished from legacy any-height',
        adjustable_scope='Source/schema repairs and focused verification; no new model/score/population/validation tuning',
        stop='Complete fixed36 cells/9cuts and paired timing/cost ledgers; stop this fraction recipe, no retuning',
        limits='Post-validation-directed Explore; correlated seeds/K/frames; cal caps not validation equivalence; no App/hardware/quieting'))
    try:
        for path, digest in inputs.items():
            if C.sha(ROOT/path) != digest: raise ValueError(f'Input drift: {path}')
        data = C.load()
        fit_mask, partition = C.split_cal(data['cal']['rows'])
        C.save(OUT/'cal_partition.json', partition)
        thresholds = C.read(C.PARENT/'thresholds.json')
        old_cuts = C.read(JOINT/'calibrations.json')
        scores = {}
        for split in data:
            with np.load(JOINT/f'{split}_scores.npz', allow_pickle=False) as a:
                scores[split] = dict(zip(a['keys'].tolist(), a['scores']))
        cuts = {}
        for si, seed in enumerate(G.SEEDS):
            parent_theta = old_cuts[f'{seed}/score_current/c15_p64']['theta']
            if old_cuts[f'{seed}/score_current/c15_p64']['nonbinding']:
                raise ValueError('Parent finite cutoff required for this declared comparison')
            cal = data['cal']; base = baseline(cal, si, thresholds[str(seed)])
            score = scores['cal'][f'{seed}/score_current']
            _, body = variants(base, score, parent_theta)
            for fraction, policy in zip(FRACTIONS, POLICIES[3:]):
                cuts[f'{seed}/{policy}'] = calibrate_head(base, body, score, cal['category'], ~fit_mask, parent_theta, fraction)
        C.save(OUT/'calibrations.json', cuts)
        metrics, summary, ledger = {}, [], []
        for split, d in data.items():
            keys, saved = [], []
            for si, seed in enumerate(G.SEEDS):
                if time.monotonic()-began >= 90: raise TimeoutError('Main phase90s cap')
                base = baseline(d, si, thresholds[str(seed)])
                score = scores[split][f'{seed}/score_current']
                parent_theta = old_cuts[f'{seed}/score_current/c15_p64']['theta']
                both, body = variants(base, score, parent_theta)
                grades = dict(baseline=base, both=both, body_only=body)
                for policy in POLICIES[3:]:
                    grade = body.copy()
                    head = (base[..., 0] == 0) & np.isfinite(score[..., 0]) & (score[..., 0] >= cuts[f'{seed}/{policy}']['theta'])
                    grade[..., 0] = np.where(head, 1, base[..., 0])
                    assert np.all(grade <= both) and np.all(grade >= body)
                    np.testing.assert_array_equal(grade[..., 1], both[..., 1])
                    grades[policy] = grade
                refs = dict(prior_light=base > 0, ordinary_OR=base == 2,
                    M3=d['m3'] >= E.M3_THETA, old_fusion=E.old_fusion(d['m3'], d['local']))
                for policy, grade in grades.items():
                    np.testing.assert_array_equal(grade == 2, base == 2)
                    result = G.describe(grade, d, refs)
                    result['paired_vs_both'] = G.paired_timing(both > 0, grade > 0, d['category'])
                    result['physical_paired_vs_baseline'] = B.physical_pair(base > 0, grade > 0, d['category'])
                    result['physical_paired_vs_both'] = B.physical_pair(both > 0, grade > 0, d['category'])
                    result['physical_contact_timely'] = result['physical_paired_vs_baseline']['candidate']
                    result['addition_costs'] = B.added_cost((grade > 0) & (base == 0), d['category'])
                    result['new_costs_vs_baseline'] = B.new_cost(base > 0, grade > 0, d['category'])
                    result['new_costs_vs_body_only'] = B.new_cost(body > 0, grade > 0, d['category'])
                    key = f'{seed}/{policy}'; metrics[f'{split}/{key}'] = result
                    keys.append(key); saved.append(grade)
                    pair = result['paired_any']['prior_light']; costs = result['new_costs_vs_baseline']
                    summary.append(dict(split=split, seed=seed, policy=policy,
                        HEAD=result['any']['counts'][0], BODY=result['any']['counts'][1],
                        HEAD_rescue=pair[0]['rescue'], BODY_rescue=pair[1]['rescue'], HEAD_earlier=pair[0]['earlier'], BODY_earlier=pair[1]['earlier'],
                        HEAD_loss=pair[0]['loss'], BODY_loss=pair[1]['loss'], HEAD_later=pair[0]['later'], BODY_later=pair[1]['later'],
                        HEAD_foregone_rescue=result['paired_vs_both'][0]['loss'], HEAD_foregone_advance=result['paired_vs_both'][0]['later'],
                        clear_slots=result['any']['clear_slots'], clear_clips=result['any']['clear_clips'], pass_clips=result['any']['pass_clips'],
                        extra_clear_slots=costs['clear']['new_slots'], extra_clear_clips=costs['clear']['new_clips'],
                        extra_pass_slots=costs['pass']['new_slots'], extra_pass_clips=costs['pass']['new_clips'],
                        HEAD_incremental_clear_slots=result['new_costs_vs_body_only']['clear']['new_slots'],
                        HEAD_incremental_pass_slots=result['new_costs_vs_body_only']['pass']['new_slots']))
                    clocks = dict(first_any=G.first(grade > 0), first_timely=G.first(grade > 0, 11),
                        first_strong=G.first(grade == 2), first_light=G.first(grade == 1),
                        before_first_any=G.first(base > 0), before_first_timely=G.first(base > 0, 11),
                        both_first_any=G.first(both > 0), both_first_timely=G.first(both > 0, 11))
                    for n, row in enumerate(d['rows']):
                        for k in range(4):
                            for q, height in enumerate(E.HEIGHTS):
                                ledger.append(dict(split=split, seed=seed, policy=policy, scene=int(d['scene_ids'][n]),
                                    replica=k, height=height, category=d['category'][n, q],
                                    shape_family=row['shape_family'], background_family=row['background_family'],
                                    **{name:int(value[n, k, q]) for name, value in clocks.items()}))
            np.savez_compressed(OUT/f'{split}_grades.npz', keys=np.array(keys), grades=np.array(saved),
                scene_ids=d['scene_ids'], category=d['category'])
        G.write_csv(OUT/'summary.csv', summary); G.write_csv(OUT/'ledger.csv', ledger)
        C.save(OUT/'metrics.json', metrics)
        C.save(OUT/'receipt.json', dict(status='COMPLETE', seconds=time.monotonic()-began, cells=len(summary),
            cuts=len(cuts), ledger_rows=len(ledger), source_sha256=C.sha(Path(__file__)), GPU_seconds=0, fit=0, prediction=0, new_raw=0))
        print(f'COMPLETE: {len(summary)} cells, {len(cuts)} cuts, {len(ledger)} ledger, {time.monotonic()-began:.3f}s')
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json', dict(error=repr(error), seconds=time.monotonic()-began)); raise


if __name__ == '__main__': run()
