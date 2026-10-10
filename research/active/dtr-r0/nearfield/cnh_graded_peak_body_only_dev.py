"""Fixed cached joint-current cutoff: both-height versus BODY-only light addition.

Consumed simulated Development; no refit, prediction, cutoff selection or quieting.
Evaluator categories affect reporting only, never the policy.
"""
from pathlib import Path
import time

import numpy as np
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

ROOT = C.ROOT
JOINT = ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
OUT = ROOT/'artifacts.local/work/cnh-graded-peak-body-only-dev-20261010'
POLICIES = ('baseline', 'both', 'body_only')


def physical_pair(before, after, category):
    contact = category == 'contact'
    mask = contact.any(1)
    # Only contact-labelled queries count here; E.summary's legacy any-height
    # statistic also counts alerts from other queries on a contact scene.
    a = G.first((before & contact[:, None, None, :]).any(-1)[..., None], 11)[mask, ..., 0]
    b = G.first((after & contact[:, None, None, :]).any(-1)[..., None], 11)[mask, ..., 0]
    both = (a >= 0) & (b >= 0)
    delta = a[both]-b[both]
    return dict(denominator=int(a.size), baseline=int((a >= 0).sum()), candidate=int((b >= 0).sum()),
        rescue=int(((a < 0) & (b >= 0)).sum()), loss=int(((a >= 0) & (b < 0)).sum()),
        both_timely=int(both.sum()), earlier=int((delta > 0).sum()), same=int((delta == 0).sum()),
        later=int((delta < 0).sum()), median_advance_frames=float(np.median(delta)) if len(delta) else None,
        advance_frame_histogram={str(int(v)):int((delta == v).sum()) for v in np.unique(delta)})


def added_cost(added, category):
    masks = dict(clear=(category == 'clear').all(1),
                 pass_=(category == 'pass').any(1) & ~(category == 'contact').any(1))
    return {name.rstrip('_'):dict(joint=G.cost(added.any(-1), mask),
        **{height:G.cost(added[..., q], mask) for q, height in enumerate(E.HEIGHTS)})
        for name, mask in masks.items()}


def new_cost(before, after, category):
    masks = dict(clear=(category == 'clear').all(1),
                 pass_=(category == 'pass').any(1) & ~(category == 'contact').any(1))
    a, b = before.any(-1), after.any(-1)
    fa, fb = G.first(a[..., None])[..., 0], G.first(b[..., None])[..., 0]
    return {name.rstrip('_'):dict(new_slots=int((b[mask] & ~a[mask]).sum()),
        new_clips=int(((fa[mask] < 0) & (fb[mask] >= 0)).sum()),
        existing_clips_earlier=int(((fa[mask] >= 0) & (fb[mask] >= 0) & (fb[mask] < fa[mask])).sum()))
        for name, mask in masks.items()}


def run():
    began = time.monotonic()
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Preserve completed or failed run; use a separate declared repair output')
    inherited = C.read(C.OUT/'input_manifest.json')['input_sha256']
    inputs = dict(inherited)
    for path in [JOINT/'PLAN.json', JOINT/'calibrations.json'] + [
        JOINT/f'{split}_{kind}.npz' for split in ('cal', 'validation') for kind in ('scores', 'grades')]:
        inputs[str(path.relative_to(ROOT))] = C.sha(path)
    C.save(OUT/'PLAN.json', dict(task='CNH_GRADED_PEAK_BODY_ONLY_DEV_20261010',
        lane='EXPLORE consumed simulated Development', authorization='User 推进 fixed BODY-only comparison',
        goal='Measure BODY benefit and clear/pass savings alongside foregone HEAD additions',
        budget=dict(main_CPU_command_wall_seconds=90, audit_CPU_command_wall_seconds=90,
                    integration_CPU_command_wall_seconds=120, total_CPU_command_wall_seconds=300, GPU_seconds=0),
        backend='TASK_NOT_GPU_SUITABLE: small cached arrays and scalar counts; local CPU',
        inputs_sha256=inputs, source_sha256=C.sha(Path(__file__)), seeds=list(G.SEEDS), policies=POLICIES,
        scoring='Reuse score_current only, saved per-seed c15_p64 cutoff, no recalibration',
        baseline='ordinary_OR strong2 plus complete ordinary single light1',
        both='baseline0 AND finite cachedscore>=cutoff -> light1, both heights',
        body_only='same addition restricted to BODY q1; HEAD exactly baseline; original strong/light retained',
        population='all ideal cal and validation, each384 scenes x4 replicas x13 frames x2 queries',
        timing='Timely f3..13 inclusive, full clock f3..15 inclusive; advances in frames only',
        decision_check='HEAD/BODY each384 contacts per split; clear512 clips/6656 union slots, purepass256/3328; one event is1/384. BODY identity and strong/baseline preservation are construction checks, not noninferiority evidence. Full foregone HEAD rescue/earlier and contact-query physical union reported; no scalar utility weight or success gate imposed.',
        adjustable_scope='Source/schema repairs and proportionate checks only; fixed scores/cutoffs/policies/population',
        stop='Complete fixed18 cells, ledgers and cost/timing decomposition; no threshold/model/raw/new sample sweeps',
        limits='Seed/K/frame correlation; post-validation-selected restriction is exploration, not independent confirmation; no App/hardware/quieting adoption'))
    try:
        for path, digest in inputs.items():
            if C.sha(ROOT/path) != digest:
                raise ValueError(f'Input drift: {path}')
        data = C.load()
        thresholds = C.read(C.PARENT/'thresholds.json')
        cuts = C.read(JOINT/'calibrations.json')
        metrics, summary, ledger = {}, [], []
        for split, d in data.items():
            with np.load(JOINT/f'{split}_scores.npz', allow_pickle=False) as archive:
                scores = dict(zip(archive['keys'].tolist(), archive['scores']))
            with np.load(JOINT/f'{split}_grades.npz', allow_pickle=False) as archive:
                old_grades = dict(zip(archive['keys'].tolist(), archive['grades']))
            keys, saved = [], []
            for si, seed in enumerate(G.SEEDS):
                if time.monotonic()-began >= 90:
                    raise TimeoutError('Main phase90s cap')
                th = thresholds[str(seed)]
                ordinary = d['candidates'][0, si]
                strong = E.old_fusion(d['m3'], d['local']) | (ordinary >= th['addition'])
                baseline = np.where(strong, 2, np.where(ordinary >= th['single'], 1, 0)).astype(np.int8)
                score = scores[f'{seed}/score_current']
                cut = cuts[f'{seed}/score_current/c15_p64']
                theta = -np.inf if cut['nonbinding'] else cut['theta']
                added = (baseline == 0) & np.isfinite(score) & (score >= theta)
                both = np.where(added, 1, baseline).astype(np.int8)
                np.testing.assert_array_equal(both, old_grades[f'{seed}/score_current/c15_p64'])
                body_added = added.copy()
                body_added[..., 0] = False
                body_only = np.where(body_added, 1, baseline).astype(np.int8)
                np.testing.assert_array_equal(body_only[..., 1], both[..., 1])
                np.testing.assert_array_equal(body_only[..., 0], baseline[..., 0])
                refs = dict(prior_light=baseline > 0, ordinary_OR=strong, M3=d['m3'] >= E.M3_THETA,
                            old_fusion=E.old_fusion(d['m3'], d['local']))
                before_full, before_timely = G.first(baseline > 0), G.first(baseline > 0, 11)
                both_full, both_timely = G.first(both > 0), G.first(both > 0, 11)
                for policy, grade in zip(POLICIES, (baseline, both, body_only)):
                    np.testing.assert_array_equal(grade == 2, strong)
                    assert np.all(grade >= baseline)
                    result = G.describe(grade, d, refs)
                    result['paired_vs_both'] = G.paired_timing(both > 0, grade > 0, d['category'])
                    result['physical_paired_vs_baseline'] = physical_pair(baseline > 0, grade > 0, d['category'])
                    result['physical_paired_vs_both'] = physical_pair(both > 0, grade > 0, d['category'])
                    result['physical_contact_timely'] = result['physical_paired_vs_baseline']['candidate']
                    result['addition_costs'] = added_cost((grade > 0) & (baseline == 0), d['category'])
                    result['new_costs_vs_baseline'] = new_cost(baseline > 0, grade > 0, d['category'])
                    key = f'{seed}/{policy}'
                    metrics[f'{split}/{key}'] = result
                    keys.append(key); saved.append(grade)
                    p = result['paired_any']['prior_light']
                    cost = result['new_costs_vs_baseline']
                    summary.append(dict(split=split, seed=seed, policy=policy,
                        HEAD=result['any']['counts'][0], BODY=result['any']['counts'][1],
                        HEAD_strong=result['strong']['counts'][0], BODY_strong=result['strong']['counts'][1],
                        HEAD_light_only=result['light_only_timely_contact'][0], BODY_light_only=result['light_only_timely_contact'][1],
                        HEAD_rescue=p[0]['rescue'], BODY_rescue=p[1]['rescue'], HEAD_loss=p[0]['loss'], BODY_loss=p[1]['loss'],
                        HEAD_earlier=p[0]['earlier'], BODY_earlier=p[1]['earlier'], HEAD_later=p[0]['later'], BODY_later=p[1]['later'],
                        physical_contact_query_timely=result['physical_contact_timely'],
                        physical_rescue=result['physical_paired_vs_baseline']['rescue'],
                        clear_slots=result['any']['clear_slots'], clear_clips=result['any']['clear_clips'],
                        pass_slots=result['joint_costs']['pass']['strong']['slots']+result['joint_costs']['pass']['light']['slots'],
                        pass_clips=result['any']['pass_clips'],
                        extra_clear_slots=cost['clear']['new_slots'], extra_clear_clips=cost['clear']['new_clips'],
                        extra_pass_slots=cost['pass']['new_slots'], extra_pass_clips=cost['pass']['new_clips'],
                        clear_light_longest_frames=result['joint_costs']['clear']['light']['longest_run_frames'],
                        pass_light_longest_frames=result['joint_costs']['pass']['light']['longest_run_frames'],
                        added_clear_longest_frames=result['addition_costs']['clear']['joint']['longest_run_frames'],
                        added_pass_longest_frames=result['addition_costs']['pass']['joint']['longest_run_frames']))
                    clocks = dict(first_any=G.first(grade > 0), first_timely=G.first(grade > 0, 11),
                        first_strong=G.first(grade == 2), first_light=G.first(grade == 1),
                        before_first_any=before_full, before_first_timely=before_timely,
                        both_first_any=both_full, both_first_timely=both_timely)
                    for n, row in enumerate(d['rows']):
                        for k in range(4):
                            for q, height in enumerate(E.HEIGHTS):
                                ledger.append(dict(split=split, seed=seed, policy=policy, scene=int(d['scene_ids'][n]),
                                    replica=k, height=height, category=d['category'][n, q],
                                    shape_family=row['shape_family'], background_family=row['background_family'],
                                    **{name:int(value[n, k, q]) for name, value in clocks.items()}))
            np.savez_compressed(OUT/f'{split}_grades.npz', keys=np.array(keys), grades=np.array(saved),
                                scene_ids=d['scene_ids'], category=d['category'])
        G.write_csv(OUT/'summary.csv', summary)
        G.write_csv(OUT/'ledger.csv', ledger)
        C.save(OUT/'metrics.json', metrics)
        C.save(OUT/'receipt.json', dict(status='COMPLETE', seconds=time.monotonic()-began,
            cells=len(summary), ledger_rows=len(ledger), source_sha256=C.sha(Path(__file__)), GPU_seconds=0,
            fit=0, prediction=0, new_raw=0))
        print(f'COMPLETE: {len(summary)} cells, {len(ledger)} ledger rows, {time.monotonic()-began:.3f}s')
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json', dict(error=repr(error), seconds=time.monotonic()-began))
        raise


if __name__ == '__main__':
    run()
