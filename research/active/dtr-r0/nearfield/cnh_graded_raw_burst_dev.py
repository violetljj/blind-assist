"""Fixed same-threshold raw evidence additions; consumed ideal Development only.

Primary current-frame raw single; causal 2-of-3 comparator. Neither sees truth.
"""
from pathlib import Path
import time
import numpy as np
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E
import cnh_graded_peak_body_only_dev as B
import cnh_graded_peak_head_cal_dev as H

ROOT = C.ROOT
PARENT = H.OUT
OUT = ROOT/'artifacts.local/work/cnh-graded-raw-burst-dev-20261010'
POLICIES = ('baseline', 'both', 'head50')
MECHANISMS = ('fixed', 'raw_single', 'raw_k2of3')


def triggers(raw, theta):
    hit = np.isfinite(raw) & (raw >= theta)
    twice = np.zeros_like(hit)
    for t in range(raw.shape[-2]):
        twice[..., t, :] = hit[..., max(0, t-2):t+1, :].sum(-2) >= 2
    return dict(raw_single=hit, raw_k2of3=twice)


def pair(before, after, category, end):
    a, b = G.first(before, end), G.first(after, end)
    result = []
    for q, height in enumerate(E.HEIGHTS):
        mask = np.broadcast_to((category[:, q] == 'contact')[:, None], a[..., q].shape)
        aa, bb = a[..., q][mask], b[..., q][mask]
        both = (aa >= 0) & (bb >= 0)
        delta = aa[both]-bb[both]
        result.append(dict(height=height, denominator=int(mask.sum()), before=int((aa >= 0).sum()),
            after=int((bb >= 0).sum()), rescue=int(((aa < 0) & (bb >= 0)).sum()),
            loss=int(((aa >= 0) & (bb < 0)).sum()), both=int(both.sum()),
            earlier=int((delta > 0).sum()), same=int((delta == 0).sum()), later=int((delta < 0).sum()),
            advance_frame_histogram={str(int(v)):int((delta == v).sum()) for v in np.unique(delta)}))
    return result


def outcomes(flags, category):
    full, timely = G.first(flags), G.first(flags, 11)
    rows = []
    for q, height in enumerate(E.HEIGHTS):
        mask = np.broadcast_to((category[:, q] == 'contact')[:, None], full[..., q].shape)
        rows.append(dict(height=height, denominator=int(mask.sum()),
            timely=int((timely[..., q][mask] >= 0).sum()),
            late=int(((timely[..., q][mask] < 0) & (full[..., q][mask] >= 0)).sum()),
            silent=int((full[..., q][mask] < 0).sum())))
    return rows


def run():
    began = time.monotonic()
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Preserve declared attempt, no implicit overwrite')
    inputs = dict(C.read(PARENT/'PLAN.json')['inputs_sha256'])
    for path in [PARENT/'calibrations.json', H.JOINT/'calibrations.json', C.PARENT/'thresholds.json',
                 Path(C.__file__), Path(G.__file__), Path(E.__file__), Path(B.__file__), Path(H.__file__)]:
        inputs[str(path.relative_to(ROOT))] = C.sha(path)
    for split in ('cal', 'validation'):
        for path in [PARENT/f'{split}_grades.npz', H.JOINT/f'{split}_scores.npz'] + [
                C.SOURCE/f'scores/ordinary_seed{seed}_{split}.npz' for seed in G.SEEDS]:
            inputs[str(path.relative_to(ROOT))] = C.sha(path)
    C.save(OUT/'PLAN.json', dict(task='CNH_GRADED_RAW_BURST_DEV_20261010', lane='EXPLORE consumed ideal simulated Development',
        authorization='User 推进 fixed detection diagnosis and short-evidence comparator',
        goal='Measure whether smoothing suppresses useful short raw evidence, with paired detection and cost',
        baseline='Fixed head50 primary, both and ordinary_OR/single baseline controls, all three seeds',
        budget=dict(main_CPU_command_wall_seconds=120, verification_CPU_command_wall_seconds=90,
                    report_CPU_command_wall_seconds=90, total_CPU_command_wall_seconds=300, GPU_seconds=0),
        adjustable_scope='Implementation/schema repair and focused verification only',
        policies=POLICIES, mechanisms=MECHANISMS, primary='raw_single additive on head50', seeds=list(G.SEEDS),
        runtime='Only grade0 becomes grade1; raw_single current finite raw >= frozen ordinary single; raw_k2of3 at least2 crossings in last<=3 observed frames, same threshold. No truth, smoothing, calibration, threshold selection or trigger persistence beyond fixed past3.',
        reset='Each scene/replica/query starts with empty history at observed f3. raw_k2of3 cannot trigger at f3; f4 can use f3+f4. No scene/replica/query carryover. Unobserved prior frames not fabricated.',
        timing='f3..13 timely, f14..15 late, f3..15 full; first cache sample f3 is not first lifetime evidence; frames only, nominal200ms not device latency',
        denominator='Each split384scenes x4replicas x13frames x2query; each contact height384streams, each shape96; clear512clips/6656 union slots, purepass256clips/3328slots',
        decision_check='One timely contact is1/384. Useful endpoint is full paired timely rescue/advance; report cost without imposing zero loss or cost. Raw-cross diagnosis is not an already evaluated rescue. No winner selection between seeds.',
        evaluator_only='Contact/pass/clear, shapes/rho and parent late/silent cohort assignments for reporting only',
        stop='Complete fixed54cells and focused parity/clock/cost verification; no window/threshold sweep, fitting, training, forward, raw reconstruction, new data or notifications',
        limits='Post-diagnostic-selected consumed Development, correlated seeds/replicas/frames; not device, hardware, user, safety or independent confirmation evidence',
        backend='TASK_NOT_GPU_SUITABLE: saved scalar arrays and integer counts',
        inputs_sha256=inputs, source_sha256=C.sha(Path(__file__))))
    try:
        for path, digest in inputs.items():
            if C.sha(ROOT/path) != digest: raise ValueError(f'Input drift: {path}')
        data = C.load()
        thresholds = C.read(C.PARENT/'thresholds.json')
        cuts = C.read(PARENT/'calibrations.json')
        old_cuts = C.read(H.JOINT/'calibrations.json')
        metrics, summary, ledger, cohorts = {}, [], [], []
        for split, d in data.items():
            with np.load(PARENT/f'{split}_grades.npz', allow_pickle=False) as a:
                parents = dict(zip(a['keys'].tolist(), a['grades']))
                np.testing.assert_array_equal(a['scene_ids'], d['scene_ids'])
                np.testing.assert_array_equal(a['category'], d['category'])
            with np.load(H.JOINT/f'{split}_scores.npz', allow_pickle=False) as a:
                scores = dict(zip(a['keys'].tolist(), a['scores']))
            keys, saved = [], []
            for si, seed in enumerate(G.SEEDS):
                th = thresholds[str(seed)]
                base = H.baseline(d, si, th)
                score = scores[f'{seed}/score_current']
                parent_theta = old_cuts[f'{seed}/score_current/c15_p64']['theta']
                both, _ = H.variants(base, score, parent_theta)
                head50 = np.where((base == 0) & np.isfinite(score) &
                    (score >= np.array([cuts[f'{seed}/head50']['theta'], parent_theta])), 1, base).astype(np.int8)
                for policy, rebuilt in dict(baseline=base, both=both, head50=head50).items():
                    np.testing.assert_array_equal(rebuilt, parents[f'{seed}/{policy}'])
                raw = E.read_job(C.SOURCE/f'scores/ordinary_seed{seed}_{split}.npz', 'ordinary', seed, split, 'ideal')
                addition = triggers(raw, th['single'])
                for policy in POLICIES:
                    before = parents[f'{seed}/{policy}']
                    before_full, before_timely = G.first(before > 0), G.first(before > 0, 11)
                    before_strong = G.first(before == 2)
                    for mechanism in MECHANISMS:
                        if time.monotonic()-began >= 120: raise TimeoutError('Main120s cap')
                        grade = before if mechanism == 'fixed' else np.where((before == 0) & addition[mechanism], 1, before).astype(np.int8)
                        assert np.all(grade >= before)
                        np.testing.assert_array_equal(grade == 2, before == 2)
                        first, timely = G.first(grade > 0), G.first(grade > 0, 11)
                        first_strong = G.first(grade == 2)
                        result = dict(outcomes=outcomes(grade > 0, d['category']),
                            paired_timely=pair(before > 0, grade > 0, d['category'], 11),
                            paired_full=pair(before > 0, grade > 0, d['category'], 13),
                            physical_contact_timely=B.physical_pair(before > 0, grade > 0, d['category']),
                            new_cost=B.new_cost(before > 0, grade > 0, d['category']),
                            added_cost=B.added_cost((grade > 0) & (before == 0), d['category']),
                            total_cost=B.added_cost(grade > 0, d['category']),
                            light_cost=B.added_cost(grade == 1, d['category']),
                            strong_cost=B.added_cost(grade == 2, d['category']))
                        key = f'{seed}/{policy}/{mechanism}'
                        metrics[f'{split}/{key}'] = result
                        keys.append(key); saved.append(grade)
                        row = dict(split=split, seed=seed, policy=policy, mechanism=mechanism)
                        for q, height in enumerate(E.HEIGHTS):
                            for name in ('timely', 'late', 'silent'): row[f'{height}_{name}'] = result['outcomes'][q][name]
                            for name in ('rescue', 'loss', 'earlier', 'later'): row[f'{height}_{name}'] = result['paired_timely'][q][name]
                            row[f'{height}_full_rescue'] = result['paired_full'][q]['rescue']
                        for kind in ('clear', 'pass'):
                            for name in ('slots', 'clips'): row[f'{kind}_{name}'] = result['total_cost'][kind]['joint'][name]
                            for name in ('new_slots', 'new_clips', 'existing_clips_earlier'): row[f'{kind}_{name}'] = result['new_cost'][kind][name]
                        summary.append(row)
                        contact_rows = []
                        for n, author in enumerate(d['rows']):
                            for k in range(4):
                                for q, height in enumerate(E.HEIGHTS):
                                    old_outcome = 'timely' if before_timely[n,k,q] >= 0 else 'late' if before_full[n,k,q] >= 0 else 'silent'
                                    new_outcome = 'timely' if timely[n,k,q] >= 0 else 'late' if first[n,k,q] >= 0 else 'silent'
                                    entry = dict(split=split, seed=seed, policy=policy, mechanism=mechanism, scene=int(d['scene_ids'][n]),
                                        replica=k, height=height, category=d['category'][n,q], shape_family=author['shape_family'], rho=author['rho'],
                                        before_outcome=old_outcome, after_outcome=new_outcome,
                                        before_first=int(before_full[n,k,q]), after_first=int(first[n,k,q]),
                                        before_timely=int(before_timely[n,k,q]), after_timely=int(timely[n,k,q]),
                                        before_first_strong=int(before_strong[n,k,q]),
                                        after_first_strong=int(first_strong[n,k,q]),
                                        grades=''.join(map(str, grade[n,k,:,q].tolist())))
                                    ledger.append(entry)
                                    if entry['category'] == 'contact': contact_rows.append(entry)
                        for height in E.HEIGHTS:
                            for old_outcome in ('timely', 'late', 'silent'):
                                for family in ('ALL', *sorted(set(d['families'].tolist()))):
                                    selected = [r for r in contact_rows if r['height'] == height and r['before_outcome'] == old_outcome and (family == 'ALL' or r['shape_family'] == family)]
                                    cohorts.append(dict(split=split, seed=seed, policy=policy, mechanism=mechanism, height=height,
                                        before_outcome=old_outcome, shape_family=family, denominator=len(selected),
                                        after_timely=sum(r['after_outcome'] == 'timely' for r in selected),
                                        after_late=sum(r['after_outcome'] == 'late' for r in selected),
                                        after_silent=sum(r['after_outcome'] == 'silent' for r in selected)))
            np.savez_compressed(OUT/f'{split}_grades.npz', keys=np.array(keys), grades=np.array(saved),
                                scene_ids=d['scene_ids'], category=d['category'])
        G.write_csv(OUT/'summary.csv', summary); G.write_csv(OUT/'ledger.csv', ledger); G.write_csv(OUT/'cohorts.csv', cohorts)
        C.save(OUT/'metrics.json', metrics)
        C.save(OUT/'receipt.json', dict(status='COMPLETE', cells=len(summary), ledger_rows=len(ledger), cohorts=len(cohorts),
            seconds=time.monotonic()-began, source_sha256=C.sha(Path(__file__)), GPU_seconds=0, fit=0, prediction=0, raw_reconstruction=0, threshold_changes=0))
        print(f'COMPLETE {len(summary)}cells {len(ledger)}ledger {time.monotonic()-began:.3f}s')
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json', dict(error=repr(error), seconds=time.monotonic()-began)); raise


if __name__ == '__main__': run()
