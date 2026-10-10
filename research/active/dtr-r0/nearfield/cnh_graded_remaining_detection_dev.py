"""Fixed-point remaining contact diagnosis; cached scores/support only.

Author metadata is evaluator-only stratification. No fitting, forward pass,
threshold selection, raw readout, notification or baseline changes.
"""
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

ROOT = C.ROOT
PARENT = ROOT/'artifacts.local/work/cnh-graded-peak-head-cal-dev-20261010'
JOINT = ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
TRACK = ROOT/'artifacts.local/work/cnh-graded-peak-track-dev-20261010/tracks'
OUT = ROOT/'artifacts.local/work/cnh-graded-remaining-detection-dev-20261010'
POLICIES = ('baseline', 'both', 'head50')
FIELDS = ('rank0_peak_log', 'strongest_inner_peak_log', 'inner_weighted_sum_share',
          'inner_supported_candidate_count', 'rank0_depth_m', 'strongest_inner_depth_m',
          'strongest_inner_share', 'rank0_support_x_extent_m', 'rank0_support_y_extent_m',
          'rank0_support_z_extent_m')


def onset(flags):
    hit = np.flatnonzero(flags)
    return int(hit[0]+3) if len(hit) else -1


def stats(values):
    a = np.array([v for v in values if v is not None], float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), min=float(a.min()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p90=float(np.quantile(a, .9)) if len(a) else None,
                max=float(a.max()) if len(a) else None)


def aggregate(rows):
    names = ('first_any', 'first_inner', 'first_rank0', 'first_raw_single',
             'first_smooth_single', 'score_f3_margin', 'score_max_timely_margin',
             'ordinary_raw_max_timely_margin', 'ordinary_smooth_max_timely_margin',
             'm3_max_timely_margin', 'local_max_timely_margin', 'inner_timely_frames',
             'inner_timely_longest_run', 'rank0_outside_with_inner_timely_frames',
             'first_inner_to_alert_frames', 'inner_peak_max_timely', 'rank0_peak_max_timely')
    return dict(streams=len(rows),
        outcomes={s:sum(r['outcome'] == s for r in rows) for s in ('timely', 'late', 'silent')},
        raw_single_cross_without_smooth=sum(r['raw_single_cross_without_smooth'] for r in rows),
        raw_old_cross_without_smooth=sum(r['raw_old_cross_without_smooth'] for r in rows),
        no_timely_inner_support=sum(r['inner_timely_frames'] == 0 for r in rows),
        inner_at_f3=sum(r['first_inner'] == 3 for r in rows),
        first_any_histogram=dict(sorted(Counter(str(r['first_any']) for r in rows).items())),
        fields={n:stats([r[n] if not n.startswith('first_') or r[n] >= 0 else None for r in rows]) for n in names})


def run():
    started = time.monotonic()
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Preserve existing diagnostic attempt')
    # Parent hashes pin the inherited scores, author rows, grades and cutoffs.
    inputs = dict(C.read(PARENT/'PLAN.json')['inputs_sha256'])
    direct = [PARENT/'calibrations.json', PARENT/'cal_grades.npz', PARENT/'validation_grades.npz',
              JOINT/'features/schema.json', TRACK/'schema.json']
    for split in ('cal', 'validation'):
        direct += [JOINT/f'{split}_scores.npz', JOINT/'features'/f'{split}_features.npz',
                   TRACK/f'{split}_tracks.npz']
    for path in direct:
        inputs[str(path.relative_to(ROOT))] = C.sha(path)
    C.save(OUT/'PLAN.json', dict(task='CNH_GRADED_REMAINING_DETECTION_DEV_20261010',
        authorization='User agrees to remaining weak/thin miss, late and full-window silence diagnosis',
        lane='EXPLORE consumed ideal simulated Development', goal='Separate early support availability, score rejection, smoothing and fixed-point cost tradeoff',
        primary='head50; both high-detection and baseline controls, all seeds and both splits',
        budget=dict(main_CPU_command_wall_seconds=120, verification_CPU_command_wall_seconds=120,
                    integration_CPU_command_wall_seconds=180, total_CPU_command_wall_seconds=420, GPU_seconds=0),
        adjustable_scope='Diagnostic implementation/schema repair and scoped verification only',
        stop='Complete fixed cohort diagnostic and report; no threshold/model/policy/notification change',
        denominator='Each split/policy/seed has384contact streams per height,96per family; sceneK correlated',
        timing='f3..13 timely, f14..15 late, f3..15 silent. first score=f3 observed cache boundary, not first lifetime signal. nominal200ms only',
        support='Existing current retained top8 positive gated native peaks; inner_share>0. No support is missing proxy, not target absence/free. Present support is not target attribution.',
        score='Frozen ordinary raw/smooth, M3/local smooth, joint score_current and inherited per-height cutoff; margins in native separate scales, no probability or cross-model comparison',
        smoothing='Raw crossings without any timely smooth crossing describe a cache discrepancy only, not an evaluated replacement rule',
        evaluator_only='Shape/rho/placement/background strata and constant contact labels; never runtime feature inputs',
        decision_check='384contact streams per height and96per family,1event resolution. Report all outcomes and cost-point foregone detections; no success gate or selected subset.',
        backend='TASK_NOT_GPU_SUITABLE: saved arrays and descriptive statistics',
        inputs_sha256=inputs, source_sha256=C.sha(Path(__file__))))
    try:
        for path, digest in inputs.items():
            if C.sha(ROOT/path) != digest:
                raise ValueError(f'Input drift: {path}')
        data = C.load()
        thresholds = C.read(C.PARENT/'thresholds.json')
        cuts = C.read(PARENT/'calibrations.json')
        old_cuts = C.read(JOINT/'calibrations.json')
        events, snapshots, pairs = [], [], []
        for split, d in data.items():
            with np.load(PARENT/f'{split}_grades.npz', allow_pickle=False) as a:
                grades = dict(zip(a['keys'].tolist(), a['grades']))
                np.testing.assert_array_equal(a['scene_ids'], d['scene_ids'])
                np.testing.assert_array_equal(a['category'], d['category'])
            with np.load(JOINT/f'{split}_scores.npz', allow_pickle=False) as a:
                scores = dict(zip(a['keys'].tolist(), a['scores']))
            with np.load(JOINT/'features'/f'{split}_features.npz', allow_pickle=False) as a:
                np.testing.assert_array_equal(a['scene_ids'], d['scene_ids'])
                feature, valid = a['current_features'], a['current_valid']
                names = a['current_names'].tolist()
            with np.load(TRACK/f'{split}_tracks.npz', allow_pickle=False) as a:
                np.testing.assert_array_equal(a['scene_ids'], d['scene_ids'])
                np.testing.assert_array_equal(a['frames'], np.arange(3, 16))
                cand_valid, inner_share = a['candidate_valid'], a['candidate_inner_share']
            inner = (cand_valid & (inner_share > 0)).any(-1)
            rank0 = cand_valid[..., 0]
            outside_with_inner = rank0 & (inner_share[..., 0] == 0) & inner
            def descriptor(n, k, q, t, name):
                j = names.index(name)
                return float(feature[n,k,t,q,j]) if valid[n,k,t,q,j] else None
            for si, seed in enumerate(G.SEEDS):
                th = thresholds[str(seed)]
                raw = E.read_job(C.SOURCE/f'scores/ordinary_seed{seed}_{split}.npz', 'ordinary', seed, split, 'ideal')
                smooth = d['candidates'][0, si]
                with np.load(C.SOURCE/f'baselines/{split}_ideal.npz', allow_pickle=False) as a:
                    raw_old = (a['m3_raw'] >= E.OLD_RAISED) | (a['local_raw'] >= E.OLD_LOCAL)
                score = scores[f'{seed}/score_current']
                parent_cut = old_cuts[f'{seed}/score_current/c15_p64']['theta']
                # Rebuild exact fixed-point grades to detect source/schema mistakes.
                base = np.where(E.old_fusion(d['m3'], d['local']) | (smooth >= th['addition']), 2,
                                np.where(smooth >= th['single'], 1, 0)).astype(np.int8)
                np.testing.assert_array_equal(base, grades[f'{seed}/baseline'])
                for policy in POLICIES:
                    if time.monotonic()-started >= 120:
                        raise TimeoutError('Main120s cap')
                    grade = grades[f'{seed}/{policy}']
                    cut = np.array([cuts[f'{seed}/head50']['theta'] if policy == 'head50' else parent_cut, parent_cut])
                    expected = base if policy == 'baseline' else np.where((base == 0) & np.isfinite(score) & (score >= cut), 1, base)
                    np.testing.assert_array_equal(expected, grade)
                    for n, author in enumerate(d['rows']):
                        for q, height in enumerate(E.HEIGHTS):
                            if d['category'][n,q] != 'contact':
                                continue
                            for k in range(4):
                                g = grade[n,k,:,q]
                                first = onset(g > 0)
                                outcome = 'silent' if first < 0 else 'timely' if first <= 13 else 'late'
                                ins = inner[n,k,:,q]
                                first_inner = onset(ins)
                                sm = score[n,k,:,q]-cut[q]
                                om = smooth[n,k,:,q]-th['single']
                                rm = raw[n,k,:,q]-th['single']
                                best = int(np.argmax(sm[:11]))
                                peak_inner = [descriptor(n,k,q,t,'strongest_inner_peak_log') for t in range(11)]
                                peak_rank0 = [descriptor(n,k,q,t,'rank0_peak_log') for t in range(11)]
                                row = dict(split=split, seed=seed, policy=policy, scene=int(d['scene_ids'][n]), replica=k, height=height,
                                    shape_family=author['shape_family'], rho=author['rho'], placement=author['placement'],
                                    background_family=author['background_family'], background_id=author['background_id'],
                                    outcome=outcome, grades=''.join(map(str,g.tolist())), first_any=first,
                                    first_strong=onset(g == 2), first_inner=first_inner,
                                    first_rank0=onset(rank0[n,k,:,q]), first_raw_single=onset(rm >= 0), first_smooth_single=onset(om >= 0),
                                    score_f3=float(score[n,k,0,q]), score_cut=float(cut[q]), score_f3_margin=float(sm[0]),
                                    score_max_timely_margin=float(sm[best]), score_max_timely_frame=best+3,
                                    ordinary_raw_max_timely_margin=float(rm[:11].max()), ordinary_smooth_max_timely_margin=float(om[:11].max()),
                                    m3_max_timely_margin=float((d['m3'][n,k,:11,q]-E.OLD_RAISED).max()),
                                    local_max_timely_margin=float((d['local'][n,k,:11,q]-E.OLD_LOCAL).max()) if np.isfinite(d['local'][n,k,:11,q]).any() else None,
                                    raw_single_cross_without_smooth=int((rm[:11] >= 0).any() and not (om[:11] >= 0).any()),
                                    raw_old_cross_without_smooth=int(raw_old[n,k,:11,q].any() and not E.old_fusion(d['m3'][n,k,:11,q], d['local'][n,k,:11,q]).any()),
                                    inner_timely_frames=int(ins[:11].sum()), inner_timely_longest_run=G.max_run(ins[:11]),
                                    rank0_outside_with_inner_timely_frames=int(outside_with_inner[n,k,:11,q].sum()),
                                    first_inner_to_alert_frames=first-first_inner if first >= 0 and first_inner >= 0 else -1,
                                    inner_peak_max_timely=max((v for v in peak_inner if v is not None), default=None),
                                    rank0_peak_max_timely=max((v for v in peak_rank0 if v is not None), default=None))
                                # For failed timely streams every actual detection path must stay below its fixed cut.
                                if outcome != 'timely':
                                    assert row['ordinary_smooth_max_timely_margin'] < 0 and row['m3_max_timely_margin'] < 0
                                    assert row['local_max_timely_margin'] is None or row['local_max_timely_margin'] < 0
                                    if policy != 'baseline':
                                        assert row['score_max_timely_margin'] < 0
                                events.append(row)
                                for stage, frame in (('f3',3), ('first_inner',first_inner), ('best_score_timely',best+3), ('first_any',first)):
                                    snap = dict(split=split,seed=seed,policy=policy,scene=int(d['scene_ids'][n]),replica=k,height=height,
                                        outcome=outcome,stage=stage,frame=frame,
                                        ordinary_raw_margin=float(rm[frame-3]) if frame >= 0 else None,
                                        ordinary_smooth_margin=float(om[frame-3]) if frame >= 0 else None,
                                        joint_score_margin=float(sm[frame-3]) if frame >= 0 else None,
                                        **{f:descriptor(n,k,q,frame-3,f) if frame >= 0 else None for f in FIELDS})
                                    snapshots.append(snap)
                for n in range(384):
                    for q,height in enumerate(E.HEIGHTS):
                        if d['category'][n,q] != 'contact': continue
                        for k in range(4):
                            bf = onset(grades[f'{seed}/both'][n,k,:,q] > 0)
                            hf = onset(grades[f'{seed}/head50'][n,k,:,q] > 0)
                            pairs.append(dict(split=split,seed=seed,scene=int(d['scene_ids'][n]),replica=k,height=height,
                                shape_family=d['rows'][n]['shape_family'],both_first=bf,head50_first=hf,
                                foregone_timely=int(0 <= bf <= 13 and (hf < 0 or hf > 13)),
                                delayed_common_timely=int(0 <= bf <= 13 and 0 <= hf <= 13 and hf > bf)))
        groups = defaultdict(list)
        for r in events:
            core = f"{r['split']}/{r['seed']}/{r['policy']}/{r['height']}"
            for suffix in ('all', f"family:{r['shape_family']}", f"rho:{r['rho']}", f"placement:{r['placement']}",
                           f"background:{r['background_family']}", f"family_rho:{r['shape_family']}:{r['rho']}"):
                for outcome in ('all',r['outcome']):
                    groups[f'{core}/{suffix}/{outcome}'].append(r)
        G.write_csv(OUT/'events.csv', events)
        G.write_csv(OUT/'snapshots.csv', snapshots)
        G.write_csv(OUT/'working_point_pairs.csv', pairs)
        C.save(OUT/'summary.json', {key:aggregate(rows) for key,rows in sorted(groups.items())})
        C.save(OUT/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-started,events=len(events),snapshots=len(snapshots),
            pairs=len(pairs),groups=len(groups),source_sha256=C.sha(Path(__file__)),GPU_seconds=0,fit=0,prediction=0,new_raw=0,threshold_changes=0))
        print(f'COMPLETE events={len(events)} snapshots={len(snapshots)} groups={len(groups)} seconds={time.monotonic()-started:.3f}')
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    run()
