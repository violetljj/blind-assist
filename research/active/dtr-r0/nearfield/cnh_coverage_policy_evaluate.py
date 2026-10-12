"""Frozen M3 coverage intervention evaluation; evaluator-only truth selection.

No model, calibration, renderer or protected test access. The target-band
metrics match the original pilot2 units. Both public queries' actual-clear
first stops are separately retained so query-band truth is not an alert gate.
"""
import argparse
from pathlib import Path
import time

import numpy as np

import cnh_location_reference_evaluate as L
import cnh_readout_pilot2_evaluate as P2
import cnh_temporal_readout_evaluate as T
import cnh_three_level_sequence as SE
import cnh_unknown_target_reference_evaluate as U

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-coverage-policy-20261004'
RC = np.asarray([2.5, 2.1, 1.7, 1.3, 1.0])
FRAMES = np.arange(3, 16)
THRESHOLD = .8557642486787612
BOOT_SEED = 2026100507
DELTAS = np.asarray([1., 2., 5., -5., -10., -15., -20.])


def smooth(raw):
    """Keep two query outputs and the original causal retained-frame warmup."""
    raw = np.asarray(raw, np.float64)
    if raw.shape[-2:] != (13, 2) or not np.isfinite(raw).all():
        raise ValueError('Finite two-query raw logits with 13 retained frames required')
    return SE.smooth(raw.reshape(-1, 1, 13, 2)).reshape(raw.shape)


def load_baseline():
    scores, scenes, categories, visible, refs, covered, receipt = L.load_baselines()
    ids, all_categories = [], []
    for i, scene in enumerate(scenes):
        truth = U.read(L.OLD/'truth/evaluation'/f"unit{scene['unit']}.json")
        if truth['mode'] == 0:
            ids.append(i)
            all_categories.append(np.asarray(truth['categories'])[:, FRAMES, :])
    if len(ids) != 24:
        raise ValueError('All 24 original mode0 scenes must be retained')
    rows, _ = T.rows_for(L.OLD, 'fresh_evaluation')
    raw, _ = T.prediction(L.OLD, 'fresh_evaluation', 'M3')
    all_smoothed, keys = T.ordered_episodes(raw, rows, fresh=True)
    units = [s['unit'] for s in scenes]
    if keys != [(u, d, k) for u in units for d in range(7) for k in range(4)]:
        raise ValueError('Original two-query baseline identity differs')
    full = all_smoothed.reshape(48, 7, 4, 13, 2)[ids]
    selected = [scenes[i] for i in ids]
    baseline = scores['M3'][ids]
    band = np.stack([full[i, ..., s['group']] for i, s in enumerate(selected)])
    np.testing.assert_array_equal(baseline, band)
    return dict(units=np.asarray([s['unit'] for s in selected]), scenes=selected,
                scores=baseline, both_query_scores=full,
                categories=categories[ids], all_categories=np.asarray(all_categories),
                refs=refs[ids], covered=covered[ids], receipt=receipt)


def group_masks(scenes):
    masks = L.groups_for(scenes)
    if (int(masks['FOV_OUT'].sum()), int(masks['FOV_IN'].sum())) != (11, 13):
        raise ValueError('Frozen mode0 cohort must retain FOV_OUT11 and FOV_IN13')
    return masks


def ledger(score, ranges, covered):
    return L.events(score, dict(threshold=THRESHOLD, operator='>='), ranges, covered)


def rate(flags, denominator, keep, boot):
    axes = tuple(range(1, flags.ndim))
    return U.pooled_rate(flags.sum(axes), denominator.sum(axes), keep, boot)


def sequence_metrics(score, both, baseline_events, base_both, data, boot):
    scenes, categories = data['scenes'], data['categories']
    ranges = np.asarray([s['front_range_m'] for s in scenes])
    groups = group_masks(scenes)
    events = ledger(score, ranges, data['covered'])
    public_stops = (both >= THRESHOLD).any(-2)
    base_public_stops = (base_both >= THRESHOLD).any(-2)
    output = {}
    for group, keep in groups.items():
        cell = dict(scenes=int(keep.sum()), branches={})
        for title, ids in [('inside1_2cm', [0, 1]), ('inside5cm', [2]),
                           ('outside5_10cm', [3, 4]), ('outside15_20cm', [5, 6])]:
            den = np.ones((len(scenes), len(ids), 4), bool)
            valid = np.broadcast_to(data['covered'][:, ids, None], den.shape)
            first, _ = rate(events['stopped'][:, ids], den, keep, boot)
            timely, draws = rate(events['timely'][:, ids] & valid, valid, keep, boot)
            base_timely, base_draws = rate(baseline_events['timely'][:, ids] & valid, valid, keep, boot)
            rescues = events['timely'][:, ids] & ~baseline_events['timely'][:, ids] & valid
            losses = baseline_events['timely'][:, ids] & ~events['timely'][:, ids] & valid
            hit = keep[:, None, None] & events['timely'][:, ids] & valid
            leads = (events['first_range'][:, ids]-.5)/.8
            cell['branches'][title] = dict(first_stops=first, timely=timely,
                censored_n=int((~valid)[keep].sum()),
                median_lead_conditional_timely_s=float(np.median(leads[hit])) if hit.any() else None,
                paired_minus_original=dict(delta_stops=timely['stops']-base_timely['stops'],
                    delta_rate=timely['rate']-base_timely['rate'] if timely['rate'] is not None else None,
                    ci95=U.interval(draws-base_draws), rescues=int(rescues[keep].sum()),
                    losses=int(losses[keep].sum()), n=timely['n']))
        # Strict all-object-clear episode truth, frozen body-frame trajectories.
        clear = np.all(categories[:, [5, 6]] == 'clear', axis=-1)
        clear_den = np.broadcast_to(clear[..., None], (len(scenes), 2, 4))
        metric, _ = rate(events['stopped'][:, [5, 6]] & clear_den, clear_den, keep, boot)
        minutes = metric['n']*13*.2/60
        base_metric, _ = rate(baseline_events['stopped'][:, [5, 6]] & clear_den, clear_den, keep, boot)
        cell['actual_clear_outer15_20_target_band'] = dict(**metric, proxy_minutes=minutes,
            first_stops_per_proxy_minute=metric['stops']/minutes if minutes else None,
            delta_stops_vs_original=metric['stops']-base_metric['stops'])
        clear_two = np.all(data['all_categories'][:, [5, 6]] == 'clear', axis=-2)
        clear_den_two = np.broadcast_to(clear_two[:, :, None, :], (len(scenes), 2, 4, 2))
        metric_two, _ = rate(public_stops[:, [5, 6]] & clear_den_two, clear_den_two, keep, boot)
        base_two, _ = rate(base_public_stops[:, [5, 6]] & clear_den_two, clear_den_two, keep, boot)
        two_minutes = metric_two['n']*13*.2/60
        cell['actual_clear_outer15_20_both_public_queries'] = dict(**metric_two,
            proxy_minutes=two_minutes,
            denominator_unit='query episodes: separate HEAD and BODY exposure',
            first_stops_per_proxy_minute=metric_two['stops']/two_minutes if two_minutes else None,
            delta_stops_vs_original=metric_two['stops']-base_two['stops'])
        # One physical episode, either query alarms; both query boxes must be
        # all-object clear throughout. This does not double exposure minutes.
        joint_clear = np.all(data['all_categories'][:, [5, 6]] == 'clear', axis=(-2, -1))
        joint_den = np.broadcast_to(joint_clear[..., None], (len(scenes), 2, 4))
        joint_stopped = public_stops[:, [5, 6]].any(-1)
        base_joint_stopped = base_public_stops[:, [5, 6]].any(-1)
        joint_metric, _ = rate(joint_stopped & joint_den, joint_den, keep, boot)
        base_joint, _ = rate(base_joint_stopped & joint_den, joint_den, keep, boot)
        joint_minutes = joint_metric['n']*13*.2/60
        cell['actual_clear_outer15_20_joint_physical_episodes'] = dict(**joint_metric,
            proxy_minutes=joint_minutes,
            denominator_unit='physical episodes; both queries clear, either query stops',
            first_stops_per_proxy_minute=joint_metric['stops']/joint_minutes if joint_minutes else None,
            delta_stops_vs_original=joint_metric['stops']-base_joint['stops'])
        output[group] = cell
    return output, events


def auc_metrics(scores, data, boot):
    scenes = data['scenes']
    ranges = np.asarray([s['front_range_m'] for s in scenes])
    domains = {f'{lo:g}-{hi:g}m': (ranges >= lo)&(ranges < hi) for lo, hi in U.BINS}
    domains[L.PRIMARY] = (ranges >= 1.2)&(ranges < 2.1)
    output, per_scene = {}, {}
    for domain, masks in domains.items():
        values = {arm: np.asarray([U.binary_auc(score[i, [0, 1]][..., mask],
                                             score[i, [4, 5, 6]][..., mask])
                                 for i, mask in enumerate(masks)], float)
                  for arm, score in scores.items()}
        per_scene[domain] = {arm: [None if not np.isfinite(x) else float(x) for x in value]
                             for arm, value in values.items()}
        output[domain] = {}
        for group, keep in group_masks(scenes).items():
            output[domain][group] = dict(scenes=int(keep.sum()),
                positive_frame_realizations_n=int(masks[keep].sum()*2*4),
                negative_frame_realizations_n=int(masks[keep].sum()*3*4),
                arms={arm: U.macro_summary(value, keep, boot)[0] for arm, value in values.items()},
                paired_minus_original={arm: U.macro_summary(value-values['M3_original'], keep, boot)[0]
                                       for arm, value in values.items() if arm != 'M3_original'})
    return output, per_scene


def decide(sequence):
    passing = [float(rc) for rc in RC if sequence[f'return_at_{rc:g}m']['FOV_OUT']['branches']['inside1_2cm']['timely']['stops'] >= 75]
    latest = min(passing) if passing else None
    if latest is None:
        branch = 'INTERMEDIATE'
    elif latest <= 1.3:
        branch = 'COVERAGE_STRATEGY_CANDIDATE'
    elif latest >= 2.1:
        branch = 'EARLY_RETURN_SENSOR_COMPARISON_PROPOSED'
    else:
        branch = 'INTERMEDIATE'
    return dict(branch=branch, passing_return_distances_m=passing, latest_successful_return_distance_m=latest,
        recovery_status='TARGET_NOT_REACHED_AT_ANY_RC' if not passing else 'TARGET_REACHED',
        frozen_target='FOV_OUT shallow timely >=75/88',
        automatic_training=False, automatic_sensor_comparison=False,
        interpretation='Controlled simulator yaw intervention; does not include detection delay or human reaction. Geometric branch is not a real-user practicality verdict.')


def evaluate(out=OUT, score_path=None):
    out = Path(out)
    started = time.monotonic()
    data = load_baseline()
    groups = group_masks(data['scenes'])
    plan_path = out/'PLAN.json'
    plan_sha = None
    if plan_path.exists():
        plan = U.read(plan_path)
        if (plan['units'] != data['units'].tolist() or plan['rc_m'] != RC.tolist()
                or plan['frames'] != FRAMES.tolist() or plan['K'] != 4):
            raise ValueError('Frozen coverage PLAN does not match full retained axes')
        for group in ('FOV_OUT', 'FOV_IN'):
            selected_units = data['units'][groups[group]].tolist()
            if plan['cohorts'][group] != selected_units:
                raise ValueError('Frozen baseline-visibility cohort differs: '+group)
        plan_sha = U.sha(plan_path)
    ranges = np.asarray([s['front_range_m'] for s in data['scenes']])
    base_events = ledger(data['scores'], ranges, data['covered'])
    baseline_counts = {g: int(base_events['timely'][groups[g], :2].sum()) for g in ('FOV_OUT', 'FOV_IN')}
    if baseline_counts['FOV_OUT'] != 20:
        raise ValueError('Frozen baseline FOV_OUT timely identity failed: '+str(baseline_counts))
    boot = P2.boot_weights(24, BOOT_SEED)
    path = Path(score_path) if score_path else out/'predictions/return_scores.npz'
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z['units'], data['units']) or not np.array_equal(z['frames'], FRAMES) or not np.array_equal(z['rc'], RC):
            raise ValueError('Intervention units/frames/rc axes differ from frozen full mode0 design')
        raw = np.asarray(z['raw_logits'] if 'raw_logits' in z else z['raw_scores'], np.float64)
    if raw.shape != (24, 5, 7, 4, 13, 2):
        raise ValueError('Every mode0 scene x5rc x7delta xK4 x13frames x2queries required')
    unchanged_first = data['both_query_scores'][:, None, :, :, 0, :]
    first_delta = np.abs(raw[:, :, :, :, 0, :]-unchanged_first)
    first_max_abs = float(first_delta.max())
    if first_max_abs > 1e-5:
        raise ValueError('Unchanged pre-intervention frame3 M3 logit parity failed: '+str(first_max_abs))
    full = smooth(raw)
    score = np.stack([full[i, ..., s['group']] for i, s in enumerate(data['scenes'])])
    scores = {'M3_original': data['scores']}
    sequence, ledger_arrays = {}, {}
    baseline_seq, _ = sequence_metrics(data['scores'], data['both_query_scores'], base_events,
                                      data['both_query_scores'], data, boot)
    sequence['M3_original'] = baseline_seq
    for j, rc in enumerate(RC):
        arm = f'return_at_{rc:g}m'
        scores[arm] = score[:, j]
        sequence[arm], events = sequence_metrics(score[:, j], full[:, j], base_events,
                                                data['both_query_scores'], data, boot)
        for field in ('alarm', 'stopped', 'timely', 'first_index', 'first_range'):
            ledger_arrays[arm+'_'+field] = events[field]
    auc, per_scene = auc_metrics(scores, data, boot)
    ledger_path = out/'evaluation/ledger.npz'
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ledger_path, units=data['units'], rc=RC, frames=FRAMES,
        baseline_scores=data['scores'], smoothed_scores=score,
        baseline_both_query_scores=data['both_query_scores'], both_query_smoothed_scores=full,
        both_query_stopped=(full >= THRESHOLD).any(-2),
        both_query_first_index=np.where((full >= THRESHOLD).any(-2), (full >= THRESHOLD).argmax(-2), -1),
        baseline_timely=base_events['timely'], baseline_stopped=base_events['stopped'], **ledger_arrays)
    result = dict(status='COMPLETE', task='coverage policy A; consumed synthetic Development',
        units=data['units'].tolist(), rc_m=RC.tolist(), intrusion_cm=DELTAS.tolist(),
        realizations_per_scene_rc_delta=4, decision_frames=FRAMES.tolist(),
        scenes=dict(all=24, FOV_OUT=11, FOV_IN=13), baseline_shallow_timely=baseline_counts,
        threshold=dict(value=THRESHOLD, operator='>=', origin='Original frozen M3 natural calibration; no refit'),
        smoothing=dict(weights=[1, 2, 4, 8, 16], retained_frames_only=True,
                       warmup='Use available trailing weights normalized by their sum'),
        sequence=sequence, auc=auc, per_scene_auc=per_scene, decision=decide(sequence),
        baseline_identity_receipt=data['receipt'], score_sha256=U.sha(path), ledger_sha256=U.sha(ledger_path),
        plan_sha256=plan_sha,
        pre_intervention_identity=dict(status='PASS', frame=3, front_range_m=float(ranges[0, 0]),
            all_queries_all_5rc_n=int(first_delta.size), max_abs_raw_logit=first_max_abs, tolerance=1e-5),
        evaluation_s=time.monotonic()-started,
        limits=['FOV group fixed using original visibility, never recategorized after intervention',
                'Truth band selection is evaluator-only; both public query clear cost separately retained',
                'First-stop proxy exposure includes all 13x0.2s; not a real walking burden',
                'Repeated noise/pose realizations are not independent scenes; bootstrap whole scenes',
                'GT range schedules prescribed intervention; cannot be used as an online cue'])
    L.save(out/'result.json', result)
    lines = ['arm,FOV_OUT_timely_n,FOV_OUT_timely_den,FOV_IN_timely_n,FOV_IN_timely_den,OUT_clear_stops,OUT_clear_n,IN_clear_stops,IN_clear_n']
    for arm, cells in sequence.items():
        oo, ii = cells['FOV_OUT'], cells['FOV_IN']
        ot, it = oo['branches']['inside1_2cm']['timely'], ii['branches']['inside1_2cm']['timely']
        oc, ic = oo['actual_clear_outer15_20_target_band'], ii['actual_clear_outer15_20_target_band']
        lines.append(f"{arm},{ot['stops']},{ot['n']},{it['stops']},{it['n']},{oc['stops']},{oc['n']},{ic['stops']},{ic['n']}")
    with (out/'evaluation/counts.csv').open('w', encoding='utf8', newline='\n') as stream:
        stream.write('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--scores', type=Path)
    parser.add_argument('--baseline-check', action='store_true')
    args = parser.parse_args()
    if args.baseline_check:
        data = load_baseline()
        ranges = np.asarray([s['front_range_m'] for s in data['scenes']])
        events = ledger(data['scores'], ranges, data['covered'])
        print({g: dict(scenes=int(keep.sum()), shallow_timely=int(events['timely'][keep, :2].sum()),
                       shallow_n=int(keep.sum()*8)) for g, keep in group_masks(data['scenes']).items()})
    else:
        result = evaluate(args.out, args.scores)
        print(result['decision'])
        print((args.out/'evaluation/counts.csv').read_text(encoding='utf8'))


if __name__ == '__main__':
    main()
