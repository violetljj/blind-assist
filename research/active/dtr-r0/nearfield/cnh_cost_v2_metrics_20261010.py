"""Fixed cost-v2 calibration and descriptive scene-cluster evaluation.

Only new cal negatives select the one additional-evidence gate. Every negative
score tie is replayed, including nonmonotone episode/upgrade costs. All frozen
model scoring uses the unchanged E2E score_split (HGB forward is CPU evaluation
time). Hold readout opens only after immutable calibration sealing. No training,
contact utility selection, old hold/validation outcomes or protected access.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import cnh_counterfactual_common_dev as C
import cnh_graded_evidence_dev as G
import cnh_frozen_e2e_tof_20261010 as F
import cnh_frozen_e2e_metrics_20261010 as M

HEIGHTS = ('HEAD', 'BODY')
MAIN_WEIGHT = .25
PASS_LAYERS = ('0-5cm', '5-10cm', '10-20cm', '20-35cm')


def negative_masks(category, rows):
    """Evaluator-only physical surface gap, independent of query alert grade."""
    category = np.asarray(category)
    passed = (category == 'pass').any(1) & ~(category == 'contact').any(1)
    clear = (category == 'clear').all(1)
    gap = np.asarray([float(r['lateral_gap_m']) for r in rows])
    if np.any(passed & ((gap < 0) | ~np.isfinite(gap))):
        raise ValueError('Pass requires a finite nonnegative physical surface gap')
    layer = np.full(len(rows), '', dtype='U16')
    layer[passed & (gap <= .05)] = PASS_LAYERS[0]
    layer[passed & (gap > .05) & (gap <= .10)] = PASS_LAYERS[1]
    layer[passed & (gap > .10) & (gap <= .20)] = PASS_LAYERS[2]
    layer[passed & (gap > .20) & (gap <= .35)] = PASS_LAYERS[3]
    if np.any(passed & (layer == '')):
        raise ValueError('Pass outside sealed 0-35cm layer range')
    near = passed & (gap <= .10)
    far = passed & (gap > .10)
    masks = dict(all_negative=passed | clear, near_pass=near, far_pass=far,
                 clear=clear, far_pass_plus_clear=far | clear)
    masks.update({name: passed & (layer == name) for name in PASS_LAYERS})
    return masks, near, layer, gap


def _clip_quarters(emitted, near):
    joint = emitted.max(-1)
    light_weight = np.where(near, 1, 4)[:, None]
    return ((joint == 1).sum(-1) * light_weight + (joint == 2).sum(-1) * 4)


def matching(both, old5, margin, category, rows, check):
    """Lowest canonical feasible tau, complete ties, joint weighted total cap.

    old5 bool grades always survive as strong. Weight .25 is represented in
    integer quarters; all negative partitions share one total calibration cap.
    Removing a light onset can expose a later strong onset, so costs need not
    decrease with threshold. No monotonicity shortcut or contact chooser.
    """
    masks, near, _, _ = negative_masks(category, rows)
    negative = masks['all_negative']
    grade = np.array(both[negative], dtype=np.int8, copy=True)
    original = np.asarray(old5[negative], dtype=bool)
    evidence = np.asarray(margin[negative], dtype=float)
    if grade.shape != original.shape or grade.shape != evidence.shape:
        raise ValueError('Calibrating aligned grades/margins required')
    grade = grade.reshape(-1, 13, 2)
    original = original.reshape(-1, 13, 2)
    evidence = evidence.reshape(-1, 13, 2)
    near_clip = np.repeat(near[negative], both.shape[1])
    baseline = 2 * original.astype(np.int8)
    # Helper expects [N,K,T,2]; calibration clips are kept correlated in reports.
    replay = lambda g: F.emit_fast(g[:, None])[:, 0]
    quarters = lambda e: _clip_quarters(e[:, None], near_clip)[:, 0]
    cap = int(quarters(replay(baseline)).sum())
    count = quarters(replay(grade))
    total = int(count.sum())
    additional = (grade > 0) & ~original
    if np.any(additional & (~np.isfinite(evidence) | (evidence < 0))):
        raise ValueError('Frozen additional positive grades require finite >=0 margin')
    clips, frames, queries = np.where(additional)
    values = evidence[additional]
    order = np.argsort(values, kind='stable')
    values, clips, frames, queries = [a[order] for a in (values, clips, frames, queries)]
    bounds = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)] if len(values) else np.array([0])
    chosen = 0. if total <= cap else None
    curve = [dict(tau=0., tau_kind='finite', weighted_cost_quarters=total,
                  weighted_cost=total/4, feasible=total <= cap)]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        check()
        affected = np.unique(clips[start:stop])
        previous = count[affected].copy()
        grade[clips[start:stop], frames[start:stop], queries[start:stop]] = 0
        # near_clip must match affected subset in the exact local replay.
        emission = replay(grade[affected])
        joint = emission.max(-1)
        count[affected] = ((joint == 1).sum(-1)*np.where(near_clip[affected], 1, 4)
                           + (joint == 2).sum(-1)*4)
        total += int((count[affected]-previous).sum())
        tau = float(np.nextafter(values[start], np.inf))
        feasible = total <= cap
        curve.append(dict(tau=tau, tau_kind='finite', weighted_cost_quarters=total,
                          weighted_cost=total/4, feasible=bool(feasible)))
        if chosen is None and feasible:
            chosen = tau
    np.testing.assert_array_equal(grade, baseline)
    assert total == cap
    curve.append(dict(tau=None, tau_kind='positive_infinity', weighted_cost_quarters=cap,
                      weighted_cost=cap/4, feasible=True))
    if chosen is None:
        chosen = np.inf
    matched = apply_gate(both, old5, margin, chosen)
    record = dict(tau=float(chosen) if np.isfinite(chosen) else None,
                  tau_kind='finite' if np.isfinite(chosen) else 'positive_infinity',
                  weighted_cap_quarters=cap, weighted_cap=cap/4,
                  candidate_thresholds=len(curve), negative_margin_ties=len(bounds)-1,
                  negative_scenes=int(negative.sum()), contact_utility_access=False,
                  main_near_light_weight=MAIN_WEIGHT,
                  selection='Lowest canonical tau: 0, nextafter(each finite eligible negative margin), +inf; all ties replayed, no monotonicity assumption; one joint all-negative weighted cap')
    return matched, record, curve


def apply_gate(both, old5, margin, tau):
    return np.where(old5, 2, np.where((both > 0) & (margin >= tau), both, 0)).astype(np.int8)


def contact_masks(category, rows):
    contact = category == 'contact'
    dark = np.asarray([bool(r['dark_thin']) for r in rows])
    sign = np.asarray([r['shape_family'] == 'sign_edge' for r in rows])
    answer = {}
    for q, height in enumerate(HEIGHTS):
        height_mask = np.zeros_like(contact)
        height_mask[:, q] = contact[:, q]
        answer[height] = height_mask
        answer[height+'_dark_thin'] = height_mask & dark[:, None]
        answer[height+'_sign_edge'] = height_mask & sign[:, None]
    answer['HEAD_BODY'] = contact
    return answer


def first_notice(emitted):
    positive = emitted > 0
    return np.where(positive.any(2), positive.argmax(2), -1)


def contact_summary(first, baseline, masks):
    result = {}
    timely = (first >= 0) & (first < 11)
    base_timely = (baseline >= 0) & (baseline < 11)
    for name, mask in masks.items():
        truth = np.broadcast_to(mask[:, None], first.shape)
        rescued = truth & timely & ~base_timely
        lost = truth & base_timely & ~timely
        hit = truth & timely
        selected = first[truth]
        scene_den = mask.sum(-1)*first.shape[1]
        scene_hit = hit.sum((1, 2))
        result[name] = dict(denominator=int(truth.sum()), scene_denominator=int(mask.any(1).sum()),
            timely=int(hit.sum()), late=int((selected >= 11).sum()), silent=int((selected < 0).sum()),
            baseline_timely=int((truth & base_timely).sum()), rescue=int(rescued.sum()),
            loss=int(lost.sum()), net=int(rescued.sum()-lost.sum()),
            scenes_with_timely=int((scene_hit > 0).sum()),
            scenes_all_events_timely=int(((scene_den > 0) & (scene_hit == scene_den)).sum()),
            scenes_with_rescue=int(rescued.any((1, 2)).sum()),
            scenes_with_loss=int(lost.any((1, 2)).sum()),
            scene_counts=[dict(scene=int(n), denominator=int(scene_den[n]), timely=int(scene_hit[n]),
                               rescue=int(rescued[n].sum()), loss=int(lost[n].sum()))
                          for n in np.flatnonzero(scene_den)])
    return result


def cost_summary(emitted, category, rows):
    masks, near, _, _ = negative_masks(category, rows)
    joint = emitted.max(-1)
    output = {}
    for name, mask in masks.items():
        subset = joint[mask]
        positive = subset > 0
        light, strong = int((subset == 1).sum()), int((subset == 2).sum())
        near_light = int(((joint == 1) & (mask & near)[:, None, None]).sum())
        full_light = light-near_light
        denom = int(mask.sum()*joint.shape[1])
        notified = int(positive.any(-1).sum())
        scene_notified = int(positive.any((1, 2)).sum())
        output[name] = dict(scene_denominator=int(mask.sum()), clip_denominator=denom,
            light_notifications=light, strong_notifications=strong,
            notifications=light+strong, near_light_notifications=near_light,
            weighted_cost=full_light+strong+MAIN_WEIGHT*near_light,
            weighted_cost_w0=full_light+strong,
            weighted_cost_w05=full_light+strong+.5*near_light,
            clips_with_any_notification=notified,
            clips_with_light_notification=int((subset == 1).any(-1).sum()),
            clips_with_strong_notification=int((subset == 2).any(-1).sum()),
            clip_notification_rate=notified/denom if denom else None,
            scenes_with_any_notification=scene_notified,
            scene_notification_rate=scene_notified/int(mask.sum()) if mask.any() else None,
            scene_counts=[dict(scene=int(n), light=int((joint[n] == 1).sum()),
                               strong=int((joint[n] == 2).sum()),
                               clips_with_any_notification=int((joint[n] > 0).any(-1).sum()))
                          for n in np.flatnonzero(mask)])
    return output


def _metadata(row, split, n, k):
    return dict(split=split, scene=n, replica=k, scene_uid=row['scene_uid'],
                physical_key=json.dumps(row['physical_key'], sort_keys=True),
                shape_family=row['shape_family'], size_variant=row['size_variant'],
                target_thickness_m=row['target_thickness_m'], rho=row['rho'],
                dark_thin=bool(row['dark_thin']), background_id=row['background_id'],
                background_family=row['background_family'], lateral_gap_m=row['lateral_gap_m'],
                pass_layer=row.get('pass_layer', ''), placement=row['placement'])


def strong_signal(reports):
    baseline = reports['hold/fixed/old5']
    base_cost = baseline['costs']['all_negative']['weighted_cost']
    base_far = baseline['costs']['far_pass_plus_clear']['notifications']
    answer = {}
    for arm in ('both', 'matched'):
        main = reports[f'hold/{G.SEEDS[0]}/{arm}']
        nets = {str(s): reports[f'hold/{s}/{arm}']['contacts']['HEAD_BODY']['net'] for s in G.SEEDS}
        net = nets[str(G.SEEDS[0])]
        total = main['contacts']['HEAD_BODY']['denominator']
        candidate_cost = main['costs']['all_negative']['weighted_cost']
        far = main['costs']['far_pass_plus_clear']['notifications']
        cost_eligible = arm == 'matched' or candidate_cost <= base_cost
        signal = bool(cost_eligible and total == 512 and net >= 10 and
                      far <= 1.10*base_far and all(nets[str(s)] > 0 for s in G.SEEDS[1:]))
        answer[arm] = dict(eligible_by_declared_arm_rule=bool(cost_eligible),
            hold_weighted_cost=candidate_cost, baseline_hold_weighted_cost=base_cost,
            main_net=net, main_denominator=total, required_net=10,
            far_pass_plus_clear_notifications=far, baseline_far_pass_plus_clear_notifications=base_far,
            far_pass_plus_clear_cap=1.10*base_far, seed_nets=nets,
            sensitivity_direction_consistent=all(nets[str(s)] > 0 for s in G.SEEDS[1:]),
            strong_signal=signal)
    any_signal = any(r['strong_signal'] for r in answer.values())
    return dict(arms=answer, any_strong_signal=any_signal,
                recommendation='建议 both/匹配臂升为默认候选' if any_signal else '保留原5格',
                interpretation='Predeclared signal evaluated on fixed arms; no hold threshold or mechanism selection; simulator Development evidence')


def calibrate(out, rowscal, check):
    """Seal cal before the controller begins hold rendering/frozen forwards."""
    out = Path(out)
    began = time.monotonic()
    check()
    with np.load(out/'data/cal/geometry.npz', allow_pickle=False) as geometry:
        cal_category = geometry['category'].copy()
    hgb_began = time.monotonic()
    with threadpool_limits(limits=2):
        cal = F.score_split(out, 'cal')
    hgb_seconds = time.monotonic()-hgb_began
    calibrations, curves, matched = {}, [], []
    for si, seed in enumerate(G.SEEDS):
        check()
        grade, record, curve = matching(cal['both'][si], cal['old5'] > 0,
                                        cal['margin'][si], cal_category, rowscal, check)
        calibrations[str(seed)] = record
        matched.append(grade)
        curves.extend([dict(seed=int(seed), **r) for r in curve])
    cal['matched'] = np.asarray(matched)
    G.write_csv(out/'calibration_curve.csv', curves)
    np.savez_compressed(out/'cal_scores_readout.npz', **cal)
    keys = ['fixed/m3', 'fixed/old5'] + [f'{s}/{arm}' for s in G.SEEDS for arm in ('both', 'matched')]
    grades = [cal['m3'], cal['old5']] + [cal[arm][si] for si, s in enumerate(G.SEEDS) for arm in ('both', 'matched')]
    np.savez_compressed(out/'cal_grades_notifications.npz', keys=np.asarray(keys), grades=np.asarray(grades),
        notifications=np.asarray([M.replay_gap1(g) for g in grades]), category=cal_category,
        scene_ids=np.arange(len(rowscal)), margin=cal['margin'], joint_scores=cal['joint_scores'], seeds=np.asarray(G.SEEDS))
    F.save_new(out/'sealed_calibration.json', dict(calibrations=calibrations,
        sealed_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=C.sha(out/'PLAN.json'), source_sha256=C.sha(Path(__file__)),
        cal_scores_sha256=C.sha(out/'data/cal/scores.npz'),
        cal_geometry_sha256=C.sha(out/'data/cal/geometry.npz'),
        cal_scores_readout_sha256=C.sha(out/'cal_scores_readout.npz'),
        cal_grades_notifications_sha256=C.sha(out/'cal_grades_notifications.npz'),
        calibration_curve_sha256=C.sha(out/'calibration_curve.csv'),
        selection='All negative joint weighted cost only; contact utility unused; hold geometry/scores unopened by evaluator',
        scientific_order='Controller seals cal before hold rendering and all frozen hold forwards',
        hgb_forward_seconds=hgb_seconds))
    return dict(duration_seconds=time.monotonic()-began, hgb_forward_seconds=hgb_seconds,
                sealed_calibration_sha256=C.sha(out/'sealed_calibration.json'))


def evaluate(out, rows, check):
    """Reuse the immutable cal seal; evaluate the subsequently rendered hold."""
    out = Path(out)
    began = time.monotonic()
    check()
    if not (out/'sealed_calibration.json').exists():
        calibrate(out, rows['cal'], check)
    seal = C.sha(out/'sealed_calibration.json')
    binding = C.read(out/'sealed_calibration.json')
    for name, field in [('PLAN.json', 'plan_sha256'), ('cal_scores_readout.npz', 'cal_scores_readout_sha256'),
                        ('cal_grades_notifications.npz', 'cal_grades_notifications_sha256'),
                        ('calibration_curve.csv', 'calibration_curve_sha256'),
                        ('data/cal/scores.npz', 'cal_scores_sha256'),
                        ('data/cal/geometry.npz', 'cal_geometry_sha256')]:
        if C.sha(out/name) != binding[field]:
            raise ValueError('Sealed calibration binding changed: '+name)
    calibrations = binding['calibrations']
    with np.load(out/'cal_scores_readout.npz', allow_pickle=False) as saved:
        cal = {name:saved[name].copy() for name in saved.files}
    with np.load(out/'data/cal/geometry.npz', allow_pickle=False) as geometry:
        cal_category = geometry['category'].copy()
    # First hold evaluator truth/scores access is after all seeds are bound.
    with np.load(out/'data/hold/geometry.npz', allow_pickle=False) as geometry:
        hold_category = geometry['category'].copy()
    hgb_began = time.monotonic()
    with threadpool_limits(limits=2):
        hold = F.score_split(out, 'hold')
    hgb_seconds = time.monotonic()-hgb_began
    held_matched = []
    for si, seed in enumerate(G.SEEDS):
        record = calibrations[str(seed)]
        tau = record['tau'] if record['tau_kind'] == 'finite' else np.inf
        held_matched.append(apply_gate(hold['both'][si], hold['old5'] > 0, hold['margin'][si], tau))
    hold['matched'] = np.asarray(held_matched)
    reports, event_rows, notification_rows = {}, [], []
    for split, scores, category in (('cal', cal, cal_category), ('hold', hold, hold_category)):
        check()
        base_first = first_notice(M.replay_gap1(scores['old5']))
        masks = contact_masks(category, rows[split])
        cost_masks, near, layer, gap = negative_masks(category, rows[split])
        negative = cost_masks['all_negative']
        arms = [('fixed/m3', scores['m3']), ('fixed/old5', scores['old5'])]
        arms += [(f'{s}/{arm}', scores[arm][si]) for si, s in enumerate(G.SEEDS)
                 for arm in ('both', 'matched')]
        saved_grades, saved_notices, saved_keys = [], [], []
        for key, grade in arms:
            check()
            notice = M.replay_gap1(grade)
            np.testing.assert_array_equal(notice, F.emit_fast(grade))
            first = first_notice(notice)
            reports[f'{split}/{key}'] = dict(contacts=contact_summary(first, base_first, masks),
                                           costs=cost_summary(notice, category, rows[split]))
            seed, arm = key.split('/')
            for n, row in enumerate(rows[split]):
                for k in range(grade.shape[1]):
                    metadata = _metadata(row, split, n, k)
                    for q, height in enumerate(HEIGHTS):
                        index = int(first[n, k, q])
                        base_index = int(base_first[n, k, q])
                        timely = 0 <= index < 11
                        baseline_timely = 0 <= base_index < 11
                        contact = category[n, q] == 'contact'
                        event_rows.append(dict(**metadata, seed=seed, arm=arm,
                            height=height, category=str(category[n, q]), first_index=index,
                            first_nominal_frame=index+3 if index >= 0 else -1,
                            outcome='timely' if timely else 'late' if index >= 11 else 'silent',
                            old5_first_index=base_index,
                            rescue=int(contact and timely and not baseline_timely),
                            loss=int(contact and baseline_timely and not timely),
                            light_notifications=int((notice[n, k, :, q] == 1).sum()),
                            strong_notifications=int((notice[n, k, :, q] == 2).sum())))
                    joint = notice[n, k].max(-1)
                    for fj in np.flatnonzero(joint):
                        union_grade = int(joint[fj])
                        light_near = bool(near[n] and union_grade == 1)
                        weight = lambda w: (w if light_near else 1.) if negative[n] else 0.
                        notification_rows.append(dict(**metadata, seed=seed, arm=arm,
                            nominal_frame=int(fj)+3, frame_index=int(fj), union_grade=union_grade,
                            HEAD_grade=int(notice[n, k, fj, 0]), BODY_grade=int(notice[n, k, fj, 1]),
                            HEAD_truth=str(category[n, 0]), BODY_truth=str(category[n, 1]),
                            negative_cost=int(negative[n]), partition='near_pass' if near[n] else
                            'far_pass' if cost_masks['far_pass'][n] else 'clear' if cost_masks['clear'][n] else 'contact',
                            evaluated_pass_layer=str(layer[n]), weight_w0=weight(0.),
                            weight_w025=weight(.25), weight_w05=weight(.5)))
            saved_keys.append(key)
            saved_grades.append(grade)
            saved_notices.append(notice)
        if split == 'hold':
            np.savez_compressed(out/f'{split}_grades_notifications.npz', keys=np.asarray(saved_keys),
                grades=np.asarray(saved_grades), notifications=np.asarray(saved_notices),
                category=category, scene_ids=np.arange(len(rows[split])),
                margin=scores['margin'], joint_scores=scores['joint_scores'], seeds=np.asarray(G.SEEDS))
        else:
            with np.load(out/'cal_grades_notifications.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['keys'], saved_keys)
                np.testing.assert_array_equal(saved['grades'], saved_grades)
                np.testing.assert_array_equal(saved['notifications'], saved_notices)
    G.write_csv(out/'event_ledger.csv', event_rows)
    G.write_csv(out/'notification_ledger.csv', notification_rows)
    decision = strong_signal(reports)
    assert C.sha(out/'sealed_calibration.json') == seal
    F.save_new(out/'metrics.json', dict(reports=reports, calibrations=calibrations,
        strong_signal=decision, contract=dict(main_weight=.25, sensitivity_weights=[0., .5],
            near_pass_definition='closest target surface lateral gap to co-directed corridor +/-0.30m <=0.10m',
            far_pass_definition='gap >0.10m, all notification grades cost 1',
            clear_definition='joint evaluator clear, all notification grades cost 1',
            cost='gap1 per query; simultaneous emitted grades joined by max; sum weight * joined notification count',
            contact='query-specific f3-13 timely, f14-15 late, no emission silent',
            independent_unit='physical scene; K and frames/seeds are correlated observations',
            uncertainty='descriptive scene counts retained; no iid replica inference or bootstrap required',
            fixed_seeds=list(G.SEEDS), main_seed=int(G.SEEDS[0]), training=0, protected_access=0)))
    outputs = ['sealed_calibration.json', 'calibration_curve.csv', 'cal_scores_readout.npz', 'metrics.json',
               'event_ledger.csv', 'notification_ledger.csv',
               'cal_grades_notifications.npz', 'hold_grades_notifications.npz']
    return dict(duration_seconds=time.monotonic()-began, cells=len(reports), events=len(event_rows),
        hgb_forward_seconds=hgb_seconds,
        joined_notifications=len(notification_rows), sealed_calibration_sha256=seal,
        source_sha256=C.sha(Path(__file__)), training=0, protected_access=0,
        hgb_forward='unchanged frozen F.score_split; CPU evaluation stage with threadpool limit 2',
        hold_first_readout='after immutable all-seed cal seal',
        outputs_sha256={name:C.sha(out/name) for name in outputs})
