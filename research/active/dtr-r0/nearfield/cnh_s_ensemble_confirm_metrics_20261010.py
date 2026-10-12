"""Frozen S confirmation: new-cal 1.25 cost gates, exact transfer and scene ledgers.

No fitting or model forward occurs here. The inherited task-cost sweep,
gap1 notifier and v2 summaries are imported unchanged. Hold has no chooser.
"""
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import numpy as np

import cnh_counterfactual_common_dev as C
import cnh_graded_evidence_dev as G
import cnh_frozen_e2e_metrics_20261010 as M
import cnh_cost_v2_metrics_20261010 as V
import cnh_task_cost_metrics_20261010 as T

ARM_KEYS = ('E', 'S955', 'S956', 'S957', 'Sensemble')
CAL_MULTIPLIER = 1.25
TRANSFER_TAU = 2.4142577648162846


def load_scores(out, split):
    scores = T.load_scores(out, split)
    if tuple(str(k) for k in scores['arm_keys']) != ARM_KEYS:
        raise ValueError('Required fixed arms/order: '+str(ARM_KEYS))
    if np.isnan(scores['light_scores']).any():
        raise ValueError('NaN evidence must not silently become a negative decision')
    return scores


def matching(old5, score, category, rows, check):
    taus, stats, ties = T.sweep(old5, score, category, rows, check, True)
    base_quarters = int(stats[-1, 0])
    cap_quarters = base_quarters*CAL_MULTIPLIER
    candidates = np.flatnonzero(stats[:, 0] <= cap_quarters)
    if not len(candidates):
        raise AssertionError('Fixed strong endpoint must be feasible')
    chosen = int(candidates[0])
    record = dict(**T.threshold_record(taus[chosen]),
        weighted_cap_quarters=cap_quarters, weighted_cap=cap_quarters/4,
        baseline_cal_weighted_cost=base_quarters/4,
        cal_weighted_cost=float(stats[chosen, 0]/4), candidate_index=chosen,
        candidate_thresholds=len(taus), negative_margin_ties=ties,
        contact_utility_access=False, cap_multiplier=CAL_MULTIPLIER)
    return record, taus, stats


def calibrate(out, rowscal, check, transfer_seal=None):
    out = Path(out)
    began = time.monotonic()
    check()
    if (out/'data/hold/task_scores.npz').exists():
        raise ValueError('Hold scores already exist before calibration seal')
    scores = load_scores(out, 'cal')
    transfer_seal = Path(transfer_seal) if transfer_seal else \
        out.parent/'cnh-task-cost-retrain-dev-20261010'/'sealed_calibration.json'
    old_binding = C.read(transfer_seal)
    transfer_record = old_binding['calibrations']['Sensemble']['secondary']
    if T.record_threshold(transfer_record) != TRANSFER_TAU:
        raise ValueError('Exact sealed secondary Sensemble transfer tau changed')
    records = {}
    fields = ['arm', 'tau', 'tau_kind', *T.STAT_NAMES, 'weighted_cost',
              'feasible', 'weighted_cap_quarters']
    with (out/'calibration_curve.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ai, arm in enumerate(ARM_KEYS):
            check()
            record, taus, stats = matching(scores['old5'], scores['light_scores'][ai],
                scores['category'], rowscal, check)
            records[arm] = {'main': record}
            for tau, counts in zip(taus, stats):
                writer.writerow(dict(arm=arm, **T.threshold_record(tau),
                    **{k:int(v) for k, v in zip(T.STAT_NAMES, counts)},
                    weighted_cost=float(counts[0]/4),
                    feasible=bool(counts[0] <= record['weighted_cap_quarters']),
                    weighted_cap_quarters=record['weighted_cap_quarters']))
    diagnostic = T.score_diagnostic(scores, scores['category'], rowscal)
    T.save_new(out/'cal_score_diagnostic.json', diagnostic)
    files = ['PLAN.json', 'data/cal/task_scores.npz', 'calibration_curve.csv',
             'cal_score_diagnostic.json']
    for optional in ('data/cal/labels.npz', 'execution_manifest.json',
                     'frozen_models_manifest.json'):
        if (out/optional).exists():
            files.append(optional)
    T.save_new(out/'sealed_calibration.json', dict(calibrations=records,
        transfer=dict(**T.threshold_record(TRANSFER_TAU), arm='Sensemble',
            provenance='Prior retrain sealed secondary point, exact float unchanged',
            source_path=str(transfer_seal.resolve()), source_sha256=C.sha(transfer_seal)),
        sealed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=C.sha(Path(__file__)), inherited_metrics_sha256=C.sha(Path(T.__file__)),
        bindings={p:C.sha(out/p) for p in files},
        selection='Lowest canonical tau: -inf, nextafter(every finite eligible new-cal negative score tie), +inf; exact joint gap1 cost; nonmonotone enumeration; contact utility never selects',
        main_cap_multiplier=CAL_MULTIPLIER,
        fixed_strong='Original old5 grade2 every slot; light additions only',
        hold_access='Seal required before any hold rendering or frozen forward'))
    return dict(duration_seconds=time.monotonic()-began,
                sealed_calibration_sha256=C.sha(out/'sealed_calibration.json'), arms=list(records))


def strong_signal(reports):
    base = reports['hold/fixed/old5']
    candidate = reports['hold/Sensemble/main']
    contact = candidate['contacts']
    den = contact['HEAD_BODY']['denominator']
    total_required = math.ceil(den*40/512)
    height_required = {h:math.ceil(contact[h]['denominator']*.02) for h in V.HEIGHTS}
    seed_required = math.ceil(total_required/2)
    seed_nets = {arm:reports['hold/'+arm+'/main']['contacts']['HEAD_BODY']['net']
                 for arm in ('S955', 'S956', 'S957')}
    net = contact['HEAD_BODY']['net']
    height_nets = {h:contact[h]['net'] for h in V.HEIGHTS}
    base_far = base['costs']['far_pass_plus_clear']['notifications']
    far = candidate['costs']['far_pass_plus_clear']['notifications']
    base_cost = base['costs']['all_negative']['weighted_cost']
    cost = candidate['costs']['all_negative']['weighted_cost']
    a = bool(net >= total_required and all(height_nets[h] >= height_required[h] for h in V.HEIGHTS))
    b = bool(far <= 1.10*base_far)
    c = bool(cost <= CAL_MULTIPLIER*1.05*base_cost)
    d = bool(all(v >= seed_required for v in seed_nets.values()))
    gates = dict(a=dict(pass_=a, net=net, denominator=den, required_net=total_required,
                       height_nets=height_nets, height_required=height_required,
                       height_denominators={h:contact[h]['denominator'] for h in V.HEIGHTS}),
        b=dict(pass_=b, far_pass_plus_clear_notifications=far, baseline_notifications=base_far,
               cap=1.10*base_far),
        c=dict(pass_=c, weighted_cost=cost, baseline_weighted_cost=base_cost,
               cap=CAL_MULTIPLIER*1.05*base_cost, multiplier=CAL_MULTIPLIER*1.05),
        d=dict(pass_=d, seed_nets=seed_nets, required_each_seed_net=seed_required))
    for gate in gates.values():
        gate['pass'] = gate.pop('pass_')
    passed = a and b and c and d
    return dict(gates=gates, failed_gates=[k for k, g in gates.items() if not g['pass']],
        all_pass=bool(passed), any_strong_signal=bool(passed),
        recommendation='S 集成 @1.25× 建议升为 ToF 默认候选' if passed else '保留原5格',
        interpretation='Frozen predeclared fresh synthetic confirmation; no hold threshold or arm selection')


def evaluate(out, rows, check):
    out = Path(out)
    began = time.monotonic()
    check()
    binding = C.read(out/'sealed_calibration.json')
    seal_hash = C.sha(out/'sealed_calibration.json')
    for name, digest in binding['bindings'].items():
        if C.sha(out/name) != digest:
            raise ValueError('Cal seal binding changed: '+name)
    transfer = binding['transfer']
    if C.sha(transfer['source_path']) != transfer['source_sha256']:
        raise ValueError('Prior frozen transfer seal changed')
    reports, event_rows, notification_rows = {}, [], []
    for split in ('cal', 'hold'):
        scores = load_scores(out, split)
        category, old5 = scores['category'], scores['old5']
        base_first = V.first_notice(M.replay_gap1(old5))
        both = scores['both'][0] if scores['both'].ndim == 5 else scores['both']
        both_first = V.first_notice(M.replay_gap1(both))
        masks = V.contact_masks(category, rows[split])
        cost_masks, near, layer, _ = V.negative_masks(category, rows[split])
        arms = [('fixed/m3', scores['m3']), ('fixed/old5', old5), ('both955', both)]
        for ai, arm in enumerate(ARM_KEYS):
            tau = T.record_threshold(binding['calibrations'][arm]['main'])
            arms.append((arm+'/main', T.apply_gate(old5, scores['light_scores'][ai], tau)))
        arms.append(('Sensemble/transfer', T.apply_gate(old5, scores['light_scores'][4],
                                                       T.record_threshold(transfer))))
        keys, grades, notices = [], [], []
        for key, grade in arms:
            check()
            if key not in ('fixed/m3', 'fixed/old5', 'both955'):
                np.testing.assert_array_equal(grade == 2, old5 == 2)
            notice = M.replay_gap1(grade)
            first = V.first_notice(notice)
            reports[split+'/'+key] = dict(contacts=V.contact_summary(first, base_first, masks),
                contacts_vs_both955=V.contact_summary(first, both_first, masks),
                costs=V.cost_summary(notice, category, rows[split]))
            keys.append(key); grades.append(grade); notices.append(notice)
            for n, row in enumerate(rows[split]):
                for k in range(grade.shape[1]):
                    metadata = V._metadata(row, split, n, k)
                    for q, height in enumerate(V.HEIGHTS):
                        index, base, oldboth = int(first[n,k,q]), int(base_first[n,k,q]), int(both_first[n,k,q])
                        timely, basetime, bothtime = 0 <= index < 11, 0 <= base < 11, 0 <= oldboth < 11
                        contact = category[n,q] == 'contact'
                        event_rows.append(dict(**metadata, arm=key, height=height, category=str(category[n,q]),
                            first_index=index, first_nominal_frame=index+3 if index >= 0 else -1,
                            outcome='timely' if timely else 'late' if index >= 11 else 'silent',
                            old5_first_index=base, both955_first_index=oldboth,
                            rescue=int(contact and timely and not basetime), loss=int(contact and basetime and not timely),
                            rescue_vs_both955=int(contact and timely and not bothtime),
                            loss_vs_both955=int(contact and bothtime and not timely),
                            light_notifications=int((notice[n,k,:,q] == 1).sum()),
                            strong_notifications=int((notice[n,k,:,q] == 2).sum())))
                    joint = notice[n,k].max(-1)
                    for f in np.flatnonzero(joint):
                        union = int(joint[f]); negative = bool(cost_masks['all_negative'][n])
                        def weight(w):
                            return (w if near[n] and union == 1 else 1.) if negative else 0.
                        notification_rows.append(dict(**metadata, arm=key, frame_index=int(f), nominal_frame=int(f)+3,
                            union_grade=union, HEAD_grade=int(notice[n,k,f,0]), BODY_grade=int(notice[n,k,f,1]),
                            HEAD_truth=str(category[n,0]), BODY_truth=str(category[n,1]), negative_cost=int(negative),
                            partition='near_pass' if near[n] else 'far_pass' if cost_masks['far_pass'][n] else
                                      'clear' if cost_masks['clear'][n] else 'contact',
                            evaluated_pass_layer=str(layer[n]), weight_w0=weight(0.),
                            weight_w025=weight(.25), weight_w05=weight(.5)))
        np.savez_compressed(out/(split+'_grades_notifications.npz'), keys=np.asarray(keys),
            grades=np.asarray(grades), notifications=np.asarray(notices), category=category,
            scene_ids=np.arange(len(rows[split])), rawlight_scores=scores['light_scores'],
            arm_keys=scores['arm_keys'])
    if len(rows['hold']) != 1536 or reports['hold/fixed/old5']['contacts']['HEAD_BODY']['denominator'] != 1024:
        raise AssertionError('Declared full hold/contact denominator not completed')
    G.write_csv(out/'event_ledger.csv', event_rows)
    G.write_csv(out/'notification_ledger.csv', notification_rows)
    decision = strong_signal(reports)
    T.save_new(out/'metrics.json', dict(reports=reports, calibrations=binding['calibrations'],
        transfer=binding['transfer'], strong_signal=decision,
        cal_score_diagnostic=C.read(out/'cal_score_diagnostic.json'),
        contract=dict(main_weight=.25, sensitivity_weights=[0., .5],
            main_cal_cap_multiplier=CAL_MULTIPLIER, transfer_tau=TRANSFER_TAU,
            cost='Per-query gap1; max same-frame query emissions; near light .25, all other negative notices 1',
            fixed_strong='Original old5 grade2 every slot; light additions only',
            contact='f3..13 timely, f14..15 late, no emissions silent',
            independent_unit='Physical scene; K/frame/seed correlated, per-scene counts retained',
            hold_selection=False, hold_curve='Not requested; no hold tie sweep performed', protected_access=0)))
    if C.sha(out/'sealed_calibration.json') != seal_hash:
        raise AssertionError('Cal seal mutated')
    outputs = ['sealed_calibration.json', 'calibration_curve.csv', 'cal_score_diagnostic.json',
               'metrics.json', 'event_ledger.csv', 'notification_ledger.csv',
               'cal_grades_notifications.npz', 'hold_grades_notifications.npz']
    return dict(duration_seconds=time.monotonic()-began, cells=len(reports), events=len(event_rows),
        joined_notifications=len(notification_rows), sealed_calibration_sha256=seal_hash,
        source_sha256=C.sha(Path(__file__)), protected_access=0,
        outputs_sha256={name:C.sha(out/name) for name in outputs})
