"""Task-cost evaluation: fixed strong grades, new-cal negative-only light gates.

This module never renders, fits or runs a model. Cached causal scores enter an
exact all-tie sweep; hold curves are descriptive and cannot alter sealed gates.
Counts keep physical-scene identities and correlated K replicas together.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from numba import njit
from sklearn.metrics import roc_auc_score

import cnh_counterfactual_common_dev as C
import cnh_graded_evidence_dev as G
import cnh_frozen_e2e_metrics_20261010 as M
import cnh_cost_v2_metrics_20261010 as V

ARM_KEYS = ('E', 'E955', 'E956', 'E957', 'S955', 'S956', 'S957', 'Sensemble',
            'B955', 'B956', 'B957', 'Bensemble')
POINTS = {'main': 1., 'secondary': 1.25}
SEEDS = (955, 956, 957)
STAT_NAMES = ('weighted_cost_quarters', 'far_pass_plus_clear_notifications',
              'HEAD_timely', 'HEAD_rescue', 'HEAD_loss',
              'BODY_timely', 'BODY_rescue', 'BODY_loss')


def save_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_scores(out, split):
    with np.load(Path(out)/'data'/split/'task_scores.npz', allow_pickle=False) as saved:
        result = {k: saved[k].copy() for k in saved.files}
    keys = [str(s) for s in result['arm_keys']]
    scores = result['light_scores']
    if scores.shape[0] != len(keys) or scores.ndim != 5 or scores.shape[-2:] != (13, 2):
        raise ValueError('Aligned arm scores [A,N,K,13,2] required')
    if len(keys) != len(set(keys)) or any(k not in ARM_KEYS for k in keys):
        raise ValueError('Unknown or duplicate fixed arm key')
    old5 = result['old5']
    if old5.shape != scores.shape[1:] or not np.isin(old5, (0, 2)).all():
        raise ValueError('Old5 cache must contain original fixed strong 0/2 grades')
    if np.isposinf(scores).any():
        raise ValueError('Positive-infinite model evidence is invalid')
    labels_path = Path(out)/'data'/split/'labels.npz'
    if labels_path.exists():
        with np.load(labels_path, allow_pickle=False) as labels:
            for original, target in [('positive', 'labelspositive'), ('valid', 'labelsvalid'),
                                     ('negative_weight', 'negative_weight')]:
                result[target] = labels[original].copy()
    return result


def threshold_record(tau):
    return dict(tau=float(tau) if np.isfinite(tau) else None,
                tau_kind='finite' if np.isfinite(tau) else
                'negative_infinity' if tau < 0 else 'positive_infinity')


def record_threshold(record):
    return record['tau'] if record['tau_kind'] == 'finite' else \
        (-np.inf if record['tau_kind'] == 'negative_infinity' else np.inf)


def apply_gate(old5, score, tau):
    return np.where(old5 == 2, 2,
                    np.where(np.isfinite(score) & (score >= tau), 1, 0)).astype(np.int8)


@njit(cache=False)
def _clip_stats(grade, negative, near, contact, base_timely):
    stats = np.zeros(8, np.int64)
    peak = np.zeros(2, np.int8)
    quiet = np.zeros(2, np.int8)
    first = np.full(2, -1, np.int64)
    for f in range(13):
        joint = 0
        for q in range(2):
            g = grade[f, q]
            if g > 0:
                quiet[q] = 0
            else:
                quiet[q] += 1
                if quiet[q] > 1:
                    peak[q] = 0
            emit = g if g > 0 and g > peak[q] else 0
            if emit > 0 and first[q] < 0:
                first[q] = f
            if g > peak[q]:
                peak[q] = g
            if emit > joint:
                joint = emit
        if negative and joint > 0:
            stats[0] += 1 if near and joint == 1 else 4
            if not near:
                stats[1] += 1
    for q in range(2):
        if contact[q]:
            timely = first[q] >= 0 and first[q] < 11
            offset = 2+3*q
            stats[offset] = int(timely)
            stats[offset+1] = int(timely and not base_timely[q])
            stats[offset+2] = int(base_timely[q] and not timely)
    return stats


@njit(cache=False)
def _sweep(grade, clips, frames, queries, bounds, negative, near, contact, base_timely):
    n = grade.shape[0]
    local = np.zeros((n, 8), np.int64)
    total = np.zeros(8, np.int64)
    for c in range(n):
        local[c] = _clip_stats(grade[c], negative[c], near[c], contact[c], base_timely[c])
        total += local[c]
    answer = np.zeros((len(bounds)+1, 8), np.int64)
    answer[0] = total
    marked = np.full(n, -1, np.int64)
    for tie in range(len(bounds)-1):
        start, stop = bounds[tie], bounds[tie+1]
        for j in range(start, stop):
            grade[clips[j], frames[j], queries[j]] = 0
        for j in range(start, stop):
            c = clips[j]
            if marked[c] == tie:
                continue
            marked[c] = tie
            fresh = _clip_stats(grade[c], negative[c], near[c], contact[c], base_timely[c])
            total += fresh-local[c]
            local[c] = fresh
        answer[tie+1] = total
    answer[-1] = total
    return answer


def sweep(old5, score, category, rows, check, negatives_only):
    """Remove each complete tie, replay only affected clips, retain all costs."""
    check()
    masks, near, _, _ = V.negative_masks(category, rows)
    negative = masks['all_negative']
    if negatives_only:
        selected = negative
    else:
        selected = np.ones(len(rows), dtype=bool)
    strong = np.asarray(old5[selected], dtype=np.int8)
    evidence = np.asarray(score[selected], dtype=float)
    grade = apply_gate(strong, evidence, -np.inf).reshape(-1, 13, 2)
    eligible = np.isfinite(evidence) & (strong != 2)
    c, f, q = np.where(eligible.reshape(-1, 13, 2))
    values = evidence.reshape(-1, 13, 2)[c, f, q]
    order = np.argsort(values, kind='stable')
    values, c, f, q = [x[order] for x in (values, c, f, q)]
    bounds = np.r_[0, np.flatnonzero(np.diff(values))+1, len(values)] if len(values) else np.array([0])
    taus = np.r_[-np.inf, np.nextafter(values[bounds[:-1]], np.inf), np.inf] if len(values) else np.array([-np.inf, np.inf])
    replicas = old5.shape[1]
    neg_clip = np.repeat(negative[selected], replicas)
    near_clip = np.repeat(near[selected], replicas)
    contact_clip = np.repeat((category[selected] == 'contact'), replicas, axis=0)
    baseline = V.first_notice(M.replay_gap1(strong)).reshape(-1, 2)
    base_timely = (baseline >= 0) & (baseline < 11)
    stats = _sweep(grade, c, f, q, bounds, neg_clip, near_clip, contact_clip, base_timely)
    check()
    if len(taus) != len(stats):
        raise AssertionError('Every finite tie plus both endpoints required')
    np.testing.assert_array_equal(grade, strong.reshape(-1, 13, 2))
    return taus, stats, int(len(bounds)-1)


def matching(old5, score, category, rows, check):
    taus, stats, ties = sweep(old5, score, category, rows, check, True)
    cap = int(stats[-1, 0])
    records = {}
    for point, multiplier in POINTS.items():
        feasible = np.flatnonzero(stats[:, 0] <= cap*multiplier)
        chosen = int(feasible[0])
        records[point] = dict(**threshold_record(taus[chosen]),
            weighted_cap_quarters=cap*multiplier, weighted_cap=cap*multiplier/4,
            cal_weighted_cost=float(stats[chosen, 0]/4), candidate_index=chosen,
            candidate_thresholds=len(taus), negative_margin_ties=ties,
            contact_utility_access=False, cap_multiplier=multiplier)
    return records, taus, stats


def _write_curve(writer, arm, taus, stats, cap=None):
    for tau, counts in zip(taus, stats):
        row = dict(arm=arm, **threshold_record(tau),
                   **{k:int(v) for k, v in zip(STAT_NAMES, counts)},
                   weighted_cost=float(counts[0]/4),
                   HEAD_net=int(counts[3]-counts[4]), BODY_net=int(counts[6]-counts[7]))
        if cap is not None:
            row.update(feasible_main=bool(counts[0] <= cap),
                       feasible_secondary=bool(counts[0] <= 1.25*cap))
        writer.writerow(row)


def _curve_writer(handle, cal):
    fields = ['arm', 'tau', 'tau_kind', *STAT_NAMES, 'weighted_cost', 'HEAD_net', 'BODY_net']
    if cal:
        fields += ['feasible_main', 'feasible_secondary']
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return writer


def score_diagnostic(scores, category, rows):
    """Cal separability description only; labels never choose operating points."""
    from sklearn.metrics import roc_auc_score
    if not all(k in scores for k in ('labelspositive', 'labelsvalid', 'negative_weight')):
        return dict(status='NOT_EVALUABLE', reason='Task-label cache absent; no labels inferred from scores')
    positive = np.asarray(scores['labelspositive'], dtype=bool)
    valid = np.asarray(scores['labelsvalid'], dtype=bool)
    weights = np.asarray(scores['negative_weight'], dtype=float)
    weights = np.broadcast_to(weights, positive.shape)
    result = {}
    for ai, key in enumerate(scores['arm_keys']):
        score = scores.get('diagnostic_scores', scores['light_scores'])[ai]
        result[str(key)] = {}
        for q, height in enumerate(V.HEIGHTS):
            use = valid[..., q] & (scores['old5'][..., q] != 2) & np.isfinite(score[..., q])
            y = positive[..., q][use]
            x = score[..., q][use]
            w = np.where(y, 1., weights[..., q][use])
            entry = dict(positive_slots=int(y.sum()), negative_slots=int((~y).sum()),
                         auc=float(roc_auc_score(y, x)) if len(np.unique(y)) == 2 else None,
                         cost_weighted_auc=float(roc_auc_score(y, x, sample_weight=w)) if len(np.unique(y)) == 2 else None)
            grouped = {}
            for name, scene_mask in [('contact_positive', category[:, q] == 'contact'),
                                     ('near_pass', np.array([r['placement']=='pass' and r['lateral_gap_m']<=.10 for r in rows])),
                                     ('far_pass', np.array([r['placement']=='pass' and r['lateral_gap_m']>.10 for r in rows])),
                                     ('clear', np.array([r['placement']=='clear' for r in rows]))]:
                pick = valid[..., q] & (scores['old5'][..., q] != 2) & np.isfinite(score[..., q]) & scene_mask[:, None, None]
                if name == 'contact_positive':
                    pick &= positive[..., q]
                values = score[..., q][pick]
                grouped[name] = dict(slots=len(values), scene_denominator=int(scene_mask.sum()),
                    quantiles={str(p):float(np.quantile(values, p)) for p in (.05, .25, .5, .75, .95)} if len(values) else {})
            entry['class_score_overlap'] = grouped
            result[str(key)][height] = entry
    return dict(status='DESCRIPTIVE_CAL_ONLY', arms=result,
                score_source='Raw unmasked margins/logits' if 'diagnostic_scores' in scores else 'Gate-eligible finite scores',
                evaluation_slots='Valid task-label slots where original old5 is nonstrong',
                independent_unit='Physical scene; AUC is a correlated-slot ranking descriptor, not iid uncertainty')


def calibrate(out, rowscal, check):
    out = Path(out)
    began = time.monotonic()
    scores = load_scores(out, 'cal')
    records = {}
    with (out/'calibration_curve.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = _curve_writer(handle, True)
        for ai, key in enumerate(scores['arm_keys']):
            check()
            record, taus, stats = matching(scores['old5'], scores['light_scores'][ai],
                                          scores['category'], rowscal, check)
            records[str(key)] = record
            _write_curve(writer, str(key), taus, stats, record['main']['weighted_cap_quarters'])
    diagnostic = score_diagnostic(scores, scores['category'], rowscal)
    save_new(out/'cal_score_diagnostic.json', diagnostic)
    files = ['PLAN.json', 'data/cal/task_scores.npz', 'data/cal/labels.npz',
             'calibration_curve.csv', 'cal_score_diagnostic.json']
    save_new(out/'sealed_calibration.json', dict(calibrations=records,
        sealed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=C.sha(Path(__file__)), bindings={p:C.sha(out/p) for p in files},
        selection='Lowest canonical tau: -inf, nextafter(all finite eligible new-cal negative-scene score ties), +inf; exact joint gap1 cost; no monotonicity shortcut; no contact utility selection',
        points=POINTS, fixed_strong='Original old5 grade2 at every slot; additions light only',
        hold_access='Hold geometry/scores unopened; seal required before hold rendering'))
    return dict(duration_seconds=time.monotonic()-began,
                sealed_calibration_sha256=C.sha(out/'sealed_calibration.json'), arms=list(records))


def strong_signal(reports, arms):
    base = reports['hold/fixed/old5']
    base_cost = base['costs']['all_negative']['weighted_cost']
    base_far = base['costs']['far_pass_plus_clear']['notifications']
    decisions = {}
    for arm in arms:
        if arm in ('E955', 'E956', 'E957'):
            continue
        report = reports[f'hold/{arm}/main']
        family = arm[0]
        seed_arms = [family+str(s) for s in SEEDS]
        seed_nets = {a:reports[f'hold/{a}/main']['contacts']['HEAD_BODY']['net']
                     for a in seed_arms if f'hold/{a}/main' in reports}
        complete = len(seed_nets) == len(seed_arms)
        consistent = complete and all(v > 0 for v in seed_nets.values())
        net = report['contacts']['HEAD_BODY']['net']
        den = report['contacts']['HEAD_BODY']['denominator']
        cost = report['costs']['all_negative']['weighted_cost']
        far = report['costs']['far_pass_plus_clear']['notifications']
        signal = bool(den == 512 and net >= 10 and far <= 1.10*base_far and
                      cost <= 1.05*base_cost and consistent)
        decisions[arm] = dict(net=net, denominator=den, required_net=10,
            weighted_cost=cost, baseline_weighted_cost=base_cost, weighted_cost_cap=1.05*base_cost,
            far_pass_plus_clear_notifications=far, baseline_far_pass_plus_clear_notifications=base_far,
            far_pass_plus_clear_cap=1.10*base_far, seed_nets=seed_nets,
            all_seeds_completed=complete, seed_direction_consistent=consistent, strong_signal=signal)
    passing = [a for a in arms if a in decisions and decisions[a]['strong_signal']]
    # E wins the declared preference. Within training families use ensemble first,
    # then the predeclared first seed; this is report ordering, not hold fitting.
    order = ['E', 'Sensemble', 'S955', 'S956', 'S957', 'Bensemble', 'B955', 'B956', 'B957']
    winner = next((a for a in order if a in passing), None)
    return dict(arms=decisions, any_strong_signal=bool(passing), passing_arms=passing,
        preferred_candidate=winner, recommendation=f'建议 {winner} 升为默认候选' if winner else '保留原5格',
        interpretation='Predeclared simulator Development gate; hold descriptive curves do not select thresholds')


def evaluate(out, rows, check):
    out = Path(out)
    began = time.monotonic()
    binding = C.read(out/'sealed_calibration.json')
    seal = C.sha(out/'sealed_calibration.json')
    for name, digest in binding['bindings'].items():
        if C.sha(out/name) != digest:
            raise ValueError('Cal seal binding changed: '+name)
    reports, event_rows, notification_rows = {}, [], []
    for split in ('cal', 'hold'):
        scores = load_scores(out, split)
        category, old5 = scores['category'], scores['old5']
        base_first = V.first_notice(M.replay_gap1(old5))
        both_first = V.first_notice(M.replay_gap1(scores['both'][0]))
        masks = V.contact_masks(category, rows[split])
        cost_masks, near, layer, _ = V.negative_masks(category, rows[split])
        arms = [('fixed/m3', scores['m3']), ('fixed/old5', old5)]
        arms += [(f'both{s}', scores['both'][i]) for i, s in enumerate(SEEDS)]
        for ai, arm in enumerate(scores['arm_keys']):
            for point in POINTS:
                tau = record_threshold(binding['calibrations'][str(arm)][point])
                arms.append((f'{arm}/{point}', apply_gate(old5, scores['light_scores'][ai], tau)))
        keys, grades, notices = [], [], []
        for key, grade in arms:
            check()
            if key not in ('fixed/m3', 'fixed/old5', 'both955', 'both956', 'both957'):
                np.testing.assert_array_equal(grade == 2, old5 == 2)
            notice = M.replay_gap1(grade)
            first = V.first_notice(notice)
            reports[f'{split}/{key}'] = dict(contacts=V.contact_summary(first, base_first, masks),
                contacts_vs_both955=V.contact_summary(first, both_first, masks),
                costs=V.cost_summary(notice, category, rows[split]))
            keys.append(key); grades.append(grade); notices.append(notice)
            for n, row in enumerate(rows[split]):
                for k in range(grade.shape[1]):
                    metadata = V._metadata(row, split, n, k)
                    for q, height in enumerate(V.HEIGHTS):
                        index, base, both = int(first[n,k,q]), int(base_first[n,k,q]), int(both_first[n,k,q])
                        timely, basetime, bothtime = 0 <= index < 11, 0 <= base < 11, 0 <= both < 11
                        contact = category[n,q] == 'contact'
                        event_rows.append(dict(**metadata, arm=key, height=height, category=str(category[n,q]),
                            first_index=index, first_nominal_frame=index+3 if index >= 0 else -1,
                            outcome='timely' if timely else 'late' if index >= 11 else 'silent',
                            old5_first_index=base, both955_first_index=both,
                            rescue=int(contact and timely and not basetime), loss=int(contact and basetime and not timely),
                            rescue_vs_both955=int(contact and timely and not bothtime),
                            loss_vs_both955=int(contact and bothtime and not timely),
                            light_notifications=int((notice[n,k,:,q] == 1).sum()),
                            strong_notifications=int((notice[n,k,:,q] == 2).sum())))
                    joint = notice[n,k].max(-1)
                    for f in np.flatnonzero(joint):
                        union = int(joint[f]); negative = bool(cost_masks['all_negative'][n])
                        weight = lambda w: (w if near[n] and union == 1 else 1.) if negative else 0.
                        notification_rows.append(dict(**metadata, arm=key, frame_index=int(f), nominal_frame=int(f)+3,
                            union_grade=union, HEAD_grade=int(notice[n,k,f,0]), BODY_grade=int(notice[n,k,f,1]),
                            HEAD_truth=str(category[n,0]), BODY_truth=str(category[n,1]), negative_cost=int(negative),
                            partition='near_pass' if near[n] else 'far_pass' if cost_masks['far_pass'][n] else
                                      'clear' if cost_masks['clear'][n] else 'contact',
                            evaluated_pass_layer=str(layer[n]), weight_w0=weight(0.), weight_w025=weight(.25), weight_w05=weight(.5)))
        np.savez_compressed(out/f'{split}_grades_notifications.npz', keys=np.asarray(keys), grades=np.asarray(grades),
                            notifications=np.asarray(notices), category=category, scene_ids=np.arange(len(rows[split])),
                            rawlight_scores=scores['light_scores'], arm_keys=scores['arm_keys'])
        if split == 'hold':
            with (out/'hold_cost_benefit_curve.csv').open('x', encoding='utf-8', newline='') as handle:
                writer = _curve_writer(handle, False)
                for ai, arm in enumerate(scores['arm_keys']):
                    taus, stats, _ = sweep(old5, scores['light_scores'][ai], category, rows[split], check, False)
                    _write_curve(writer, str(arm), taus, stats)
    G.write_csv(out/'event_ledger.csv', event_rows)
    G.write_csv(out/'notification_ledger.csv', notification_rows)
    arms = [str(k) for k in scores['arm_keys']]
    decision = strong_signal(reports, arms)
    save_new(out/'metrics.json', dict(reports=reports, calibrations=binding['calibrations'], strong_signal=decision,
        cal_score_diagnostic=C.read(out/'cal_score_diagnostic.json'),
        contract=dict(main_weight=.25, sensitivity_weights=[0., .5], points=POINTS,
            cost='Per-query gap1; max same-frame query emissions; near light .25, all other negative notices 1',
            fixed_strong='Every new arm exactly preserves original old5 grade2 slots; additions light only',
            contact='f3..13 timely, f14..15 late, no emissions silent',
            independent_unit='Physical scene; K/frame/seed correlated, retained scene counts',
            curves='All hold score ties and +/-inf; descriptive only, no threshold selection',
            protected_access=0)))
    if C.sha(out/'sealed_calibration.json') != seal:
        raise AssertionError('Cal seal mutated')
    outputs = ['sealed_calibration.json','calibration_curve.csv','cal_score_diagnostic.json','metrics.json',
               'hold_cost_benefit_curve.csv','event_ledger.csv','notification_ledger.csv',
               'cal_grades_notifications.npz','hold_grades_notifications.npz']
    return dict(duration_seconds=time.monotonic()-began, cells=len(reports), events=len(event_rows),
        joined_notifications=len(notification_rows), sealed_calibration_sha256=seal,
        source_sha256=C.sha(Path(__file__)), protected_access=0,
        outputs_sha256={name:C.sha(out/name) for name in outputs})
