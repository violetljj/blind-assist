"""Independent CPU reconstruction and fixed score-stratified DEV description.

No production grade, summary, calibration, or smoothing functions are imported.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
RUN = ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010'
OUT = RUN/'audit/recheck_v2'
SEEDS = (2026100955, 2026100956, 2026100957)
POLICIES = ('dual', 'dual_max3', 'dual_k2of3')
M3, RAISED, LOCAL = .8557642486787612, .9404184587540165, 4.625390338985158


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth(x):
    x = np.asarray(x, np.float64)
    answer = np.zeros_like(x)
    for t in range(x.shape[-2]):
        indices = list(range(max(0, t-4), t+1))
        weights = [2**i for i in range(len(indices))]
        for ix, weight in zip(indices, weights):
            answer[..., t, :] += x[..., ix, :]*weight
        answer[..., t, :] /= sum(weights)
    return answer


def temporal(x, kind):
    answer = np.full(x.shape, -np.inf, np.float64)
    for t in range(x.shape[-2]):
        ordered = np.sort(x[..., max(0, t-2):t+1, :].astype(np.float64), axis=-2)
        if kind == 'max3':
            answer[..., t, :] = ordered[..., -1, :]
        elif ordered.shape[-2] >= 2:
            answer[..., t, :] = ordered[..., -2, :]
    return answer


def trend(x):
    slope, residual = np.zeros(x.shape, np.float64), np.zeros(x.shape, np.float64)
    for t in range(x.shape[-2]):
        y = x[..., max(0, t-4):t+1, :].astype(np.float64)
        if y.shape[-2] > 1:
            design = np.column_stack((np.ones(y.shape[-2]), np.arange(y.shape[-2])))
            flat = np.moveaxis(y, -2, 0).reshape(y.shape[-2], -1)
            coefficients = np.linalg.lstsq(design, flat, rcond=None)[0]
            fit = design@coefficients
            slope[..., t, :] = coefficients[1].reshape(x.shape[:-2]+(x.shape[-1],))
            residual[..., t, :] = np.sqrt(np.mean((flat-fit)**2, axis=0)).reshape(x.shape[:-2]+(x.shape[-1],))
    return slope, residual


def grade(m3, local, ordinary, th, raw, policy):
    strong = (m3 >= RAISED) | ((local >= LOCAL) & (ordinary >= th['single'])) | (ordinary >= th['addition'])
    total = (m3 >= RAISED) | (local >= LOCAL) | (ordinary >= th['single']) | strong
    if policy != 'dual':
        kind = policy.removeprefix('dual_')
        total |= temporal(raw, kind) >= th['temporal'][kind]['theta']
    answer = total.astype(np.int8)
    answer[strong] = 2
    return answer


def first(flags, end=13):
    answer = np.full(flags.shape[:2]+(2,), -1, np.int64)
    for t in range(end):
        take = (answer == -1) & flags[:, :, t]
        answer[take] = t+3
    return answer


def cost(flags, mask):
    traces = flags[mask].reshape(-1, 13)
    longest, segments = 0, 0
    for trace in traces:
        previous, run = False, 0
        for active in trace:
            if active and not previous:
                segments += 1
            run = run+1 if active else 0
            longest = max(longest, run)
            previous = active
    return dict(slots=int(traces.sum()), slot_denominator=traces.size,
        clips=int(traces.any(1).sum()), clip_denominator=len(traces),
        segments=segments, longest_run_frames=longest)


def summary(flags, category):
    contact = category == 'contact'
    timely = (first(flags, 11) >= 0) & contact[:, None, :]
    late = (first(flags[:, :, 11:], 2) >= 0) & contact[:, None, :] & ~timely
    clear = (category == 'clear').all(1)
    passed = (category == 'pass').any(1) & ~contact.any(1)
    joint = flags.any(3)
    cc = cost(joint, clear)
    return dict(counts=timely.sum((0, 1)).tolist(), late_counts=late.sum((0, 1)).tolist(),
        clear_slots=cc['slots'], clear_denominator=cc['slot_denominator'], clear_segments=cc['segments'],
        clear_clips=cc['clips'], physical_contact_any_height=int(joint[contact.any(1), :, :11].any(2).sum()),
        pass_clips=int(joint[passed].any(2).sum()), contact_event_denominators=(contact.sum(0)*flags.shape[1]).tolist(),
        physical_contact_denominator=int(contact.any(1).sum()*flags.shape[1]),
        clear_clip_denominator=cc['clip_denominator'], pass_clip_denominator=int(passed.sum()*flags.shape[1]))


def paired(before, after, category):
    a, b = first(before, 11), first(after, 11)
    result = []
    for q, height in enumerate(('HEAD', 'BODY')):
        aa, bb = a[category[:, q] == 'contact', :, q].ravel(), b[category[:, q] == 'contact', :, q].ravel()
        both = (aa >= 0) & (bb >= 0)
        delta = aa[both]-bb[both]
        unique, count = np.unique(delta, return_counts=True)
        result.append(dict(height=height, denominator=len(aa), rescue=int(((aa < 0) & (bb >= 0)).sum()),
            loss=int(((aa >= 0) & (bb < 0)).sum()), both_timely=int(both.sum()), earlier=int((delta > 0).sum()),
            same=int((delta == 0).sum()), later=int((delta < 0).sum()),
            median_advance_frames=float(np.median(delta)) if len(delta) else None,
            advance_frame_histogram={str(int(k)): int(v) for k, v in zip(unique, count)}))
    return result


def nearest(values, target):
    values = values.ravel()
    options = np.r_[np.unique(values[np.isfinite(values)]), np.inf]
    counts = np.asarray([np.count_nonzero(values >= v) for v in options])
    distance = abs(counts-target)
    index = np.flatnonzero(distance == distance.min())[-1]
    return float(options[index]), int(counts[index])


def csv_write(path, rows):
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run():
    began = time.monotonic()
    if OUT.exists():
        raise FileExistsError('Preserve previous independent audit')
    OUT.mkdir(parents=True)
    save(OUT/'PLAN.json', dict(task='INDEPENDENT_CNH_GRADED_EVIDENCE_AUDIT', budget_CPU_seconds=600,
        scope='All 2 splits, 3 seeds, 3 policies; numerical reconstruction, causal prefixes, fixed score strata',
        production_functions_used=[], source_sha256=sha(Path(__file__)),
        first_attempt= 'audit/receipt.json preserved; its only failures were FP32 mean vs FP64 OLS residual rounding',
        residual_absolute_tolerance=3e-7, rationale='Independent OLS residual vs production FP32 raw mean; integer grid classifications still checked exactly',
        margin_score_bins=['-infinity', -1, 0, 1, 'infinity'], membership='left-inclusive, right-exclusive',
        matching='f13 smoothed ordinary minus frozen single theta, shared HEAD/BODY bins; within-score-stratum descriptive comparison, not exact matching',
        trend='OLS raw past5 f9..13, slope logit per nominal frame; residual RMS; frozen cal-all-f13 median slope and residual boundaries',
        categories='Evaluator-only category; no rule selection, threshold tuning, inferencing or deployment claim'))
    failures, checks, distribution, rounding = [], 0, [], []
    def compare(name, actual, expected):
        nonlocal checks
        checks += 1
        if actual != expected:
            failures.append(dict(check=name, actual=actual, expected=expected))
    try:
        metrics, thresholds = read(RUN/'metrics.json'), read(RUN/'thresholds.json')
        scene_rows = read(SOURCE/'scene_rows.json')
        preserve_metrics = read(RUN/'strong_preserve/metrics.json')
        summary_index = {}
        for subpath in ('', 'strong_preserve'):
            with (RUN/subpath/'summary.csv').open(encoding='utf8', newline='') as stream:
                for row in csv.DictReader(stream):
                    summary_index[subpath, row['split'], int(row['seed']), row.get('policy', 'preserve')] = row
        def check_summary(label, grade_value, category_value, saved):
            any_metrics, strong_metrics = summary(grade_value > 0, category_value), summary(grade_value == 2, category_value)
            cc, pp = (category_value == 'clear').all(1), (category_value == 'pass').any(1) & ~(category_value == 'contact').any(1)
            joint_value = grade_value.max(3)
            strong_cost = cost(joint_value == 2, cc)
            light_cost = cost(joint_value == 1, cc)
            light_contact = ((grade_value[:, :, :11] == 1).any(2) & ~(grade_value[:, :, :11] == 2).any(2)
                & (category_value == 'contact')[:, None, :]).sum((0, 1))
            expected = dict(HEAD_any=any_metrics['counts'][0], BODY_any=any_metrics['counts'][1],
                HEAD_strong=strong_metrics['counts'][0], BODY_strong=strong_metrics['counts'][1],
                HEAD_light_only=int(light_contact[0]), BODY_light_only=int(light_contact[1]),
                clear_any_slots=any_metrics['clear_slots'], clear_any_clips=any_metrics['clear_clips'],
                clear_strong_slots=strong_cost['slots'], clear_light_slots=light_cost['slots'],
                pass_any_clips=any_metrics['pass_clips'], pass_strong_clips=cost(joint_value == 2, pp)['clips'],
                pass_light_clips=cost(joint_value == 1, pp)['clips'])
            compare(label+'/summary_csv', {k: int(saved[k]) for k in expected}, expected)
        old = read(SOURCE/'evaluation/thresholds.json')['arms']['ordinary']
        with (RUN/'event_ledger.csv').open(encoding='utf8', newline='') as stream:
            events = list(csv.DictReader(stream))
        event_index = {(r['split'], int(r['seed']), r['policy'], int(r['scene']), int(r['replica']), r['height']): r for r in events}
        with (RUN/'deadline_grid.csv').open(encoding='utf8', newline='') as stream:
            grids = list(csv.DictReader(stream))
        group_boundaries = {}
        for split in ('cal', 'validation'):
            with np.load(SOURCE/f'data/{split}/physics.npz') as payload:
                category, ids = payload['category'], payload['scene_ids']
            with np.load(SOURCE/f'baselines/{split}_ideal.npz') as payload:
                m3, local = smooth(payload['m3_raw']), smooth(payload['local_raw'])
            with np.load(RUN/f'{split}_grades.npz') as payload:
                stored_grades, stored_slope, stored_residual = payload['grades'], payload['slope'], payload['residual']
                compare(split+'/cached_category', payload['category'].tolist(), category.tolist())
                compare(split+'/cached_ids', payload['scene_ids'].tolist(), ids.tolist())
            with np.load(RUN/f'strong_preserve/{split}_grades.npz') as payload:
                preserved = payload['grades']
            clear, passed = (category == 'clear').all(1), (category == 'pass').any(1) & ~(category == 'contact').any(1)
            for si, seed in enumerate(SEEDS):
                th = thresholds[str(seed)]
                with np.load(SOURCE/f'scores/ordinary_seed{seed}_{split}.npz') as payload:
                    raw = payload['raw'].astype(np.float64)
                ordinary, (slope, residual) = smooth(raw), trend(raw)
                compare(f'{split}/{seed}/slope', bool(np.allclose(stored_slope[si], slope, atol=2e-14, rtol=2e-14)), True)
                max_residual_error = float(abs(stored_residual[si]-residual).max())
                rounding.append(dict(split=split, seed=seed, maximum_residual_difference=max_residual_error))
                compare(f'{split}/{seed}/residual', bool(np.allclose(stored_residual[si], residual, atol=3e-7, rtol=0)), True)
                if split == 'cal':
                    boundaries = dict(slope=float(np.median(slope[..., 10, :])), residual=float(np.median(residual[..., 10, :])))
                    group_boundaries[str(seed)] = boundaries
                    compare(f'{seed}/residual_cal_boundary', bool(np.isclose(th['residual_cutoff'], boundaries['residual'], atol=3e-7, rtol=0)), True)
                    compare(f'{seed}/single_threshold', th['single'], old[str(seed)]['standalone']['value'])
                    compare(f'{seed}/addition_threshold', th['addition'], old[str(seed)]['old_fusion_plus_candidate']['value'])
                    for kind in ('max3', 'k2of3'):
                        theta, slots = nearest(temporal(raw, kind)[clear].max(3), old[str(seed)]['standalone_actual_clear_slots'])
                        compare(f'{seed}/{kind}/theta', theta, th['temporal'][kind]['theta'])
                        compare(f'{seed}/{kind}/slots', slots, th['temporal'][kind]['actual_slots'])
                references = dict(M3=m3 >= M3, old_fusion=(m3 >= RAISED) | (local >= LOCAL),
                    ordinary_single=ordinary >= th['single'], ordinary_OR=(m3 >= RAISED) | (local >= LOCAL) | (ordinary >= th['addition']))
                dual = grade(m3, local, ordinary, th, raw, 'dual')
                successor = (references['ordinary_OR'] | references['ordinary_single']).astype(np.int8)
                successor[references['ordinary_OR']] = 2
                successor_label = f'{split}/{seed}/strong_preserve'
                compare(successor_label+'/all_grades', bool(np.array_equal(successor, preserved[si])), True)
                compare(successor_label+'/strong_OR_identity', bool(np.array_equal(successor == 2, references['ordinary_OR'])), True)
                compare(successor_label+'/any_dual_identity', bool(np.array_equal(successor > 0, dual > 0)), True)
                successor_report = preserve_metrics[split][str(seed)]
                for level, flags in (('any', successor > 0), ('strong', successor == 2), ('light', successor == 1)):
                    compare(successor_label+'/'+level, summary(flags, category), successor_report[level])
                for field, flags in (('paired_any', successor > 0), ('paired_strong', successor == 2)):
                    for name, reference in references.items():
                        compare(successor_label+'/'+field+'/'+name, paired(reference, flags, category), successor_report[field][name])
                compare(successor_label+'/paired_strong_vs_initial_dual', paired(dual == 2, successor == 2, category), successor_report['paired_strong_vs_initial_dual'])
                for kind, mask in (('clear', clear), ('pass', passed)):
                    for level, number in (('light', 1), ('strong', 2)):
                        compare(successor_label+'/joint/'+kind+'/'+level, cost(successor.max(3) == number, mask), successor_report['joint_costs'][kind][level])
                check_summary(successor_label, successor, category, summary_index['strong_preserve', split, seed, 'preserve'])
                for pi, policy in enumerate(POLICIES):
                    reconstructed = grade(m3, local, ordinary, th, raw, policy)
                    label = f'{split}/{seed}/{policy}'
                    compare(label+'/all_grades', bool(np.array_equal(stored_grades[si, pi], reconstructed)), True)
                    report = metrics[split][str(seed)][policy]
                    check_summary(label, reconstructed, category, summary_index['', split, seed, policy])
                    for level, flags in (('any', reconstructed > 0), ('strong', reconstructed == 2), ('light', reconstructed == 1)):
                        compare(label+'/'+level+'/metrics', summary(flags, category), report[level])
                    for field, flags in (('paired_any', reconstructed > 0), ('paired_strong', reconstructed == 2)):
                        for name, reference in references.items():
                            compare(label+'/'+field+'/'+name, paired(reference, flags, category), report[field][name])
                    compare(label+'/vs_dual', paired(dual > 0, reconstructed > 0, category), report['paired_any_vs_dual'])
                    joint = reconstructed.max(3)
                    for kind, mask in (('clear', clear), ('pass', passed)):
                        for level, value in (('light', 1), ('strong', 2)):
                            compare(label+'/joint/'+kind+'/'+level, cost(joint == value, mask), report['joint_costs'][kind][level])
                    s, l = (reconstructed[:, :, :11] == 2).any(2), (reconstructed[:, :, :11] == 1).any(2)
                    contact = (category == 'contact')[:, None, :]
                    compare(label+'/light_only', (l & ~s & contact).sum((0, 1)).tolist(), report['light_only_timely_contact'])
                    compare(label+'/old_to_light_only', (references['old_fusion'][:, :, :11].any(2) & l & ~s & contact).sum((0, 1)).tolist(), report['contact_old_timely_to_light_only'])
                    for name, field in (('old_fusion', 'old_alert_query_slots_to_light'), ('ordinary_OR', 'ordinary_OR_query_slots_to_light')):
                        compare(label+'/'+field, int((references[name] & (reconstructed == 1)).sum()), report[field])
                    firsts = dict(first_any_frame=first(reconstructed > 0), first_strong_frame=first(reconstructed == 2),
                        old_first_frame=first(references['old_fusion']), ordinary_OR_first_frame=first(references['ordinary_OR']))
                    for i, scene in enumerate(ids):
                        for k in range(4):
                            for q, height in enumerate(('HEAD', 'BODY')):
                                row = event_index[(split, seed, policy, int(scene), k, height)]
                                compare(label+f'/event/{scene}/{k}/{height}',
                                    [int(row[field]) for field in firsts], [int(value[i, k, q]) for value in firsts.values()])
                    for t in range(13):
                        changed_raw, changed_m3, changed_local = raw.copy(), m3.copy(), local.copy()
                        changed_raw[:, :, t+1:] = 9876.5
                        changed_m3[:, :, t+1:] = -9876.5
                        changed_local[:, :, t+1:] = 9876.5
                        changed = grade(changed_m3, changed_local, smooth(changed_raw), th, changed_raw, policy)
                        compare(label+f'/future_mutation/{t}', bool(np.array_equal(changed[:, :, :t+1], reconstructed[:, :, :t+1])), True)
                        prefix = grade(m3[:, :, :t+1], local[:, :, :t+1], smooth(raw[:, :, :t+1]), th, raw[:, :, :t+1], policy)
                        compare(label+f'/prefix/{t}', bool(np.array_equal(prefix, reconstructed[:, :, :t+1])), True)
                    for group, group_report in report['groups'].items():
                        key, value = group.split(':', 1)
                        key = 'shape_family' if key == 'family' else key
                        mask = np.array([str(row[key]) == value for row in scene_rows[split]])
                        compare(label+'/group/'+group+'/any', summary((reconstructed > 0)[mask], category[mask]), group_report['any'])
                        compare(label+'/group/'+group+'/paired', paired(references['old_fusion'][mask], (reconstructed > 0)[mask], category[mask]), group_report['vs_old_fusion'])
                exists, intrusion = local[:, :, 10] >= LOCAL, ordinary[:, :, 10] >= th['single']
                enhancing, volatile = slope[:, :, 10] > 0, residual[:, :, 10] > th['residual_cutoff']
                total = 0
                for row in grids:
                    if row['split'] != split or int(row['seed']) != seed:
                        continue
                    q = ('HEAD', 'BODY').index(row['height'])
                    selected = (category[:, q] == row['category'])[:, None]
                    for value, name in ((exists, 'exists'), (intrusion, 'intrusion'), (enhancing, 'support_enhancing'), (volatile, 'high_detrended_residual')):
                        selected = selected & (value[..., q] == int(row[name]))
                    count = int(selected.sum()); total += count
                    compare(f'{split}/{seed}/grid/'+str(checks), count, int(row['count']))
                compare(f'{split}/{seed}/grid_denominator', total, category.size*4)
                bins = np.array([-np.inf, -1., 0., 1., np.inf])
                margin = ordinary[:, :, 10]-th['single']
                boundary = group_boundaries[str(seed)]
                for q, height in enumerate(('HEAD', 'BODY')):
                    for b in range(4):
                        score = (margin[..., q] >= bins[b]) & (margin[..., q] < bins[b+1])
                        for cat in ('contact', 'pass', 'clear'):
                            selected = score & (category[:, q] == cat)[:, None]
                            row = dict(split=split, seed=seed, height=height, score_bin=b,
                                margin_lower=str(bins[b]), margin_upper=str(bins[b+1]), category=cat, count=int(selected.sum()),
                                score_bin_denominator=int(score.sum()), cal_slope_median=boundary['slope'], cal_residual_median=boundary['residual'])
                            for name, values in (('margin', margin[..., q]), ('slope', slope[..., 10, q]), ('residual', residual[..., 10, q])):
                                observed = values[selected]
                                for quantile, label in ((.1, 'p10'), (.5, 'median'), (.9, 'p90')):
                                    row[name+'_'+label] = float(np.quantile(observed, quantile)) if len(observed) else None
                            row['slope_above_cal_median'] = int((slope[..., 10, q][selected] > boundary['slope']).sum())
                            row['residual_above_cal_median'] = int((residual[..., 10, q][selected] > boundary['residual']).sum())
                            distribution.append(row)
                if time.monotonic()-began >= 600:
                    raise TimeoutError('Independent audit CPU600s cap reached')
        compare('event_ledger_denominator', len(events), 2*3*3*384*4*2)
        compare('event_ledger_unique_keys', len(event_index), len(events))
        compare('grid_rows', len(grids), 2*3*2*16*3)
        save(OUT/'score_strata_boundaries.json', group_boundaries)
        csv_write(OUT/'score_matched_trend_residual.csv', distribution)
        save(OUT/'receipt.json', dict(status='PASS' if not failures else 'FAIL', checks=checks, failures=failures,
            seconds=time.monotonic()-began, cells=18, event_rows=len(events), grid_rows=len(grids), descriptive_rows=len(distribution),
            causal='All18 cells, each of13 cuts: arbitrary future mutation and truncated-prefix reconstruction; no production functions imported',
            residual_rounding=rounding, first_attempt_preserved=True, preserve_strong_cells=6,
            source_sha256=sha(Path(__file__)), production_source_sha256=sha(Path(__file__).with_name('cnh_graded_evidence_dev.py')),
            limitations='Consumed synthetic Development; broad score bins describe strata and do not make scores exactly matched; no labels in grade computation'))
        print(json.dumps(dict(status='PASS' if not failures else 'FAIL', checks=checks, failures=failures[:10], seconds=time.monotonic()-began)))
    except BaseException as error:
        save(OUT/'failure.json', dict(error=repr(error), seconds=time.monotonic()-began, failures=failures))
        raise


if __name__ == '__main__':
    run()
