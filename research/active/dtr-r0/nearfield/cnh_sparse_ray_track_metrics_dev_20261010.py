"""Negative-calibration-only sparse-ray score evaluation on consumed Development.

Three fixed additive arms and one primary standalone arm are evaluated; the
aligned_fixed4 score remains a descriptor. No training, inference, E2E holdout
access, contact-based operating-point selection or source mechanism changes.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import cnh_frozen_e2e_metrics_20261010 as M

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
DEFAULT_OUT = ROOT/'artifacts.local/work/cnh-sparse-ray-track-dev-20261010'
ARMS = ('aligned_adaptive', 'unaligned_adaptive', 'current_only')
DESCRIPTOR = 'aligned_fixed4'
M3_THETA = .8557642486787612
RAISED_M3 = .9404184587540165
LOCAL_THETA = 4.625390338985158


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def save_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def smooth(raw):
    result = np.empty_like(raw, dtype=float)
    for t in range(raw.shape[-2]):
        values = raw[..., max(0, t-4):t+1, :].astype(float)
        result[..., t, :] = np.average(values, axis=-2, weights=np.exp2(np.arange(values.shape[-2])))
    return result


def notification_formula(grade):
    """Stateless episode indexing for exact batched calibration costs."""
    t = np.arange(grade.shape[-2]).reshape((1,) * (grade.ndim-2) + (-1, 1))
    zero = grade == 0
    previous_zero = np.zeros_like(zero); previous_zero[..., 1:, :] = zero[..., :-1, :]
    reset = np.maximum.accumulate(np.where(zero & previous_zero, t, -1), axis=-2)
    def previous_hit(flags):
        last = np.maximum.accumulate(np.where(flags, t, -1), axis=-2)
        previous = np.full_like(last, -1); previous[..., 1:, :] = last[..., :-1, :]
        return previous
    return np.where((grade == 2) & (previous_hit(grade == 2) <= reset), 2,
        np.where((grade == 1) & (previous_hit(grade > 0) <= reset), 1, 0)).astype(np.int8)


def cost_masks(category):
    return dict(clear=(category == 'clear').all(1),
                purepass=(category == 'pass').any(1) & ~(category == 'contact').any(1))


def make_grade(score, old5, tau, standalone=False):
    flagged = np.isfinite(score) & (score >= tau)
    if standalone:
        return flagged.astype(np.int8)
    return np.where(old5, 2, flagged.astype(np.int8)).astype(np.int8)


def cutoff_record(tau):
    return dict(tau=float(tau) if np.isfinite(tau) else None,
                tau_kind='finite' if np.isfinite(tau) else 'positive_infinity' if tau > 0 else 'negative_infinity')


def decode_tau(record):
    return record['tau'] if record['tau_kind'] == 'finite' else np.inf if record['tau_kind'] == 'positive_infinity' else -np.inf


def calibrate(score, old5, category, check=lambda: None, standalone=False, chunk=32):
    """Enumerate every complete negative score tie; never inspect contact utility.

    Joint costs are max emitted HEAD/BODY grade per frame. Notification costs
    need not be monotonic in tau. Return first feasible point in ascending tau.
    """
    if score.shape != old5.shape or score.ndim != 4 or score.shape[-2:] != (13, 2):
        raise ValueError('Paired scores and old5 flags must be [N,K,13,2]')
    if np.isposinf(score).any() or np.isneginf(score).any():
        raise ValueError('Cache scores must be finite or NaN missing; infinities not admitted')
    masks = cost_masks(category)
    negative = masks['clear'] | masks['purepass']
    if not masks['clear'].any() or not masks['purepass'].any():
        raise ValueError('Both negative calibration cost strata required')
    s, old = score[negative], old5[negative]
    isclear = masks['clear'][negative]
    original = M.replay_gap1((2*old).astype(np.int8)).max(-1)
    caps = [int((original[isclear] > 0).sum()), int((original[~isclear] > 0).sum())]
    eligible = np.isfinite(s) & (np.ones_like(old, bool) if standalone else ~old)
    ties = np.unique(s[eligible])
    points = np.r_[-np.inf, np.nextafter(ties, np.inf), np.inf]
    curve, selected = [], None
    for begin in range(0, len(points), chunk):
        check()
        thresholds = points[begin:begin+chunk]
        flagged = np.isfinite(s)[None] & (s[None] >= thresholds[:, None, None, None, None])
        grades = flagged.astype(np.int8) if standalone else np.where(old[None], 2, flagged).astype(np.int8)
        joint = notification_formula(grades).max(-1)
        clear = (joint[:, isclear] > 0).sum((1, 2, 3))
        passed = (joint[:, ~isclear] > 0).sum((1, 2, 3))
        for tau, cc, pc in zip(thresholds, clear, passed):
            feasible = bool(cc <= caps[0] and pc <= caps[1])
            curve.append(dict(**cutoff_record(tau), clear=int(cc), purepass=int(pc), feasible=feasible))
            if selected is None and feasible:
                selected = float(tau)
    if selected is None:
        raise AssertionError('Infinity fallback must satisfy negative cost caps')
    return dict(**cutoff_record(selected), clear_cap=caps[0], purepass_cap=caps[1],
        finite_negative_score_ties=len(ties), points=len(curve),
        contact_utility_access=False, standalone=standalone,
        selection='First feasible ascending complete negative-score ties; all points replayed, nonmonotonic costs allowed'), curve


def evaluator_rows(rows):
    """Only evaluation strata: source HEAD horizontal4cm is not new1.7cm."""
    result = []
    for row in rows:
        updated = dict(row)
        thickness = row['target_box']['hi'][1]-row['target_box']['lo'][1]
        updated['size_variant'] = 0 if row['shape_family'] == 'horizontal' and abs(thickness-.04) < 1e-8 else 1
        result.append(updated)
    return result


def rename_thin(value):
    if isinstance(value, dict):
        return {('HEAD_horizontal4cm' if k == 'HEAD_horizontal_thin' else k): rename_thin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rename_thin(v) for v in value]
    return value


def physical_ci(scene_numerator, scene_denominator, rows):
    grouped = {}
    for n, d, row in zip(scene_numerator, scene_denominator, rows):
        if d > 0:
            key = row['physical_key']
            a, b = grouped.get(key, (0, 0)); grouped[key] = (a+int(n), b+int(d))
    numerator = np.asarray([v[0] for v in grouped.values()])
    denominator = np.asarray([v[1] for v in grouped.values()])
    count = len(denominator)
    estimate = float(numerator.sum()/denominator.sum()) if count else None
    if count < 2:
        return dict(estimate=estimate, ci95=None, independent_physical_worlds=count, status='NOT_EVALUABLE_FEWER_THAN_TWO_PHYSICAL_WORLDS')
    rng = np.random.default_rng(20261010)
    ix = rng.integers(0, count, (2000, count))
    rates = numerator[ix].sum(1)/denominator[ix].sum(1)
    return dict(estimate=estimate, ci95=np.quantile(rates, [.025, .975]).tolist(),
                independent_physical_worlds=count, status='DESCRIPTIVE_PHYSICAL_CLUSTER_PERCENTILE')


def contact_masks(category, rows):
    result = M._contact_masks(category, np.asarray([r['shape_family'] for r in rows]), rows)
    dark = np.asarray([r['rho'] == .25 for r in rows])[:, None]
    result['HEAD_horizontal4cm_dark'] = result['HEAD_horizontal_thin'] & dark
    return result


def summary(grade, category, rows):
    report = M.summarize(grade, category, rows)
    notice = M.replay_gap1(grade)
    darkmask = contact_masks(category, rows)['HEAD_horizontal4cm_dark']
    report['contacts']['HEAD_horizontal4cm_dark'] = M._contact_counts(M._first(notice), M._first(notice, 2), darkmask, 11)
    report['contract']['primary'] = 'HEAD contact horizontal4cm; dark rho.25 and all horizontal secondary'
    report['contract']['independent_unit'] = 'physical_key cluster; retain all row occurrences and K4'
    report['contract']['cost_CI'] = 'Not reported; all notification counts and exposure denominators retained'
    report['physical_worlds'] = len({r['physical_key'] for r in rows})
    return rename_thin(report)


def paired(base, candidate, category, rows):
    # Reuse integer counts/timing, then replace row-based CIs with physical-world CIs.
    report = M.compare(base, candidate, category, rows, bootstrap_samples=1)
    before, after = M.replay_gap1(base), M.replay_gap1(candidate)
    first_base, first_candidate = M._first(before), M._first(after)
    strong_base, strong_candidate = M._first(before, 2), M._first(after, 2)
    for name, mask in contact_masks(category, rows).items():
        truth = np.broadcast_to(mask[:, None, :], first_base.shape)
        oldtimely = truth & (first_base >= 0) & (first_base < 11)
        newtimely = truth & (first_candidate >= 0) & (first_candidate < 11)
        if name not in report['contacts']:
            # The dark secondary stratum is a pure evaluator subset.
            subset = [dict(r, shape_family='horizontal' if r['rho'] == .25 else 'other') for r in rows]
            secondary = M.compare(base, candidate, category, subset, bootstrap_samples=1)
            report['contacts'][name] = secondary['contacts']['HEAD_horizontal_thin']
        item = report['contacts'][name]
        item['timely_rate_difference_ci'] = physical_ci(
            (newtimely.astype(int)-oldtimely.astype(int)).sum((1, 2)), truth.sum((1, 2)), rows)
        item['strong_to_light_timely'] = int((truth & (strong_base >= 0) & (strong_base < 11)
            & newtimely & ((strong_candidate < 0) | (strong_candidate >= 11))).sum())
    for cost in report['costs'].values():
        for value in cost.values():
            if isinstance(value, dict):
                value.pop('per_clip_difference_ci', None)
    report['bootstrap'] = dict(seed=20261010, samples=2000, confidence=.95,
        unit='physical_key clusters, all source occurrences and K4 retained; contact strata only',
        interpretation='descriptive consumed Development; shared backgrounds and design factors remain correlated')
    return rename_thin(report)


def load(split, out):
    folder = SOURCE/'data'/split
    rows = evaluator_rows(read(SOURCE/'scene_rows.json')[split])
    with np.load(folder/'geometry.npz', allow_pickle=False) as a:
        category, ids, uids = a['category'], a['scene_ids'], a['scene_uids']
        np.testing.assert_array_equal(a['frame_category'], np.broadcast_to(category[:, None, :], (384, 13, 2)))
    with np.load(SOURCE/'baselines'/f'{split}_ideal.npz', allow_pickle=False) as a:
        m3raw, localraw = a['m3_raw'], a['local_raw']
    if not np.isfinite(m3raw).all() or np.isnan(localraw).any() or np.isposinf(localraw).any():
        raise ValueError('Finite M3 required; local may be -inf, never nan/+inf')
    with np.load(out/f'{split}_scores.npz', allow_pickle=False) as a:
        scores, arms = a['scores'], a['arm_names'].tolist()
        anchor = a['anchor_native_index']
        np.testing.assert_array_equal(a['scene_ids'], ids)
        np.testing.assert_array_equal(a['scene_uids'], uids)
        np.testing.assert_array_equal(a['frames'], np.arange(3, 16))
        np.testing.assert_array_equal(a['queries'], ['HEAD', 'BODY'])
    if arms != [*ARMS, DESCRIPTOR] or scores.shape != (4, 384, 4, 13, 2):
        raise ValueError('Exactly three fixed evaluator arms plus fixed4 descriptor required')
    if np.isinf(scores).any():
        raise ValueError('Scores must be finite or NaN indicating unavailable anchor')
    if anchor.shape != scores.shape or not np.array_equal(np.isnan(scores), anchor == -1):
        raise ValueError('Missing score must match missing native anchor index, never negative evidence')
    if not ((anchor >= -1) & (anchor < 1024)).all():
        raise ValueError('Native anchor indices must be -1 or 0..1023')
    np.testing.assert_array_equal(ids, [r['scene_id'] for r in rows])
    np.testing.assert_array_equal(uids, [r['scene_uid'] for r in rows])
    m3, local = smooth(m3raw), smooth(localraw)
    old5 = (m3 >= RAISED_M3) | (local >= LOCAL_THETA)
    return dict(rows=rows, category=category, ids=ids, uids=uids, scores=scores,
        m3=(2*(m3 >= M3_THETA)).astype(np.int8), old5=(2*old5).astype(np.int8),
        inputs_sha256={str(p): sha(p) for p in (SOURCE/'scene_rows.json', folder/'geometry.npz',
            SOURCE/'baselines'/f'{split}_ideal.npz', out/f'{split}_scores.npz')})


def run(out):
    start = time.monotonic()
    attempts = out/'evaluation_attempts'; attempts.mkdir(exist_ok=True)
    previous = sum(read(p)['seconds'] for p in attempts.glob('*.json'))
    receipt = dict(status='FAILED', source_sha256=sha(Path(__file__)), previous_seconds=previous,
                   model_forward=0, training=0, protected_access=0, fresh_e2e_access=0)
    def check():
        if previous+time.monotonic()-start >= 300:
            raise TimeoutError('Cumulative evaluation300 command-wall seconds reached')
    try:
        if (out/'evaluation_receipt.json').exists():
            raise FileExistsError('Preserve completed evaluator result')
        cal = load('cal', out)
        cases = [(arm+'/addon', si, False) for si, arm in enumerate(ARMS)]
        cases.append(('aligned_adaptive/standalone', 0, True))
        calibrations, curves = {}, []
        for name, si, standalone in cases:
            record, curve = calibrate(cal['scores'][si], cal['old5'] > 0, cal['category'], check, standalone)
            calibrations[name] = record
            curves.extend(dict(arm=name, **r) for r in curve)
            print('RAYTRACK_CALIBRATED', name, record['tau_kind'], record['tau'], len(curve), flush=True)
        sealed = dict(status='SEALED', calibrations=calibrations,
            inputs_sha256=cal['inputs_sha256'], source_sha256=sha(Path(__file__)),
            main_arm='aligned_adaptive/addon', descriptor_only=DESCRIPTOR,
            selection='Negative costs only; no contact utility in selection; no validation loaded')
        save_new(out/'sealed_calibration.json', sealed)
        write_csv(out/'calibration_curve.csv', curves)
        # Validation was consumed Development before this task; still never retune on it.
        validation = load('validation', out)
        reports, comparisons, ledger = {}, {}, []
        inputs = {'cal': cal['inputs_sha256'], 'validation': validation['inputs_sha256']}
        for split, data in (('cal', cal), ('validation', validation)):
            grades = dict(M3=data['m3'], old5=data['old5'])
            for name, si, standalone in cases:
                grades[name] = make_grade(data['scores'][si], data['old5'] > 0,
                                          decode_tau(calibrations[name]), standalone)
            notices, keys, saved = [], list(grades), []
            for name, grade in grades.items():
                check()
                notice = M.replay_gap1(grade)
                np.testing.assert_array_equal(notice, notification_formula(grade))
                np.testing.assert_array_equal(M._first(grade), M._first(notice))
                np.testing.assert_array_equal(M._first(grade, 2), M._first(notice, 2))
                if name.endswith('/addon'):
                    np.testing.assert_array_equal(grade[data['old5'] > 0], data['old5'][data['old5'] > 0])
                reports[f'{split}/{name}'] = summary(grade, data['category'], data['rows'])
                if name not in ('M3', 'old5'):
                    comparisons[f'{split}/{name}'] = paired(data['old5'], grade, data['category'], data['rows'])
                onset, strong = M._first(notice), M._first(notice, 2)
                for n, row in enumerate(data['rows']):
                    for k in range(4):
                        for q, height in enumerate(('HEAD', 'BODY')):
                            ledger.append(dict(split=split, arm=name, scene_id=int(data['ids'][n]),
                                scene_uid=row['scene_uid'], physical_key=row['physical_key'], replica=k,
                                height=height, category=data['category'][n, q], shape_family=row['shape_family'],
                                horizontal4cm=row['size_variant'] == 0, rho=row['rho'],
                                background_id=row['background_id'], background_family=row['background_family'],
                                first_any=int(onset[n, k, q]+3) if onset[n, k, q] >= 0 else -1,
                                first_strong=int(strong[n, k, q]+3) if strong[n, k, q] >= 0 else -1,
                                timely=bool(0 <= onset[n, k, q] < 11),
                                light_notifications=int((notice[n, k, :, q] == 1).sum()),
                                strong_notifications=int((notice[n, k, :, q] == 2).sum())))
                saved.append(grade); notices.append(notice)
            np.savez_compressed(out/f'{split}_grades_notifications.npz', keys=np.asarray(keys),
                grades=np.asarray(saved), notifications=np.asarray(notices), category=data['category'],
                scene_ids=data['ids'], scene_uids=data['uids'])
        write_csv(out/'event_ledger.csv', ledger)
        save_new(out/'metrics.json', dict(reports=reports, comparisons=comparisons,
            calibrations=calibrations, scope='Consumed ideal simulated Development, no fresh confirmation',
            negative_cost_exposure='clear512/purepass256clip per split; all authored occurrences/K4 retained',
            independence='384 authored rows but324physical worlds;64absence occurrences are4worlds. No cost CI. Contact CI groups physical keys and K4; background factors still shared.',
            standalone='All standalone positive grades are light; paired strong-to-light timely count reported, no default promotion'))
        receipt.update(status='COMPLETE', inputs_sha256=inputs, cells=len(reports), pairs=len(comparisons),
            ledger_query_rows=len(ledger), calibration_points=len(curves),
            sealed_calibration_sha256=sha(out/'sealed_calibration.json'),
            outputs_sha256={name:sha(out/name) for name in ('metrics.json', 'calibration_curve.csv',
                'event_ledger.csv', 'cal_grades_notifications.npz', 'validation_grades_notifications.npz')})
        print('RAYTRACK_EVALUATION_COMPLETE', round(time.monotonic()-start, 3), flush=True)
    except BaseException as error:
        receipt.update(error=repr(error), traceback=traceback.format_exc()); raise
    finally:
        receipt['seconds'] = time.monotonic()-start
        receipt['cumulative_seconds'] = previous+receipt['seconds']
        save_new(attempts/f'evaluation_{time.time_ns()}.json', receipt)
        if receipt['status'] == 'COMPLETE':
            save_new(out/'evaluation_receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    destination = args.out.resolve()
    if not destination.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Canonical artifacts.local required')
    run(destination)
