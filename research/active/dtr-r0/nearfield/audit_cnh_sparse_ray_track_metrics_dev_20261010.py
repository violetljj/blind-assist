"""Independent scalar notification/metric and saved calibration-point audit.

No reader of fresh E2E data, no mechanism recomputation, inference or selection.
Checks all curve identities/feasibility flags, and independently recounts selected,
preceding and endpoint costs. Cumulative audit command wall is capped at120s.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
OUT = ROOT/'artifacts.local/work/cnh-sparse-ray-track-dev-20261010'
ARMS = ('aligned_adaptive', 'unaligned_adaptive', 'current_only')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def replay(values):
    answer, peak, zeros = [], 0, 0
    for value in values:
        g = int(value)
        if g == 0:
            zeros += 1
            if zeros >= 2:
                peak = 0
            answer.append(0)
        else:
            answer.append(g if g > peak else 0)
            peak = max(peak, g); zeros = 0
    return np.asarray(answer, np.int8)


def replay_all(grade):
    result = np.zeros_like(grade)
    for n in range(grade.shape[0]):
        for k in range(grade.shape[1]):
            for q in range(2):
                result[n, k, :, q] = replay(grade[n, k, :, q])
    return result


def first(values, strong=False):
    flags = values == 2 if strong else values > 0
    return np.where(flags.any(2), flags.argmax(2), -1)


def smooth(values):
    result = np.empty_like(values, dtype=float)
    for t in range(13):
        w = np.exp2(np.arange(min(5, t+1)))
        result[..., t, :] = np.average(values[..., max(0, t-4):t+1, :].astype(float), axis=-2, weights=w)
    return result


def masks(category, rows):
    contact = category == 'contact'
    head = np.zeros_like(contact); head[:, 0] = contact[:, 0]
    body = np.zeros_like(contact); body[:, 1] = contact[:, 1]
    horizontal = np.array([r['shape_family'] == 'horizontal' for r in rows])[:, None]
    thin = np.array([abs(r['target_box']['hi'][1]-r['target_box']['lo'][1]-.04) < 1e-8 for r in rows])[:, None]
    edge = np.array([r['shape_family'] == 'sign_edge' for r in rows])[:, None]
    dark = np.array([r['rho'] == .25 for r in rows])[:, None]
    return dict(HEAD=head, BODY=body, all_contact_queries=contact, HEAD_horizontal4cm=head & horizontal & thin,
                HEAD_horizontal=head & horizontal, HEAD_sign_edge=head & edge,
                HEAD_horizontal4cm_dark=head & horizontal & thin & dark)


def cost_masks(category):
    return dict(clear=(category == 'clear').all(1),
                purepass=(category == 'pass').any(1) & ~(category == 'contact').any(1))


def stats(values):
    return dict(n=int(values.size), min=float(values.min()) if values.size else None,
                median=float(np.median(values)) if values.size else None,
                max=float(values.max()) if values.size else None, sum=float(values.sum()))


def same(actual, expected):
    for key, value in actual.items():
        if isinstance(value, dict):
            same(value, expected[key])
        else:
            assert value == expected[key], (key, value, expected[key])


def counts(emitted, category, rows):
    f, s = first(emitted), first(emitted, True)
    contacts = {}
    for name, mask in masks(category, rows).items():
        truth = np.broadcast_to(mask[:, None, :], f.shape)
        a, b = f[truth], s[truth]
        contacts[name] = dict(denominator=int(truth.sum()), scene_denominator=int(mask.any(1).sum()),
            timely=int(((a >= 0) & (a < 11)).sum()), timely_strong=int(((b >= 0) & (b < 11)).sum()),
            late=int((a >= 11).sum()), silent=int((a < 0).sum()), any_notification=int((a >= 0).sum()),
            first_index_histogram={str(int(t)):int((a == t).sum()) for t in np.unique(a)},
            first_strong_index_histogram={str(int(t)):int((b == t).sum()) for t in np.unique(b)},
            first_index_stats=stats(a[a >= 0]), first_strong_index_stats=stats(b[b >= 0]))
    costs = {}
    joint = emitted.max(-1)
    for name, mask in cost_masks(category).items():
        v = joint[mask]
        costs[name] = dict(scene_denominator=int(mask.sum()), clip_denominator=v.shape[0]*v.shape[1],
            frame_denominator=int(v.size), notifications=int((v > 0).sum()),
            light_notifications=int((v == 1).sum()), strong_notifications=int((v == 2).sum()),
            clips_with_any_notification=int((v > 0).any(2).sum()),
            clips_with_light_notification=int((v == 1).any(2).sum()),
            clips_with_strong_notification=int((v == 2).any(2).sum()))
    valid = (category == 'contact')[:, None, :]
    physical = dict(denominator=int((category == 'contact').any(1).sum()*emitted.shape[1]),
        scene_denominator=int((category == 'contact').any(1).sum()),
        timely=int((valid & (f >= 0) & (f < 11)).any(-1).sum()))
    return dict(contacts=contacts, costs=costs, physical_contact_clips=physical)


def physical_ci(num, den, rows):
    clusters = {}
    for n, d, row in zip(num, den, rows):
        if d > 0:
            a, b = clusters.get(row['physical_key'], (0, 0))
            clusters[row['physical_key']] = (a+int(n), b+int(d))
    numerator = np.array([r[0] for r in clusters.values()]); denominator = np.array([r[1] for r in clusters.values()])
    if len(denominator) < 2:
        return None, len(denominator)
    rng = np.random.default_rng(20261010)
    resample = rng.integers(0, len(denominator), (2000, len(denominator)))
    rates = numerator[resample].sum(1)/denominator[resample].sum(1)
    return np.quantile(rates, [.025, .975]).tolist(), len(denominator)


def pairs(before, after, category, rows, expected):
    a, b, sa, sb = first(before), first(after), first(before, True), first(after, True)
    for name, mask in masks(category, rows).items():
        truth = np.broadcast_to(mask[:, None, :], a.shape)
        at, bt = truth & (a >= 0) & (a < 11), truth & (b >= 0) & (b < 11)
        common = truth & (a >= 0) & (b >= 0); delta = b-a
        n, d = (bt.astype(int)-at.astype(int)).sum((1, 2)), truth.sum((1, 2))
        row = dict(denominator=int(truth.sum()), scene_denominator=int(mask.any(1).sum()),
            baseline_timely=int(at.sum()), candidate_timely=int(bt.sum()), rescue=int((~at & bt).sum()),
            loss=int((at & ~bt).sum()), net=int(n.sum()),
            baseline_timely_strong=int((truth & (sa >= 0) & (sa < 11)).sum()),
            candidate_timely_strong=int((truth & (sb >= 0) & (sb < 11)).sum()),
            common_any_notified=int(common.sum()), common_timely=int((at & bt).sum()),
            earlier_common=int((common & (delta < 0)).sum()), later_common=int((common & (delta > 0)).sum()),
            unchanged_common=int((common & (delta == 0)).sum()),
            delta_frames_common=stats(delta[common]), delta_frames_common_timely=stats(delta[at & bt]),
            strong_to_light_timely=int((truth & (sa >= 0) & (sa < 11) & bt & ((sb < 0) | (sb >= 11))).sum()))
        same(row, expected['contacts'][name])
        interval, groups = physical_ci(n, d, rows)
        saved_ci = expected['contacts'][name]['timely_rate_difference_ci']
        assert saved_ci['ci95'] == interval and saved_ci['independent_physical_worlds'] == groups
    # No cost CI pretending repeated absent occurrences are independent.
    for cost in expected['costs'].values():
        assert not any(isinstance(v, dict) and 'per_clip_difference_ci' in v for v in cost.values())


def tau(row):
    return float(row['tau']) if row['tau_kind'] == 'finite' else np.inf if row['tau_kind'] == 'positive_infinity' else -np.inf


def audit(out):
    start = time.monotonic()
    folder = out/'audit_attempts'; folder.mkdir(exist_ok=True)
    previous = sum(read(p)['seconds'] for p in folder.glob('*.json'))
    receipt = dict(status='FAILED', previous_seconds=previous, source_sha256=sha(Path(__file__)),
                   model_forward=0, training=0, new_noise=0, fresh_e2e_access=0)
    def check():
        if previous+time.monotonic()-start >= 120:
            raise TimeoutError('Cumulative metrics audit120 command-wall reached')
    try:
        execution = read(out/'evaluation_receipt.json')
        assert execution['status'] == 'COMPLETE' and 'result_directory' not in execution
        frozen = read(out/'evaluation_sources.json')
        for path, expected in frozen.items():
            assert sha(ROOT/path) == expected
        frozen_metrics = frozen['research/active/dtr-r0/nearfield/cnh_sparse_ray_track_metrics_dev_20261010.py']
        assert sha(out/'metrics_source_before_resume.py') == frozen_metrics == execution['source_sha256']
        lineage = read(out/'metrics_source_lineage.json')
        assert lineage['frozen_sha256'] == frozen_metrics
        assert sha(out/'metrics_source_after_resume.py') == lineage['after_resume_sha256']
        a = ast.parse((out/'metrics_source_before_resume.py').read_text())
        b = ast.parse((out/'metrics_source_after_resume.py').read_text())
        fa = {n.name:ast.dump(n) for n in a.body if isinstance(n, ast.FunctionDef)}
        fb = {n.name:ast.dump(n) for n in b.body if isinstance(n, ast.FunctionDef)}
        assert [n for n in fa if fa[n] != fb[n]] == ['run']
        for bysplit in execution['inputs_sha256'].values():
            for path, expected in bysplit.items():
                assert sha(path) == expected
        for path, expected in execution['outputs_sha256'].items():
            assert sha(out/path) == expected
        report = read(out/'metrics.json'); sealed = read(out/'sealed_calibration.json')
        assert 'curve_sha256' not in sealed and sealed['source_sha256'] == frozen_metrics
        assert sha(out/'sealed_calibration.json') == execution['sealed_calibration_sha256']
        assert sealed['calibrations'] == report['calibrations']
        author = read(SOURCE/'scene_rows.json')
        curve = csv_read(out/'calibration_curve.csv')
        archives, scorearchives, tally = {}, {}, {}
        streams, cells, paircells = 0, 0, 0
        for split in ('cal', 'validation'):
            check()
            with np.load(out/f'{split}_grades_notifications.npz', allow_pickle=False) as a:
                saved = {k:a[k] for k in a.files}
            archives[split] = saved
            with np.load(out/f'{split}_scores.npz', allow_pickle=False) as a:
                scores = a['scores']; arms = a['arm_names'].tolist()
                assert arms == [*ARMS, 'aligned_fixed4']
                np.testing.assert_array_equal(np.isnan(scores), a['anchor_native_index'] == -1)
                np.testing.assert_array_equal(a['frames'], np.arange(3, 16))
                np.testing.assert_array_equal(a['scene_ids'], saved['scene_ids'])
                np.testing.assert_array_equal(a['scene_uids'], saved['scene_uids'])
            scorearchives[split] = scores
            with np.load(SOURCE/'baselines'/f'{split}_ideal.npz', allow_pickle=False) as a:
                assert np.isfinite(a['m3_raw']).all()
                assert not np.isnan(a['local_raw']).any() and not np.isposinf(a['local_raw']).any()
                m3, local = smooth(a['m3_raw']), smooth(a['local_raw'])
            category, rows = saved['category'], author[split]
            assert len({r['physical_key'] for r in rows}) == 324
            base = (m3 >= .9404184587540165) | (local >= 4.625390338985158)
            expected_grades = dict(M3=2*(m3 >= .8557642486787612), old5=2*base)
            for name, record in sealed['calibrations'].items():
                arm, kind = name.split('/')
                score = scores[arms.index(arm)]
                flags = np.isfinite(score) & (score >= tau(record))
                expected_grades[name] = flags.astype(np.int8) if kind == 'standalone' else np.where(base, 2, flags)
            names = saved['keys'].tolist()
            assert names == list(expected_grades)
            notices = {}
            for name, grade, emitted in zip(names, saved['grades'], saved['notifications']):
                check()
                np.testing.assert_array_equal(grade, expected_grades[name])
                independent = replay_all(grade)
                np.testing.assert_array_equal(independent, emitted)
                np.testing.assert_array_equal(first(grade), first(emitted))
                np.testing.assert_array_equal(first(grade, True), first(emitted, True))
                notices[name] = independent; streams += grade.shape[0]*grade.shape[1]*2
                counted = counts(independent, category, rows)
                same(counted, report['reports'][f'{split}/{name}'])
                tally[f'{split}/{name}'] = counted; cells += 1
            for name in sealed['calibrations']:
                check()
                pairs(notices['old5'], notices[name], category, rows, report['comparisons'][f'{split}/{name}'])
                paircells += 1
        cal = archives['cal']; category = cal['category']; masks_negative = cost_masks(category)
        negative = masks_negative['clear'] | masks_negative['purepass']
        old = cal['grades'][cal['keys'].tolist().index('old5')] > 0
        baselinecost = tally['cal/old5']['costs']; caps = [baselinecost['clear']['notifications'], baselinecost['purepass']['notifications']]
        assert caps == [39, 60]
        verified_points = {}
        for name, record in sealed['calibrations'].items():
            check()
            arm, kind = name.split('/')
            score = scorearchives['cal'][ARMS.index(arm)]
            eligible = negative[:, None, None, None] & np.isfinite(score)
            if kind == 'addon':
                eligible &= ~old
            ties = np.unique(score[eligible]); expected_tau = np.r_[-np.inf, np.nextafter(ties, np.inf), np.inf]
            rows = [r for r in curve if r['arm'] == name]
            assert len(rows) == len(expected_tau) == record['points']
            assert len(ties) == record['finite_negative_score_ties']
            feasible_indices = []
            for j, (point, row) in enumerate(zip(expected_tau, rows)):
                assert point == tau(row)
                feasible = int(row['clear']) <= caps[0] and int(row['purepass']) <= caps[1]
                assert (row['feasible'] == 'True') == feasible
                if feasible:
                    feasible_indices.append(j)
            selected = feasible_indices[0]
            assert tau(record) == expected_tau[selected]
            assert record['contact_utility_access'] is False
            assert record['clear_cap'] == caps[0] and record['purepass_cap'] == caps[1]
            probes = sorted({0, max(0, selected-1), selected, len(rows)-2, len(rows)-1})
            for j in probes:
                point = expected_tau[j]
                flagged = np.isfinite(score[negative]) & (score[negative] >= point)
                grade = flagged.astype(np.int8) if kind == 'standalone' else np.where(old[negative], 2, flagged).astype(np.int8)
                joint = replay_all(grade).max(-1)
                clear = masks_negative['clear'][negative]
                observed = [int((joint[clear] > 0).sum()), int((joint[~clear] > 0).sum())]
                assert observed == [int(rows[j]['clear']), int(rows[j]['purepass'])]
            verified_points[name] = dict(curve_points_identity_verified=len(rows), first_feasible_index=selected,
                independently_replayed_indices=probes, selected_tau=record['tau'], selected_tau_kind=record['tau_kind'])
        # One predeclared actual stream per case exercises the producer formula again.
        import cnh_sparse_ray_track_metrics_dev_20261010 as producer
        for split in ('cal', 'validation'):
            for caseindex in range(6):
                actual = archives[split]['grades'][caseindex, 0:1, 0:1]
                np.testing.assert_array_equal(producer.notification_formula(actual), replay_all(actual))
        receipt.update(status='PASS', scalar_streams_verified=streams, metric_cells_verified=cells,
            paired_cells_verified=paircells, calibration=verified_points, counts=tally,
            frozen_execution_body_identity='b75 snapshot, repo file restored byte-for-byte, receipt/seal match; no resume metadata fields; numerical function AST unchanged during source-file edit',
            limitations='Consumed Development. Physical cluster contact CI preserves occurrences/K; shared background design remains correlated. No cost CI. Complete curve costs not all independently rerun; all tie identities/feasibility flags and five selected/adjacent/endpoint probes per case verified.',
            model_forward=0, training=0, new_noise=0, fresh_e2e_access=0)
        print('SPARSE_METRIC_AUDIT_PASS', streams, cells, paircells, round(time.monotonic()-start, 3), flush=True)
    except BaseException as error:
        receipt.update(error=repr(error), traceback=traceback.format_exc()); raise
    finally:
        receipt['seconds'] = time.monotonic()-start
        receipt['cumulative_seconds'] = previous+receipt['seconds']
        save_new(folder/f'metrics_audit_{time.time_ns()}.json', receipt)
        if receipt['status'] == 'PASS':
            save_new(out/'metrics_audit_receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    audit(args.out.resolve())
