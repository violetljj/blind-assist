"""Independent descriptive recount and expectation audit on consumed ToF scenes.

No model forward, training, noise sampling, threshold selection or protected data.
This file intentionally does not import the primary decomposition implementation.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = ROOT/'artifacts.local/work/cnh-baseline-miss-decomposition-dev-20261010'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def near(actual, expected, label):
    np.testing.assert_allclose(float(actual), expected, rtol=2e-11, atol=1e-8,
                               err_msg=str(label))


def scalar_replay(grades):
    result = []
    remembered_grade = 0
    consecutive_zeros = 0
    for value in grades:
        value = int(value)
        if value == 0:
            consecutive_zeros += 1
            if consecutive_zeros >= 2:
                remembered_grade = 0
            result.append(0)
        else:
            consecutive_zeros = 0
            result.append(value if value > remembered_grade else 0)
            remembered_grade = max(remembered_grade, value)
    return np.array(result, np.int8)


def outcome(notice):
    positives = np.flatnonzero(notice)
    if not len(positives):
        return 'silent', -1
    first = int(positives[0])
    return ('timely' if first < 11 else 'late'), first+3


def band(value):
    return '<2' if value < 2 else '2-4' if value < 4 else '4-8' if value < 8 else '>=8'


def expectation_statistics(target, present, ambient):
    variance = present + 16*ambient[..., None]
    flat_signal = target.reshape(16, -1)
    flat_variance = variance.reshape(16, -1)
    standardized = flat_signal/np.sqrt(flat_variance)
    best = standardized.max(1)
    selected = standardized.argmax(1)
    frame = np.arange(16)
    return dict(best=best, best_bin=selected,
                best_T=flat_signal[frame, selected],
                best_V=flat_variance[frame, selected],
                matched=np.linalg.norm(standardized, axis=1))


def cpu_geometric_visibility(source, poses, row):
    rays, weights = source.angular_rays(16)
    mass = weights.reshape(64, -1)
    mass = mass/mass.sum(1, keepdims=True)
    parameters, _ = source.nominal_parameters()
    sensor_module = __import__(source.synthesize_response.__module__)
    occupancies, count, fov = [], [], []
    for pose in poses:
        hits = source.raycast_boxes(pose[:3, 3], rays @ pose[:3, :3].T,
                                   [row['target_box'], *row['background_boxes']])
        target_hit = hits['object_id'].reshape(64, -1) == 0
        occupancies.append((target_hit*mass).sum(1).reshape(8, 8))
        ranges = hits['distance'].reshape(64, -1)
        raw_bin = np.floor((np.where(target_hit, ranges, parameters.range_zero_m)
                            - parameters.range_zero_m)/sensor_module.RAW_BIN_M)
        count.append(int((target_hit & (raw_bin >= 0) & (raw_bin < 128)).sum()))
        target_only = source.raycast_boxes(pose[:3, 3], rays @ pose[:3, :3].T,
                                          [row['target_box']])
        fov.append(int((target_only['object_id'] == 0).sum()))
    return np.asarray(occupancies), np.asarray(count), np.asarray(fov)


def smooth(raw):
    raw = np.asarray(raw, float)
    result = np.empty_like(raw)
    for frame in range(13):
        first = max(0, frame-4)
        numerator = np.zeros(raw.shape[:-2]+(2,), float)
        denominator = 0.
        for previous in range(first, frame+1):
            weight = 2.**(previous-first)
            numerator += weight*raw[..., previous, :]
            denominator += weight
        result[..., frame, :] = numerator/denominator
    return result


def dist_stats(values):
    if not values:
        return None
    values = np.asarray(values, float)
    return dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                    map(float, np.quantile(values, [0, .25, .5, .75, 1]))))


def same_stats(actual, values, label):
    expected = dist_stats(values)
    assert (actual is None) == (expected is None), label
    if expected is not None:
        for name, value in expected.items():
            near(actual[name], value, (label, name))


def lateral_gap(box):
    left, right = float(box['lo'][0]), float(box['hi'][0])
    return (left if left >= 0 else -right)-.30


def audit(out, cap):
    began = time.monotonic()
    attempts = out/'audit_attempts'
    attempts.mkdir(exist_ok=True)
    previous = sum(read(path)['seconds'] for path in attempts.glob('*.json'))
    receipt = dict(status='FAILED', source_sha256=sha(Path(__file__)), training=0,
                   model_forward=0, photon_sampling=0, threshold_changes=0,
                   protected_access=0, previous_seconds=previous)
    def check():
        if previous+time.monotonic()-began >= cap:
            raise TimeoutError('Independent audit cumulative CPU command-wall cap reached')
    try:
        assert out.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
        assert read(out/'analysis_receipt.json')['status'] == 'COMPLETE'
        manifest = read(out/'input_manifest.json')
        for path, digest in manifest.items():
            check()
            assert sha(ROOT/path) == digest, ('input SHA', path)
        events = csv_read(out/'event_ledger.csv')
        clips = csv_read(out/'clip_ledger.csv')
        frames = csv_read(out/'frame_ledger.csv')
        cross = csv_read(out/'cross_table.csv')
        summary = read(out/'summary.json')
        event_lookup = {(r['cohort'], int(r['scene_id']), int(r['replica']), r['height']): r for r in events}
        clip_lookup = {(r['cohort'], int(r['scene_id']), int(r['replica'])): r for r in clips}
        frame_lookup = {(r['cohort'], int(r['scene_id']), int(r['frame'])): r for r in frames}
        assert len(event_lookup) == len(events)
        assert len(clip_lookup) == len(clips)
        assert len(frame_lookup) == len(frames)
        source = {'e2e_hold': (ROOT/'artifacts.local/work/cnh-frozen-e2e-20261010', 'hold'),
                  'old_ideal_validation': (ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009', 'validation')}
        CPU_samples = []
        event_count = clip_count = field_count = scalar_streams = 0
        for cohort, (folder, split) in source.items():
            check()
            rows = read(folder/'scene_rows.json')[split]
            with np.load(folder/'data'/split/'geometry.npz', allow_pickle=False) as z:
                category, poses = z['category'], z['sensor']
            rawpath = folder/'data/hold/scores.npz' if cohort == 'e2e_hold' else folder/'baselines/validation_ideal.npz'
            with np.load(rawpath, allow_pickle=False) as z:
                rawm3, rawlocal = z['m3_raw'], z['local_raw']
            with np.load(out/f'{cohort}_frozen.npz', allow_pickle=False) as z:
                frozen = {name: z[name] for name in z.files}
            np.testing.assert_array_equal(category, frozen['category'])
            np.testing.assert_allclose(smooth(rawm3), frozen['m3'], rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(smooth(rawlocal), frozen['local'], rtol=1e-14, atol=1e-14)
            expected_grades = dict(m3=2*(frozen['m3'] >= .8557642486787612),
                old5=2*((frozen['m3'] >= .9404184587540165) | (frozen['local'] >= 4.625390338985158)))
            if cohort == 'e2e_hold':
                with np.load(folder/'hold_grades_notifications.npz', allow_pickle=False) as z:
                    key = z['keys'].tolist().index('2026100955/both')
                    expected_grades['both'] = z['grades'][key]
            else:
                with np.load(ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010/validation_grades.npz', allow_pickle=False) as z:
                    key = z['keys'].tolist().index('2026100955/score_current/c15_p64')
                    expected_grades['both'] = z['grades'][key]
            for arm, expected in expected_grades.items():
                np.testing.assert_array_equal(expected, frozen[arm+'_grade'])
                for n in range(len(rows)):
                    for k in range(expected.shape[1]):
                        for q in range(2):
                            notice = scalar_replay(expected[n, k, :, q])
                            np.testing.assert_array_equal(notice, frozen[arm+'_notice'][n, k, :, q])
                            scalar_streams += 1
            with np.load(out/f'{cohort}_expectations.npz', allow_pickle=False) as z:
                payload = {name: z[name] for name in z.files}
            np.testing.assert_array_equal(payload['sensor'], poses)
            np.testing.assert_array_equal(payload['scene_ids'], np.arange(len(rows)))
            assert payload['target'].min() >= -1e-8
            assert payload['present'].min() >= 0
            assert len(rows)*16 == sum(r['cohort'] == cohort for r in frames)
            for n, row in enumerate(rows):
                check()
                stats = expectation_statistics(payload['target'][n], payload['present'][n], payload['ambient'])
                occupancy = payload['occupancy'][n]
                visibility = payload['visible_counts'][n]
                fov = payload['fov_counts'][n]
                best = stats['best']
                for f in range(16):
                    record = frame_lookup[cohort, n, f]
                    for name, value in dict(best_bin_snr=best[f], target_counts=payload['target'][n, f].sum(),
                        best_snr_flatbin=stats['best_bin'][f], visible_counts=visibility[f], fov_counts=fov[f],
                        max_zone_visible_fraction=occupancy[f].max(), total_zone_visible_fraction=occupancy[f].sum()/64).items():
                        near(record[name], value, (cohort, n, f, name)); field_count += 1
                    assert record['physical_key'] == row['physical_key']
                target_gap = lateral_gap(row['target_box']) if row['presence'] else None
                scene_gap = min(lateral_gap(box) for box in row['boxes'])
                vv, fv = np.flatnonzero(visibility), np.flatnonzero(fov)
                exits = [f for f in range(1, 16) if fov[f-1] > 0 and fov[f] == 0]
                visible_exits = [f for f in range(1, 16) if visibility[f-1] > 0 and visibility[f] == 0]
                timely_vis = np.flatnonzero(visibility[3:14])
                shared = dict(max_best_bin_snr=best[3:14].max(), frames_snr_ge2=(best[3:14] >= 2).sum(),
                    frames_snr_ge4=(best[3:14] >= 4).sum(), frames_snr_ge8=(best[3:14] >= 8).sum(),
                    first_visible_frame=int(vv[0]) if len(vv) else -1,
                    first_visible_timely_frame=int(timely_vis[0]+3) if len(timely_vis) else -1,
                    exit_fov_frame=exits[0] if exits else -1, exit_visible_frame=visible_exits[0] if visible_exits else -1,
                    permanent_exit_fov_frame=int(fv[-1]+1) if len(fv) and fv[-1] < 15 else -1,
                    fov_reentry_after_exit=int(bool(exits and np.any(fov[exits[0]+1:]))),
                    visible_timely_frames=len(timely_vis), max_zone_visible_fraction=occupancy[3:14].max(),
                    max_total_zone_visible_fraction=occupancy[3:14].sum((-1, -2)).max()/64,
                    max_visible_zone_share=(occupancy[3:14] > 0).mean((-1, -2)).max(), min_scene_clearance_m=scene_gap)
                kind = 'clear' if all(category[n] == 'clear') else 'purepass' if any(category[n] == 'pass') and not any(category[n] == 'contact') else None
                for k in range(frozen['m3'].shape[1]):
                    record_list = []
                    for q, height in enumerate(('HEAD', 'BODY')):
                        if category[n, q] != 'contact':
                            continue
                        event_count += 1
                        record = event_lookup[cohort, n, k, height]
                        record_list.append(record)
                        for arm in expected_grades:
                            expected_outcome, first_frame = outcome(frozen[arm+'_notice'][n, k, :, q])
                            assert record[arm+'_outcome'] == expected_outcome
                            assert int(record[arm+'_first_frame']) == first_frame
                        for name, value in dict(m3_margin=frozen['m3'][n, k, :11, q].max()-.8557642486787612,
                            m3_raised_margin=frozen['m3'][n, k, :11, q].max()-.9404184587540165,
                            local_margin=frozen['local'][n, k, :11, q].max()-4.625390338985158).items():
                            near(record[name], value, (cohort, n, k, height, name)); field_count += 1
                    if kind:
                        clip_count += 1
                        record = clip_lookup[cohort, n, k]
                        record_list.append(record)
                        assert record['kind'] == kind
                        for arm in expected_grades:
                            joint = frozen[arm+'_notice'][n, k].max(-1)
                            for suffix, value in [('notifications', sum(joint > 0)), ('light', sum(joint == 1)),
                                                  ('strong', sum(joint == 2)), ('max_grade', max(joint))]:
                                assert int(record[arm+'_'+suffix]) == value
                            for q, height in enumerate(('HEAD', 'BODY')):
                                notice = frozen[arm+'_notice'][n, k, :, q]
                                name = arm+'_'+height+'_notifications'
                                if name in record:
                                    assert int(record[name]) == np.count_nonzero(notice)
                    for record in record_list:
                        assert record['physical_key'] == row['physical_key']
                        assert record['snr_band'] == band(shared['max_best_bin_snr'])
                        assert record['shape'] == row['shape_family']
                        assert record['background_family'] == row['background_family']
                        for name, value in shared.items():
                            near(record[name], value, (cohort, n, k, name)); field_count += 1
                        if target_gap is None:
                            assert record['signed_target_clearance_m'] == ''
                        else:
                            near(record['signed_target_clearance_m'], target_gap, (cohort, n, 'gap'))
                if not row['presence']:
                    assert not payload['target'][n].any()
                    np.testing.assert_array_equal(payload['present'][n], payload['absent'][n])
            import cnh_counterfactual_data_dev as D
            scene_module, cpu, _, _ = D.frozen_imports()
            probe_frames = np.array([3, 13])
            for height in range(2):
                check()
                n = min(i for i, row in enumerate(rows) if row['presence'] and category[i, height] == 'contact')
                row = rows[n]
                opaque = cpu.expected(dict(poses=poses[probe_frames], boxes=[dict(row['target_box'], rho=0), *row['background_boxes']]))
                present = cpu.expected(dict(poses=poses[probe_frames], boxes=[row['target_box'], *row['background_boxes']]))
                target = present['expectation']-opaque['expectation']
                np.testing.assert_allclose(payload['present'][n, probe_frames], present['expectation'], rtol=2e-11, atol=1e-8)
                np.testing.assert_allclose(payload['target'][n, probe_frames], target, rtol=2e-11, atol=1e-8)
                np.testing.assert_array_equal(payload['ambient'][probe_frames], present['ambient'])
                occ, vis, fov = cpu_geometric_visibility(scene_module, poses[probe_frames], row)
                np.testing.assert_allclose(payload['occupancy'][n, probe_frames], occ, rtol=2e-11, atol=1e-9)
                np.testing.assert_array_equal(payload['visible_counts'][n, probe_frames], vis)
                np.testing.assert_array_equal(payload['fov_counts'][n, probe_frames], fov)
                CPU_samples.append(dict(cohort=cohort, scene_id=n, height=('HEAD', 'BODY')[height], frames=probe_frames.tolist(),
                    present_max_abs=float(abs(payload['present'][n, probe_frames]-present['expectation']).max()),
                    target_max_abs=float(abs(payload['target'][n, probe_frames]-target).max())))
            del payload, frozen
            print('MISS_AUDIT_COHORT', cohort, 'seconds', round(time.monotonic()-began, 3), flush=True)
        assert event_count == len(events) and clip_count == len(clips)
        # Independent tuple-key census, retaining physical-scene IDs per cell.
        dimensions = ('all', 'shape', 'thickness_m', 'rho', 'side', 'signed_target_clearance_m',
                      'background_family', 'shape_rho', 'shape_size_rho_background')
        census = defaultdict(list)
        for event in events:
            for dimension in dimensions:
                group = 'all' if dimension == 'all' else '/'.join(event[k] for k in ('shape', 'rho')) if dimension == 'shape_rho' else '/'.join(event[k] for k in ('shape', 'thickness_m', 'rho', 'background_family')) if dimension == 'shape_size_rho_background' else event[dimension]
                census[event['cohort'], event['height'], dimension, group].append(event)
        cross_keys = set()
        for record in cross:
            check()
            key = (record['cohort'], record['height'], record['dimension'], record['group'])
            sub = census[key]
            cell = [e for e in sub if e['snr_band'] == record['snr_band'] and e[record['arm']+'_outcome'] == record['outcome']]
            assert int(record['events']) == len(cell)
            assert int(record['physical_scenes']) == len({e['physical_key'] for e in cell})
            assert int(record['group_events']) == len(sub)
            assert int(record['group_physical_scenes']) == len({e['physical_key'] for e in sub})
            fullkey = (*key, record['snr_band'], record['arm'], record['outcome'])
            assert fullkey not in cross_keys
            cross_keys.add(fullkey)
        assert len(cross_keys) == len(census)*4*3*3
        for cohort, heights in summary['contact'].items():
            for height, actual in heights.items():
                sub = [e for e in events if e['cohort'] == cohort and e['height'] == height]
                miss = [e for e in sub if e['old5_outcome'] != 'timely']
                hi = [e for e in miss if float(e['max_best_bin_snr']) >= 4]
                lo = [e for e in miss if float(e['max_best_bin_snr']) < 2]
                expected = dict(events=len(sub), physical_scenes=len({e['physical_key'] for e in sub}), misses=len(miss),
                    high=len(hi), low=len(lo), intermediate=len(miss)-len(hi)-len(lo),
                    high_scene_any=len({e['physical_key'] for e in hi}), low_scene_any=len({e['physical_key'] for e in lo}))
                for name, value in expected.items():
                    assert actual[name] == value, (cohort, height, name)
                for arm in ('m3', 'old5', 'both'):
                    counts = Counter(e[arm+'_outcome'] for e in sub)
                    for result in ('timely', 'late', 'silent'):
                        assert actual['outcomes'][arm][result] == counts[result]
        for actual in summary['shape_groups']:
            sub = [e for e in events if e['cohort'] == actual['cohort'] and e['height'] == actual['height'] and e['shape'] == actual['shape']]
            hi = [e for e in sub if e['old5_outcome'] != 'timely' and float(e['max_best_bin_snr']) >= 4]
            expected = dict(denominator=len(sub), physical_scenes=len({e['physical_key'] for e in sub}), high_miss=len(hi),
                high_miss_scenes=len({e['physical_key'] for e in hi}), high_late=sum(e['old5_outcome'] == 'late' for e in hi),
                high_silent=sum(e['old5_outcome'] == 'silent' for e in hi),
                low_miss=sum(e['old5_outcome'] != 'timely' and float(e['max_best_bin_snr']) < 2 for e in sub))
            for name, value in expected.items():
                assert actual[name] == value, ('shape', actual['shape'], name)
            for name, field in [('m3_margin', 'm3_margin'), ('local_margin', 'local_margin'), ('high_snr', 'max_best_bin_snr'), ('ge4_frames', 'frames_snr_ge4')]:
                same_stats(actual[name], [float(e[field]) for e in hi], ('shape', name))
        for actual in summary['cost']:
            sub = [c for c in clips if c['cohort'] == actual['cohort'] and c['kind'] == actual['kind']]
            arm = actual['arm']
            selected = [c for c in sub if bool(int(c[arm+'_notifications'])) == actual['notified']]
            present = [c for c in selected if int(c['presence'])]
            distances = [float(c['signed_target_clearance_m']) for c in present]
            expected = dict(clips=len(selected), denominator=len(sub), physical_scenes=len({c['physical_key'] for c in selected}),
                notifications=sum(int(c[arm+'_notifications']) for c in selected), light=sum(int(c[arm+'_light']) for c in selected),
                strong=sum(int(c[arm+'_strong']) for c in selected), present_clips=len(present), absent_clips=len(selected)-len(present),
                near10cm=sum(d < .1 for d in distances))
            for name, value in expected.items():
                assert actual[name] == value, ('cost', actual['cohort'], arm, name)
            same_stats(actual['distance'], distances, ('cost', 'distance'))
            same_stats(actual['snr'], [float(c['max_best_bin_snr']) for c in present], ('cost', 'snr'))
            for distance, stratum in actual['distance_strata'].items():
                members = [c for c in present if float(c['signed_target_clearance_m']) == float(distance)]
                assert stratum['clips'] == len(members)
                assert stratum['notifications'] == sum(int(c[arm+'_notifications']) for c in members)
        supplement_cells = 0
        if (out/'supplement.json').exists():
            supplement = read(out/'supplement.json')
            for key, actual in supplement['details'].items():
                cohort, height = key.split('/')
                miss = [e for e in events if e['cohort'] == cohort and e['height'] == height and e['old5_outcome'] != 'timely']
                hi = [e for e in miss if float(e['max_best_bin_snr']) >= 4]
                lo = [e for e in miss if float(e['max_best_bin_snr']) < 2]
                assert actual['high_ge4_frames'] == dict(Counter(e['frames_snr_ge4'] for e in hi))
                for name, value in dict(high_both_timely=sum(e['both_outcome'] == 'timely' for e in hi),
                    low_both_timely=sum(e['both_outcome'] == 'timely' for e in lo),
                    low_zero_visible=sum(int(e['visible_timely_frames']) == 0 for e in lo),
                    high_late=sum(e['old5_outcome'] == 'late' for e in hi),
                    low_late=sum(e['old5_outcome'] == 'late' for e in lo)).items():
                    assert actual[name] == value, ('supplement', key, name)
                low_counts = Counter((e['shape'], e['rho'], e['thickness_m']) for e in lo)
                actual_low = {(e['shape'], e['rho'], e['thickness_m']): e['misses'] for e in actual['low_groups']}
                assert actual_low == dict(low_counts)
                supplement_cells += 1
            for actual in supplement['fine_groups']:
                sub = [e for e in events if all(e[k] == actual[k] for k in ('cohort', 'height', 'shape', 'rho', 'thickness_m'))]
                hi = [e for e in sub if e['old5_outcome'] != 'timely' and float(e['max_best_bin_snr']) >= 4]
                assert actual['high_miss'] == len(hi)
                assert actual['den'] == len(sub)
                assert actual['scenes'] == len({e['physical_key'] for e in hi})
                assert actual['backgrounds'] == dict(Counter(e['background_family'] for e in hi))
                supplement_cells += 1
            for actual in supplement['height_cost']:
                sub = [c for c in clips if c['cohort'] == actual['cohort'] and c['kind'] == 'purepass']
                arm = actual['arm']
                assert actual['query_notifications'] == sum(int(c[arm+'_HEAD_notifications'])+int(c[arm+'_BODY_notifications']) for c in sub)
                assert actual['opposite_target_height_notifications'] == sum(int(c[arm+'_'+('BODY' if c['target_height'] == 'HEAD' else 'HEAD')+'_notifications']) for c in sub)
                supplement_cells += 1
        check()
        receipt.update(status='PASS', input_files_verified=len(manifest), events_verified=len(events), clips_verified=len(clips),
            frame_rows_verified=len(frames), metric_fields_verified=field_count, scalar_streams_verified=scalar_streams,
            cross_table_cells_verified=len(cross), summary_contact=summary['contact'], cpu_expectation_samples=CPU_samples,
            supplement_rows_verified=supplement_cells,
            checks=['frozen raw score smoothing and fixed inherited grades', 'independent scalar gap1 replay',
                'oracle T / sqrt(mu_present + 16ambient)', 'all frame and contact SNR bands and timing',
                'physical-scene identity and correlated replica denominators', 'lateral signed AABB corridor clearance',
                'full cross-table contact census and high/low group counts', 'all notification grades/cost/distance/SNR distributions',
                'four prespecified CPU expectation and independent geometric visibility recomputations'],
            limits=['Consumed controlled Development reuse; descriptive only', 'CPU probes verify finite quadrature implementation, not hardware detectability',
                    'SNR bands and saved notification outcomes occupy different statistic spaces; no threshold selected'])
        print('MISS_AUDIT_PASS', len(events), len(clips), len(frames), len(cross), flush=True)
    except BaseException as error:
        receipt.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['seconds'] = time.monotonic()-began
        receipt['cumulative_seconds'] = previous+receipt['seconds']
        save_new(attempts/f'audit_{time.time_ns()}.json', receipt)
        if receipt['status'] == 'PASS':
            save_new(out/'audit_receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--cpu-cap-seconds', type=float, default=120)
    args = parser.parse_args()
    audit(args.out.resolve(), args.cpu_cap_seconds)
