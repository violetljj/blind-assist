"""Independent CPU recount of frozen spatial-notice evidence.

Shares the run's cumulative analyze budget. No inference, rendering, threshold
search objective, or calls to candidate event/summary/decision implementations.
The geometric masks are accepted as evaluator inputs; this does not independently
prove object attribution or real-world localization.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
import numpy as np
import cnh_sector_notice_dev as S


def manual_point(values, budget):
    flat = np.asarray(values, float).ravel()
    assert len(flat) and np.isfinite(flat).all()
    assert 0 <= budget <= len(flat)
    # Descending unique levels count entire ties; the next rejected level
    # determines the smallest feasible representable threshold.
    levels, counts = np.unique(flat, return_counts=True)
    used = 0
    boundary = None
    for level, count in zip(levels[::-1], counts[::-1]):
        if used + int(count) > budget:
            boundary = float(level)
            break
        used += int(count)
    theta = -np.inf if boundary is None else np.nextafter(boundary, np.inf)
    assert int((flat >= theta).sum()) == used
    return dict(threshold=float(theta), actual_notices=used,
                unused_notices=budget-used, target_notices=budget,
                boundary_tie_count=0 if boundary is None else int((flat == boundary).sum()))


def manual_notices(score, theta, policy):
    if policy == 'all_sectors':
        return score >= theta
    out = np.zeros(score.shape, bool)
    # Explicit comparisons preserve the frozen LEFT-first exact tie ordering.
    choice = np.zeros(score.shape[:2], int)
    best = score[..., 0].copy()
    for label in (1, 2):
        update = score[..., label] > best
        choice[update] = label
        best[update] = score[..., label][update]
    for label in range(3):
        out[..., label] = (choice == label) & (best >= theta)
    return out


def manual_baseline(alarm, contact, deadline):
    main = np.zeros(len(contact), bool)
    full = main.copy()
    for row in np.flatnonzero(contact):
        for frame in range(int(deadline[row])+1):
            if alarm[row, frame]:
                full[row] = True
                if frame >= 2:
                    main[row] = True
    return main, full


def manual_events(notice, truth, contact, deadline):
    n = len(contact)
    fields = {k: np.zeros(n, bool) for k in
              ('timely', 'first_supported', 'first_clean', 'first_unique', 'ever_supported')}
    fields['first'] = np.full(n, -1, int)
    fields['first_labels'] = np.zeros((n, 3), bool)
    correct = wrong = ambiguous = supported_all = unsupported_all = 0
    full = np.zeros(n, bool)
    for row in np.flatnonzero(contact):
        for frame in range(int(deadline[row])+1):
            selected = [label for label in range(3) if notice[row, frame, label]]
            if selected:
                full[row] = True
            if frame < 2:
                continue
            hits = sum(bool(truth[row, frame, label]) for label in selected)
            supported_all += hits
            unsupported_all += len(selected)-hits
            if hits:
                fields['ever_supported'][row] = True
            if not selected or fields['first'][row] >= 0:
                continue
            fields['first'][row] = frame
            fields['first_labels'][row] = notice[row, frame]
            fields['timely'][row] = True
            fields['first_supported'][row] = hits > 0
            fields['first_clean'][row] = hits > 0 and hits == len(selected)
            fields['first_unique'][row] = (fields['first_clean'][row] and
                                           int(truth[row, frame].sum()) == 1)
            correct += hits
            wrong += len(selected)-hits
            ambiguous += int(truth[row, frame].sum() > 1)
    counts = {key: int(fields[key].sum()) for key in
              ('first_supported', 'first_clean', 'first_unique')}
    counts.update(timely_main=int(fields['timely'].sum()),
                  timely_including_startup=int(full.sum()),
                  later_or_first_supported=int(fields['ever_supported'].sum()),
                  first_timed_command_correct_labels=correct,
                  first_timed_command_incorrect_labels=wrong,
                  first_ambiguous_truth_events=ambiguous,
                  contact_predeadline_main_supported_notices=supported_all,
                  contact_predeadline_main_unsupported_notices=unsupported_all)
    return fields, counts


def manual_pair(candidate, baseline, contact):
    rescued = lost = 0
    for row in np.flatnonzero(contact):
        rescued += int(candidate[row] and not baseline[row])
        lost += int(baseline[row] and not candidate[row])
    return dict(rescued=rescued, lost=lost, diff=rescued-lost)


def manual_smooth(raw):
    raw = np.asarray(raw, np.float64)
    smoothed = np.zeros_like(raw)
    for frame in range(13):
        length = min(frame+1, 5)
        for lag in range(length):
            smoothed[..., frame, :] += (raw[..., frame-lag, :] *
                                        (1 << (length-lag-1)) / ((1 << length)-1))
    return smoothed


def rotation_y(degrees):
    angle = np.deg2rad(degrees); c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def audit():
    started = time.monotonic()
    with S.stage('analyze') as check:
        result = S.B.A.read(S.OUT/'result.json')
        assert result['status'] == 'COMPLETE'
        with np.load(S.OUT/'ledger.npz') as f:
            ledger = {k: f[k] for k in f.files}
        with np.load(S.E.OUT/'ledger.npz') as f:
            baseline = {k: f[k] for k in ('score', 'unit', 'config', 'contact', 'control', 'deadline')}
        for key in ('unit', 'config', 'contact', 'control', 'deadline'):
            np.testing.assert_array_equal(ledger[key], baseline[key])
        assert ledger['score'].shape == (5, 2, 3840, 13, 3)
        assert ledger['truth'].shape == (5, 3840, 13, 3)
        contact, control, deadline = (ledger[k] for k in ('contact', 'control', 'deadline'))
        assert int(contact.sum()) == 229 and int(control.sum()) == 384
        assert not (contact & control).any()
        assert np.all((deadline[contact] >= 0) & (deadline[contact] <= 12))
        assert not ledger['truth'][:, ~contact].any()
        assert S.B.A.sha(S.OUT/'ledger.npz') == result['ledger_sha256']
        parent = S.B.A.read(S.E.OUT/'result.json')
        receipts = S.B.A.read(S.OUT/'prepare_result.json')
        assert S.B.A.sha(Path(__file__).with_name('cnh_sector_geometry_dev.py')) == receipts['geometry_source_sha256']
        raw_arrays = query_frames = 0
        max_score_delta = 0.
        for unit in S.Y.UNITS:
            check()
            path = S.OUT/'units'/f'unit{unit}.npz'
            assert S.B.A.sha(path) == result['input_units_sha256'][str(unit)]
            rows = np.flatnonzero(ledger['unit'] == unit)
            np.testing.assert_array_equal(ledger['config'][rows], np.arange(40))
            with np.load(path) as saved:
                for ni, name in enumerate(S.Y.NAMES):
                    raw = saved[name+'/raw']
                    assert raw.shape == (3, 3, 40, 13, 2)
                    smooth = manual_smooth(raw)
                    for si in range(2):
                        branch_ids = (0,) if si == 0 else (1, 2)
                        expected = np.max(smooth[:, branch_ids], axis=(1, 4)).transpose(1, 2, 0)
                        target = ledger['score'][ni, si][rows]
                        max_score_delta = max(max_score_delta, float(np.abs(expected-target).max()))
                        np.testing.assert_allclose(expected, target, atol=1e-12, rtol=0)
                    raw_arrays += 1
                # Query-axis/origin checks on three spaced source units only.
                if unit not in (S.Y.UNITS[0], S.Y.UNITS[len(S.Y.UNITS)//2], S.Y.UNITS[-1]):
                    continue
                with np.load(S.Y.M.OUT/'units'/f'unit{unit}.npz') as native, np.load(S.Y.OUT/'units'/f'unit{unit}.npz') as physical:
                    for name in S.Y.NAMES:
                        data = native if name == 'zero' else physical
                        prefix = '' if name == 'zero' else name+'/'
                        noisy, sensor = data[prefix+'noisy'], data[prefix+'sensor']
                        displacement = sensor[..., :3, 3]-native['travel'][..., :3, 3]
                        for ai, angle in enumerate((-20., 0., 20.)):
                            q = saved[name+'/query'][ai]
                            pitch = np.deg2rad(-10.)
                            pitch_matrix = np.array([[1., 0., 0.], [0., np.cos(pitch), -np.sin(pitch)], [0., np.sin(pitch), np.cos(pitch)]])
                            np.testing.assert_allclose(q[..., :3, :3], np.broadcast_to(rotation_y(-angle) @ pitch_matrix, q[..., :3, :3].shape), atol=1e-12, rtol=0)
                            for config in range(40):
                                for frame in range(16):
                                    yaw = np.rad2deg(np.arctan2(noisy[config, frame, 0, 2], noisy[config, frame, 2, 2]))
                                    reconstructed = rotation_y(yaw+angle) @ q[config, frame, :3, 3]
                                    np.testing.assert_allclose(reconstructed, displacement[config, frame], atol=1e-12, rtol=0)
                            query_frames += 40*16
        point_count = events_count = paired_count = 0
        for ni, name in enumerate(S.Y.NAMES):
            for si, sensor in enumerate(('single', 'dual')):
                check(); key = name+'/'+sensor; cell = result['metrics'][key]
                e1 = baseline['score'][ni, si, 1]
                old_theta = parent['metrics'][key]['e1']['threshold']['threshold']
                budget = int((e1 >= old_theta).sum())
                assert int((e1[control] >= old_theta).sum()) == 124
                baselines = {}
                for arm, index in (('e1', 1), ('ema', 4)):
                    data = baseline['score'][ni, si, index]
                    point = manual_point(data, budget)
                    saved_point = cell[arm]['working_point']
                    for field in ('threshold', 'target_notices', 'actual_notices', 'unused_notices'):
                        assert saved_point[field] == point[field], (key, arm, field)
                    alarm = data >= point['threshold']
                    main, full = manual_baseline(alarm, contact, deadline)
                    np.testing.assert_array_equal(main, ledger[key+'/'+arm+'/timely'])
                    assert cell[arm]['timely_main'] == int(main.sum())
                    assert cell[arm]['timely_including_startup'] == int(full.sum())
                    baselines[arm] = main
                    point_count += 1
                reference = 'ema' if baselines['ema'].sum() >= baselines['e1'].sum() else 'e1'
                for policy in ('top1', 'all_sectors'):
                    rec = cell[policy]
                    score = ledger['score'][ni, si]
                    values = score.max(-1) if policy == 'top1' else score
                    point = manual_point(values, budget)
                    for field in point:
                        assert rec['working_point'][field] == point[field], (key, policy, field)
                    notice = manual_notices(score, point['threshold'], policy)
                    np.testing.assert_array_equal(notice, ledger[key+'/'+policy+'/notice'])
                    fields, counts = manual_events(notice, ledger['truth'][ni], contact, deadline)
                    for field, expected in fields.items():
                        np.testing.assert_array_equal(expected, ledger[key+'/'+policy+'/'+field])
                    for field, value in counts.items():
                        assert rec[field] == value, (key, policy, field, rec[field], value)
                    for label, mask in (('total_notices', np.ones(3840, bool)), ('contact_window_notices', contact), ('strict_clear_notices', control), ('other_window_notices', ~contact & ~control)):
                        assert rec[label] == int(notice[mask].sum())
                    assert rec['reference_arm'] == reference
                    assert rec['first_clean_minus_reference'] == counts['first_clean']-int(baselines[reference].sum())
                    for arm, base in baselines.items():
                        assert rec['against'][arm]['detection'] == manual_pair(fields['timely'], base, contact)
                        assert rec['against'][arm]['first_clean'] == manual_pair(fields['first_clean'], base, contact)
                        paired_count += 2
                    assert rec['label_removed_timely'] == counts['timely_main']
                    n = counts['timely_main']
                    for field, numerator in (('first_supported_fraction', counts['first_supported']), ('first_clean_fraction', counts['first_clean'])):
                        expected = None if not n else numerator/n
                        assert rec[field] == expected
                    point_count += 1; events_count += int(contact.sum())
        for sensor in ('single', 'dual'):
            diffs = [result['metrics'][name+'/'+sensor]['top1']['first_clean_minus_reference'] for name in S.Y.NAMES[1:]]
            if min(diffs) >= 0 and max(diffs) > 0:
                outcome = 'SUPPORT_DEDUP_CHECK'
            elif max(diffs) <= 0 and min(diffs) < 0:
                outcome = 'LOWER_CURRENT_SECTOR_READOUT_PRIORITY'
            else:
                outcome = 'RETAIN_MIXED_OR_ZERO_NO_AUTOMATIC_FOLLOWUP'
            assert result['decisions'][sensor] == dict(diffs=diffs, decision=outcome)
        csv_count = 0
        lookup = {(int(u), int(c)): i for i, (u, c) in enumerate(zip(ledger['unit'], ledger['config']))}
        with (S.OUT/'events.csv').open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                index = lookup[int(row['unit']), int(row['config'])]
                prefix = row['name']+'/'+row['sensor']+'/'+row['policy']+'/'
                first = int(ledger[prefix+'first'][index])
                assert int(row['first_output']) == first
                assert row['first_input_frame'] == ('' if first < 0 else str(first+3))
                assert int(row['deadline']) == int(deadline[index])
                for key in ('timely', 'first_clean', 'first_unique', 'ever_supported'):
                    assert row[key] == str(bool(ledger[prefix+key][index]))
                csv_count += 1
        assert csv_count == 5*2*2*229
        S.B.save(S.OUT/'audit_result.json', dict(status='PASS', seconds=time.monotonic()-started,
            raw_arrays=raw_arrays, query_frames_sampled=query_frames, maximum_score_difference=max_score_delta,
            working_points=point_count, event_recounts=events_count, paired_recounts=paired_count,
            csv_rows=csv_count, configurations_decisions=2,
            source_sha256=S.B.A.sha(Path(__file__)), ledger_sha256=S.B.A.sha(S.OUT/'ledger.npz'),
            limits='Geometry masks are evaluator inputs. Query checks use3source units; no independent GPU/rerender or user-response validation.'))
        print('SECTOR_AUDIT_PASS', json.dumps(dict(points=point_count, events=events_count, csv_rows=csv_count)), flush=True)


if __name__ == '__main__':
    audit()
