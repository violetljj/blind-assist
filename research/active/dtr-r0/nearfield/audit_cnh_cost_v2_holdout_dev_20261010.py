"""Independent saved-output audit for cost-v2; no rendering or inference.

Scalar gap1 replay, evaluator geometry and full negative score-tie sweep are
implemented here without importing the production evaluator. K stays within
physical scene. Inputs are only this run and old inventory metadata/model hashes.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SEEDS = (2026100955, 2026100956, 2026100957)
HEIGHTS = ('HEAD', 'BODY')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def scalar_gap1(grades):
    grades = np.asarray(grades)
    assert grades.shape[-2:] == (13, 2)
    assert np.isin(grades, (0, 1, 2)).all()
    flat = grades.reshape(-1, 13, 2)
    result = np.zeros_like(flat)
    for clip in range(len(flat)):
        for query in range(2):
            previous_peak, consecutive_zeros = 0, 0
            for frame in range(13):
                current = int(flat[clip, frame, query])
                if current == 0:
                    consecutive_zeros += 1
                    if consecutive_zeros == 2:
                        previous_peak = 0
                else:
                    consecutive_zeros = 0
                    if current > previous_peak:
                        result[clip, frame, query] = current
                    previous_peak = max(previous_peak, current)
    return result.reshape(grades.shape)


def smooth(raw):
    out = np.empty(raw.shape, dtype=float)
    for t in range(13):
        start = max(0, t - 4)
        weights = np.array([2. ** i for i in range(t - start + 1)])
        out[..., t, :] = (raw[..., start:t+1, :] * weights[:, None]).sum(-2) / weights.sum()
    return out


def geometry(rows, sensor, category):
    import cnh_counterfactual_data_dev as D
    keys = set()
    counts = {}
    for n, row in enumerate(rows):
        key = hashlib.sha256(json.dumps(row['boxes'], sort_keys=True,
            separators=(',', ':')).encode('utf8')).hexdigest()
        assert key == row['physical_key'] and key not in keys
        keys.add(key)
        box = row['target_box']
        gaps = [max(float(box['lo'][0]) - float(sensor[f, 0, 3]) - .30,
                    float(sensor[f, 0, 3]) - float(box['hi'][0]) - .30)
                for f in D.FRAMES]
        gap = min(gaps)
        assert abs(gap - row['lateral_gap_m']) < 1e-12
        placement = row['placement']
        expected = ['clear', 'clear']
        expected[row['group']] = placement
        assert category[n].tolist() == expected
        for f in D.FRAMES:
            assert D.category_boxes(row['background_boxes'], sensor[f, :3, 3]) == ['clear', 'clear']
            legacy = expected.copy()
            if placement == 'pass' and gap > .10:
                legacy[row['group']] = 'clear'
            assert D.category_boxes(row['boxes'], sensor[f, :3, 3]) == legacy
        if placement == 'pass':
            layer = ('0-5' if gap <= .05 else '5-10' if gap <= .10 else
                     '10-20' if gap <= .20 else '20-35')
            assert str(row['pass_layer']).replace('cm', '') == layer
            assert 0 < gap <= .35
        counts[placement] = counts.get(placement, 0) + 1
    return keys, counts


def masks(rows):
    placement = np.array([r['placement'] for r in rows])
    gap = np.array([r['lateral_gap_m'] for r in rows])
    passed = placement == 'pass'
    return dict(nearpass=passed & (gap <= .10), farpass=passed & (gap > .10),
        pass_0_5=passed & (gap <= .05), pass_5_10=passed & (gap > .05) & (gap <= .10),
        pass_10_20=passed & (gap > .10) & (gap <= .20),
        pass_20_35=passed & (gap > .20) & (gap <= .35), clear=placement == 'clear')


def raw_cost(emissions, selected, near_weight=.25):
    all_joint = emissions.max(-1)
    joint = all_joint[selected]
    light, strong = int((joint == 1).sum()), int((joint == 2).sum())
    return dict(light=light, strong=strong, notifications=light+strong,
        weighted=near_weight*light+strong, scene_denominator=int(selected.sum()),
        clip_denominator=int(joint.shape[0] * joint.shape[1]),
        clips_with_any=int((joint > 0).any(-1).sum()),
        clips_with_light=int((joint == 1).any(-1).sum()), clips_with_strong=int((joint == 2).any(-1).sum()),
        scenes_with_any=int((joint > 0).any((1,2)).sum()),
        scene_counts=[dict(scene=int(n), light=int((all_joint[n] == 1).sum()),
            strong=int((all_joint[n] == 2).sum()),
            clips_with_any_notification=int((all_joint[n] > 0).any(-1).sum()))
            for n in np.flatnonzero(selected)])


def total_cost(emissions, rows, near_weight=.25):
    selected = masks(rows)
    return (raw_cost(emissions, selected['nearpass'], near_weight)['weighted'] +
            raw_cost(emissions, selected['farpass'])['notifications'] +
            raw_cost(emissions, selected['clear'])['notifications'])


def enumerate_ties(both, old5, margin, rows, deadline):
    """Ascending ties, scalar replay of affected clips, no monotone assumption."""
    selected = np.array([r['placement'] != 'contact' for r in rows])
    light_weight = np.repeat(np.array([.25 if r['placement'] == 'pass' and
        r['lateral_gap_m'] <= .10 else 1. for r in rows])[selected], both.shape[1])
    grades = both[selected].reshape(-1, 13, 2).copy()
    baseline = old5[selected].reshape(-1, 13, 2)
    margins = margin[selected].reshape(-1, 13, 2)
    def weighted(e, clip_ids=None):
        joint = e.max(-1)
        w = light_weight if clip_ids is None else light_weight[clip_ids]
        return ((joint == 1).sum(-1) * w + (joint == 2).sum(-1)).astype(float)
    emitted = scalar_gap1(grades)
    byclip = weighted(emitted)
    cap = float(weighted(scalar_gap1(baseline)).sum())
    cost = float(byclip.sum())
    eligible = (grades > 0) & (baseline == 0) & np.isfinite(margins)
    clip, frame, query = np.where(eligible)
    values = margins[eligible]
    order = np.argsort(values, kind='stable')
    values, clip, frame, query = [v[order] for v in (values, clip, frame, query)]
    assert not len(values) or values.min() >= 0
    boundaries = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)] if len(values) else np.array([0])
    curve = [dict(tau=0., weighted=cost, feasible=cost <= cap)]
    selected_tau = 0. if cost <= cap else None
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        assert time.monotonic() < deadline, 'Audit CPU budget reached'
        affected = np.unique(clip[begin:end])
        previous = byclip[affected].sum()
        grades[clip[begin:end], frame[begin:end], query[begin:end]] = 0
        byclip[affected] = weighted(scalar_gap1(grades[affected]), affected)
        cost += float(byclip[affected].sum() - previous)
        tau = float(np.nextafter(values[begin], np.inf))
        feasible = cost <= cap
        curve.append(dict(tau=tau, weighted=cost, feasible=feasible))
        if selected_tau is None and feasible:
            selected_tau = tau
    np.testing.assert_array_equal(grades, baseline)
    assert cost == cap
    curve.append(dict(tau=None, weighted=cap, feasible=True))
    return np.inf if selected_tau is None else selected_tau, cap, curve


def first(emissions):
    result = np.full(emissions.shape[:2] + (2,), -1, dtype=int)
    for frame in range(13):
        positive = emissions[:, :, frame, :] > 0
        result[(result < 0) & positive] = frame
    return result


def paired(emission, baseline, category, rows, group='all'):
    a, b = first(baseline), first(emission)
    result = {}
    for q, height in enumerate(HEIGHTS):
        selected = category[:, q] == 'contact'
        if group == 'dark_thin':
            selected &= np.array([r['dark_thin'] for r in rows], bool)
        elif group == 'sign_edge':
            selected &= np.array([r['shape_family'] == 'sign_edge' for r in rows])
        at, bt = ((v[:, :, q] >= 0) & (v[:, :, q] < 11) for v in (a, b))
        bf = b[selected, :, q]
        rescued, lost = (~at & bt)[selected], (at & ~bt)[selected]
        result[height] = dict(denominator=int(selected.sum()*emission.shape[1]),
            scene_denominator=int(selected.sum()), baseline_timely=int(at[selected].sum()),
            candidate_timely=int(bt[selected].sum()), rescue=int((~at & bt)[selected].sum()),
            loss=int((at & ~bt)[selected].sum()),
            net=int(bt[selected].sum()-at[selected].sum()), late=int((bf >= 11).sum()),
            silent=int((bf < 0).sum()), scenes_with_timely=int(bt[selected].any(-1).sum()),
            scenes_all_events_timely=int(bt[selected].all(-1).sum()),
            scenes_with_rescue=int(rescued.any(-1).sum()), scenes_with_loss=int(lost.any(-1).sum()),
            scene_counts=[dict(scene=int(n), denominator=int(emission.shape[1]), timely=int(bt[n].sum()),
                rescue=int((~at & bt)[n].sum()), loss=int((at & ~bt)[n].sum()))
                for n in np.flatnonzero(selected)])
    return result


def inventory_check(out, newkeys, rows):
    inventory = read(out/'inventory.json')
    previous_keys, previous_families, previous_bg = set(), set(), set()
    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get('physical_key'), str):
                previous_keys.add(value['physical_key'])
            if isinstance(value.get('background_family'), str):
                previous_families.add(value['background_family'])
            for name, dest in (('boxes', previous_keys), ('background_boxes', previous_bg)):
                boxes = value.get(name)
                if isinstance(boxes, list) and boxes and all(isinstance(b, dict) and
                        set(('lo', 'hi', 'rho')) <= set(b) for b in boxes):
                    dest.add(hashlib.sha256(json.dumps(boxes, sort_keys=True,
                        separators=(',', ':')).encode()).hexdigest())
                    if name == 'boxes' and value.get('background_family') and 'scene_id' not in value and 'scene_uid' not in value:
                        previous_bg.add(hashlib.sha256(json.dumps(boxes, sort_keys=True,
                            separators=(',', ':')).encode()).hexdigest())
            for v in value.values():
                collect(v)
        elif isinstance(value, list):
            for v in value:
                collect(v)
    for entry in inventory['files']:
        path = ROOT/entry['path']
        assert sha(path) == entry['sha256']
        collect(read(path))
    assert len(previous_keys) == inventory['unique_physical_keys']
    assert len(previous_bg) == inventory['unique_background_keys']
    for split in rows:
        assert not newkeys[split] & previous_keys
        assert not {r['background_family'] for r in rows[split]} & previous_families
        backgrounds = {hashlib.sha256(json.dumps(r['background_boxes'], sort_keys=True,
            separators=(',', ':')).encode()).hexdigest() for r in rows[split]}
        assert not backgrounds & previous_bg
    return dict(metadata_files=len(inventory['files']), prior_keys=len(previous_keys),
                prior_background_keys=len(previous_bg), overlaps=0)


def verify_metrics(reconstructed, production):
    mapping = {'nearpass': 'near_pass', 'farpass':'far_pass', 'pass_0_5':'0-5cm',
        'pass_5_10':'5-10cm', 'pass_10_20':'10-20cm', 'pass_20_35':'20-35cm', 'clear':'clear'}
    for key, ours in reconstructed.items():
        theirs = production['reports'][key]
        for group, heights in ours['paired'].items():
            for height, row in heights.items():
                name = height if group == 'all' else height+'_'+group
                expected = theirs['contacts'][name]
                for field, value in row.items():
                    field = 'timely' if field == 'candidate_timely' else field
                    assert expected[field] == value, (key, name, field, expected[field], value)
        for name, row in ours['costs'].items():
            expected = theirs['costs'][mapping[name]]
            for source, dest in (('light','light_notifications'), ('strong','strong_notifications'),
                                ('weighted','weighted_cost'), ('notifications','notifications'),
                                ('scene_denominator','scene_denominator'), ('clip_denominator','clip_denominator'),
                                ('clips_with_any','clips_with_any_notification'),
                                ('clips_with_light','clips_with_light_notification'),
                                ('clips_with_strong','clips_with_strong_notification'),
                                ('scenes_with_any','scenes_with_any_notification'), ('scene_counts','scene_counts')):
                assert row[source] == expected[dest], (key, name, source)
            if row['clip_denominator']:
                assert expected['clip_notification_rate'] == row['clips_with_any']/row['clip_denominator']
                assert expected['scene_notification_rate'] == row['scenes_with_any']/row['scene_denominator']
        for weight, field in (('0.0','weighted_cost_w0'), ('0.25','weighted_cost'), ('0.5','weighted_cost_w05')):
            assert ours['weighted_cost'][weight] == theirs['costs']['all_negative'][field]
        full = theirs['contacts']['HEAD_BODY']
        for field in ('denominator', 'scene_denominator', 'baseline_timely', 'rescue', 'loss', 'net', 'late', 'silent'):
            assert full[field] == sum(ours['paired']['all'][h][field] for h in HEIGHTS)


def verify_signal(reconstructed, production):
    baseline = reconstructed['hold/fixed/old5']
    base_cost = baseline['weighted_cost']['0.25']
    fullcost = lambda r: r['costs']['farpass']['notifications'] + r['costs']['clear']['notifications']
    base_far = fullcost(baseline)
    answers = {}
    for arm in ('both', 'matched'):
        main = reconstructed[f'hold/{SEEDS[0]}/{arm}']
        nets = {str(seed): sum(reconstructed[f'hold/{seed}/{arm}']['paired']['all'][h]['net']
                              for h in HEIGHTS) for seed in SEEDS}
        den = sum(main['paired']['all'][h]['denominator'] for h in HEIGHTS)
        eligible = arm == 'matched' or main['weighted_cost']['0.25'] <= base_cost
        consistent = all(nets[str(s)] > 0 for s in SEEDS[1:])
        signal = eligible and den == 512 and nets[str(SEEDS[0])] >= 10 and fullcost(main) <= 1.1*base_far and consistent
        expected = production['strong_signal']['arms'][arm]
        for field, value in dict(eligible_by_declared_arm_rule=eligible, main_net=nets[str(SEEDS[0])],
                main_denominator=den, seed_nets=nets, sensitivity_direction_consistent=consistent,
                strong_signal=signal, far_pass_plus_clear_notifications=fullcost(main),
                baseline_far_pass_plus_clear_notifications=base_far).items():
            assert expected[field] == value, (arm, field)
        answers[arm] = signal
    assert production['strong_signal']['any_strong_signal'] == any(answers.values())
    return answers


def validate_ledgers(out, archives, rows):
    events_seen, notifications_seen = set(), set()
    event_counts = notification_counts = 0
    with (out/'event_ledger.csv').open(encoding='utf-8-sig', newline='') as stream:
        for r in csv.DictReader(stream):
            split, seed, arm = r['split'], r['seed'], r['arm']
            key = f'{seed}/{arm}'
            n, k = int(r['scene']), int(r['replica'])
            q = HEIGHTS.index(r['height'])
            identity = (split, key, n, k, q)
            assert identity not in events_seen
            events_seen.add(identity)
            a = archives[split]
            emitted = a['emissions'][key][n, k, :, q]
            first_any = next((i for i, v in enumerate(emitted) if v), -1)
            first_timely = first_any if 0 <= first_any < 11 else -1
            assert int(r['first_index']) == first_any
            assert r['outcome'] == ('timely' if first_timely >= 0 else 'late' if first_any >= 11 else 'silent')
            assert int(r['light_notifications']) == int((emitted == 1).sum())
            assert int(r['strong_notifications']) == int((emitted == 2).sum())
            assert r['scene_uid'] == rows[split][n]['scene_uid']
            assert json.loads(r['physical_key']) == rows[split][n]['physical_key']
            assert r['category'] == a['category'][n, q]
            base_emitted = a['emissions']['fixed/old5'][n, k, :, q]
            base_first = next((i for i, v in enumerate(base_emitted) if v), -1)
            base_timely = 0 <= base_first < 11
            contact = a['category'][n, q] == 'contact'
            assert int(r['old5_first_index']) == base_first
            assert int(r['rescue']) == int(contact and first_timely >= 0 and not base_timely)
            assert int(r['loss']) == int(contact and base_timely and first_timely < 0)
            event_counts += 1
    expected_events = sum(sum(v.size//13 for v in a['emissions'].values()) for a in archives.values())
    assert event_counts == expected_events
    with (out/'notification_ledger.csv').open(encoding='utf-8-sig', newline='') as stream:
        for r in csv.DictReader(stream):
            split, seed, arm = r['split'], r['seed'], r['arm']
            key = f'{seed}/{arm}'
            n, k = int(r['scene']), int(r['replica'])
            frame = int(r['nominal_frame']) - 3
            identity = (split, key, n, k, frame)
            assert identity not in notifications_seen
            notifications_seen.add(identity)
            joint = int(archives[split]['emissions'][key][n, k, frame].max())
            assert joint > 0 and int(r['union_grade']) == joint
            row = rows[split][n]
            w = (0. if row['placement'] == 'contact' else .25 if row['placement'] == 'pass' and row['lateral_gap_m'] <= .10 and joint == 1 else 1.)
            assert r['scene_uid'] == row['scene_uid']
            assert abs(float(r['weight_w025']) - w) < 1e-12
            for weight, field in ((0., 'weight_w0'), (.5,'weight_w05')):
                expected_weight = (0. if row['placement'] == 'contact' else weight if
                    row['placement'] == 'pass' and row['lateral_gap_m'] <= .10 and joint == 1 else 1.)
                assert float(r[field]) == expected_weight
            for q, height in enumerate(HEIGHTS):
                assert int(r[height+'_grade']) == archives[split]['emissions'][key][n, k, frame, q]
            notification_counts += 1
    expected_notifications = sum(int((v.max(-1) > 0).sum())
        for a in archives.values() for v in a['emissions'].values())
    assert notification_counts == expected_notifications
    return dict(events=event_counts, joint_notifications=notification_counts)


def run(out):
    began = time.monotonic()
    deadline = began + 190.
    assert not (out/'independent_audit.json').exists(), 'Preserve completed independent audit'
    rows = read(out/'scene_rows.json')
    archives, checks = {}, []
    manifest = read(out/'execution_manifest.json')
    hashes = manifest['inherited']
    repair_path = out/'serialization_repair_binding.json'
    repair = read(repair_path) if repair_path.exists() else None
    for name, digest in manifest['new_sources'].items():
        if repair and str(Path(name)) == repair['repaired_metric_path']:
            assert sha(out/'execution_manifest.json') == repair['original_manifest_sha256']
            assert digest == repair['original_metric_sha256']
            assert sha(name) == repair['repaired_metric_sha256']
            text = Path(name).read_text(encoding='utf8')
            original = text.replace('scene_counts=[dict(scene=int(n), denominator=',
                                    'scene_counts=[dict(scene=n, denominator=', 1)
            assert original != text
            candidates = [original.encode(), original.replace('\n','\r\n').encode()]
            assert digest in {hashlib.sha256(v).hexdigest() for v in candidates}
            assert repair['inference'] == 0 and repair['calibration_reselection'] == 0
            for path, retained_digest in repair['retained_outputs_sha256'].items():
                assert sha(out/path) == retained_digest
        else:
            assert sha(name) == digest
    plan = read(out/'PLAN.json')
    assert plan['near_pass_max_gap_m'] == .10 and plan['near_pass_light_weight'] == .25
    assert plan['training'] == 0 and plan['protected_access'] == 0
    frozen_models = {}
    for name, digest in hashes.items():
        if name.endswith(('.pt', '.pickle')) or name.endswith(('thresholds.json', 'calibrations.json')):
            assert sha(name) == digest
            frozen_models[name] = digest
    assert len([p for p in frozen_models if p.endswith('.pt')]) == 8
    assert len([p for p in frozen_models if p.endswith('.pickle')]) == 6
    keys_by_split = {}
    for split in ('cal', 'hold'):
        with np.load(out/f'{split}_grades_notifications.npz', allow_pickle=False) as a:
            keys = a['keys'].tolist()
            grades = {key: a['grades'][i].copy() for i, key in enumerate(keys)}
            notices = {key: a['notifications'][i].copy() for i, key in enumerate(keys)}
            category = a['category'].copy()
            margins = a['margin'].copy()
            joint_scores = a['joint_scores'].copy()
        with np.load(out/'data'/split/'geometry.npz', allow_pickle=False) as a:
            np.testing.assert_array_equal(category, a['category'])
            keys_by_split[split], geometry_counts = geometry(rows[split], a['sensor'], category)
        assert geometry_counts == {'contact':256, 'clear':256, 'pass':256}
        assert (category == 'contact').sum(0).tolist() == [128,128]
        for layer in ('0-5cm','5-10cm','10-20cm','20-35cm'):
            selected_rows = [r for r in rows[split] if r['pass_layer'] == layer]
            assert len(selected_rows) >= 64 and {r['group'] for r in selected_rows} == {0,1}
        assert set(keys) == {'fixed/m3', 'fixed/old5'} | {f'{seed}/{arm}' for seed in SEEDS for arm in ('both', 'matched')}
        emissions = {key: scalar_gap1(value) for key, value in grades.items()}
        for key in keys:
            np.testing.assert_array_equal(emissions[key], notices[key])
        with np.load(out/'data'/split/'scores.npz', allow_pickle=False) as a:
            m3, local = smooth(a['m3_raw']), smooth(a['local_raw'])
            ordinary_raw = a['ordinary_raw'].copy()
        expected5 = (2*((m3 >= .9404184587540165) | (local >= 4.625390338985158))).astype(np.int8)
        expected3 = (2*(m3 >= .8557642486787612)).astype(np.int8)
        thresholds = read(ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
        cuts = read(ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010/calibrations.json')
        for si, seed in enumerate(SEEDS):
            np.testing.assert_array_equal(grades['fixed/old5'], expected5)
            np.testing.assert_array_equal(grades['fixed/m3'], expected3)
            ordinary = smooth(ordinary_raw[si])
            th = thresholds[str(seed)]
            cutoff = cuts[f'{seed}/score_current/c15_p64']['theta']
            strong = (expected5 > 0) | (ordinary >= th['addition'])
            expected = np.where(strong, 2, np.where(ordinary >= th['single'], 1, 0)).astype(np.int8)
            expected[(expected == 0) & np.isfinite(joint_scores[si]) & (joint_scores[si] >= cutoff)] = 1
            np.testing.assert_array_equal(expected, grades[f'{seed}/both'])
            np.testing.assert_allclose(np.maximum(ordinary-th['single'], joint_scores[si]-cutoff), margins[si], rtol=0, atol=1e-12)
        archives[split] = dict(grades=grades, emissions=emissions, category=category,
                               margin=margins, geometry_counts=geometry_counts)
    assert not keys_by_split['cal'] & keys_by_split['hold']
    inventory_details = inventory_check(out, keys_by_split, rows)
    checks.append('All scene geometry, scalar gap1, frozen m3/old5/both reconstruction and disjoint splits')
    seal = read(out/'sealed_calibration.json')
    for name, field in [('PLAN.json','plan_sha256'), ('cal_scores_readout.npz','cal_scores_readout_sha256'),
                       ('cal_grades_notifications.npz','cal_grades_notifications_sha256'),
                       ('calibration_curve.csv','calibration_curve_sha256'),
                       ('data/cal/scores.npz','cal_scores_sha256'),
                       ('data/cal/geometry.npz','cal_geometry_sha256')]:
        assert sha(out/name) == seal[field]
    scientific_hold = read(out/'scientific_hold_receipt.json')
    assert scientific_hold['sealed_calibration_sha256'] == sha(out/'sealed_calibration.json')
    assert (out/'sealed_calibration.json').stat().st_mtime_ns <= (out/'data/hold/hist.npy').stat().st_mtime_ns
    assert (out/'sealed_calibration.json').stat().st_mtime_ns <= (out/'data/hold/scores.npz').stat().st_mtime_ns
    with (out/'calibration_curve.csv').open(encoding='utf-8-sig', newline='') as stream:
        production_curve = list(csv.DictReader(stream))
    calibration_details = {}
    for si, seed in enumerate(SEEDS):
        cal = archives['cal']
        tau, cap, curve = enumerate_ties(cal['grades'][f'{seed}/both'],
            cal['grades']['fixed/old5'], cal['margin'][si], rows['cal'], deadline)
        record = seal['calibrations'][str(seed)]
        sealed_tau = record['tau'] if record['tau_kind'] == 'finite' else np.inf
        assert tau == sealed_tau
        assert cap == record['weighted_cap'] and record['contact_utility_access'] is False
        actual_curve = [r for r in production_curve if int(r['seed']) == seed]
        assert len(actual_curve) == len(curve)
        for ours, theirs in zip(curve, actual_curve):
            recorded_tau = None if theirs['tau'] in ('', 'None') else float(theirs['tau'])
            assert ours['tau'] == recorded_tau
            assert ours['weighted'] == float(theirs['weighted_cost'])
            assert ours['feasible'] == (theirs['feasible'].lower() == 'true')
        for split, a in archives.items():
            baseline, both = a['grades']['fixed/old5'], a['grades'][f'{seed}/both']
            expected = np.where(baseline > 0, 2,
                np.where((both > 0) & (a['margin'][si] >= tau), both, 0)).astype(np.int8)
            np.testing.assert_array_equal(expected, a['grades'][f'{seed}/matched'])
        calibration_details[str(seed)] = dict(tau=None if not np.isfinite(tau) else float(tau),
            cap=cap, exhaustive_tie_points=len(curve))
    checks.append('Every negative tie independently enumerated and lowest feasible cost-only tau reconstructed')
    ledger_counts = validate_ledgers(out, archives, rows)
    independent = {}
    for split, a in archives.items():
        for key, emission in a['emissions'].items():
            seed, arm = key.split('/')
            baseline = a['emissions']['fixed/old5']
            independent[f'{split}/{key}'] = dict(
                paired={group: paired(emission, baseline, a['category'], rows[split], group)
                        for group in ('all', 'dark_thin', 'sign_edge')},
                weighted_cost={str(w): total_cost(emission, rows[split], w) for w in (0., .25, .5)},
                costs={name: raw_cost(emission, mask, .25 if name.startswith(('near', 'pass_0', 'pass_5')) else 1.)
                       for name, mask in masks(rows[split]).items()})
    checks.append('All event/notification ledger rows and independent paired subgroup/raw weighted counts')
    production = read(out/'metrics.json')
    verify_metrics(independent, production)
    signals = verify_signal(independent, production)
    evaluation = read(out/'evaluation_receipt.json')
    for name, digest in evaluation['outputs_sha256'].items():
        assert sha(out/name) == digest
    checks.append('Production metrics and declared strong signal exactly match independent reconstruction; seal precedes hold')
    result = dict(status='PASS', seconds=time.monotonic()-began,
        source_sha256=sha(Path(__file__)), run=str(out), checks=checks,
        calibrations=calibration_details, ledger_counts=ledger_counts,
        reconstructed_metrics=independent, frozen_model_count=len(frozen_models),
        inventory=inventory_details, strong_signals=signals,
        independent_physical_scenes={s: len(k) for s, k in keys_by_split.items()},
        training=0, inference=0, rendering=0, protected_access=0)
    (out/'independent_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k != 'reconstructed_metrics'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts.local/work/cnh-cost-v2-holdout-dev-20261010')
    output = parser.parse_args().out
    started = time.monotonic()
    try:
        run(output)
    except BaseException as error:
        failure = dict(status='FAILED', seconds=time.monotonic()-started,
                       error=repr(error), source_sha256=sha(Path(__file__)))
        (output/f'independent_audit_failure_{time.time_ns()}.json').write_text(
            json.dumps(failure, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        raise
