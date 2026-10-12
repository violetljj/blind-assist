"""Independent saved-output audit; no model forward, fit or point selection.

Reads only the completed fresh controlled ToF result and its pinned inherited
inputs. Keeps a cumulative 180-second audit limit and preserves failure logs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-frozen-e2e-20261010'
SEEDS = (2026100955, 2026100956, 2026100957)
OLD = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
JOINT = ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
M3_THETA = .8557642486787612
M3_FUSION_THETA = .9404184587540165
LOCAL_THETA = 4.625390338985158


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def save_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def scalar_replay(values):
    """Separate scalar episode transition implementation for each stream."""
    output, last_peak, zero_run = [], 0, 0
    for value in values:
        g = int(value)
        if g == 0:
            zero_run += 1
            if zero_run == 2:
                last_peak = 0
            output.append(0)
            continue
        output.append(g if g > last_peak else 0)
        last_peak = max(g, last_peak)
        zero_run = 0
    return np.asarray(output, np.int8)


def stateless_replay(values):
    """Independent vector formula using previous hit/reset indices, not peak state."""
    values = np.asarray(values)
    t = np.arange(values.shape[-2]).reshape((1,) * (values.ndim-2) + (-1, 1))
    positive, strong, zero = values > 0, values == 2, values == 0
    previous_zero = np.zeros_like(zero)
    previous_zero[..., 1:, :] = zero[..., :-1, :]
    reset = np.maximum.accumulate(np.where(zero & previous_zero, t, -1), axis=-2)
    def preceding(flags):
        last = np.maximum.accumulate(np.where(flags, t, -1), axis=-2)
        result = np.full_like(last, -1)
        result[..., 1:, :] = last[..., :-1, :]
        return result
    first_positive = preceding(positive) <= reset
    first_strong = preceding(strong) <= reset
    return np.where(strong & first_strong, 2,
                    np.where((values == 1) & first_positive, 1, 0)).astype(np.int8)


def first(values, strong=False):
    flags = values == 2 if strong else values > 0
    return np.where(flags.any(2), flags.argmax(2), -1)


def smooth(raw):
    result = np.empty_like(raw, dtype=float)
    for t in range(raw.shape[-2]):
        selected = raw[..., max(0, t-4):t+1, :].astype(float)
        weights = np.exp2(np.arange(selected.shape[-2]))
        result[..., t, :] = np.average(selected, weights=weights, axis=-2)
    return result


def ci(delta, denominator):
    selected = denominator > 0
    numerator, denominator = delta[selected], denominator[selected]
    if len(denominator) < 2:
        return None
    rng = np.random.default_rng(20261010)
    draws = rng.integers(0, len(denominator), (2000, len(denominator)))
    result = numerator[draws].sum(1) / denominator[draws].sum(1)
    return np.quantile(result, [.025, .975]).tolist()


def masks(category, rows):
    contact = category == 'contact'
    head, body = np.zeros_like(contact), np.zeros_like(contact)
    head[:, 0], body[:, 1] = contact[:, 0], contact[:, 1]
    horizontal = np.asarray([r['shape_family'] == 'horizontal' for r in rows])[:, None]
    thin = np.asarray([r['size_variant'] == 0 for r in rows])[:, None]
    edge = np.asarray([r['shape_family'] == 'sign_edge' for r in rows])[:, None]
    return dict(HEAD=head, BODY=body, all_contact_queries=contact,
                HEAD_horizontal_thin=head & horizontal & thin,
                HEAD_horizontal=head & horizontal, HEAD_sign_edge=head & edge)


def negative_masks(category):
    return dict(clear=(category == 'clear').all(1),
                purepass=(category == 'pass').any(1) & ~(category == 'contact').any(1))


def report_counts(notice, category, rows):
    hit, strong = first(notice), first(notice, True)
    contacts = {}
    for name, mask in masks(category, rows).items():
        truth = np.broadcast_to(mask[:, None, :], hit.shape)
        a, s = hit[truth], strong[truth]
        contacts[name] = dict(denominator=int(truth.sum()), scene_denominator=int(mask.any(1).sum()),
            timely=int(((a >= 0) & (a < 11)).sum()),
            timely_strong=int(((s >= 0) & (s < 11)).sum()),
            late=int((a >= 11).sum()), silent=int((a < 0).sum()), any_notification=int((a >= 0).sum()),
            first_index_histogram={str(int(t)): int((a == t).sum()) for t in np.unique(a)},
            first_strong_index_histogram={str(int(t)): int((s == t).sum()) for t in np.unique(s)})
    joint = notice.max(-1)
    costs = {}
    for name, mask in negative_masks(category).items():
        v = joint[mask]
        costs[name] = dict(scene_denominator=int(mask.sum()), clip_denominator=v.shape[0]*v.shape[1],
            frame_denominator=int(v.size), notifications=int((v > 0).sum()),
            light_notifications=int((v == 1).sum()), strong_notifications=int((v == 2).sum()),
            clips_with_any_notification=int((v > 0).any(-1).sum()),
            clips_with_light_notification=int((v == 1).any(-1).sum()),
            clips_with_strong_notification=int((v == 2).any(-1).sum()))
    return dict(contacts=contacts, costs=costs)


def subset_equal(actual, expected):
    for key, value in actual.items():
        if isinstance(value, dict):
            subset_equal(value, expected[key])
        else:
            assert value == expected[key], (key, value, expected[key])


def paired_counts(a, b, category, rows):
    af, bf = first(a), first(b)
    ast, bst = first(a, True), first(b, True)
    result = {}
    for name, mask in masks(category, rows).items():
        truth = np.broadcast_to(mask[:, None, :], af.shape)
        at = truth & (af >= 0) & (af < 11)
        bt = truth & (bf >= 0) & (bf < 11)
        common = truth & (af >= 0) & (bf >= 0)
        delta = bf - af
        numerator = (bt.astype(int)-at.astype(int)).sum((1, 2))
        denominator = truth.sum((1, 2))
        result[name] = dict(denominator=int(truth.sum()), scene_denominator=int(mask.any(1).sum()),
            baseline_timely=int(at.sum()), candidate_timely=int(bt.sum()), rescue=int((~at & bt).sum()),
            loss=int((at & ~bt).sum()), net=int(numerator.sum()),
            baseline_timely_strong=int((truth & (ast >= 0) & (ast < 11)).sum()),
            candidate_timely_strong=int((truth & (bst >= 0) & (bst < 11)).sum()),
            common_any_notified=int(common.sum()), common_timely=int((at & bt).sum()),
            earlier_common=int((common & (delta < 0)).sum()), later_common=int((common & (delta > 0)).sum()),
            unchanged_common=int((common & (delta == 0)).sum()))
        result[name]['ci95'] = ci(numerator, denominator)
    return result


def audit_curve(out, cal, category, sealed, check):
    with (out/'calibration_curve.csv').open(encoding='utf-8-sig', newline='') as stream:
        csvrows = list(csv.DictReader(stream))
    negmasks = negative_masks(category)
    neg = negmasks['clear'] | negmasks['purepass']
    subset_clear = negmasks['clear'][neg]
    old = cal['grades'][cal['keys'].tolist().index(f'{SEEDS[0]}/old5')][neg]
    oldnotice = stateless_replay(old)
    oldjoint = oldnotice.max(-1)
    caps = [int((oldjoint[subset_clear] > 0).sum()), int((oldjoint[~subset_clear] > 0).sum())]
    records = {}
    for si, seed in enumerate(SEEDS):
        check()
        base = cal['grades'][cal['keys'].tolist().index(f'{seed}/both')][neg]
        margin = cal['margin'][si][neg]
        eligible = (base > 0) & (old == 0) & np.isfinite(margin)
        ties = np.unique(margin[eligible])
        assert not len(ties) or ties[0] >= 0
        cutoffs = np.r_[0., np.nextafter(ties, np.inf), np.inf]
        rows = [r for r in csvrows if int(r['seed']) == seed]
        assert len(rows) == len(cutoffs)
        result = []
        # Full vector formula per cutoff; no incremental cost cache or monotonicity assumption.
        for start in range(0, len(cutoffs), 32):
            check()
            points = cutoffs[start:start+32]
            g = np.where(old[None] > 0, 2,
                np.where((base[None] > 0) & (margin[None] >= points[:, None, None, None, None]), base[None], 0))
            joint = stateless_replay(g).max(-1)
            clear = (joint[:, subset_clear] > 0).sum((1, 2, 3))
            passed = (joint[:, ~subset_clear] > 0).sum((1, 2, 3))
            result.extend(zip(clear.tolist(), passed.tolist()))
        for j, (point, expected) in enumerate(zip(cutoffs, result)):
            row = rows[j]
            recorded_tau = np.inf if row['tau_kind'] == 'positive_infinity' else float(row['tau'])
            assert point == recorded_tau, ('tau tie identity', seed, j, point, recorded_tau)
            assert expected == (int(row['clear']), int(row['pass_'])), ('curve cost', seed, j, expected, row)
            assert (row['feasible'] == 'True') == (expected[0] <= caps[0] and expected[1] <= caps[1])
        feasible = [j for j, cost in enumerate(result) if cost[0] <= caps[0] and cost[1] <= caps[1]]
        lowest = float(cutoffs[feasible[0]])
        chosen = sealed[str(seed)]
        tau = chosen['tau'] if chosen['tau_kind'] == 'finite' else np.inf
        assert tau == lowest, ('lowest feasible', seed, tau, lowest)
        assert chosen['clear_cap'] == caps[0] and chosen['pass_cap'] == caps[1]
        assert chosen['thresholds'] == len(rows) and chosen['negative_margin_ties'] == len(ties)
        assert chosen['contact_utility_access'] is False
        records[str(seed)] = dict(all_points_verified=len(rows), eligible_ties=len(ties),
            first_feasible_index=feasible[0], tau=None if np.isinf(lowest) else lowest,
            clear_cap=caps[0], pass_cap=caps[1], selected_cost=list(result[feasible[0]]))
        print('AUDIT_ALL_TIES', seed, len(rows), flush=True)
    return records


def audit(out):
    began = time.monotonic()
    attempts = out/'audit_attempts'
    attempts.mkdir(exist_ok=True)
    previous = sum(read(p)['seconds'] for p in attempts.glob('*.json'))
    def check():
        if previous + time.monotonic()-began >= 180:
            raise TimeoutError('Cumulative independent audit180s limit reached')
    receipt = dict(status='FAILED', previous_seconds=previous, source_sha256=sha(Path(__file__)))
    try:
        assert read(out/'receipt.json')['status'] == 'COMPLETE'
        prep, plan = read(out/'prepare_receipt.json'), read(out/'PLAN.json')
        assert read(out/'receipt.json')['plan_sha256'] == sha(out/'PLAN.json')
        inputs = read(out/plan.get('runtime_input_hashes', 'input_hashes.json'))
        for path, digest in inputs.items():
            check()
            assert sha(path) == digest, ('input SHA', path)
        rows = read(out/'scene_rows.json')
        assert sha(out/'scene_rows.json') == prep['scene_rows_sha256']
        prior = read(OLD/'scene_rows.json')
        assert sha(OLD/'scene_rows.json') == prep['old_geometry_sha256']
        oldkeys = {r['physical_key'] for split in ('ordinary_train', 'cf_train', 'cal', 'validation') for r in prior[split]}
        keys = {s: {r['physical_key'] for r in rows[s]} for s in ('cal', 'hold')}
        assert not keys['cal'] & keys['hold']
        for s in keys:
            assert len(keys[s]) == len(rows[s]) == 768
            assert not keys[s] & oldkeys
        for name in ('scientific_receipt.json', 'render_receipt.json'):
            for path, digest in read(out/name)['outputs_sha256'].items():
                check()
                assert sha(out/path) == digest, ('output SHA', path)
        import torch
        identities = {}
        for seed in SEEDS:
            state = torch.load(OLD/f'models/ordinary_seed{seed}.pt', map_location='cpu', weights_only=True)
            assert state['arm'] == 'ordinary' and state['seed'] == seed and state['steps'] == 7488
            identities[str(seed)] = dict(ordinary_arm=state['arm'], ordinary_steps=state['steps'])
            del state
        models, inherited_plan = read(JOINT/'models.json'), read(JOINT/'PLAN.json')
        for seed in SEEDS:
            for height in ('HEAD', 'BODY'):
                record = models[f'{seed}/score_current/{height}']
                path = JOINT/record['path']
                assert sha(path) == record['sha256'] and record['dimensions'] == 47
                with path.open('rb') as stream:
                    model = pickle.load(stream)
                assert type(model).__name__ == 'HistGradientBoostingClassifier'
                assert model.n_features_in_ == 47 and model.n_iter_ == record['iterations']
                for key, value in inherited_plan['params'].items():
                    assert model.get_params()[key] == value
                del model
        thresholds = read(ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
        cuts = read(JOINT/'calibrations.json')
        sealed = read(out/'sealed_calibration.json')
        assert sealed['plan_sha256'] == sha(out/'PLAN.json')
        assert sealed['cal_scores_sha256'] == sha(out/'data/cal/scores.npz')
        reports = read(out/'metrics.json')
        assert reports['calibrations'] == sealed['calibrations']
        archives, group_rows, recount = {}, [], {}
        stream_count = 0
        for split in ('cal', 'hold'):
            check()
            folder = out/'data'/split
            assert sha(folder/'geometry.npz') == prep['cohorts'][split]['geometry_sha256']
            assert sha(folder/'scores.npz') == read(folder/'scores_receipt.json')['sha256']
            with np.load(folder/'geometry.npz', allow_pickle=False) as a:
                category = a['category']
                np.testing.assert_array_equal(a['scene_uids'], [r['scene_uid'] for r in rows[split]])
                np.testing.assert_array_equal(a['scene_ids'], np.arange(768))
            expected_category = np.full((768, 2), 'clear', dtype='<U7')
            for n, row in enumerate(rows[split]):
                expected_category[n, row['group']] = row['placement']
            np.testing.assert_array_equal(category, expected_category)
            with np.load(folder/'scores.npz', allow_pickle=False) as a:
                raw = {k: a[k] for k in a.files}
            np.testing.assert_array_equal(raw['seeds'], SEEDS)
            assert np.isfinite(raw['m3_raw']).all() and np.isfinite(raw['ordinary_raw']).all()
            assert not np.isnan(raw['local_raw']).any() and not np.isposinf(raw['local_raw']).any()
            assert raw['current'].shape[-1] == 22
            np.testing.assert_array_equal(raw['current_valid'], np.isfinite(raw['current']))
            features = np.concatenate((np.where(raw['current_valid'], raw['current'], np.nan),
                                       (~raw['current_valid']).astype(float)), -1)
            np.testing.assert_array_equal(np.isnan(features[..., :22]), features[..., 22:] > 0)
            with np.load(out/f'{split}_grades_notifications.npz', allow_pickle=False) as a:
                archive = {k: a[k] for k in a.files}
            archives[split] = archive
            np.testing.assert_array_equal(archive['category'], category)
            np.testing.assert_array_equal(archive['scene_ids'], np.arange(768))
            np.testing.assert_array_equal(archive['seeds'], SEEDS)
            assert np.isfinite(archive['joint_scores']).all() and np.isfinite(archive['margin']).all()
            m3, local = smooth(raw['m3_raw']), smooth(raw['local_raw'])
            old5 = (m3 >= M3_FUSION_THETA) | (local >= LOCAL_THETA)
            expected_keys = [f'{seed}/{arm}' for seed in SEEDS for arm in ('m3', 'old5', 'both', 'matched')]
            assert archive['keys'].tolist() == expected_keys
            grades = dict(zip(expected_keys, archive['grades']))
            emissions = dict(zip(expected_keys, archive['notifications']))
            for si, seed in enumerate(SEEDS):
                ordinary = smooth(raw['ordinary_raw'][si])
                single, addition = thresholds[str(seed)]['single'], thresholds[str(seed)]['addition']
                base = np.where(old5 | (ordinary >= addition), 2, np.where(ordinary >= single, 1, 0))
                joint = archive['joint_scores'][si]
                cutoff = cuts[f'{seed}/score_current/c15_p64']['theta']
                both = np.where((base == 0) & (joint >= cutoff), 1, base)
                margin = np.maximum(ordinary-single, joint-cutoff)
                np.testing.assert_array_equal(archive['margin'][si], margin)
                record = sealed['calibrations'][str(seed)]
                tau = record['tau'] if record['tau_kind'] == 'finite' else np.inf
                expected = dict(m3=2*(m3 >= M3_THETA), old5=2*old5,
                    both=both, matched=np.where(old5, 2, np.where((both > 0) & (margin >= tau), both, 0)))
                for arm in ('m3', 'old5', 'both', 'matched'):
                    check()
                    key = f'{seed}/{arm}'
                    grade, emitted = grades[key], emissions[key]
                    np.testing.assert_array_equal(grade, expected[arm])
                    # All streams inspected by scalar transitions, independent of main/helper.
                    for n in range(len(grade)):
                        for k in range(grade.shape[1]):
                            for q in range(2):
                                np.testing.assert_array_equal(scalar_replay(grade[n, k, :, q]), emitted[n, k, :, q])
                                stream_count += 1
                    np.testing.assert_array_equal(stateless_replay(grade), emitted)
                    np.testing.assert_array_equal(first(grade), first(emitted))
                    np.testing.assert_array_equal(first(grade, True), first(emitted, True))
                    counts = report_counts(emitted, category, rows[split])
                    subset_equal(counts, reports['reports'][f'{split}/{key}'])
                    recount[f'{split}/{key}'] = counts
                    if arm in ('both', 'matched'):
                        reference = emissions[f'{seed}/old5']
                        contact = (category == 'contact')[:, None, :]
                        firstold, firstnew = first(reference), first(emitted)
                        assert not (contact & (firstold >= 0) & (firstold < 11) & ((firstnew < 0) | (firstnew >= 11))).any()
                        paired = paired_counts(reference, emitted, category, rows[split])
                        for name, values in paired.items():
                            expected_paired = reports['comparisons'][f'{split}/{key}']['contacts'][name]
                            interval = values.pop('ci95')
                            subset_equal(values, expected_paired)
                            assert interval == expected_paired['timely_rate_difference_ci']['ci95']
                    for family in sorted({r['background_family'] for r in rows[split]}):
                        indices = [i for i, r in enumerate(rows[split]) if r['background_family'] == family]
                        grouped = report_counts(emitted[indices], category[indices], [rows[split][i] for i in indices])
                        group_rows.append(dict(split=split, seed=seed, arm=arm, background_family=family, **grouped))
        curve = audit_curve(out, archives['cal'], archives['cal']['category'], sealed['calibrations'], check)
        hold_matched = {str(seed): recount[f'hold/{seed}/matched'] for seed in SEEDS}
        assert all(recount[f'hold/{seed}/old5']['contacts']['HEAD_horizontal_thin']['denominator'] == 32 for seed in SEEDS)
        assert all(recount[f'hold/{seed}/old5']['contacts']['HEAD_horizontal_thin']['scene_denominator'] == 16 for seed in SEEDS)
        check()
        save_new(out/'audit_background_groups.json', dict(scope='All split/seed/arm/background-family rows; no selected subgroup', rows=group_rows))
        receipt.update(status='PASS', inputs_verified=len(inputs), model_identities=identities,
            new_old_physical_overlap=0, new_cal_hold_overlap=0, scalar_streams_verified=stream_count,
            metric_cells_verified=len(recount), all_tie_curve=curve, hold_matched=hold_matched,
            checks=['pinned input/model/output SHA', 'ordinary metadata and HGB saved identity without forward',
                'finite M3/ordinary; local -inf allowed, nan/+inf rejected', 'missing descriptor indicators',
                'fixed M3 and original5 thresholds; fixed ordinary/HGB saved score grades',
                'independent scalar/stateless episode replay and first-any/strong conservation',
                'actual contact query, HEAD thin vs thick/edge and BODY numerators',
                'paired rescue/loss/first-time shifts and fixed scene bootstrap CI',
                'all cal whole-score ties; first feasible dual-cost point; fixed hold application',
                'original5 timely retention; grouped background costs and contact counts'],
            limits='Fresh same-simulator controlled ToF offline notification replay; inference not rerun, RGB end-to-end not established; resource release uses main receipt, not separate host-process inspection',
            main_resource_release_claim=read(out/'receipt.json')['resource_release'], training=0, model_forward=0, protected_access=0)
        print('AUDIT_PASS', stream_count, len(recount), flush=True)
    except BaseException as error:
        receipt.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['seconds'] = time.monotonic()-began
        receipt['cumulative_seconds'] = previous + receipt['seconds']
        save_new(attempts/f'audit_{time.time_ns()}.json', receipt)
        if receipt['status'] == 'PASS':
            save_new(out/'audit_receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    audit(args.out.resolve())
