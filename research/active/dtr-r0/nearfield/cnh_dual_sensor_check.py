"""Independent CPU reconstruction of dual alarm evidence from raw logits.

Uses no production evaluator, smoothing, threshold or event helper. Requires
the complete result and writes one engineering receipt under the task tree.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-dual-sensor-alarm-20261005'
OLD = ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
PROFILE = ROOT/'artifacts.local/work/cnh-location-reference-20261004/performance/cohort_profiles.json'
FRAMES = np.arange(3, 16)
THRESHOLD = .8557642486787612
ARMS = ('single', 'dual_frozen', 'dual_matched', 'alternating_frozen')
BRANCHES = {'shallow': [0, 1], 'deep': [2], 'outside5_10cm': [3, 4], 'outside15_20cm': [5, 6]}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(raw, acquired=None):
    """Loop over causal observation histories; missing sensor starts at -inf."""
    raw = np.asarray(raw, float)
    acquired = np.arange(13) if acquired is None else np.asarray(acquired)
    output = np.full_like(raw, -np.inf)
    history, last = [], None
    for tick in range(13):
        if tick in acquired:
            history.append(tick)
            ids = history[-5:]
            weights = 2. ** np.arange(len(ids))
            # The frozen trailing weights are proportional to these powers.
            value = sum(raw[..., t, :]*w for t, w in zip(ids, weights))/weights.sum()
            last = tick
        if last is not None and tick-last <= 2:
            output[..., tick, :] = value
    return output


def rx(degrees):
    a = np.deg2rad(degrees); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(degrees):
    a = np.deg2rad(degrees); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def run():
    started = time.monotonic()
    result = read(OUT/'result.json'); plan = read(OUT/'PLAN.json')
    assert result['status'] == 'COMPLETE'
    assert result['provenance']['plan_sha256'] == sha(OUT/'PLAN.json')
    units = plan['units']; N = len(units)
    profiles = {p['unit']: p for p in read(PROFILE)['profiles']}
    with np.load(OLD/'inputs/fresh_evaluation/rows.npz', allow_pickle=False) as z:
        rows = {k: z[k] for k in z.files}
    with np.load(OLD/'predictions/M3_fresh_evaluation.npz', allow_pickle=False) as z:
        original_raw = z['raw'].copy()
    index = {}
    for i, key in enumerate(zip(rows['unit'], rows['variant'], rows['replica'], rows['frame'])):
        assert key not in index
        index[key] = i
    assert len(index) == N*7*4*13
    single_raw = np.array([[[[original_raw[index[(u, v, k, f)]] for f in FRAMES]
                            for k in range(4)] for v in range(7)] for u in units])
    single = smooth(single_raw)
    full, alternate, covered, categories, ranges, sources = [], [], [], [], [], {}
    pose_errors = []
    ex = np.repeat(np.eye(4)[None], 2, axis=0)
    ex[0, :3, :3] = rx(10)@ry(-15)@rx(-10)
    ex[1, :3, :3] = rx(10)@ry(15)@rx(-10)
    for u in units:
        scorepath = OUT/'scores'/f'unit{u}.npz'
        sources[str(scorepath)] = sha(scorepath)
        with np.load(scorepath, allow_pickle=False) as z:
            raw, alt = z['raw_full'].copy(), z['raw_alternating'].copy()
            assert z['frames'].tolist() == FRAMES.tolist() and int(z['unit']) == u
        assert raw.shape == alt.shape == (2, 7, 4, 13, 2)
        assert np.isfinite(raw).all()
        full.append(smooth(raw))
        rebuilt = []
        for s in range(2):
            observed = np.flatnonzero(FRAMES % 2 == s)
            absent = np.flatnonzero(FRAMES % 2 != s)
            assert np.isfinite(np.take(alt[s], observed, axis=-2)).all()
            assert np.isnan(np.take(alt[s], absent, axis=-2)).all()
            rebuilt.append(smooth(alt[s], observed))
        alternate.append(rebuilt)
        truth = read(OLD/'truth/evaluation'/f'unit{u}.json')
        categories.append(np.asarray(truth['categories'])[:, FRAMES])
        ranges.append(np.asarray(truth['front_range_m'])[FRAMES])
        with np.load(OLD/'observations/evaluation'/f'unit{u}.npz', allow_pickle=False) as z:
            old_sensor, old_noisy, travel = z['sensor'].copy(), z['noisy'].copy(), z['travel'].copy()
        with np.load(OUT/'observations'/f'unit{u}.npz', allow_pickle=False) as z:
            actual_noisy, actual_query = z['noisy'].copy(), z['query'].copy()
        expected_noisy = old_noisy[None]@ex[:, None, None]
        expected_physical = old_sensor[None]@ex[:, None]
        expected_query = np.linalg.inv(travel)[None]@expected_physical
        pose_errors.append(max(float(np.abs(actual_noisy-expected_noisy).max()),
                               float(np.abs(actual_query-expected_query).max())))
        np.testing.assert_allclose(actual_noisy, expected_noisy, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual_query, expected_query, rtol=0, atol=1e-12)
        support = []
        for boxes in truth['boxes']:
            target = boxes[0]
            corners = np.array([[x, y, z] for x in (target['lo'][0], target['hi'][0])
                                for y in (target['lo'][1], target['hi'][1])
                                for z in (target['lo'][2], target['hi'][2])])
            inverse = np.linalg.inv(travel[FRAMES])
            local = np.einsum('tij,vj->tvi', inverse[:, :3, :3], corners)+inverse[:, None, :3, 3]
            front = local[..., 2].min(1)
            np.testing.assert_allclose(front, ranges[-1], atol=1e-12, rtol=0)
            support.append(bool(front[0] >= .9 and (front <= .9).any()))
        covered.append(support)
    full, alternate, covered = np.asarray(full), np.asarray(alternate), np.asarray(covered)
    categories, ranges = np.asarray(categories), np.asarray(ranges)
    both = {'single': single, 'dual_frozen': full.max(1), 'dual_matched': full.max(1),
            'alternating_frozen': alternate.max(1)}
    clear = np.repeat(np.all(categories[:, [5, 6]] == 'clear', axis=(2, 3))[:, :, None], 4, axis=2)
    peaks = {a: score[:, [5, 6]].max(axis=(3, 4)) for a, score in both.items()}
    masks = {'all': np.ones(N, bool), 'calibration': np.isin(units, plan['calibration_units']),
             'evaluation': np.isin(units, plan['evaluation_units'])}
    assert masks['calibration'].sum() == masks['evaluation'].sum() == 24
    assert not (masks['calibration'] & masks['evaluation']).any()
    budget = int((peaks['single'][clear & masks['calibration'][:, None, None]] >= THRESHOLD).sum())
    values = peaks['dual_matched'][clear & masks['calibration'][:, None, None]]
    # Enumerate attainable >= sets directly, with no production threshold helper.
    candidates = np.r_[np.nextafter(np.unique(values), np.inf), values.min()]
    feasible = [float(t) for t in candidates if np.count_nonzero(values >= t) <= budget]
    matched = min(feasible)
    thresholds = {a: matched if a == 'dual_matched' else THRESHOLD for a in ARMS}
    assert matched == result['thresholds']['dual_matched']['threshold']
    assert budget == result['thresholds']['dual_matched']['allowed_stops']
    target = {a: np.array([value[i, ..., profiles[u]['group']] for i, u in enumerate(units)])
              for a, value in both.items()}
    alarm = {a: value >= thresholds[a] for a, value in target.items()}
    stopped = {a: value.any(-1) for a, value in alarm.items()}
    first = {a: value.argmax(-1) for a, value in alarm.items()}
    timely = {a: stopped[a] & (np.take_along_axis(ranges[:, None, None, :], first[a][..., None], -1)[..., 0] >= .9)
              & covered[..., None] for a in ARMS}
    joint = {a: peaks[a] >= thresholds[a] for a in ARMS}
    errors = {}
    with np.load(OUT/'evaluation/ledger.npz', allow_pickle=False) as ledger:
        for name, actual in [('full_sensor_smoothed', full), ('alternating_sensor_smoothed', alternate),
                             ('covered', covered), ('joint_clear_den', clear)]:
            np.testing.assert_allclose(actual, ledger[name], atol=1e-12, rtol=0)
            finite = np.isfinite(actual)
            errors[name] = float(np.abs(actual[finite].astype(float)-ledger[name][finite].astype(float)).max())
        for a in ARMS:
            for field, value in [('alarm', alarm[a]), ('stopped', stopped[a]), ('timely', timely[a]),
                                 ('first_index', first[a]), ('both_query_scores', both[a]), ('joint_clear_stopped', joint[a])]:
                np.testing.assert_allclose(value, ledger[a+'_'+field], atol=1e-12, rtol=0)
    comparisons = 0
    for split, split_mask in masks.items():
        for group, cells in result['sequence'][split].items():
            group_mask = np.ones(N, bool)
            for token in group.split('/'):
                if token == 'all': continue
                if token in ('HEAD', 'BODY'):
                    group_mask &= np.array([profiles[u]['group'] == ('HEAD', 'BODY').index(token) for u in units])
                elif token in ('none', 'panel'):
                    group_mask &= np.array([profiles[u]['context'] == token for u in units])
                elif token in ('FOV_IN', 'FOV_OUT'):
                    group_mask &= np.array([bool(profiles[u]['fov_in']) == (token == 'FOV_IN') for u in units])
                elif token.startswith('mode'):
                    group_mask &= np.array([profiles[u]['mode'] == int(token[4:]) for u in units])
                else: raise ValueError(token)
            keep = group_mask & split_mask
            for a, cell in cells.items():
                assert cell['scenes'] == int(keep.sum())
                for branch, ids in BRANCHES.items():
                    valid = np.repeat(covered[:, ids, None], 4, 2)
                    expected_n = int(valid[keep].sum())
                    expected_count = int(timely[a][:, ids][keep].sum())
                    metric = cell['branches'][branch]
                    assert metric['timely']['n'] == expected_n
                    assert metric['timely']['stops'] == expected_count
                    assert metric['first_stops']['stops'] == int(stopped[a][:, ids][keep].sum())
                    assert metric['first_stops']['n'] == int(keep.sum())*len(ids)*4
                    assert metric['paired_minus_single']['delta_stops'] == expected_count-int(timely['single'][:, ids][keep].sum())
                    assert metric['paired_minus_single']['rescues'] == int((timely[a][:, ids] & ~timely['single'][:, ids] & valid)[keep].sum())
                    assert metric['paired_minus_single']['losses'] == int((timely['single'][:, ids] & ~timely[a][:, ids] & valid)[keep].sum())
                    if a == 'single':
                        observed_sources = {'single': int(stopped[a][:, ids][keep].sum())}
                    else:
                        observed_sources = dict(L=0, R=0, tie=0, both_above_threshold=0)
                        sensor_scores = alternate if a == 'alternating_frozen' else full
                        for i in np.flatnonzero(keep):
                            q = profiles[units[i]]['group']
                            for d in ids:
                                for k in range(4):
                                    if not stopped[a][i, d, k]: continue
                                    tick = first[a][i, d, k]
                                    left, right = sensor_scores[i, :, d, k, tick, q]
                                    observed_sources['L' if left > right else 'R' if right > left else 'tie'] += 1
                                    observed_sources['both_above_threshold'] += int(left >= thresholds[a] and right >= thresholds[a])
                    assert metric['first_report_source'] == observed_sources
                    comparisons += 8
                metric = cell['joint_clear']
                assert metric['n'] == int(clear[keep].sum())
                assert metric['stops'] == int((joint[a] & clear)[keep].sum())
                assert metric['paired_minus_single']['delta_stops'] == int(((joint[a] & clear)[keep]).sum()-((joint['single'] & clear)[keep]).sum())
                comparisons += 3
    ev = masks['evaluation']
    fov = np.array([profiles[u]['fov_in'] for u in units], bool)
    def difference(keep):
        den = int(np.repeat(covered[:, :2, None], 4, 2)[keep].sum())
        assert den > 0
        return (int(timely['dual_matched'][:, :2][keep].sum())-int(timely['single'][:, :2][keep].sum()))/den
    gain, inside = difference(ev & ~fov), difference(ev & fov)
    clear_delta = int((joint['dual_matched'] & clear)[ev].sum())-int((joint['single'] & clear)[ev].sum())
    branch = ('DUAL_ALARM_NOT_SUPPORTED_SIM' if gain < .1-1e-12 or inside < -.03-1e-12 else
              'DUAL_ALARM_SUPPORTED_SIM' if gain >= .3-1e-12 and clear_delta <= 1 else 'MIXED')
    assert branch == result['decision']['branch']
    receipt = dict(status='PASS',scope='Independent stdlib/numpy raw-logit reconstruction; no production evaluator helpers',
        threshold=matched,calibration_single_budget=budget,calibration_dual_stops=int((values>=matched).sum()),
        decision=branch,FOV_OUT_shallow_delta_rate=gain,FOV_IN_shallow_delta_rate=inside,clear_delta_stops=clear_delta,
        metric_integer_comparisons=comparisons,ledger_max_abs_errors=errors,shared_pose_query_max_abs_error=max(pose_errors),
        complete_scenes=N,whole_split_scene_counts={k:int(v.sum()) for k,v in masks.items()},
        score_sha256=sources,result_sha256=sha(OUT/'result.json'),plan_sha256=sha(OUT/'PLAN.json'),
        checker_sha256=sha(__file__),seconds=time.monotonic()-started)
    path = OUT/'engineering/independent_check.json'
    assert not path.exists(), 'Independent evidence already exists; do not overwrite'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in receipt.items() if not k.endswith('sha256')},indent=2))
    return receipt


def run_secondary():
    """Rebuild optional22.5 scores independently; never fit a new threshold."""
    tick = time.monotonic(); secondary = OUT/'secondary22p5'
    main_hash = sha(OUT/'PLAN.json')
    parent = read(OUT/'PLAN.json'); main = read(OUT/'result.json')
    plan = read(secondary/'PLAN.json'); result = read(secondary/'result.json')
    assert result['status'] == 'COMPLETE'
    assert plan['parent_plan_sha256'] == main_hash
    assert plan['deadline_unix'] == parent['deadline_unix']
    assert plan['started_unix'] == parent['started_unix']
    assert plan['splay_deg'] == [-22.5, 22.5] and plan['pitch_deg'] == -10
    assert plan['units'] == parent['units']
    assert plan['evaluation_units'] == parent['evaluation_units']
    units = plan['units']; N = len(units)
    profiles = {p['unit']: p for p in read(PROFILE)['profiles']}
    with np.load(OLD/'inputs/fresh_evaluation/rows.npz', allow_pickle=False) as z:
        rows = {k:z[k] for k in z.files}
    with np.load(OLD/'predictions/M3_fresh_evaluation.npz', allow_pickle=False) as z:
        original = z['raw'].copy()
    lookup = dict(zip(zip(rows['unit'], rows['variant'], rows['replica'], rows['frame']), range(len(original))))
    assert len(lookup) == N*7*4*13
    single = smooth(np.array([[[[original[lookup[(u,v,k,f)]] for f in FRAMES]
                                for k in range(4)] for v in range(7)] for u in units]))
    full, alternate, categories, ranges, covered, pose_error, hashes = [], [], [], [], [], [], {}
    extrinsics = np.repeat(np.eye(4)[None], 2, 0)
    for s, angle in enumerate((-22.5, 22.5)):
        extrinsics[s,:3,:3] = rx(10)@ry(angle)@rx(-10)
    for u in units:
        scorepath = secondary/'scores'/f'unit{u}.npz'; hashes[str(scorepath)] = sha(scorepath)
        with np.load(scorepath, allow_pickle=False) as z:
            raw, alt = z['raw_full'].copy(), z['raw_alternating'].copy()
            assert int(z['unit']) == u and np.array_equal(z['frames'], FRAMES)
        assert raw.shape == alt.shape == (2,7,4,13,2) and np.isfinite(raw).all()
        full.append(smooth(raw)); sampled = []
        for s in range(2):
            observed = np.flatnonzero(FRAMES%2 == s); absent = np.flatnonzero(FRAMES%2 != s)
            assert np.isfinite(np.take(alt[s], observed, axis=-2)).all()
            assert np.isnan(np.take(alt[s], absent, axis=-2)).all()
            sampled.append(smooth(alt[s], observed))
        alternate.append(sampled)
        truth = read(OLD/'truth/evaluation'/f'unit{u}.json')
        categories.append(np.asarray(truth['categories'])[:,FRAMES]); ranges.append(np.asarray(truth['front_range_m'])[FRAMES])
        with np.load(OLD/'observations/evaluation'/f'unit{u}.npz', allow_pickle=False) as z:
            sensor, noisy, travel = z['sensor'].copy(),z['noisy'].copy(),z['travel'].copy()
        with np.load(secondary/'observations'/f'unit{u}.npz', allow_pickle=False) as z:
            actual_noisy, actual_query = z['noisy'].copy(),z['query'].copy()
        expected_noisy = noisy[None]@extrinsics[:,None,None]
        expected_physical = sensor[None]@extrinsics[:,None]
        expected_query = np.linalg.inv(travel)[None]@expected_physical
        np.testing.assert_allclose(actual_noisy, expected_noisy, atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual_query, expected_query, atol=1e-12, rtol=0)
        with np.load(secondary/'templates'/f'unit{u}.npz', allow_pickle=False) as z:
            np.testing.assert_allclose(z['physical'], expected_physical, atol=1e-12, rtol=0)
        pose_error.append(max(float(np.abs(actual_noisy-expected_noisy).max()),float(np.abs(actual_query-expected_query).max())))
        supports = []
        for boxes in truth['boxes']:
            box = boxes[0]
            vertices = np.array([[x,y,z] for x in (box['lo'][0],box['hi'][0]) for y in (box['lo'][1],box['hi'][1]) for z in (box['lo'][2],box['hi'][2])])
            inverse = np.linalg.inv(travel[FRAMES])
            front = (np.einsum('tij,vj->tvi',inverse[:,:3,:3],vertices)+inverse[:,None,:3,3])[...,2].min(1)
            np.testing.assert_allclose(front,ranges[-1],atol=1e-12,rtol=0)
            supports.append(bool(front[0]>=.9 and (front<=.9).any()))
        covered.append(supports)
    full, alternate, ranges, categories, covered = map(np.asarray,(full,alternate,ranges,categories,covered))
    both = dict(single=single,secondary_frozen=full.max(1),secondary_transferred_main_threshold=full.max(1),secondary_alternating_frozen=alternate.max(1))
    thresholds = {a: main['thresholds']['dual_matched']['threshold'] if a=='secondary_transferred_main_threshold' else THRESHOLD for a in both}
    assert thresholds == result['thresholds'] and result['threshold_fit'].startswith('NONE')
    target = {a:np.array([value[i,...,profiles[u]['group']] for i,u in enumerate(units)]) for a,value in both.items()}
    alarms = {a:value>=thresholds[a] for a,value in target.items()}
    stopped = {a:value.any(-1) for a,value in alarms.items()}
    first = {a:value.argmax(-1) for a,value in alarms.items()}
    timely = {a:stopped[a] & (np.take_along_axis(ranges[:,None,None,:],first[a][...,None],-1)[...,0]>=.9) & covered[...,None] for a in both}
    clear = np.repeat(np.all(categories[:,[5,6]]=='clear',axis=(2,3))[:,:,None],4,2)
    joint = {a:value[:,[5,6]].max(axis=(3,4))>=thresholds[a] for a,value in both.items()}
    splits = dict(all=np.ones(N,bool),calibration=np.isin(units,parent['calibration_units']),evaluation=np.isin(units,parent['evaluation_units']))
    comparisons = 0
    for split,split_mask in splits.items():
        for group,cells in result['sequence'][split].items():
            group_mask = np.ones(N,bool)
            for token in group.split('/'):
                if token=='all': continue
                if token in ('HEAD','BODY'): group_mask &= np.array([profiles[u]['group']==('HEAD','BODY').index(token) for u in units])
                elif token in ('none','panel'): group_mask &= np.array([profiles[u]['context']==token for u in units])
                elif token in ('FOV_IN','FOV_OUT'): group_mask &= np.array([bool(profiles[u]['fov_in'])==(token=='FOV_IN') for u in units])
                elif token.startswith('mode'): group_mask &= np.array([profiles[u]['mode']==int(token[4:]) for u in units])
                else: raise ValueError(token)
            keep = split_mask&group_mask
            for a,cell in cells.items():
                assert cell['scenes']==int(keep.sum())
                for name,ids in BRANCHES.items():
                    den = np.repeat(covered[:,ids,None],4,2)
                    actual = int(timely[a][:,ids][keep].sum()); baseline = int(timely['single'][:,ids][keep].sum())
                    metric=cell['branches'][name]
                    assert metric['timely']['stops']==actual and metric['timely']['n']==int(den[keep].sum())
                    assert metric['paired_minus_single']['delta_stops']==actual-baseline
                    comparisons+=3
                assert cell['joint_clear']['stops']==int((joint[a]&clear)[keep].sum())
                assert cell['joint_clear']['n']==int(clear[keep].sum())
                assert cell['joint_clear']['paired_minus_single']['delta_stops']==int((joint[a]&clear)[keep].sum())-int((joint['single']&clear)[keep].sum())
                comparisons+=3
    assert sha(OUT/'PLAN.json')==main_hash and plan['deadline_unix']==read(OUT/'PLAN.json')['deadline_unix']
    receipt=dict(status='PASS',scope='Independent raw-logit secondary22.5 reconstruction; no production evaluator helpers, no recalibration',
        scenes=N,actual_splay_deg=[-22.5,22.5],pitch_deg=-10,shared_pose_query_max_abs_error=max(pose_error),
        metric_integer_comparisons=comparisons,transferred_main_threshold=thresholds['secondary_transferred_main_threshold'],
        parent_plan_unchanged=True,parent_deadline_inherited=True,
        summary={split:{group:{a:{name:cell['branches'][name]['timely'] for name in ('shallow','deep')}|{'joint_clear':cell['joint_clear']} for a,cell in result['sequence'][split][group].items()} for group in ('all','FOV_OUT','FOV_IN')} for split in ('all','evaluation')},
        score_sha256=hashes,secondary_result_sha256=sha(secondary/'result.json'),secondary_plan_sha256=sha(secondary/'PLAN.json'),parent_plan_sha256=main_hash,
        checker_sha256=sha(__file__),seconds=time.monotonic()-tick)
    path=OUT/'engineering/secondary_independent_check.json'
    assert not path.exists(), 'Independent secondary evidence already exists'
    path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('summary','score_sha256')},indent=2))
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--secondary',action='store_true'); args=parser.parse_args()
    run_secondary() if args.secondary else run()
