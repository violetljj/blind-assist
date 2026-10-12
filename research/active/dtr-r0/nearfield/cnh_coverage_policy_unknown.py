"""Frozen, causal single-side coverage proxy from estimated ego-motion only.

This flags directional coverage uncertainty; it is not a obstacle detector or a
validated false-positive label. Translation-bearing estimates require estimated
SE3 and a gravity-aligned horizontal plane, unavailable from head IMU alone.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
OUT = WORK / 'cnh-coverage-policy-20261004'
PILOT = WORK / 'cnh-readout-pilot2-20261005'
NATURAL = WORK / 'cnh-margin-confirm-20261002'
RULE = dict(dt_s=.2, intervals=5, window_s=1., head_vectors=6,
            yaw_threshold_deg=8., min_displacement_m=.05,
            min_mean_head_horizontal_norm=.05,
            direction='1s net estimated translation projected into gravity-aligned XZ',
            head='mean of six estimated head-forward vectors projected into XZ',
            angle='signed atan2(travel_z*head_x-travel_x*head_z, dot)',
            decision='abs(angle)>=8 degrees; opposite side UNKNOWN',
            warmup='first five frames WARMUP_UNKNOWN',
            unavailable='insufficient displacement or horizontal head norm => DIRECTION_UNAVAILABLE_UNKNOWN',
            input_contract='estimated SE3 only; dt and gravity-axis are declared public assumptions; no true motion, target, range, visibility or mode enters marker')


def marker(estimated):
    estimated = np.asarray(estimated, dtype=np.float64)
    if estimated.ndim != 3 or estimated.shape[1:] != (4, 4) or not np.isfinite(estimated).all():
        raise ValueError('Expected finite estimated SE3 sequence')
    count = len(estimated)
    angle = np.full(count, np.nan)
    displacement = np.full(count, np.nan)
    # 0=WARMUP_UNKNOWN, 1=DIRECTION_UNAVAILABLE_UNKNOWN, 2=NO_BIAS_FLAG,
    # 3=LEFT_UNKNOWN, 4=RIGHT_UNKNOWN. NO_BIAS_FLAG does not mean clear.
    state = np.zeros(count, dtype=np.int8)
    for frame in range(5, count):
        travel = (estimated[frame, :3, 3] - estimated[frame-5, :3, 3])[[0, 2]]
        head = estimated[frame-5:frame+1, :3, 2].mean(0)[[0, 2]]
        displacement[frame] = np.linalg.norm(travel)
        if displacement[frame] < .05 or np.linalg.norm(head) < .05:
            state[frame] = 1
            continue
        angle[frame] = np.rad2deg(np.arctan2(travel[1]*head[0]-travel[0]*head[1], np.dot(travel, head)))
        state[frame] = 3 if angle[frame] >= 8. else 4 if angle[frame] <= -8. else 2
    return dict(angle_deg=angle, displacement_m=displacement, state=state)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def summarize(rows, states):
    states = np.asarray(states)
    if len(states) == 0:
        return dict(sequences=0, evaluated_frames=0, flagged_frames=0, flagged_sequences=0)
    flag = states >= 3
    evaluable = states >= 2
    first = [float(np.flatnonzero(x)[0]*.2) for x in flag if x.any()]
    return dict(sequences=len(states), total_frames=int(states.size),
                post_warmup_frames=int((states != 0).sum()),
                evaluated_frames=int(evaluable.sum()), flagged_frames=int(flag.sum()),
                flagged_frame_fraction=float(flag.sum()/evaluable.sum()) if evaluable.any() else None,
                flagged_sequences=int(flag.any(1).sum()),
                flagged_sequence_fraction=float(flag.any(1).mean()),
                left_unknown_frames=int((states == 3).sum()), right_unknown_frames=int((states == 4).sum()),
                warmup_unknown_frames=int((states == 0).sum()),
                direction_unavailable_frames=int((states == 1).sum()),
                first_flag_time_from_sequence_start_s=dict(n=len(first), median=float(np.median(first)) if first else None,
                    min=float(min(first)) if first else None, max=float(max(first)) if first else None),
                mode0_detection_reference='known simulated bias begins at sequence start; first flag includes required 1s warmup')


def engineering_checks():
    def poses(yaw=15.):
        p = np.repeat(np.eye(4)[None], 16, axis=0)
        a = np.deg2rad(yaw); c, s = np.cos(a), np.sin(a)
        p[:, :3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        p[:, 2, 3] = np.arange(16)*.16
        return p
    p = poses()
    got = marker(p)
    assert np.all(got['state'][:5] == 0) and np.all(got['state'][5:] == 3)
    assert np.allclose(got['angle_deg'][5:], 15.)
    assert np.all(marker(poses(0.))['state'][5:] == 2)
    stationary = p.copy(); stationary[:, :3, 3] = 0
    assert np.all(marker(stationary)['state'][5:] == 1)
    a = np.deg2rad(71.); c, s = np.cos(a), np.sin(a)
    gauge = np.eye(4); gauge[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    gauge[:3, 3] = [7., 2., -3.]
    moved = marker(gauge[None] @ p)
    assert np.array_equal(moved['state'], got['state'])
    assert np.allclose(moved['angle_deg'][5:], got['angle_deg'][5:])
    return dict(status='PASS', checks=['15deg bias triggers exactly after 1s warmup',
        'zero head/travel offset remains unflagged', 'stationary motion => direction unavailable',
        'world yaw/translation gauge leaves decision unchanged'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-rendered', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    deadline = json.loads((OUT/'budget.json').read_text())['deadline_unix_s']
    plan = json.loads((OUT/'PLAN.json').read_text())
    target = OUT/'unknown'
    target.mkdir(parents=True, exist_ok=True)
    checks = engineering_checks()
    records, states, angles, lengths = [], [], [], []
    def add(split, unit, config, replica, estimated):
        if time.time() >= deadline:
            raise TimeoutError('Authorized coverage wall budget reached')
        decision = marker(estimated)
        # Evaluation-only mode annotation is added after prediction, not read by marker.
        records.append(dict(split=split, unit=int(unit), config=int(config), replica=int(replica), mode=int(unit%3)))
        states.append(decision['state']); angles.append(decision['angle_deg']); lengths.append(decision['displacement_m'])
    pilot_plan = json.loads((PILOT/'PLAN.json').read_text())
    units = pilot_plan.get('eval_units', pilot_plan.get('evaluation_units'))
    if units is None:
        units = [u for u in range(231001, 231073) if u%3 in (0, 1)]
    for unit in units:
        if unit%3 != 0:
            continue
        with np.load(PILOT/'observations/evaluation'/f'unit{unit}.npz', allow_pickle=False) as obs:
            for replica, estimated in enumerate(obs['noisy']):
                add('controlled_original_mode0', unit, -1, replica, estimated)
    import cnh_cvr_pilot as CP
    for split, ids, subdir in [('natural_95000', range(95000, 95048), 'calib'), ('natural_96000', range(96000, 96096), 'evaluation')]:
        for unit in ids:
            with np.load(NATURAL/'features'/subdir/f'unit{unit}.npz', allow_pickle=False) as data:
                configs = np.unique(data['scene'])
            for config in configs:
                _, _, estimated = CP.motion_metadata(unit, int(config))
                add(split, unit, config, 0, estimated)
    # Optional separate rendered-input pass can be added without altering rule.
    if args.with_rendered:
        for path in sorted((OUT/'observations').glob('unit*.npz')):
            with np.load(path, allow_pickle=False) as obs:
                unit = int(path.stem.removeprefix('unit'))
                noisy = obs['noisy']
                for intervention in range(len(noisy)):
                    for replica, estimated in enumerate(noisy[intervention]):
                        add(f'controlled_return_{intervention}', unit, -1, replica, estimated)
    states = np.asarray(states, dtype=np.int8)
    np.savez_compressed(target/'ledger.npz', state=states, angle_deg=np.asarray(angles), displacement_m=np.asarray(lengths),
        split=np.asarray([r['split'] for r in records]), unit=np.asarray([r['unit'] for r in records]),
        config=np.asarray([r['config'] for r in records]), replica=np.asarray([r['replica'] for r in records]),
        mode=np.asarray([r['mode'] for r in records]))
    summary = {}
    for split in sorted({r['split'] for r in records}):
        for mode in sorted({r['mode'] for r in records if r['split'] == split}):
            selection = [i for i, r in enumerate(records) if r['split'] == split and r['mode'] == mode]
            summary[f'{split}/mode{mode}'] = summarize([records[i] for i in selection], states[selection])
    result = dict(status='COMPLETE', rule=RULE, summaries=summary, engineering_checks=checks,
        elapsed_s=time.monotonic()-start, plan_sha256=hashlib.sha256((OUT/'PLAN.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        natural_estimator_source_sha256=hashlib.sha256((CP.SOURCE/'cnh_track_a_readout.py').read_bytes()).hexdigest(),
        limitations=['Assumed estimated SE3 includes translation; this is not available from head IMU alone.',
        'Horizontal/gravity axis is a declared estimator assumption.',
        'Mode1/mode2 trigger fractions are generator mixture diagnostics, not verified false positives or user-frequency estimates.',
        'NO_BIAS_FLAG does not certify clear space; flagged opposite-side UNKNOWN is a coverage proxy.',
        'Natural inputs reuse consumed Development and deterministic estimator replay, not new real-user observations.'])
    save(target/'result.json', result)
    print(json.dumps(dict(status=result['status'], elapsed_s=result['elapsed_s'], summaries=summary), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
