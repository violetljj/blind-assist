"""Controlled return-to-center observations; no model, threshold or training.

Physical truth enters generation only. Public inference files retain signed
photons, nominal ambient, estimated relative poses and commanded query frames.
Unchanged observations and noisy-pose prefixes are copied bitwise from pilot2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import cnh_displacement_ceiling_render as R
import cnh_location_reference_gpu as G
from cnh_track_a_readout import noisy_poses

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
OUT = ROOT/'artifacts.local/work/cnh-coverage-policy-20261004'
PROFILE = ROOT/'artifacts.local/work/cnh-location-reference-20261004/performance/cohort_profiles.json'
RC = np.asarray([2.5, 2.1, 1.7, 1.3, 1.0], np.float64)
DT, SPEED, RETURN_S = .2, .8, .4
PHOTON_SEED, POSE_SEED = 2026100406, 2026100407


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def check_budget():
    remaining = read(OUT/'budget.json')['deadline_unix_s']-time.time()
    if remaining <= 0:
        raise TimeoutError('Authorized coverage wall-clock limit reached')
    return remaining


def cohort():
    records = [p for p in read(PROFILE)['profiles'] if p['mode'] == 0]
    assert len(records) == 24 and sum(not p['fov_in'] for p in records) == 11
    return records


def trajectory(sensor, travel, front_range_m, rc):
    """Scheduler truth gives crossing time; only command is exported to reader."""
    sensor, travel = np.asarray(sensor), np.asarray(travel)
    ranges = np.asarray(front_range_m)
    np.testing.assert_allclose(np.diff(ranges), -SPEED*DT, atol=1e-12, rtol=0)
    crossing_s = max(0., (float(ranges[0])-float(rc))/SPEED)
    times = np.arange(len(sensor))*DT
    yaw = 15.*(1.-np.clip((times-crossing_s)/RETURN_S, 0., 1.))
    query = np.repeat(np.eye(4)[None], len(sensor), axis=0)
    for f, y in enumerate(yaw):
        query[f, :3, :3] = R.S.ry(y)@R.S.rx(-10.)
    physical = sensor.copy()
    changed = yaw != 15.
    for f in np.flatnonzero(changed):
        physical[f, :3, :3] = travel[f, :3, :3]@query[f, :3, :3]
    np.testing.assert_allclose(np.linalg.inv(travel)@physical, query, atol=1e-12, rtol=0)
    return physical, query, yaw, changed, crossing_s


def target_ray_counts(engine, targets):
    """Evaluator-only visibility from the same background first-hit rays."""
    cp = engine.cp
    p, n = engine.npose, engine.nray
    directions = engine.directions.reshape(p, n, 3)
    origins = engine.poses[:, None, :3, 3]
    back = engine.distance.reshape(p, n)
    result = []
    for box in targets:
        lo, hi = cp.asarray(box['lo']), cp.asarray(box['hi'])
        parallel = cp.abs(directions) < 1e-14
        safe = cp.where(parallel, 1., directions)
        a, b = (lo-origins)/safe, (hi-origins)/safe
        near = cp.max(cp.where(parallel, -cp.inf, cp.minimum(a, b)), axis=-1)
        far = cp.min(cp.where(parallel, cp.inf, cp.maximum(a, b)), axis=-1)
        outside = cp.any(parallel & ((origins < lo) | (origins > hi)), axis=-1)
        hit = cp.where(near > 1e-10, near, far)
        take = (~outside) & (far >= cp.maximum(near, 0.)) & (hit > 1e-10) & cp.isfinite(hit) & (hit <= back)
        result.append(cp.asnumpy(take.sum(axis=-1)).astype(np.int32))
    return np.stack(result)


def render_one(unit):
    check_budget(); started = time.monotonic()
    paths = dict(obs=OLD/'observations/evaluation'/f'unit{unit}.npz',
                 templates=OLD/'templates/evaluation'/f'unit{unit}.npz',
                 truth=OLD/'truth/evaluation'/f'unit{unit}.json')
    truth = read(paths['truth']); assert truth['mode'] == 0
    with np.load(paths['obs']) as obs:
        old = {name: obs[name] for name in obs.files}
    with np.load(paths['templates']) as template:
        expected_old = template['expected']
        visible_old = (template['object_id'] == 0).sum(axis=(-3, -2, -1)).astype(np.int32)
    trajectories = [trajectory(old['sensor'], old['travel'], truth['front_range_m'], rc) for rc in RC]
    sensor = np.stack([x[0] for x in trajectories]); query = np.stack([x[1] for x in trajectories])
    yaw = np.stack([x[2] for x in trajectories]); changed = np.stack([x[3] for x in trajectories])
    expected = np.repeat(expected_old[None], len(RC), axis=0)
    visible = np.repeat(visible_old[None], len(RC), axis=0)
    # Reuse identical final-zero poses across return schedules. Only changed
    # exposure matrices are rendered, without redrawing the target/background.
    positions = np.argwhere(changed)
    unique, inverse = np.unique(sensor[changed].reshape(-1, 16), axis=0, return_inverse=True)
    targets = [boxes[0] for boxes in truth['boxes']]
    background = truth['boxes'][0][1:]
    assert all(boxes[1:] == background for boxes in truth['boxes'])
    engine = G.ExpectedRenderer(unique.reshape(-1, 4, 4), background)
    try:
        endpoints = engine.render(targets, candidate_batch=4, pose_batch=32, deadline_check=check_budget)
        rhos = np.asarray([b['rho'] for b in targets], np.float64)
        means = endpoints[:, 0]+rhos[:, None, None, None, None]/G.ENDPOINT_RHO*(endpoints[:, 1]-endpoints[:, 0])
        ray_counts = target_ray_counts(engine, targets)
        ambient = np.repeat(old['ambient'][None], len(RC), axis=0)
        np.testing.assert_array_equal(engine.ambient, np.repeat(old['ambient'][:1], len(unique), axis=0))
        device_meta = engine.metadata
    finally:
        engine.close()
    for (r, f), ix in zip(positions, inverse):
        expected[r, :, f] = means[:, ix]
        visible[r, :, f] = ray_counts[:, ix]
    hist = np.repeat(old['hist'][None], len(RC), axis=0)
    noisy = np.empty((len(RC), 4, 16, 4, 4), np.float64)
    for r in range(len(RC)):
        check_budget()
        for d in range(7):
            for k in range(4):
                seed = int(np.random.SeedSequence([PHOTON_SEED, unit, d, k]).generate_state(1)[0])
                h, _, _ = R.sample(expected[r, d], ambient[r], seed)
                hist[r, d, k, changed[r]] = h[changed[r]]
        for k in range(4):
            seed = int(np.random.SeedSequence([POSE_SEED, unit, k]).generate_state(1)[0])
            noisy[r, k] = noisy_poses(sensor[r], seed, dt=DT)
            np.testing.assert_array_equal(noisy[r, k, ~changed[r]], old['noisy'][k, ~changed[r]])
        np.testing.assert_array_equal(hist[r][:, :, ~changed[r]], old['hist'][:, :, ~changed[r]])
    obs_path = OUT/'observations'/f'unit{unit}.npz'; obs_path.parent.mkdir(exist_ok=True)
    private_path = OUT/'templates'/f'unit{unit}.npz'; private_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(obs_path, hist=hist, ambient=ambient, noisy=noisy,
                        query=query, yaw_deg=yaw, changed=changed, rc_m=RC)
    np.savez_compressed(private_path, expected=expected, target_ray_count=visible, sensor=sensor,
                        travel=old['travel'], crossing_s=np.asarray([x[4] for x in trajectories]))
    save(OUT/'truth'/f'unit{unit}.json', truth)
    record = dict(status='COMPLETE', unit=unit, elapsed_s=time.monotonic()-started,
                  changed_exposures=int(changed.sum()), unique_physical_exposures=len(unique),
                  unchanged_prefix_bitwise=True, paired_noise_innovations=True,
                  latent_counts='NOT_AVAILABLE in pilot2 source; omitted for all schedules',
                  photon_seed_prefix=PHOTON_SEED, pose_seed_prefix=POSE_SEED,
                  renderer=device_meta, source_sha256=R.source_sha256(),
                  original_input_sha256={k:G.hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
                  outputs_sha256={str(p.relative_to(OUT)):G.hashlib.sha256(p.read_bytes()).hexdigest() for p in (obs_path,private_path)})
    save(OUT/'render_receipts'/f'unit{unit}.json', record)
    return record


def geometry_check():
    """Planar formula checked against finite sub16 rays; full3D is separate."""
    rays, _ = R.S.angular_rays(16)
    rays = rays.reshape(-1, 3)
    edge = -np.rad2deg(np.arctan2(rays[:, 0], rays[:, 2])).min()
    rows = []
    for yaw in (0., 5., 10., 15., 20.):
        closed = .29/np.tan(np.deg2rad(22.5-yaw))
        finite = .29/np.tan(np.deg2rad(edge-yaw))
        direction = rays@R.S.ry(yaw).T
        found = []
        for distance in (finite*.999, finite*1.001):
            box = dict(lo=[-8.,-10.,distance], hi=[-.29,10.,distance+1e-8],rho=.5)
            hits = R.S.raycast_boxes(np.zeros(3),direction,[box])
            found.append(bool((hits['object_id'] == 0).any()))
        assert found == [False, True], (yaw, finite, found)
        rows.append(dict(yaw_deg=yaw, planar_continuous_m=closed, finite_sub16_m=finite,
                         immediately_below_above_visible=found))
    value = dict(status='PASS', role='constructed horizontal geometry; not finite-height pitched-target visibility',
                 fov_half_continuous_deg=22.5, fov_half_finite_ray_deg=float(edge),rows=rows)
    save(OUT/'engineering/geometry_check.json', value)
    return value


def engineering_check():
    check_budget(); started = time.monotonic(); geometry = geometry_check()
    unit = cohort()[0]['unit']; truth = read(OLD/'truth/evaluation'/f'unit{unit}.json')
    with np.load(OLD/'observations/evaluation'/f'unit{unit}.npz') as data:
        sensor, travel = data['sensor'], data['travel']
    physical, _, _, changed, _ = trajectory(sensor, travel, truth['front_range_m'], RC[0])
    frames = [0, int(np.flatnonzero(changed)[0]), 15]
    poses = physical[frames]
    boxes = truth['boxes'][0]
    reference = R.expected(dict(poses=poses, boxes=boxes))
    engine = G.ExpectedRenderer(poses, boxes[1:])
    try:
        endpoints = engine.render([boxes[0]], deadline_check=check_budget)[0]
        expected = endpoints[0]+boxes[0]['rho']/G.ENDPOINT_RHO*(endpoints[1]-endpoints[0])
        error = float(np.max(np.abs(expected-reference['expectation'])))
        np.testing.assert_allclose(expected, reference['expectation'], atol=1e-8, rtol=1e-9)
        np.testing.assert_array_equal(engine.ambient, reference['ambient'])
        np.testing.assert_array_equal(target_ray_counts(engine, [boxes[0]])[0],
                                      (reference['object_id'] == 0).sum(axis=(-3,-2,-1)))
        meta = engine.metadata
    finally:
        engine.close()
    value = dict(status='PASS', unit=unit, frames=frames, max_abs_expected_error=error,
                 target_ray_counts_equal=True, ambient_equal=True, query_contract_equal=True,
                 planar_geometry=geometry, renderer=meta,elapsed_s=time.monotonic()-started)
    save(OUT/'engineering/render_check.json', value)
    return value


def run():
    started = time.monotonic(); records = []
    engineering_check()
    for i, profile in enumerate(cohort()):
        record = render_one(profile['unit']); records.append(record)
        print(json.dumps(dict(stage='render', completed=i+1, total=24, unit=profile['unit'],
                              unit_seconds=round(record['elapsed_s'],3), total_seconds=round(time.monotonic()-started,3))), flush=True)
    result = dict(status='COMPLETE', units=[p['unit'] for p in cohort()], scenes=24,
                  schedules=RC.tolist(), replicas=4, displacement_variants=7,
                  elapsed_s=time.monotonic()-started, total_changed_exposures=sum(r['changed_exposures'] for r in records),
                  total_unique_physical_exposures=sum(r['unique_physical_exposures'] for r in records))
    save(OUT/'generation_receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    if args.check: print(json.dumps(engineering_check()), flush=True)
    elif args.run: print(json.dumps(run()), flush=True)
    else: parser.error('Choose --check or --run')
