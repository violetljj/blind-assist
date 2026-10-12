"""Fresh extrinsic-augmented V inputs; original CVR/query/label contract.

Physical boxes and all poses are mirrored together. Installation yaw is known
to projection, not an extra model input. Evaluation generation is pilot gated.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import cnh_cvr_pilot as CP
import cnh_dual_sensor_envelope_natural as N
import cnh_margin_confirm as MC
import cnh_near_range as NR
import cnh_proposal_attribution_scenes as S
import cnh_sequence_observed_geometry as GE
import cnh_structure_space as SS
import cnh_m3_pose_ensemble as PE

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-extrinsic-aug-20261006'
FRAMES = np.arange(3, 16)
REFLECT = np.diag([-1., 1., 1., 1.])
DLL_HANDLES = []


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    assert not p.exists(), 'Preserve sealed input: '+str(p)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    for key, folder in [('TEMP', 'tmp'), ('TMP', 'tmp'), ('CUPY_CACHE_DIR', 'cupy-cache')]:
        p = OUT/folder
        p.mkdir(exist_ok=True)
        os.environ[key] = str(p)
    if os.name == 'nt' and not DLL_HANDLES:
        dll = Path(CP.torch.__file__).parent.parent/'nvidia/cublas/bin'
        if dll.is_dir():
            DLL_HANDLES.append(os.add_dll_directory(str(dll)))
            os.environ['PATH'] = str(dll)+os.pathsep+os.environ['PATH']


def plan():
    p = read(OUT/'PLAN.json')
    if time.time() >= p['deadline_unix']:
        raise TimeoutError('Inherited 4 hour wall-clock deadline reached')
    # Parent stores inclusive ID ranges; materialize IDs without touching data.
    train, cal, ev = p['training'], p['calibration'], p['evaluation']
    p['natural_train_units'] = list(range(train['natural_units'][0], train['natural_units'][1]+1))
    p['train_units'] = [u for u in range(train['displacement_units'][0], train['displacement_units'][1]+1)
                        if u % 3 in train['displacement_mod3_in']]
    p['natural_calibration_units'] = list(range(cal['units'][0], cal['units'][1]+1))
    p['natural_evaluation_units'] = list(range(ev['units'][0], ev['units'][1]+1))
    return p


def require_evaluation():
    p = plan()
    gate = read(OUT/p['calibration']['pilot_gate_path'])
    assert gate.get('judgment') == 'PASS', 'Do not access fresh evaluation before pilot PASS'
    return p


def mirror_pose(pose, mirrored):
    return REFLECT@np.asarray(pose)@REFLECT if mirrored else np.asarray(pose).copy()


def mirror_boxes(boxes, mirrored):
    out = copy.deepcopy(boxes)
    if mirrored:
        for b in out:
            oldlo, oldhi = b['lo'][0], b['hi'][0]
            b['lo'][0], b['hi'][0] = -oldhi, -oldlo
    return out


def natural_train_scenes(unit):
    """Exact old V 10 none + 12 panel distribution with fresh unit-keyed RNG."""
    ref = NR.known(unit)[0]
    rng = np.random.default_rng([2026100104, int(unit)])
    out = []
    for config in range(22):
        t = NR._target(rng, S.MARGINS[config % 6]+rng.uniform(-.006, .006), (unit+config) % 2)
        if config < 10:
            scene = NR._none_scene(S, unit, config, ref, t)
        else:
            shape = SS.draw_shape(rng)
            scene = SS.panel_scene(unit, config, ref, t, shape,
                SS.EDGES[0]+float(rng.uniform())*(SS.EDGES[5]-SS.EDGES[0]))
        scene['meta'] = dict(scene.get('meta', {}), range=t['z'])
        out.append(scene)
    return out


def displacement_scene(unit, index, p):
    """Old V displacement support and fixed-delta nuisance rejection."""
    from cnh_expected_separability import target_at
    rng = np.random.default_rng([p.get('scene_seed_prefix', 2026100605), int(unit)])
    delta = float(rng.uniform(-25., 10.))
    ref = NR.known(unit)[0]
    assert unit % 3 in (0, 1)
    group, context = (index//2) % 2, 'none' if (index//4) % 2 == 0 else 'panel'
    attempts = 0
    while True:
        target = NR._target(rng, -.01, group)
        shift = .65-target['z']
        target['box']['lo'][2] += shift
        target['box']['hi'][2] += shift
        target['z'] = .65
        if context == 'none':
            scene = NR._none_scene(S, unit, 0, ref, target)
        else:
            shape = dict(SS.draw_shape(rng), same_side=1, gap=float(rng.uniform(.34, .40)))
            scene = SS.panel_scene(unit, 0, ref, target, shape,
                float(rng.uniform(SS.EDGES[0], SS.EDGES[5])))
        scene['boxes'][0] = target_at(target, -delta/100)
        collision = any(np.all(np.minimum(scene['boxes'][0]['hi'], b['hi'])-
                               np.maximum(scene['boxes'][0]['lo'], b['lo']) > 0)
                        for b in scene['boxes'][1:])
        if not collision:
            scene['meta'] = dict(scene.get('meta', {}), range=.65)
            scene['family'] = context
            return scene, delta, attempts
        attempts += 1
        assert attempts <= 1000, 'Fixed-delta nuisance rejection exhausted'


def angle_for(unit, config, replica, sequence_index, p):
    if sequence_index % 4 == 0:
        return 0.
    rng = np.random.default_rng([p.get('augmentation_seed_prefix', 2026100608), unit, config, replica])
    return float(rng.uniform(-30., 30.))


def base_metadata(unit, config, replica, mirrored, p):
    from cnh_track_a_readout import noisy_poses
    sensor, travel, _ = CP.motion_metadata(unit, config)
    seed = int(np.random.SeedSequence([p.get('noise_seed_prefix', 2026100607), unit, config, replica]).generate_state(1)[0])
    noisy = noisy_poses(sensor, seed, dt=.2)
    query = np.stack([PE.public_query(unit, frame) for frame in range(16)])
    result = tuple(mirror_pose(x, mirrored) for x in (sensor, travel, noisy, query))
    # Public nominal query equals the frozen mount schedule; no scene anchor.
    np.testing.assert_allclose(np.linalg.inv(result[1])@result[0], result[3], atol=1e-12, rtol=0)
    return result


def categories(boxes, travel):
    mesh = np.concatenate([S.box_mesh(b['lo'], b['hi']) for b in boxes])
    return np.asarray([[GE.surface_category(GE.transform(mesh, pose), query)
                        for query in (0, 1)] for pose in travel])


class CachedProjector:
    """Admitted config-batched geometry, without allocating predictor networks."""
    paired_project = N.Predictor.paired_project

    def __init__(self):
        import torch
        from cnh_cvr_v2_materialize import BatchedProjector
        self.torch = torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.projector = BatchedProjector()

    def features(self, z, noisy, public_query):
        n = len(z)
        result = np.empty((n, 13, 3, 24, 17, 33), np.float16)
        for begin in range(0, n, 4):
            nn, qq = noisy[begin:begin+4], public_query[begin:begin+4]
            for fi, frame in enumerate(FRAMES):
                plan()
                ix = np.arange(max(0, int(frame)-7), int(frame)+1)
                matrices = (qq[:, frame]@np.linalg.inv(nn[:, frame]))[:, None]@nn[:, ix]
                result[begin:begin+4, fi] = self.paired_project(z[begin:begin+4][:, ix], matrices).cpu().numpy().astype(np.float16)
        return result

    def close(self):
        self.projector = None
        gc.collect()
        self.torch.cuda.empty_cache()


def render(boxes, sensor_center, angles, unit, config, p):
    import cnh_location_reference_gpu as G
    import cnh_displacement_ceiling_render as R
    from cnh_temporal_readout_data import normalized_z
    poses = np.stack([sensor_center@N.extrinsic(a) for a in angles])
    engine = G.ExpectedRenderer(poses.reshape(-1, 4, 4), boxes)
    try:
        expectation = engine.background_expected(return_device=False).reshape(len(angles), 16, 8, 8, 16)
        ambient = engine.ambient.reshape(len(angles), 16, 8, 8).copy()
        backend = engine.metadata
    finally:
        engine.close()
    hist = np.stack([R.sample(expectation[branch], ambient[branch],
        int(np.random.SeedSequence([p.get('photon_seed_prefix', 2026100606), unit, config, branch]).generate_state(1)[0]))[0]
        for branch in range(len(angles))])
    return hist, ambient, normalized_z(hist, ambient), backend


def training_unit(domain, unit, index, projector):
    p = plan()
    folder = OUT/'train_units'/domain
    receipt_path = folder/f'unit{unit}.json'
    if receipt_path.exists():
        old = read(receipt_path)
        assert sha(folder/f'unit{unit}.npz') == old['payload_sha256']
        return old
    tick = time.monotonic()
    mirrored = (unit//3) % 2 == 1 if domain == 'natural' else (index//8) % 2 == 1
    if domain == 'natural':
        scenes = natural_train_scenes(unit)
        repeats = 1
        nuisance = 0
    else:
        scene, delta, nuisance = displacement_scene(unit, index, p)
        scenes, repeats = [scene], 2
    zs, noises, queries, angles, histograms, ambients, rows, descriptions = [], [], [], [], [], [], [], []
    for config, scene in enumerate(scenes):
        boxes = mirror_boxes(scene['boxes'], mirrored)
        sensor, travel, _, query = base_metadata(unit, config, 0, mirrored, p)
        cats = categories(boxes, travel)
        front = np.asarray([GE.target_front(GE.corners(boxes[0]), pose) for pose in travel])
        sequence_index = index*22+config if domain == 'natural' else len(p['natural_train_units'])*22+index
        # Physical installation is fixed across photon replicas of one scene.
        angle = angle_for(unit, config, 0, sequence_index, p)
        aa = [angle]*repeats
        hist, ambient, z, backend = render(boxes, sensor, aa, unit, config, p)
        descriptions.append(dict(config=config, boxes=boxes, group=int(scene['group']),
            family=scene['family'], range=float(scene['meta']['range']),
            intrusion_cm=delta if domain == 'displacement' else None))
        for replica, angle in enumerate(aa):
            _, _, noisy, _ = base_metadata(unit, config, replica, mirrored, p)
            ex = N.extrinsic(angle)
            zs.append(z[replica]); noises.append(noisy@ex); queries.append(query@ex)
            angles.append(angle); histograms.append(hist[replica]); ambients.append(ambient[replica])
            for frame in FRAMES:
                rows.append(dict(unit=unit, config=config, replica=replica, frame=int(frame),
                    domain=0 if domain == 'natural' else 1, group=int(scene['group']),
                    front=float(front[frame]), category=cats[frame].tolist(),
                    mirror=mirrored, angle_deg=angle))
    voxels = projector.features(np.stack(zs), np.stack(noises), np.stack(queries)).reshape(-1, 3, 24, 17, 33)
    cc = np.asarray([r['category'] for r in rows])
    labels = np.char.startswith(cc, 'contact').astype(np.float32)
    mask = (cc != 'pass0-10cm').astype(np.float32)
    front = np.asarray([r['front'] for r in rows])
    weights = mask*np.where((front >= 1.6)&(front < 2.6), 2., 1.)[:, None]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder/f'unit{unit}.npz'
    assert not dest.exists()
    np.savez_compressed(dest, voxels=voxels, labels=labels, mask=mask,
        weights=weights.astype(np.float32), z1=np.stack(zs), hist=np.stack(histograms),
        ambient=np.stack(ambients), noisy=np.stack(noises), public_query=np.stack(queries),
        travel=travel, sensor_center=sensor, angles=angles)
    truth_path = OUT/'truth/train'/domain/f'unit{unit}.json'
    save(truth_path, dict(unit=unit, mirror=mirrored, domain=domain, scenes=descriptions, rows=rows,
                         travel=travel.tolist(), sensor_center=sensor.tolist()))
    result = dict(status='COMPLETE', unit=unit, domain=domain, samples=len(rows), mirror=mirrored,
         angles_deg=angles, nuisance_rejections=nuisance, elapsed_seconds=time.monotonic()-tick,
         payload_sha256=sha(dest), truth_sha256=sha(truth_path), backend=backend,
         plan_sha256=sha(OUT/'PLAN.json'), script_sha256=sha(__file__))
    save(receipt_path, result)
    return result


def generate_train():
    setup()
    p = plan()
    projector = CachedProjector()
    receipts = []
    try:
        for domain, units in [('natural', p['natural_train_units']), ('displacement', p['train_units'])]:
            for index, unit in enumerate(units):
                r = training_unit(domain, unit, index, projector)
                receipts.append(r)
                print('train inputs', domain, index+1, len(units), unit, round(r['elapsed_seconds'], 2), flush=True)
    finally:
        projector.close()
    assemble_train(p, receipts)


def assemble_train(p, receipts):
    n = sum(r['samples'] for r in receipts)
    assert n == (len(p['natural_train_units'])*22+len(p['train_units'])*2)*13
    folder = OUT/'inputs/train'
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder/'receipt.json').exists()
    arrays = dict(voxels=np.lib.format.open_memmap(folder/'voxels.npy', mode='w+', dtype=np.float16,
                       shape=(n, 3, 24, 17, 33)))
    for name in ('labels', 'mask', 'weights'):
        arrays[name] = np.lib.format.open_memmap(folder/f'{name}.npy', mode='w+', dtype=np.float32, shape=(n, 2))
    row_keys = ('unit', 'config', 'replica', 'frame', 'domain', 'group', 'front', 'category', 'mirror', 'angle_deg')
    metadata = {k: [] for k in row_keys}
    start = 0
    for r in receipts:
        unit, domain = r['unit'], r['domain']
        with np.load(OUT/'train_units'/domain/f'unit{unit}.npz') as d:
            stop = start+r['samples']
            for name in arrays:
                arrays[name][start:stop] = d[name]
        rows = read(OUT/'truth/train'/domain/f'unit{unit}.json')['rows']
        for key in row_keys:
            metadata[key].extend(row[key] for row in rows)
        start = stop
    assert start == n
    md = {k: np.asarray(v) for k,v in metadata.items()}
    mass = {}
    for domain in (0, 1):
        keep = md['domain'] == domain
        mass[str(domain)] = float(arrays['weights'][keep].sum(dtype=np.float64))
        arrays['weights'][keep] *= np.float32((n/2)/mass[str(domain)])
    for a in arrays.values():
        a.flush()
    arrays.clear()
    np.savez_compressed(folder/'rows.npz', **md)
    save(folder/'receipt.json', dict(status='COMPLETE', samples=n, rows=n, raw_loss_mass=mass,
        plan_sha256=sha(OUT/'PLAN.json'), outputs={name: sha(folder/name) for name in
        ('voxels.npy', 'labels.npy', 'mask.npy', 'weights.npy', 'rows.npz')},
        weight_rule='Original V: contact hard1/clear0/pass mask, far2, each domain sumN/2'))


def natural_unit(split, unit, index, projector, p):
    dest = OUT/'features'/split/f'unit{unit}.npy'
    receipt_path = dest.with_suffix('.json')
    if receipt_path.exists():
        r = read(receipt_path)
        assert r['feature_sha256'] == sha(dest)
        return r
    tick = time.monotonic()
    mirrored = (unit//3) % 2 == 1
    scenes = MC.scenes_for(unit)
    histograms, ambients, zs, nn, qq, descriptions = [], [], [], [], [], []
    angles = [0., -15., 15.]
    for config, scene in enumerate(scenes):
        plan()
        boxes = mirror_boxes(scene['boxes'], mirrored)
        sensor, travel, noisy, query = base_metadata(unit, config, 0, mirrored, p)
        hist, ambient, z, backend = render(boxes, sensor, angles, unit, config, p)
        histograms.append(hist); ambients.append(ambient); zs.append(z)
        nn.append(noisy); qq.append(query)
        descriptions.append(dict(unit=unit, config=config, group=int(scene['group']),
             off=-float(scene['margin']), range=float(scene['meta']['range']), cond=scene['family'], boxes=boxes))
    z = np.stack(zs, axis=1)
    noisy_center = np.stack(nn)
    public_query = np.stack(qq)
    ex = np.stack([N.extrinsic(a) for a in angles])
    noisy = noisy_center[None]@ex[:, None, None]
    queries = public_query[None]@ex[:, None, None]
    features = projector.features(z.reshape(-1, 16, 8, 8, 16),
        noisy.reshape(-1, 16, 4, 4), queries.reshape(-1, 16, 4, 4))
    features = features.reshape(3, 40, 13, 3, 24, 17, 33)
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()
    np.save(dest, features)
    obs = OUT/'observations'/split/f'unit{unit}.npz'
    obs.parent.mkdir(parents=True, exist_ok=True)
    assert not obs.exists()
    np.savez_compressed(obs, hist=np.stack(histograms, axis=1), ambient=np.stack(ambients, axis=1),
         z1=z, sensor_center=sensor, travel=travel, noisy_center=noisy_center,
         public_query=query, angles=angles, configs=np.arange(40), mirror=mirrored, sensors=['S','L','R'])
    truth_path = OUT/'truth'/split/f'unit{unit}.json'
    save(truth_path, dict(unit=unit, mirror=mirrored, mode=unit % 3, scenes=descriptions,
         travel=travel.tolist(), public_query=query.tolist(), sensor_center=sensor.tolist()))
    result = dict(status='COMPLETE', unit=unit, samples=3*40*13, mirror=mirrored,
         feature_shape=list(features.shape), feature_sha256=sha(dest), observation_sha256=sha(obs),
         truth_sha256=sha(truth_path), elapsed_seconds=time.monotonic()-tick,
         plan_sha256=sha(OUT/'PLAN.json'), script_sha256=sha(__file__), backend=backend)
    save(receipt_path, result)
    return result


def generate_natural(split):
    setup()
    p = require_evaluation() if split == 'evaluation' else plan()
    units = p['natural_evaluation_units'] if split == 'evaluation' else p['natural_calibration_units']
    projector = CachedProjector()
    receipts = []
    try:
        for index, unit in enumerate(units):
            r = natural_unit(split, unit, index, projector, p)
            receipts.append(r)
            print('natural features', split, index+1, len(units), unit, round(r['elapsed_seconds'], 2), flush=True)
    finally:
        projector.close()
    metadata = {name: [] for name in ('unit', 'sensor', 'config', 'frame')}
    for unit in units:
        for sensor in range(3):
            for config in range(40):
                for frame in FRAMES:
                    for key, value in zip(metadata, (unit, sensor, config, int(frame))):
                        metadata[key].append(value)
    folder = OUT/'inputs'/split
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'rows.npz', **{k:np.asarray(v) for k,v in metadata.items()})
    save(OUT/'features'/split/'receipt.json', dict(status='COMPLETE', units=units,
        configs=list(range(40)), angles_deg=[0., -15., 15.], samples=len(units)*3*40*13,
        total_seconds=sum(r['elapsed_seconds'] for r in receipts),
        rows_sha256=sha(folder/'rows.npz'), plan_sha256=sha(OUT/'PLAN.json'), script_sha256=sha(__file__)))


def engineering_cpu():
    # Proper reflected rotations, equal sensor-relative geometry, unchanged
    # public query labels and correct known-extrinsic/history conjugation.
    rows = []
    for unit in (100000, 100001, 100002):
        sensor, travel, noisy = CP.motion_metadata(unit, 0)
        mirrored_sensor = mirror_pose(sensor, True)
        np.testing.assert_allclose(np.linalg.det(mirrored_sensor[:, :3, :3]), 1., atol=1e-12)
        np.testing.assert_allclose(mirror_pose(mirrored_sensor, True), sensor, atol=1e-12)
        for angle in (-30., -15., 0., 15., 30.):
            np.testing.assert_allclose(REFLECT@N.extrinsic(angle)@REFLECT, N.extrinsic(-angle), atol=1e-12)
            frame = 10
            q = PE.public_query(unit, frame)
            ex = N.extrinsic(angle)
            actual = (q@ex)@np.linalg.inv(noisy[frame]@ex)@(noisy[3:frame+1]@ex)
            expected = q@np.linalg.inv(noisy[frame])@noisy[3:frame+1]@ex
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        boxes = MC.scenes_for(unit)[0]['boxes']
        original = categories(boxes, travel)
        mirrored = categories(mirror_boxes(boxes, True), mirror_pose(travel, True))
        np.testing.assert_array_equal(original, mirrored)
        rows.append(dict(unit=unit, determinant=1, public_query_mirror_and_branch_swap=True,
                         all_object_labels_reflection_invariant=True))
    save(OUT/'engineering/cpu.json', dict(status='PASS', rows=rows,
         backend='CPU geometry/algebra; TASK_NOT_GPU_SUITABLE', script_sha256=sha(__file__)))


def engineering_gpu():
    setup()
    p = plan()
    import cnh_displacement_ceiling as D
    import cnh_location_reference_gpu as G
    projector = CachedProjector()
    rows = []
    try:
        old = ROOT/'artifacts.local/work/cnh-margin-confirm-20261002/features/calib/unit95000.npz'
        with np.load(old) as d:
            z = np.stack([d['z1'][d['scene'] == config] for config in range(4)])
        noisy = np.stack([CP.motion_metadata(95000, config)[2] for config in range(4)])
        for frame in (3, 8, 15):
            ix = np.arange(max(0, frame-7), frame+1)
            q = PE.public_query(95000, frame)
            ts = (q@np.linalg.inv(noisy[:, frame]))[:, None]@noisy[:, ix]
            actual = projector.paired_project(z[:, ix], ts)
            expected = projector.torch.cat([D.project_many(projector.projector, z[c, ix][None], ts[c]) for c in range(4)])
            assert projector.torch.equal(actual, expected)
            rows.append(dict(frame=frame, configs=4, projected_voxels_bitwise_equal=True))
        scene = MC.scenes_for(p['natural_calibration_units'][0])[0]
        sensor = scene['poses'][[0, 13]]
        original = G.ExpectedRenderer(sensor@N.extrinsic(-15), scene['boxes'])
        reflected = G.ExpectedRenderer(mirror_pose(sensor, True)@N.extrinsic(15), mirror_boxes(scene['boxes'], True))
        try:
            expected = original.background_expected(return_device=False)[:, :, ::-1, :]
            actual = reflected.background_expected(return_device=False)
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-12)
            rows.append(dict(physical_reflection_max_abs_error=float(np.max(np.abs(actual-expected))),
                             horizontal_zone_reflection=True, backend=original.metadata))
        finally:
            original.close(); reflected.close()
    finally:
        projector.close()
    save(OUT/'engineering/gpu.json', dict(status='PASS', rows=rows, script_sha256=sha(__file__)))
    snapshot = OUT/'source'/Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    assert not snapshot.exists()
    snapshot.write_bytes(Path(__file__).read_bytes())
    save(OUT/'DATA_CONTRACT.json', dict(status='FROZEN', parent_plan_sha256=sha(OUT/'PLAN.json'),
         script_sha256=sha(__file__), snapshot_sha256=sha(snapshot),
         seeds=dict(scene=2026100605, photon=2026100606, pose=2026100607, augmentation=2026100608),
         natural_scene_seed_prefixes=[2026100104,2026100105],
         training_rows=39936, zero_yaw='Physical sequence global indexmod4==0, repeated photons shareyaw',
         interface='Train inputs/train/*.npy; natural features/split/unit.npy[S,L,R;C40;F13;3;24;17;33]',
         models_not_allocated=True, backend='Admitted CuPy renderer and Torch FP64 batchedProjector'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['engineering-cpu', 'engineering-gpu', 'train', 'calibration', 'train-calibration', 'evaluation'])
    args = ap.parse_args()
    try:
        if args.stage == 'engineering-cpu':
            engineering_cpu()
        elif args.stage == 'engineering-gpu':
            engineering_gpu()
        elif args.stage == 'train':
            generate_train()
        elif args.stage == 'train-calibration':
            generate_train()
            generate_natural('calibration')
        else:
            generate_natural(args.stage)
    except BaseException as error:
        save(OUT/'failures'/f'{args.stage}-{time.time_ns()}.json', dict(stage=args.stage,
             status='FAILED', error=repr(error), unix=time.time()))
        raise
