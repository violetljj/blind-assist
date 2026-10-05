"""Fresh natural dual gated-fusion inputs; frozen M3, no training.

Reuse admitted renderer and batched projection unchanged. Truth is written by
the separate CPU-only geometry stage; predictor inputs contain no truth labels.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import cnh_dual_sensor_envelope_natural as N

ROOT = Path(__file__).resolve().parents[4]
PARENT = ROOT/'artifacts.local/work/cnh-dual-gated-fusion-20261005'
SOURCE = ROOT/'artifacts.local/work/cnh-margin-confirm-20261002'
FRAMES = np.arange(3, 16)
PHOTON_PREFIX = 2026100531


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def save(p, v):
    N.save(p, v)


def deadline():
    if time.time() >= read(PARENT/'PLAN.json')['deadline_unix']:
        raise TimeoutError('Inherited 2 hour wall-clock deadline reached')


def setup(batch):
    out = PARENT/f'natural{batch}'
    out.mkdir(parents=True, exist_ok=True)
    N.OUT = out
    N.deadline = deadline
    for key, name in [('TEMP', 'tmp'), ('TMP', 'tmp'), ('CUPY_CACHE_DIR', 'cupy-cache')]:
        p = PARENT/name
        p.mkdir(exist_ok=True)
        os.environ[key] = str(p)
    return out


def freeze():
    import cnh_margin_confirm as MC
    import cnh_displacement_ceiling_render as R
    deadline()
    parent = read(PARENT/'PLAN.json')
    sources = [Path(__file__), Path(N.__file__)]
    sources += [Path(__file__).with_name(name) for name in (
        'cnh_margin_confirm.py', 'cnh_near_range.py', 'cnh_structure_space.py',
        'cnh_proposal_attribution_scenes.py', 'cnh_cvr_pilot.py',
        'cnh_m3_pose_ensemble.py', 'cnh_cvr_projection.py', 'cnh_cvr_v2_materialize.py',
        'cnh_location_reference_gpu.py', 'cnh_displacement_ceiling_render.py',
        'cnh_temporal_readout_data.py', 'cnh_sequence_observed_geometry.py')]
    for batch, angles, units in [
        (95000, [-15, 15], parent['natural_calibration_units']),
        (97000, [0, -15, 15], parent['natural_evaluation_units'])]:
        out = setup(batch)
        save(out/'PLAN.json', dict(
            parent_plan_sha256=N.sha(PARENT/'PLAN.json'),
            deadline_unix=parent['deadline_unix'], units=units, angles_deg=angles,
            configs=parent['configs'], fallback_config_sets=parent['fallback_config_sets'],
            models_sha256={str(p): N.sha(p) for p in MC.model_paths('M3')},
            source_sha256={str(p): N.sha(p) for p in sources},
            renderer_sources=R.source_sha256(), photon_prefix=PHOTON_PREFIX,
            photon_rule='SeedSequence[prefix,unit,config,sensor] coarse signed aggregate law, independent streams',
            scene_rule='Original MC.scenes_for(unit), RNG SeedSequence[2026100105,unit]; natural97000 fresh unit IDs',
            pose_rule='Original shared head/noisy pose trajectory, CP.motion_metadata(unit,config); branch fixed extrinsic',
            inference='Unchanged admitted N.Predictor float64 geometry/float32 sequential history/float16 cache/5 seed M3',
            input_contract='raw[sensor,config,13,HEAD/BODY], sensors L/R for95000; S/L/R for97000',
            threshold=parent['dual_default_threshold'], training=False))
    save(PARENT/'NATURAL_SOURCE_CONTRACT.json', dict(
        source_sha256={str(p): N.sha(p) for p in sources},
        inherited_projection_receipt_sha256=N.sha(ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005/natural/engineering/batched_projection.json'),
        script_sha256=N.sha(__file__)))


def scenes_for(batch, unit):
    import cnh_margin_confirm as MC
    if batch == 95000:
        rows = read(SOURCE/'manifest_units/calib'/f'unit{unit}.json')
        return rows
    scenes = MC.scenes_for(unit)
    return [dict(split='evaluation', unit=unit, config=s['config'], group=int(s['group']),
                 off=-float(s['margin']), range=float(s['meta']['range']),
                 cond=s['family'], boxes=s['boxes']) for s in scenes]


def unit_run(batch, unit, configs, predictor):
    import cnh_displacement_ceiling_render as R
    import cnh_location_reference_gpu as G
    import cnh_cvr_pilot as CP
    from cnh_temporal_readout_data import normalized_z
    out = setup(batch)
    plan = read(out/'PLAN.json')
    angles = plan['angles_deg']
    path = out/'scores'/f'unit{unit}.npz'
    receipt = path.with_suffix('.json')
    if receipt.exists():
        old = read(receipt)
        assert old['score_sha256'] == N.sha(path)
        assert old['configs'] == configs
        return old
    deadline()
    tick = time.monotonic()
    rows = scenes_for(batch, unit)
    scene_path = out/'scene_units'/f'unit{unit}.json'
    if not scene_path.exists():
        save(scene_path, rows)
    ex = np.stack([N.extrinsic(a) for a in angles])
    sensor, _, _ = CP.motion_metadata(unit, 0)
    poses = (sensor[None]@ex[:, None]).reshape(len(angles)*16, 4, 4)
    hist, ambient = [], []
    for config in configs:
        deadline()
        row = rows[config]
        assert row['config'] == config
        engine = G.ExpectedRenderer(poses, row['boxes'])
        try:
            means = engine.background_expected(return_device=False).reshape(len(angles), 16, 8, 8, 16)
            amb = engine.ambient.reshape(len(angles), 16, 8, 8).copy()
            backend = engine.metadata
        finally:
            engine.close()
        samples = []
        for branch in range(len(angles)):
            seed = int(np.random.SeedSequence([PHOTON_PREFIX, unit, config, branch]).generate_state(1)[0])
            samples.append(R.sample(means[branch], amb[branch], seed)[0])
        hist.append(np.stack(samples))
        ambient.append(amb)
    hist = np.stack(hist, axis=1)
    ambient = np.stack(ambient, axis=1)
    z = normalized_z(hist, ambient)
    render_seconds = time.monotonic()-tick
    obs = out/'observations'/f'unit{unit}.npz'
    obs.parent.mkdir(exist_ok=True)
    assert not obs.exists()
    np.savez_compressed(obs, hist=hist, ambient=ambient, z1=z, configs=configs, unit=unit)
    it = time.monotonic()
    raw = predictor.predict(unit, configs, z, angles)
    infer_seconds = time.monotonic()-it
    path.parent.mkdir(exist_ok=True)
    assert not path.exists()
    np.savez_compressed(path, raw=raw, configs=configs, unit=unit, frames=FRAMES,
                        sensors=['L', 'R'] if batch == 95000 else ['S', 'L', 'R'])
    result = dict(status='COMPLETE', unit=unit, configs=configs, raw_shape=list(raw.shape),
                  score_sha256=N.sha(path), observation_sha256=N.sha(obs),
                  scene_sha256=N.sha(scene_path), render_seconds=render_seconds,
                  infer_seconds=infer_seconds, total_seconds=time.monotonic()-tick,
                  plan_sha256=N.sha(out/'PLAN.json'), backend=backend)
    save(receipt, result)
    return result


def parity(predictor):
    """Same retained single observation must produce original logits."""
    receipt = PARENT/'natural95000/engineering/parity.json'
    if receipt.exists():
        assert read(receipt)['status'] == 'PASS'
        return
    with np.load(SOURCE/'features/calib/unit95000.npz') as d:
        z = d['z1'][d['scene'] == 0][None, None]
    raw = predictor.predict(95000, [0], z, [0])[0, 0]
    with np.load(SOURCE/'frame_scores_M3_early.npz') as early, np.load(SOURCE/'frame_scores_M3.npz') as late:
        old = np.concatenate((early['95000'][0], late['95000'][0]))
    error = float(np.max(np.abs(raw-old)))
    assert error < 1e-5, error
    save(receipt, dict(status='PASS', raw_logit_max_abs_error=error,
         geometry_contract='Original admitted paired_project unchanged', gpu_device=predictor.torch.cuda.get_device_name()))


def run():
    setup(95000)
    parent = read(PARENT/'PLAN.json')
    predictor = N.Predictor()
    try:
        parity(predictor)
        receipts = []
        for batch, units in [(95000, parent['natural_calibration_units']),
                             (97000, parent['natural_evaluation_units'])]:
            out = setup(batch)
            selection = out/'selected_configs.json'
            if selection.exists():
                configs = read(selection)['configs']
            else:
                configs = parent['configs']
                if batch == 97000:
                    mean_seconds = float(np.mean([r['total_seconds'] for r in receipts]))
                    remaining = parent['deadline_unix']-time.time()-600
                    for candidate in parent['fallback_config_sets']:
                        configs = candidate
                        predicted = mean_seconds*1.5*len(units)*len(configs)/40
                        if predicted <= remaining:
                            break
                    details = dict(predicted_seconds=predicted, available_seconds_after_600s_reserve=remaining,
                                   calibration_mean_unit_seconds=mean_seconds)
                else:
                    details = {}
                save(selection, dict(configs=configs, reason='Runtime-only uniform config selection before batch scientific scores; no outcome-dependent reduction', **details))
            batch_receipts = []
            for unit in units:
                r = unit_run(batch, unit, configs, predictor)
                receipts.append(r)
                batch_receipts.append(r)
                progress = dict(stage='render/infer', batch=batch, completed=len(batch_receipts),
                                total=len(units), last_unit=unit, configs=configs, last_activity_unix=time.time())
                temp = out/'progress.tmp.json'
                temp.write_text(json.dumps(progress, indent=2)+'\n', encoding='utf8')
                temp.replace(out/'progress.json')
                print('natural', batch, unit, len(configs), round(r['total_seconds'], 2), flush=True)
            save(out/'run.json', dict(status='COMPLETE', units=units, configs=configs,
                 total_seconds=sum(r['total_seconds'] for r in batch_receipts), script_sha256=N.sha(__file__)))
    finally:
        predictor.close()


def geometry(batch):
    """Truth only. Preserve original positive-area surface and deadline rules."""
    import cnh_cvr_pilot as CP
    import cnh_sequence_observed_geometry as GE
    import cnh_proposal_attribution_scenes as S
    out = setup(batch)
    plan = read(out/'PLAN.json')
    configs = read(out/'selected_configs.json')['configs']
    keys = ('unit', 'config', 'query', 'split', 'frame_ranges', 'frame_category',
            'clear_all', 'ref_category', 'covered', 'last_category', 'frame_poses',
            'reference_pose', 'reference_fraction', 'reference_time_s',
            'crossing_left_frame', 'crossing_right_frame', 'interpolation_alpha', 'censor_reason')
    values = {k: [] for k in keys}
    tick = time.monotonic()
    manifest = []
    for unit in plan['units']:
        deadline()
        rows = scenes_for(batch, unit)
        _, travel, _ = CP.motion_metadata(unit, 0)
        poses = travel[FRAMES]
        for config in configs:
            row = rows[config]
            manifest.append(row)
            mesh = np.concatenate([S.box_mesh(b['lo'], b['hi']) for b in row['boxes']])
            ranges, reference = GE.deadline_reference(GE.corners(row['boxes'][0]), poses)
            local = [GE.transform(mesh, pose) for pose in poses]
            local_ref = GE.transform(mesh, reference['reference_pose']) if reference['covered'] else None
            for query in (0, 1):
                cats = np.asarray([GE.surface_category(t, query) for t in local])
                entry = dict(unit=unit, config=config, query=query,
                    split='calib' if batch == 95000 else 'evaluation', frame_ranges=ranges,
                    frame_category=cats, clear_all=bool(np.all(cats == 'clear')),
                    ref_category=GE.surface_category(local_ref, query) if reference['covered'] else 'censored',
                    last_category=cats[-1], frame_poses=poses, **reference)
                for key in keys:
                    values[key].append(entry[key])
        print('geometry', batch, unit, round(time.monotonic()-tick, 2), flush=True)
    dest = out/'geometry.npz'
    assert not dest.exists()
    payload = {k: np.asarray(v) for k,v in values.items()}
    payload['frames'] = FRAMES
    assert np.array_equal(payload['clear_all'], (payload['frame_category'] == 'clear').all(1))
    assert np.array_equal(payload['covered'], payload['ref_category'] != 'censored')
    assert (np.diff(payload['frame_ranges'], axis=1) <= 1e-8).all()
    np.savez_compressed(dest, **payload)
    save(out/'scene_manifest.json', manifest)
    save(out/'geometry_receipt.json', dict(status='COMPLETE', geometry_sha256=N.sha(dest),
         elapsed_seconds=time.monotonic()-tick, query_episodes=len(payload['unit']), scores_read=False,
         source_sha256=N.sha(GE.__file__), scene_manifest_sha256=N.sha(out/'scene_manifest.json')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['freeze', 'run', 'geometry'])
    ap.add_argument('--batch', type=int, choices=[95000, 97000], default=97000)
    args = ap.parse_args()
    try:
        if args.stage == 'freeze':
            freeze()
        elif args.stage == 'run':
            run()
        else:
            geometry(args.batch)
    except BaseException as error:
        save(PARENT/'natural_failures'/f'{args.stage}-{time.time_ns()}.json',
             dict(status='FAILED', error=repr(error), stage=args.stage, unix=time.time()))
        raise
