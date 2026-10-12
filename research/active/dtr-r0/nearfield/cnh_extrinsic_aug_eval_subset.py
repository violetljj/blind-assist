"""Pilot-gated fresh evaluation with runtime-only uniform config fallback.

Selection count is frozen before any evaluation scene or score is opened.
Physical boxes, three views, noise law, geometry and labels remain unchanged.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import cnh_extrinsic_aug_data as D

OUT = D.OUT


def configs_for(unit, count):
    assert 1 <= count <= 40
    return list(range(40)) if count == 40 else sorted(int(x) for x in
        np.random.default_rng(2026100609+int(unit)).choice(40, count, replace=False))


def unit_run(unit, index, configs, projector, p):
    dest = OUT/'features/evaluation'/f'unit{unit}.npy'
    receipt_path = dest.with_suffix('.json')
    if receipt_path.exists():
        old = D.read(receipt_path)
        assert old['configs'] == configs and old['feature_sha256'] == D.sha(dest)
        return old
    tick = time.monotonic()
    mirrored = (unit//3) % 2 == 1
    all_scenes = D.MC.scenes_for(unit)
    histograms, ambients, zs, nn, qq, descriptions = [], [], [], [], [], []
    angles = [0., -15., 15.]
    for config in configs:
        D.plan()
        scene = all_scenes[config]
        boxes = D.mirror_boxes(scene['boxes'], mirrored)
        sensor, travel, noisy, query = D.base_metadata(unit, config, 0, mirrored, p)
        hist, ambient, z, backend = D.render(boxes, sensor, angles, unit, config, p)
        histograms.append(hist); ambients.append(ambient); zs.append(z)
        nn.append(noisy); qq.append(query)
        descriptions.append(dict(unit=unit, config=config, group=int(scene['group']),
            off=-float(scene['margin']), range=float(scene['meta']['range']), cond=scene['family'], boxes=boxes))
    z = np.stack(zs, axis=1)
    noisy_center = np.stack(nn)
    public_query = np.stack(qq)
    ex = np.stack([D.N.extrinsic(angle) for angle in angles])
    noisy = noisy_center[None]@ex[:, None, None]
    queries = public_query[None]@ex[:, None, None]
    features = projector.features(z.reshape(-1, 16, 8, 8, 16),
        noisy.reshape(-1, 16, 4, 4), queries.reshape(-1, 16, 4, 4))
    features = features.reshape(3, len(configs), 13, 3, 24, 17, 33)
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()
    np.save(dest, features)
    obs = OUT/'observations/evaluation'/f'unit{unit}.npz'
    obs.parent.mkdir(parents=True, exist_ok=True)
    assert not obs.exists()
    np.savez_compressed(obs, hist=np.stack(histograms, axis=1), ambient=np.stack(ambients, axis=1),
        z1=z, sensor_center=sensor, travel=travel, noisy_center=noisy_center,
        public_query=query, angles=angles, configs=configs, mirror=mirrored, sensors=['S','L','R'])
    truth_path = OUT/'truth/evaluation'/f'unit{unit}.json'
    D.save(truth_path, dict(unit=unit, mirror=mirrored, mode=unit % 3, configs=configs,
        scenes=descriptions, travel=travel.tolist(), public_query=query.tolist(), sensor_center=sensor.tolist()))
    r = dict(status='COMPLETE', unit=unit, samples=3*len(configs)*13, configs=configs,
        feature_shape=list(features.shape), feature_sha256=D.sha(dest), observation_sha256=D.sha(obs),
        truth_sha256=D.sha(truth_path), elapsed_seconds=time.monotonic()-tick,
        plan_sha256=D.sha(OUT/'PLAN.json'), script_sha256=D.sha(__file__),
        base_data_source_sha256=D.sha(D.__file__), backend=backend)
    D.save(receipt_path, r)
    return r


def run(count, reason):
    D.setup()
    p = D.require_evaluation()
    units = p['natural_evaluation_units']
    selection_path = OUT/'evaluation_config_selection.json'
    selections = {str(unit): configs_for(unit, count) for unit in units}
    if selection_path.exists():
        selection = D.read(selection_path)
        assert selection['config_count_per_unit'] == count and selection['configs_by_unit'] == selections
    else:
        D.save(selection_path, dict(status='FROZEN_BEFORE_EVALUATION_SCENES',
            config_count_per_unit=count, configs_by_unit=selections,
            rule='Sorted uniform sample without replacement; RNG default_rng(2026100609+unit); common views use same scene IDs',
            selection_basis='Only runtime forecast after pilot gate; no evaluation scene/scores inspected',
            reason=reason, plan_sha256=D.sha(OUT/'PLAN.json'), script_sha256=D.sha(__file__)))
    projector = D.CachedProjector()
    receipts = []
    try:
        for index, unit in enumerate(units):
            r = unit_run(unit, index, selections[str(unit)], projector, p)
            receipts.append(r)
            print('evaluation subset', index+1, len(units), unit, count, round(r['elapsed_seconds'], 2), flush=True)
    finally:
        projector.close()
    metadata = {name: [] for name in ('unit', 'sensor', 'config', 'frame')}
    for unit in units:
        for sensor in range(3):
            for config in selections[str(unit)]:
                for frame in D.FRAMES:
                    for key, value in zip(metadata, (unit, sensor, config, int(frame))):
                        metadata[key].append(value)
    folder = OUT/'inputs/evaluation'
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'rows.npz', **{key:np.asarray(value) for key,value in metadata.items()})
    D.save(OUT/'features/evaluation/receipt.json', dict(status='COMPLETE', units=units,
        config_count_per_unit=count, configs_by_unit=selections, angles_deg=[0., -15., 15.],
        samples=len(units)*3*count*13, total_seconds=sum(r['elapsed_seconds'] for r in receipts),
        rows_sha256=D.sha(folder/'rows.npz'), selection_sha256=D.sha(selection_path),
        plan_sha256=D.sha(OUT/'PLAN.json'), script_sha256=D.sha(__file__)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', choices=range(1, 41), type=int, default=40)
    ap.add_argument('--reason', required=True, help='Runtime-only frozen rationale')
    args = ap.parse_args()
    try:
        run(args.count, args.reason)
    except BaseException as error:
        D.save(OUT/'failures'/f'eval-subset-{time.time_ns()}.json',
               dict(status='FAILED', error=repr(error), unix=time.time()))
        raise
