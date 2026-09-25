"""Execution-location recovery for only the frozen 960 worker Street frames.

Consumes already fitted alley bias/thresholds; imports unchanged frozen arms.
No alley execution, fitting, K selection, new capture or other source access.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from cnh_r1_readout_run import arms, synthesize, sha, write
from cnh_rgb_visible_depth_audit import load_scene_depth
from cnh_street_e2e_materialize import frame_identity


RAW_MANIFEST_SHA = '25ced0da709c149cd0d78db3fbe50115aa5a8165fca8ebf68c69e8594b97f758'
MATERIAL_MANIFEST_SHA = 'f093a5ed14ba9efb1002e642f33e2ed309c09f5fb0d2808cf2248079a8a5540e'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run(bundle, capture, output):
    started = time.monotonic()
    binding = read(bundle / 'binding.json')
    for name, expected in binding['files'].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path) != expected:
            raise ValueError('Frozen bundle file differs: ' + name)
    if sha(Path(__file__)) != binding['files']['code/' + Path(__file__).name]:
        raise ValueError('Runner identity differs')
    if sha(capture / 'raw-manifest.json') != RAW_MANIFEST_SHA:
        raise ValueError('Only original Street worker raw source allowed')
    material_root = bundle / 'inputs/worker'
    if sha(material_root / 'manifest.json') != MATERIAL_MANIFEST_SHA:
        raise ValueError('Only original Street worker materialization allowed')
    manifest = read(material_root / 'manifest.json')
    raw_rows = read(capture / 'raw-manifest.json')['frames']
    rows = manifest['frames']
    if (len(rows) != 960 or len(raw_rows) != 960 or
            any(r.get('data_role') != 'Development' for r in rows + raw_rows)):
        raise ValueError('Exactly frozen 960 Development frames required')
    by_id = {row['id']: row for row in raw_rows}
    frames = []
    for row in rows:
        raw = by_id[row['id']]
        if any(raw[key] != row[key] for key in ('layout_id', 'clip_id', 'pose_index', 'nominal_time_s')):
            raise ValueError('Raw/materialized trajectory differs')
        folder = (capture / raw['folder']).resolve()
        if not folder.is_relative_to(capture.resolve()):
            raise ValueError('Frame escapes frozen source')
        frames.append(dict(original_files={name: dict(path=str(folder / name), sha256=row[field])
            for name, field in [('camera.json', 'camera_sha256'), ('depth_left.exr', 'depth_sha256'),
                                ('depth_left_valid.npy', 'depth_valid_sha256')]}))
    with np.load(material_root / 'observations.npz', allow_pickle=False) as obs, np.load(material_root / 'targets.npz', allow_pickle=False) as target:
        keys = np.asarray([r['frame_key'] for r in rows])
        if not np.array_equal(keys, obs['frame_key']) or not np.array_equal(keys, target['frame_key']):
            raise ValueError('Original observation/label frame identity differs')
        histogram = obs['histogram'].reshape(960, 64, 16).copy()
        labels = target['labels'].copy()
    clips = np.asarray([r['layout_id'] + '/' + r['clip_id'] for r in rows])
    steps = np.asarray([r['pose_index'] for r in rows], dtype=np.int64)
    for clip in set(clips):
        ids = np.flatnonzero(clips == clip)
        if not np.array_equal(steps[ids], np.arange(40)):
            raise ValueError('Frozen clip sequence differs')
    data = dict(rows=rows, labels=labels, histogram=histogram, clip_ids=clips, steps=steps)
    transforms = []

    def loader(_, index):
        row = rows[index]
        key, seed = frame_identity(RAW_MANIFEST_SHA, row, row['camera_sha256'], row['depth_sha256'])
        if key != row['frame_key'] or seed != row['seed']:
            raise ValueError('Frame/seed derivation differs')
        depth, camera = load_scene_depth(frames[index])
        if not np.allclose(camera['T_camera_tof'], np.eye(4), atol=1e-12):
            raise ValueError('Sensor/camera alignment differs')
        transforms.append(np.asarray(camera['T_world_camera']))
        return dict(depth=depth, camera=camera, seed=seed)

    output.mkdir(parents=True, exist_ok=False)
    try:
        bias = np.load(bundle / 'inputs/train-bias.npy', allow_pickle=False)
        alley = read(bundle / 'inputs/alley-result.json')
        raw, quiet, samples, seeds = synthesize(data, loader, started + 1800)
        # No sample cache is needed once the original-response parity gate passes.
        del quiet, samples
        directions = []
        rotations = []
        for clip in set(clips):
            ids = np.flatnonzero(clips == clip)
            for i, j in zip(ids, ids[1:]):
                a, b = transforms[i], transforms[j]
                directions.append((b[:3, 3] - a[:3, 3]) @ a[:3, :3])
                rotations.append(float(np.max(np.abs(b[:3, :3] - a[:3, :3]))))
        directions = np.asarray(directions)
        if not np.allclose(directions, [0, 0, .1], rtol=0, atol=1e-10) or max(rotations) > 1e-12:
            raise ValueError('Original worker forward camera motion differs')
        scores, full, matched = arms(raw, data, bias)
        del raw, matched
        if len(scores) != 22 or set(scores) != set(alley['results']):
            raise ValueError('Frozen arm membership differs')
        score_path = output / 'worker-scores.npz'
        np.savez_compressed(score_path, labels=labels, frame_key=keys, **scores)
        report = dict(status='COMPLETE_ORIGINAL_STREET_WORKER_960_ONLY', frames=960, arms=22,
                      parity_frames=960, scores_sha256=sha(score_path),
                      bias_sha256=sha(bundle / 'inputs/train-bias.npy'),
                      source_hashes={name: sha(bundle / 'code' / name) for name in alley['source_hashes']},
                      thresholds={name: alley['results'][name]['threshold'] for name in scores},
                      bundle_binding_sha256=sha(bundle / 'binding.json'), binding=binding,
                      source_capture=str(capture), source_manifest_sha256=RAW_MANIFEST_SHA,
                      materialized_manifest_sha256=MATERIAL_MANIFEST_SHA,
                      camera_step=dict(count=len(directions), min=directions.min(0).tolist(), max=directions.max(0).tolist(),
                                       max_abs_error=float(np.max(np.abs(directions - [0, 0, .1]))), max_rotation_delta=max(rotations)),
                      threshold_source='Unchanged saved alley train thresholds for each of 22 arms',
                      no_fitting=True, no_alley_run=True, no_new_capture=True, wall_s=time.monotonic()-started)
        write(output / 'worker-result.json', report)
        return report
    except BaseException as error:
        write(output / 'failure.json', dict(status='STOP_WORKER_STREET_RECOVERY', error=repr(error), wall_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.bundle, args.capture, args.output)
    print(json.dumps(dict(status=result['status'], frames=result['frames'], wall_s=result['wall_s'])))
