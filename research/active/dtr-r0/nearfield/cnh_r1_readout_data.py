"""R1 input-only loader: frozen alley and 1920 Street Development frames.

No sensor synthesis, score, training, download, City or protected-test access.
Canonical EXR/mask/camera inputs are checked lazily before each frame is read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cnh_rgb_dev_comparison import checked, sha
from cnh_street_e2e_train import load_inputs
from cnh_street_e2e_partitions import planned_split
from cnh_street_e2e_materialize import frame_identity
from cnh_rgb_visible_depth_audit import load_scene_depth


REPO = Path(__file__).resolve().parents[4]
EVIDENCE = REPO / 'artifacts.local/evidence'
PARTITION = REPO / 'artifacts.local/work/cnh-route-comparison-20260924/plan/street-alley-merged-partitions-20260924-v1.json'
PARTITION_HASH = '7ae115183bab26876eedc4d9bd3b11cb0b7613c2b56dddb3055266fe1a586a9b'
COLLECTION = EVIDENCE / 'cnh-alley-rgb-replay-six-20260925-v2/collection-overlay.json'
COLLECTION_HASH = 'f72425577e4e06f9a95e39ba2ef0a9b3d6fc087157e74d31bcfce7927ba4840f'
STREET_ROOTS = (EVIDENCE / 'cnh-street-e2e-main-20260924-v2', EVIDENCE / 'cnh-street-e2e-worker-20260924-v1')
STREET_MANIFEST_HASHES = ('224522daa0f3d1357538ad453cfa7a26389d4d2ee6201301fc4d1bf3f6aa68c1',
                          'f093a5ed14ba9efb1002e642f33e2ed309c09f5fb0d2808cf2248079a8a5540e')
STREET_CAPTURES = (EVIDENCE / 'cnh-street-development-batch-20260924-v1/capture',
                   EVIDENCE / 'cnh-street-development-worker-batch-thin-20260924-v1/payload/capture')
QUERY_NAMES = ('left_HEAD', 'left_BODY', 'centre_HEAD', 'centre_BODY', 'right_HEAD', 'right_BODY')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _assemble(roots, frames, partition, expected_frames, dataset, require_depth):
    if sha(partition) != PARTITION_HASH:
        raise ValueError('Frozen Development partition differs')
    plan = read(partition)
    _, labels, rows, names, sources = load_inputs(roots, plan)
    train, dev, split = planned_split(rows, plan)
    if len(rows) != expected_frames or tuple(names) != QUERY_NAMES:
        raise ValueError('Frozen frame count/query order differs')
    if train.sum() != expected_frames // 2 or dev.sum() != expected_frames // 2:
        raise ValueError('Frozen equal train/dev split differs')
    if [f['frame_key'] for f in frames] != [r['frame_key'] for r in rows]:
        raise ValueError('Source/materialized frame order differs')
    histogram, ambient, scalar, valid = [], [], [], []
    for root in roots:
        with np.load(Path(root) / 'observations.npz', allow_pickle=False) as obs:
            histogram.append(obs['histogram'].reshape(-1, 64, 16).copy())
            ambient.append(obs['ambient'].reshape(-1, 64).copy())
            scalar.append(obs['distance_m'].reshape(-1, 64).copy())
            valid.append(obs['valid'].reshape(-1, 64).copy())
    groups = {}
    for i, (frame, row) in enumerate(zip(frames, rows)):
        if frame['frame_id'] != row['id'] or frame['layout_id'] != row['layout_id']:
            raise ValueError('Source frame identity differs')
        key = (row['layout_id'], row['clip_id'])
        groups.setdefault(key, []).append(i)
    clips = []
    for key, indices in groups.items():
        indices = sorted(indices, key=lambda i: rows[i]['pose_index'])
        if len(indices) != 40 or [rows[i]['pose_index'] for i in indices] != list(range(40)):
            raise ValueError('Expected complete 40-frame clips')
        times = np.asarray([rows[i]['nominal_time_s'] for i in indices])
        if not np.allclose(times, np.arange(40) * .1, rtol=0, atol=1e-12):
            raise ValueError('Nominal clip timing differs')
        if len({bool(train[i]) for i in indices}) != 1:
            raise ValueError('Clip crosses split')
        clips.append(dict(clip_key=key, indices=indices, nominal_interval_s=.1,
                          nominal_translation_step_m=.1, timing_authority='STATIC_SPATIAL_POSES_NOT_OBSERVED_VIDEO_TIME'))
    complete = np.asarray([all(Path(binding['path']).is_file() for binding in f['original_files'].values()) for f in frames])
    if require_depth and not complete.all():
        first = int(np.flatnonzero(~complete)[0])
        missing = [binding['path'] for binding in frames[first]['original_files'].values() if not Path(binding['path']).is_file()]
        raise FileNotFoundError(f'{dataset}: canonical sources complete for {int(complete.sum())}/{len(complete)} frames; first missing {missing}')
    return dict(dataset=dataset, rows=rows, labels=labels, train=train, dev=dev, split=split,
                query_names=names, histogram=np.concatenate(histogram), ambient=np.concatenate(ambient),
                scalar=np.concatenate(scalar), valid=np.concatenate(valid), frames=frames, clips=clips,
                clip_ids=np.asarray([row['layout_id'] + '/' + row['clip_id'] for row in rows]),
                steps=np.asarray([row['pose_index'] for row in rows], dtype=np.int64),
                source_available=complete, materialized_sources=sources, partition_sha256=sha(partition),
                identity=dict(dataset=dataset, partition_sha256=sha(partition), materialized_sources=sources),
                raw128_stored=False, timing_authority='nominal_time_s resets each clip; static spatial recipe, not measured cadence')


def load_alley(collection=COLLECTION, partition=PARTITION, *, require_depth=True):
    """Return 960 frozen alley rows; no RGB payload is decoded."""
    if sha(collection) != COLLECTION_HASH:
        raise ValueError('Frozen alley collection differs')
    collection_data = read(collection)
    if collection_data.get('status') != 'PASS_DEVELOPMENT_RGB_TOF_OVERLAY_ONLY' or collection_data.get('test_collection') != 'NOT_RUN':
        raise ValueError('Only accepted Development overlays allowed')
    roots, frames = [], []
    for item in collection_data['layouts']:
        if item['split'] not in ('train', 'dev'):
            raise ValueError('Protected partition prohibited')
        overlay = read(checked(dict(path=item['overlay'], sha256=item['overlay_sha256'])))
        manifest_path = checked(overlay['materialized_manifest'])
        manifest = read(manifest_path)
        roots.append(manifest_path.parent)
        for frame in overlay['frames']:
            frames.append({**frame, 'source_manifest_sha256': manifest['source_manifest_sha256'],
                           'original_files': {name: frame['original_files'][name] for name in
                                              ('camera.json', 'depth_left.exr', 'depth_left_valid.npy')}})
    data = _assemble(roots, frames, partition, 960, 'ALLEY_DEVELOPMENT', require_depth)
    data['collection_sha256'] = sha(collection)
    return data


def load_street(partition=PARTITION, *, main_capture=None, worker_capture=None, require_depth=True):
    """Return original 1920 Street rows; missing worker source fails explicitly.

    Overrides only relocate captures with the exact frozen raw-manifest hash;
    they cannot substitute another dataset. require_depth=False permits a
    metadata/H3 audit but load_frame still refuses unavailable canonical inputs.
    """
    captures = (Path(main_capture) if main_capture is not None else STREET_CAPTURES[0],
                Path(worker_capture) if worker_capture is not None else STREET_CAPTURES[1])
    frames = []
    for root, capture, manifest_hash in zip(STREET_ROOTS, captures, STREET_MANIFEST_HASHES):
        manifest = read(checked(dict(path=str(root / 'manifest.json'), sha256=manifest_hash)))
        raw_path = capture / 'raw-manifest.json'
        if sha(raw_path) != manifest['source_manifest_sha256']:
            raise ValueError('Relocated Street source manifest differs')
        raw = read(raw_path)
        if len(raw['frames']) != 960 or any(r.get('data_role') != 'Development' or
                                           str(r.get('split', '')).lower() in ('test', 'locked_test', 'blind') for r in raw['frames']):
            raise ValueError('Only frozen Street Development frames permitted')
        raw_rows = {r['id']: r for r in raw['frames']}
        for row in manifest['frames']:
            source_row = raw_rows[row['id']]
            if any(row[k] != source_row[k] for k in ('layout_id', 'clip_id', 'pose_index', 'nominal_time_s')):
                raise ValueError('Raw/materialized trajectory identity differs')
            folder = (capture / source_row['folder']).resolve()
            if not folder.is_relative_to(capture.resolve()):
                raise ValueError('Frame folder escaped frozen capture')
            files = {name: dict(path=str(folder / name), sha256=row[field]) for name, field in
                     [('camera.json', 'camera_sha256'), ('depth_left.exr', 'depth_sha256'), ('depth_left_valid.npy', 'depth_valid_sha256')]}
            frames.append(dict(frame_id=row['id'], frame_key=row['frame_key'], layout_id=row['layout_id'],
                               clip_id=row['clip_id'], pose_index=row['pose_index'], original_files=files,
                               source_manifest_sha256=manifest['source_manifest_sha256']))
    return _assemble(STREET_ROOTS, frames, partition, 1920, 'STREET_DEVELOPMENT', require_depth)


def load_frame(data, index):
    """Read one checked depth frame; does not synthesize or compute a score."""
    row, frame = data['rows'][index], data['frames'][index]
    for name, field in [('camera.json', 'camera_sha256'), ('depth_left.exr', 'depth_sha256'), ('depth_left_valid.npy', 'depth_valid_sha256')]:
        if frame['original_files'][name]['sha256'] != row[field]:
            raise ValueError('Frame input identity differs')
    key, seed = frame_identity(frame['source_manifest_sha256'], row, row['camera_sha256'], row['depth_sha256'])
    if key != row['frame_key'] or key != frame['frame_key'] or seed != row['seed']:
        raise ValueError('Frame/seed identity differs')
    depth, camera = load_scene_depth(frame)
    if not np.allclose(camera['T_camera_tof'], np.eye(4), atol=1e-12):
        raise ValueError('Current frozen sensor assumes collocated aligned camera and ToF')
    return dict(depth=depth, camera=camera, seed=seed, original_h3=data['histogram'][index],
                clip_key=(row['layout_id'], row['clip_id']), step=int(row['pose_index']),
                nominal_time_s=float(row['nominal_time_s']), labels=data['labels'][index], frame_key=key)
