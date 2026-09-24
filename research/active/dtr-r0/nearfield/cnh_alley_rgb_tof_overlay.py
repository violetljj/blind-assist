"""Bind repaired Development RGB to immutable alley ToF and label arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_file(path: Path, expected: str) -> dict:
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f'Original or replay file hash differs: {path}')
    return {'path': str(path), 'sha256': expected}


def build_overlay(rgb_capture: Path, materialized: Path) -> dict:
    import numpy as np

    rgb_capture = rgb_capture.resolve()
    materialized = materialized.resolve()
    replay_receipt_path = rgb_capture / 'format-receipt.json'
    replay = json.loads(replay_receipt_path.read_text())
    if (replay.get('status') != 'RGB_NUMERIC_PASS_VISUAL_REVIEW_REQUIRED' or
            replay.get('rgb_usability', {}).get('failed_image_count') != 0 or
            replay.get('frame_count') != 160):
        raise ValueError('Complete numeric RGB replay required before overlay')
    old = Path(replay['source_capture']).resolve()
    old_receipt_path = old / 'format-receipt.json'
    old_receipt = json.loads(old_receipt_path.read_text())
    if old_receipt.get('status') != 'PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY':
        raise ValueError('Original seven-pass receipt is incomplete')
    replay_spec = json.loads((rgb_capture / 'source/spec.json').read_text(encoding='utf-8-sig'))
    if (digest(old_receipt_path) != replay_spec['rgb_replay_source']['format_receipt_sha256'] or
            digest(old / 'raw-manifest.json') != replay_spec['rgb_replay_source']['raw_manifest_sha256']):
        raise ValueError('Historical capture receipts changed')
    e2e_receipt_path = materialized / 'receipt.json'
    e2e_receipt = json.loads(e2e_receipt_path.read_text())
    e2e_manifest_path = materialized / 'manifest.json'
    e2e_manifest = json.loads(e2e_manifest_path.read_text())
    if (e2e_receipt.get('status') != 'PASS_DEVELOPMENT_MATERIALIZATION_ONLY' or
            e2e_receipt.get('count') != 160 or e2e_manifest.get('count') != 160 or
            e2e_manifest.get('source_manifest_sha256') != digest(old / 'raw-manifest.json')):
        raise ValueError('Materialized ToF/labels are not bound to original raw manifest')
    observation = checked_file(materialized / 'observations.npz',
                               e2e_receipt['outputs']['observations.npz'])
    targets = checked_file(materialized / 'targets.npz',
                           e2e_receipt['outputs']['targets.npz'])
    checked_file(e2e_manifest_path, e2e_receipt['outputs']['manifest.json'])
    old_rows = {row['id']: row for row in old_receipt['frames']}
    new_rows = {row['frame_id']: row for row in replay['frames']}
    e2e_rows = e2e_manifest['frames']
    if (len(old_rows) != 160 or len(new_rows) != 160 or len(e2e_rows) != 160 or
            set(old_rows) != set(new_rows) or set(old_rows) != {row['id'] for row in e2e_rows}):
        raise ValueError('Replay, original and materialized frame identities differ')
    with np.load(observation['path'], allow_pickle=False) as data:
        observation_keys = data['frame_key'].tolist()
        if data['distance_m'].shape[0] != 160 or data['histogram'].shape[0] != 160:
            raise ValueError('Original ToF observation arrays have wrong frame count')
    with np.load(targets['path'], allow_pickle=False) as data:
        target_keys = data['frame_key'].tolist()
        if data['labels'].shape != (160, 6):
            raise ValueError('Original label array has wrong shape')
    frames = []
    for index, materialized_row in enumerate(e2e_rows):
        frame_id = materialized_row['id']
        prior, current = old_rows[frame_id], new_rows[frame_id]
        if (materialized_row['frame_key'] != observation_keys[index] or
                materialized_row['frame_key'] != target_keys[index]):
            raise ValueError('ToF/label frame keys differ')
        for key in ('layout_id', 'clip_id', 'pose_index', 'physical_site_id'):
            if materialized_row[key] != prior[key] or materialized_row[key] != current.get(key, prior[key]):
                raise ValueError(f'Frame metadata differs: {key}')
        expected = prior['hashes']
        for key, name in (('camera_sha256', 'camera.json'),
                          ('depth_sha256', 'depth_left.exr'),
                          ('depth_valid_sha256', 'depth_left_valid.npy'),
                          ('inserted_geometry_sha256', 'inserted-geometry.json')):
            if materialized_row[key] != expected[name]:
                raise ValueError(f'Materialized frame source differs: {key}')
        old_folder = (old / prior['folder']).resolve()
        new_folder = (rgb_capture / f'frames/{frame_id}').resolve()
        if not old_folder.is_relative_to(old) or not new_folder.is_relative_to(rgb_capture):
            raise ValueError('Frame path escaped its capture root')
        old_files = {name: checked_file(old_folder / name, expected[name])
                     for name in ('camera.json', 'inserted-geometry.json',
                                  'depth_left.exr', 'depth_left_valid.npy')}
        rgb_files = {name: checked_file(new_folder / name, current['new_rgb_sha256'][name])
                     for name in ('left.png', 'right.png')}
        frames.append(dict(frame_id=frame_id, layout_id=prior['layout_id'],
                           physical_site_id=prior['physical_site_id'],
                           clip_id=prior['clip_id'], pose_index=prior['pose_index'],
                           frame_key=materialized_row['frame_key'],
                           original_files=old_files, repaired_rgb=rgb_files,
                           tof_observation_index=index, target_label_index=index,
                           original_right_depth_sha256=expected['depth_right.exr'],
                           original_right_depth_local_verification='UNAVAILABLE_IN_THIN_SOURCE'))
    return dict(schema='cnh-alley-rgb-tof-overlay-v1',
                status='JOINED_DEVELOPMENT_NUMERIC_RGB_VISUAL_REVIEW_REQUIRED',
                benchmark_eligible=False, data_role='Development', frame_count=160,
                rgb_replay_receipt=checked_file(replay_receipt_path, digest(replay_receipt_path)),
                original_receipt=checked_file(old_receipt_path, digest(old_receipt_path)),
                materialized_manifest=checked_file(e2e_manifest_path, digest(e2e_manifest_path)),
                materialized_receipt=checked_file(e2e_receipt_path, digest(e2e_receipt_path)),
                observations=observation, targets=targets,
                arrays=dict(observations=('histogram', 'ambient', 'distance_m', 'valid', 'status', 'frame_key'),
                            targets=('labels', 'frame_key')),
                frames=frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb-capture', type=Path, required=True)
    parser.add_argument('--materialized', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_overlay(args.rgb_capture, args.materialized)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], frames=result['frame_count'],
                          output=str(args.output.resolve()))))


if __name__ == '__main__':
    main()
